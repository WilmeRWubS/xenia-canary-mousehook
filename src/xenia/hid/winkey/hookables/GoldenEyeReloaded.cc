/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2026 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#define _USE_MATH_DEFINES

#include "xenia/hid/winkey/hookables/GoldenEyeReloaded.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <map>
#include <mutex>

#include "xenia/base/platform_win.h"
#include "xenia/emulator.h"
#include "xenia/hid/hid_flags.h"
#include "xenia/hid/input_system.h"
#include "xenia/kernel/util/shim_utils.h"
#include "xenia/xbox.h"

using namespace xe::kernel;

DECLARE_double(sensitivity);
DECLARE_double(fov_sensitivity);
DECLARE_bool(invert_y);
DECLARE_bool(invert_x);

const uint32_t kTitleIdGoldenEyeReloaded = 0x415608AF;

namespace xe {
namespace hid {
namespace winkey {

// Mouse look works the same way as the GoldenEye 007: Reloaded static
// recompilation (github.com/nikolaygorb/GoldenEye007ReloadedRecomp): the
// mouse turn is injected into the player look update after the game's stick
// processing, so there's no dead zone, response curve or turn speed cap.
namespace {

struct ReloadedBuildAddrs {
  uint32_t check_addr;   // instruction to check
  uint32_t check_value;  // expected instruction at check_addr

  // lbz r11,2618(r31): aim lock / snap checks, the look update returns right
  // after while either is set.
  uint32_t aim_assist_hook_addr;
  // Right after the aim friction call: r1+104 holds the yaw rate (rad/s).
  uint32_t yaw_hook_addr;
  // The pitch clamp every look update path ends in (r31+2140 is loaded).
  uint32_t pitch_hook_addr;

  uint32_t unzoomed_fov_addr;  // float, radians
};

const std::map<GoldenEyeReloadedGame::GameBuild, ReloadedBuildAddrs>
    kSupportedBuilds = {
        {GoldenEyeReloadedGame::GameBuild::GoldenEyeReloaded_Retail,
         {0x821FDD00, 0x897F0A3A, 0x821FDD00, 0x821FDD64, 0x821FDEE4,
          0x827F2854}},
};

// Offsets into the player struct (r31 in the look update).
constexpr uint32_t kPlayerDeltaTime = 428;  // float, seconds
constexpr uint32_t kPlayerPitch = 2140;     // float, radians, + = look down
constexpr uint32_t kPlayerFov = 2576;       // float, radians
constexpr uint32_t kPlayerAimLock = 2618;   // uint8
constexpr uint32_t kPlayerAimLockCached = 2661;  // uint8
constexpr uint32_t kPlayerAimSnap = 2662;        // uint8
constexpr uint32_t kStackYawRate = 104;  // float on r1, rad/s, + = right

// Same scale as the Call of Duty hooks: degrees per mouse count at
// sensitivity 1.
constexpr double kCountsPerDegree = 10.5;
// Mouse motion (|dx| + |dy|) that cancels an aim snap / lock.
constexpr double kSnapCancelCounts = 4.0;
// Guards against a stalled frame turning into one huge snap.
constexpr double kMaxRadiansPerUpdate = 1.5;
// If the look update hasn't run for this long we're in a menu / cutscene /
// vehicle; mouse motion is dropped instead of piling up.
constexpr std::chrono::milliseconds kLookUpdateTimeout{300};

std::mutex mouse_mutex_;
double pending_dx_ = 0.0;
double pending_dy_ = 0.0;
std::chrono::steady_clock::time_point last_look_update_{};

// Pitch taken by the yaw hook, applied by the pitch hook of the same update.
thread_local double pending_pitch_ = 0.0;
thread_local bool pitch_pending_ = false;

const ReloadedBuildAddrs* active_build_ = nullptr;

float LoadFloat(uint32_t address) {
  return *kernel_memory()->TranslateVirtual<xe::be<float>*>(address);
}

void StoreFloat(uint32_t address, float value) {
  *kernel_memory()->TranslateVirtual<xe::be<float>*>(address) = value;
}

bool HookActive() {
  return active_build_ &&
         kernel_state()->title_id() == kTitleIdGoldenEyeReloaded;
}

void MarkLookUpdate() {
  std::lock_guard<std::mutex> lock(mouse_mutex_);
  last_look_update_ = std::chrono::steady_clock::now();
}

bool PlausibleFov(double fov) {
  return std::isfinite(fov) && fov > 0.02 && fov < 3.0;
}

// Lowers sensitivity while zoomed, like cg_fovscale in the COD hooks.
double ZoomScale(uint32_t player) {
  const double fov = LoadFloat(player + kPlayerFov);
  const double unzoomed = LoadFloat(active_build_->unzoomed_fov_addr);
  if (!PlausibleFov(fov) || !PlausibleFov(unzoomed)) {
    return 1.0;
  }
  double scale = std::tan(fov * 0.5) / std::tan(unzoomed * 0.5);
  if (scale >= 0.999) {
    return 1.0;
  }
  scale = std::clamp(scale, 0.02, 1.0);
  const double a = cvars::fov_sensitivity;
  return (1.0 - a) * (scale * scale) + a * scale;
}

void AimAssistHook(PPCContext* context, void* arg0, void* arg1) {
  if (!HookActive()) {
    return;
  }
  MarkLookUpdate();

  const uint32_t player = static_cast<uint32_t>(context->r[31]);
  uint8_t* lock =
      kernel_memory()->TranslateVirtual<uint8_t*>(player + kPlayerAimLock);
  uint8_t* snap =
      kernel_memory()->TranslateVirtual<uint8_t*>(player + kPlayerAimSnap);
  if (!*lock && !*snap) {
    return;
  }

  double moved;
  {
    std::lock_guard<std::mutex> guard(mouse_mutex_);
    moved = std::abs(pending_dx_) + std::abs(pending_dy_);
  }
  // Moving the mouse ends the aim snap / lock, as moving the stick does.
  if (moved >= kSnapCancelCounts) {
    *lock = 0;
    *kernel_memory()->TranslateVirtual<uint8_t*>(player +
                                                 kPlayerAimLockCached) = 0;
    *snap = 0;
  }
}

void YawHook(PPCContext* context, void* arg0, void* arg1) {
  pitch_pending_ = false;
  if (!HookActive()) {
    return;
  }

  const uint32_t player = static_cast<uint32_t>(context->r[31]);
  const double dt = LoadFloat(player + kPlayerDeltaTime);

  double dx, dy;
  {
    std::lock_guard<std::mutex> lock(mouse_mutex_);
    last_look_update_ = std::chrono::steady_clock::now();
    if (!(dt > 1e-4 && dt < 0.5)) {
      return;  // keep the motion for the next update
    }
    dx = pending_dx_;
    dy = pending_dy_;
    pending_dx_ = pending_dy_ = 0.0;
  }
  if (dx == 0.0 && dy == 0.0) {
    return;
  }

  const double radians_per_count = (cvars::sensitivity / kCountsPerDegree) *
                                   (M_PI / 180.0) * ZoomScale(player);
  double yaw = dx * radians_per_count * (cvars::invert_x ? -1.0 : 1.0);
  double pitch = dy * radians_per_count * (cvars::invert_y ? -1.0 : 1.0);
  if (!std::isfinite(yaw) || !std::isfinite(pitch)) {
    return;
  }
  yaw = std::clamp(yaw, -kMaxRadiansPerUpdate, kMaxRadiansPerUpdate);
  pitch = std::clamp(pitch, -kMaxRadiansPerUpdate, kMaxRadiansPerUpdate);

  const uint32_t rate_address =
      static_cast<uint32_t>(context->r[1]) + kStackYawRate;
  StoreFloat(rate_address,
             LoadFloat(rate_address) + static_cast<float>(yaw / dt));

  pending_pitch_ = pitch;
  pitch_pending_ = true;
}

void PitchHook(PPCContext* context, void* arg0, void* arg1) {
  if (!pitch_pending_) {
    return;
  }
  pitch_pending_ = false;
  if (!HookActive()) {
    return;
  }
  const uint32_t address = static_cast<uint32_t>(context->r[31]) + kPlayerPitch;
  StoreFloat(address, LoadFloat(address) + static_cast<float>(pending_pitch_));
}

}  // namespace

GoldenEyeReloadedGame::~GoldenEyeReloadedGame() = default;

bool GoldenEyeReloadedGame::IsGameSupported(GameVersion title_version) {
  if (kernel_state()->title_id() != kTitleIdGoldenEyeReloaded) {
    return false;
  }

  if (game_build_ != GameBuild::Unknown) {
    return true;
  }

  for (auto& build : kSupportedBuilds) {
    auto* check = kernel_memory()->TranslateVirtual<xe::be<uint32_t>*>(
        build.second.check_addr);
    if (*check == build.second.check_value) {
      game_build_ = build.first;
      return true;
    }
  }

#ifdef XENIA_MOUSEHOOK_MESSAGE
  mousehook_message_wrapper(
      "MOUSEHOOK: Supported Title ID, but unsupported version of GoldenEye "
      "007: Reloaded, refer to the Github repo for supported titles.",
      true);
#endif
  return false;
}

bool GoldenEyeReloadedGame::DoHooks(uint32_t user_index,
                                    RawInputState& input_state,
                                    X_INPUT_STATE* out_state) {
  if (!input_state.mouse.x_delta && !input_state.mouse.y_delta) {
    return false;
  }

  std::lock_guard<std::mutex> lock(mouse_mutex_);
  if (std::chrono::steady_clock::now() - last_look_update_ >
      kLookUpdateTimeout) {
    // Not in gameplay, don't let motion pile up for when we get back in.
    pending_dx_ = pending_dy_ = 0.0;
    return false;
  }
  pending_dx_ += input_state.mouse.x_delta;
  pending_dy_ += input_state.mouse.y_delta;
  return true;
}

std::string GoldenEyeReloadedGame::ChooseBinds() { return "Default"; }

bool GoldenEyeReloadedGame::ModifierKeyHandler(uint32_t user_index,
                                               RawInputState& input_state,
                                               X_INPUT_STATE* out_state) {
  return false;
}

void GoldenEyeReloadedGame::WeaponSwitchHandler(uint32_t user_index,
                                                RawInputState& input_state,
                                                X_INPUT_STATE* out_state,
                                                int weapon, uint16_t buttons) {}

void GoldenEyeReloadedGame::MidHookInit() {
  if (midhook_status == HOOKED || game_build_ == GameBuild::Unknown) {
    return;
  }

  active_build_ = &kSupportedBuilds.at(game_build_);
  xe::cpu::ppc::RegisterMidHookASM(active_build_->aim_assist_hook_addr,
                                   AimAssistHook);
  xe::cpu::ppc::RegisterMidHookASM(active_build_->yaw_hook_addr, YawHook);
  xe::cpu::ppc::RegisterMidHookASM(active_build_->pitch_hook_addr, PitchHook);

  midhook_status = HOOKED;
}

}  // namespace winkey
}  // namespace hid
}  // namespace xe
