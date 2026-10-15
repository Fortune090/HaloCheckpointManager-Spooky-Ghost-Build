#pragma once
#include "pch.h"
#include "IOptionalCheat.h"
#include "GameState.h"
#include "DIContainer.h"

// Chase Camera (MCC 1.3528 only): Halo 2, Halo 3, Halo 3: ODST, Halo: Reach, Halo 4.
//
// Forces the engine's own attached third-person CHASE camera (the director's "following" camera - the one it already
// uses for vehicles with a third-person seat) on the local player while on foot, with optional no camera collision,
// field of view and vertical offset. Nothing in game memory is permanently written: turning it off just makes the
// hooks answer honestly again and the director swaps itself back to first person on its next update.
//
// One cheat object per game (template in the .cpp, same pattern as ThirdPersonRendering), all driven by the same four
// settings (chaseCameraToggle / chaseCameraNoClip / chaseCameraFov / chaseCameraHeight). Only the object for the game
// that is currently playing ever enables itself. Per-game mechanism:
//
//   Halo 2        hook the director's "should this unit use the following camera?" predicate (answer yes + a sentinel
//                 perspective code so the director's change-detection swaps cameras).
//   Halo 3 / ODST hook director_set_perspective and rewrite its code argument 1 (first person) -> 0 (follow). The
//                 director calls it every tick on foot with code 1, so this is a pure argument rewrite.
//   Reach / Halo 4 hook director_set_camera_mode and rewrite mode 3 (first person) -> 1 (following) when not forced.
//   All games     hook the FOLLOWING camera's update (the function that fills the observer command) and post-edit
//                 its output: |= 0x10 (no camera collision), FOV, focus z += height. Because it is the following
//                 camera's own update, the edits can only ever apply while the following camera is the active one.
//
// A game whose two InternalPointerData.xml entries (chaseCameraSelectFunction / chaseCameraFollowFunction) are missing,
// or whose function prologues do not match the build the RE was done on, simply has no Chase Camera (construction
// throws HCMInitException / the toggle refuses) - it never hooks blindly.
//
// COEXISTENCE
//   * ThirdPersonRendering (a Free Camera option; Halo 3 / ODST / Reach / Halo 4 mid-hook a `je` inside the SAME
//     select function that forces its "same camera already? skip" early-out to never be taken, i.e. the camera object is
//     REBUILT on every call): our select hook sits at the function entry and, once the chase camera is already the
//     current one, returns "no change" without calling the original, so that forced rebuild cannot reset the following
//     camera (Reach/Halo 4 would otherwise lose its yaw/pitch every tick). Halo 2's ThirdPersonRendering hooks a
//     different function and does not interact.
//   * Free Camera hooks the FINAL camera data (setCameraData), after everything here, so while Free Camera is on you
//     see the free camera; the chase camera is simply still the underlying camera and is what you get back when Free
//     Camera is turned off. Turning chase on/off while Free Camera is on is safe (the director swaps in the background).
class ChaseCameraImplBase { public: virtual ~ChaseCameraImplBase() = default; };

class ChaseCamera : public IOptionalCheat
{
private:
	std::unique_ptr<ChaseCameraImplBase> pimpl;

public:
	ChaseCamera(GameState gameImpl, IDIContainer& dicon);
	~ChaseCamera();

	std::string_view getName() override { return nameof(ChaseCamera); }
};
