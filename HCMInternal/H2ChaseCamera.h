#pragma once
#include "pch.h"
#include "IOptionalCheat.h"
#include "GameState.h"
#include "DIContainer.h"

// Halo 2 (MCC halo2.dll 1.3528 only). Forces the engine's own attached third-person CHASE camera (the director's
// "following" camera - the one it uses for vehicles with a third-person seat) on the local player while on foot.
//
// There is no engine flag/global for this. The director re-chooses its camera from ONE predicate, sub_1806D4C50
// (RVA 0x6D4C50): "does this unit want the following camera?" - true only for vehicle seats flagged third-person
// (or a weapon with the third-person bit). This cheat hooks that predicate and answers "yes" for any live unit, plus a
// sentinel perspective code so the director's own change-detection (sub_1806D4B20) swaps cameras - with the engine's
// normal blend - on the next director update, and swaps straight back when the hook goes inert. Nothing in the
// game's memory is ever written; turning it off is just the hook answering honestly again.
//
// Why a hook and not a poke: aim (sub_1806D4730), the camera swap, level-load init and cutscene exit all call that
// same predicate, so one answer keeps all of them consistent. Poking director.update_function would be undone by the
// engine on the next perspective change and would leave the aim path on the first-person branch.
class H2ChaseCamera : public IOptionalCheat
{
private:
	class Impl;
	std::unique_ptr<Impl> pimpl;

public:
	H2ChaseCamera(GameState gameImpl, IDIContainer& dicon);
	~H2ChaseCamera();

	std::string_view getName() override { return nameof(H2ChaseCamera); }
};
