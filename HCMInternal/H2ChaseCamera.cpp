#include "pch.h"
#include "H2ChaseCamera.h"
#include "SettingsStateAndEvents.h"
#include "IMessagesGUI.h"
#include "IMCCStateHook.h"
#include "RuntimeExceptionHandler.h"
#include "PointerDataStore.h"
#include "MultilevelPointer.h"
#include "ModuleHook.h"

// ============================================================================
//  Halo 2 chase camera (MCC halo2.dll build 1.3528).
//
//  RE (all addresses are halo2.dll RVAs = IDA EA - 0x180000000):
//   * The director (Blam director.cpp) is a 4-element array of 0x150-byte structs at RVA 0x15EB220 (header at
//     0x15EB210; header+4 = director mode, 0 = game). Element +0x10 is the camera's UPDATE FUNCTION POINTER, which
//     IS the camera mode:
//         0x77DEB0 first person   0x77C9F0 following (chase)   0x77D570 dead   0x77E990 editor   0x7186D0 scripted
//     director_get_perspective (0x6D4EC0) is just a classifier over that pointer: first person -> 0 (or 3 while the
//     change-pause at +0x04 is running), following -> 1, scripted/editor -> 2. Every renderer consumer treats
//     "!= 0" as third person (draw the body, skip the first-person gun pass 0x7E0C60, etc.).
//   * The per-frame chooser is 0x6D4B20 (director_choose_camera(user, force)), reached from 0x6D4910 each
//     director_update. It asks 0x6D4C50(unit_index, short* perspective_code) "should this unit use the following
//     camera?". For a biped on foot the answer is always 0 (the only 1-sources are a vehicle seat with the
//     third-person-camera flag [seat flags bit 4/6], or a held weapon whose definition has bit 31 at +0x12C).
//     If the answer is 1 and the camera is currently first person, 0x6D4B20 calls the following-camera init
//     (0x77C9C0) and installs 0x77C9F0 + the blend timer (0x6D59A0); if 0 and currently following, it swaps back.
//     It only re-evaluates when the stored perspective code (element +0x70) differs from the new one, or on force
//     (respawn / cutscene exit).
//   * The aim path 0x6D4730 and the level-init 0x6D5160 / cutscene-exit 0x6D58A0 call the SAME predicate, so
//     hooking it keeps aim (camera-track branch 0x77C700 vs first-person 0x77DAA0) and every camera swap consistent.
//   * The following camera needs no vehicle: with no unit camera tracks the track evaluators (0x77C080/0x77C400)
//     fall back to the default camera track in game globals (+0xEC), which is what the engine's own debug camera
//     cycle (first person -> flying -> following, name table 0x180DFC420, order table 0x180B15DC0) uses on foot.
//
//  This cheat: inline-hook 0x6D4C50. While enabled, when the original answers 0 for a live unit it answers 1 and
//  writes a sentinel perspective code, so 0x6D4B20 sees "code changed" and switches on the next director update
//  (with the engine's normal blend). When the toggle goes off the hook answers honestly (code returns to its real
//  value), the stored sentinel differs again, and 0x6D4B20 swaps back to first person by itself. NOTHING in the
//  game's memory is written by this cheat. The hook is attached once (after a prologue check) and left inert while
//  off - detaching an inline hook while the game thread is inside it races (see ModuleHook.cpp, hook graveyard).
// ============================================================================

namespace
{
	// sub_1806D4C50 prologue up to (not including) its first `call` - no relative operands, so it is build-stable
	// but still refuses another build or a function someone else already hooked.
	//   mov [rsp+18h],rbx; push rbp,rsi,rdi,r13,r14; sub rsp,20h; xor ebx,ebx; mov r14d,ecx; mov rdi,rdx;
	//   mov ebp,ebx; lea edx,[rbx+3]
	constexpr uint8_t kPredicateEntryBytes[] = {
		0x48, 0x89, 0x5C, 0x24, 0x18, 0x55, 0x56, 0x57, 0x41, 0x55, 0x41, 0x56, 0x48, 0x83, 0xEC, 0x20,
		0x33, 0xDB, 0x44, 0x8B, 0xF1, 0x48, 0x8B, 0xFA, 0x8B, 0xEB, 0x8D, 0x53, 0x03 };

	// sub_18077C9F0 (following camera update, the 0x77C9F0 mode pointer) prologue up to its first call: register/stack
	// ops only, no relative operands.
	//   mov rax,rsp; push rbx,rsi,rdi,r12,r14; sub rsp,0A0h; mov r14,rdx; movaps [rax-38h],xmm6; mov rsi,rcx;
	//   mov [rax+18h],r15; lea rdx,[rax-78h]; mov rdi,r8; mov ecx,[r14]
	constexpr uint8_t kFollowUpdateEntryBytes[] = {
		0x48, 0x8B, 0xC4, 0x53, 0x56, 0x57, 0x41, 0x54, 0x41, 0x56, 0x48, 0x81, 0xEC, 0xA0, 0x00, 0x00, 0x00, 0x4C, 0x8B, 0xF2,
		0x0F, 0x29, 0x70, 0xC8, 0x48, 0x8B, 0xF1, 0x4C, 0x89, 0x78, 0x18, 0x48, 0x8D, 0x50, 0x88, 0x49, 0x8B, 0xF8, 0x41, 0x8B, 0x0E };

	// Observer command flag bit 4 (Blam observer.cpp: TEST_BIT(pending_command.flags, 4)). MCC's observer update
	// (halo2.dll 1806F0000.c decompile: `if ((observer[16] & 0x10) == 0 && distance != 0) distance = sub_1806F05A0(...)`)
	// skips the camera-collision "safe distance" pull-in when it is set; observer_apply_camera_effect skips its
	// collision_test_line (sub_18075B2C0) too. The following camera writes this command every frame (flags dword
	// at +0: zeroed, then |= 0x80, |= 1), so setting the bit right after it = a no-clip chase camera.
	constexpr uint32_t kObserverCommandNoCollision = 0x10;

	// Real codes the predicate writes are 0..4. Any other value is just "different from the real one" so that
	// 0x6D4B20's `stored != new` check fires on the way in AND on the way out.
	constexpr int16_t kForcedPerspectiveCode = 0x7C;

	bool entryBytesMatch(uintptr_t at, const uint8_t* expect, size_t n)
	{
		uint8_t got[64] = {};
		if (n > sizeof(got)) return false;
		__try { memcpy(got, (const void*)at, n); }
		__except (EXCEPTION_EXECUTE_HANDLER) { return false; }
		return memcmp(got, expect, n) == 0;
	}
}

class H2ChaseCamera::Impl
{
private:
	GameState mGame;
	std::weak_ptr<SettingsStateAndEvents> settingsWeak;
	std::weak_ptr<IMessagesGUI> messagesWeak;

	std::shared_ptr<MultilevelPointer> mPredicateFunction;   // halo2.dll+0x6D4C50 (InternalPointerData.xml)
	std::unique_ptr<ModuleInlineHook> mHook;
	std::mutex mArmMutex;                                    // toggle thread vs MCC-state thread
	bool mRefused = false;                                   // prologue didn't match: never hook this build

	static inline std::atomic<Impl*> sInstance{ nullptr };
	static inline std::atomic_bool sEnabled{ false };

	// no-clip: second hook on the following camera's update (halo2.dll+0x77C9F0)
	std::shared_ptr<MultilevelPointer> mFollowUpdateFunction;
	std::unique_ptr<ModuleInlineHook> mFollowHook;
	bool mFollowRefused = false;
	static inline std::atomic_bool sNoClip{ false };
	std::unique_ptr<ScopedCallback<ToggleEvent>> mNoClipCallback;
	std::shared_ptr<BinarySetting<float>> mFovSetting, mHeightSetting;   // read by the game thread each chase update

	// __int64 __fastcall sub_18077C9F0(camera_state*, camera_input*, s_observer_command* out)
	static int64_t __fastcall followUpdateDetour(void* state, void* input, uint8_t* outCommand)
	{
		Impl* self = sInstance.load(std::memory_order_acquire);
		if (!self || !self->mFollowHook) return 0;
		const int64_t r = self->mFollowHook->getInlineHook().call<int64_t>(state, input, outCommand);
		if (outCommand && sEnabled.load(std::memory_order_relaxed))
		{
			if (sNoClip.load(std::memory_order_relaxed))
				*(uint32_t*)outCommand |= kObserverCommandNoCollision;
			// FOV: command float[10] (+0x28) = field of view (radians); the update mirrors it into float[34] (+0x88) under
			// flag 0x80 and blends on change (decompile of sub_18077C9F0). 0 = leave the game's own value.
			const float fovDeg = self->mFovSetting ? self->mFovSetting->GetValue() : 0.f;
			if (fovDeg > 1.f && fovDeg < 179.f)
			{
				const float rad = fovDeg * 0.017453292f;
				*(float*)(outCommand + 0x28) = rad;
				*(float*)(outCommand + 0x88) = rad;
			}
			// vertical offset: command float[1..3] (+0x04..+0x0C) = the focus point the camera orbits (the unit's camera
			// position from sub_1806C3A90); raising its z lifts both the camera and what it looks at.
			const float dz = self->mHeightSetting ? self->mHeightSetting->GetValue() : 0.f;
			if (dz != 0.f && std::isfinite(dz)) *(float*)(outCommand + 0x0C) += dz;
		}
		return r;
	}

	// Arms the no-clip hook; a mismatch only disables no-clip, never the chase camera itself. Caller holds mArmMutex.
	void ensureFollowArmedLocked()
	{
		if (mFollowRefused || !mFollowHook || mFollowHook->isHookInstalled()) return;
		uintptr_t addr = 0;
		if (!mFollowUpdateFunction || !mFollowUpdateFunction->resolve(&addr)) return;
		if (!entryBytesMatch(addr, kFollowUpdateEntryBytes, sizeof(kFollowUpdateEntryBytes)))
		{
			mFollowRefused = true;
			PLOG_ERROR << "[H2ChaseCamera] following-camera update prologue differs - no-clip unavailable";
			return;
		}
		mFollowHook->setWantsToBeAttached(true);
		if (mFollowHook->isHookInstalled()) PLOG_INFO << "[H2ChaseCamera] no-clip hook armed";
		else { mFollowRefused = true; PLOG_ERROR << "[H2ChaseCamera] safetyhook could not install the no-clip hook"; }
	}

	// Subscribed LAST-declared / reset FIRST in ~Impl: the settings object and MCC state hook outlive this cheat.
	std::unique_ptr<ScopedCallback<ToggleEvent>> mToggleCallback;
	std::unique_ptr<ScopedCallback<eventpp::CallbackList<void(const MCCState&)>>> mStateCallback;

	// int __fastcall sub_1806D4C50(uint32 unit_index, int16* out_perspective_code)
	// Returns 1 = use the following (chase) camera. Called by the game/director thread every director update.
	static int32_t __fastcall predicateDetour(uint32_t unitIndex, int16_t* outPerspectiveCode)
	{
		Impl* self = sInstance.load(std::memory_order_acquire);
		if (!self || !self->mHook)
		{
			// teardown race: answer "no unit" the way the original does, never leave the out-param unwritten
			if (outPerspectiveCode) *outPerspectiveCode = 0;
			return 0;
		}

		const int32_t original = self->mHook->getInlineHook().call<int32_t>(unitIndex, outPerspectiveCode);

		// original == 0 for a unit on foot / in a first-person seat. unitIndex == NONE (-1) is "no unit" (dead,
		// not spawned): leave that to the engine so the dead camera and respawn path stay untouched.
		if (original == 0 && sEnabled.load(std::memory_order_relaxed) && unitIndex != 0xFFFFFFFFu && outPerspectiveCode)
		{
			*outPerspectiveCode = kForcedPerspectiveCode;
			return 1;
		}
		return original;
	}

	void say(const std::string& msg)
	{
		PLOG_INFO << msg;
		try { if (auto m = messagesWeak.lock()) m->addMessage(msg); }
		catch (...) {}
	}

	// Attach the hook once halo2.dll is present and the predicate's prologue is the one the RE was done on.
	// Returns true if the hook is (now) installed. A module that isn't loaded yet is NOT a refusal - the next MCC
	// Ingame state retries.
	bool ensureArmed()
	{
		std::scoped_lock lk(mArmMutex);
		if (mRefused) return false;
		ensureFollowArmedLocked();
		if (mHook->isHookInstalled()) return true;

		uintptr_t addr = 0;
		if (!mPredicateFunction || !mPredicateFunction->resolve(&addr)) return false;   // halo2.dll not loaded yet

		if (!entryBytesMatch(addr, kPredicateEntryBytes, sizeof(kPredicateEntryBytes)))
		{
			mRefused = true;
			PLOG_ERROR << "[H2ChaseCamera] camera predicate prologue differs (another build, or already hooked by another tool) - refusing to hook";
			return false;
		}

		mHook->setWantsToBeAttached(true);   // ModuleHookManager attaches it now (module is loaded) and on every re-load
		if (!mHook->isHookInstalled())
		{
			mRefused = true;
			PLOG_ERROR << "[H2ChaseCamera] safetyhook could not install the hook";
			return false;
		}
		PLOG_INFO << "[H2ChaseCamera] camera predicate hook armed";
		return true;
	}

	void refuseToggle(const char* why)
	{
		sEnabled = false;
		if (auto settings = settingsWeak.lock())
		{
			// untick the checkbox without re-firing the event (same trick as UncapRenderSections / FarClipDistance)
			settings->h2ChaseCameraToggle->GetValueDisplay() = false;
			settings->h2ChaseCameraToggle->GetValue() = false;
		}
		say(why);
	}

	void onToggle(bool& on)
	{
		if (on)
		{
			sEnabled = true;
			if (ensureArmed())
				say("Chase Camera: on (takes effect on the next game update; unpause if paused)");
			else if (mRefused)
				refuseToggle("Chase Camera: unavailable - halo2.dll is not the 1.3528 build this was made for.");
			else
				PLOG_DEBUG << "[H2ChaseCamera] halo2.dll not loaded yet; will arm when Halo 2 is in-game";
		}
		else
		{
			// Hook stays installed but answers honestly from now on; the director swaps back to first person itself.
			sEnabled = false;
			say("Chase Camera: off");
		}
	}

	void onMCCStateChanged(const MCCState& s)
	{
		if (s.currentGameState != mGame) return;
		if (s.currentPlayState != PlayState::Ingame) return;
		if (!sEnabled.load()) return;
		if (!ensureArmed() && mRefused)
			refuseToggle("Chase Camera: unavailable - halo2.dll is not the 1.3528 build this was made for.");
	}

public:
	Impl(GameState game, IDIContainer& dicon)
		: mGame(game),
		settingsWeak(dicon.Resolve<SettingsStateAndEvents>()),
		messagesWeak(dicon.Resolve<IMessagesGUI>())
	{
		auto ptr = dicon.Resolve<PointerDataStore>().lock();
		auto h2ChaseCameraPredicateFunction = ptr->getData<std::shared_ptr<MultilevelPointer>>(nameof(h2ChaseCameraPredicateFunction), game);
		mPredicateFunction = h2ChaseCameraPredicateFunction;

		mHook = ModuleInlineHook::make(mGame.toModuleName(), mPredicateFunction, (void*)&predicateDetour);
		try
		{
			auto h2ChaseCameraFollowUpdateFunction = ptr->getData<std::shared_ptr<MultilevelPointer>>(nameof(h2ChaseCameraFollowUpdateFunction), game);
			mFollowUpdateFunction = h2ChaseCameraFollowUpdateFunction;
			mFollowHook = ModuleInlineHook::make(mGame.toModuleName(), mFollowUpdateFunction, (void*)&followUpdateDetour);
		}
		catch (HCMInitException& ex) { mFollowRefused = true; PLOG_ERROR << "[H2ChaseCamera] no-clip unavailable: " << ex.what(); }
		sInstance = this;

		auto settings = settingsWeak.lock();
		sNoClip = settings->h2ChaseCameraNoClip->GetValue();
		mFovSetting = settings->h2ChaseCameraFov;
		mHeightSetting = settings->h2ChaseCameraHeight;
		mNoClipCallback = std::make_unique<ScopedCallback<ToggleEvent>>(settings->h2ChaseCameraNoClip->valueChangedEvent,
			[](bool& on) { sNoClip = on; });
		mToggleCallback = std::make_unique<ScopedCallback<ToggleEvent>>(settings->h2ChaseCameraToggle->valueChangedEvent,
			[this](bool& on) { onToggle(on); });
		mStateCallback = std::make_unique<ScopedCallback<eventpp::CallbackList<void(const MCCState&)>>>(
			dicon.Resolve<IMCCStateHook>().lock()->getMCCStateChangedEvent(),
			[this](const MCCState& s) { onMCCStateChanged(s); });

		// toggle was already on when this cheat was (re)built (e.g. service rebuilt after a game switch)
		if (settings->h2ChaseCameraToggle->GetValue())
		{
			bool on = true;
			onToggle(on);
		}
	}

	~Impl()
	{
		// UNSUBSCRIBE FIRST so no toggle / state event can reach a half-destroyed object.
		if (mToggleCallback) mToggleCallback->removeCallback();
		if (mStateCallback) mStateCallback->removeCallback();
		if (mNoClipCallback) mNoClipCallback->removeCallback();
		mToggleCallback.reset();
		mStateCallback.reset();
		mNoClipCallback.reset();

		// Answer honestly before the hook goes away: the stored sentinel then differs from the real code and the
		// director swaps itself back to first person on its next update (if the hook is still live that long).
		sEnabled = false;
		if (mHook) mHook->setWantsToBeAttached(false);   // restores the original bytes
		mHook.reset();
		if (mFollowHook) mFollowHook->setWantsToBeAttached(false);
		mFollowHook.reset();
		sInstance = nullptr;
	}
};

// Only ever constructed for Halo 2: the GUI row that requires it is registered (Halo2) only.
H2ChaseCamera::H2ChaseCamera(GameState gameImpl, IDIContainer& dicon)
{
	switch (gameImpl)
	{
	case GameState::Value::Halo2:
		pimpl = std::make_unique<Impl>(gameImpl, dicon);
		break;
	default:
		throw HCMInitException("H2ChaseCamera not impl for this game (Halo 2 only)");
	}
}

H2ChaseCamera::~H2ChaseCamera() { PLOG_VERBOSE << "~" << getName(); }
