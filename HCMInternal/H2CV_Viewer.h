#pragma once
#include <d3d11.h>
#include <dxgi.h>
#include <atomic>
#include <deque>
#include <cstdint>
#include <string>
#include <vector>
#include "H2CV_Collision.h"
#include "H2CV_Renderer.h"

namespace h2cv
{
	// InFrame: drawn inside the game's frame at the first-person pass, against the game's real world depth (default).
	// Own: drawn at Present against the viewer's own collision depth (automatic fallback when InFrame can't run,
	//      e.g. remastered graphics). XRay: no occlusion.
	enum class Occlusion : int { InFrame = 0, PostBloom, Own, XRay, Count };
	const char* occlusionName(Occlusion o);

	struct Settings
	{
		std::atomic_bool show[kCatCount] = { true, false, true, true, false, false, false, false, true };
		std::atomic_bool fills{ true };            // translucent invisible-barrier and kill-trigger faces
		std::atomic<float> barrierOpacity{ 0.03f };
		std::atomic<float> killOpacity{ 0.04f };
		std::atomic_int occlusion{ (int)Occlusion::InFrame };
		std::atomic_bool gameZReversed{ true };   // F12 diagnostic: game depth stored as 1 - z/w (W1: reversed-Z on by default)
		std::atomic<float> lineWidth{ 2.0f };     // px
		std::atomic<float> pull{ 0.0015f };       // relative depth pull (0.15% of distance)
		std::atomic<float> hiddenAlpha{ 0.0f };   // 0 = hidden lines invisible
		std::atomic<float> nearRadius{ 60.0f };   // wu around the camera
		std::atomic_bool tim{ true };             // TIM overlay (player's level-pair children)
		std::atomic_bool timLabels{ true };
		std::atomic_bool pills{ true };           // Havok capsules of bipeds
		std::atomic_bool bipedOccluders{ true };  // capsules of OTHER bipeds/vehicles occlude lines
		std::atomic_bool hud{ true };
		std::atomic_bool creases{ false };             // static acute-crease PREDICTION (CollisionWorld::Crease) - disproved as a rule (90 deg corners ghost too); kept as a visual aid
		std::atomic_bool creaseFaces{ true };          // also outline both faces of each crease
		// ghost desync LOGGER: while Ghost debug runs, writes backstep / rewind / over-credit episodes (with the polygon,
		// direction, velocities and the surfaces being touched) to GhostDesyncLog_<date>.jsonl next to the module
		std::atomic_bool ghostLog{ false };
		// ---- ghosting desync debug view (Home / End, see dllmain.cpp) ----
		std::atomic_bool ghost{ false };                // master toggle [Home]
		std::atomic_bool ghostPrev{ true };             // (b) Havok step-start pose (motion PREV)
		std::atomic_bool ghostHalo{ true };             // (c) Halo's own object position
		std::atomic_bool ghostTrail{ true };            // (d) last N ticks of Havok poses            [End]
		std::atomic_bool ghostRewind{ true };           // (e/g) backstep rewind target + pre-rewind pose
		std::atomic_bool ghostBelieved{ true };         // (f) poses the over-credited TIMs believe     [Shift+Home]
		std::atomic_bool ghostEdges{ true };            // lime/red edges of skipped / over-credited polygons [Shift+End]
		std::atomic_bool ghostEdgesSuspectOnly{ false };// false = every skipped child (TIM>0), true = only stale / over-credited ones
		std::atomic_bool ghostStrictSkip{ false };      // true = ANY skipped child turns the capsule lime (very common on floors)
		std::atomic_int ghostTrailLen{ 8 };             // ticks
		std::atomic_int ghostHold{ 8 };                 // ticks a backstep keeps the capsule lime / the rewind target visible
		std::atomic_int ghostBelievedMax{ 6 };          // max believed-pose capsules per player
		std::atomic<float> ghostPrevMinDelta{ 0.001f }; // wu: draw (b) only when it differs from the current pose by more than this
		std::atomic<float> ghostHaloMinDelta{ 0.0005f };// wu: draw (c) only when Halo's position differs from Havok's by more than this
		std::atomic<float> ghostOverMin{ 0.005f };      // wu of TIM over-credit that counts as "suspect"
		std::atomic_bool ghostShowReal{ true };         // draw (a), the capsule where Havok really is (off = only the desync poses)
		std::atomic_bool ghostVehicles{ true };         // seated in a vehicle (any seat): show the vehicle's Havok body instead of the biped's
		// capsule colours, packed like rgba() (r | g<<8 | b<<16 | a<<24)
		std::atomic<uint32_t> ghostColNormal{ 0xFFB469FFu };   // (a) no backstep / nothing suspect  (pink 255,105,180)
		std::atomic<uint32_t> ghostColDesync{ 0xFF00FFAAu };   // (a) backstep / suspect child          (lime 170,255,0)
		std::atomic<uint32_t> ghostColBelieved{ 0xFF00FFAAu }; // (f) believed poses                   (lime)
	};

	class Viewer
	{
	public:
		Settings settings;

		// ---- ghost desync LOGGER (render thread only; JSON lines next to the module, see logEvent) ----
		struct LogKey { uint64_t k; bool operator==(const LogKey& o) const { return k == o.k; } };
		struct LogState { uint32_t startTick = 0, lastFrame = 0; float peakOver = 0.f; std::string startRec; };
		std::vector<std::pair<uint64_t, LogState>> mLogActive;   // over-credit episodes in progress
		FILE* mLogFile = nullptr;
		std::string mLogPath;
		uint32_t mLogFrame = 0, mLogLines = 0;
		bool logEnsureOpen();
		void logClose();
		void logEvent(const std::string& json);

		// called from HCM's ForegroundDirectXRenderEvent: inside Present, ImGui frame open, HCM's UI not drawn yet
		void onRender(ID3D11Device* dev, ID3D11DeviceContext* ctx, ID3D11RenderTargetView* rtv, uint32_t w, uint32_t h);
		void onFirstPersonPass(); // engine hook: entry of sub_1807E0C60 (render thread)
		void onResolveDone();     // engine hook: exit of sub_180951EC0 (bloom + composite done, HUD not yet)
		void restoreBloom();      // put the game's bloom gate back (unload / mode change)
		void beforeResize();
		void shutdown(); // render thread must be stopped first

		// "game depth" input set by the engine hooks (see EngineDepth)
		void setGameDepth(const GameDepthInput& g) { mGame = g; }

	private:
		bool initDevice(ID3D11Device* dev, ID3D11DeviceContext* ctx);
		void drawCollision(const h2::Vec3& camPos, bool depthTest);
		void drawTim();
		void drawPills();
		void drawGhost();
		void drawHud(const h2::Camera& cam);

		ID3D11Device* mDev = nullptr;
		ID3D11DeviceContext* mCtx = nullptr;
		ID3D11RenderTargetView* mRtv = nullptr;
		uint32_t mW = 0, mH = 0;
		bool mImgui = false;
		Renderer mR;
		CollisionWorld mWorld;
		GameDepthInput mGame;
		std::vector<LineVertex> mScratch;
		uint64_t mLastTime = 0;

		// ghost debug state, one slot per local user 0..3
		struct GhostSample { uint32_t tick = 0; h2::Vec3 cur{}, prev{}; int32_t S = 0; h2::Xf pose; };
		struct GhostPlayer
		{
			uintptr_t obj = 0; uint32_t lastTick = 0; bool have = false;
			std::deque<GhostSample> trail;
			uint32_t lastBackstepTick = 0; bool everBackstep = false; int32_t prevS = 0;
			bool haveRewind = false; uint32_t rewindTick = 0; h2::Vec3 rewindFrom{}, rewindTo{}; float rewindDist = 0, tStar = 0;
		};
		struct GhostInfo
		{
			bool valid = false; int user = 0; std::string note;
			std::string subject = "biped", detail; bool vehicle = false; int nPrims = 0, nWelders = 0;
			bool pairFound = false, welder = false; int32_t S = 0;
			int nChildren = 0, nSkipped = 0, nSuspect = 0; float maxOver = 0;
			float dOrigin = 0, dCom = 0, comOff = 0; bool haloRefCom = false; float prevDelta = 0;
			float rewindDist = 0, tStar = 0; int rewindAge = -1;
			bool lime = false, hold = false; int nBelieved = 0;
		};
		GhostPlayer mGP[4];
		GhostInfo mGI[4];

		// HUD stats
		struct TimRow { uint32_t key; int cat; uint32_t inst, surf; float tim; int manifold; uint8_t kind; float trueDist; };
		std::vector<TimRow> mTimRows;
		int32_t mPairS = 0; bool mPairFound = false; bool mPairWelder = false; uint32_t mTick = 0;
		std::string mStatus;
		// in-frame bookkeeping
		uint32_t mPresentCount = 0, mInFrameCount = 0, mLastInFramePresent = 0;
		bool mInFrameThisFrame = false;
		std::string mInFrameNote;
		float mLabelScaleX = 1.f, mLabelScaleY = 1.f;
		// post-bloom: per-view snapshot taken at the first-person pass, consumed after the resolve
		bool mPbPending = false; h2::GameFrame mPbFrame{}; D3D11_VIEWPORT mPbVp{};
		// bloom gate dword_180E19438 (single reader in sub_180951EC0; 0 = bloom on)
		bool mBloomTouched = false; uint32_t mBloomOrig = 0;
		bool drawViewInto(ID3D11RenderTargetView* rtv, ID3D11DepthStencilView* dsv, const h2::GameFrame& f, const D3D11_VIEWPORT& vp);
		void setBloomSuppressed(bool off);
	};
}
