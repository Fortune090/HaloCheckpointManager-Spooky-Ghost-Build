#include "pch.h"
#include "H2CV_Viewer.h"
#include "H2CV_Halo2.h"
#include "H2CV_Mem.h"
#include "H2CV_Log.h"
#include "H2CV_TimKeys.h"
#include "H2CV_ViewerUtil.h"
#include "H2CV_Engine.h"
#include <imgui.h>
#include <cmath>
#include <cstdio>
#include <algorithm>

namespace h2cv
{
	using h2::Vec3;
	using namespace vu;

	const char* occlusionName(Occlusion o)
	{
		switch (o)
		{
		case Occlusion::InFrame: return "in-frame, game depth - bloom OFF while shown (Cartographer style)";
		case Occlusion::PostBloom: return "after bloom, game depth copy + gun mask (bloom kept)";
		case Occlusion::Own: return "own collision depth (at Present)";
		case Occlusion::XRay: return "X-ray (no occlusion)";
		default: return "?";
		}
	}

	namespace
	{
		// what the ghost view is showing for one local user: the vehicle the unit is seated in, else the biped
		struct GhostSubject
		{
			bool ok = false, vehicle = false;
			uintptr_t obj = 0, ent = 0;     // object whose Havok body is shown (the vehicle, or the unit itself)
			std::string name, detail;
			h2::Motion m; h2::PairInfo pair; h2::BodyShape body; h2::Xf pose;
		};

		// Vehicle subject. Any seat counts (driver, passenger, gunner): the vehicle's body desyncs the same way whoever sits
		// in it. Picks the first DYNAMIC rigid body of the vehicle's havok component that has a (body, level) pair, trying the
		// primary record (comp+0x18) first; falls back to the primary dynamic body without a pair. sj.ok stays false (and
		// `why` says why) when anything is unreadable -- the caller then shows the biped.
		void buildVehicleSubject(uintptr_t unit, GhostSubject& sj, std::string& why)
		{
			uint32_t vdatum = 0;
			const uintptr_t veh = h2::vehicleOfUnit(unit, &vdatum);
			if (!veh) return;
			uint32_t grp = 0;
			const uint32_t tag = h2::objectDefinitionTag(veh, &grp);
			char fc[5] = { (char)(grp >> 24), (char)(grp >> 16), (char)(grp >> 8), (char)grp, 0 };
			for (int i = 0; i < 4; ++i) if (fc[i] < 32 || fc[i] > 126) fc[i] = '?';
			char nm[128];
			snprintf(nm, sizeof(nm), "vehicle [tag %04X '%s', object %08X]", tag & 0xFFFF, fc, vdatum);
			sj.name = nm;
			uintptr_t bodies[8];
			const int nb = h2::havokBodies(veh, bodies);
			if (nb <= 0) { why = "seated in a vehicle, but it has no Havok rigid body"; return; }
			int pick = -1, dyn = 0; h2::Motion pm; h2::PairInfo pp;
			for (int i = 0; i < nb; ++i)
			{
				h2::Motion m;
				if (!h2::readMotion(bodies[i], m, true)) continue;
				++dyn;
				const h2::PairInfo pr = h2::findBspPair(bodies[i]);
				if (pick < 0 || (pr.found && !pp.found)) { pick = i; pm = m; pp = pr; }
				if (pp.found) break;
			}
			if (pick < 0) { why = "seated in a vehicle, but none of its " + std::to_string(nb) + " Havok bodies has a live (non-fixed) motion"; return; }
			if (!h2::readBodyShape(bodies[pick], pp.found ? pp.agent : 0, sj.body)) { why = "seated in a vehicle, but its shape tree has no readable primitive (" + sj.body.desc + ")"; return; }
			sj.ok = true; sj.vehicle = true; sj.obj = veh; sj.ent = bodies[pick];
			sj.m = pm; sj.pair = pp; sj.pose = h2::motionPose(pm);
			char dt[200];
			snprintf(dt, sizeof(dt), "%s motion, body %d of %d (%d dynamic), %zu primitives; shape: %s", h2::motionKindName(pm.kind), pick, nb, dyn, sj.body.prims.size(), sj.body.desc.c_str());
			sj.detail = dt;
		}

		void buildBipedSubject(uintptr_t unit, GhostSubject& sj, std::string& note)
		{
			const uintptr_t ent = h2::havokEntityForObject(unit);
			if (!ent) { note = "no dynamic Havok body"; return; }
			h2::Motion m;
			if (!h2::readMotion(ent, m)) { note = "motion not a live stabilized-box (asleep / dead / seated in a vehicle that cannot be shown)"; return; }
			const h2::Capsule cap = h2::entityCapsuleLocal(ent);
			if (!cap.ok) { note = "capsule unreadable"; return; }
			const h2::PairInfo pair = h2::findBspPair(ent);
			h2::ShapePrim p; p.capsule = true; p.a = cap.a; p.b = cap.b; p.r = cap.r; p.hkType = 7; p.agent = pair.found ? pair.agent : 0;
			sj.body = h2::BodyShape{}; sj.body.prims.push_back(p);
			sj.ok = true; sj.vehicle = false; sj.obj = unit; sj.ent = ent; sj.name = "biped";
			sj.m = m; sj.pair = pair; sj.pose = h2::motionPose(m);
		}

	}

	bool Viewer::initDevice(ID3D11Device* dev, ID3D11DeviceContext* ctx)
	{
		// HCM owns the device, context and ImGui; the viewer only borrows them
		if (!dev || !ctx) return false;
		if (!mR.ready() && !mR.init(dev)) return false;
		mDev = dev; mCtx = ctx;
		LOGF("device %p ready", (void*)mDev);
		return true;
	}

	void Viewer::beforeResize() { mRtv = nullptr; }

	void Viewer::shutdown()
	{
		restoreBloom();
		mR.shutdown();
		mRtv = nullptr; mCtx = nullptr; mDev = nullptr;
	}

	void Viewer::drawCollision(const h2::Vec3& camPos, bool depthTest)
	{
		const float R = settings.nearRadius.load(), R2 = R * R;
		const bool own = depthTest, game = false;
		const h2::Camera cam = [&] { h2::Camera c; c.pos = camPos; return c; }();
		// translucent faces first (Cartographer's barrier / kill-trigger fill), depth-tested like the lines
		if (settings.fills)
		{
			auto fill = [&](const std::vector<Tri>& tris, uint32_t rgb, float opacity)
			{
				if (tris.empty() || opacity <= 0.f) return;
				const uint32_t col = (rgb & 0x00FFFFFFu) | ((uint32_t)(std::clamp(opacity, 0.f, 1.f) * 255.f) << 24);
				mScratch.clear();
				for (const auto& t : tris)
				{
					if (pointSegDist2(cam.pos, t.a, t.b) > R2 && pointSegDist2(cam.pos, t.b, t.c) > R2) continue;
					mScratch.push_back({ t.a.x, t.a.y, t.a.z, col });
					mScratch.push_back({ t.b.x, t.b.y, t.b.z, col });
					mScratch.push_back({ t.c.x, t.c.y, t.c.z, col });
				}
				mR.drawTris(mScratch.data(), mScratch.size(), own);
			};
			if (settings.show[(int)Cat::BspInvisible] || settings.show[(int)Cat::InstancedInvisible])
				fill(mWorld.barrierFills(), kCatColor[(int)Cat::BspInvisible], settings.barrierOpacity);
			if (settings.show[(int)Cat::KillTrigger])
				fill(mWorld.killTriggerFills(), kCatColor[(int)Cat::KillTrigger], settings.killOpacity);
		}
		mScratch.clear();
		for (int c = 0; c < kCatCount; ++c)
		{
			if (!settings.show[c]) continue;
			mWorld.forNear((Cat)c, cam.pos, R, [&](const Line& l)
			{
				if (pointSegDist2(cam.pos, l.a, l.b) > R2) return;
				addLine(mScratch, l.a, l.b, kCatColor[c]);
			});
		}
		mR.drawLines(mScratch.data(), mScratch.size(), settings.lineWidth, settings.pull, own, game, settings.hiddenAlpha);

		// desyncable creases: concave edges whose two faces meet at < 90 degrees (yellow = just under 90, red = very
		// acute), thick, over the normal lines; optional thin outline of both faces
		if (settings.creases)
		{
			auto heat = [](float sharp, uint8_t a) {
				const float t = std::clamp(sharp / 0.7f, 0.f, 1.f);              // nA.nB = -0.7 (~135 deg between normals) and sharper = red
				return rgba(255, (uint8_t)(230 - 200 * t), 0, a); };
			if (settings.creaseFaces)
			{
				mScratch.clear();
				for (const auto& l : mWorld.creaseFaces())
				{
					if (pointSegDist2(cam.pos, l.a, l.b) > R2) continue;
					addLine(mScratch, l.a, l.b, rgba(255, 200, 0, 150));
				}
				mR.drawLines(mScratch.data(), mScratch.size(), settings.lineWidth + 0.5f, settings.pull, own, game, settings.hiddenAlpha);
			}
			mScratch.clear();
			for (const auto& c : mWorld.creases())
			{
				if (pointSegDist2(cam.pos, c.edge.a, c.edge.b) > R2) continue;
				addLine(mScratch, c.edge.a, c.edge.b, heat(c.sharp, 255));
			}
			mR.drawLines(mScratch.data(), mScratch.size(), settings.lineWidth + 3.0f, settings.pull, own, game, settings.hiddenAlpha);
		}
	}

	void Viewer::drawPills()
	{
		std::vector<h2::ObjectRef> objs;
		h2::enumerateObjects(objs);
		const uintptr_t me = h2::localPlayerUnitObject();
		uintptr_t locals[4] = {};
		if (settings.ghost) h2::localPlayerUnits(locals);   // drawn (state-coloured) by drawGhost instead
		mScratch.clear();
		for (const auto& o : objs)
		{
			if (o.type != h2::Biped) continue;
			if (settings.ghost && (o.obj == locals[0] || o.obj == locals[1] || o.obj == locals[2] || o.obj == locals[3])) continue;
			const uintptr_t ent = h2::havokEntityForObject(o.obj);
			if (!ent) continue;
			const h2::Capsule c = h2::entityCapsule(ent);
			if (!c.ok) continue;
			capsuleWire(mScratch, c, o.obj == me ? rgba(255, 230, 0) : rgba(255, 255, 255, 200));
		}
		// pills are drawn through walls (like Cartographer's), thin
		mR.drawLines(mScratch.data(), mScratch.size(), 1.5f, 0.f, false, false, 0.f);
	}

	// ------------------------------------------------------------------------------------------------
	// GHOSTING DESYNC DEBUG VIEW (notes: Halo2_Ghosting_Analysis.md; read-only).
	// The subject per local user is the VEHICLE the unit is seated in (any seat) when its Havok body can be read, else the biped.
	//   pink/lime/red body = Havok's current pose, coloured by backstep state (capsule for a biped, the vehicle's shape
	//                        primitives -- boxes, convex hulls, spheres, capsules -- for a vehicle)
	//   thin state-coloured body = Havok step-start pose (motion PREV; Havok keeps no previous rotation, so current rotation)
	//   white body = where HALO's object position puts it (only when it differs from Havok)
	//   fading trail = Havok poses of the last N game ticks (with their own rotation)
	//   yellow body + red-orange pre-rewind body = last backstep rewind (target / where it was drawn)
	//   thin lime bodies + line = "believed" poses: where each over-credited TIM thinks the body is
	//   lime/red thick polygon edges = welder children currently skipped (TIM>0) / over-credited
	// ------------------------------------------------------------------------------------------------
	// ---- ghost desync logger ---------------------------------------------------------------------------------------
	// One JSON object per line, written next to the module that contains this code (HCMInternal.dll / h2cv.dll):
	// GhostDesyncLog_<yyyymmdd_hhmmss>.jsonl. Events: backstep (S 0 -> >0), rewind (detected tick rewind),
	// overcredit_start / overcredit_end (a polygon's TIM over-credit episode). Raw data for finding the ground truth.
	namespace
	{
	}

	bool Viewer::logEnsureOpen()
	{
		if (mLogFile) return true;
		if (!mLogPath.empty()) return false;          // opening failed once this session: don't retry every frame
		HMODULE self = nullptr;
		wchar_t dir[MAX_PATH] = {};
		if (GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
			(LPCWSTR)&vu::jv, &self) && GetModuleFileNameW(self, dir, MAX_PATH))
		{
			if (wchar_t* slash = wcsrchr(dir, L'\\')) slash[1] = 0;
		}
		SYSTEMTIME st; GetLocalTime(&st);
		wchar_t path[MAX_PATH + 64];
		swprintf(path, MAX_PATH + 64, L"%sGhostDesyncLog_%04d%02d%02d_%02d%02d%02d.jsonl", dir, st.wYear, st.wMonth, st.wDay, st.wHour, st.wMinute, st.wSecond);
		char narrow[MAX_PATH * 2] = {};
		WideCharToMultiByte(CP_UTF8, 0, path, -1, narrow, sizeof(narrow), nullptr, nullptr);
		mLogPath = narrow;
		if (_wfopen_s(&mLogFile, path, L"ab") != 0) { mLogFile = nullptr; LOGF("ghost logger: could not open %s", narrow); return false; }
		LOGF("ghost logger: writing %s", narrow);
		mLogLines = 0;
		return true;
	}

	void Viewer::logClose()
	{
		if (mLogFile)
		{
			// close every open episode so the file never ends mid-episode
			for (auto& e : mLogActive)
			{
				std::string r = "{\"event\":\"overcredit_end\",\"reason\":\"logger_off\",\"key64\":" + std::to_string(e.first) +
					",\"peak_over\":" + jf(e.second.peakOver) + "}\n";
				fwrite(r.data(), 1, r.size(), mLogFile);
			}
			fclose(mLogFile);
			LOGF("ghost logger: closed %s (%u lines)", mLogPath.c_str(), mLogLines);
		}
		mLogFile = nullptr; mLogPath.clear(); mLogActive.clear();
	}

	void Viewer::logEvent(const std::string& json)
	{
		if (!mLogFile) return;
		fwrite(json.data(), 1, json.size(), mLogFile);
		fputc('\n', mLogFile);
		fflush(mLogFile);
		++mLogLines;
	}

	void Viewer::drawGhost()
	{
		const uint32_t kLime = rgba(170, 255, 0), kRed = rgba(255, 40, 40);
		// user-selectable capsule colours (state colours of the real capsule + trail, and the believed poses)
		const uint32_t kPink = settings.ghostColNormal.load(), kDesync = settings.ghostColDesync.load(), kBelieved = settings.ghostColBelieved.load();
		const uint32_t kHalo = rgba(255, 255, 255), kRewindTo = rgba(255, 230, 0), kRewindFrom = rgba(255, 90, 0);
		constexpr float kTol = 0.0328f;

		const uint32_t tick = h2::gameTick();
		uintptr_t units[4];
		h2::localPlayerUnits(units);
		const int trailLen = std::clamp(settings.ghostTrailLen.load(), 0, 60);
		const uint32_t hold = (uint32_t)std::clamp(settings.ghostHold.load(), 0, 120);
		const int belMax = std::clamp(settings.ghostBelievedMax.load(), 0, 32);
		const float overMin = settings.ghostOverMin, prevMin = settings.ghostPrevMinDelta, haloMin = settings.ghostHaloMinDelta;

		std::vector<LineVertex> capsV, thinV, edgeLime, edgeRed;
		std::vector<Vec3> poly;
		auto withA = [](uint32_t c, uint8_t a) { return (c & 0x00FFFFFFu) | ((uint32_t)a << 24); };
		const Vec3 zero{ 0, 0, 0 };
		++mLogFrame;
		if (!settings.ghostLog && mLogFile) logClose();
		if (!settings.ghostLog) mLogPath.clear();          // re-enabling opens a fresh file
		const bool logOn = settings.ghostLog && logEnsureOpen();
		const std::string lvl = logOn ? eng::levelCode() : std::string();

		for (int u = 0; u < 4; ++u)
		{
			GhostInfo& gi = mGI[u];
			gi = GhostInfo{}; gi.user = u;
			GhostPlayer& gp = mGP[u];
			const uintptr_t unit = units[u];
			if (!unit) { gp = GhostPlayer{}; continue; }
			gi.valid = true;

			// ---- the subject: the vehicle the unit is riding in, else its biped -------------------------------
			GhostSubject sj;
			std::string vehWhy;
			if (settings.ghostVehicles) buildVehicleSubject(unit, sj, vehWhy);
			if (!sj.ok)
			{
				sj = GhostSubject{};
				std::string bipedWhy;
				buildBipedSubject(unit, sj, bipedWhy);
				if (!sj.ok)
				{
					gp = GhostPlayer{};
					gi.subject = "biped";
					gi.note = vehWhy.empty() ? bipedWhy : (vehWhy + "; biped: " + bipedWhy);
					continue;
				}
				gi.detail = vehWhy.empty() ? std::string() : (vehWhy + " -- showing the biped");
			}
			else gi.detail = sj.detail;
			gi.subject = sj.name; gi.vehicle = sj.vehicle; gi.nPrims = (int)sj.body.prims.size();
			const h2::Motion& m = sj.m;
			const h2::PairInfo& pair = sj.pair;
			const h2::BodyShape& body = sj.body;
			const h2::Xf& pose = sj.pose;
			bool newRewind = false;
			std::vector<std::pair<uint32_t, Vec3>> contacts;          // touching children (manifold > 0): key, cached normal
			std::vector<std::pair<uint64_t, std::string>> pendStart;  // over-credit episodes that started this frame
			auto head = [&](const char* ev) {
				return std::string("{\"event\":\"") + ev + "\",\"engine\":\"halo2\",\"tick\":" + std::to_string(tick) + ",\"map\":\"" + lvl +
					"\",\"user\":" + std::to_string(u) + ",\"subject\":\"" + (sj.vehicle ? "vehicle" : "biped") + "\",\"cur\":" + jv(m.cur) +
					",\"prev\":" + jv(m.prev) + ",\"v\":" + jv(m.v) + ",\"vint\":" + jv(m.vint) + ",\"S\":" + std::to_string(pair.found ? pair.S : 0); };
			if (gp.obj != sj.obj) { gp = GhostPlayer{}; gp.obj = sj.obj; }
			// anchor of the link lines and of the "which side of the polygon" test: the biped's capsule midpoint, a vehicle's centre of mass
			const Vec3 mid = sj.vehicle ? m.cur : primCentre(body.prims[0], pose);
			gi.pairFound = pair.found; gi.welder = pair.agentIsWelder; gi.S = pair.found ? pair.S : 0;
			if (pair.found && pair.S > 0) { gp.lastBackstepTick = tick; gp.everBackstep = true; }

			// ---- once per game tick: trail sample + rewind detection --------------------------------------
			// motion time (+0x8C) is 1.0 after integrate (sub_1809E05F0 end: `a1[8].m128_i32[3] = 1065353216`); anything else = mid-step
			if (std::fabs(m.time - 1.f) < 1e-4f && tick != 0 && tick != gp.lastTick)
			{
				if (gp.have && tick < gp.lastTick) { gp.trail.clear(); gp.have = false; gp.haveRewind = false; }
				GhostSample sm; sm.tick = tick; sm.cur = m.cur; sm.prev = m.prev; sm.S = gi.S; sm.pose = pose;
				if (gp.have && tick == gp.lastTick + 1 && !gp.trail.empty())
				{
					// integrate does PREV := CUR, so after a clean tick PREV_k == CUR_(k-1) exactly. A backstep setTime
					// (sub_1809DFEB0) moves CUR to P(0)+t*dt*vInt before that copy -> PREV_k = rewind target M.
					const GhostSample& p = gp.trail.back();
					const Vec3 d = sub(m.prev, p.cur);
					const float dist = len(d);
					if (dist > 1e-4f && dist < 1.0f)   // > 1 wu is a teleport, not a rewind
					{
						const Vec3 D = sub(p.cur, p.prev);   // the step the previous tick integrated: A -> B
						const float D2 = dot(D, D);
						gp.haveRewind = true; gp.rewindTick = tick; gp.rewindFrom = p.cur; gp.rewindTo = m.prev; gp.rewindDist = dist;
						gp.tStar = D2 > 1e-10f ? dot(sub(m.prev, p.prev), D) / D2 : 0.f;
						newRewind = true;
						gp.lastBackstepTick = tick; gp.everBackstep = true;
					}
				}
				gp.trail.push_back(sm);
				while (gp.trail.size() > 64) gp.trail.pop_front();
				gp.lastTick = tick; gp.have = true;
			}
			const bool holdActive = gp.everBackstep && tick >= gp.lastBackstepTick && tick - gp.lastBackstepTick <= hold;
			const bool rewindActive = gp.haveRewind && tick >= gp.rewindTick && tick - gp.rewindTick <= hold;
			gi.hold = holdActive;
			if (gp.haveRewind) { gi.rewindDist = gp.rewindDist; gi.tStar = gp.tStar; gi.rewindAge = (int)(tick - gp.rewindTick); }

			// ---- welder children of the (body, level) pair ---------------------------------------------------
			// One welder agent per primitive (a vehicle's list / transform agents fan out to one welder per convex leaf; the
			// biped's pair agent IS the welder). Primitives that share one agent (the spheres of a multi-sphere) are one group
			// and the true gap is the smallest of theirs.
			struct Bel { float over; Vec3 dir; };
			std::vector<Bel> bel;
			struct Group { uintptr_t agent; std::vector<const h2::ShapePrim*> prims; };
			std::vector<Group> groups;
			for (const auto& pr : body.prims)
			{
				if (h2::agentKind(pr.agent) != h2::AgentKind::Welder) continue;
				bool found = false;
				for (auto& g : groups) if (g.agent == pr.agent) { g.prims.push_back(&pr); found = true; break; }
				if (!found) groups.push_back({ pr.agent, { &pr } });
			}
			gi.nWelders = (int)groups.size();
			if (!pair.found) gi.note = std::string("no (") + (sj.vehicle ? "vehicle" : "player") + ", level) pair";
			else if (groups.empty())
			{
				if (!sj.vehicle) gi.note = "pair agent is not the welder (inactive / rebuilt)";
				else if (pair.kind == h2::AgentKind::Unknown) gi.note = "vehicle pair agent is not a list / transform / MOPP / welder agent (rebuilt or unsupported?): no TIM data";
				else gi.note = "vehicle pair agent tree has no welder leaf: no TIM data";
			}
			for (const auto& grp : groups)
			{
				std::vector<h2::Child> kids;
				if (!h2::readChildren(grp.agent, kids)) { gi.note = "children unreadable"; continue; }
				const Vec3 gMid = primCentre(*grp.prims[0], pose);
				for (const auto& k : kids)
				{
					if (k.kind != h2::Child::Sweep && k.kind != h2::Child::Gsk) continue;
					++gi.nChildren;
					const bool skipping = k.tim > 0 && k.manifold == 0;
					if (logOn && k.manifold > 0) contacts.push_back({ k.key, k.normal });
					if (!skipping) continue;   // only skipped children matter below (and the polygon lookup is the costly part)
					++gi.nSkipped;
					const KeyInfo ki = decodeKey(k.key);
					bool have = false;
					if (ki.kind == KeyInfo::WorldPolygon) have = mWorld.worldPolygon(ki.surface, poly);
					else if (ki.kind == KeyInfo::IgPolygon) have = mWorld.igPolygon(ki.instance, ki.surface, poly);
					if (!have || poly.size() < 3) continue;
					// same measure as drawTim(): true gap between the primitive's surface and the polygon surface
					float gap = 1e9f;
					for (const h2::ShapePrim* pr : grp.prims) gap = std::min(gap, primSurfaceDist(*pr, pose, poly));
					const float trueDist = gap - 0.01f;
					const float over = trueDist > -0.5f ? k.tim - (trueDist - 1.1f * kTol) : 0.f;
					const bool stale = trueDist < kTol;                 // GSK would report a contact, the cache still skips it
					const bool suspect = over > overMin || stale;
					if (suspect) ++gi.nSuspect;
					gi.maxOver = std::max(gi.maxOver, over);
					if (logOn && over > overMin)
					{
						const uint64_t key64 = ((uint64_t)u << 40) | ((uint64_t)(sj.vehicle ? 1 : 0) << 39) | (uint64_t)k.key;
						auto it = std::find_if(mLogActive.begin(), mLogActive.end(), [&](const auto& e) { return e.first == key64; });
						if (it != mLogActive.end()) { it->second.peakOver = std::max(it->second.peakOver, over); it->second.lastFrame = mLogFrame; }
						else
						{
							Vec3 n = norm(cross(sub(poly[1], poly[0]), sub(poly[2], poly[0])));
							const float side = dot(sub(sj.vehicle ? gMid : mid, poly[0]), n);
							if (side < 0) n = mul(n, -1.f);                          // the believed shift direction: away from the polygon, on the body's side
							Vec3 c{}; for (const auto& v : poly) c = add(c, v); c = mul(c, 1.f / (float)poly.size());
							std::string pv = "["; for (size_t i = 0; i < poly.size(); ++i) { if (i) pv += ","; pv += jv(poly[i]); } pv += "]";
							std::string sf;
							if (ki.kind == KeyInfo::WorldPolygon && ki.surface < mWorld.worldBsp().surfaces.size())
							{ const auto& su = mWorld.worldBsp().surfaces[ki.surface]; sf = ",\"flags\":" + std::to_string(su.flags) + ",\"material\":" + std::to_string(su.material) + ",\"breakable\":" + std::to_string(su.breakable); }
							const Vec3 step = sub(m.cur, m.prev);
							std::string rec = head("overcredit_start") + ",\"key\":" + std::to_string(k.key) + ",\"key64\":" + std::to_string(key64) +
								",\"poly_kind\":\"" + (ki.kind == KeyInfo::WorldPolygon ? "world" : ki.kind == KeyInfo::IgPolygon ? "instanced" : "other") +
								"\",\"instance\":" + std::to_string(ki.instance) + ",\"surface\":" + std::to_string(ki.surface) + sf +
								",\"child_kind\":\"" + (k.kind == h2::Child::Sweep ? "sweep" : "gsk") + "\",\"poly\":" + pv + ",\"centroid\":" + jv(c) +
								",\"normal\":" + jv(n) + ",\"cached_normal\":" + jv(k.normal) + ",\"tim\":" + jf(k.tim) + ",\"true_dist\":" + jf(trueDist) +
								",\"over\":" + jf(over) + ",\"believed_shift\":" + jv(mul(n, over)) + ",\"n_dot_vint\":" + jf(dot(n, m.vint)) +
								",\"n_dot_v\":" + jf(dot(n, m.v)) + ",\"n_dot_step\":" + jf(dot(n, step)) + ",\"stale\":" + (stale ? "true" : "false");
							pendStart.push_back({ key64, rec });
							LogState ls; ls.startTick = tick; ls.lastFrame = mLogFrame; ls.peakOver = over;
							mLogActive.push_back({ key64, ls });
						}
					}
					if (settings.ghostEdges && (suspect || !settings.ghostEdgesSuspectOnly))
					{
						auto& dst = suspect ? edgeRed : edgeLime;
						for (size_t i = 0; i < poly.size(); ++i) addLine(dst, poly[i], poly[(i + 1) % poly.size()], suspect ? kRed : kLime);
					}
					if (over > overMin && belMax > 0)
					{
						Vec3 n = norm(cross(sub(poly[1], poly[0]), sub(poly[2], poly[0])));
						const float side = dot(sub(sj.vehicle ? gMid : mid, poly[0]), n);
						if (std::fabs(side) < 1e-6f) n = norm(k.normal); else if (side < 0) n = mul(n, -1.f);   // away from the polygon, on the body's side
						bel.push_back({ over, n });
					}
				}
			}
			if (logOn)
			{
				std::string cj = "[";
				for (size_t i = 0; i < contacts.size(); ++i)
				{
					const KeyInfo ck = decodeKey(contacts[i].first);
					if (i) cj += ",";
					cj += "{\"key\":" + std::to_string(contacts[i].first) + ",\"poly_kind\":\"" + (ck.kind == KeyInfo::WorldPolygon ? "world" : ck.kind == KeyInfo::IgPolygon ? "instanced" : "other") +
						"\",\"instance\":" + std::to_string(ck.instance) + ",\"surface\":" + std::to_string(ck.surface) + ",\"normal\":" + jv(contacts[i].second) + "}";
				}
				cj += "]";
				for (const auto& ps : pendStart) logEvent(ps.second + ",\"contacts\":" + cj + "}");
				if (pair.found && pair.S > 0 && gp.prevS <= 0) logEvent(head("backstep") + ",\"contacts\":" + cj + "}");
				if (newRewind)
					logEvent(head("rewind") + ",\"rewind_from\":" + jv(gp.rewindFrom) + ",\"rewind_to\":" + jv(gp.rewindTo) + ",\"rewind_dist\":" + jf(gp.rewindDist) +
						",\"t_star\":" + jf(gp.tStar) + ",\"contacts\":" + cj + "}");
			}
			gp.prevS = pair.found ? pair.S : 0;
			gi.lime = gi.S > 0 || holdActive || gi.nSuspect > 0 || (settings.ghostStrictSkip && gi.nSkipped > 0);
			const uint32_t st = gi.S == -3 ? kRed : (gi.lime ? kDesync : kPink);
			auto drawBody = [&](std::vector<LineVertex>& dst, const h2::Xf& ps, const Vec3& shift, uint32_t col)
			{
				for (const auto& pr : body.prims) primWire(dst, pr, ps, shift, col);
			};

			// (a) Havok current pose (optional: off shows only the desync poses, without the overlap)
			if (settings.ghostShowReal) drawBody(capsV, pose, zero, st);
			// (b) Havok step-start pose = motion PREV (+0x70) shifted from CUR (+0x80); the previous rotation is not stored, so the current one
			{
				const Vec3 d = sub(m.prev, m.cur);
				gi.prevDelta = len(d);
				if (settings.ghostPrev && gi.prevDelta > prevMin && gi.prevDelta < 5.f) drawBody(thinV, pose, d, withA(st, 120));
			}
			// (c) Halo's object position (obj+0x64); the reference point (body origin +0xC0 vs centre of mass +0x80) is whichever is nearer
			{
				float hp[3];
				if (mem::copy(hp, sj.obj + 0x64, 12) && std::isfinite(hp[0]) && std::isfinite(hp[1]) && std::isfinite(hp[2]))
				{
					const Vec3 P{ hp[0], hp[1], hp[2] };
					gi.dOrigin = len(sub(P, m.origin)); gi.dCom = len(sub(P, m.cur)); gi.comOff = len(m.com);
					gi.haloRefCom = gi.dCom < gi.dOrigin;
					const Vec3 d = sub(P, gi.haloRefCom ? m.cur : m.origin);
					// a vehicle's Halo position is its node-0 origin, which need not be the origin of the body Havok moves (a body
					// attached to another node): a constant offset of more than half a world unit is that mismatch, not a desync
					const float haloMax = sj.vehicle ? 0.5f : 50.f;
					if (settings.ghostHalo && len(d) > haloMin && len(d) < haloMax) drawBody(capsV, pose, d, kHalo);
				}
			}
			// (d) trail of the last N tick poses (the newest sample is the current pose: skipped when identical)
			if (settings.ghostTrail && trailLen > 0)
			{
				const int n = (int)gp.trail.size(), first = std::max(0, n - trailLen - 1);
				for (int i = first; i < n; ++i)
				{
					const GhostSample& sm = gp.trail[i];
					const Vec3 d = sub(sm.cur, m.cur);
					if (len(d) < 5e-4f || len(d) > 20.f) continue;
					const float f = (float)(i - first + 1) / (float)(n - first);
					drawBody(thinV, sm.pose, zero, withA(sm.S > 0 ? kDesync : kPink, (uint8_t)(35 + 150 * f)));
				}
			}
			// (e/g) the last backstep rewind: where it was drawn (B) -> the rewind target (M = PREV after the step)
			if (settings.ghostRewind && rewindActive)
			{
				drawBody(thinV, pose, sub(gp.rewindFrom, m.cur), withA(kRewindFrom, 170));
				drawBody(capsV, pose, sub(gp.rewindTo, m.cur), kRewindTo);
				addLine(thinV, add(gp.rewindFrom, sub(mid, m.cur)), add(gp.rewindTo, sub(mid, m.cur)), kRewindTo);
			}
			// (f) believed poses: each over-credited child "thinks" the body is `over` farther along its normal
			if (settings.ghostBelieved && !bel.empty())
			{
				std::sort(bel.begin(), bel.end(), [](const Bel& x, const Bel& y) { return x.over > y.over; });
				if ((int)bel.size() > belMax) bel.resize((size_t)belMax);
				gi.nBelieved = (int)bel.size();
				for (const auto& b : bel)
				{
					const Vec3 d = mul(b.dir, b.over);
					// full-detail and solid like the real body (the user watches these); the link line stays thin
					drawBody(capsV, pose, d, kBelieved);
					addLine(thinV, add(mid, d), mid, withA(kBelieved, 130));
				}
			}
		}
		// drawn through walls (like the pills): thin/faded ones, then the solid capsules, then the thick polygon edges
		// over-credit episodes not seen this frame have ended (TIM dropped back / polygon left the welder / subject changed)
		if (logOn)
			for (auto it = mLogActive.begin(); it != mLogActive.end();)
			{
				if (it->second.lastFrame == mLogFrame) { ++it; continue; }
				logEvent("{\"event\":\"overcredit_end\",\"tick\":" + std::to_string(tick) + ",\"key64\":" + std::to_string(it->first) +
					",\"key\":" + std::to_string((uint32_t)(it->first & 0xFFFFFFFFu)) + ",\"user\":" + std::to_string((int)(it->first >> 40)) +
					",\"start_tick\":" + std::to_string(it->second.startTick) + ",\"ticks\":" + std::to_string((int64_t)tick - (int64_t)it->second.startTick) +
					",\"peak_over\":" + jf(it->second.peakOver) + "}");
				it = mLogActive.erase(it);
			}

		mR.drawLines(thinV.data(), thinV.size(), 1.3f, 0.f, false, false, 0.f);
		mR.drawLines(capsV.data(), capsV.size(), 2.5f, 0.f, false, false, 0.f);
		mR.drawLines(edgeLime.data(), edgeLime.size(), settings.lineWidth + 3.0f, 0.f, false, false, 0.f);
		mR.drawLines(edgeRed.data(), edgeRed.size(), settings.lineWidth + 4.5f, 0.f, false, false, 0.f);
	}

	void Viewer::drawTim()
	{
		mTimRows.clear(); mPairFound = false; mPairWelder = false;
		const uintptr_t obj = h2::localPlayerUnitObject();
		if (!obj) { mStatus = "no local player unit"; return; }
		const uintptr_t ent = h2::havokEntityForObject(obj);
		if (!ent) { mStatus = "player has no dynamic Havok body"; return; }
		const h2::PairInfo pair = h2::findBspPair(ent);
		mPairFound = pair.found; mPairS = pair.S; mPairWelder = pair.agentIsWelder;
		if (!pair.found) { mStatus = "no (player, level) pair"; return; }
		if (!pair.agentIsWelder) { mStatus = "pair agent is not the welder (inactive / rebuilt)"; return; }
		std::vector<h2::Child> kids;
		if (!h2::readChildren(pair.agent, kids)) { mStatus = "children unreadable"; return; }
		mStatus = "ok";
		const h2::Capsule cap = h2::entityCapsule(ent);

		mScratch.clear();
		std::vector<LineVertex> featLines;
		std::vector<Vec3> poly;
		for (const auto& k : kids)
		{
			TimRow row{ k.key, -1, 0, 0, k.tim, k.manifold, (uint8_t)k.kind, -1.f };
			const KeyInfo ki = decodeKey(k.key);
			row.cat = ki.category; row.inst = ki.instance; row.surf = ki.surface;
			bool have = false;
			if (ki.kind == KeyInfo::WorldPolygon) have = mWorld.worldPolygon(ki.surface, poly);
			else if (ki.kind == KeyInfo::IgPolygon) have = mWorld.igPolygon(ki.instance, ki.surface, poly);
			if (have && cap.ok) row.trueDist = segPolyDist(cap.a, cap.b, poly) - cap.r - 0.01f;
			mTimRows.push_back(row);
			if (!have || (k.kind != h2::Child::Sweep && k.kind != h2::Child::Gsk)) continue;

			const bool skipping = k.tim > 0 && k.manifold == 0;
			const float estTrueTim = row.trueDist - 1.1f * 0.0328f;
			const float over = (skipping && row.trueDist > -0.5f) ? k.tim - estTrueTim : 0.f;
			const uint32_t col = timColor(over, skipping, k.manifold);
			// which ring edge / vertex the last GSK run (the one that wrote the TIM) was closest to
			const int n = (int)poly.size();
			int featEdge = -1, featVert = -1;
			if (k.feature == 1 && k.featA >= 0 && k.featB >= 0 && k.featA < n && k.featB < n)
			{
				if (k.featB == (k.featA + 1) % n) featEdge = k.featA;
				else if (k.featA == (k.featB + 1) % n) featEdge = k.featB;
			}
			else if (k.feature == 0 && k.featA >= 0 && k.featA < n) featVert = k.featA;
			for (int i = 0; i < n; ++i)
			{
				const Vec3& a = poly[i]; const Vec3& b = poly[(i + 1) % n];
				addLine(mScratch, a, b, col);
				if (i == featEdge) addLine(featLines, a, b, rgba(255, 255, 255));
				if (!settings.timLabels) continue;
				const Vec3 mid = mul(add(a, b), 0.5f);
				Label L{}; L.col = col;
				if (mR.project(mid, L.x, L.y))
				{
					L.x *= mLabelScaleX; L.y *= mLabelScaleY;
					const char* mark = (i == featEdge) ? " <" : "";
					if (k.manifold > 0) snprintf(L.text, sizeof(L.text), "c%d%s", k.manifold, mark);
					else if (skipping) snprintf(L.text, sizeof(L.text), "%.3f%s%s", k.tim, over > 0.005f ? "!" : "", mark);
					else snprintf(L.text, sizeof(L.text), "%.3f%s", k.tim, mark);
					gLabels.push_back(L);
				}
			}
			if (featVert >= 0)
			{
				const Vec3& v = poly[featVert];
				const float s = 0.03f;
				addLine(featLines, Vec3{ v.x - s, v.y, v.z }, Vec3{ v.x + s, v.y, v.z }, rgba(255, 255, 255));
				addLine(featLines, Vec3{ v.x, v.y - s, v.z }, Vec3{ v.x, v.y + s, v.z }, rgba(255, 255, 255));
				addLine(featLines, Vec3{ v.x, v.y, v.z - s }, Vec3{ v.x, v.y, v.z + s }, rgba(255, 255, 255));
			}
		}
		// the touched / skipped polygons are the point of the overlay: draw them on top, thicker
		mR.drawLines(mScratch.data(), mScratch.size(), settings.lineWidth + 1.5f, 0.f, false, false, 0.f);
		mR.drawLines(featLines.data(), featLines.size(), settings.lineWidth + 4.0f, 0.f, false, false, 0.f);
	}

	void Viewer::drawHud(const h2::Camera& cam)
	{
		ImGui::SetNextWindowPos(ImVec2(12, 12), ImGuiCond_Always);
		ImGui::SetNextWindowBgAlpha(0.55f);
		if (ImGui::Begin("Collision Viewer##h2cv", nullptr, ImGuiWindowFlags_NoInputs | ImGuiWindowFlags_AlwaysAutoResize |
			ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoFocusOnAppearing | ImGuiWindowFlags_NoNav))
		{
			ImGui::Text("occlusion: %s", occlusionName((Occlusion)settings.occlusion.load()));
			if (!mInFrameNote.empty()) ImGui::TextColored(ImVec4(1, 0.75f, 0.2f, 1), "%s", mInFrameNote.c_str());
			ImGui::Text("width %.1fpx  pull %.4f  hidden %.2f  radius %.0f",
				settings.lineWidth.load(), settings.pull.load(), settings.hiddenAlpha.load(), settings.nearRadius.load());
			for (int c = 0; c < (int)Cat::SoftCeiling; ++c)   // soft ceilings are a Halo 3 / ODST category
			{
				const uint32_t col = kCatColor[c];
				ImGui::TextColored(ImVec4((col & 0xFF) / 255.f, ((col >> 8) & 0xFF) / 255.f, ((col >> 16) & 0xFF) / 255.f, 1),
					"%s %s  %zu", settings.show[c] ? "[x]" : "[ ]", catName((Cat)c), mWorld.lines((Cat)c).size());
				if (c % 3 != 2 && c != (int)Cat::SoftCeiling - 1) ImGui::SameLine();
			}
			ImGui::Separator();
			ImGui::Text("TIM overlay %s  labels %s  capsules %s", settings.tim ? "on" : "off",
				settings.timLabels ? "on" : "off", settings.pills ? "on" : "off");
			if (settings.tim)
			{
				ImGui::Text("player<->level pair: %s  S = %d  tick %u  (%s)", mPairFound ? "found" : "-", mPairS, mTick, mStatus.c_str());
				if (!mTimRows.empty() && ImGui::BeginTable("tim", 6, ImGuiTableFlags_SizingFixedFit))
				{
					ImGui::TableSetupColumn("key"); ImGui::TableSetupColumn("poly"); ImGui::TableSetupColumn("TIM");
					ImGui::TableSetupColumn("pts"); ImGui::TableSetupColumn("true d"); ImGui::TableSetupColumn("over");
					ImGui::TableHeadersRow();
					for (const auto& r : mTimRows)
					{
						const bool skip = r.tim > 0 && r.manifold == 0;
						const float over = (skip && r.trueDist > -0.5f) ? r.tim - (r.trueDist - 1.1f * 0.0328f) : 0.f;
						ImGui::TableNextRow();
						ImGui::TableNextColumn(); ImGui::Text("%08X", r.key);
						ImGui::TableNextColumn();
						if (r.cat == 1) ImGui::Text("bsp %u", r.surf);
						else if (r.cat == 5) ImGui::Text("ig %u/%u", r.inst, r.surf);
						else ImGui::Text("cat %d", r.cat);
						ImGui::TableNextColumn(); ImGui::Text(r.kind == h2::Child::Null ? "null" : "%.4f", r.tim);
						ImGui::TableNextColumn(); ImGui::Text("%d", r.manifold);
						ImGui::TableNextColumn(); if (r.trueDist > -0.5f) ImGui::Text("%.4f", r.trueDist); else ImGui::Text("-");
						ImGui::TableNextColumn();
						if (over > 0.005f) ImGui::TextColored(ImVec4(1, 0.2f, 0.2f, 1), "+%.4f", over); else ImGui::Text(skip ? "skip" : "");
					}
					ImGui::EndTable();
				}
			}
			ImGui::Separator();
			ImGui::Text("GHOST debug [Home] %s  believed poses [Shift+Home] %s  trail [End] %s  skipped edges [Shift+End] %s",
				settings.ghost ? "on" : "off", settings.ghostBelieved ? "on" : "off", settings.ghostTrail ? "on" : "off", settings.ghostEdges ? "on" : "off");
			if (settings.ghost)
			{
				for (const auto& g : mGI)
				{
					if (!g.valid) continue;
					const ImVec4 col = g.S == -3 ? ImVec4(1, 0.2f, 0.2f, 1) : (g.lime ? ImVec4(0.67f, 1, 0, 1) : ImVec4(1, 0.41f, 0.71f, 1));
					if (!g.note.empty() && !g.pairFound)
					{
						ImGui::TextColored(col, "P%d [%s]: %s", g.user, g.subject.c_str(), g.note.c_str());
						if (!g.detail.empty()) ImGui::TextDisabled("    %s", g.detail.c_str());
						continue;
					}
					ImGui::TextColored(col, "P%d [%s]  S=%d%s  skipped %d/%d  suspect %d  max over-credit %.4f wu  believed poses %d",
						g.user, g.subject.c_str(), g.S, g.hold && g.S <= 0 ? " (backstep hold)" : "", g.nSkipped, g.nChildren, g.nSuspect, g.maxOver, g.nBelieved);
					if (g.vehicle) ImGui::TextDisabled("    vehicle body: %d primitives, %d welder leaves with TIMs", g.nPrims, g.nWelders);
					ImGui::Text("    Havok-Halo: %.4f wu (vs %s; origin %.4f, COM %.4f, COM offset %.4f)   step-start delta %.4f wu",
						g.haloRefCom ? g.dCom : g.dOrigin, g.haloRefCom ? "COM" : "origin", g.dOrigin, g.dCom, g.comOff, g.prevDelta);
					if (g.rewindAge >= 0) ImGui::Text("    last rewind %.4f wu, t*=%.2f, %d ticks ago   (g = rewind target %s)", g.rewindDist, g.tStar, g.rewindAge,
						g.rewindAge <= settings.ghostHold ? "shown" : "expired; g = step-start pose");
					else ImGui::Text("    no rewind seen yet   (g = step-start pose)");
					if (!g.detail.empty()) ImGui::TextDisabled("    %s", g.detail.c_str());
					if (!g.note.empty()) ImGui::TextDisabled("    %s", g.note.c_str());
				}
				ImGui::TextDisabled("pink = no backstep   LIME = S>0 / hold / stale or over-credited child   RED = S=-3   white = Halo position   yellow = rewind target   red-orange = pre-rewind");
				ImGui::TextDisabled("seated in a vehicle (any seat) the view follows the VEHICLE's Havok body [%s]; on exit it returns to the biped", settings.ghostVehicles ? "on" : "off");
			}
			ImGui::Separator();
			ImGui::TextDisabled("options: HCM menu > Overlays > Collision Viewer");
		}
		ImGui::End();
		(void)cam;
	}

	namespace
	{
		DirectX::XMFLOAT4X4 gameViewProj(const h2::GameFrame& f, bool reversed)
		{
			// world_to_view = real_matrix4x3 {scale, forward, left, up, position}: view = s*(x*F + y*L + z*U) + P
			// (Cartographer matrix4x3_transform_point), then clip = view * projection (row-vector, standard z)
			const float s = f.w2v[0];
			const DirectX::XMMATRIX W(
				s * f.w2v[1], s * f.w2v[2], s * f.w2v[3], 0,
				s * f.w2v[4], s * f.w2v[5], s * f.w2v[6], 0,
				s * f.w2v[7], s * f.w2v[8], s * f.w2v[9], 0,
				f.w2v[10], f.w2v[11], f.w2v[12], 1);
			const DirectX::XMMATRIX M(f.proj);
			DirectX::XMFLOAT4X4 vp;
			DirectX::XMStoreFloat4x4(&vp, W * M);
			if (reversed) // the depth buffer stores 1 - z/w  ->  clip.z' = clip.w - clip.z
				for (int i = 0; i < 4; ++i) vp.m[i][2] = vp.m[i][3] - vp.m[i][2];
			return vp;
		}

		D3D11_VIEWPORT frameViewport(const h2::GameFrame& f)
		{
			// render_camera viewport_bounds: i16 top, left, bottom, right
			D3D11_VIEWPORT vp{};
			vp.TopLeftX = f.vp[1]; vp.TopLeftY = f.vp[0];
			vp.Width = (float)(f.vp[3] - f.vp[1]); vp.Height = (float)(f.vp[2] - f.vp[0]);
			vp.MinDepth = 0; vp.MaxDepth = 1;
			return vp;
		}

		bool comOk(IUnknown* p)
		{
			// the engine globals hold live COM pointers; make sure one is readable before calling through it
			if (!mem::okp(reinterpret_cast<uintptr_t>(p))) return false;
			const uintptr_t vt = mem::q(reinterpret_cast<uintptr_t>(p));
			return mem::okp(vt) && mem::okp(mem::q(vt));
		}
	}

	void Viewer::setBloomSuppressed(bool off)
	{
		if (isH3()) return;   // the bloom gate is a Halo 2 global
		const uintptr_t gate = h2::base() + 0xE19438; // dword_180E19438: sub_180951EC0 skips bloom (sub_18096FBF0) when non-zero
		if (!h2::base()) return;
		if (off)
		{
			if (!mBloomTouched) { mBloomOrig = mem::u32(gate, 0); mBloomTouched = true; }
			*reinterpret_cast<volatile uint32_t*>(gate) = 1;
		}
		else if (mBloomTouched)
		{
			*reinterpret_cast<volatile uint32_t*>(gate) = mBloomOrig;
			mBloomTouched = false;
		}
	}

	void Viewer::restoreBloom() { setBloomSuppressed(false); }

	bool Viewer::drawViewInto(ID3D11RenderTargetView* rtv, ID3D11DepthStencilView* dsv, const h2::GameFrame& f, const D3D11_VIEWPORT& vp)
	{
		const bool h3g = isH3();
		// Halo 2: the stored projection is standard-Z while the depth buffer holds 1 - z/w (flip applied in gameViewProj).
		// Halo 3 / ODST: the engine builds the projection for the depth convention in use (reversed when its byte is set) - used as stored.
		const bool reversed = h3g ? h3::depthReversed() : settings.gameZReversed.load();
		const bool wantObjects = settings.show[(int)Cat::Scenery] || settings.show[(int)Cat::Crate] || settings.show[(int)Cat::Machine];
		if (h3g) mWorld.setWantCreases(settings.creases);
		mWorld.update(wantObjects, settings.show[(int)Cat::Breakable]);
		if (!mWorld.valid()) return false;
		mR.setStandardZ(!reversed);
		if (!mR.beginExternal(mCtx, rtv, dsv, vp, gameViewProj(f, h3g ? false : reversed))) return false;
		mR.setStandardZ(!reversed);
		drawCollision(f.pos, true);
		if (h3g)
		{
			if (settings.tim) drawTim3();
			if (settings.pills) drawPills3();
			if (settings.ghost) drawGhost3();
		}
		else
		{
			if (settings.tim) drawTim();
			if (settings.pills) drawPills();
			if (settings.ghost) drawGhost();
		}
		mR.end();
		return true;
	}

	void Viewer::onFirstPersonPass()
	{
		if (!mDev || !mCtx || !mR.ready()) return;
		const int occ = settings.occlusion.load();
		if (occ != (int)Occlusion::InFrame) setBloomSuppressed(false);
		if (occ != (int)Occlusion::InFrame && occ != (int)Occlusion::PostBloom) return;
		h2::GameFrame f;
		if (!h2::readFrame(f) || f.textureCamera) return;
		const uintptr_t b = h2::base();
		auto* rtv = reinterpret_cast<ID3D11RenderTargetView*>(mem::q(b + h2::rva::SCENE_RTV));
		auto* dsv = reinterpret_cast<ID3D11DepthStencilView*>(mem::q(b + h2::rva::SCENE_DSV));
		if (!comOk(rtv) || !comOk(dsv)) return;
		const D3D11_VIEWPORT vp = frameViewport(f);
		// labels are drawn on the back buffer at Present: scale from scene-target to back-buffer pixels
		{
			ID3D11Resource* res = nullptr; rtv->GetResource(&res);
			ID3D11Texture2D* t = nullptr;
			if (res) { res->QueryInterface(__uuidof(ID3D11Texture2D), (void**)&t); res->Release(); }
			if (t)
			{
				D3D11_TEXTURE2D_DESC d{}; t->GetDesc(&d); t->Release();
				mLabelScaleX = (d.Width && mW) ? (float)mW / (float)d.Width : 1.f;
				mLabelScaleY = (d.Height && mH) ? (float)mH / (float)d.Height : 1.f;
			}
		}
		if (occ == (int)Occlusion::InFrame)
		{
			// Cartographer's recipe: draw into the scene target against the live world depth BEFORE the first-person pass
			// (the gun then paints over the lines, the HUD comes later) and switch the bloom stage off while we do.
			setBloomSuppressed(true);
			if (drawViewInto(rtv, dsv, f, vp)) { mInFrameThisFrame = true; ++mInFrameCount; }
		}
		else
		{
			// post-bloom: keep a copy of the complete world depth now; draw after the resolve/bloom
			if (mR.copyWorldDepth(mCtx, dsv)) { mPbFrame = f; mPbVp = vp; mPbPending = true; }
		}
	}

	void Viewer::onResolveDone()
	{
		if (!mPbPending) return;
		mPbPending = false;
		if (settings.occlusion.load() != (int)Occlusion::PostBloom || !mDev || !mCtx) return;
		const uintptr_t b = h2::base();
		auto* dsv = reinterpret_cast<ID3D11DepthStencilView*>(mem::q(b + h2::rva::SCENE_DSV));   // now holds first-person-only depth
		auto* rtv = reinterpret_cast<ID3D11RenderTargetView*>(mem::q(b + 0x197EE58));           // swap-chain RTV (composited scene)
		if (!comOk(dsv)) return;
		if (!comOk(rtv)) rtv = mRtv;
		if (!rtv) return;
		if (!mR.copyFpDepthAndMerge(mCtx, dsv, mPbVp, settings.gameZReversed.load())) return;
		// back buffer may be larger than the scene target (resolution scaling): scale the viewport to match
		D3D11_VIEWPORT vp = mPbVp;
		vp.TopLeftX *= mLabelScaleX; vp.Width *= mLabelScaleX; vp.TopLeftY *= mLabelScaleY; vp.Height *= mLabelScaleY;
		if (std::fabs(mLabelScaleX - 1.f) > 0.01f || std::fabs(mLabelScaleY - 1.f) > 0.01f)
		{
			mInFrameNote = "post-bloom: render scale != 100% - depth copy and back buffer differ in size; set Occlusion to InFrame_NoBloom";
			return;
		}
		const float sx = mLabelScaleX, sy = mLabelScaleY;
		mLabelScaleX = mLabelScaleY = 1.f;
		if (drawViewInto(rtv, mR.worldCopyDsv(), mPbFrame, vp)) { mInFrameThisFrame = true; ++mInFrameCount; }
		mLabelScaleX = sx; mLabelScaleY = sy;
	}

	void Viewer::onRender(ID3D11Device* dev, ID3D11DeviceContext* ctx, ID3D11RenderTargetView* rtv, uint32_t w, uint32_t h)
	{
		if (!mDev && !initDevice(dev, ctx)) return;
		if (!rtv || !w || !h) return;
		mRtv = rtv; mW = w; mH = h;

		++mPresentCount;
		mWantSim3 = isH3() && (settings.ghost || settings.tim);
		const bool gameOk = eng::base() && eng::buildOk();
		h2::Camera cam;
		if (gameOk) cam = isH3() ? camera3() : h2::readCamera();
		mTick = gameOk ? eng::gameTick() : 0;
		int occ = settings.occlusion.load();
		if (isH3() && occ == (int)Occlusion::PostBloom) occ = (int)Occlusion::Own;   // there is no post-bloom hook on Halo 3 / ODST
		const bool engineMode = occ == (int)Occlusion::InFrame || occ == (int)Occlusion::PostBloom;
		const bool engineDrew = mInFrameThisFrame;
		mInFrameThisFrame = false;
		if (engineDrew) mLastInFramePresent = mPresentCount;
		if (!engineMode) { mInFrameNote.clear(); setBloomSuppressed(false); }
		else if (!engineDrew && mPresentCount - mLastInFramePresent > 2)
		{
			// the hooks did not draw (remastered graphics, menu, cinematic, texture camera): fall back to own depth
			mInFrameNote = mInFrameCount == 0
				? "render hooks not drawing (remastered graphics? menu?) - showing own collision depth instead"
				: "render hooks idle this frame (cinematic / menu?) - showing own collision depth";
			setBloomSuppressed(false);
		}
		else mInFrameNote.clear();

		if (gameOk && cam.ok && !(engineMode && (engineDrew || mPresentCount - mLastInFramePresent <= 2)))
		{
			const bool own = occ != (int)Occlusion::XRay;
			const bool wantObjects = (own && !isH3()) || settings.show[(int)Cat::Scenery] || settings.show[(int)Cat::Crate] || settings.show[(int)Cat::Machine];
			if (isH3()) mWorld.setWantCreases(settings.creases);
			mWorld.update(wantObjects, settings.show[(int)Cat::Breakable]);
			mLabelScaleX = mLabelScaleY = 1.f;
			if (mWorld.valid() && mR.begin(mCtx, mRtv, mW, mH, cam.pos, cam.fwd, cam.up, cam.vfov, 0.01f))
			{
				if (own)
				{
					mR.clearOwnDepth();
					mR.setStaticOccluders(mWorld.staticOccluders(), mWorld.sbspId() ^ (mWorld.staticOccluders().size() << 1));
					mR.drawStaticOccluders();
					mR.drawOccluders(mWorld.objectOccluders());
					if (settings.bipedOccluders && isH3())
					{
						std::vector<Tri> caps;
						bipedOccluders3(caps, cam.pos);
						mR.drawOccluders(caps);
					}
					else if (settings.bipedOccluders)
					{
						// other bipeds (and the player's own capsule when the camera is outside it, e.g. third person)
						std::vector<Tri> caps;
						std::vector<h2::ObjectRef> objs;
						h2::enumerateObjects(objs);
						for (const auto& o : objs)
						{
							if (o.type != h2::Biped && o.type != h2::Vehicle) continue;
							const uintptr_t e = h2::havokEntityForObject(o.obj);
							if (!e) continue;
							const h2::Capsule c = h2::entityCapsule(e);
							if (!c.ok) continue;
							if (std::sqrt(pointSegDist2(cam.pos, c.a, c.b)) < c.r + 0.05f) continue; // camera inside it
							capsuleTris(caps, c);
						}
						mR.drawOccluders(caps);
					}
				}
				drawCollision(cam.pos, own);
				if (isH3())
				{
					if (settings.tim) drawTim3();
					if (settings.pills) drawPills3();
					if (settings.ghost) drawGhost3();
				}
				else
				{
					if (settings.tim) drawTim();
					if (settings.pills) drawPills();
					if (settings.ghost) drawGhost();
				}
				mR.end();
			}
		}

		{
			ImDrawList* dl = ImGui::GetBackgroundDrawList();
			for (const auto& L : gLabels)
			{
				const ImVec2 sz = ImGui::CalcTextSize(L.text);
				const ImVec2 p(L.x - sz.x * 0.5f, L.y - sz.y * 0.5f);
				dl->AddRectFilled(ImVec2(p.x - 2, p.y - 1), ImVec2(p.x + sz.x + 2, p.y + sz.y + 1), IM_COL32(0, 0, 0, 160), 3.f);
				dl->AddText(p, L.col | 0xFF000000u, L.text);
			}
			gLabels.clear();
			if (settings.hud)
			{
				if (!eng::base()) { dl->AddText(ImVec2(12, 12), IM_COL32(255, 200, 0, 255), "Collision Viewer: game module not loaded"); }
				else if (!eng::buildOk()) { dl->AddText(ImVec2(12, 12), IM_COL32(255, 60, 60, 255), "Collision Viewer: unsupported game build (needs MCC 1.3528)"); }
				else if (isH3()) drawHud3();
				else drawHud(cam);
			}
		}
	}
}
