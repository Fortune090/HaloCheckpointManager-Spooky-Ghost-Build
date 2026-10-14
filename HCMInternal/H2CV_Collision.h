#pragma once
// Collision geometry cache for Halo 2 MCC: world collision BSP, instanced geometry (IG), breakables, and dynamic
// object collision (scenery / crates / machines). Ported from HCM's wireframe-private CollisionWireframeOverlay
// (the "sorting" by collision type), plus what the new viewer adds: per-surface polygon outlines (for the TIM
// labels) and occluder triangles (for the collision-depth occlusion pass).
//
// collision_bsp layout (dual-pool tag blocks {u32 count, i32 dataOffset}):
//   surfaces @+0x28 (stride 8):  {plane u16, first_edge u16, flags u8, breakable i8, material u16}
//        flags: two_sided 1, invisible 2, climbable 4, breakable 8, invalid 0x10, conveyor 0x20
//   edges    @+0x30 (stride 0xC): {start_vertex, end_vertex, forward_edge, reverse_edge, left_surface, right_surface} u16
//   vertices @+0x38 (stride 0x10): xyz f32
#include "H2CV_Halo2.h"
#include <vector>
#include <unordered_map>
#include <cstdint>

namespace h2cv
{
	using h2::Vec3;

	enum class Cat : uint8_t { Bsp, BspInvisible, Breakable, Instanced, InstancedInvisible, Scenery, Crate, Machine, KillTrigger, Count };
	constexpr int kCatCount = (int)Cat::Count;
	const char* catName(Cat c);

	struct Line { Vec3 a, b; };
	struct Tri { Vec3 a, b, c; };

	// a parsed (copied) collision_bsp
	struct CollBsp
	{
		struct Surface { uint16_t plane, firstEdge; uint8_t flags; int8_t breakable; uint16_t material; };
		struct Edge { uint16_t v0, v1, fwd, rev, left, right; };
		std::vector<Surface> surfaces;
		std::vector<Edge> edges;
		std::vector<Vec3> verts;
		std::vector<float> planes;   // planes @+0x08 (stride 0x10): {i, j, k, d}; surface.plane bit 15 = flipped
		bool parse(uintptr_t collbsp, const h2::TagPools& p);
		// front-facing unit normal of a surface (its plane, negated when the plane index is flagged); false if malformed
		bool normal(uint32_t surface, Vec3& out) const;
		// ordered polygon outline of a surface (Halo edge-ring walk); false if malformed
		bool polygon(uint32_t surface, std::vector<Vec3>& out) const;
	};

	// IG instance placement: [0]=scale, [1..3]=forward, [4..6]=left, [7..9]=up, [10..12]=position
	struct IgInstance { float m[13]; int def; };
	Vec3 igTransform(const float* m, const Vec3& v);

	class CollisionWorld
	{
	public:
		// call once per frame from the render thread; cheap when nothing changed
		void update(bool wantObjects, bool wantBreakableLive);
		void invalidate() { mSbsp = 0; }

		const std::vector<Line>& lines(Cat c) const { return mLines[(int)c]; }
		const std::vector<Tri>& staticOccluders() const { return mStaticTris; }   // world + IG, non-invisible
		const std::vector<Tri>& objectOccluders() const { return mObjectTris; }   // scenery/crates/machines
		const std::vector<Tri>& barrierFills() const { return mBarrierTris; }     // invisible (*) surfaces, world + IG
		const std::vector<Tri>& killTriggerFills() const { return mKillTris; }    // scenario kill-trigger volume faces

		bool valid() const { return mSbsp != 0; }
		uintptr_t sbspId() const { return mSbsp; }   // changes on every BSP (re)load
		const CollBsp& worldBsp() const { return mWorld; }

		// polygon outline for a Havok BSP collection child (category decoded by the caller)
		bool worldPolygon(uint32_t surface, std::vector<Vec3>& out) const { return mWorld.polygon(surface, out); }
		bool igPolygon(uint32_t instance, uint32_t surface, std::vector<Vec3>& out) const;
		size_t igInstanceCount() const { return mIgInstances.size(); }

		size_t edgeCountTotal() const;

		// "desyncable creases": two collision surfaces sharing an edge whose fronts face each other at less than 90
		// degrees (concave AND nA.nB < 0). Ghosting model: a head-on bump into A rewinds along -nA, which over-credits
		// the TIM of any nearby polygon B with nB.(-nA) > 0, i.e. nA.nB < 0 - a flat floor meeting a vertical wall
		// (nA.nB = 0) can't be charged, a floor sloping down into the wall can. sharp = -nA.nB (0..1).
		struct Crease { Line edge; float sharp; };
		const std::vector<Crease>& creases() const { return mCreases; }
		const std::vector<Line>& creaseFaces() const { return mCreaseFaces; }   // outlines of both faces of each crease
		int planeOrientationAgree() const { return mPlaneAgree; }               // diagnostics: winding vs signed plane
		int planeOrientationDisagree() const { return mPlaneDisagree; }

	private:
		void rebuildStatic(uintptr_t sbsp);
		void gatherObjects();
		void classify(const CollBsp& bsp, const float* m, Cat normal, Cat invisible, bool withBreakable,
		              std::vector<Tri>* trisOut, std::vector<Tri>* invisTrisOut);

		uintptr_t mSbsp = 0;
		CollBsp mWorld;
		std::vector<CollBsp> mIgDefs;
		std::vector<IgInstance> mIgInstances;

		// edge bordering a breakable collision surface; left/right are surface indices (0xFFFF = none)
		struct BreakableEdge { Line l; uint16_t left, right; };
		std::vector<BreakableEdge> mBreakable;

		std::vector<Crease> mCreases;
		std::vector<Line> mCreaseFaces;
		int mPlaneAgree = 0, mPlaneDisagree = 0;
		void findCreases(const CollBsp& bsp, const float* m);   // m = IG placement, nullptr for the world bsp

		std::vector<Line> mLines[kCatCount];
		std::vector<Tri> mStaticTris, mObjectTris, mBarrierTris, mKillTris;
		void buildKillTriggers();

		// object collision, parsed from the engine's own walk (sub_180734FB0):
		//   coll tag: regions {count @+0x1C, offset @+0x20} stride 12 -> permutations {count @+4, offset @+8} stride 20
		//   -> bsps {count @+4, offset @+8} stride 0x44 {node_index u16 @0, collision_bsp @+4}
		struct ObjBsp { uint16_t node; std::vector<Line> lines; std::vector<Tri> tris; };
		struct ObjPerm { std::vector<ObjBsp> bsps; };
		struct ObjRegion { std::vector<ObjPerm> perms; };
		struct ObjColl { std::vector<ObjRegion> regions; };
		std::unordered_map<uintptr_t, ObjColl> mCollCache;     // coll tag data address -> geometry
		// model (hlmt) regions {count @+0x70, offset @+0x74} stride 16: collision_region_index i8 @+4, permutations
		// {count @+8, offset @+0xC} stride 8, collision_permutation_index i8 @+5 (sub_180793C40 / sub_180793760)
		struct ModelRegion { int8_t collRegion; std::vector<int8_t> collPerm; };
		struct ModelMap { std::vector<ModelRegion> regions; bool anyMapped = false; };
		std::unordered_map<uintptr_t, ModelMap> mModelCache;   // hlmt tag data address -> region mapping
		const ObjColl* loadColl(uintptr_t coll, const h2::TagPools& p);
		const ModelMap* loadModelMap(uintptr_t hlmt, const h2::TagPools& p);
		uint32_t mFrame = 0;
	};
}
