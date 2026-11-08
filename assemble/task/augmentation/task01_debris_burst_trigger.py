"""Triggered 56-rock falling-debris event for Task01."""
from __future__ import annotations

import hashlib
import json
import math
import os
import random
import time
import uuid
from pathlib import Path

from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

SOURCE_ROCK_NAMES = tuple(f"Rock_{i:02d}" for i in range(8))
INSTANCE_NAMES = tuple(f"Debris_{i:02d}" for i in range(56))
TOTAL = len(INSTANCE_NAMES)
INTERVAL = 0.12
STAGING_Z = -20.0
SCHEMA_VERSION = 2
CATEGORY_TOTALS = {"table": 16, "ground": 32, "contact": 8}
SIZE_TOTALS = {"large": 8, "medium": 20, "small": 28}
TABLE_HALF_SCALE_IDS = {"Debris_01", "Debris_23", "Debris_51"}
TABLE_PATH = "/World/envs/env_0/Worktable"
ROBOT_PATH = "/World/TrackedBase"
UPPER_BODY_PATH = f"{ROBOT_PATH}/G1UpperBody"
TASK_OBJECT_PATHS = (
    "/World/envs/env_0/Leiguan",
    "/World/envs/env_0/ChaganShengziHolder",
    "/World/envs/env_0/ChaganShengzi",
)
CONTACT_LINK_NAMES = tuple(
    f"{side}_{link}"
    for side in ("left", "right")
    for link in ("shoulder_pitch_link", "shoulder_roll_link", "shoulder_yaw_link", "elbow_link")
)


def _finite(name, default, low=None, high=None):
    value = float(os.environ.get(name, str(default)))
    if not math.isfinite(value) or (low is not None and value < low) or (high is not None and value > high):
        raise ValueError(f"{name} is outside its allowed range")
    return value


def _normalize_xy(vector, fallback):
    length = math.hypot(float(vector[0]), float(vector[1]))
    return fallback if length < 1e-6 else (float(vector[0]) / length, float(vector[1]) / length)


def _native_row(array, values):
    """Build a row on the tensor's own device and with its dtype.

    Isaac's local CPU view accepts Python lists, while the Quest runtime uses
    CUDA tensors and rejects assigning a host-side list into them.
    """
    return array.new_tensor(values) if hasattr(array, "new_tensor") else values


class TriggeredFallingDebris3D:
    ROOT_PATH = "/World/G1TriggeredDebris"

    def __init__(self, stage, *, seed=260822, prepare_only=False):
        self.stage = stage
        self._prepare_only = bool(prepare_only)
        self.base_seed = int(os.environ.get("G1_DEBRIS_SEED", seed))
        self.interval = _finite("G1_DEBRIS_INTERVAL", INTERVAL, 0.05, 2.0)
        self.height_scale = _finite("G1_DEBRIS_RELEASE_HEIGHT_SCALE", 1.0, 0.5, 2.0)
        self.release_height_offset = _finite("G1_DEBRIS_RELEASE_HEIGHT_OFFSET", .70, 0.0, 2.0)
        # Preserve the earlier, faster physical fall. These values are kept
        # separate because tabletop rocks use a little more damping to settle.
        self.fall_linear_damping = _finite("G1_DEBRIS_LINEAR_DAMPING", .12, 0.0, 20.0)
        self.table_linear_damping = _finite("G1_DEBRIS_TABLE_LINEAR_DAMPING", .55, 0.0, 20.0)
        self.request_path = Path(os.environ.get("G1_DEBRIS_BURST_REQUEST", "/tmp/g1_task01_debris_request.json"))
        self.status_path = Path(os.environ.get("G1_DEBRIS_BURST_STATUS", "/tmp/g1_task01_debris_status.json"))
        self.source_path = Path(os.environ.get(
            "G1_DEBRIS_SOURCE_USD",
            os.environ.get("BENCH_ASSEMBLE_ROOT", str(Path(__file__).resolve().parents[2])) + "/task/g1_tabletop_grasp_task/demo_mining_conditions/task01_falling_debris_grasp_editor.usda",
        )).expanduser()
        self.session, self.phase, self.trigger_id = uuid.uuid4().hex, "idle", ""
        self.seed, self.started, self.released, self.error = self.base_seed, 0.0, 0, ""
        self._seen_request, self._closed, self._status_at = "", False, 0.0
        self._rocks, self._view, self._source_stage, self._source_cache = [], None, None, {}
        self._released_counts = dict.fromkeys(CATEGORY_TOTALS, 0)
        self._settled_counts = dict.fromkeys(CATEGORY_TOTALS, 0)
        self._table_retained = 0
        self._scene_frame()

        existing = stage.GetPrimAtPath(self.ROOT_PATH)
        old_version = existing.GetAttribute("g1:debris:schemaVersion").Get() if existing and existing.IsValid() else None
        if existing and existing.IsValid() and old_version not in (None, SCHEMA_VERSION):
            stage.RemovePrim(Sdf.Path(self.ROOT_PATH))
        self._root = UsdGeom.Xform.Define(stage, self.ROOT_PATH).GetPrim()
        self._root.SetActive(True)
        self._root.CreateAttribute("g1:debris:schemaVersion", Sdf.ValueTypeNames.Int).Set(SCHEMA_VERSION)
        self._root.CreateAttribute("g1:debris:mode", Sdf.ValueTypeNames.Token).Set("triggered")
        self._root.CreateAttribute("g1:debris:source", Sdf.ValueTypeNames.Asset).Set(str(self.source_path))
        self._root.CreateAttribute("g1:debris:sourceRockNames", Sdf.ValueTypeNames.String).Set(",".join(SOURCE_ROCK_NAMES))
        self._root.CreateAttribute("g1:debris:total", Sdf.ValueTypeNames.Int).Set(TOTAL)
        self._root.CreateAttribute("g1:debris:interval", Sdf.ValueTypeNames.Float).Set(self.interval)
        self._visual_material = self._author_source_material()

        prepared = self._root.GetAttribute("g1:debris:prepared")
        if prepared and bool(prepared.Get()):
            self._attach_existing()
        else:
            specs = self._build_specs()
            self._rocks = [self._author_rock(index, spec) for index, spec in enumerate(specs)]
            self._author_collision_groups()
            self._root.CreateAttribute("g1:debris:prepared", Sdf.ValueTypeNames.Bool).Set(True)
        if len(self._rocks) != TOTAL:
            raise RuntimeError(f"prepared {len(self._rocks)} debris bodies; expected {TOTAL}")
        if not self._prepare_only:
            self._try_init_view()
            self._write_status(force=True)
        print(f"[debris-burst] {'prepared' if self._prepare_only else 'idle'} session={self.session} total={TOTAL}", flush=True)

    def _scene_frame(self):
        center, half = Gf.Vec3d(-3.057, 3.536, 0.420), Gf.Vec3d(0.625, 0.713, 0.421)
        table = self.stage.GetPrimAtPath(TABLE_PATH)
        try:
            if table and table.IsValid():
                bounds = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"]).ComputeWorldBound(table).ComputeAlignedRange()
                center, half = (bounds.GetMin() + bounds.GetMax()) * .5, (bounds.GetMax() - bounds.GetMin()) * .5
        except Exception as exc:
            print(f"[debris-burst] table bounds unavailable: {exc}", flush=True)
        self.table_center, self.table_half = Gf.Vec3d(center), Gf.Vec3d(half)
        self.table_min, self.table_max = Gf.Vec3d(center - half), Gf.Vec3d(center + half)
        self.table_top = float(self.table_max[2])
        robot_xy = (-1.2885, 2.4958)
        robot = self.stage.GetPrimAtPath(ROBOT_PATH)
        try:
            if robot and robot.IsValid():
                p = UsdGeom.Xformable(robot).ComputeLocalToWorldTransform(Usd.TimeCode.Default()).ExtractTranslation()
                robot_xy = (float(p[0]), float(p[1]))
        except Exception:
            pass
        self.robot_xy = robot_xy
        self.forward_xy = _normalize_xy((float(center[0]) - robot_xy[0], float(center[1]) - robot_xy[1]), (-.86, .51))
        self.lateral_xy = (-self.forward_xy[1], self.forward_xy[0])

    def _author_source_material(self):
        path = self.ROOT_PATH + "/RockMaterial"
        prim = self.stage.GetPrimAtPath(path)
        if not prim or not prim.IsValid():
            prim = self.stage.DefinePrim(path, "Material")
            prim.GetReferences().AddReference(str(self.source_path), "/World/G1FallingDebris/RockMaterial")
        return UsdShade.Material(prim)

    def _bounds(self, path):
        prim = self.stage.GetPrimAtPath(path)
        if not prim or not prim.IsValid():
            return None
        try:
            return UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"]).ComputeWorldBound(prim).ComputeAlignedRange()
        except Exception:
            return None

    def _table_targets(self):
        min_x, min_y = float(self.table_min[0]), float(self.table_min[1])
        max_x, max_y = float(self.table_max[0]), float(self.table_max[1])
        margin = .15
        xs = [min_x + margin + (max_x - min_x - 2 * margin) * i / 5 for i in range(6)]
        ys = [min_y + margin + (max_y - min_y - 2 * margin) * i / 4 for i in range(5)]
        forbidden = []
        for path in TASK_OBJECT_PATHS:
            bounds = self._bounds(path)
            if bounds is not None:
                forbidden.append((float(bounds.GetMin()[0]) - .12, float(bounds.GetMax()[0]) + .12,
                                  float(bounds.GetMin()[1]) - .12, float(bounds.GetMax()[1]) + .12))
        candidates = [(min(x - min_x, max_x - x, y - min_y, max_y - y), x, y)
                      for y in ys for x in xs
                      if not any(a <= x <= b and c <= y <= d for a, b, c, d in forbidden)]
        if len(candidates) < 16:
            candidates = [(min(x - min_x, max_x - x, y - min_y, max_y - y), x, y) for y in ys for x in xs]
        stable = sorted(candidates, reverse=True)[:12]
        selected = {(x, y) for _, x, y in stable}
        edge = [item for item in sorted(candidates) if (item[1], item[2]) not in selected][:4]
        return [(x, y, True) for _, x, y in stable] + [(x, y, False) for _, x, y in edge]

    def _ground_targets(self, side):
        sign, (fx, fy), (lx, ly) = (1 if side == "left" else -1), self.forward_xy, self.lateral_xy
        sizes = ("large", "medium", "small", "medium", "small", "large", "medium", "small",
                 "medium", "large", "small", "medium", "small", "large", "medium", "small")
        result = []
        for index, size_class in enumerate(sizes):
            row, column = divmod(index, 4)
            forward, lateral = .66 + .18 * row, .52 + .16 * column
            result.append((self.robot_xy[0] + fx * forward + sign * lx * lateral,
                           self.robot_xy[1] + fy * forward + sign * ly * lateral, size_class))
        return result

    def _build_specs(self):
        table = [{"category": "table", "side": "center", "size_class": ("medium" if i % 2 == 0 else "small"),
                  "target": Gf.Vec3f(x, y, self.table_top + .13), "release_height": (.56 if stable else .72) * self.height_scale,
                  "stable": stable, "contact_link": ""}
                 for i, (x, y, stable) in enumerate(self._table_targets())]
        ground = {side: [{"category": "ground", "side": side, "size_class": size_class,
                          "target": Gf.Vec3f(x, y, .04), "release_height": 1.58 * self.height_scale,
                          "stable": False, "contact_link": ""}
                         for x, y, size_class in self._ground_targets(side)] for side in ("left", "right")}
        contact = {side: [] for side in ("left", "right")}
        for side in contact:
            for link in ("shoulder_pitch_link", "shoulder_roll_link", "shoulder_yaw_link", "elbow_link"):
                path = f"{UPPER_BODY_PATH}/{side}_{link}"
                bounds = self._bounds(path)
                if bounds is None:
                    target = Gf.Vec3f(float(self.table_center[0]), float(self.table_center[1]), 1.20)
                else:
                    center = (bounds.GetMin() + bounds.GetMax()) * .5
                    target = Gf.Vec3f(float(center[0]), float(center[1]), float(bounds.GetMax()[2]))
                contact[side].append({"category": "contact", "side": side, "size_class": "small", "target": target,
                                      "release_height": .38 * self.height_scale, "stable": False, "contact_link": path})
        ordered = []
        for group in range(4):
            left, right, top = ground["left"][group * 4:(group + 1) * 4], ground["right"][group * 4:(group + 1) * 4], table[group * 4:(group + 1) * 4]
            ordered += [left[0], top[0], right[0], contact["left"][group], left[1], top[1], right[1], contact["right"][group],
                        left[2], top[2], right[2], left[3], top[3], right[3]]
        source_by_size = {"large": (0, 2, 3, 5, 0, 2, 3, 5),
                          "medium": tuple(i % 8 for i in range(20)), "small": tuple(i % 8 for i in range(28))}
        scales = {"large": (1.48, 1.56, 1.64, 1.54), "medium": (.94, 1.0, 1.06, 1.1), "small": (.56, .62, .68, .72)}
        counts = dict.fromkeys(SIZE_TOTALS, 0)
        for index, spec in enumerate(ordered):
            size_class, n = spec["size_class"], counts[spec["size_class"]]
            counts[size_class] += 1
            spec["instance_name"] = INSTANCE_NAMES[index]
            spec["source_name"] = SOURCE_ROCK_NAMES[source_by_size[size_class][n]]
            spec["scale"] = scales[size_class][n % len(scales[size_class])]
            # Desktop rocks should read as scattered debris, not boulders.
            # Keep the larger authored scale range on the floor only.
            if spec["category"] == "table":
                spec["scale"] *= .78
            # Only the three still-dominant desktop stones are reduced again;
            # the authored small stones keep their current visual size.
            if spec["instance_name"] in TABLE_HALF_SCALE_IDS:
                spec["scale"] *= .5

        # The prominent right-side Rock_03 exposes any contact gap from the
        # third-person camera. Swap it with the left-edge Rock_02; the latter
        # is one of the three half-scale table stones above.
        by_name = {spec["instance_name"]: spec for spec in ordered}
        by_name["Debris_29"]["target"], by_name["Debris_51"]["target"] = (
            by_name["Debris_51"]["target"], by_name["Debris_29"]["target"]
        )
        categories = {name: sum(s["category"] == name for s in ordered) for name in CATEGORY_TOTALS}
        if len(ordered) != TOTAL or categories != CATEGORY_TOTALS or counts != SIZE_TOTALS:
            raise RuntimeError(f"invalid debris layout: total={len(ordered)} categories={categories} sizes={counts}")
        return ordered

    def _source_info(self, name):
        if name in self._source_cache:
            return self._source_cache[name]
        default = (Gf.Vec3f(0, 0, 0), Gf.Vec3f(.16, .16, .16))
        try:
            self._source_stage = self._source_stage or Usd.Stage.Open(str(self.source_path))
            prim = self._source_stage.GetPrimAtPath(f"/World/G1FallingDebris/{name}")
            if not prim or not prim.IsValid():
                return default
            bounds = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"]).ComputeWorldBound(prim).ComputeAlignedRange()
            size = bounds.GetMax() - bounds.GetMin()
            result = (Gf.Vec3f(bounds.GetMin() + size * .5), Gf.Vec3f(max(float(size[0]), .04), max(float(size[1]), .04), max(float(size[2]), .04)))
            self._source_cache[name] = result
            return result
        except Exception as exc:
            print(f"[debris-burst] source bounds unavailable for {name}: {exc}", flush=True)
            return default

    @staticmethod
    def _mass(spec):
        return .028 if spec["category"] == "contact" else {"small": .055, "medium": .15, "large": .48}[spec["size_class"]]

    def _author_rock(self, index, spec):
        path = f"{self.ROOT_PATH}/{spec['instance_name']}"
        root = UsdGeom.Xform.Define(self.stage, path).GetPrim()
        xf = UsdGeom.Xformable(root)
        staging = Gf.Vec3f(spec["target"][0], spec["target"][1], STAGING_Z)
        translate, orient, scale_op = xf.AddTranslateOp(), xf.AddOrientOp(), xf.AddScaleOp()
        translate.Set(staging); orient.Set(Gf.Quatf(1.0, Gf.Vec3f(0, 0, 0))); scale_op.Set(Gf.Vec3f(spec["scale"], spec["scale"], spec["scale"]))
        rigid = UsdPhysics.RigidBodyAPI.Apply(root)
        rigid.CreateRigidBodyEnabledAttr().Set(True); rigid.CreateKinematicEnabledAttr().Set(False)
        rigid.CreateVelocityAttr().Set(Gf.Vec3f(0, 0, 0)); rigid.CreateAngularVelocityAttr().Set(Gf.Vec3f(0, 0, 0))
        UsdPhysics.MassAPI.Apply(root).CreateMassAttr(self._mass(spec))
        root.CreateAttribute("physxRigidBody:disableGravity", Sdf.ValueTypeNames.Bool).Set(True)
        root.CreateAttribute("physxRigidBody:enableSpeculativeCCD", Sdf.ValueTypeNames.Bool).Set(True)
        root.CreateAttribute("physxRigidBody:linearDamping", Sdf.ValueTypeNames.Float).Set(
            self.table_linear_damping if spec["category"] == "table" else self.fall_linear_damping
        )
        root.CreateAttribute("physxRigidBody:angularDamping", Sdf.ValueTypeNames.Float).Set(2.8 if spec["category"] == "table" else .45)
        attrs = {"index": index, "target": spec["target"], "sourceRock": spec["source_name"], "category": spec["category"],
                 "side": spec["side"], "sizeClass": spec["size_class"], "scale": spec["scale"], "releaseHeight": spec["release_height"],
                 "stableTableTarget": spec["stable"], "contactLink": spec["contact_link"]}
        types = {"index": Sdf.ValueTypeNames.Int, "target": Sdf.ValueTypeNames.Float3, "sourceRock": Sdf.ValueTypeNames.Token,
                 "category": Sdf.ValueTypeNames.Token, "side": Sdf.ValueTypeNames.Token, "sizeClass": Sdf.ValueTypeNames.Token,
                 "scale": Sdf.ValueTypeNames.Float, "releaseHeight": Sdf.ValueTypeNames.Float, "stableTableTarget": Sdf.ValueTypeNames.Bool,
                 "contactLink": Sdf.ValueTypeNames.String}
        for key, value in attrs.items(): root.CreateAttribute(f"g1:debris:{key}", types[key]).Set(value)
        center, size = self._source_info(spec["source_name"])
        if spec["category"] == "table":
            # Match the visual mesh's scaled lower bound directly to the
            # tabletop. The small clearance used previously still read as a
            # visible air gap in the third-person camera.
            target = spec["target"]
            spec = dict(spec)
            spec["target"] = Gf.Vec3f(float(target[0]), float(target[1]),
                                      self.table_top + max(.001, float(size[2]) * float(spec["scale"]) * .500))
            root.GetAttribute("g1:debris:target").Set(spec["target"])
        visual = UsdGeom.Xform.Define(self.stage, f"{path}/Visual").GetPrim()
        UsdGeom.Xformable(visual).AddTranslateOp().Set(-center)
        source = UsdGeom.Xform.Define(self.stage, f"{path}/Visual/Source").GetPrim()
        source.GetReferences().AddReference(str(self.source_path), f"/World/G1FallingDebris/{spec['source_name']}")
        UsdShade.MaterialBindingAPI(visual).Bind(self._visual_material, UsdShade.Tokens.strongerThanDescendants)
        shape = UsdGeom.Mesh.Define(self.stage, f"{path}/CollisionProxy")
        half = size * .5
        shape.CreatePointsAttr([Gf.Vec3f(x, y, z) for x in (-half[0], half[0]) for y in (-half[1], half[1]) for z in (-half[2], half[2])])
        faces = ((0,1,3,2),(4,6,7,5),(0,4,5,1),(2,3,7,6),(0,2,6,4),(1,5,7,3))
        shape.CreateFaceVertexCountsAttr([4] * 6); shape.CreateFaceVertexIndicesAttr([i for face in faces for i in face]); shape.CreatePurposeAttr().Set("guide")
        UsdPhysics.CollisionAPI.Apply(shape.GetPrim()).CreateCollisionEnabledAttr().Set(True)
        UsdPhysics.MeshCollisionAPI.Apply(shape.GetPrim()).CreateApproximationAttr().Set("convexHull")
        UsdGeom.Imageable(shape.GetPrim()).CreateVisibilityAttr().Set("invisible")
        return self._record(root, translate, orient, scale_op, spec, staging)

    def _record(self, root, translate, orient, scale_op, spec, staging):
        target = Gf.Vec3f(spec["target"])
        return {"root": root, "translate": translate, "orient": orient, "scale_op": scale_op, "instance_name": spec["instance_name"],
                "source_name": spec["source_name"], "category": spec["category"], "side": spec["side"], "size_class": spec["size_class"],
                "scale": float(spec["scale"]), "target": target, "release_height": float(spec["release_height"]), "stable": bool(spec["stable"]),
                "contact_link": spec["contact_link"], "release_position": Gf.Vec3f(target[0], target[1], target[2] + spec["release_height"]),
                "staging_position": Gf.Vec3f(staging), "release_orientation": Gf.Quatf(1.0, Gf.Vec3f(0,0,0)),
                "linear": Gf.Vec3f(0,0,0), "angular": Gf.Vec3f(0,0,0), "released": False, "released_at": None,
                "pinned": False}

    def _attach_existing(self):
        for name in INSTANCE_NAMES:
            root = self.stage.GetPrimAtPath(f"{self.ROOT_PATH}/{name}")
            if not root or not root.IsValid(): raise RuntimeError(f"prepared debris body is missing: {name}")
            ops = {op.GetOpName(): op for op in UsdGeom.Xformable(root).GetOrderedXformOps()}
            get = lambda key: root.GetAttribute(f"g1:debris:{key}").Get()
            spec = {"instance_name": name, "source_name": str(get("sourceRock")), "category": str(get("category")), "side": str(get("side")),
                    "size_class": str(get("sizeClass")), "scale": float(get("scale")), "target": Gf.Vec3f(get("target")),
                    "release_height": float(get("releaseHeight")), "stable": bool(get("stableTableTarget")), "contact_link": str(get("contactLink") or "")}
            target = spec["target"]
            self._rocks.append(self._record(root, ops["xformOp:translate"], ops["xformOp:orient"], ops["xformOp:scale"], spec, Gf.Vec3f(target[0],target[1],STAGING_Z)))

    def _group(self, name, paths):
        group = UsdPhysics.CollisionGroup.Define(self.stage, f"{self.ROOT_PATH}/{name}")
        includes = Usd.CollectionAPI.Apply(group.GetPrim(), "colliders").CreateIncludesRel()
        for path in paths:
            prim = self.stage.GetPrimAtPath(path)
            if prim and prim.IsValid(): includes.AddTarget(Sdf.Path(path))
        return group

    @staticmethod
    def _filter(a, b):
        a.CreateFilteredGroupsRel().AddTarget(b.GetPath()); b.CreateFilteredGroupsRel().AddTarget(a.GetPath())

    def _author_collision_groups(self):
        try:
            normal_paths = [f"{self.ROOT_PATH}/{r['instance_name']}" for r in self._rocks if r["category"] != "contact"]
            normal = self._group("NormalDebrisCollisionGroup", normal_paths)
            all_protected = self._group("AllProtectedCollisionGroup", (ROBOT_PATH,) + TASK_OBJECT_PATHS)
            protected = list(TASK_OBJECT_PATHS)
            robot = self.stage.GetPrimAtPath(ROBOT_PATH)
            upper = self.stage.GetPrimAtPath(UPPER_BODY_PATH)
            if robot and robot.IsValid(): protected += [str(p.GetPath()) for p in robot.GetChildren() if p.GetPath() != Sdf.Path(UPPER_BODY_PATH)]
            if upper and upper.IsValid(): protected += [str(p.GetPath()) for p in upper.GetChildren() if p.GetName() not in CONTACT_LINK_NAMES]
            self._filter(normal, all_protected)
            # A PhysX collider may belong to only one collision group.  The
            # eight contact stones therefore use direct filtered pairs, so the
            # robot/task colliders do not need a second, overlapping group.
            for rock in self._rocks:
                if rock["category"] != "contact":
                    continue
                proxy = self.stage.GetPrimAtPath(f"{rock['root'].GetPath()}/CollisionProxy")
                pairs = UsdPhysics.FilteredPairsAPI.Apply(proxy).CreateFilteredPairsRel()
                for path in protected:
                    pairs.AddTarget(Sdf.Path(path))
            self._root.CreateAttribute("g1:debris:collisionFilter", Sdf.ValueTypeNames.String).Set("normal protects robot/task; contact admits shoulders and upper arms")
        except Exception as exc:
            print(f"[debris-burst] collision groups unavailable: {exc}", flush=True)

    def _try_init_view(self):
        try:
            from isaacsim.core.prims import RigidPrim
            view = RigidPrim(prim_paths_expr=f"{self.ROOT_PATH}/Debris_.*", name="task01_debris_bodies", reset_xform_properties=False)
            view.initialize()
            if view.is_physics_handle_valid() and view.count == TOTAL: self._view = view
            else: print(f"[debris-burst] tensor view count={getattr(view,'count',None)} expected={TOTAL}", flush=True)
        except Exception as exc:
            print(f"[debris-burst] tensor view unavailable: {exc}", flush=True)

    def _set_velocity_zero(self):
        if self._view is not None:
            try:
                values = self._view.get_velocities(clone=True); values[...] = 0.; self._view.set_velocities(values)
            except Exception: pass

    def _refresh_contact_targets(self):
        cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"])
        for rock in self._rocks:
            if rock["category"] != "contact": continue
            prim = self.stage.GetPrimAtPath(rock["contact_link"])
            if not prim or not prim.IsValid(): continue
            try:
                bounds = cache.ComputeWorldBound(prim).ComputeAlignedRange(); center = (bounds.GetMin() + bounds.GetMax()) * .5
                rock["target"] = Gf.Vec3f(float(center[0]), float(center[1]), float(bounds.GetMax()[2]))
            except Exception: pass

    def _trigger(self, trigger, delay):
        if self.phase != "idle": raise ValueError(f"event already {self.phase}; reset Task01 before another trigger")
        if not isinstance(trigger, str) or not trigger: raise ValueError("trigger must be a nonempty string")
        if not math.isfinite(delay) or not 0 <= delay <= 60: raise ValueError("delay must be 0..60 seconds")
        self.trigger_id = trigger
        try: self.seed = uuid.UUID(trigger).int & 0xFFFFFFFF
        except ValueError: self.seed = int.from_bytes(hashlib.sha256(trigger.encode()).digest()[:4], "big")
        rng = random.Random(self.seed); self._refresh_contact_targets()
        for rock in self._rocks:
            axis = Gf.Vec3d(rng.uniform(-1,1),rng.uniform(-1,1),rng.uniform(-1,1))
            if axis.GetLength() < 1e-6: axis = Gf.Vec3d(0,0,1)
            rock["release_orientation"] = Gf.Quatf(Gf.Rotation(axis.GetNormalized(), rng.uniform(0,360)).GetQuat())
            spin = 1.6 if rock["category"] == "table" and rock["stable"] else (2.2 if rock["category"] == "contact" else 3.2)
            rock["angular"] = Gf.Vec3f(rng.uniform(-spin,spin),rng.uniform(-spin,spin),rng.uniform(-spin,spin))
            jitter = {"table": .025 if rock["stable"] else .04, "ground": .075, "contact": .012}[rock["category"]]
            p = rock["target"]
            rock["release_position"] = Gf.Vec3f(
                float(p[0]) + rng.uniform(-jitter, jitter),
                float(p[1]) + rng.uniform(-jitter, jitter),
                float(p[2]) + rock["release_height"] + self.release_height_offset,
            )
            rock["linear"] = Gf.Vec3f(0,0,-.05 if rock["category"] == "table" else -.12)
            rock["released"], rock["released_at"], rock["pinned"] = False, None, False
            rock["root"].GetAttribute("physxRigidBody:disableGravity").Set(True)
            UsdPhysics.RigidBodyAPI(rock["root"]).CreateKinematicEnabledAttr().Set(False)
        self.started, self.released, self.phase = time.monotonic()+delay, 0, ("waiting" if delay > 0 else "playing")
        self._released_counts, self._settled_counts, self._table_retained = dict.fromkeys(CATEGORY_TOTALS,0), dict.fromkeys(CATEGORY_TOTALS,0), 0

    def _release_one(self, index):
        rock = self._rocks[index]
        try:
            if self._view is not None:
                positions, orientations = self._view.get_world_poses(clone=True); p,q = rock["release_position"],rock["release_orientation"]
                positions[index] = _native_row(positions, [float(p[0]),float(p[1]),float(p[2])]); qi = q.GetImaginary(); orientations[index] = _native_row(orientations, [float(q.GetReal()),float(qi[0]),float(qi[1]),float(qi[2])])
                self._view.set_world_poses(positions=positions, orientations=orientations)
                velocities = self._view.get_velocities(clone=True); linear,angular = rock["linear"],rock["angular"]
                velocities[index,:3] = _native_row(velocities, [float(linear[0]),float(linear[1]),float(linear[2])]); velocities[index,3:] = _native_row(velocities, [float(angular[0]),float(angular[1]),float(angular[2])])
                self._view.set_velocities(velocities)
            else:
                rock["translate"].Set(rock["release_position"]); rock["orient"].Set(rock["release_orientation"])
        except Exception as exc:
            self.error = f"failed to place {rock['instance_name']}: {exc}"; print(f"[debris-burst] {self.error}", flush=True); return False
        rock["root"].GetAttribute("physxRigidBody:disableGravity").Set(False)
        rock["released"], rock["released_at"] = True, time.monotonic(); self.released = index + 1; self._released_counts[rock["category"]] += 1
        print(f"[debris-burst] released {rock['instance_name']} category={rock['category']} size={rock['size_class']} ({self.released}/{TOTAL})", flush=True)
        return True

    def _stabilize_table_rocks(self):
        """Pin the 12 central table drops after a visible physical fall.

        The real trolley's tabletop collider is thin and its edge slopes make
        free rolling unpredictable.  Keeping central stones at their settled
        poses gives the requested persistent tabletop obstruction while the
        release and first impact remain native PhysX.
        """
        now = time.monotonic()
        indexes = [i for i, rock in enumerate(self._rocks)
                   if rock["category"] == "table" and rock["stable"] and rock["released"]
                   and not rock["pinned"] and now - rock["released_at"] >= .35]
        if not indexes:
            return
        if self._view is not None:
            try:
                positions, orientations = self._view.get_world_poses(clone=True)
                velocities = self._view.get_velocities(clone=True)
                ready = []
                for index in indexes:
                    rock = self._rocks[index]
                    # Do not snap a rock to the table while it is still high
                    # in its descent; that was the source of the apparent
                    # first-frame "pop" for early tabletop releases.
                    if float(positions[index, 2]) <= float(rock["target"][2]) + .08:
                        ready.append(index)
                for index in ready:
                    rock = self._rocks[index]
                    p = rock["target"]
                    positions[index] = _native_row(positions, [float(p[0]), float(p[1]), float(p[2])])
                    velocities[index, :] = _native_row(velocities, [0.0] * 6)
                    rock["root"].GetAttribute("physxRigidBody:disableGravity").Set(True)
                    UsdPhysics.RigidBodyAPI(rock["root"]).CreateKinematicEnabledAttr().Set(True)
                    rock["pinned"] = True
                self._view.set_world_poses(positions=positions, orientations=orientations)
                self._view.set_velocities(velocities)
                return
            except Exception as exc:
                self.error = f"failed to settle table debris: {exc}"
        for index in indexes:
            rock = self._rocks[index]
            rock["translate"].Set(rock["target"])
            rock["root"].GetAttribute("physxRigidBody:disableGravity").Set(True)
            UsdPhysics.RigidBodyAPI(rock["root"]).CreateKinematicEnabledAttr().Set(True)
            rock["pinned"] = True

    def _update_observed_counts(self):
        if self._view is None: return
        try:
            import numpy as np
            positions,_ = self._view.get_world_poses(clone=True); velocities = self._view.get_velocities(clone=True)
            if hasattr(positions,"detach"): positions,velocities = positions.detach().cpu().numpy(),velocities.detach().cpu().numpy()
            settled, retained, now = dict.fromkeys(CATEGORY_TOTALS,0),0,time.monotonic()
            for i,rock in enumerate(self._rocks):
                if not rock["released"] or now-rock["released_at"] < .75: continue
                is_settled = float(np.linalg.norm(velocities[i,:3])) < .30
                if is_settled: settled[rock["category"]] += 1
                x,y,z = map(float, positions[i,:3])
                if rock["category"] == "table" and is_settled and float(self.table_min[0])+.015 <= x <= float(self.table_max[0])-.015 and float(self.table_min[1])+.015 <= y <= float(self.table_max[1])-.015 and self.table_top-.03 <= z <= self.table_top+.35: retained += 1
            self._settled_counts, self._table_retained = settled,retained
        except Exception: pass

    def update(self, dt):
        del dt
        if self._closed or self._prepare_only: return
        try:
            raw = self.request_path.read_text()
            if raw != self._seen_request:
                self._seen_request = raw; req = json.loads(raw)
                if req.get("session") != self.session: raise ValueError("request belongs to a different simulator session")
                if req.get("action","trigger") != "trigger": raise ValueError("unknown debris action")
                self._trigger(req.get("trigger"),float(req.get("delay",0.))); self.error = ""
        except FileNotFoundError: pass
        except (ValueError,TypeError,KeyError,OSError,json.JSONDecodeError) as exc: self.error = str(exc)
        now = time.monotonic()
        if self.phase in {"waiting","playing"} and now >= self.started:
            self.phase = "playing"; target_count = min(TOTAL,1+int((now-self.started)/self.interval))
            while self.released < target_count:
                if not self._release_one(self.released): break
            self._stabilize_table_rocks()
            self._update_observed_counts()
            if self.released == TOTAL and now-self.started >= (TOTAL-1)*self.interval+4.:
                self.phase = "settled"; self._update_observed_counts()
        self._write_status()

    def reset(self):
        for rock in self._rocks:
            rock["root"].GetAttribute("physxRigidBody:disableGravity").Set(True); UsdPhysics.RigidBodyAPI(rock["root"]).CreateKinematicEnabledAttr().Set(False); rock["translate"].Set(rock["staging_position"]); rock["orient"].Set(Gf.Quatf(1.,Gf.Vec3f(0,0,0))); rock["released"],rock["released_at"],rock["pinned"] = False,None,False
        if self._view is not None:
            try:
                positions,orientations = self._view.get_world_poses(clone=True)
                for i,rock in enumerate(self._rocks):
                    p=rock["staging_position"]; positions[i]=_native_row(positions,[float(p[0]),float(p[1]),float(p[2])]); orientations[i]=_native_row(orientations,[1.,0.,0.,0.])
                self._view.set_world_poses(positions=positions,orientations=orientations)
            except Exception as exc: self.error=f"failed to park debris on reset: {exc}"
        self._set_velocity_zero(); self.phase,self.trigger_id,self.released,self.error = "idle","",0,""; self._released_counts,self._settled_counts,self._table_retained = dict.fromkeys(CATEGORY_TOTALS,0),dict.fromkeys(CATEGORY_TOTALS,0),0; self._write_status(force=True)

    def _write_status(self, force=False):
        now=time.monotonic()
        if not force and now-self._status_at < .15: return
        self.status_path.parent.mkdir(parents=True,exist_ok=True)
        payload={"session":self.session,"phase":self.phase,"trigger":self.trigger_id,"wall_time":time.time(),"seed":int(self.seed),"released":self.released,"total":TOTAL,"interval":self.interval,"released_by_category":dict(self._released_counts),"settled_by_category":dict(self._settled_counts),"table_retained":int(self._table_retained),"category_totals":dict(CATEGORY_TOTALS),"size_totals":dict(SIZE_TOTALS),"error":self.error}
        temp=self.status_path.with_suffix(self.status_path.suffix+".tmp"); temp.write_text(json.dumps(payload),encoding="utf-8"); temp.replace(self.status_path); self._status_at=now

    def close(self):
        if self._closed: return
        self._closed=True
        if self._prepare_only: return
        try:
            if self.status_path.exists() and json.loads(self.status_path.read_text()).get("session")==self.session: self.status_path.unlink()
        except (OSError,ValueError,json.JSONDecodeError): pass
        # PhysX reports undefined behavior if collision groups are removed
        # while the scene is still stopping.  The stage owns these transient
        # prims and will discard them on shutdown; leaving them parked avoids
        # that teardown hazard.


def install_triggered_debris(base):
    del base
    return TriggeredFallingDebris3D
