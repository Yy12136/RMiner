"""Isaac-Sim-native lighting and dust effects for dataset replay.

The module is enabled only through environment variables.  It changes USD
lights and RTX volumetric settings before camera render products are sampled;
it never edits an already encoded video.
"""

from __future__ import annotations

import os
import json
import math
import sys
import importlib.util
from pathlib import Path

import omni.usd
from pxr import Gf, Sdf, UsdGeom, UsdLux


VALID_LIGHTING_PROFILES = {
    "none",
    "low_illumination",
    "warm_worklamp",
    "cool_led",
    "headlamp_spot",
    "headlamp_only",
    "headlamp_aux0",
    "headlamp_aux1",
    "headlamp_aux2",
    "headlamp_aux123",
}


def _parse_vector(name: str, default: tuple[float, float, float]):
    raw = os.environ.get(name)
    if not raw:
        return default
    if raw.strip().lower() == "auto":
        return None
    values = tuple(float(item.strip()) for item in raw.split(","))
    if len(values) != 3:
        raise ValueError(f"{name} must contain exactly three comma-separated values")
    return values


class ReplayAugmentationEffects:
    """Apply one deterministic lighting profile and optional spatial dust."""

    ROOT_PATH = "/World/ReplayAugmentation"

    def __init__(self) -> None:
        self.stage = omni.usd.get_context().get_stage()
        self.profile = os.environ.get(
            "ISAAC_REPLAY_LIGHTING_PROFILE", "none"
        ).strip().lower()
        if self.profile not in VALID_LIGHTING_PROFILES:
            raise ValueError(
                "ISAAC_REPLAY_LIGHTING_PROFILE must be one of "
                + ", ".join(sorted(VALID_LIGHTING_PROFILES))
            )
        self.center = _parse_vector(
            "ISAAC_REPLAY_EFFECT_CENTER", (-3.0606229131, 3.5318245338, 0.0)
        )
        if self.center is None:
            self.center = self._camera_center()
        self._original_light_values: list[tuple[object, object]] = []
        self._original_nurec_values: list[tuple[object, object]] = []
        self._created_paths: list[str] = []
        self._dust = None
        self._falling_debris_prepare = None
        self._apply_lighting()
        if os.environ.get("ISAAC_REPLAY_DUST", "0") == "1":
            # Reuse the approved 5000-particle neutral-gray dust renderer.
            module_root = str(Path(__file__).resolve().parent)
            if module_root not in sys.path:
                sys.path.insert(0, module_root)
            from mine_dust_3d import MineDust3D

            self._dust = MineDust3D(self.center)
        if os.environ.get("G1_FALLING_DEBRIS", "0").strip().lower() in {
            "1", "true", "yes", "on"
        }:
            # Dynamic debris must be authored before env.sim.reset so GPU
            # PhysX includes these bodies in its rigid-body tensor view.  The
            # task runtime controller attaches to these prims after reset and
            # owns their release/update lifecycle.
            source = Path(__file__).resolve().with_name("falling_debris_3d.py")
            module_name = "task_replay_falling_debris_3d"
            module = sys.modules.get(module_name)
            if module is None:
                spec = importlib.util.spec_from_file_location(module_name, source)
                if spec is None or spec.loader is None:
                    raise ImportError(f"Unable to load falling-debris module: {source}")
                module = importlib.util.module_from_spec(spec)
                sys.modules[module_name] = module
                spec.loader.exec_module(module)
            self._falling_debris_prepare = module.FallingDebris3D(
                self.stage, seed=int(os.environ.get("G1_DEBRIS_SEED", "260822")),
                prepare_only=True,
            )
        print(
            "[replay-augmentation] profile="
            f"{self.profile}, dust={self._dust is not None}, center={self.center}"
        )

    def _camera_center(self) -> tuple[float, float, float]:
        positions = []
        for prim in self.stage.Traverse():
            if not prim.IsA(UsdGeom.Camera):
                continue
            name = prim.GetName().lower()
            if not any(token in name for token in ("front", "head", "wrist")):
                continue
            matrix = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(0.0)
            translation = matrix.ExtractTranslation()
            positions.append(tuple(float(value) for value in translation))
        if not positions:
            print(
                "[replay-augmentation] No task camera was found for automatic "
                "centering; using world origin."
            )
            return (0.0, 0.0, 0.0)
        center = tuple(
            sum(point[axis] for point in positions) / len(positions)
            for axis in range(3)
        )
        return (center[0], center[1], 0.0)

    def _scale_existing_lights(self, scale: float) -> None:
        augmentation_root = Sdf.Path(self.ROOT_PATH)
        for prim in self.stage.Traverse():
            if prim.GetPath().HasPrefix(augmentation_root):
                continue
            intensity = prim.GetAttribute("inputs:intensity")
            if not intensity or not intensity.HasAuthoredValueOpinion():
                continue
            value = intensity.Get()
            if value is None:
                continue
            self._original_light_values.append((intensity, value))
            intensity.Set(float(value) * float(scale))

    def _darken_nurec_emission(self, scale: float) -> int:
        """Reduce baked NuRec emission so shaped lights remain visible."""

        changed = 0
        for prim in self.stage.Traverse():
            role = prim.GetAttribute("fieldRole")
            if not role or role.Get() != "emissiveColor":
                continue
            for name, value in (
                ("omni:nurec:ccmR", Gf.Vec4f(scale, 0.0, 0.0, 0.0)),
                ("omni:nurec:ccmG", Gf.Vec4f(0.0, scale, 0.0, 0.0)),
                ("omni:nurec:ccmB", Gf.Vec4f(0.0, 0.0, scale, 0.0)),
            ):
                attribute = prim.GetAttribute(name)
                if attribute and attribute.HasAuthoredValueOpinion():
                    self._original_nurec_values.append((attribute, attribute.Get()))
                else:
                    attribute = prim.CreateAttribute(name, Sdf.ValueTypeNames.Float4)
                attribute.Set(value)
            changed += 1
        print(
            f"[replay-augmentation] NuRec emissive scale={scale:.3f}, "
            f"fields={changed}"
        )
        return changed

    @staticmethod
    def _disable_auto_exposure(camera) -> None:
        prim = camera if hasattr(camera, "GetAppliedSchemas") else camera.GetPrim()
        for schema in (
            "OmniRtxCameraAutoExposureAPI_1",
            "OmniRtxCameraExposureAPI_1",
        ):
            if schema not in prim.GetAppliedSchemas():
                prim.AddAppliedSchema(schema)
        exposure = float(os.environ.get("G1_HEADLAMP_CAMERA_EXPOSURE", "0.0"))
        for name, value in {
            "exposure": exposure,
            "exposure:fStop": 1.0,
            "exposure:iso": 0.0,
            "exposure:responsivity": 1.0,
            "exposure:time": 1.0,
        }.items():
            prim.CreateAttribute(name, Sdf.ValueTypeNames.Float).Set(value)
        prim.CreateAttribute(
            "omni:rtx:autoExposure:enabled", Sdf.ValueTypeNames.Bool
        ).Set(False)

    @staticmethod
    def _translate(prim, position: tuple[float, float, float]) -> None:
        xform = UsdGeom.Xformable(prim)
        operation = next(
            (
                op
                for op in xform.GetOrderedXformOps()
                if op.GetOpName() == "xformOp:translate"
            ),
            None,
        )
        if operation is None:
            operation = xform.AddTranslateOp()
        operation.Set(Gf.Vec3d(*position))

    def _create_sphere_light(
        self,
        path: str,
        position: tuple[float, float, float],
        intensity: float,
        color: tuple[float, float, float],
        radius: float,
    ):
        light = UsdLux.SphereLight.Define(self.stage, path)
        light.CreateIntensityAttr(float(intensity))
        light.CreateColorAttr(Gf.Vec3f(*color))
        light.CreateRadiusAttr(float(radius))
        self._translate(light.GetPrim(), position)
        return light

    def _find_head_camera(self):
        preferred = []
        for prim in self.stage.Traverse():
            if not prim.IsA(UsdGeom.Camera):
                continue
            name = prim.GetName().lower()
            score = 0
            if "front_cam" in name:
                score += 10
            if "front" in name or "head" in name:
                score += 5
            if "wrist" in name:
                score -= 10
            preferred.append((score, str(prim.GetPath()), prim))
        if not preferred:
            raise RuntimeError("No head/front camera found for replay lighting")
        preferred.sort(key=lambda item: (-item[0], item[1]))
        return preferred[0][2]

    @staticmethod
    def _camera_world_point(matrix, point: tuple[float, float, float]) -> Gf.Vec3d:
        return matrix.Transform(Gf.Vec3d(*point))

    @staticmethod
    def _aim_quaternion(source: Gf.Vec3d, target: Gf.Vec3d) -> Gf.Quatd:
        direction = target - source
        if direction.GetLength() <= 1.0e-8:
            raise ValueError("Replay spotlight source and target must differ")
        direction.Normalize()
        return Gf.Rotation(Gf.Vec3d(0.0, 0.0, -1.0), direction).GetQuat()

    def _create_replay_spot(self, path: str, position: Gf.Vec3d, target: Gf.Vec3d, intensity: float, cone_angle: float, softness: float, focus: float, *, local_space: bool = False, color_temperature: float = 4100.0) -> None:
        light = UsdLux.SphereLight.Define(self.stage, path)
        light.CreateIntensityAttr(float(intensity)).Set(float(intensity))
        light.CreateExposureAttr(0.0).Set(0.0)
        light.CreateRadiusAttr(0.025).Set(0.025)
        light.CreateNormalizeAttr(False).Set(False)
        light.CreateColorAttr(Gf.Vec3f(1.0, 1.0, 1.0)).Set(Gf.Vec3f(1.0, 1.0, 1.0))
        light.CreateEnableColorTemperatureAttr(True).Set(True)
        light.CreateColorTemperatureAttr(float(color_temperature)).Set(
            float(color_temperature)
        )
        shaping = UsdLux.ShapingAPI.Apply(light.GetPrim())
        shaping.CreateShapingConeAngleAttr(float(cone_angle)).Set(float(cone_angle))
        shaping.CreateShapingConeSoftnessAttr(float(softness)).Set(float(softness))
        shaping.CreateShapingFocusAttr(float(focus)).Set(float(focus))
        shaping.CreateShapingFocusTintAttr(Gf.Vec3f(0.0, 0.0, 0.0)).Set(Gf.Vec3f(0.0, 0.0, 0.0))
        UsdLux.ShadowAPI.Apply(light.GetPrim()).CreateShadowEnableAttr(True).Set(True)
        xform = UsdGeom.Xformable(light.GetPrim())
        xform.ClearXformOpOrder()
        xform.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(position)
        if not local_space or position != target:
            xform.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(
                self._aim_quaternion(position, target)
            )
        self._created_paths.append(path)

    @staticmethod
    def _parse_aux_indices(profile: str) -> tuple[int, ...]:
        if os.environ.get("ISAAC_REPLAY_AUX_LIGHTS", "on").strip().lower() == "off":
            return ()
        explicit = os.environ.get("ISAAC_REPLAY_AUX_LIGHT_INDICES", "").strip()
        if explicit:
            values = tuple(sorted({int(item) for item in explicit.split(",") if item.strip()}))
            if any(index not in (0, 1, 2) for index in values):
                raise ValueError("G1_AUX_LIGHT_INDICES must contain only 0,1,2")
            return values
        return {
            "headlamp_only": (),
            "headlamp_aux0": (0,),
            "headlamp_aux1": (1,),
            "headlamp_aux2": (2,),
            "headlamp_aux123": (0, 1, 2),
        }[profile]

    def _metadata_object_target(self, camera_matrix):
        metadata_file = os.environ.get("G1_REPLAY_INITIAL_STATE_FILE", "").strip()
        if not metadata_file:
            return None
        payload = json.loads(Path(metadata_file).read_text(encoding="utf-8"))
        sim_state = payload.get("sim_state") or {}
        runtime = sim_state.get("runtime_state") or {}
        replay_initial = runtime.get("replay_initial_state") or {}
        pose = runtime.get("object_root_pose_wxyz")
        object_name = runtime.get("active_object")
        if pose is None:
            singular = replay_initial.get("object") or {}
            pose = singular.get("root_pose_wxyz")
            object_name = singular.get("name", object_name)
        if pose is None:
            objects = replay_initial.get("objects") or {}
            preferred = (
                "rod",
                "coupling_sleeve",
                "hand_cart",
                "handwheel",
                "loaded_detonator_assembly",
            )
            selected = next((name for name in preferred if name in objects), None)
            if selected is None and objects:
                selected = sorted(objects)[0]
            if selected is not None:
                pose = (objects[selected] or {}).get("root_pose_wxyz")
                object_name = selected
        if pose is None:
            rigid_objects = (sim_state.get("init_state") or {}).get("rigid_object") or {}
            if rigid_objects:
                object_name = sorted(rigid_objects)[0]
                pose = (rigid_objects[object_name] or {}).get("root_pose", [[None]])[0]
        if not pose or len(pose) < 3:
            return None
        world = Gf.Vec3d(float(pose[0]), float(pose[1]), float(pose[2]))
        local = camera_matrix.GetInverse().Transform(world)
        if float(local[2]) >= -0.10:
            return None
        task_id = str(sim_state.get("task_name") or runtime.get("task_id") or "")
        spread = {
            "Isaac-TabletopGrasp-G129-Dex1-Joint": 0.12,
            "Isaac-RoofBoltSleeve-G129-Dex1-Joint": 0.10,
            "Isaac-BinPickPlace-G129-Dex1-Joint": 0.12,
            "Isaac-HandCartPush-G129-Dex1-Joint": 0.24,
            "Isaac-ValveHandwheel-G129-Dex1-Joint": 0.12,
            "Isaac-WallHoleDetonator-G129-Dex1-Joint": 0.07,
        }.get(task_id, 0.10)
        return (local, spread, str(object_name or "unknown"), task_id)

    def _apply_batch_lighting(self) -> None:
        ambient_scale = float(os.environ.get("G1_REPLAY_AMBIENT_SCALE", "0.08"))
        self._scale_existing_lights(ambient_scale)
        self._darken_nurec_emission(
            float(os.environ.get("G1_HEADLAMP_NUREC_EMISSIVE_SCALE", "0.14"))
        )
        camera = self._find_head_camera()
        self._disable_auto_exposure(camera)
        camera_path = str(camera.GetPath())
        camera_matrix = UsdGeom.Xformable(camera).ComputeLocalToWorldTransform(0.0)
        headlamp_enabled = os.environ.get("ISAAC_REPLAY_HEADLAMP", "on").strip().lower() == "on"
        indices = self._parse_aux_indices(self.profile)
        if headlamp_enabled:
            # This prim is a child of the camera.  Its transform must therefore
            # be local; authoring a world transform here double-transformed the
            # old batch headlamp and made the five cases visually collapse.
            self._create_replay_spot(
                f"{camera_path}/ReplayBatchHeadlamp",
                Gf.Vec3d(0.0, 0.035, -0.015),
                Gf.Vec3d(0.0, 0.035, -4.0),
                float(os.environ.get("G1_HEADLAMP_INTENSITY", "14000000")),
                float(os.environ.get("G1_HEADLAMP_CONE_ANGLE_DEG", "30")),
                float(os.environ.get("G1_HEADLAMP_CONE_SOFTNESS", "0.40")),
                float(os.environ.get("G1_HEADLAMP_FOCUS", "2.5")),
                local_space=True,
            )
        if not indices:
            self._write_assignments([], camera_path, ambient_scale)
            return
        import random
        seed = int(os.environ.get("G1_AUX_LIGHT_SEED", "260904"))
        episode = int(os.environ.get("G1_AUX_LIGHT_EPISODE", "0"))
        rng = random.Random(seed + episode)
        # Positions are expressed in the initial head-camera frame, then
        # converted to fixed world positions.  Targets span left/centre/right
        # visible rays at mixed depths instead of all landing below the floor.
        sources = ((-0.28, 0.10, -0.02), (0.28, 0.10, -0.02), (0.0, 0.26, -0.02))
        metadata_target = self._metadata_object_target(camera_matrix)
        if metadata_target is not None:
            object_local, spread, target_object_name, target_task_id = metadata_target
            offsets = ((-spread, -0.02), (0.0, 0.05), (spread, -0.02))
            targets = [
                (
                    float(object_local[0] + dx + rng.uniform(-0.01, 0.01)),
                    float(object_local[1] + dy + rng.uniform(-0.01, 0.01)),
                    float(object_local[2]),
                )
                for dx, dy in offsets
            ]
            target_source = "replay_metadata_task_object"
        else:
            lanes = (-0.55, 0.0, 0.55)
            targets = []
            for lane in lanes:
                depth = rng.uniform(1.0, 1.5)
                targets.append(
                    (
                        (lane + rng.uniform(-0.07, 0.07)) * depth,
                        rng.uniform(-0.08, 0.22),
                        -depth,
                    )
                )
            target_source = "camera_view_fallback"
            target_object_name = None
            target_task_id = None
        color_temperatures = tuple(
            float(item.strip())
            for item in os.environ.get(
                "G1_AUX_LIGHT_COLOR_TEMPERATURES", "3000,4100,6000"
            ).split(",")
        )
        if len(color_temperatures) != 3:
            raise ValueError(
                "G1_AUX_LIGHT_COLOR_TEMPERATURES must contain three values"
            )
        assignments = []
        for index in indices:
            source_local = sources[index]
            target_local = targets[index]
            source = self._camera_world_point(camera_matrix, source_local)
            target = self._camera_world_point(camera_matrix, target_local)
            path = f"{camera_path}/ReplayBatchAuxLight{index}"
            self._create_replay_spot(
                path,
                Gf.Vec3d(*source_local),
                Gf.Vec3d(*target_local),
                float(os.environ.get("G1_AUX_LIGHT_INTENSITY", "30000000")),
                float(os.environ.get("G1_AUX_LIGHT_CONE_ANGLE_DEG", "20")),
                float(os.environ.get("G1_AUX_LIGHT_CONE_SOFTNESS", "0.28")),
                float(os.environ.get("G1_AUX_LIGHT_FOCUS", "4.0")),
                local_space=True,
                color_temperature=color_temperatures[index],
            )
            distance = math.dist(tuple(source), tuple(target))
            assignments.append({
                "index": index,
                "source_local_m": list(source_local),
                "target_local_m": list(target_local),
                "source_world_m": [float(v) for v in source],
                "target_world_m": [float(v) for v in target],
                "normalized_image_target": [
                    float(target_local[0] / max(1.0e-6, -target_local[2])),
                    float(target_local[1] / max(1.0e-6, -target_local[2])),
                ],
                "source_target_distance_m": distance,
                "color_temperature_k": color_temperatures[index],
                "target_source": target_source,
                "target_object_name": target_object_name,
                "target_task_id": target_task_id,
            })
        self._write_assignments(assignments, camera_path, ambient_scale)

    def _write_assignments(self, assignments: list[dict], camera_path: str, ambient_scale: float) -> None:
        assignment_file = os.environ.get("G1_AUX_LIGHT_ASSIGNMENT_FILE", "").strip()
        if assignment_file:
            Path(assignment_file).parent.mkdir(parents=True, exist_ok=True)
            Path(assignment_file).write_text(json.dumps({
                "schema": "lighting_replay_r2",
                "seed": int(os.environ.get("G1_AUX_LIGHT_SEED", "260904")),
                "episode": int(os.environ.get("G1_AUX_LIGHT_EPISODE", "0")),
                "profile": self.profile,
                "camera_path": camera_path,
                "headlamp_enabled": os.environ.get("ISAAC_REPLAY_HEADLAMP", "on").strip().lower() == "on",
                "ambient_scale": ambient_scale,
                "nurec_emissive_scale": float(os.environ.get("G1_HEADLAMP_NUREC_EMISSIVE_SCALE", "0.14")),
                "lights": assignments,
            }, indent=2) + "\n", encoding="utf-8")

    def _apply_lighting(self) -> None:
        x, y, z = self.center
        if self.profile == "none":
            return
        if self.profile == "low_illumination":
            self._scale_existing_lights(0.34)
            return
        if self.profile == "warm_worklamp":
            self._scale_existing_lights(0.58)
            self._create_sphere_light(
                f"{self.ROOT_PATH}/WarmWorkLamp",
                (x - 0.45, y - 0.35, z + 1.55),
                4200.0,
                (1.0, 0.64, 0.36),
                0.18,
            )
            return
        if self.profile == "cool_led":
            self._scale_existing_lights(0.72)
            light = UsdLux.RectLight.Define(
                self.stage, f"{self.ROOT_PATH}/CoolLedPanel"
            )
            light.CreateIntensityAttr(2600.0)
            light.CreateColorAttr(Gf.Vec3f(0.66, 0.82, 1.0))
            light.CreateWidthAttr(1.6)
            light.CreateHeightAttr(0.9)
            self._translate(light.GetPrim(), (x, y, z + 2.15))
            return
        if self.profile == "headlamp_spot":
            self._scale_existing_lights(0.27)
            parent = os.environ.get(
                "ISAAC_REPLAY_HEADLAMP_PARENT",
                "/World/TrackedBase/G1UpperBody/d435_link/front_cam",
            ).rstrip("/")
            light = UsdLux.SphereLight.Define(
                self.stage, f"{parent}/ReplayProfileHeadlamp"
            )
            light.CreateIntensityAttr(2400.0)
            light.CreateColorAttr(Gf.Vec3f(0.86, 0.92, 1.0))
            light.CreateRadiusAttr(0.022)
            shaping = UsdLux.ShapingAPI.Apply(light.GetPrim())
            shaping.CreateShapingConeAngleAttr(27.0)
            shaping.CreateShapingConeSoftnessAttr(0.32)
            return
        if self.profile in {"headlamp_only", "headlamp_aux0", "headlamp_aux1", "headlamp_aux2", "headlamp_aux123"}:
            self._apply_batch_lighting()
            return

    def update(self, dt: float) -> None:
        if self._dust is not None:
            self._dust.update(dt)

    def close(self) -> None:
        if self._falling_debris_prepare is not None:
            self._falling_debris_prepare.close()
            self._falling_debris_prepare = None
        if self._dust is not None:
            self._dust.close()
            self._dust = None
        if self.stage:
            self.stage.RemovePrim(Sdf.Path(self.ROOT_PATH))
            for path in self._created_paths:
                try:
                    self.stage.RemovePrim(Sdf.Path(path))
                except Exception:
                    pass
        self._created_paths.clear()
        for attribute, value in self._original_light_values:
            try:
                attribute.Set(value)
            except Exception:
                pass
        self._original_light_values.clear()
        for attribute, value in self._original_nurec_values:
            try:
                attribute.Set(value)
            except Exception:
                pass
        self._original_nurec_values.clear()


def effects_requested() -> bool:
    return (
        os.environ.get("ISAAC_REPLAY_LIGHTING_PROFILE", "none") != "none"
        or os.environ.get("ISAAC_REPLAY_DUST", "0") == "1"
        or os.environ.get("G1_FALLING_DEBRIS", "0").strip().lower()
        in {"1", "true", "yes", "on"}
    )
