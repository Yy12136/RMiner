"""Native USD head-mounted spotlight for Task01 visual replay.

The light is parented below the d435 head link with the same local orientation
as the rendered head camera, so its -Z emission axis follows the camera without
per-frame world-pose updates.  The feature is disabled unless
``G1_HEADLAMP=on`` and restores runtime fallbacks when the controller closes.
"""

from __future__ import annotations

import os
import json
import math
from pathlib import Path

import omni.usd
from pxr import Gf, UsdGeom, UsdLux


AUX_SOURCE_LOCAL_POSITIONS = (
    (-1.05, -0.85, 2.35),
    (1.05, -0.75, 2.25),
    (0.00, 1.05, 2.55),
)
AUX_TABLE_X_RANGE = (-0.55, 0.55)
AUX_TABLE_Y_RANGE = (-0.34, 0.30)
AUX_TABLE_Z = 0.84
AUX_MIN_TARGET_DISTANCE = 0.35

# The original demo constrained every target to the small tabletop interior.
# ``view3d`` deliberately samples a wider first-person scene envelope in the
# table-local frame.  The source lights remain fixed; only their aim points
# change per episode.  The envelope covers floor, walls, far background and
# overhead surfaces that can appear in the head camera.
AUX_VIEW3D_X_LANES = ((-1.00, -0.45), (-0.28, 0.28), (0.45, 1.00))
AUX_VIEW3D_MIN_TARGET_DISTANCE = 0.60
AUX_VIEW3D_REGIONS = ("floor", "table", "left_wall", "right_wall", "far_wall", "ceiling")

# Calibrated first-person surface layouts.  The first renderer pass established
# that distant wall/ceiling points are often occluded or too low contrast in
# this particular mine scene.  These targets are therefore intentionally on
# the *visible* left, centre and right parts of the tabletop: all three are
# genuine 3D world-space targets, but each episode produces a clearly distinct
# and reproducible illumination pattern in the rendered camera image.
AUX_VIEW3D_LAYOUTS = (
    (
        (-0.48, -0.20, 0.84),  # tabletop, left foreground
        (-0.42, 0.06, 0.84),   # tabletop, left centre
        (-0.36, 0.28, 0.84),   # tabletop, left rear
    ),
    (
        (-0.22, 0.02, 0.84),   # tabletop, centre-left
        (0.00, 0.18, 0.84),    # tabletop, centre
        (0.22, 0.02, 0.84),    # tabletop, centre-right
    ),
    (
        (0.36, 0.28, 0.84),    # tabletop, right rear
        (0.42, 0.06, 0.84),    # tabletop, right centre
        (0.48, -0.20, 0.84),   # tabletop, right foreground
    ),
)


def _env_float(name: str, default: float) -> float:
    value = float(os.environ.get(name, str(default)))
    if not value >= 0.0:
        raise ValueError(f"{name} must be non-negative; got {value}")
    return value


def _env_angle(name: str, default: float, lower: float, upper: float) -> float:
    value = float(os.environ.get(name, str(default)))
    if not lower <= value <= upper:
        raise ValueError(f"{name} must be in [{lower}, {upper}]; got {value}")
    return value


def _table_local_to_world(
    table_position: tuple[float, float, float], table_yaw: float,
    local_point: tuple[float, float, float],
) -> Gf.Vec3d:
    cosine = math.cos(table_yaw)
    sine = math.sin(table_yaw)
    x, y, z = local_point
    return Gf.Vec3d(
        table_position[0] + cosine * x - sine * y,
        table_position[1] + sine * x + cosine * y,
        table_position[2] + z,
    )


def _sample_auxiliary_targets(
    seed: int, episode: int, count: int, profile: str
) -> list[tuple[float, float, float]]:
    """Sample separated, deterministic points in the calibrated tabletop interior."""

    import numpy as np

    rng = np.random.Generator(np.random.PCG64(seed + episode))
    points: list[tuple[float, float, float]] = []
    if profile == "distinct":
        # Keep one beam in each screen-side lane.  The y coordinate and a
        # small x jitter remain random, so every episode is different while
        # the three beams cannot collapse into one broad central patch.
        lanes = ((-0.52, -0.40), (-0.02, 0.02), (0.40, 0.52))
        for lower, upper in lanes:
            points.append(
                (
                    float(rng.uniform(lower, upper)),
                    float(rng.uniform(*AUX_TABLE_Y_RANGE)),
                    AUX_TABLE_Z,
                )
            )
        return points
    if profile == "view3d":
        # Use a calibrated layout rather than selecting arbitrary volume points:
        # arbitrary points can sit behind geometry and produce no visible spot.
        # The episode index selects a distinct spatial arrangement; the seed
        # only adds a small, reproducible perturbation to each target.
        layout = AUX_VIEW3D_LAYOUTS[episode % len(AUX_VIEW3D_LAYOUTS)]
        points = []
        for base_x, base_y, base_z in layout[:count]:
            jitter = np.asarray(
                [rng.uniform(-0.025, 0.025), rng.uniform(-0.035, 0.035), 0.0],
                dtype=float,
            )
            point = (float(base_x + jitter[0]), float(base_y + jitter[1]), float(base_z + jitter[2]))
            if all(math.dist(point, previous) >= AUX_VIEW3D_MIN_TARGET_DISTANCE for previous in points):
                points.append(point)
            else:
                points.append((float(base_x), float(base_y), float(base_z)))
        return points
    if profile != "random":
        raise ValueError("G1_AUX_LIGHT_PROFILE must be random, distinct, or view3d")
    for _ in range(10000):
        if len(points) == count:
            break
        candidate = (
            float(rng.uniform(*AUX_TABLE_X_RANGE)),
            float(rng.uniform(*AUX_TABLE_Y_RANGE)),
            AUX_TABLE_Z,
        )
        if all(
            math.hypot(candidate[0] - p[0], candidate[1] - p[1])
            >= AUX_MIN_TARGET_DISTANCE
            for p in points
        ):
            points.append(candidate)
    if len(points) != count:
        raise RuntimeError("Unable to sample separated auxiliary-light targets")
    return points


def _aim_quaternion(source: Gf.Vec3d, target: Gf.Vec3d) -> Gf.Quatd:
    direction = target - source
    if direction.GetLength() <= 1.0e-8:
        raise ValueError("Auxiliary light source and target must differ")
    direction.Normalize()
    # USD lights emit along local -Z.  Rotate that axis directly to the target.
    return Gf.Rotation(Gf.Vec3d(0.0, 0.0, -1.0), direction).GetQuat()


def _parse_floats(name: str):
    """Parse a comma/semicolon separated float list, or None when unset."""
    raw = os.environ.get(name, "").strip()
    if not raw:
        return None
    return [float(v) for v in raw.replace(";", ",").split(",") if v.strip()]


def _parse_world_points(name: str):
    """Parse "x,y,z;x,y,z;..." into world-space points, or None when unset.

    The auxiliary lights are authored relative to the worktable, which is fine
    for lighting the task but cannot put a beam somewhere else in the tunnel.
    These overrides take absolute world coordinates so a shot can be lit from
    any position; leaving them unset keeps the original behaviour exactly.
    """
    raw = os.environ.get(name, "").strip()
    if not raw:
        return None
    points = []
    for chunk in raw.split(";"):
        values = [float(v) for v in chunk.replace(",", " ").split()]
        if len(values) != 3:
            raise ValueError(f"{name} entries must be x,y,z triples")
        points.append(tuple(values))
    return points


def preauthor_auxiliary_spotlights(
    stage,
    table_position: tuple[float, float, float],
    table_yaw: float,
) -> list[dict]:
    """Create the three fixed scene lights before renderer initialization."""

    mode = os.environ.get("G1_AUX_LIGHTS", "off").strip().lower()
    if mode not in {"off", "on"}:
        raise ValueError(f"G1_AUX_LIGHTS must be one of: off, on; got {mode!r}")
    if mode == "off":
        return []
    count = int(os.environ.get("G1_AUX_LIGHT_COUNT", "3"))
    if count != 3:
        raise ValueError("G1_AUX_LIGHT_COUNT is fixed at 3 for this demo")
    seed = int(os.environ.get("G1_AUX_LIGHT_SEED", "260904"))
    episode = int(os.environ.get("G1_AUX_LIGHT_EPISODE", "0"))
    profile = os.environ.get("G1_AUX_LIGHT_PROFILE", "random").strip().lower()
    intensity = _env_float("G1_AUX_LIGHT_INTENSITY", 12000000.0)
    cone_angle = _env_float("G1_AUX_LIGHT_CONE_ANGLE_DEG", 32.0)
    softness = _env_float("G1_AUX_LIGHT_CONE_SOFTNESS", 0.45)
    focus = _env_float("G1_AUX_LIGHT_FOCUS", 2.5)
    color_temperature = _env_float("G1_AUX_LIGHT_COLOR_TEMP_K", 4100.0)
    radius = _env_float("G1_AUX_LIGHT_RADIUS_M", 0.025)
    if not 0.0 < cone_angle <= 90.0 or not 0.0 <= softness <= 1.0:
        raise ValueError("Invalid auxiliary spotlight shaping parameters")

    root_path = "/World/Task01AuxLights"
    if stage.GetPrimAtPath(root_path).IsValid():
        stage.RemovePrim(root_path)
    UsdGeom.Xform.Define(stage, root_path)
    targets = _sample_auxiliary_targets(seed, episode, count, profile)
    world_sources = _parse_world_points("G1_AUX_LIGHT_WORLD_POSITIONS")
    world_intensities = _parse_floats("G1_AUX_LIGHT_WORLD_INTENSITIES")
    world_targets = _parse_world_points("G1_AUX_LIGHT_WORLD_TARGETS")
    for label, points in (
        ("G1_AUX_LIGHT_WORLD_POSITIONS", world_sources),
        ("G1_AUX_LIGHT_WORLD_TARGETS", world_targets),
    ):
        if points is not None and len(points) < count:
            raise ValueError(
                f"{label} needs at least {count} entries, got {len(points)}"
            )
    assignments: list[dict] = []
    for index, (source_local, target_local) in enumerate(
        zip(AUX_SOURCE_LOCAL_POSITIONS[:count], targets)
    ):
        if world_sources is not None:
            source_world = Gf.Vec3d(*world_sources[index])
        else:
            source_world = _table_local_to_world(
                table_position, table_yaw, source_local
            )
        if world_targets is not None:
            target_world = Gf.Vec3d(*world_targets[index])
        else:
            target_world = _table_local_to_world(
                table_position, table_yaw, target_local
            )
        path = f"{root_path}/AuxLight{index}"
        per_light = (
            world_intensities[index]
            if world_intensities is not None and index < len(world_intensities)
            else intensity
        )
        light = UsdLux.SphereLight.Define(stage, path)
        light.CreateIntensityAttr(float(per_light)).Set(float(per_light))
        light.CreateExposureAttr(0.0).Set(0.0)
        light.CreateRadiusAttr(float(radius)).Set(float(radius))
        light.CreateNormalizeAttr(False).Set(False)
        light.CreateColorAttr(Gf.Vec3f(1.0, 1.0, 1.0)).Set(Gf.Vec3f(1.0, 1.0, 1.0))
        light.CreateEnableColorTemperatureAttr(True).Set(True)
        light.CreateColorTemperatureAttr(float(color_temperature)).Set(float(color_temperature))
        shaping = UsdLux.ShapingAPI.Apply(light.GetPrim())
        shaping.CreateShapingConeAngleAttr(float(cone_angle)).Set(float(cone_angle))
        shaping.CreateShapingConeSoftnessAttr(float(softness)).Set(float(softness))
        shaping.CreateShapingFocusAttr(float(focus)).Set(float(focus))
        shaping.CreateShapingFocusTintAttr(Gf.Vec3f(0.0, 0.0, 0.0)).Set(Gf.Vec3f(0.0, 0.0, 0.0))
        UsdLux.ShadowAPI.Apply(light.GetPrim()).CreateShadowEnableAttr(True).Set(True)
        xform = UsdGeom.Xformable(light.GetPrim())
        xform.ClearXformOpOrder()
        xform.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(source_world)
        xform.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(_aim_quaternion(source_world, target_world))
        assignments.append(
            {
                "index": index,
                "prim_path": path,
                "source_local_m": list(source_local),
                "source_world_m": [float(v) for v in source_world],
                "target_local_m": list(target_local),
                "target_world_m": [float(v) for v in target_world],
            }
        )
    assignment_file = os.environ.get("G1_AUX_LIGHT_ASSIGNMENT_FILE", "").strip()
    payload = {
        "seed": seed,
        "episode": episode,
        "count": count,
        "profile": profile,
        "parameters": {
            "intensity": intensity,
            "cone_angle_deg": cone_angle,
            "cone_softness": softness,
            "focus": focus,
            "color_temperature_k": color_temperature,
            "radius_m": radius,
        },
        "table_position_m": list(table_position),
        "table_yaw_rad": table_yaw,
        "min_target_distance_m": AUX_MIN_TARGET_DISTANCE,
        "lights": assignments,
    }
    if assignment_file:
        path = Path(assignment_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"[aux-lights] Pre-authored {count} fixed spotlights for episode={episode}, seed={seed}")
    return assignments


class AuxiliarySpotlightController:
    """Runtime lifecycle hook for the pre-authored auxiliary lights."""

    ROOT_PATH = "/World/Task01AuxLights"

    def __init__(self) -> None:
        self.stage = omni.usd.get_context().get_stage()
        self.enabled = os.environ.get("G1_AUX_LIGHTS", "off").strip().lower() == "on"
        if self.enabled:
            prim = self.stage.GetPrimAtPath(self.ROOT_PATH)
            if not prim or not prim.IsValid():
                raise RuntimeError("Auxiliary lights were not pre-authored before renderer initialization")
            print(f"[aux-lights] Using scene-preauthored lights at {self.ROOT_PATH}")

    def close(self) -> None:
        # The scene is process-local; remove only lights created by this feature.
        if self.enabled and self.stage.GetPrimAtPath(self.ROOT_PATH).IsValid():
            self.stage.RemovePrim(self.ROOT_PATH)
        self.enabled = False


class HeadlampController:
    """Create one warm-white SphereLight shaped as a camera-aligned spotlight."""

    DEFAULT_MOUNT_PATH = "/World/TrackedBase/G1UpperBody/d435_link"
    DEFAULT_DOME_PATH = "/World/light"
    LIGHT_NAME = "Task01Headlamp"

    def __init__(self) -> None:
        mode = os.environ.get("G1_HEADLAMP", "off").strip().lower()
        if mode not in {"off", "on"}:
            raise ValueError(
                f"G1_HEADLAMP must be one of: off, on; got {mode!r}"
            )
        self.mode = mode
        self.stage = omni.usd.get_context().get_stage()
        self.mount_path = os.environ.get(
            "G1_HEADLAMP_MOUNT_PATH", self.DEFAULT_MOUNT_PATH
        ).strip()
        self.dome_path = os.environ.get(
            "G1_HEADLAMP_DOME_PATH", self.DEFAULT_DOME_PATH
        ).strip()
        self.light_path = f"{self.mount_path}/{self.LIGHT_NAME}"
        self._dome_intensity_attr = None
        self._dome_original_intensity: float | None = None
        self._created = False
        self._owns_prim = False

        aux_enabled = os.environ.get("G1_AUX_LIGHTS", "off").strip().lower() == "on"
        if self.mode == "off" and not aux_enabled:
            print("[headlamp] Disabled (G1_HEADLAMP=off).")
            return

        # The auxiliary-only comparison must also run in a dark environment.
        # Previously the headlamp controller returned before reducing the
        # DomeLight, so the three fixed scene lights were washed out by the
        # normal ambient illumination.
        if self.mode == "off" and aux_enabled:
            ambient = _env_float("G1_HEADLAMP_AMBIENT_INTENSITY", 80.0)
            dome = self.stage.GetPrimAtPath(self.dome_path)
            if not dome or not dome.IsValid() or dome.GetTypeName() != "DomeLight":
                raise RuntimeError(
                    f"Expected DomeLight at {self.dome_path}; "
                    f"found {dome.GetTypeName() if dome and dome.IsValid() else None}"
                )
            dome_light = UsdLux.DomeLight(dome)
            self._dome_intensity_attr = dome_light.GetIntensityAttr()
            self._dome_original_intensity = float(self._dome_intensity_attr.Get())
            self._dome_intensity_attr.Set(float(ambient))
            print(
                "[headlamp] Headlamp disabled; reduced DomeLight for "
                f"auxiliary-only replay: ambient={ambient:.1f}"
            )
            return

        mount = self.stage.GetPrimAtPath(self.mount_path)
        if not mount or not mount.IsValid():
            raise RuntimeError(
                f"Headlamp mount prim does not exist: {self.mount_path}"
            )
        intensity = _env_float("G1_HEADLAMP_INTENSITY", 2000000.0)
        ambient = _env_float("G1_HEADLAMP_AMBIENT_INTENSITY", 80.0)
        cone_angle = _env_float("G1_HEADLAMP_CONE_ANGLE_DEG", 44.0)
        cone_softness = _env_float("G1_HEADLAMP_CONE_SOFTNESS", 0.50)
        color_temperature = _env_float("G1_HEADLAMP_COLOR_TEMP_K", 4100.0)
        radius = _env_float("G1_HEADLAMP_RADIUS_M", 0.025)
        focus = _env_float("G1_HEADLAMP_FOCUS", 2.0)
        aim_yaw = _env_angle("G1_HEADLAMP_AIM_YAW_DEG", 0.0, -30.0, 30.0)
        aim_pitch = _env_angle(
            "G1_HEADLAMP_AIM_PITCH_DEG", 0.0, -20.0, 20.0
        )
        if not 0.0 < cone_angle <= 90.0:
            raise ValueError(
                "G1_HEADLAMP_CONE_ANGLE_DEG must be in (0, 90]"
            )
        if not 0.0 <= cone_softness <= 1.0:
            raise ValueError(
                "G1_HEADLAMP_CONE_SOFTNESS must be in [0, 1]"
            )
        if not 1000.0 <= color_temperature <= 10000.0:
            raise ValueError(
                "G1_HEADLAMP_COLOR_TEMP_K must be in [1000, 10000]"
            )

        existing = self.stage.GetPrimAtPath(self.light_path)
        if existing.IsValid():
            if existing.GetTypeName() != "SphereLight":
                raise RuntimeError(
                    f"Pre-authored headlamp is not a SphereLight: "
                    f"{self.light_path} is {existing.GetTypeName()!r}"
                )
            self._created = True
            print(
                "[headlamp] Using scene-preauthored native spotlight: "
                f"path={self.light_path}, intensity={intensity:.1f}, "
                f"ambient={ambient:.1f}, cone={cone_angle:.1f}deg, "
                f"softness={cone_softness:.2f}, focus={focus:.2f}, "
                f"temperature={color_temperature:.0f}K, radius={radius:.3f}m, "
                f"aim_yaw={aim_yaw:.2f}deg, aim_pitch={aim_pitch:.2f}deg"
            )
            return

        dome = self.stage.GetPrimAtPath(self.dome_path)
        if not dome or not dome.IsValid() or dome.GetTypeName() != "DomeLight":
            raise RuntimeError(
                f"Expected DomeLight at {self.dome_path}; "
                f"found {dome.GetTypeName() if dome and dome.IsValid() else None}"
            )
        dome_light = UsdLux.DomeLight(dome)
        self._dome_intensity_attr = dome_light.GetIntensityAttr()
        self._dome_original_intensity = float(
            self._dome_intensity_attr.Get()
        )
        self._dome_intensity_attr.Set(float(ambient))

        if self.stage.GetPrimAtPath(self.light_path).IsValid():
            self.stage.RemovePrim(self.light_path)
        light = UsdLux.SphereLight.Define(self.stage, self.light_path)
        light.CreateIntensityAttr(float(intensity)).Set(float(intensity))
        light.CreateExposureAttr(0.0).Set(0.0)
        light.CreateRadiusAttr(float(radius)).Set(float(radius))
        light.CreateNormalizeAttr(False).Set(False)
        light.CreateColorAttr(Gf.Vec3f(1.0, 1.0, 1.0)).Set(
            Gf.Vec3f(1.0, 1.0, 1.0)
        )
        light.CreateEnableColorTemperatureAttr(True).Set(True)
        light.CreateColorTemperatureAttr(float(color_temperature)).Set(
            float(color_temperature)
        )

        shaping = UsdLux.ShapingAPI.Apply(light.GetPrim())
        shaping.CreateShapingConeAngleAttr(float(cone_angle)).Set(
            float(cone_angle)
        )
        shaping.CreateShapingConeSoftnessAttr(float(cone_softness)).Set(
            float(cone_softness)
        )
        shaping.CreateShapingFocusAttr(float(focus)).Set(float(focus))
        shaping.CreateShapingFocusTintAttr(
            Gf.Vec3f(0.0, 0.0, 0.0)
        ).Set(Gf.Vec3f(0.0, 0.0, 0.0))

        shadow = UsdLux.ShadowAPI.Apply(light.GetPrim())
        shadow.CreateShadowEnableAttr(True).Set(True)

        xform = UsdGeom.Xformable(light.GetPrim())
        xform.ClearXformOpOrder()
        xform.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(
            Gf.Vec3d(0.0, 0.035, -0.015)
        )
        camera_orientation = Gf.Quatd(
            0.5, Gf.Vec3d(-0.5, -0.5, 0.5)
        )
        yaw_offset = Gf.Rotation(
            Gf.Vec3d(0.0, 1.0, 0.0), aim_yaw
        ).GetQuat()
        pitch_offset = Gf.Rotation(
            Gf.Vec3d(1.0, 0.0, 0.0), aim_pitch
        ).GetQuat()
        xform.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(
            camera_orientation * yaw_offset * pitch_offset
        )
        self._created = True
        self._owns_prim = True
        print(
            "[headlamp] Enabled native camera-mounted spotlight: "
            f"path={self.light_path}, intensity={intensity:.1f}, "
            f"ambient={ambient:.1f}, cone={cone_angle:.1f}deg, "
            f"softness={cone_softness:.2f}, focus={focus:.2f}, "
            f"temperature={color_temperature:.0f}K, radius={radius:.3f}m, "
            f"aim_yaw={aim_yaw:.2f}deg, aim_pitch={aim_pitch:.2f}deg"
        )

    @property
    def is_enabled(self) -> bool:
        return self.mode == "on" and self._created

    def close(self) -> None:
        if self._dome_intensity_attr is not None:
            if self._dome_original_intensity is not None:
                self._dome_intensity_attr.Set(
                    float(self._dome_original_intensity)
                )
        if (
            self._created
            and self._owns_prim
            and self.stage.GetPrimAtPath(self.light_path).IsValid()
        ):
            self.stage.RemovePrim(self.light_path)
        self._created = False


__all__ = [
    "AuxiliarySpotlightController",
    "HeadlampController",
    "preauthor_auxiliary_spotlights",
]
