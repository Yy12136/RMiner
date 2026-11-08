"""Optional RTX volumetric mine dust for Isaac Sim replay augmentation.

This module is intentionally task-local and disabled by default.  It combines
the renderer's camera-frustum volumetric scattering with real USD 3D points,
so head/wrist cameras receive depth, occlusion, parallax, and lighting instead
of a screen-space overlay.
"""

from __future__ import annotations

import math
import os

import carb.settings
import numpy as np
import omni.usd
from pxr import Gf, Sdf, UsdGeom, UsdLux, Vt


def _env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, str(default)))


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, str(default)))


class MineDust3D:
    """Renderer-owned volume plus animated spatial dust motes."""

    POINTS_PATH = "/World/G1MineDustPreview/SpatialDust"
    HEADLAMP_PATH = (
        "/World/TrackedBase/G1UpperBody/d435_link/front_cam/DustHeadlamp"
    )

    def __init__(self, table_position: tuple[float, float, float]) -> None:
        self.stage = omni.usd.get_context().get_stage()
        self.settings = carb.settings.get_settings()
        self.headlamp_path = os.environ.get(
            "ISAAC_REPLAY_DUST_HEADLAMP_PATH", self.HEADLAMP_PATH
        )
        self._previous_settings: dict[str, object] = {}
        self._frame = 0
        self._rng = np.random.default_rng(_env_int("G1_DUST_SEED", 260822))
        self._configure_volume()
        self._create_headlamp()
        self._create_particles(table_position)

    def _set(self, path: str, value) -> None:
        self._previous_settings[path] = self.settings.get(path)
        if isinstance(value, bool):
            self.settings.set_bool(path, value)
        elif isinstance(value, int):
            self.settings.set_int(path, value)
        elif isinstance(value, float):
            self.settings.set_float(path, value)
        elif isinstance(value, (tuple, list)):
            self.settings.set_float_array(path, list(value))
        else:
            self.settings.set(path, value)

    def _configure_volume(self) -> None:
        density = _env_float("G1_DUST_VOLUME_DENSITY", 0.55)
        measurement_distance = _env_float(
            "G1_DUST_TRANSMITTANCE_DISTANCE", 180.0
        )
        settings = {
            "/rtx/raytracing/globalVolumetricEffects/enabled": True,
            "/rtx/raytracing/inscattering/atmosphereHeight": 300.0,
            "/rtx/raytracing/inscattering/maxDistance": 650.0,
            "/rtx/raytracing/inscattering/densityMult": density,
            "/rtx/raytracing/inscattering/transmittanceMeasurementDistance": measurement_distance,
            "/rtx/raytracing/inscattering/transmittanceColor": (0.50, 0.50, 0.50),
            "/rtx/raytracing/inscattering/singleScatteringAlbedo": (0.68, 0.68, 0.68),
            "/rtx/raytracing/inscattering/anisotropyFactor": 0.32,
            "/rtx/raytracing/inscattering/useDetailNoise": True,
            "/rtx/raytracing/inscattering/detailNoiseScale": 0.018,
            "/rtx/raytracing/inscattering/noiseAnimationSpeedX": 0.018,
            "/rtx/raytracing/inscattering/noiseAnimationSpeedY": 0.010,
            "/rtx/raytracing/inscattering/noiseAnimationSpeedZ": 0.004,
            "/rtx/raytracing/inscattering/noiseScaleRangeMin": 0.10,
            "/rtx/raytracing/inscattering/noiseScaleRangeMax": 1.0,
            "/rtx/raytracing/inscattering/noiseNumOctaves": 4,
            "/rtx/raytracing/inscattering/maxAccumulationFrames": 8,
            "/rtx/raytracing/inscattering/depthSlices": 64,
            "/rtx/raytracing/inscattering/pixelRatio": 4,
            # Depth-aware absorption remains useful in camera render products
            # whose RTX mode does not expose the full noisy voxel grid.
            "/rtx/fog/enabled": True,
            "/rtx/fog/fogColor": (0.44, 0.44, 0.44),
            "/rtx/fog/fogColorIntensity": 1.0,
            "/rtx/fog/fogZup/enabled": True,
            "/rtx/fog/fogStartHeight": 3.0,
            "/rtx/fog/fogHeightDensity": 0.025,
            "/rtx/fog/fogHeightFalloff": 0.0,
            "/rtx/fog/fogStartDist": 0.12,
            "/rtx/fog/fogEndDist": 3.2,
            "/rtx/fog/fogDistanceDensity": 0.10,
        }
        for path, value in settings.items():
            self._set(path, value)

    def _create_headlamp(self) -> None:
        # A shaped light makes real suspended particles visible chiefly inside
        # the camera beam, as in a dusty underground tunnel.
        light = UsdLux.SphereLight.Define(self.stage, self.headlamp_path)
        light.CreateIntensityAttr(
            _env_float("G1_DUST_HEADLAMP_INTENSITY", 900.0)
        )
        light.CreateColorAttr(Gf.Vec3f(0.96, 0.96, 0.96))
        light.CreateRadiusAttr(0.025)
        shaping = UsdLux.ShapingAPI.Apply(light.GetPrim())
        shaping.CreateShapingConeAngleAttr(38.0)
        shaping.CreateShapingConeSoftnessAttr(0.48)

    def _create_particles(self, table_position: tuple[float, float, float]) -> None:
        # Dense enough to remain clearly visible from all three cameras while
        # preserving the small, neutral-gray airborne-dust appearance.
        count = _env_int("G1_DUST_PARTICLE_COUNT", 5000)
        if not 1 <= count <= 10000:
            raise ValueError("G1_DUST_PARTICLE_COUNT must be in [1, 10000]")

        center = np.asarray(table_position, dtype=np.float32)
        # This box surrounds the robot head, both wrists, and the tabletop.
        self._minimum = center + np.asarray((-1.25, -1.35, 0.30), np.float32)
        self._maximum = center + np.asarray((1.25, 1.35, 2.35), np.float32)
        self._positions = self._rng.uniform(
            self._minimum, self._maximum, size=(count, 3)
        ).astype(np.float32)
        drift = np.asarray((0.018, -0.010, 0.003), dtype=np.float32)
        self._velocities = (
            drift
            + self._rng.normal(0.0, (0.008, 0.008, 0.004), size=(count, 3))
        ).astype(np.float32)
        self._phase = self._rng.uniform(0.0, 2.0 * math.pi, count).astype(
            np.float32
        )

        depth_class = self._rng.random(count)
        widths = np.where(
            depth_class < 0.80,
            self._rng.uniform(0.0015, 0.0034, count),
            np.where(
                depth_class < 0.97,
                self._rng.uniform(0.0034, 0.0065, count),
                self._rng.uniform(0.0065, 0.0120, count),
            ),
        ).astype(np.float32)
        gray = self._rng.uniform(0.08, 0.24, count).astype(np.float32)
        colors = np.repeat(gray[:, None], 3, axis=1)
        opacity = self._rng.uniform(0.24, 0.60, count).astype(np.float32)

        points = UsdGeom.Points.Define(self.stage, self.POINTS_PATH)
        self._points_attr = points.CreatePointsAttr()
        self._points_attr.Set(Vt.Vec3fArray.FromNumpy(self._positions))
        points.CreateWidthsAttr().Set(Vt.FloatArray.FromNumpy(widths))
        primvars = UsdGeom.PrimvarsAPI(points.GetPrim())
        primvars.CreatePrimvar(
            "displayColor", Sdf.ValueTypeNames.Color3fArray, UsdGeom.Tokens.vertex
        ).Set(Vt.Vec3fArray.FromNumpy(colors))
        primvars.CreatePrimvar(
            "displayOpacity", Sdf.ValueTypeNames.FloatArray, UsdGeom.Tokens.vertex
        ).Set(Vt.FloatArray.FromNumpy(opacity))
        print(
            "[mine-dust-3d] Enabled RTX noisy volumetric scattering and "
            f"{count} animated USD spatial particles."
        )

    def update(self, dt: float) -> None:
        dt = float(np.clip(dt, 0.0, 0.1))
        self._frame += 1
        self._positions += self._velocities * dt
        self._positions[:, 2] += (
            0.0035 * np.sin(self._phase + 0.025 * self._frame) * dt
        )
        extent = self._maximum - self._minimum
        self._positions = (
            (self._positions - self._minimum) % extent + self._minimum
        ).astype(np.float32)
        # Updating every other control tick is enough for 30 Hz cameras and
        # avoids unnecessary USD traffic.
        if self._frame % 2 == 0:
            self._points_attr.Set(Vt.Vec3fArray.FromNumpy(self._positions))

    def close(self) -> None:
        if self.stage:
            self.stage.RemovePrim(Sdf.Path("/World/G1MineDustPreview"))
            self.stage.RemovePrim(Sdf.Path(self.headlamp_path))
        for path, value in self._previous_settings.items():
            if value is not None:
                self.settings.set(path, value)
