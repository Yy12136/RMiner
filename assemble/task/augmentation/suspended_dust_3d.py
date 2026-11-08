"""Distributed suspended mine dust for the Task01 RTX scene.

The effect combines a low-density, animated RTX atmospheric layer with
world-space dust motes spread through the tunnel.  It deliberately has no
single source or expanding spherical front.  Everything is authored in USD;
camera pixels and source datasets are untouched.
"""

from __future__ import annotations

import math
import os
import random

import carb.settings
import omni.usd
from pxr import Gf, Sdf, UsdGeom, Vt


def _float(name: str, default: float) -> float:
    value = float(os.environ.get(name, str(default)))
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def _int(name: str, default: int) -> int:
    return int(os.environ.get(name, str(default)))


class SuspendedDust3D:
    """Sparse animated particles plus long-range heterogeneous attenuation."""

    ROOT_PATH = "/World/G1SuspendedDust"

    def __init__(
        self,
        table_position: tuple[float, float, float],
        table_yaw: float = 0.0,
    ) -> None:
        del table_position, table_yaw
        self.stage = omni.usd.get_context().get_stage()
        self.settings = carb.settings.get_settings()
        self.seed = _int("G1_SUSPENDED_DUST_SEED", 260917)
        self.rng = random.Random(self.seed)
        self.variant = os.environ.get("G1_SUSPENDED_DUST_VARIANT", "balanced").strip().lower()
        if self.variant not in {"subtle", "balanced", "dense"}:
            raise ValueError("G1_SUSPENDED_DUST_VARIANT must be subtle, balanced, or dense")
        defaults = {
            # Dust is mostly a participating medium.  Individual grains are
            # deliberately sparse: a dense field of large points reads as
            # snow or fireflies at this camera resolution.
            "subtle": (70, 0.085, 0.012),
            "balanced": (150, 0.125, 0.018),
            "dense": (280, 0.175, 0.027),
        }
        default_count, default_haze, default_fog = defaults[self.variant]
        self.count = _int("G1_SUSPENDED_DUST_COUNT", default_count)
        self.haze_density = _float("G1_SUSPENDED_DUST_HAZE", default_haze)
        self.fog_density = _float("G1_SUSPENDED_DUST_FOG", default_fog)
        self.width_min = _float("G1_SUSPENDED_DUST_WIDTH_MIN", 0.0025)
        self.width_max = _float("G1_SUSPENDED_DUST_WIDTH_MAX", 0.0090)
        self.alpha_min = _float("G1_SUSPENDED_DUST_ALPHA_MIN", 0.015)
        self.alpha_max = _float("G1_SUSPENDED_DUST_ALPHA_MAX", 0.060)
        color_text = os.environ.get("G1_SUSPENDED_DUST_COLOR", "0.40,0.33,0.24")
        self.color = tuple(float(component) for component in color_text.split(","))
        if not 0 <= self.count <= 3000:
            raise ValueError("G1_SUSPENDED_DUST_COUNT must be in [0, 3000]")
        if not 0 <= self.haze_density <= 0.5 or not 0 <= self.fog_density <= 0.1:
            raise ValueError("Suspended dust haze/fog density is outside the safe demo range")
        if not 0.0005 <= self.width_min <= self.width_max <= 0.05:
            raise ValueError("Suspended dust widths must satisfy 0.0005 <= min <= max <= 0.05")
        if not 0.0 <= self.alpha_min <= self.alpha_max <= 0.25:
            raise ValueError("Suspended dust alpha must satisfy 0 <= min <= max <= 0.25")
        if len(self.color) != 3 or any(not 0.0 <= value <= 2.0 for value in self.color):
            raise ValueError("G1_SUSPENDED_DUST_COLOR must contain three values in [0, 2]")

        self._closed = False
        self._elapsed = 0.0
        self._last_update = -1.0
        self._previous_settings: dict[str, object] = {}
        self._configure_haze()

        root = UsdGeom.Xform.Define(self.stage, self.ROOT_PATH).GetPrim()
        root.CreateAttribute("g1:dust:implementation", Sdf.ValueTypeNames.String).Set(
            "distributed USD motes plus animated RTX inscattering"
        )
        root.CreateAttribute("g1:dust:variant", Sdf.ValueTypeNames.Token).Set(self.variant)
        self._points_schema = UsdGeom.Points.Define(self.stage, f"{self.ROOT_PATH}/Motes")

        # The fixed demo camera is at roughly (-2.32, -2.94, 3.25) looking +Y.
        # Motes fill the complete visible tunnel volume with no privileged centre.
        self._particles = []
        colors = []
        opacities = []
        widths = []
        for index in range(self.count):
            # Bias some samples toward the camera to make parallax visible, but
            # retain a deep population for atmospheric perspective.
            depth_u = self.rng.random()
            y = -2.2 + 12.8 * (depth_u ** 1.35)
            x_span = 3.0 - 0.35 * max(0.0, (y - 5.0) / 5.0)
            x = -2.2 + self.rng.uniform(-x_span, x_span)
            z = self.rng.uniform(0.15, 4.25)
            phase = self.rng.uniform(0.0, math.tau)
            speed = self.rng.uniform(0.012, 0.042)
            size = self.rng.uniform(self.width_min, self.width_max)
            if self.rng.random() < 0.025:
                size *= self.rng.uniform(1.25, 1.6)
            alpha = self.rng.uniform(self.alpha_min, self.alpha_max)
            if self.variant == "dense":
                alpha *= 1.22
            warm = self.rng.uniform(0.82, 1.08)
            colors.append(Gf.Vec3f(*(component * warm for component in self.color)))
            opacities.append(min(alpha, 0.09))
            widths.append(size)
            self._particles.append(
                {
                    "base": Gf.Vec3f(x, y, z),
                    "phase": phase,
                    "speed": speed,
                    "lift": self.rng.uniform(-0.004, 0.009),
                }
            )

        self._points_schema.CreatePointsAttr(Vt.Vec3fArray([item["base"] for item in self._particles]))
        self._points_schema.CreateWidthsAttr(Vt.FloatArray(widths))
        self._points_schema.SetWidthsInterpolation(UsdGeom.Tokens.vertex)
        primvars = UsdGeom.PrimvarsAPI(self._points_schema)
        primvars.CreatePrimvar(
            "displayColor", Sdf.ValueTypeNames.Color3fArray, UsdGeom.Tokens.vertex
        ).Set(Vt.Vec3fArray(colors))
        primvars.CreatePrimvar(
            "displayOpacity", Sdf.ValueTypeNames.FloatArray, UsdGeom.Tokens.vertex
        ).Set(Vt.FloatArray(opacities))
        self._points_schema.GetPrim().CreateAttribute(
            "primvars:doNotCastShadows", Sdf.ValueTypeNames.Bool
        ).Set(True)
        print(
            f"[suspended-dust-3d] enabled variant={self.variant}, motes={self.count}, "
            f"haze={self.haze_density}, fog={self.fog_density}, seed={self.seed}",
            flush=True,
        )

    def _set_setting(self, path: str, value) -> None:
        if path not in self._previous_settings:
            self._previous_settings[path] = self.settings.get(path)
        if isinstance(value, bool):
            self.settings.set_bool(path, value)
        elif isinstance(value, int):
            self.settings.set_int(path, value)
        elif isinstance(value, float):
            self.settings.set_float(path, value)
        else:
            self.settings.set(path, value)

    def _configure_haze(self) -> None:
        # A long measurement distance produces believable attenuation without
        # turning the near field into a uniform gray veil.  Detail noise moves
        # independently in 3D so there is no radial expansion pattern.
        settings = {
            "/rtx/raytracing/globalVolumetricEffects/enabled": True,
            "/rtx/raytracing/inscattering/atmosphereHeight": 8.0,
            "/rtx/raytracing/inscattering/maxDistance": 45.0,
            "/rtx/raytracing/inscattering/densityMult": self.haze_density,
            "/rtx/raytracing/inscattering/transmittanceMeasurementDistance": 24.0,
            "/rtx/raytracing/inscattering/transmittanceColor": (0.58, 0.53, 0.46),
            "/rtx/raytracing/inscattering/singleScatteringAlbedo": (0.46, 0.41, 0.34),
            "/rtx/raytracing/inscattering/anisotropyFactor": 0.42,
            "/rtx/raytracing/inscattering/useDetailNoise": True,
            "/rtx/raytracing/inscattering/detailNoiseScale": 0.065,
            "/rtx/raytracing/inscattering/noiseAnimationSpeedX": 0.010,
            "/rtx/raytracing/inscattering/noiseAnimationSpeedY": 0.026,
            "/rtx/raytracing/inscattering/noiseAnimationSpeedZ": 0.004,
            "/rtx/raytracing/inscattering/noiseScaleRangeMin": 0.08,
            "/rtx/raytracing/inscattering/noiseScaleRangeMax": 0.62,
            "/rtx/raytracing/inscattering/noiseNumOctaves": 4,
            "/rtx/raytracing/inscattering/maxAccumulationFrames": 4,
            "/rtx/raytracing/inscattering/depthSlices": 96,
            "/rtx/raytracing/inscattering/pixelRatio": 2,
            "/rtx/raytracing/inscattering/enableFlowSampling": False,
            "/rtx/fog/enabled": True,
            "/rtx/fog/fogColor": (0.44, 0.40, 0.34),
            "/rtx/fog/fogColorIntensity": 0.54,
            "/rtx/fog/fogZup/enabled": True,
            "/rtx/fog/fogStartHeight": 4.5,
            "/rtx/fog/fogHeightDensity": self.fog_density * 0.34,
            "/rtx/fog/fogHeightFalloff": 0.06,
            "/rtx/fog/fogStartDist": 0.35,
            "/rtx/fog/fogEndDist": 42.0,
            "/rtx/fog/fogDistanceDensity": self.fog_density,
        }
        for path, value in settings.items():
            self._set_setting(path, value)

    def update(self, dt: float) -> None:
        self._elapsed += max(0.0, min(float(dt), 0.1))
        if self._elapsed - self._last_update < 0.20 or not self._particles:
            return
        self._last_update = self._elapsed
        positions = []
        for index, item in enumerate(self._particles):
            base = item["base"]
            phase = item["phase"] + self._elapsed * item["speed"]
            y = float(base[1]) + self._elapsed * (0.010 + 0.002 * (index % 5))
            if y > 10.6:
                y = -2.2 + (y - 10.6)
            positions.append(
                Gf.Vec3f(
                    float(base[0]) + 0.08 * math.sin(phase * 1.7),
                    y,
                    float(base[2]) + item["lift"] * self._elapsed + 0.055 * math.sin(phase),
                )
            )
        self._points_schema.GetPointsAttr().Set(Vt.Vec3fArray(positions))

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self.stage.GetPrimAtPath(self.ROOT_PATH):
            self.stage.RemovePrim(Sdf.Path(self.ROOT_PATH))
        for path, value in self._previous_settings.items():
            if value is not None:
                self.settings.set(path, value)


__all__ = ["SuspendedDust3D"]
