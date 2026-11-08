"""Optional NVIDIA Flow dust for Task01 visual replay.

The effect is disabled unless ``G1_DUST_MODE=flow``.  It combines a very
light RTX ambient haze with a locally simulated NVIDIA Flow volume.  The
installed Fire USD provides the only hierarchy/solver configuration that the
Flow 110 basic ray marcher renders reliably in this headless Isaac Sim build;
its output is restyled as a low-speed neutral brown-gray dust cloud.  No source
dataset samples or camera images are modified.
"""

from __future__ import annotations

import math
import json
import os
from pathlib import Path
import random
import time

import carb.settings
import omni.kit.app
import omni.usd
from pxr import Gf, Sdf, UsdGeom, Vt


_PRESETS = {
    "low": {
        "radius": 0.24,
        "couple_rate": 4.0,
        "opacity": 0.32,
        "ambient_scale": 0.75,
        "fuel": 0.45,
        "temperature": 1.20,
        "coupling_scale": 1.0,
    },
    "medium": {
        "radius": 0.34,
        "couple_rate": 6.0,
        "opacity": 0.40,
        "ambient_scale": 1.00,
        "fuel": 0.65,
        "temperature": 1.60,
        "coupling_scale": 0.30,
    },
    "heavy": {
        "radius": 0.42,
        "couple_rate": 8.0,
        "opacity": 0.65,
        "ambient_scale": 1.30,
        "fuel": 0.80,
        "temperature": 2.00,
        "coupling_scale": 1.0,
    },
}

_CINEMATIC = {
    "radius": 0.45,
    "couple_rate": 7.0,
    # The larger emitter volume already increases coverage substantially.
    # Keep per-voxel opacity lower so it reads as airborne dust rather than
    # an opaque white curtain in the head and wrist cameras.
    "opacity": 0.32,
    "ambient_scale": 1.20,
    "fuel": 0.45,
    "temperature": 1.10,
    "coupling_scale": 0.20,
}


# "dust" profile: airborne mine dust rather than a smoke plume.  The local
# emitter is deliberately faint and wide so it does not read as a cloud, while
# the global layer keeps enough density to attenuate distant geometry.  The
# glow that made earlier iterations look like a light source came from the
# scattering albedo, not from the density, so ambient_scale stays high here and
# the albedo is lowered instead (see _DUST_ALBEDO_PEAK).
_DUST = {
    "radius": 0.90,
    "couple_rate": 2.0,
    "opacity": 0.80,
    "ambient_scale": 1.10,
    "fuel": 0.65,
    "temperature": 1.60,
    "coupling_scale": 1.00,
}

# Solver defaults for the "dust" profile.  The Fire hierarchy is buoyancy driven
# by its temperature channel; zeroing that buoyancy, damping the rotation and
# shortening the temperature lifetime is what turns the rising plume into
# something that spreads and settles.
_DUST_SOLVER = {
    "buoyancy_per_temp": 0.0,
    "gravity_z": -0.30,
    "vorticity_smoke_mask": 0.25,
    "cooling_rate": 0.25,
    "smoke_damping": 0.08,
    "smoke_fade": 0.30,
    "vorticity_scale": 0.10,
}

# Peak scattering albedo for the "dust" profile.  Below the absorption
# crossover the suspended dust attenuates light instead of adding it, which is
# the difference between "hazy tunnel" and "glowing blob".
_DUST_ALBEDO_PEAK = 0.40


# Colormap ramps.  The Fire hierarchy ships an emissive flame ramp whose top
# control points reach RGB 12 and 32 -- far above 1.0 -- so high-density voxels
# render as self-lit cores.  That is what made earlier dust look like a glowing
# blob rather than suspended particulate.  The dust ramp keeps every channel
# below 1.0 and gives alpha a long tail instead of a steep step.
_LEGACY_COLORMAP_RGB = (
    (0.05, 0.047, 0.041),
    (0.18, 0.16, 0.13),
    (0.42, 0.37, 0.30),
    (3.00, 2.70, 2.20),
    (12.0, 10.5, 8.50),
    (32.0, 28.0, 22.0),
)
_LEGACY_COLORMAP_ALPHA = (0.0, 0.08, 0.18, 0.38, 0.68, 1.0)
_DUST_COLORMAP_RGB = (
    (0.05, 0.042, 0.032),
    (0.25, 0.19, 0.13),
    (0.60, 0.45, 0.30),
    (1.10, 0.82, 0.55),
    (1.60, 1.20, 0.80),
    (2.20, 1.62, 1.05),
)
_DUST_COLORMAP_ALPHA = (0.0, 0.55, 0.78, 0.90, 0.96, 1.0)


def _env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, str(default)))


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, str(default)))


class FlowDust3D:
    """Neutral-gray Flow plume plus low-density ambient suspended dust."""

    ROOT_PATH = "/World/G1MineDustFlow"

    def __init__(
        self,
        table_position: tuple[float, float, float],
        table_yaw: float = 0.0,
    ) -> None:
        self.profile_name = os.environ.get("G1_DUST_PROFILE", "train").lower()
        if self.profile_name not in {"train", "cinematic", "dust"}:
            raise ValueError(
                "G1_DUST_PROFILE must be one of: train, cinematic, dust; "
                f"got {self.profile_name!r}"
            )
        preset_name = os.environ.get("G1_DUST_PRESET", "medium").lower()
        if preset_name not in _PRESETS:
            raise ValueError(
                "G1_DUST_PRESET must be one of: low, medium, heavy; "
                f"got {preset_name!r}"
            )
        self.preset_name = preset_name
        if self.profile_name == "cinematic":
            self.preset = dict(_CINEMATIC)
        elif self.profile_name == "dust":
            self.preset = dict(_DUST)
        else:
            self.preset = dict(_PRESETS[preset_name])
        self.solver = (
            _DUST_SOLVER if self.profile_name == "dust" else {}
        )
        # Optional live placement: point this at a small text file holding
        # "x y z [radius]" in world metres and the emitters follow it each
        # frame.  Framing the dust is otherwise a restart-per-tweak loop
        # because the emitters are authored once, relative to the worktable.
        _pos_file = os.environ.get("G1_DUST_POSITION_FILE", "").strip()
        self._position_file = Path(_pos_file) if _pos_file else None
        self._last_position_text = ""
        self._colormap_prim = None
        self._colormap_rgb_table = _DUST_COLORMAP_RGB
        self._colormap_alpha_table = _DUST_COLORMAP_ALPHA
        self._colormap_gain = 1.0
        self._colormap_opacity = 1.0
        self._brightness_attrs = None
        self.wide = os.environ.get("G1_DUST_WIDE", "0").strip().lower() in {
            "1", "true", "yes", "on"
        }
        self.layout = os.environ.get("G1_DUST_LAYOUT", "default").strip().lower()
        if self.layout not in {"default", "left_right_2x", "left_right_1m", "left_right_075m", "single_1m", "single_1m_soft", "single_1m_ramp", "single_1m_tunnel_ramp", "distributed_field", "box_field", "texture_field"}:
            raise ValueError("Unsupported G1_DUST_LAYOUT: " + self.layout)
        self.left_right_2x = self.layout == "left_right_2x"
        self.left_right_1m = self.layout == "left_right_1m"
        self.left_right_075m = self.layout == "left_right_075m"
        self.single_1m = self.layout == "single_1m"
        self.single_1m_soft = self.layout == "single_1m_soft"
        self.single_1m_ramp = self.layout == "single_1m_ramp"
        self.single_1m_tunnel_ramp = self.layout == "single_1m_tunnel_ramp"
        self.distributed_field = self.layout == "distributed_field"
        self.box_field = self.layout == "box_field"
        self.texture_field = self.layout == "texture_field"
        default_start_opacity = 0.035 if self.single_1m_tunnel_ramp else 0.055
        default_end_opacity = 0.065 if self.single_1m_tunnel_ramp else 0.10
        default_start_ambient = 0.45 if self.single_1m_tunnel_ramp else 0.60
        default_end_ambient = 0.58 if self.single_1m_tunnel_ramp else 0.75
        self._ramp_start_opacity = _env_float("G1_DUST_RAMP_START_OPACITY", default_start_opacity)
        self._ramp_end_opacity = _env_float("G1_DUST_RAMP_END_OPACITY", default_end_opacity)
        self._ramp_start_ambient = _env_float("G1_DUST_RAMP_START_AMBIENT", default_start_ambient)
        self._ramp_end_ambient = _env_float("G1_DUST_RAMP_END_AMBIENT", default_end_ambient)
        self._global_haze_start = _env_float("G1_DUST_GLOBAL_HAZE_START", 0.10)
        self._global_haze_end = _env_float("G1_DUST_GLOBAL_HAZE_END", 0.16)
        if self.single_1m_ramp or self.single_1m_tunnel_ramp:
            for name, value in (("start", self._ramp_start_ambient), ("end", self._ramp_end_ambient)):
                if not 0.2 <= value <= 1.5:
                    raise ValueError(f"G1_DUST_RAMP_{name.upper()}_AMBIENT must be in [0.2, 1.5]")
            if self._ramp_start_opacity < 0.0 or self._ramp_end_opacity < self._ramp_start_opacity:
                raise ValueError("Dust ramp opacity must be non-decreasing and non-negative")
        if self.single_1m_tunnel_ramp and not (0.0 < self._global_haze_start <= self._global_haze_end <= 0.5):
            raise ValueError("G1_DUST_GLOBAL_HAZE_START/END must satisfy 0 < start <= end <= 0.5")
        self.seed = _env_int("G1_DUST_SEED", 260822)
        self._rng = random.Random(self.seed)
        self._velocity_phase = self._rng.uniform(0.0, 2.0 * math.pi)
        self._table_yaw = float(table_yaw)
        self.stage = omni.usd.get_context().get_stage()
        self.settings = carb.settings.get_settings()
        self.layer = _env_int("G1_DUST_FLOW_LAYER", 0)
        self._previous_settings: dict[str, object] = {}
        self._elapsed = 0.0
        self._marker_seen = False
        marker = os.environ.get("G1_DUST_RECORDING_MARKER")
        self._recording_marker = Path(marker) if marker else None
        # No burst parameters, file I/O or USD changes on the default path.
        self._burst_enabled = os.environ.get("G1_DUST_BURST", "0").strip().lower() in {
            "1", "true", "yes", "on"
        }
        if self._burst_enabled and (
            self.profile_name != "dust" or self.layout != "single_1m"
            or os.environ.get("G1_FLOW_DEBUG_FIRE", "0") == "1"
            or os.environ.get("G1_FLOW_EXPERIMENTAL_SMOKE_ONLY", "0") == "1"
        ):
            raise ValueError("Dust burst requires profile=dust, layout=single_1m and the standard Flow hierarchy")

        manager = omni.kit.app.get_app().get_extension_manager()
        if not manager.is_extension_enabled("omni.flowusd"):
            manager.set_extension_enabled_immediate("omni.flowusd", True)
        if not manager.is_extension_enabled("omni.flowusd"):
            raise RuntimeError("Unable to enable the omni.flowusd extension")

        self._previous_layer_data = dict(self.stage.GetRootLayer().customLayerData)
        self._enable_flow_rendering()
        self._configure_ambient_haze()
        self._create_flow_prims(table_position, self._table_yaw)
        if self._burst_enabled:
            self._init_burst()
        print(
            "[flow-dust-3d] Enabled NVIDIA Flow volumetric dust "
            f"(profile={self.profile_name}, preset={self.preset_name}, "
            f"wide={self.wide}, layout={self.layout}, seed={self.seed}, "
            f"layer={self.layer}, emitters={len(getattr(self, '_emitter_enabled', []))}, "
            "densityCellSize=0.06m, neutral-gray Fire-solver fallback)."
        )

    @staticmethod
    def _set_attr(prim, name: str, type_name, value) -> None:
        attr = prim.GetAttribute(name)
        if not attr:
            attr = prim.CreateAttribute(name, type_name, custom=True)
        attr.Set(value)

    def _define(self, path: str, type_name: str):
        prim = self.stage.DefinePrim(path, type_name)
        if not prim:
            raise RuntimeError(f"Unable to define {type_name} at {path}")
        return prim

    def _set_setting(self, path: str, value) -> None:
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

    def _enable_flow_rendering(self) -> None:
        layer_data = dict(self.stage.GetRootLayer().customLayerData)
        render_settings = dict(layer_data.get("renderSettings", {}))
        render_settings.update(
            {
                "rtx:flow:enabled": True,
                "rtx:flow:pathTracingEnabled": True,
                "rtx:flow:rayTracedReflectionsEnabled": True,
                "rtx:flow:rayTracedTranslucencyEnabled": True,
            }
        )
        layer_data["renderSettings"] = render_settings
        self.stage.GetRootLayer().customLayerData = layer_data
        self._set_setting("/rtx/flow/enabled", True)
        self._set_setting("/rtx/translucency/enabled", True)
        self._set_setting("/rtx/reflections/enabled", True)
        self._set_setting("/rtx/indirectDiffuse/enabled", True)

    def _configure_ambient_haze(self) -> None:
        multiplier = _env_float("G1_DUST_AMBIENT_MULTIPLIER", 1.0)
        if not 0.6 <= multiplier <= 1.5:
            raise ValueError("G1_DUST_AMBIENT_MULTIPLIER must be in [0.6, 1.5]")
        scale = float(self.preset["ambient_scale"]) * multiplier
        if self.single_1m_ramp or self.single_1m_tunnel_ramp:
            multiplier = self._ramp_start_ambient
        if self.wide and not self.left_right_2x and not self.left_right_1m and not self.left_right_075m and not self.single_1m and not self.single_1m_soft and not self.single_1m_ramp:
            # The wide demo deliberately increases the scene-wide suspended
            # dust layer, while keeping its color and noise model unchanged.
            scale *= 1.35
        density = 0.24 * scale
        measurement_distance = 12.0 / scale
        fog_distance_density = 0.06 * scale
        self.ambient_multiplier = multiplier
        # Scattering albedo near 1 makes the dust read as a light source
        # (the scene gets brighter).  Lowering it past the absorption
        # crossover makes suspended dust attenuate instead, which is what
        # Beer-Lambert transmittance does to distant geometry.
        _base_albedo = (0.66, 0.63, 0.57)
        _albedo_default = (
            _DUST_ALBEDO_PEAK
            if self.profile_name == "dust"
            else _base_albedo[0]
        )
        _albedo_peak = _env_float(
            "G1_DUST_SCATTER_ALBEDO", _albedo_default
        )
        if not 0.0 <= _albedo_peak <= 1.0:
            raise ValueError("G1_DUST_SCATTER_ALBEDO must be in [0, 1]")
        _albedo_gain = _albedo_peak / _base_albedo[0]
        scatter_albedo = tuple(c * _albedo_gain for c in _base_albedo)
        fog_intensity = _env_float("G1_DUST_FOG_INTENSITY", 0.82)
        settings = {
            "/rtx/raytracing/globalVolumetricEffects/enabled": True,
            "/rtx/raytracing/inscattering/atmosphereHeight": 12.0,
            "/rtx/raytracing/inscattering/maxDistance": 18.0,
            "/rtx/raytracing/inscattering/densityMult": density,
            "/rtx/raytracing/inscattering/transmittanceMeasurementDistance": measurement_distance,
            "/rtx/raytracing/inscattering/transmittanceColor": (0.55, 0.52, 0.46),
            "/rtx/raytracing/inscattering/singleScatteringAlbedo": scatter_albedo,
            "/rtx/raytracing/inscattering/anisotropyFactor": 0.24,
            "/rtx/raytracing/inscattering/useDetailNoise": True,
            "/rtx/raytracing/inscattering/detailNoiseScale": 0.020,
            "/rtx/raytracing/inscattering/noiseAnimationSpeedX": 0.008,
            "/rtx/raytracing/inscattering/noiseAnimationSpeedY": 0.004,
            "/rtx/raytracing/inscattering/noiseAnimationSpeedZ": 0.0015,
            "/rtx/raytracing/inscattering/noiseScaleRangeMin": 0.12,
            "/rtx/raytracing/inscattering/noiseScaleRangeMax": 0.85,
            "/rtx/raytracing/inscattering/noiseNumOctaves": 3,
            "/rtx/raytracing/inscattering/maxAccumulationFrames": 8,
            "/rtx/raytracing/inscattering/depthSlices": _env_int(
                "G1_DUST_INSCT_DEPTH_SLICES", 64
            ),
            # Full-resolution inscattering is what lets a narrow light
            # shaft survive; at the default 1/4 resolution it is blurred
            # away entirely.
            "/rtx/raytracing/inscattering/pixelRatio": _env_int(
                "G1_DUST_INSCT_PIXEL_RATIO", 4
            ),
            "/rtx/fog/enabled": True,
            "/rtx/fog/fogColor": (0.46, 0.43, 0.38),
            "/rtx/fog/fogColorIntensity": fog_intensity,
            "/rtx/fog/fogZup/enabled": True,
            "/rtx/fog/fogStartHeight": 3.0,
            "/rtx/fog/fogHeightDensity": 0.018 * scale,
            "/rtx/fog/fogHeightFalloff": 0.0,
            "/rtx/fog/fogStartDist": 0.05,
            "/rtx/fog/fogEndDist": 4.5,
            "/rtx/fog/fogDistanceDensity": fog_distance_density,
        }
        if self.single_1m_tunnel_ramp:
            # A separate low-density tunnel layer supplies the slow airborne
            # haze; the local Flow source remains responsible for near-field
            # volume and does not need to become opaque.
            settings["/rtx/raytracing/inscattering/densityMult"] = self._global_haze_start
            settings["/rtx/raytracing/inscattering/transmittanceMeasurementDistance"] = 12.0 / max(self._global_haze_start, 1e-3)
            settings["/rtx/raytracing/inscattering/detailNoiseScale"] = 0.045
            settings["/rtx/raytracing/inscattering/noiseAnimationSpeedX"] = 0.030
            settings["/rtx/raytracing/inscattering/noiseAnimationSpeedY"] = 0.018
            settings["/rtx/raytracing/inscattering/noiseAnimationSpeedZ"] = 0.006
            settings["/rtx/raytracing/inscattering/noiseScaleRangeMin"] = 0.05
            settings["/rtx/raytracing/inscattering/noiseScaleRangeMax"] = 0.65
            settings["/rtx/fog/fogDistanceDensity"] = 0.025
            settings["/rtx/fog/fogHeightDensity"] = 0.008
        # Flow's density field can drive the inscattering volume, which --
        # unlike the Flow ray marcher, IS integrated against scene lights.
        # That is the only route to a real light shaft (Tyndall) here.
        if _env_int("G1_DUST_FLOW_SAMPLING", 0) == 1:
            settings["/rtx/raytracing/inscattering/enableFlowSampling"] = True
            settings["/rtx/raytracing/inscattering/flowDensityScale"] = _env_float("G1_DUST_FLOW_DENSITY_SCALE", 1.0)
            settings["/rtx/raytracing/inscattering/flowDensityOffset"] = _env_float("G1_DUST_FLOW_DENSITY_OFFSET", 0.0)
            settings["/rtx/raytracing/inscattering/minFlowLayer"] = 0
            settings["/rtx/raytracing/inscattering/maxFlowLayer"] = 1
        for path, value in settings.items():
            self._set_setting(path, value)
        print(
            "[flow-dust-3d] Ambient haze: "
            f"scale={scale:.3f}, densityMult={density:.3f}, "
            f"measurementDistance={measurement_distance:.3f}m, "
            f"fogDistanceDensity={fog_distance_density:.3f}, "
            f"flowSampling={_env_int('G1_DUST_FLOW_SAMPLING', 0)}, "
            f"scatterAlbedo={scatter_albedo[0]:.3f}/{scatter_albedo[1]:.3f}/"
            f"{scatter_albedo[2]:.3f}, fogIntensity={fog_intensity:.3f}."
        )

    @staticmethod
    def _rotate_xy(x: float, y: float, yaw: float) -> tuple[float, float]:
        cosine = math.cos(yaw)
        sine = math.sin(yaw)
        return cosine * x - sine * y, sine * x + cosine * y

    def _configure_emitter(
        self,
        emitter,
        position: Gf.Vec3f,
        radius: float,
        velocity: Gf.Vec3f,
    ) -> None:
        values = {
            "allocationScale": (Sdf.ValueTypeNames.Float, 1.0),
            "applyPostPressure": (Sdf.ValueTypeNames.Bool, False),
            "burn": (Sdf.ValueTypeNames.Float, 0.0),
            "coupleRateBurn": (Sdf.ValueTypeNames.Float, 0.0),
            "coupleRateDivergence": (Sdf.ValueTypeNames.Float, 0.0),
            "coupleRateFuel": (
                Sdf.ValueTypeNames.Float,
                2.0 * self.preset["coupling_scale"],
            ),
            "coupleRateSmoke": (Sdf.ValueTypeNames.Float, 0.0),
            "coupleRateTemperature": (
                Sdf.ValueTypeNames.Float,
                10.0 * self.preset["coupling_scale"],
            ),
            "coupleRateVelocity": (Sdf.ValueTypeNames.Float, 2.0),
            "divergence": (Sdf.ValueTypeNames.Float, 0.0),
            "enabled": (Sdf.ValueTypeNames.Bool, True),
            "fuel": (Sdf.ValueTypeNames.Float, self.preset["fuel"]),
            "layer": (Sdf.ValueTypeNames.Int, self.layer),
            "multisample": (Sdf.ValueTypeNames.Bool, False),
            "numSubSteps": (Sdf.ValueTypeNames.UInt, 1),
            "physicsVelocityScale": (Sdf.ValueTypeNames.Float, 0.0),
            "position": (Sdf.ValueTypeNames.Float3, position),
            "radius": (Sdf.ValueTypeNames.Float, radius),
            "radiusIsWorldSpace": (Sdf.ValueTypeNames.Bool, True),
            "smoke": (Sdf.ValueTypeNames.Float, 0.0),
            "temperature": (
                Sdf.ValueTypeNames.Float,
                self.preset["temperature"],
            ),
            "velocity": (Sdf.ValueTypeNames.Float3, velocity),
            "velocityIsWorldSpace": (Sdf.ValueTypeNames.Bool, True),
        }
        for name, (type_name, value) in values.items():
            self._set_attr(emitter, name, type_name, value)

    def _configure_box_emitter(
        self,
        emitter,
        position: Gf.Vec3f,
        half_size: Gf.Vec3f,
        velocity: Gf.Vec3f,
    ) -> None:
        """Configure a low-rate tunnel-sized source without a radial front."""
        values = {
            "allocationScale": (Sdf.ValueTypeNames.Float, 1.0),
            "applyPostPressure": (Sdf.ValueTypeNames.Bool, False),
            "burn": (Sdf.ValueTypeNames.Float, 0.0),
            "coupleRateBurn": (Sdf.ValueTypeNames.Float, 0.0),
            "coupleRateDivergence": (Sdf.ValueTypeNames.Float, 0.0),
            "coupleRateFuel": (
                Sdf.ValueTypeNames.Float,
                _env_float("G1_DUST_BOX_FUEL_RATE", 0.18),
            ),
            "coupleRateSmoke": (Sdf.ValueTypeNames.Float, 0.0),
            "coupleRateTemperature": (
                Sdf.ValueTypeNames.Float,
                _env_float("G1_DUST_BOX_TEMPERATURE_RATE", 0.55),
            ),
            "coupleRateVelocity": (Sdf.ValueTypeNames.Float, 0.35),
            "divergence": (Sdf.ValueTypeNames.Float, 0.0),
            "enabled": (Sdf.ValueTypeNames.Bool, True),
            "fuel": (Sdf.ValueTypeNames.Float, 0.42),
            "halfSize": (Sdf.ValueTypeNames.Float3, half_size),
            "isPhysicsCollision": (Sdf.ValueTypeNames.Bool, False),
            "layer": (Sdf.ValueTypeNames.Int, self.layer),
            "level": (Sdf.ValueTypeNames.Int, 0),
            "physicsVelocityScale": (Sdf.ValueTypeNames.Float, 0.0),
            "position": (Sdf.ValueTypeNames.Float3, position),
            "smoke": (Sdf.ValueTypeNames.Float, 0.0),
            "temperature": (Sdf.ValueTypeNames.Float, 0.72),
            "velocity": (Sdf.ValueTypeNames.Float3, velocity),
            "velocityIsWorldSpace": (Sdf.ValueTypeNames.Bool, True),
        }
        for name, (type_name, value) in values.items():
            self._set_attr(emitter, name, type_name, value)

    def _make_dust_texture(
        self, width: int, height: int, depth: int
    ) -> Vt.FloatArray:
        """Build deterministic low-frequency 3D density without a source point."""
        gain = _env_float("G1_DUST_TEXTURE_DENSITY", 0.72)
        values = []
        for z_index in range(depth):
            z = z_index / max(depth - 1, 1)
            for y_index in range(height):
                y = y_index / max(height - 1, 1)
                for x_index in range(width):
                    x = x_index / max(width - 1, 1)
                    low = (
                        0.50
                        + 0.23 * math.sin(math.tau * (1.15 * x + 0.42 * y + 0.31 * z))
                        + 0.16 * math.sin(math.tau * (2.30 * y - 0.71 * z + 0.17))
                        + 0.11 * math.sin(math.tau * (2.05 * x - 1.37 * y + 1.62 * z))
                    )
                    # Small deterministic cell variation breaks smooth bands;
                    # the Flow solver then advects the coarse pattern.
                    jitter = self._rng.uniform(-0.07, 0.07)
                    value = max(0.015, min(1.0, (low + jitter - 0.24) * gain))
                    values.append(value)
        return Vt.FloatArray(values)

    def _configure_texture_emitter(self, emitter) -> None:
        width, height, depth = 24, 44, 16
        density = self._make_dust_texture(width, height, depth)
        fuel = Vt.FloatArray([0.38 * value for value in density])
        values = {
            "allocationScale": (Sdf.ValueTypeNames.Float, 1.0),
            "applyPostPressure": (Sdf.ValueTypeNames.Bool, False),
            "burn": (Sdf.ValueTypeNames.Float, 0.0),
            "coupleRateBurn": (Sdf.ValueTypeNames.Float, 0.0),
            "coupleRateDivergence": (Sdf.ValueTypeNames.Float, 0.0),
            "coupleRateFuel": (
                Sdf.ValueTypeNames.Float,
                _env_float("G1_DUST_TEXTURE_FUEL_RATE", 0.55),
            ),
            "coupleRateSmoke": (Sdf.ValueTypeNames.Float, 0.0),
            "coupleRateTemperature": (
                Sdf.ValueTypeNames.Float,
                _env_float("G1_DUST_TEXTURE_TEMPERATURE_RATE", 1.4),
            ),
            "coupleRateVelocity": (Sdf.ValueTypeNames.Float, 0.55),
            "divergence": (Sdf.ValueTypeNames.Float, 0.0),
            "enabled": (Sdf.ValueTypeNames.Bool, True),
            "fuel": (Sdf.ValueTypeNames.Float, 0.0),
            "fuelScale": (Sdf.ValueTypeNames.Float, 1.0),
            "halfSize": (Sdf.ValueTypeNames.Float3, Gf.Vec3f(3.35, 5.10, 1.82)),
            "layer": (Sdf.ValueTypeNames.Int, self.layer),
            "level": (Sdf.ValueTypeNames.Int, 0),
            "position": (Sdf.ValueTypeNames.Float3, Gf.Vec3f(-1.0, 1.85, 1.92)),
            "smoke": (Sdf.ValueTypeNames.Float, 0.0),
            "temperature": (Sdf.ValueTypeNames.Float, 0.0),
            "temperatureScale": (Sdf.ValueTypeNames.Float, 1.0),
            "textureDepth": (Sdf.ValueTypeNames.UInt, depth),
            "textureFuels": (Sdf.ValueTypeNames.FloatArray, fuel),
            "textureHeight": (Sdf.ValueTypeNames.UInt, height),
            "textureTemperatures": (Sdf.ValueTypeNames.FloatArray, density),
            "textureWidth": (Sdf.ValueTypeNames.UInt, width),
            "velocity": (Sdf.ValueTypeNames.Float3, Gf.Vec3f(0.018, 0.040, 0.004)),
            "velocityIsWorldSpace": (Sdf.ValueTypeNames.Bool, True),
            "velocityScale": (Sdf.ValueTypeNames.Float, 1.0),
        }
        for name, (type_name, value) in values.items():
            self._set_attr(emitter, name, type_name, value)

    def _create_flow_prims(
        self,
        table_position: tuple[float, float, float],
        table_yaw: float,
    ) -> None:
        root = UsdGeom.Xform.Define(self.stage, self.ROOT_PATH).GetPrim()
        if not root:
            raise RuntimeError(f"Unable to define Flow root at {self.ROOT_PATH}")
        # Start from NVIDIA's locally installed, tested Flow hierarchy.  This
        # supplies every required channel/debug/cloud child prim.  Local USD
        # opinions below turn that hierarchy into a smoke-only dust plume;
        # there is no network dependency and the source preset is not edited.
        preset_candidates = sorted(
            Path(os.environ.get("ISAACSIM_ROOT", str(Path.home() / "isaacsim"))).joinpath(
                "extscache"
            ).glob(
                "omni.flowusd-*/data/presets/Fire/Fire.usda"
            )
        )
        if not preset_candidates:
            raise FileNotFoundError("Installed NVIDIA Flow Fire preset not found")
        preset_path = preset_candidates[-1]
        root.GetReferences().AddReference(str(preset_path), "/Fire")
        print(f"[flow-dust-3d] Base preset: {preset_path}")

        if os.environ.get("G1_FLOW_DEBUG_FIRE", "0") == "1":
            emitter = self.stage.GetPrimAtPath(
                f"{self.ROOT_PATH}/flowEmitterSphere"
            )
            simulate = self.stage.GetPrimAtPath(f"{self.ROOT_PATH}/flowSimulate")
            offscreen = self.stage.GetPrimAtPath(f"{self.ROOT_PATH}/flowOffscreen")
            render = self.stage.GetPrimAtPath(f"{self.ROOT_PATH}/flowRender")
            table_x, table_y, table_z = map(float, table_position)
            emitter.GetAttribute("position").Set(
                Gf.Vec3f(table_x, table_y, table_z + 1.0)
            )
            emitter.GetAttribute("radius").Set(0.12)
            emitter.GetAttribute("radiusIsWorldSpace").Set(True)
            emitter.GetAttribute("enabled").Set(True)
            emitter.GetAttribute("layer").Set(0)
            simulate.GetAttribute("densityCellSize").Set(0.06)
            simulate.GetAttribute("forceSimulate").Set(True)
            simulate.GetAttribute("layer").Set(0)
            offscreen.GetAttribute("layer").Set(0)
            render.GetAttribute("layer").Set(0)
            advection = self.stage.GetPrimAtPath(
                f"{self.ROOT_PATH}/flowSimulate/advection"
            )
            advection.GetAttribute("gravity").Set(Gf.Vec3f(0.0, 0.0, -0.2))
            advection.GetAttribute("buoyancyPerTemp").Set(0.3)
            self._emitter_enabled = [emitter.GetAttribute("enabled")]
            self._emitter_velocity = [emitter.GetAttribute("velocity")]
            self._base_velocities = [Gf.Vec3f(0.0, 0.0, 0.0)]
            print("[flow-dust-3d] DEBUG: using unmodified NVIDIA Fire rendering")
            return

        if os.environ.get("G1_FLOW_EXPERIMENTAL_SMOKE_ONLY", "0") != "1":
            emitter = self.stage.GetPrimAtPath(
                f"{self.ROOT_PATH}/flowEmitterSphere"
            )
            simulate = self.stage.GetPrimAtPath(f"{self.ROOT_PATH}/flowSimulate")
            offscreen = self.stage.GetPrimAtPath(f"{self.ROOT_PATH}/flowOffscreen")
            render = self.stage.GetPrimAtPath(f"{self.ROOT_PATH}/flowRender")
            advection = self.stage.GetPrimAtPath(
                f"{self.ROOT_PATH}/flowSimulate/advection"
            )
            colormap = self.stage.GetPrimAtPath(
                f"{self.ROOT_PATH}/flowOffscreen/colormap"
            )
            table_x, table_y, table_z = map(float, table_position)
            # Emitters were only ever placed relative to the worktable, so
            # the dust could not be moved somewhere it reads better (a
            # tunnel with real depth, for instance).  An explicit world
            # centre replaces that origin without touching any layout.
            _centre = os.environ.get("G1_DUST_EMITTER_CENTER", "").strip()
            if _centre:
                _parts = [float(v) for v in _centre.split(",")]
                if len(_parts) != 3:
                    raise ValueError(
                        "G1_DUST_EMITTER_CENTER must be three "
                        "comma-separated world coordinates"
                    )
                table_x, table_y, table_z = _parts
            if self.texture_field:
                emitter.GetAttribute("enabled").Set(False)
                texture = self._define(
                    f"{self.ROOT_PATH}/flowEmitterTexture",
                    "FlowEmitterTexture",
                )
                self._configure_texture_emitter(texture)
                self._emitter_enabled = [texture.GetAttribute("enabled")]
                self._emitter_velocity = [texture.GetAttribute("velocity")]
                self._base_velocities = [Gf.Vec3f(0.018, 0.040, 0.004)]
            elif self.box_field:
                # Disable the referenced Fire sphere and inject through broad,
                # shallow sheets staggered in depth and height.  The sheets
                # advect along the tunnel, producing drifting dust banks rather
                # than a radial puff or one uniform screen-space veil.
                emitter.GetAttribute("enabled").Set(False)
                box_specs = [
                    (
                        Gf.Vec3f(-1.65, -1.55, 2.55),
                        Gf.Vec3f(1.55, 0.78, 0.48),
                        Gf.Vec3f(0.025, 0.075, -0.010),
                    ),
                    (
                        Gf.Vec3f(-0.30, -0.35, 1.20),
                        Gf.Vec3f(1.95, 0.55, 0.62),
                        Gf.Vec3f(-0.018, 0.085, 0.012),
                    ),
                    (
                        Gf.Vec3f(-1.85, 0.75, 2.18),
                        Gf.Vec3f(1.45, 0.62, 0.70),
                        Gf.Vec3f(0.030, 0.070, -0.008),
                    ),
                    (
                        Gf.Vec3f(0.15, 1.55, 1.02),
                        Gf.Vec3f(1.55, 0.52, 0.55),
                        Gf.Vec3f(-0.022, 0.090, 0.016),
                    ),
                    (
                        Gf.Vec3f(-1.00, 2.65, 1.72),
                        Gf.Vec3f(2.25, 0.78, 0.68),
                        Gf.Vec3f(0.018, 0.065, 0.006),
                    ),
                    (
                        Gf.Vec3f(-2.00, 4.05, 0.88),
                        Gf.Vec3f(1.30, 0.58, 0.48),
                        Gf.Vec3f(0.028, 0.080, 0.014),
                    ),
                    (
                        Gf.Vec3f(-0.10, 4.85, 2.42),
                        Gf.Vec3f(1.65, 0.68, 0.58),
                        Gf.Vec3f(-0.016, 0.060, -0.010),
                    ),
                    (
                        Gf.Vec3f(-1.05, 6.15, 1.48),
                        Gf.Vec3f(2.05, 0.82, 0.72),
                        Gf.Vec3f(0.020, 0.055, 0.005),
                    ),
                ]
                boxes = []
                velocities = []
                for index, (position, half_size, velocity) in enumerate(box_specs):
                    box = self._define(
                        f"{self.ROOT_PATH}/flowEmitterBox{index}",
                        "FlowEmitterBox",
                    )
                    self._configure_box_emitter(box, position, half_size, velocity)
                    boxes.append(box)
                    velocities.append(velocity)
                self._emitter_enabled = [item.GetAttribute("enabled") for item in boxes]
                self._emitter_velocity = [item.GetAttribute("velocity") for item in boxes]
                self._base_velocities = velocities
            elif self.distributed_field:
                # Sparse emitters throughout the visible tunnel create a
                # heterogeneous field with no privileged centre or radial
                # front.  Each source is smaller than the old one-metre sphere
                # and moves slowly in a different direction.  The Flow solver
                # still provides true world-space advection and occlusion.
                radius = _env_float("G1_DUST_EMITTER_RADIUS", 0.34)
                local_offsets = [
                    (-1.65, -0.55, 0.62),
                    (-0.25, -0.15, 1.45),
                    (1.25, 0.25, 0.88),
                    (-1.15, 1.55, 2.35),
                    (0.65, 1.95, 1.65),
                    (1.65, 3.15, 2.65),
                    (-1.55, 3.75, 1.15),
                    (0.10, 4.65, 0.72),
                    (1.25, 5.55, 1.78),
                ]
                local_velocities = [
                    (0.025, 0.020, 0.010),
                    (-0.020, 0.030, 0.006),
                    (0.015, 0.024, -0.004),
                    (0.030, -0.012, 0.008),
                    (-0.028, 0.014, 0.004),
                    (0.012, 0.025, -0.006),
                    (-0.018, 0.022, 0.005),
                    (0.026, -0.010, 0.003),
                    (-0.014, 0.018, -0.003),
                ]
            elif self.left_right_2x or self.left_right_1m or self.left_right_075m or self.single_1m or self.single_1m_soft or self.single_1m_ramp or self.single_1m_tunnel_ramp:
                # Two broad, overlapping sources remove the isolated circular
                # plume impression while keeping a deterministic left/right
                # composition. The 0.68 m radius is the 2x Medium layout;
                # left_right_1m uses two 1.0 m sources, left_right_075m uses
                # two 0.75 m sources, and single_1m variants use one centered
                # 1.0 m source for a simpler full-screen cloud.
                if self.left_right_075m:
                    radius = 0.75
                else:
                    radius = 1.0 if (self.left_right_1m or self.single_1m or self.single_1m_soft or self.single_1m_ramp or self.single_1m_tunnel_ramp) else float(self.preset["radius"]) * 2.0
                if self.single_1m or self.single_1m_soft or self.single_1m_ramp or self.single_1m_tunnel_ramp:
                    local_offsets = [(0.00, 0.00, 1.02)]
                    local_velocities = [(0.08, 0.00, 0.038)]
                else:
                    local_offsets = [(-0.70, 0.20, 1.05), (0.70, -0.15, 1.00)]
                    local_velocities = [(0.18, -0.035, 0.040), (-0.14, 0.055, 0.032)]
            else:
                radius = float(self.preset["radius"]) * (1.70 if self.wide else 1.0)
                local_offsets = (
                [(-0.52, 0.38, 1.12), (0.52, -0.30, 1.02)]
                if self.wide
                else [(-0.35, 0.20, 1.00), (0.35, -0.15, 0.93)]
                )
                local_velocities = (
                    [(0.15, -0.03, 0.045), (-0.10, 0.06, 0.035)]
                    if self.wide
                    else [(0.10, -0.02, 0.035), (-0.06, 0.04, 0.025)]
                )
            if not self.box_field and not self.texture_field:
                positions: list[Gf.Vec3f] = []
                velocities = []
                for index, ((ox, oy, oz), (vx, vy, vz)) in enumerate(
                    zip(local_offsets, local_velocities, strict=True)
                ):
                    ox += self._rng.uniform(-0.035, 0.035)
                    oy += self._rng.uniform(-0.035, 0.035)
                    vx += self._rng.uniform(-0.012, 0.012)
                    vy += self._rng.uniform(-0.012, 0.012)
                    if not self.distributed_field:
                        radius = _env_float("G1_DUST_EMITTER_RADIUS", radius)
                    if self.distributed_field:
                        # This layout is authored in tunnel/world axes.  The
                        # worktable itself is rotated by about 60 degrees;
                        # applying that yaw would push the deep sources through
                        # the left wall and out of the camera frustum.
                        world_x, world_y = ox, oy
                        velocity_x, velocity_y = vx, vy
                    else:
                        world_x, world_y = self._rotate_xy(ox, oy, table_yaw)
                        velocity_x, velocity_y = self._rotate_xy(vx, vy, table_yaw)
                    positions.append(
                        Gf.Vec3f(table_x + world_x, table_y + world_y, table_z + oz)
                    )
                    velocities.append(Gf.Vec3f(velocity_x, velocity_y, vz))

                self._configure_emitter(emitter, positions[0], radius, velocities[0])
                if self.single_1m or self.single_1m_soft or self.single_1m_ramp or self.single_1m_tunnel_ramp:
                    self._emitter_enabled = [emitter.GetAttribute("enabled")]
                    self._emitter_velocity = [emitter.GetAttribute("velocity")]
                else:
                    emitters = [emitter]
                    for index in range(1, len(positions)):
                        suffix = chr(ord("A") + index)
                        extra = self._define(
                            f"{self.ROOT_PATH}/flowEmitterSphere{suffix}",
                            "FlowEmitterSphere",
                        )
                        self._configure_emitter(extra, positions[index], radius, velocities[index])
                        emitters.append(extra)
                    self._emitter_enabled = [item.GetAttribute("enabled") for item in emitters]
                    self._emitter_velocity = [item.GetAttribute("velocity") for item in emitters]
                self._base_velocities = velocities
            simulate.GetAttribute("densityCellSize").Set(
                _env_float(
                    "G1_DUST_FLOW_CELL_SIZE",
                    0.11 if (self.box_field or self.texture_field) else 0.06,
                )
            )
            simulate.GetAttribute("enableLowPrecisionDensity").Set(False)
            simulate.GetAttribute("enableLowPrecisionVelocity").Set(False)
            simulate.GetAttribute("forceSimulate").Set(True)
            simulate.GetAttribute("layer").Set(self.layer)
            offscreen.GetAttribute("layer").Set(self.layer)
            render.GetAttribute("layer").Set(self.layer)
            advection.GetAttribute("combustionEnabled").Set(True)
            advection.GetAttribute("burnPerTemp").Set(4.0)
            advection.GetAttribute("fuelPerBurn").Set(0.25)
            advection.GetAttribute("smokePerBurn").Set(3.0)
            advection.GetAttribute("tempPerBurn").Set(5.0)
            advection.GetAttribute("gravity").Set(
                Gf.Vec3f(
                    0.0,
                    0.0,
                    _env_float(
                        "G1_DUST_GRAVITY_Z",
                        self.solver.get("gravity_z", -0.20),
                    ),
                )
            )
            advection.GetAttribute("buoyancyPerSmoke").Set(0.0)
            advection.GetAttribute("buoyancyPerTemp").Set(
                _env_float(
                    "G1_DUST_BUOYANCY_PER_TEMP",
                    self.solver.get("buoyancy_per_temp", 0.30),
                )
            )
            advection.GetAttribute("coolingRate").Set(
                _env_float(
                    "G1_DUST_COOLING_RATE",
                    self.solver.get("cooling_rate", 1.5),
                )
            )
            smoke_channel = self.stage.GetPrimAtPath(
                f"{self.ROOT_PATH}/flowSimulate/advection/smoke"
            )
            smoke_channel.GetAttribute("damping").Set(
                _env_float(
                    "G1_DUST_SMOKE_DAMPING",
                    self.solver.get("smoke_damping", 0.25),
                )
            )
            if self.single_1m_tunnel_ramp:
                fade = 0.65
            elif self.single_1m_soft:
                fade = 0.55
            elif self.single_1m or self.single_1m_ramp:
                fade = 0.40
            elif self.left_right_075m:
                fade = 0.55
            elif self.left_right_1m:
                fade = 0.45
            elif self.left_right_2x:
                fade = 0.58
            elif self.wide:
                fade = 0.80
            else:
                fade = 1.2
            if self.profile_name == "dust":
                fade = self.solver["smoke_fade"]
            if self.distributed_field:
                fade = 0.52
            if self.box_field:
                fade = 0.50
            if self.texture_field:
                fade = 0.46
            fade = _env_float("G1_DUST_SMOKE_FADE", fade)
            smoke_channel.GetAttribute("fade").Set(fade)
            vorticity = self.stage.GetPrimAtPath(
                f"{self.ROOT_PATH}/flowSimulate/vorticity"
            )
            vorticity.GetAttribute("enabled").Set(True)
            if self.single_1m_tunnel_ramp:
                force_scale = 0.45
            elif self.single_1m_soft:
                force_scale = 0.60
            elif self.single_1m or self.single_1m_ramp:
                force_scale = 0.70
            elif self.left_right_075m:
                force_scale = 0.72
            elif self.left_right_1m:
                force_scale = 0.76
            elif self.left_right_2x:
                force_scale = 0.72
            elif self.wide:
                force_scale = 0.65
            else:
                force_scale = 0.55
            if self.profile_name == "dust":
                force_scale = self.solver["vorticity_scale"]
            if self.box_field:
                force_scale = 0.34
            force_scale = _env_float("G1_DUST_VORTICITY_SCALE", force_scale)
            vorticity.GetAttribute("forceScale").Set(force_scale)
            vorticity.GetAttribute("burnMask").Set(0.0)
            vorticity.GetAttribute("fuelMask").Set(0.0)
            vorticity.GetAttribute("temperatureMask").Set(0.0)
            vorticity.GetAttribute("smokeMask").Set(
                _env_float(
                    "G1_DUST_VORTICITY_SMOKE_MASK",
                    self.solver.get("vorticity_smoke_mask", 1.0),
                )
            )
            vorticity.GetAttribute("velocityMask").Set(0.35)
            summary = self.stage.GetPrimAtPath(
                f"{self.ROOT_PATH}/flowSimulate/summaryAllocate"
            )
            summary.GetAttribute("smokeThreshold").Set(0.005)
            summary.GetAttribute("speedThreshold").Set(0.02)
            summary.GetAttribute("speedThresholdMinSmoke").Set(0.001)
            opacity = _env_float("G1_FLOW_OPACITY", self.preset["opacity"])
            if self.single_1m_ramp or self.single_1m_tunnel_ramp:
                opacity = self._ramp_start_opacity
            if self.profile_name == "dust":
                rgb_table = _DUST_COLORMAP_RGB
                alpha_table = _DUST_COLORMAP_ALPHA
            else:
                rgb_table = _LEGACY_COLORMAP_RGB
                alpha_table = _LEGACY_COLORMAP_ALPHA
            color_gain = _env_float("G1_DUST_COLOR_GAIN", 1.0)
            self._colormap_rgb_table = rgb_table
            self._colormap_alpha_table = alpha_table
            self._colormap_gain = color_gain
            self._colormap_opacity = opacity
            self._colormap_prim = colormap
            self._colormap_alpha_multipliers = alpha_table
            colormap.GetAttribute("colorScale").Set(
                _env_float(
                    "G1_DUST_COLOR_SCALE",
                    1.0 if self.profile_name == "dust" else 2.0,
                )
            )
            colormap.GetAttribute("xPoints").Set(
                Vt.FloatArray([0.0, 0.05, 0.15, 0.45, 0.75, 1.0])
            )
            colormap.GetAttribute("colorScalePoints").Set(
                Vt.FloatArray([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
            )
            colormap.GetAttribute("rgbaPoints").Set(
                Vt.Vec4fArray(
                    [
                        Gf.Vec4f(
                            rgb[0] * color_gain,
                            rgb[1] * color_gain,
                            rgb[2] * color_gain,
                            alpha * opacity,
                        )
                        for rgb, alpha in zip(rgb_table, alpha_table)
                    ]
                )
            )
            print(
                "[flow-dust-3d] Using the installed NVIDIA hierarchy with "
                "neutral-gray dust shading and meter-scale dynamics."
            )
            return

        emitter = self._define(
            f"{self.ROOT_PATH}/flowEmitterSphere", "FlowEmitterSphere"
        )
        table_x, table_y, table_z = map(float, table_position)
        emitter_position = Gf.Vec3f(
            table_x - 0.32,
            table_y + 0.18,
            table_z + 0.94,
        )
        emitter_values = {
            "allocationScale": (Sdf.ValueTypeNames.Float, 1.6),
            "applyPostPressure": (Sdf.ValueTypeNames.Bool, False),
            "burn": (Sdf.ValueTypeNames.Float, 0.0),
            "coupleRateBurn": (Sdf.ValueTypeNames.Float, 0.0),
            "coupleRateDivergence": (Sdf.ValueTypeNames.Float, 0.0),
            "coupleRateFuel": (Sdf.ValueTypeNames.Float, 2.0),
            "coupleRateSmoke": (
                Sdf.ValueTypeNames.Float,
                _env_float("G1_FLOW_COUPLE_RATE", self.preset["couple_rate"]),
            ),
            # The standard Flow ray marcher indexes its colormap with the
            # temperature density.  It is used here only as a visible scalar;
            # combustion and thermal buoyancy remain disabled.
            "coupleRateTemperature": (Sdf.ValueTypeNames.Float, 8.0),
            "coupleRateVelocity": (Sdf.ValueTypeNames.Float, 3.0),
            "divergence": (Sdf.ValueTypeNames.Float, 0.0),
            "enabled": (Sdf.ValueTypeNames.Bool, False),
            "fuel": (
                Sdf.ValueTypeNames.Float,
                {"low": 0.22, "medium": 0.34, "heavy": 0.48}[
                    self.preset_name
                ],
            ),
            "layer": (Sdf.ValueTypeNames.Int, self.layer),
            "multisample": (Sdf.ValueTypeNames.Bool, True),
            "numSubSteps": (Sdf.ValueTypeNames.UInt, 2),
            "physicsVelocityScale": (Sdf.ValueTypeNames.Float, 0.0),
            "position": (Sdf.ValueTypeNames.Float3, emitter_position),
            "radius": (
                Sdf.ValueTypeNames.Float,
                _env_float("G1_FLOW_RADIUS", self.preset["radius"]),
            ),
            "radiusIsWorldSpace": (Sdf.ValueTypeNames.Bool, True),
            "smoke": (
                Sdf.ValueTypeNames.Float,
                _env_float("G1_FLOW_SMOKE", self.preset["smoke"]),
            ),
            "temperature": (
                Sdf.ValueTypeNames.Float,
                {"low": 0.55, "medium": 0.75, "heavy": 0.95}[
                    self.preset_name
                ],
            ),
            "velocity": (Sdf.ValueTypeNames.Float3, Gf.Vec3f(0.13, -0.05, 0.055)),
            "velocityIsWorldSpace": (Sdf.ValueTypeNames.Bool, True),
        }
        for name, (type_name, value) in emitter_values.items():
            self._set_attr(emitter, name, type_name, value)
        self._emitter_enabled = [emitter.GetAttribute("enabled")]
        self._emitter_velocity = [emitter.GetAttribute("velocity")]
        self._base_velocities = [Gf.Vec3f(0.13, -0.05, 0.055)]

        simulate = self._define(
            f"{self.ROOT_PATH}/flowSimulate", "FlowSimulate"
        )
        for name, type_name, value in (
            ("blockMinLifetime", Sdf.ValueTypeNames.UInt, 6),
            ("densityCellSize", Sdf.ValueTypeNames.Float, 0.05),
            ("enableLowPrecisionDensity", Sdf.ValueTypeNames.Bool, True),
            ("enableLowPrecisionVelocity", Sdf.ValueTypeNames.Bool, True),
            ("enableSmallBlocks", Sdf.ValueTypeNames.Bool, True),
            ("enableVariableTimeStep", Sdf.ValueTypeNames.Bool, False),
            ("forceClear", Sdf.ValueTypeNames.Bool, False),
            ("forceDisableCoreSimulation", Sdf.ValueTypeNames.Bool, False),
            ("forceDisableEmitters", Sdf.ValueTypeNames.Bool, False),
            # IsaacLab advances SimulationContext directly rather than driving
            # the Kit timeline. Flow therefore needs forceSimulate in headless
            # camera/replay jobs or its USD prims render but never accumulate.
            ("forceSimulate", Sdf.ValueTypeNames.Bool, True),
            ("interpolateTimeSteps", Sdf.ValueTypeNames.Bool, True),
            ("layer", Sdf.ValueTypeNames.Int, self.layer),
            ("maxStepsPerSimulate", Sdf.ValueTypeNames.UInt, 2),
            ("physicsCollisionEnabled", Sdf.ValueTypeNames.Bool, False),
            ("simulateWhenPaused", Sdf.ValueTypeNames.Bool, False),
            ("stepsPerSecond", Sdf.ValueTypeNames.Float, 60.0),
            ("timeScale", Sdf.ValueTypeNames.Float, 1.0),
            ("velocitySubSteps", Sdf.ValueTypeNames.UInt, 1),
        ):
            self._set_attr(simulate, name, type_name, value)

        advection = self._define(
            f"{self.ROOT_PATH}/flowSimulate/advection",
            "FlowAdvectionCombustionParams",
        )
        for name, type_name, value in (
            ("buoyancyMaxSmoke", Sdf.ValueTypeNames.Float, 1.0),
            ("buoyancyPerSmoke", Sdf.ValueTypeNames.Float, 0.02),
            ("buoyancyPerTemp", Sdf.ValueTypeNames.Float, 0.18),
            ("burnPerTemp", Sdf.ValueTypeNames.Float, 2.5),
            ("combustionEnabled", Sdf.ValueTypeNames.Bool, True),
            ("coolingRate", Sdf.ValueTypeNames.Float, 0.9),
            ("fuelPerBurn", Sdf.ValueTypeNames.Float, 0.25),
            ("smokePerBurn", Sdf.ValueTypeNames.Float, 2.0),
            ("tempPerBurn", Sdf.ValueTypeNames.Float, 0.8),
            ("downsampleEnabled", Sdf.ValueTypeNames.Bool, True),
            ("enabled", Sdf.ValueTypeNames.Bool, True),
            ("forceFadeEnabled", Sdf.ValueTypeNames.Bool, False),
            ("globalFetch", Sdf.ValueTypeNames.Bool, False),
            ("gravity", Sdf.ValueTypeNames.Float3, Gf.Vec3f(0.0, 0.0, -0.15)),
        ):
            self._set_attr(advection, name, type_name, value)

        smoke_channel = self._define(
            f"{self.ROOT_PATH}/flowSimulate/advection/smoke",
            "FlowAdvectionChannelParams",
        )
        for name, value in (
            ("damping", 0.12),
            ("fade", 0.35),
            ("secondOrderBlendFactor", 0.85),
            ("secondOrderBlendThreshold", 0.001),
        ):
            self._set_attr(smoke_channel, name, Sdf.ValueTypeNames.Float, value)

        velocity_channel = self._define(
            f"{self.ROOT_PATH}/flowSimulate/advection/velocity",
            "FlowAdvectionChannelParams",
        )
        for name, value in (
            ("damping", 0.04),
            ("fade", 0.18),
            ("secondOrderBlendFactor", 0.55),
            ("secondOrderBlendThreshold", 0.001),
        ):
            self._set_attr(velocity_channel, name, Sdf.ValueTypeNames.Float, value)

        burn_channel = self._define(
            f"{self.ROOT_PATH}/flowSimulate/advection/burn",
            "FlowAdvectionChannelParams",
        )
        for name, value in (
            ("damping", 0.12),
            ("fade", 0.35),
            ("secondOrderBlendFactor", 0.85),
            ("secondOrderBlendThreshold", 0.001),
        ):
            self._set_attr(burn_channel, name, Sdf.ValueTypeNames.Float, value)

        temperature_channel = self._define(
            f"{self.ROOT_PATH}/flowSimulate/advection/temperature",
            "FlowAdvectionChannelParams",
        )
        for name, value in (
            ("damping", 0.12),
            ("fade", 0.35),
            ("secondOrderBlendFactor", 0.85),
            ("secondOrderBlendThreshold", 0.001),
        ):
            self._set_attr(
                temperature_channel, name, Sdf.ValueTypeNames.Float, value
            )

        vorticity = self._define(
            f"{self.ROOT_PATH}/flowSimulate/vorticity", "FlowVorticityParams"
        )
        for name, type_name, value in (
            ("enabled", Sdf.ValueTypeNames.Bool, True),
            ("forceScale", Sdf.ValueTypeNames.Float, 0.45),
            ("burnMask", Sdf.ValueTypeNames.Float, 1.0),
            ("smokeMask", Sdf.ValueTypeNames.Float, 1.0),
            ("temperatureMask", Sdf.ValueTypeNames.Float, 1.0),
            ("velocityMask", Sdf.ValueTypeNames.Float, 0.35),
            ("velocityLinearMask", Sdf.ValueTypeNames.Float, 0.15),
            ("velocityLogScale", Sdf.ValueTypeNames.Float, 1.0),
        ):
            self._set_attr(vorticity, name, type_name, value)

        pressure = self._define(
            f"{self.ROOT_PATH}/flowSimulate/pressure", "FlowPressureParams"
        )
        self._set_attr(pressure, "enabled", Sdf.ValueTypeNames.Bool, True)

        summary = self._define(
            f"{self.ROOT_PATH}/flowSimulate/summaryAllocate",
            "FlowSummaryAllocateParams",
        )
        for name, type_name, value in (
            ("enableNeighborAllocation", Sdf.ValueTypeNames.Bool, True),
            ("smokeThreshold", Sdf.ValueTypeNames.Float, 0.005),
            ("speedThreshold", Sdf.ValueTypeNames.Float, 0.02),
            ("speedThresholdMinSmoke", Sdf.ValueTypeNames.Float, 0.001),
        ):
            self._set_attr(summary, name, type_name, value)

        offscreen = self._define(
            f"{self.ROOT_PATH}/flowOffscreen", "FlowOffscreen"
        )
        self._set_attr(offscreen, "layer", Sdf.ValueTypeNames.Int, self.layer)
        colormap = self._define(
            f"{self.ROOT_PATH}/flowOffscreen/colormap",
            "FlowRayMarchColormapParams",
        )
        opacity = _env_float("G1_FLOW_OPACITY", self.preset["opacity"])
        self._set_attr(colormap, "colorScale", Sdf.ValueTypeNames.Float, 3.0)
        self._set_attr(colormap, "resolution", Sdf.ValueTypeNames.UInt, 32)
        self._set_attr(
            colormap,
            "xPoints",
            Sdf.ValueTypeNames.FloatArray,
            Vt.FloatArray([0.0, 0.12, 0.45, 1.0]),
        )
        self._set_attr(
            colormap,
            "colorScalePoints",
            Sdf.ValueTypeNames.FloatArray,
            Vt.FloatArray([1.0, 1.0, 1.0, 1.0]),
        )
        self._set_attr(
            colormap,
            "rgbaPoints",
            Sdf.ValueTypeNames.Float4Array,
            Vt.Vec4fArray(
                [
                    Gf.Vec4f(0.16, 0.15, 0.14, 0.0),
                    Gf.Vec4f(0.32, 0.30, 0.27, 0.28 * opacity),
                    Gf.Vec4f(0.50, 0.47, 0.42, 0.78 * opacity),
                    Gf.Vec4f(0.68, 0.64, 0.58, opacity),
                ]
            ),
        )

        shadow = self._define(
            f"{self.ROOT_PATH}/flowOffscreen/shadow", "FlowShadowParams"
        )
        for name, type_name, value in (
            ("attenuation", Sdf.ValueTypeNames.Float, 0.025),
            ("coarsePropagate", Sdf.ValueTypeNames.Bool, True),
            ("enabled", Sdf.ValueTypeNames.Bool, True),
            ("isPointLight", Sdf.ValueTypeNames.Bool, False),
            ("lightDirection", Sdf.ValueTypeNames.Float3, Gf.Vec3f(0.4, -0.2, 1.0)),
            ("minIntensity", Sdf.ValueTypeNames.Float, 0.30),
            ("numSteps", Sdf.ValueTypeNames.UInt, 12),
            ("stepSizeScale", Sdf.ValueTypeNames.Float, 1.0),
        ):
            self._set_attr(shadow, name, type_name, value)

        render = self._define(f"{self.ROOT_PATH}/flowRender", "FlowRender")
        self._set_attr(render, "layer", Sdf.ValueTypeNames.Int, self.layer)
        raymarch = self._define(
            f"{self.ROOT_PATH}/flowRender/rayMarch", "FlowRayMarchParams"
        )
        for name, type_name, value in (
            ("attenuation", Sdf.ValueTypeNames.Float, 1.5),
            ("colorScale", Sdf.ValueTypeNames.Float, 3.0),
            ("enableBlockWireframe", Sdf.ValueTypeNames.Bool, False),
            ("enableRawMode", Sdf.ValueTypeNames.Bool, False),
            ("shadowFactor", Sdf.ValueTypeNames.Float, 0.55),
            ("stepSizeScale", Sdf.ValueTypeNames.Float, 0.8),
        ):
            self._set_attr(raymarch, name, type_name, value)

        cloud = self._define(
            f"{self.ROOT_PATH}/flowRender/rayMarch/cloud",
            "FlowRayMarchCloudParams",
        )
        cloud_density = {"low": 1.2, "medium": 1.8, "heavy": 2.8}[
            self.preset_name
        ]
        for name, type_name, value in (
            ("ambientColor", Sdf.ValueTypeNames.Float3, Gf.Vec3f(0.52, 0.50, 0.46)),
            ("ambientMultiplier", Sdf.ValueTypeNames.Float, 0.75),
            (
                "attenuationMultiplier",
                Sdf.ValueTypeNames.Float3,
                Gf.Vec3f(1.0, 0.96, 0.90),
            ),
            ("densityMultiplier", Sdf.ValueTypeNames.Float, cloud_density),
            ("enableCloudMode", Sdf.ValueTypeNames.Bool, False),
            ("numShadowSteps", Sdf.ValueTypeNames.Int, 8),
            ("shadowStepMultiplier", Sdf.ValueTypeNames.Float, 1.0),
            ("sunDirection", Sdf.ValueTypeNames.Float3, Gf.Vec3f(0.4, -0.2, 1.0)),
            (
                "volumeBaseColor",
                Sdf.ValueTypeNames.Float3,
                Gf.Vec3f(0.48, 0.46, 0.42),
            ),
            ("volumeColorMultiplier", Sdf.ValueTypeNames.Float, 0.85),
        ):
            self._set_attr(cloud, name, type_name, value)

    def _scene_light_attrs(self):
        """Lazily locate the DomeLight and the NuRec emissive CCM attributes."""
        if self._brightness_attrs is None:
            attrs = {"dome": None, "nurec": []}
            dome_path = os.environ.get("G1_HEADLAMP_DOME_PATH", "/World/light")
            prim = self.stage.GetPrimAtPath(dome_path)
            if prim and prim.IsValid():
                attrs["dome"] = prim.GetAttribute("inputs:intensity")
            for candidate in self.stage.Traverse():
                role = candidate.GetAttribute("fieldRole")
                if not role or role.Get() != "emissiveColor":
                    continue
                ccm = [
                    candidate.GetAttribute(name)
                    for name in ("omni:nurec:ccmR", "omni:nurec:ccmG", "omni:nurec:ccmB")
                ]
                if all(c and c.HasAuthoredValueOpinion() for c in ccm):
                    attrs["nurec"].append(ccm)
            self._brightness_attrs = attrs
        return self._brightness_attrs

    def _apply_scene_brightness(self, ambient, nurec) -> None:
        """Retune the two knobs that make the mine dark.

        ``ambient`` drives the DomeLight intensity and ``nurec`` rescales the
        baked NuRec emission; the headlamp module sets both once at startup,
        which is why the scene reads as night.  Nothing else writes them, so a
        hand-set value sticks.
        """
        attrs = self._scene_light_attrs()
        if ambient is not None and attrs["dome"]:
            attrs["dome"].Set(float(ambient))
        if nurec is not None:
            for ccm in attrs["nurec"]:
                for channel, attribute in enumerate(ccm):
                    vector = [0.0, 0.0, 0.0, 0.0]
                    vector[channel] = float(nurec)
                    attribute.Set(Gf.Vec4f(*vector))

    def _rewrite_colormap(self, gain: float, opacity: float, *, transient: bool = False) -> None:
        """Re-apply the density ramp with a new gain and opacity."""
        if self._colormap_prim is None:
            return
        if not transient:
            self._colormap_gain = gain
            self._colormap_opacity = opacity
        self._colormap_prim.GetAttribute("rgbaPoints").Set(
            Vt.Vec4fArray(
                [
                    Gf.Vec4f(
                        rgb[0] * gain,
                        rgb[1] * gain,
                        rgb[2] * gain,
                        alpha * opacity,
                    )
                    for rgb, alpha in zip(
                        self._colormap_rgb_table, self._colormap_alpha_table
                    )
                ]
            )
        )

    def _apply_position_file(self) -> None:
        """Apply a hand-edited tuning line, live.

        Positional fields are ``x y z [radius]`` in world metres; optional
        ``gain=`` and ``opacity=`` fields retune the density ramp.  ``update()``
        never writes ``position``, so hand-set values survive.  Example::

            -2.3 -1.0 2.0 1.2 gain=5.5 opacity=0.8 ambient=200 nurec=0.3
        """
        try:
            text = self._position_file.read_text(encoding="utf-8")
        except OSError:
            return
        if text == self._last_position_text:
            return
        positional: list[float] = []
        named: dict[str, float] = {}
        for token in text.replace(",", " ").replace(";", " ").split():
            if "=" in token:
                key, _, value = token.partition("=")
                try:
                    named[key.strip().lower()] = float(value)
                except ValueError:
                    return
            else:
                try:
                    positional.append(float(token))
                except ValueError:
                    return
        if len(positional) not in (3, 4):
            return
        self._last_position_text = text

        moved = []
        for name in ("flowEmitterSphere", "flowEmitterSphereB"):
            prim = self.stage.GetPrimAtPath(f"{self.ROOT_PATH}/{name}")
            if not prim or not prim.IsValid():
                continue
            attr = prim.GetAttribute("position")
            if attr:
                attr.Set(Gf.Vec3f(positional[0], positional[1], positional[2]))
            if len(positional) == 4:
                radius_attr = prim.GetAttribute("radius")
                if radius_attr:
                    radius_attr.Set(positional[3])
            moved.append(name)

        ambient = named.get("ambient")
        nurec = named.get("nurec")
        if ambient is not None or nurec is not None:
            self._apply_scene_brightness(ambient, nurec)

        gain = named.get("gain")
        opacity = named.get("opacity")
        if gain is not None or opacity is not None:
            self._rewrite_colormap(
                gain if gain is not None else self._colormap_gain,
                opacity if opacity is not None else self._colormap_opacity,
            )

        if moved:
            print(
                f"[flow-dust-3d] tuning applied: pos={positional[:3]} "
                f"radius={positional[3] if len(positional) == 4 else 'kept'} "
                f"gain={self._colormap_gain} opacity={self._colormap_opacity}",
                flush=True,
            )

    def _init_burst(self) -> None:
        """Opt-in visual demo: native Flow injection plus an RTX envelope.

        This prescribes an expanding injection region, not a pressure wave or
        a calibrated dust model. RGB and alpha both need an envelope because
        the installed Flow ray marcher also emits light through colormap RGB.
        """
        self._burst_params = {
            "delay": _env_float("G1_DUST_BURST_DELAY", 4.0),
            "rise": _env_float("G1_DUST_BURST_RISE", 1.2),
            "hold": _env_float("G1_DUST_BURST_HOLD", 2.0),
            "fade": _env_float("G1_DUST_BURST_FADE", 6.0),
            "radius_start": _env_float("G1_DUST_BURST_RADIUS_START", 0.3),
            "radius_end": _env_float("G1_DUST_BURST_RADIUS_END", 2.4),
            "peak": _env_float("G1_DUST_BURST_PEAK", 1.0),
        }
        self._validate_burst(self._burst_params)
        request = os.environ.get("G1_DUST_BURST_FILE", "").strip()
        status = os.environ.get("G1_DUST_BURST_STATUS_FILE", "").strip()
        self._burst_file = Path(request) if request else None
        self._burst_status_file = Path(status) if status else None
        self._burst_last_text = ""
        self._burst_phase = ""
        self._burst_flow_density = float(self.settings.get("/rtx/raytracing/inscattering/flowDensityScale") or 0.0)
        self._burst_emitter = self.stage.GetPrimAtPath(f"{self.ROOT_PATH}/flowEmitterSphere")
        self._burst_clear_attr = self.stage.GetPrimAtPath(f"{self.ROOT_PATH}/flowSimulate").GetAttribute("forceClear")
        self.trigger_burst("startup")
        self._update_burst()
        for enabled in self._emitter_enabled:
            enabled.Set(False)

    @staticmethod
    def _validate_burst(params: dict) -> None:
        if not all(math.isfinite(v) for v in params.values()):
            raise ValueError("Burst parameters must be finite numbers")
        if not (0 <= params["delay"] <= 60 and 0 <= params["hold"] <= 60
                and 0.1 <= params["rise"] <= 60 and 0.1 <= params["fade"] <= 60):
            raise ValueError("Burst delay/hold must be 0..60s; rise/fade must be 0.1..60s")
        if not 0.06 <= params["radius_start"] <= params["radius_end"] <= 3.5:
            raise ValueError("Burst radii must satisfy 0.06 <= start <= end <= 3.5m")
        if not 0 <= params["peak"] <= 4:
            raise ValueError("Burst peak must be 0..4")

    def trigger_burst(self, trigger: str = "event") -> None:
        """Restart one burst; callers may use this setter for a future event."""
        if not self._burst_enabled:
            return
        self._burst_trigger = str(trigger)
        self._burst_started = time.monotonic()
        self._burst_clear_pending = True

    def _read_burst_request(self) -> None:
        if self._burst_file is None:
            return
        try:
            text = self._burst_file.read_text(encoding="utf-8")
            if text == self._burst_last_text:
                return
            self._burst_last_text = text
            request = json.loads(text)
            if not isinstance(request, dict) or set(request) - (set(self._burst_params) | {"trigger"}):
                raise ValueError("Unknown burst request fields")
            params = dict(self._burst_params)
            params.update({k: float(v) for k, v in request.items() if k != "trigger"})
            self._validate_burst(params)
            self._burst_params = params
            if "trigger" in request and str(request["trigger"]) != self._burst_trigger:
                self.trigger_burst(str(request["trigger"]))
        except OSError:
            return
        except (ValueError, TypeError) as error:
            print(f"[dust-burst] Ignoring invalid request: {error}", flush=True)

    @staticmethod
    def _burst_sample(t: float, p: dict) -> tuple[str, float, float, bool]:
        """Return phase, visibility, source radius and injection permission."""
        active_t = t - p["delay"]
        if active_t < 0:
            return "waiting", 0.0, p["radius_start"], False
        if active_t < p["rise"]:
            u = active_t / p["rise"]
            return "rise", u * u * (3 - 2 * u), p["radius_start"] + (p["radius_end"] - p["radius_start"]) * u, True
        active_t -= p["rise"]
        if active_t < p["hold"]:
            return "peak", 1.0, p["radius_end"], True
        active_t -= p["hold"]
        if active_t < p["fade"]:
            u = active_t / p["fade"]
            return "fade", 1 - u * u * (3 - 2 * u), p["radius_end"], False
        return "done", 0.0, p["radius_start"], False

    def _update_burst(self) -> None:
        self._read_burst_request()
        # Use wall time only for this visual demo. The legacy update clock is
        # clamped at 0.1s and stretches durations when rendering below 10 FPS.
        t = time.monotonic() - self._burst_started
        phase, envelope, radius, inject = self._burst_sample(t, self._burst_params)
        if self._burst_clear_pending:
            phase, envelope, radius, inject = "reset", 0.0, self._burst_params["radius_start"], False
            self._burst_clear_pending = False
            self._burst_started = time.monotonic()
            t = 0.0
        self._burst_inject = inject
        self._burst_clear_attr.Set(phase in {"reset", "waiting", "done"})
        self._burst_emitter.GetAttribute("radius").Set(radius)
        self._rewrite_colormap(
            self._colormap_gain * envelope,
            self._colormap_opacity * self._burst_params["peak"] * envelope,
            transient=True,
        )
        if _env_int("G1_DUST_FLOW_SAMPLING", 0) == 1:
            self.settings.set_float("/rtx/raytracing/inscattering/flowDensityScale", self._burst_flow_density * envelope)
        if phase != self._burst_phase:
            print(f"[dust-burst] {self._burst_trigger}: {phase} t={t:.3f}s", flush=True)
            self._burst_phase = phase
        if self._burst_status_file is not None:
            status = {"pid": os.getpid(), "trigger": self._burst_trigger, "phase": phase,
                      "elapsed": t, "envelope": envelope, "radius": radius,
                      "inject": inject, "wall_time": time.time(), "params": self._burst_params}
            temp = self._burst_status_file.with_suffix(".tmp")
            try:
                temp.write_text(json.dumps(status), encoding="utf-8")
                temp.replace(self._burst_status_file)
            except OSError as error:
                print(f"[dust-burst] Status output unavailable: {error}", flush=True)
                self._burst_status_file = None

    def update(self, dt: float) -> None:
        dt = float(max(0.0, min(float(dt), 0.1)))
        if self._position_file is not None:
            self._apply_position_file()
        if self._recording_marker is not None:
            recording = self._recording_marker.exists()
            if recording and not self._marker_seen:
                self._elapsed = 0.0
                self._marker_seen = True
            elif not recording:
                self._marker_seen = False
        self._elapsed += dt
        if self.distributed_field:
            # Stagger injection so no set of spheres expands in synchrony.
            # At least four sources remain active, while old density continues
            # to advect and overlap into an irregular suspended field.
            for index, enabled in enumerate(self._emitter_enabled):
                cycle = (self._elapsed + index * 0.71) % 5.4
                enabled.Set(cycle < 2.7 + 0.18 * (index % 3))
        elif self.box_field:
            # Old density keeps moving after injection stops, so staggered
            # sheets overlap into soft, irregular airborne banks.
            for index, enabled in enumerate(self._emitter_enabled):
                cycle = (self._elapsed + 1.17 * index) % 8.4
                enabled.Set(cycle < 3.45 + 0.18 * (index % 3))
        if self.single_1m_ramp or self.single_1m_tunnel_ramp:
            ramp_duration = 35.5 if self.single_1m_tunnel_ramp else 10.0
            ramp_progress = 0.0 if self._elapsed <= 5.0 else min(1.0, (self._elapsed - 5.0) / ramp_duration)
            opacity = self._ramp_start_opacity + ramp_progress * (self._ramp_end_opacity - self._ramp_start_opacity)
            ambient = self._ramp_start_ambient + ramp_progress * (self._ramp_end_ambient - self._ramp_start_ambient)
            rgba = [
                Gf.Vec4f(0.05, 0.047, 0.041, 0.0),
                Gf.Vec4f(0.18, 0.16, 0.13, 0.08 * opacity),
                Gf.Vec4f(0.42, 0.37, 0.30, 0.18 * opacity),
                Gf.Vec4f(3.00, 2.70, 2.20, 0.38 * opacity),
                Gf.Vec4f(12.0, 10.5, 8.50, 0.68 * opacity),
                Gf.Vec4f(32.0, 28.0, 22.0, opacity),
            ]
            self._colormap_prim.GetAttribute("rgbaPoints").Set(Vt.Vec4fArray(rgba))
            if self.single_1m_tunnel_ramp:
                global_haze = self._global_haze_start + ramp_progress * (self._global_haze_end - self._global_haze_start)
                self.settings.set_float("/rtx/raytracing/inscattering/densityMult", global_haze)
                self.settings.set_float("/rtx/raytracing/inscattering/transmittanceMeasurementDistance", 12.0 / max(global_haze, 1e-3))
                self.settings.set_float("/rtx/fog/fogDistanceDensity", 0.025 + ramp_progress * 0.015)
                self.settings.set_float("/rtx/fog/fogHeightDensity", 0.008 + ramp_progress * 0.004)
            else:
                scale = float(self.preset["ambient_scale"]) * ambient
                self.settings.set_float("/rtx/raytracing/inscattering/densityMult", 0.24 * scale)
                self.settings.set_float("/rtx/raytracing/inscattering/transmittanceMeasurementDistance", 12.0 / scale)
                self.settings.set_float("/rtx/fog/fogDistanceDensity", 0.06 * scale)
                self.settings.set_float("/rtx/fog/fogHeightDensity", 0.018 * scale)
        if self._burst_enabled:
            self._update_burst()
        for index, (enabled, velocity_attr, base_velocity) in enumerate(
            zip(
                self._emitter_enabled,
                self._emitter_velocity,
                self._base_velocities,
                strict=True,
            )
        ):
            if self._burst_enabled:
                enabled.Set(self._burst_inject)
            elif not self.distributed_field and not self.box_field and not self.texture_field:
                enabled.Set(True)
            phase = self._elapsed + self._velocity_phase + index * 1.9
            local_lateral = 0.018 * math.sin(0.65 * phase)
            wind_x, wind_y = self._rotate_xy(
                0.0, local_lateral, self._table_yaw
            )
            vertical = float(base_velocity[2]) + 0.010 * math.sin(
                0.37 * phase + 0.8
            )
            velocity_attr.Set(
                Gf.Vec3f(
                    float(base_velocity[0]) + wind_x,
                    float(base_velocity[1]) + wind_y,
                    vertical,
                )
            )

    def close(self) -> None:
        if self.stage:
            self.stage.RemovePrim(Sdf.Path(self.ROOT_PATH))
            self.stage.GetRootLayer().customLayerData = self._previous_layer_data
        for path, value in self._previous_settings.items():
            if value is not None:
                self.settings.set(path, value)


__all__ = ["FlowDust3D"]
