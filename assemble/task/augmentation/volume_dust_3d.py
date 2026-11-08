"""World-space RTX/MDL volume dust for the Task01 mining scene.

This opt-in demo avoids a visible point emitter.  It fills a large tunnel
volume with low-density animated Perlin/Worley fields so dust reads as
suspended particulate rather than a plume expanding from one location.

The implementation only authors USD geometry and MDL volume materials.  It
does not modify camera RGB frames, task state, physics, or source datasets.
"""

from __future__ import annotations

import math
import os
import random

import omni.usd
from pxr import Gf, Sdf, UsdGeom, UsdShade, Vt


def _float(name: str, default: float) -> float:
    value = float(os.environ.get(name, str(default)))
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def _int(name: str, default: int) -> int:
    return int(os.environ.get(name, str(default)))


def _vec3(name: str, default: str) -> Gf.Vec3f:
    values = tuple(float(item.strip()) for item in os.environ.get(name, default).split(","))
    if len(values) != 3 or not all(math.isfinite(item) for item in values):
        raise ValueError(f"{name} must be x,y,z")
    return Gf.Vec3f(*values)


class VolumeDust3D:
    """Animated heterogeneous volume occupying the tunnel instead of a source."""

    ROOT_PATH = "/World/G1MineDustVolume"

    _POINTS = Vt.Vec3fArray(
        [
            (-0.5, -0.5, -0.5),
            (0.5, -0.5, -0.5),
            (-0.5, -0.5, 0.5),
            (0.5, -0.5, 0.5),
            (-0.5, 0.5, -0.5),
            (0.5, 0.5, -0.5),
            (-0.5, 0.5, 0.5),
            (0.5, 0.5, 0.5),
        ]
    )
    _COUNTS = Vt.IntArray([4, 4, 4, 4, 4, 4])
    _INDICES = Vt.IntArray(
        [
            0, 1, 3, 2,
            4, 6, 7, 5,
            0, 4, 5, 1,
            2, 3, 7, 6,
            0, 2, 6, 4,
            1, 5, 7, 3,
        ]
    )

    def __init__(
        self,
        table_position: tuple[float, float, float],
        table_yaw: float = 0.0,
    ) -> None:
        del table_position, table_yaw
        self.stage = omni.usd.get_context().get_stage()
        if self.stage is None:
            raise RuntimeError("No active USD stage")
        self.seed = _int("G1_VOLUME_DUST_SEED", 260917)
        self.rng = random.Random(self.seed)
        self.variant = os.environ.get("G1_VOLUME_DUST_VARIANT", "layered").strip().lower()
        if self.variant not in {"perlin", "layered", "worley"}:
            raise ValueError("G1_VOLUME_DUST_VARIANT must be perlin, layered, or worley")
        self.center = _vec3("G1_VOLUME_DUST_CENTER", "-2.2,2.8,2.0")
        self.extent = _vec3("G1_VOLUME_DUST_EXTENT", "8.0,13.0,5.0")
        if min(self.extent) <= 0.1:
            raise ValueError("G1_VOLUME_DUST_EXTENT components must exceed 0.1 m")
        self.density = _float("G1_VOLUME_DUST_DENSITY", 0.075)
        self.feature_size = _float("G1_VOLUME_DUST_FEATURE_SIZE", 3.8)
        self.phase_speed = _float("G1_VOLUME_DUST_PHASE_SPEED", 0.045)
        self.drift_speed = _float("G1_VOLUME_DUST_DRIFT_SPEED", 0.018)
        self.animate_material = os.environ.get(
            "G1_VOLUME_DUST_ANIMATE_MATERIAL", "0"
        ).strip().lower() in {"1", "true", "yes", "on"}
        self.directional_bias = _float("G1_VOLUME_DUST_DIRECTIONAL_BIAS", 0.35)
        if not 0.0 <= self.density <= 5.0:
            raise ValueError("G1_VOLUME_DUST_DENSITY must be in [0, 5]")
        if not -0.95 <= self.directional_bias <= 0.95:
            raise ValueError("G1_VOLUME_DUST_DIRECTIONAL_BIAS must be in [-0.95, 0.95]")

        self._elapsed = 0.0
        self._last_material_update = -1.0
        self._closed = False
        self._layers: list[dict] = []
        root = UsdGeom.Xform.Define(self.stage, self.ROOT_PATH).GetPrim()
        root.CreateAttribute("g1:dust:implementation", Sdf.ValueTypeNames.String).Set(
            "world-space MDL heterogeneous volume"
        )
        root.CreateAttribute("g1:dust:variant", Sdf.ValueTypeNames.Token).Set(self.variant)

        if self.variant == "layered":
            specs = [
                # A broad, faint base field keeps the tunnel atmosphere continuous.
                ("Base", self.center, self.extent, 0.55, 1.00, 0.00),
                # Two offset fields break up the box-scale pattern and introduce depth.
                (
                    "MidA",
                    self.center + Gf.Vec3f(-1.4, 1.0, -0.25),
                    Gf.Vec3f(self.extent[0] * 0.72, self.extent[1] * 0.74, self.extent[2] * 0.72),
                    0.36,
                    0.62,
                    1.73,
                ),
                (
                    "MidB",
                    self.center + Gf.Vec3f(1.2, 2.2, 0.35),
                    Gf.Vec3f(self.extent[0] * 0.62, self.extent[1] * 0.58, self.extent[2] * 0.66),
                    0.25,
                    0.43,
                    3.11,
                ),
            ]
        else:
            specs = [("Main", self.center, self.extent, 1.0, 1.0, 0.0)]

        for name, center, extent, density_scale, size_scale, phase_offset in specs:
            self._layers.append(
                self._author_layer(
                    name=name,
                    center=center,
                    extent=extent,
                    density=self.density * density_scale,
                    feature_size=self.feature_size * size_scale,
                    phase_offset=phase_offset,
                )
            )

        print(
            f"[volume-dust-3d] authored {len(self._layers)} world-space volume layer(s) "
            f"(variant={self.variant}, density={self.density}, feature={self.feature_size}, "
            f"center={tuple(self.center)}, extent={tuple(self.extent)}, seed={self.seed})",
            flush=True,
        )

    def _author_layer(
        self,
        *,
        name: str,
        center: Gf.Vec3f,
        extent: Gf.Vec3f,
        density: float,
        feature_size: float,
        phase_offset: float,
    ) -> dict:
        layer_path = f"{self.ROOT_PATH}/{name}"
        mesh = UsdGeom.Mesh.Define(self.stage, f"{layer_path}/Volume")
        mesh.CreatePointsAttr(self._POINTS)
        mesh.CreateFaceVertexCountsAttr(self._COUNTS)
        mesh.CreateFaceVertexIndicesAttr(self._INDICES)
        mesh.CreateSubdivisionSchemeAttr("none")
        UsdGeom.PrimvarsAPI(mesh).CreatePrimvar(
            "isVolume", Sdf.ValueTypeNames.Bool, UsdGeom.Tokens.constant
        ).Set(True)
        xform = UsdGeom.Xformable(mesh.GetPrim())
        translate = xform.AddTranslateOp()
        translate.Set(center)
        xform.AddScaleOp().Set(extent)

        material = UsdShade.Material.Define(self.stage, f"{layer_path}/Material")
        shader = UsdShade.Shader.Define(self.stage, f"{layer_path}/Material/Shader")
        shader.CreateImplementationSourceAttr(UsdShade.Tokens.sourceAsset)
        shader.SetSourceAsset(Sdf.AssetPath("OmniVolumeNoise.mdl"), "mdl")
        subidentifier = (
            "OmniVolumeWorleyNoise" if self.variant == "worley" else "OmniVolumePerlinNoise"
        )
        shader.SetSourceAssetSubIdentifier(subidentifier, "mdl")
        shader.CreateOutput("out", Sdf.ValueTypeNames.Token)
        for terminal in ("surface", "volume", "displacement"):
            material.CreateOutput(f"mdl:{terminal}", Sdf.ValueTypeNames.Token).ConnectToSource(
                shader.ConnectableAPI(), "out"
            )

        albedo = _vec3("G1_VOLUME_DUST_ALBEDO", "0.54,0.46,0.36")
        shader.CreateInput("volume_albedo", Sdf.ValueTypeNames.Color3f).Set(albedo)
        shader.CreateInput("volume_density_scale", Sdf.ValueTypeNames.Float).Set(density)
        shader.CreateInput("directional_bias", Sdf.ValueTypeNames.Float).Set(self.directional_bias)
        shader.CreateInput("color1", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.0, 0.0, 0.0))
        shader.CreateInput("color2", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(1.0, 1.0, 1.0))
        shader.CreateInput("size", Sdf.ValueTypeNames.Float).Set(feature_size)
        shader.CreateInput("noise_distortion", Sdf.ValueTypeNames.Float3).Set(
            Gf.Vec3f(0.31, 0.18, 0.27)
        )
        shader.CreateInput("noise_threshold_low", Sdf.ValueTypeNames.Float).Set(
            _float("G1_VOLUME_DUST_THRESHOLD_LOW", 0.43)
        )
        shader.CreateInput("noise_threshold_high", Sdf.ValueTypeNames.Float).Set(
            _float("G1_VOLUME_DUST_THRESHOLD_HIGH", 0.88)
        )
        shader.CreateInput("noise_bands", Sdf.ValueTypeNames.Float).Set(2.2)
        if self.variant != "worley":
            shader.CreateInput("apply_marble", Sdf.ValueTypeNames.Bool).Set(False)
            shader.CreateInput("apply_dent", Sdf.ValueTypeNames.Bool).Set(False)
            phase = shader.CreateInput("noise_phase", Sdf.ValueTypeNames.Float)
            phase.Set(phase_offset)
            shader.CreateInput("noise_levels", Sdf.ValueTypeNames.Int).Set(4)
            shader.CreateInput("absolute_noise", Sdf.ValueTypeNames.Bool).Set(True)
            shader.CreateInput("ridged_noise", Sdf.ValueTypeNames.Bool).Set(False)
        else:
            phase = None
            shader.CreateInput("edge", Sdf.ValueTypeNames.Float).Set(0.88)
            shader.CreateInput("worley_mode", Sdf.ValueTypeNames.Int).Set(4)
            shader.CreateInput("metric", Sdf.ValueTypeNames.Int).Set(0)
            shader.CreateInput("step_threshold", Sdf.ValueTypeNames.Float).Set(0.0)

        UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)
        return {
            "translate": translate,
            "base_center": Gf.Vec3f(center),
            "phase": phase,
            "phase_offset": phase_offset,
        }

    def update(self, dt: float) -> None:
        dt = max(0.0, min(float(dt), 0.1))
        self._elapsed += dt
        # Slow sub-centimetre drift prevents a frozen texture without creating
        # an expanding source or a smoke-like upward plume.
        for index, layer in enumerate(self._layers):
            phase = self._elapsed * (0.15 + index * 0.037) + layer["phase_offset"]
            drift = Gf.Vec3f(
                self.drift_speed * math.sin(phase),
                self.drift_speed * self._elapsed * (0.32 + 0.07 * index),
                self.drift_speed * 0.35 * math.sin(phase * 0.63),
            )
            layer["translate"].Set(layer["base_center"] + drift)

        # Material recompilation is more expensive than a transform update, so
        # advance the 4D noise at 4 Hz rather than every frame.
        if self.animate_material and self._elapsed - self._last_material_update >= 0.25:
            self._last_material_update = self._elapsed
            for index, layer in enumerate(self._layers):
                if layer["phase"] is not None:
                    layer["phase"].Set(
                        layer["phase_offset"] + self._elapsed * self.phase_speed * (1.0 + 0.17 * index)
                    )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self.stage.GetPrimAtPath(self.ROOT_PATH):
            self.stage.RemovePrim(Sdf.Path(self.ROOT_PATH))


__all__ = ["VolumeDust3D"]
