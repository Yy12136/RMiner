"""Static, hand-placed debris for the mining-condition figure.

The fourth figure is a single still image. Debris therefore stays completely
static: each fragment is an ordinary USD Xform with an editable translate,
orientation, and scale. There is deliberately no PhysX body, velocity,
gravity, timer, or image-space compositing in this module. The same layout can
also be authored as a standalone ``falling_debris_static.usda`` file and then
referenced into the tunnel scene.
"""

from __future__ import annotations

import math
import os
import random
from pathlib import Path

from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade, Vt


def _float(name: str, default: float) -> float:
    value = float(os.environ.get(name, str(default)))
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def _int(name: str, default: int) -> int:
    return int(os.environ.get(name, str(default)))


def _bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _vec3(name: str, default: str) -> Gf.Vec3f:
    raw = os.environ.get(name, default)
    values = tuple(float(v.strip()) for v in raw.split(","))
    if len(values) != 3 or not all(math.isfinite(v) for v in values):
        raise ValueError(f"{name} must be x,y,z")
    return Gf.Vec3f(*values)


class FallingDebris3D:
    """Author editable, static rock assets under one clearly named root."""

    ROOT_PATH = "/World/G1FallingDebris"

    def __init__(self, stage, *, seed: int = 260822, prepare_only: bool = False) -> None:
        del prepare_only  # compatibility with the previous replay hook
        self.stage = stage
        self.seed = _int("G1_DEBRIS_SEED", seed)
        self.rng = random.Random(self.seed)
        self._closed = False
        self._rocks: list[dict] = []

        count = _int("G1_DEBRIS_COUNT", 10)
        if not 1 <= count <= 64:
            raise ValueError("G1_DEBRIS_COUNT must be in [1, 64]")
        self.count = count
        self.center = _vec3("G1_DEBRIS_CENTER", "-1.15,0.65,3.35")
        spread = _float("G1_DEBRIS_SPREAD_RADIUS", 0.95)
        depth = _float("G1_DEBRIS_DEPTH", 0.18)
        height = _float("G1_DEBRIS_HEIGHT", 0.95)
        self.spread = Gf.Vec3f(spread, depth, height)
        self.size_min = _float("G1_DEBRIS_SIZE_MIN", 0.09)
        self.size_max = _float("G1_DEBRIS_SIZE_MAX", 0.30)
        if not (0.02 <= self.size_min <= self.size_max <= 1.0):
            raise ValueError("G1_DEBRIS_SIZE_MIN/MAX must satisfy 0.02 <= min <= max <= 1")

        root = UsdGeom.Xform.Define(stage, self.ROOT_PATH).GetPrim()
        root.SetActive(True)
        root.CreateAttribute("g1:debris:mode", Sdf.ValueTypeNames.Token).Set("static")
        root.CreateAttribute("g1:debris:editable", Sdf.ValueTypeNames.Bool).Set(True)
        root.CreateAttribute(
            "g1:debris:instructions", Sdf.ValueTypeNames.String
        ).Set("Drag Rock_XX Xforms in the USD stage; no simulation restart is needed.")
        self._root = root

        existing_rocks = [
            prim
            for prim in Usd.PrimRange(root)
            if prim.GetName().startswith("Rock_")
        ]
        if existing_rocks:
            # Replay hooks may construct the augmentation once before reset and
            # the task controller may attach it a second time. Reuse the
            # already-authored static layout instead of stacking xform ops.
            self.count = len(existing_rocks)
            print(
                f"[falling-debris] reused existing static USD layout ({self.count} rocks)",
                flush=True,
            )
            return

        layout_raw = os.environ.get("G1_DEBRIS_STATIC_USD", "").strip()
        if layout_raw:
            layout_path = Path(layout_raw).expanduser()
            if layout_path.is_file():
                root.GetReferences().AddReference(
                    str(layout_path), Sdf.Path(self.ROOT_PATH)
                )
                root.GetAttribute("g1:debris:mode").Set("static-reference")
                referenced_count = sum(
                    1
                    for prim in Usd.PrimRange(root)
                    if prim.GetName().startswith("Rock_")
                )
                if referenced_count:
                    self.count = referenced_count
                print(
                    f"[falling-debris] loaded editable static USD layout: {layout_path} "
                    f"({self.count} rocks)",
                    flush=True,
                )
                return
            print(
                f"[falling-debris] static layout not found, generating default layout: "
                f"{layout_path}",
                flush=True,
            )

        self._material = self._make_stone_material()
        for index in range(count):
            position = Gf.Vec3f(
                self.center[0] + self.rng.uniform(-self.spread[0], self.spread[0]),
                self.center[1] + self.rng.uniform(-self.spread[1], self.spread[1]),
                self.center[2] + self.rng.uniform(-self.spread[2], self.spread[2]),
            )
            size = self.rng.uniform(self.size_min, self.size_max)
            if index < 2:
                size *= 1.30
            scale = Gf.Vec3f(
                size * self.rng.uniform(0.65, 1.35),
                size * self.rng.uniform(0.65, 1.20),
                size * self.rng.uniform(0.60, 1.45),
            )
            axis = Gf.Vec3d(
                self.rng.uniform(-1.0, 1.0),
                self.rng.uniform(-1.0, 1.0),
                self.rng.uniform(-1.0, 1.0),
            )
            if axis.GetLength() < 1.0e-6:
                axis = Gf.Vec3d(0.0, 0.0, 1.0)
            orientation = Gf.Quatf(
                Gf.Rotation(axis.GetNormalized(), self.rng.uniform(0.0, 360.0)).GetQuat()
            )
            self._rocks.append(self._author_rock(index, position, scale, orientation))

        print(
            f"[falling-debris] authored {self.count} static editable USD rocks "
            f"(seed={self.seed}, root={self.ROOT_PATH})",
            flush=True,
        )

    def _make_stone_material(self) -> UsdShade.Material:
        material = UsdShade.Material.Define(self.stage, f"{self.ROOT_PATH}/RockMaterial")
        shader = UsdShade.Shader.Define(
            self.stage, f"{self.ROOT_PATH}/RockMaterial/PreviewSurface"
        )
        shader.CreateIdAttr().Set("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
            Gf.Vec3f(0.030, 0.034, 0.038)
        )
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.90)
        shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
        shader.CreateInput("specularColor", Sdf.ValueTypeNames.Color3f).Set(
            Gf.Vec3f(0.12, 0.12, 0.12)
        )
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        return material

    def _author_rock(self, index: int, position, scale, orientation) -> dict:
        path = f"{self.ROOT_PATH}/Rock_{index:02d}"
        root = UsdGeom.Xform.Define(self.stage, path).GetPrim()
        xform = UsdGeom.Xformable(root)
        translate_op = xform.AddTranslateOp()
        translate_op.Set(position)
        xform.AddOrientOp().Set(orientation)
        root.CreateAttribute("g1:debris:editable", Sdf.ValueTypeNames.Bool).Set(True)
        root.CreateAttribute("g1:debris:index", Sdf.ValueTypeNames.Int).Set(index)

        visual = UsdGeom.Xform.Define(self.stage, f"{path}/Visual").GetPrim()
        self._reference_asset(visual, scale, index)
        if not self._has_referenced_mesh(visual):
            self._author_fallback_mesh(path, scale)
        else:
            for prim in Usd.PrimRange(visual):
                if prim.IsA(UsdGeom.Mesh):
                    UsdShade.MaterialBindingAPI(prim).Bind(self._material)
        return {
            "path": path,
            "position": position,
            "scale": scale,
            "orientation": orientation,
        }

    def _reference_asset(self, visual, scale: Gf.Vec3f, index: int) -> None:
        if not _bool("G1_DEBRIS_USE_ASSETS", True):
            return
        asset_root = Path(
            os.environ.get(
                "G1_DEBRIS_ASSET_ROOT",
                os.environ.get("BENCH_ASSEMBLE_ROOT", str(Path(__file__).resolve().parents[2])) + "/assets-new/object/polyhaven_rocks_2k",
            )
        )
        asset_names = ("rock_07", "rock_09", "stone_01")
        asset_name = asset_names[index % len(asset_names)]
        asset_path = asset_root / asset_name / f"{asset_name}_2k.usdc"
        if not asset_path.is_file():
            print(
                f"[falling-debris] asset missing, using procedural stone: {asset_path}",
                flush=True,
            )
            return
        default_path = None
        try:
            asset_stage = Usd.Stage.Open(str(asset_path))
            default_prim = asset_stage.GetDefaultPrim()
            if default_prim and default_prim.IsValid():
                default_path = default_prim.GetPath()
        except Exception as error:
            print(
                f"[falling-debris] asset metadata unavailable for {asset_name}: {error}",
                flush=True,
            )
        if default_path is None:
            visual.GetReferences().AddReference(str(asset_path))
        else:
            visual.GetReferences().AddReference(str(asset_path), default_path)
        visual_xform = UsdGeom.Xformable(visual)
        visual_scale = max(float(component) for component in scale) * _float(
            "G1_DEBRIS_ASSET_SCALE_MULT", 5.0
        )
        visual_xform.AddScaleOp().Set(Gf.Vec3f(visual_scale, visual_scale, visual_scale))

    @staticmethod
    def _has_referenced_mesh(visual) -> bool:
        return any(prim.IsA(UsdGeom.Mesh) for prim in Usd.PrimRange(visual))

    def _author_fallback_mesh(self, path: str, scale: Gf.Vec3f) -> None:
        """Use an irregular faceted stone only if the PolyHaven asset is absent."""
        phi = (1.0 + math.sqrt(5.0)) * 0.5
        base = [
            (-1, phi, 0), (1, phi, 0), (-1, -phi, 0), (1, -phi, 0),
            (0, -1, phi), (0, 1, phi), (0, -1, -phi), (0, 1, -phi),
            (phi, 0, -1), (phi, 0, 1), (-phi, 0, -1), (-phi, 0, 1),
        ]
        faces = [
            (0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11),
            (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6), (7, 1, 8),
            (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9),
            (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1),
        ]
        unit = [Gf.Vec3d(*v).GetNormalized() for v in base]
        midpoint_cache = {}

        def midpoint(left: int, right: int) -> int:
            key = tuple(sorted((left, right)))
            if key not in midpoint_cache:
                midpoint_cache[key] = len(unit)
                unit.append((unit[left] + unit[right]).GetNormalized())
            return midpoint_cache[key]

        triangles = []
        for left, middle, right in faces:
            lm, mr, rl = midpoint(left, middle), midpoint(middle, right), midpoint(right, left)
            triangles.extend(((left, lm, rl), (middle, mr, lm), (right, rl, mr), (lm, mr, rl)))
        points = []
        for vertex in unit:
            jitter = self.rng.uniform(0.88, 1.12)
            points.append(
                Gf.Vec3f(
                    float(vertex[0]) * float(scale[0]) * jitter,
                    float(vertex[1]) * float(scale[1]) * jitter,
                    float(vertex[2]) * float(scale[2]) * jitter,
                )
            )
        mesh = UsdGeom.Mesh.Define(self.stage, f"{path}/FallbackStone")
        mesh.CreatePointsAttr().Set(Vt.Vec3fArray(points))
        mesh.CreateFaceVertexCountsAttr().Set(Vt.IntArray([3] * len(triangles)))
        mesh.CreateFaceVertexIndicesAttr().Set(Vt.IntArray([i for face in triangles for i in face]))
        mesh.CreateSubdivisionSchemeAttr().Set("none")
        tone = self.rng.uniform(0.72, 1.18)
        color = Gf.Vec3f(0.030 * tone, 0.034 * tone, 0.038 * tone)
        mesh.CreateDisplayColorPrimvar(UsdGeom.Tokens.constant).Set([color])
        UsdShade.MaterialBindingAPI(mesh.GetPrim()).Bind(self._material)

    def update(self, dt: float) -> None:
        # Manual USD edits must never be overwritten by a timer or a per-frame
        # position/velocity update.
        del dt

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self.stage.GetPrimAtPath(self.ROOT_PATH):
            self.stage.RemovePrim(Sdf.Path(self.ROOT_PATH))


__all__ = ["FallingDebris3D"]

# Task01 triggered debris burst opt-in (20260919)
if os.environ.get("G1_DEBRIS_BURST", "0") == "1":
    try:
        from .task01_debris_burst_trigger import TriggeredFallingDebris3D
    except ImportError:
        import importlib.util
        _burst_path = Path(__file__).with_name("task01_debris_burst_trigger.py")
        _burst_spec = importlib.util.spec_from_file_location("task01_debris_burst_trigger", _burst_path)
        _burst_module = importlib.util.module_from_spec(_burst_spec)
        _burst_spec.loader.exec_module(_burst_module)
        TriggeredFallingDebris3D = _burst_module.TriggeredFallingDebris3D
    FallingDebris3D = TriggeredFallingDebris3D
