#!/usr/bin/env python3
"""Build the portable standing rod-and-rope asset and its static holder."""

from __future__ import annotations

import math
import shutil
from pathlib import Path

from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade


ASSEMBLE_ROOT = Path(__file__).resolve().parents[3]
SAMPLES_ROOT = ASSEMBLE_ROOT / "assets-new" / "samples"
OBJECT_ROOT = ASSEMBLE_ROOT / "assets-new" / "object"

SOURCE_ASSEMBLY = SAMPLES_ROOT / "chaganshengzi001.usd"
STANDING_ASSEMBLY = OBJECT_ROOT / "chaganshengzi_standing.usd"
HOLDER_ASSET = OBJECT_ROOT / "chaganshengzi_holder.usd"

HOLDER_HEIGHT = 0.060
HOLDER_BOTTOM_THICKNESS = 0.008
HOLDER_INNER_RADIUS = 0.024
HOLDER_OUTER_RADIUS = 0.030
HOLDER_SEGMENTS = 16
HOLDER_COLOR = Gf.Vec3f(0.16, 0.18, 0.20)
HOLDER_STATIC_FRICTION = 0.10
HOLDER_DYNAMIC_FRICTION = 0.05


def _copy_dependencies() -> None:
    OBJECT_ROOT.mkdir(parents=True, exist_ok=True)
    for name in ("chagan.usd", "rope_40cm.usda"):
        source = SAMPLES_ROOT / name
        destination = OBJECT_ROOT / name
        if not source.is_file():
            raise FileNotFoundError(source)
        shutil.copy2(source, destination)


def _compact_layer(stage: Usd.Stage, destination: Path) -> None:
    temporary = destination.with_name(f".{destination.stem}.tmp{destination.suffix}")
    if temporary.exists():
        temporary.unlink()
    if not stage.GetRootLayer().Export(str(temporary)):
        raise RuntimeError(f"Failed to export {temporary}")
    temporary.replace(destination)


def build_standing_assembly() -> None:
    if not SOURCE_ASSEMBLY.is_file():
        raise FileNotFoundError(SOURCE_ASSEMBLY)
    shutil.copy2(SOURCE_ASSEMBLY, STANDING_ASSEMBLY)

    layer = Sdf.Layer.FindOrOpen(str(STANDING_ASSEMBLY))
    if layer is None:
        raise RuntimeError(f"Failed to open {STANDING_ASSEMBLY}")
    layer.subLayerPaths = [
        path for path in layer.subLayerPaths if not path.startswith("metrics:")
    ]
    custom_data = dict(layer.customLayerData)
    custom_data.pop("metricsAssembler", None)
    omni_layer = dict(custom_data.get("omni_layer", {}))
    omni_layer["authoring_layer"] = f"./{STANDING_ASSEMBLY.name}"
    custom_data["omni_layer"] = omni_layer
    layer.customLayerData = custom_data
    layer.Save()

    stage = Usd.Stage.Open(str(STANDING_ASSEMBLY), Usd.Stage.LoadAll)
    if stage is None:
        raise RuntimeError(f"Failed to compose {STANDING_ASSEMBLY}")
    stage.RemovePrim("/World/PhysicsScene")

    # The source rod uses an SDF mesh collider.  A convex hull is less prone to
    # catching on the holder rim and is sufficient for this nearly cylindrical
    # grasp object.
    rod_collision_prim = stage.GetPrimAtPath("/World/SmoothConnector/Mesh")
    if not rod_collision_prim.IsValid():
        raise RuntimeError("Missing rod collision mesh /World/SmoothConnector/Mesh")
    rod_mesh_collision = UsdPhysics.MeshCollisionAPI.Apply(rod_collision_prim)
    rod_mesh_collision.CreateApproximationAttr().Set("convexHull")

    stage.GetRootLayer().Save()
    _compact_layer(stage, STANDING_ASSEMBLY)


def _apply_static_collision(
    prim: Usd.Prim,
    physics_material: UsdShade.Material,
) -> None:
    collision = UsdPhysics.CollisionAPI.Apply(prim)
    collision.CreateCollisionEnabledAttr(True)
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(
        physics_material,
        materialPurpose="physics",
    )


def build_holder() -> None:
    temporary = HOLDER_ASSET.with_name(f".{HOLDER_ASSET.stem}.tmp{HOLDER_ASSET.suffix}")
    if temporary.exists():
        temporary.unlink()
    stage = Usd.Stage.CreateNew(str(temporary))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    root = UsdGeom.Xform.Define(stage, "/Holder")
    stage.SetDefaultPrim(root.GetPrim())
    # The holder must move during reset randomization. A kinematic rigid body
    # keeps the 17 child colliders in one PhysX actor, allowing its pose to be
    # updated immediately through the same tensor API used by the rod.
    rigid_body = UsdPhysics.RigidBodyAPI.Apply(root.GetPrim())
    rigid_body.CreateRigidBodyEnabledAttr(True)
    rigid_body.CreateKinematicEnabledAttr(True)

    material = UsdShade.Material.Define(stage, "/Holder/PhysicsMaterial")
    material_api = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    material_api.CreateStaticFrictionAttr(HOLDER_STATIC_FRICTION)
    material_api.CreateDynamicFrictionAttr(HOLDER_DYNAMIC_FRICTION)
    material_api.CreateRestitutionAttr(0.0)

    bottom = UsdGeom.Cylinder.Define(stage, "/Holder/Bottom")
    bottom.CreateAxisAttr(UsdGeom.Tokens.z)
    bottom.CreateRadiusAttr(HOLDER_OUTER_RADIUS)
    bottom.CreateHeightAttr(HOLDER_BOTTOM_THICKNESS)
    bottom.AddTranslateOp().Set(
        Gf.Vec3d(0.0, 0.0, 0.5 * HOLDER_BOTTOM_THICKNESS)
    )
    bottom.CreateDisplayColorAttr([HOLDER_COLOR])
    _apply_static_collision(bottom.GetPrim(), material)

    mean_radius = 0.5 * (HOLDER_INNER_RADIUS + HOLDER_OUTER_RADIUS)
    radial_thickness = HOLDER_OUTER_RADIUS - HOLDER_INNER_RADIUS
    tangential_length = (
        2.0
        * HOLDER_OUTER_RADIUS
        * math.tan(math.pi / HOLDER_SEGMENTS)
        * 1.03
    )
    for index in range(HOLDER_SEGMENTS):
        angle = 2.0 * math.pi * index / HOLDER_SEGMENTS
        segment = UsdGeom.Cube.Define(stage, f"/Holder/Wall_{index:02d}")
        segment.CreateSizeAttr(2.0)
        segment.AddTranslateOp().Set(
            Gf.Vec3d(
                mean_radius * math.cos(angle),
                mean_radius * math.sin(angle),
                0.5 * HOLDER_HEIGHT,
            )
        )
        segment.AddOrientOp().Set(
            Gf.Quatf(
                math.cos(0.5 * angle),
                Gf.Vec3f(0.0, 0.0, math.sin(0.5 * angle)),
            )
        )
        segment.AddScaleOp().Set(
            Gf.Vec3f(
                0.5 * radial_thickness,
                0.5 * tangential_length,
                0.5 * HOLDER_HEIGHT,
            )
        )
        segment.CreateDisplayColorAttr([HOLDER_COLOR])
        _apply_static_collision(segment.GetPrim(), material)

    stage.GetRootLayer().Save()
    temporary.replace(HOLDER_ASSET)


def main() -> None:
    _copy_dependencies()
    build_standing_assembly()
    build_holder()
    print(f"Built standing assembly: {STANDING_ASSEMBLY}")
    print(f"Built static holder: {HOLDER_ASSET}")


if __name__ == "__main__":
    main()
