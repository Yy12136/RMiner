#!/usr/bin/env python3
"""Build self-contained, task-scale valve and handwheel USD assets."""

from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade


ASSEMBLE_ROOT = Path(__file__).resolve().parents[3]
SOURCE_ROOT = ASSEMBLE_ROOT / "objects"
OUTPUT_ROOT = ASSEMBLE_ROOT / "assets-new" / "object" / "valve_handwheel_task"

HANDWHEEL_SOURCE = SOURCE_ROOT / "blue_cast_metal_valve_handwheel.usd"
VALVE_SOURCE = SOURCE_ROOT / "dark_gray_metal_three_way_ball_valve.usd"
HANDWHEEL_OUTPUT = OUTPUT_ROOT / "blue_valve_handwheel_task.usd"
VALVE_OUTPUT = OUTPUT_ROOT / "three_way_ball_valve_task.usd"

SOURCE_MESH_PATH = Sdf.Path("/world/geometry_0/geometry_0")
SOURCE_LOOKS_PATH = Sdf.Path("/world/Looks")

TASK_SIZE_SCALE = 1.8

HANDWHEEL_SCALE = 0.22 * TASK_SIZE_SCALE
HANDWHEEL_MASS = 0.5
HANDWHEEL_RIM_INNER_RADIUS = 0.087 * TASK_SIZE_SCALE
HANDWHEEL_RIM_OUTER_RADIUS = 0.109 * TASK_SIZE_SCALE
HANDWHEEL_HUB_OUTER_RADIUS = 0.033 * TASK_SIZE_SCALE
# Give the already-installed wheel generous clearance around the 16 mm stem.
# The visual bore is remapped to this same radius below, so collision and
# appearance continue to agree.
HANDWHEEL_HUB_INNER_RADIUS = 0.013 * TASK_SIZE_SCALE
HANDWHEEL_RIM_Z_MIN = 0.016 * TASK_SIZE_SCALE
HANDWHEEL_RIM_Z_MAX = 0.058 * TASK_SIZE_SCALE
HANDWHEEL_HUB_Z_MIN = 0.0
HANDWHEEL_HUB_Z_MAX = 0.050 * TASK_SIZE_SCALE
HANDWHEEL_SPOKE_Z_MIN = 0.015 * TASK_SIZE_SCALE
HANDWHEEL_SPOKE_Z_MAX = 0.052 * TASK_SIZE_SCALE
HANDWHEEL_RING_SEGMENTS = 32
HANDWHEEL_HUB_SEGMENTS = 24

VALVE_SCALE = 0.20 * TASK_SIZE_SCALE
VALVE_STEM_RADIUS = 0.008 * TASK_SIZE_SCALE
VALVE_STEM_SOURCE_BLEND_START = 0.300
VALVE_STEM_SOURCE_BLEND_END = 0.320

OUTER_CONTACT_OFFSET = 0.001 * TASK_SIZE_SCALE
INSERTION_CONTACT_OFFSET = 0.0005 * TASK_SIZE_SCALE


def _source_mesh(stage: Usd.Stage) -> UsdGeom.Mesh:
    prim = stage.GetPrimAtPath(SOURCE_MESH_PATH)
    if not prim or not prim.IsA(UsdGeom.Mesh):
        raise RuntimeError(f"Missing source mesh {SOURCE_MESH_PATH}")
    return UsdGeom.Mesh(prim)


def _copy_primvar(
    source_mesh: UsdGeom.Mesh,
    target_mesh: UsdGeom.Mesh,
    name: str,
) -> None:
    source = UsdGeom.PrimvarsAPI(source_mesh).GetPrimvar(name)
    if not source or not source.HasValue():
        return
    target = UsdGeom.PrimvarsAPI(target_mesh).CreatePrimvar(
        name,
        source.GetTypeName(),
        source.GetInterpolation(),
        source.GetElementSize(),
    )
    target.Set(source.Get())
    if source.IsIndexed():
        target.SetIndices(source.GetIndices())


def _copy_visual_mesh(
    source_stage: Usd.Stage,
    target_stage: Usd.Stage,
    root_path: Sdf.Path,
    points: np.ndarray,
) -> UsdGeom.Mesh:
    source_mesh = _source_mesh(source_stage)
    target = UsdGeom.Mesh.Define(target_stage, root_path.AppendChild("Visual"))
    target.CreatePointsAttr(
        [Gf.Vec3f(*(float(value) for value in point)) for point in points]
    )
    target.CreateFaceVertexCountsAttr(
        source_mesh.GetFaceVertexCountsAttr().Get()
    )
    target.CreateFaceVertexIndicesAttr(
        source_mesh.GetFaceVertexIndicesAttr().Get()
    )
    target.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    target.CreateDoubleSidedAttr(source_mesh.GetDoubleSidedAttr().Get() or False)
    extent_min = points.min(axis=0)
    extent_max = points.max(axis=0)
    target.CreateExtentAttr(
        [
            Gf.Vec3f(*(float(value) for value in extent_min)),
            Gf.Vec3f(*(float(value) for value in extent_max)),
        ]
    )
    normals = source_mesh.GetNormalsAttr().Get()
    if normals:
        # Source normals are Y-up. Rotate them with the same +90 degree X
        # transform used for the points.
        normals_array = np.asarray(normals, dtype=np.float32)
        transformed = np.column_stack(
            (normals_array[:, 0], -normals_array[:, 2], normals_array[:, 1])
        )
        target.CreateNormalsAttr(
            [
                Gf.Vec3f(*(float(value) for value in normal))
                for normal in transformed
            ]
        )
        target.SetNormalsInterpolation(source_mesh.GetNormalsInterpolation())
    for name in ("displayColor", "displayOpacity"):
        _copy_primvar(source_mesh, target, name)

    looks_path = root_path.AppendChild("Looks")
    if not Sdf.CopySpec(
        source_stage.GetRootLayer(),
        SOURCE_LOOKS_PATH,
        target_stage.GetRootLayer(),
        looks_path,
    ):
        raise RuntimeError("Failed to copy embedded source materials")
    material = UsdShade.Material.Get(
        target_stage, looks_path.AppendChild("VertexColorMaterial")
    )
    if not material:
        raise RuntimeError("Copied VertexColorMaterial is missing")
    UsdShade.MaterialBindingAPI.Apply(target.GetPrim()).Bind(material)
    return target


def _make_physics_material(
    stage: Usd.Stage,
    path: Sdf.Path,
    static_friction: float,
    dynamic_friction: float,
) -> UsdShade.Material:
    material = UsdShade.Material.Define(stage, path)
    api = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    api.CreateStaticFrictionAttr(static_friction)
    api.CreateDynamicFrictionAttr(dynamic_friction)
    api.CreateRestitutionAttr(0.0)
    return material


def _bind_collision(
    prim: Usd.Prim,
    material: UsdShade.Material,
    contact_offset: float,
) -> None:
    UsdPhysics.CollisionAPI.Apply(prim).CreateCollisionEnabledAttr(True)
    prim.CreateAttribute(
        "physxCollision:contactOffset", Sdf.ValueTypeNames.Float
    ).Set(contact_offset)
    prim.CreateAttribute(
        "physxCollision:restOffset", Sdf.ValueTypeNames.Float
    ).Set(0.0)
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(
        material,
        materialPurpose="physics",
    )


def _make_convex_prism(
    stage: Usd.Stage,
    path: Sdf.Path,
    xy: list[tuple[float, float]],
    z_min: float,
    z_max: float,
    material: UsdShade.Material,
    contact_offset: float,
) -> None:
    count = len(xy)
    points = [Gf.Vec3f(x, y, z_min) for x, y in xy]
    points.extend(Gf.Vec3f(x, y, z_max) for x, y in xy)
    face_counts: list[int] = [count, count]
    indices: list[int] = list(reversed(range(count))) + list(
        range(count, 2 * count)
    )
    for index in range(count):
        next_index = (index + 1) % count
        face_counts.append(4)
        indices.extend(
            (index, next_index, count + next_index, count + index)
        )
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr(points)
    mesh.CreateFaceVertexCountsAttr(face_counts)
    mesh.CreateFaceVertexIndicesAttr(indices)
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    mesh.CreatePurposeAttr(UsdGeom.Tokens.guide)
    mesh.CreateVisibilityAttr(UsdGeom.Tokens.invisible)
    mesh.CreateExtentAttr(
        [
            Gf.Vec3f(
                min(point[0] for point in xy),
                min(point[1] for point in xy),
                z_min,
            ),
            Gf.Vec3f(
                max(point[0] for point in xy),
                max(point[1] for point in xy),
                z_max,
            ),
        ]
    )
    _bind_collision(mesh.GetPrim(), material, contact_offset)
    UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr(
        "convexHull"
    )


def _annulus_wedges(
    stage: Usd.Stage,
    parent: Sdf.Path,
    inner_radius: float,
    outer_radius: float,
    z_min: float,
    z_max: float,
    segments: int,
    material: UsdShade.Material,
    contact_offset: float,
) -> None:
    for index in range(segments):
        angle0 = 2.0 * math.pi * index / segments
        angle1 = 2.0 * math.pi * (index + 1) / segments
        xy = [
            (inner_radius * math.cos(angle0), inner_radius * math.sin(angle0)),
            (outer_radius * math.cos(angle0), outer_radius * math.sin(angle0)),
            (outer_radius * math.cos(angle1), outer_radius * math.sin(angle1)),
            (inner_radius * math.cos(angle1), inner_radius * math.sin(angle1)),
        ]
        _make_convex_prism(
            stage,
            parent.AppendChild(f"Section_{index:02d}"),
            xy,
            z_min,
            z_max,
            material,
            contact_offset,
        )


def _make_spokes(
    stage: Usd.Stage,
    parent: Sdf.Path,
    material: UsdShade.Material,
) -> None:
    inner_radius = 0.027 * TASK_SIZE_SCALE
    outer_radius = 0.091 * TASK_SIZE_SCALE
    inner_half_angle = math.radians(24.0)
    outer_half_angle = math.radians(13.0)
    for index, center in enumerate(
        (math.radians(90.0), math.radians(210.0), math.radians(330.0))
    ):
        xy = [
            (
                inner_radius * math.cos(center - inner_half_angle),
                inner_radius * math.sin(center - inner_half_angle),
            ),
            (
                outer_radius * math.cos(center - outer_half_angle),
                outer_radius * math.sin(center - outer_half_angle),
            ),
            (
                outer_radius * math.cos(center + outer_half_angle),
                outer_radius * math.sin(center + outer_half_angle),
            ),
            (
                inner_radius * math.cos(center + inner_half_angle),
                inner_radius * math.sin(center + inner_half_angle),
            ),
        ]
        _make_convex_prism(
            stage,
            parent.AppendChild(f"Spoke_{index:02d}"),
            xy,
            HANDWHEEL_SPOKE_Z_MIN,
            HANDWHEEL_SPOKE_Z_MAX,
            material,
            OUTER_CONTACT_OFFSET,
        )


def _prepare_stage(path: Path, root_name: str) -> tuple[Usd.Stage, Usd.Prim]:
    path.parent.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, f"/{root_name}").GetPrim()
    stage.SetDefaultPrim(root)
    root.SetMetadata("kind", "component")
    return stage, root


def _atomic_save(stage: Usd.Stage, temporary: Path, destination: Path) -> None:
    stage.GetRootLayer().Save()
    os.replace(temporary, destination)


def _build_handwheel() -> None:
    source_stage = Usd.Stage.Open(str(HANDWHEEL_SOURCE))
    if source_stage is None:
        raise RuntimeError(f"Failed to open {HANDWHEEL_SOURCE}")
    source_points = np.asarray(
        _source_mesh(source_stage).GetPointsAttr().Get(), dtype=np.float64
    )
    points = np.column_stack(
        (source_points[:, 0], -source_points[:, 2], source_points[:, 1])
    )
    points *= HANDWHEEL_SCALE
    points[:, 2] -= points[:, 2].min()
    radial = np.linalg.norm(points[:, :2], axis=1)
    source_bore_radius = float(radial.min())
    # Enlarge only the central hub region.  The outer wheel and the visible
    # silhouette remain untouched while the rendered bore matches the 26 mm
    # collision bore used by the task.
    hub_mask = radial < HANDWHEEL_HUB_OUTER_RADIUS
    hub_radial = radial[hub_mask]
    denominator = HANDWHEEL_HUB_OUTER_RADIUS - source_bore_radius
    if denominator <= 0.0:
        raise RuntimeError("Source handwheel bore is outside the hub radius")
    remapped_radial = HANDWHEEL_HUB_INNER_RADIUS + (
        (hub_radial - source_bore_radius)
        * (HANDWHEEL_HUB_OUTER_RADIUS - HANDWHEEL_HUB_INNER_RADIUS)
        / denominator
    )
    nonzero = hub_radial > 1.0e-9
    hub_points = points[hub_mask]
    hub_points[nonzero, :2] *= (
        remapped_radial[nonzero] / hub_radial[nonzero]
    )[:, None]
    points[hub_mask] = hub_points
    points = points.astype(np.float32)
    radial = np.linalg.norm(points[:, :2], axis=1)
    bore_radius = float(radial.min())

    temporary = HANDWHEEL_OUTPUT.with_name(
        f".{HANDWHEEL_OUTPUT.stem}.{os.getpid()}.tmp{HANDWHEEL_OUTPUT.suffix}"
    )
    temporary.unlink(missing_ok=True)
    stage, root = _prepare_stage(temporary, "ValveHandwheel")
    root_path = root.GetPath()
    _copy_visual_mesh(source_stage, stage, root_path, points)

    rigid = UsdPhysics.RigidBodyAPI.Apply(root)
    rigid.CreateRigidBodyEnabledAttr(True)
    rigid.CreateKinematicEnabledAttr(False)
    mass = UsdPhysics.MassAPI.Apply(root)
    mass.CreateMassAttr(HANDWHEEL_MASS)
    mass.CreateCenterOfMassAttr(
        Gf.Vec3f(0.0, 0.0, 0.5 * float(points[:, 2].max()))
    )
    for name, value, value_type in (
        ("physxRigidBody:disableGravity", False, Sdf.ValueTypeNames.Bool),
        ("physxRigidBody:enableSpeculativeCCD", True, Sdf.ValueTypeNames.Bool),
        (
            "physxRigidBody:solverPositionIterationCount",
            16,
            Sdf.ValueTypeNames.Int,
        ),
        (
            "physxRigidBody:solverVelocityIterationCount",
            4,
            Sdf.ValueTypeNames.Int,
        ),
        (
            "physxRigidBody:maxDepenetrationVelocity",
            1.0,
            Sdf.ValueTypeNames.Float,
        ),
        ("physxRigidBody:linearDamping", 0.03, Sdf.ValueTypeNames.Float),
        ("physxRigidBody:angularDamping", 0.03, Sdf.ValueTypeNames.Float),
    ):
        root.CreateAttribute(name, value_type).Set(value)

    physics_root = root_path.AppendChild("PhysicsMaterials")
    UsdGeom.Scope.Define(stage, physics_root)
    grasp_material = _make_physics_material(
        stage, physics_root.AppendChild("Grasp"), 1.2, 1.0
    )
    insertion_material = _make_physics_material(
        stage, physics_root.AppendChild("Insertion"), 0.20, 0.15
    )
    collision_root = root_path.AppendChild("Collision")
    UsdGeom.Xform.Define(stage, collision_root).CreateVisibilityAttr(
        UsdGeom.Tokens.invisible
    )
    _annulus_wedges(
        stage,
        collision_root.AppendChild("Rim"),
        HANDWHEEL_RIM_INNER_RADIUS,
        HANDWHEEL_RIM_OUTER_RADIUS,
        HANDWHEEL_RIM_Z_MIN,
        HANDWHEEL_RIM_Z_MAX,
        HANDWHEEL_RING_SEGMENTS,
        grasp_material,
        OUTER_CONTACT_OFFSET,
    )
    _annulus_wedges(
        stage,
        collision_root.AppendChild("Hub"),
        HANDWHEEL_HUB_INNER_RADIUS,
        HANDWHEEL_HUB_OUTER_RADIUS,
        HANDWHEEL_HUB_Z_MIN,
        HANDWHEEL_HUB_Z_MAX,
        HANDWHEEL_HUB_SEGMENTS,
        insertion_material,
        INSERTION_CONTACT_OFFSET,
    )
    _make_spokes(stage, collision_root.AppendChild("Spokes"), grasp_material)

    metadata = {
        "task:outerDiameter": 2.0 * float(radial.max()),
        "task:boreDiameter": 2.0 * bore_radius,
        "task:collisionBoreDiameter": 2.0 * HANDWHEEL_HUB_INNER_RADIUS,
        "task:hubCollisionHeight": HANDWHEEL_HUB_Z_MAX - HANDWHEEL_HUB_Z_MIN,
        "task:thickness": float(points[:, 2].max()),
        "task:mass": HANDWHEEL_MASS,
        "task:axis": "Z",
        "task:sizeScale": TASK_SIZE_SCALE,
        "task:collisionStyle": "compound convex open-bore wheel",
    }
    for name, value in metadata.items():
        value_type = (
            Sdf.ValueTypeNames.String
            if isinstance(value, str)
            else Sdf.ValueTypeNames.Double
        )
        root.CreateAttribute(name, value_type).Set(value)
    _atomic_save(stage, temporary, HANDWHEEL_OUTPUT)


def _build_valve() -> None:
    source_stage = Usd.Stage.Open(str(VALVE_SOURCE))
    if source_stage is None:
        raise RuntimeError(f"Failed to open {VALVE_SOURCE}")
    source_points = np.asarray(
        _source_mesh(source_stage).GetPointsAttr().Get(), dtype=np.float64
    )
    source_radial = np.linalg.norm(source_points[:, (0, 2)], axis=1)
    stem_mask = source_points[:, 1] >= VALVE_STEM_SOURCE_BLEND_END
    source_stem_max_radius = float(source_radial[stem_mask].max())
    full_stem_scale = VALVE_STEM_RADIUS / (
        VALVE_SCALE * source_stem_max_radius
    )
    blend = np.clip(
        (source_points[:, 1] - VALVE_STEM_SOURCE_BLEND_START)
        / (VALVE_STEM_SOURCE_BLEND_END - VALVE_STEM_SOURCE_BLEND_START),
        0.0,
        1.0,
    )
    radial_scale = 1.0 + blend * (full_stem_scale - 1.0)
    remapped = source_points.copy()
    remapped[:, 0] *= radial_scale
    remapped[:, 2] *= radial_scale
    points = np.column_stack((remapped[:, 0], -remapped[:, 2], remapped[:, 1]))
    points *= VALVE_SCALE
    source_y_min = float(source_points[:, 1].min())
    points[:, 2] -= points[:, 2].min()
    points = points.astype(np.float32)
    stem_base = (
        VALVE_STEM_SOURCE_BLEND_START - source_y_min
    ) * VALVE_SCALE
    stem_tip = float(points[:, 2].max())
    stem_length = stem_tip - stem_base

    temporary = VALVE_OUTPUT.with_name(
        f".{VALVE_OUTPUT.stem}.{os.getpid()}.tmp{VALVE_OUTPUT.suffix}"
    )
    temporary.unlink(missing_ok=True)
    stage, root = _prepare_stage(temporary, "ThreeWayBallValve")
    root_path = root.GetPath()
    _copy_visual_mesh(source_stage, stage, root_path, points)

    rigid = UsdPhysics.RigidBodyAPI.Apply(root)
    rigid.CreateRigidBodyEnabledAttr(True)
    rigid.CreateKinematicEnabledAttr(True)
    mass = UsdPhysics.MassAPI.Apply(root)
    mass.CreateMassAttr(5.0)
    for name, value, value_type in (
        ("physxRigidBody:disableGravity", True, Sdf.ValueTypeNames.Bool),
        (
            "physxRigidBody:solverPositionIterationCount",
            16,
            Sdf.ValueTypeNames.Int,
        ),
        (
            "physxRigidBody:solverVelocityIterationCount",
            4,
            Sdf.ValueTypeNames.Int,
        ),
    ):
        root.CreateAttribute(name, value_type).Set(value)

    physics_root = root_path.AppendChild("PhysicsMaterials")
    UsdGeom.Scope.Define(stage, physics_root)
    body_material = _make_physics_material(
        stage, physics_root.AppendChild("Body"), 0.8, 0.6
    )
    insertion_material = _make_physics_material(
        stage, physics_root.AppendChild("Stem"), 0.20, 0.15
    )
    collision_root = root_path.AppendChild("Collision")
    collision_xform = UsdGeom.Xform.Define(stage, collision_root)
    collision_xform.CreateVisibilityAttr(UsdGeom.Tokens.invisible)

    def cylinder(
        name: str,
        radius: float,
        height: float,
        axis: str,
        position: tuple[float, float, float],
        material: UsdShade.Material,
        contact_offset: float = OUTER_CONTACT_OFFSET,
    ) -> None:
        geometry = UsdGeom.Cylinder.Define(
            stage, collision_root.AppendChild(name)
        )
        geometry.CreateRadiusAttr(radius)
        geometry.CreateHeightAttr(height)
        geometry.CreateAxisAttr(axis)
        geometry.AddTranslateOp().Set(Gf.Vec3d(*position))
        geometry.CreatePurposeAttr(UsdGeom.Tokens.guide)
        geometry.CreateVisibilityAttr(UsdGeom.Tokens.invisible)
        _bind_collision(geometry.GetPrim(), material, contact_offset)

    body = UsdGeom.Sphere.Define(stage, collision_root.AppendChild("Body"))
    body.CreateRadiusAttr(0.050 * TASK_SIZE_SCALE)
    body.AddTranslateOp().Set(
        Gf.Vec3d(0.0, 0.0, 0.060 * TASK_SIZE_SCALE)
    )
    body.CreatePurposeAttr(UsdGeom.Tokens.guide)
    body.CreateVisibilityAttr(UsdGeom.Tokens.invisible)
    _bind_collision(body.GetPrim(), body_material, OUTER_CONTACT_OFFSET)
    cylinder(
        "MainPipe",
        0.030 * TASK_SIZE_SCALE,
        0.200 * TASK_SIZE_SCALE,
        UsdGeom.Tokens.y,
        (0.0, 0.0, 0.052 * TASK_SIZE_SCALE),
        body_material,
    )
    cylinder(
        "ThirdPort",
        0.030 * TASK_SIZE_SCALE,
        0.114 * TASK_SIZE_SCALE,
        UsdGeom.Tokens.x,
        (0.0, 0.0, 0.052 * TASK_SIZE_SCALE),
        body_material,
    )
    cylinder(
        "Neck",
        0.029 * TASK_SIZE_SCALE,
        0.060 * TASK_SIZE_SCALE,
        UsdGeom.Tokens.z,
        (0.0, 0.0, 0.123 * TASK_SIZE_SCALE),
        body_material,
    )
    cylinder(
        "Collar",
        0.034 * TASK_SIZE_SCALE,
        0.012 * TASK_SIZE_SCALE,
        UsdGeom.Tokens.z,
        (0.0, 0.0, stem_base - 0.006 * TASK_SIZE_SCALE),
        body_material,
    )
    cylinder(
        "Stem",
        VALVE_STEM_RADIUS,
        stem_length,
        UsdGeom.Tokens.z,
        (0.0, 0.0, stem_base + 0.5 * stem_length),
        insertion_material,
        INSERTION_CONTACT_OFFSET,
    )

    bounds_min = points.min(axis=0)
    bounds_max = points.max(axis=0)
    metadata = {
        "task:sizeX": float(bounds_max[0] - bounds_min[0]),
        "task:sizeY": float(bounds_max[1] - bounds_min[1]),
        "task:sizeZ": float(bounds_max[2] - bounds_min[2]),
        "task:stemDiameter": 2.0 * VALVE_STEM_RADIUS,
        "task:stemBaseZ": stem_base,
        "task:stemTipZ": stem_tip,
        "task:stemLength": stem_length,
        "task:axis": "Z",
        "task:sizeScale": TASK_SIZE_SCALE,
        "task:collisionStyle": "compound primitives with continuous stem",
    }
    for name, value in metadata.items():
        value_type = (
            Sdf.ValueTypeNames.String
            if isinstance(value, str)
            else Sdf.ValueTypeNames.Double
        )
        root.CreateAttribute(name, value_type).Set(value)
    _atomic_save(stage, temporary, VALVE_OUTPUT)


def build() -> None:
    for source in (HANDWHEEL_SOURCE, VALVE_SOURCE):
        if not source.is_file():
            raise FileNotFoundError(source)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    try:
        _build_handwheel()
        _build_valve()
    finally:
        for temporary in OUTPUT_ROOT.glob(".*.tmp.usd"):
            temporary.unlink(missing_ok=True)
    print(f"Built {HANDWHEEL_OUTPUT}")
    print(f"Built {VALVE_OUTPUT}")


if __name__ == "__main__":
    build()
