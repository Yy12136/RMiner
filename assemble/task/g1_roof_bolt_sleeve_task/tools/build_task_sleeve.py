#!/usr/bin/env python3
"""Build the task-specific ID53/OD55/L250 roof-bolt sleeve."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade


ASSEMBLE_ROOT = Path(__file__).resolve().parents[3]
OBJECT_ROOT = ASSEMBLE_ROOT / "assets-new" / "object"
SOURCE = OBJECT_ROOT / "steel_mining_roof_bolt_coupling_sleeve.usd"
OUTPUT = OBJECT_ROOT / "steel_mining_roof_bolt_coupling_sleeve_ID53_OD55_COD65_L250.usd"

SOURCE_MESH_PATH = Sdf.Path("/world/geometry_0/geometry_0")
SOURCE_LOOKS_PATH = Sdf.Path("/world/Looks")
ROOT_PATH = Sdf.Path("/CouplingSleeve")
VISUAL_PATH = ROOT_PATH.AppendChild("Visual")
COLLISION_PATH = ROOT_PATH.AppendChild("CollisionMesh")
LOOKS_PATH = ROOT_PATH.AppendChild("Looks")

INNER_RADIUS = 0.0265
OUTER_RADIUS = 0.0275
TARGET_LENGTH = 0.250
# The sleeve has 4 mm nominal radial clearance over the 45 mm roof bolt.
# A 3 mm sleeve contact envelope plus the bolt's 0.5 mm envelope made the
# effective clearance negative, so PhysX rejected a geometrically valid
# insertion before the surfaces touched. Keep both insertion partners at a
# 0.5 mm envelope, leaving 4 mm of effective radial collision clearance.
CONTACT_OFFSET = 0.0005
COLLISION_SEGMENTS = 16
# Give teleoperation an additional 1 mm of radial collision clearance beyond
# the visible 53 mm bore. The polygon vertices are placed outside the desired
# tangent radius so every flat inner face resolves to an exact 55 mm bore.
COLLISION_BORE_RADIUS = 0.0275
COLLISION_INNER_RADIUS = COLLISION_BORE_RADIUS / math.cos(
    math.pi / COLLISION_SEGMENTS
)
# Keep the visible OD at 55 mm, but use a slightly wider 65 mm grip envelope.
# This gives each explicit convex ring segment a robust 5 mm radial wall so a
# closing Dex1 finger cannot tunnel through the very thin visible sleeve.
COLLISION_OUTER_RADIUS = 0.0325

# Effective mass properties of assets-new/object/leiguan.usd under its default
# 1000 kg/m^3 PhysX density. Copying the resolved values makes this differently
# sized sleeve respond like Leiguan without changing its visual/collision mesh.
TASK_MASS = 0.2834361495612364
INERTIA_TRANSVERSE = 0.00294677466
INERTIA_AXIAL = 0.000106728127


def _flatten_and_remap_points(stage: Usd.Stage, mesh_prim: Usd.Prim):
    points = np.asarray(UsdGeom.Mesh(mesh_prim).GetPointsAttr().Get(), dtype=np.float64)
    transform = np.asarray(
        UsdGeom.XformCache().GetLocalToWorldTransform(mesh_prim), dtype=np.float64
    )
    world = np.c_[points, np.ones(len(points), dtype=np.float64)] @ transform
    world = world[:, :3]

    radial = np.linalg.norm(world[:, :2], axis=1)
    source_inner = float(radial.min())
    source_outer = float(radial.max())
    source_z_min = float(world[:, 2].min())
    source_z_max = float(world[:, 2].max())
    source_z_center = 0.5 * (source_z_min + source_z_max)
    source_length = source_z_max - source_z_min
    if source_inner <= 0.0 or source_outer <= source_inner or source_length <= 0.0:
        raise RuntimeError("Source sleeve bounds are invalid")

    mapped_radius = INNER_RADIUS + (
        (radial - source_inner)
        / (source_outer - source_inner)
        * (OUTER_RADIUS - INNER_RADIUS)
    )
    xy_scale = mapped_radius / np.maximum(radial, 1.0e-12)
    output = np.empty_like(world, dtype=np.float32)
    output[:, 0] = world[:, 0] * xy_scale
    output[:, 1] = world[:, 1] * xy_scale
    output[:, 2] = (world[:, 2] - source_z_center) * (
        TARGET_LENGTH / source_length
    )
    return output, source_inner, source_outer, source_length


def _copy_display_color(source_mesh: UsdGeom.Mesh, target_mesh: UsdGeom.Mesh) -> None:
    source = UsdGeom.PrimvarsAPI(source_mesh).GetPrimvar("displayColor")
    if not source or not source.HasValue():
        return
    target = UsdGeom.PrimvarsAPI(target_mesh).CreatePrimvar(
        "displayColor",
        source.GetTypeName(),
        source.GetInterpolation(),
        source.GetElementSize(),
    )
    target.Set(source.Get())
    if source.IsIndexed():
        target.SetIndices(source.GetIndices())


def _create_simple_hollow_cylinder_collision(stage: Usd.Stage) -> None:
    """Create an explicit 16-piece convex ring with a guaranteed open bore."""

    half_length = 0.5 * TARGET_LENGTH
    collision_root = UsdGeom.Xform.Define(stage, COLLISION_PATH)
    collision_root.CreatePurposeAttr(UsdGeom.Tokens.guide)
    collision_root.CreateVisibilityAttr(UsdGeom.Tokens.invisible)

    # Both radii describe the flat face locations. Moving the vertices outward
    # by 1/cos(half-segment-angle) makes the inscribed polygon tangent to those
    # exact radii rather than shrinking the cooked inner/outer diameters.
    half_segment_angle = math.pi / COLLISION_SEGMENTS
    inner_vertex_radius = COLLISION_INNER_RADIUS
    outer_vertex_radius = COLLISION_OUTER_RADIUS / math.cos(half_segment_angle)
    face_counts = [4, 4, 4, 4, 4, 4]
    face_indices = [
        0, 3, 2, 1,
        4, 5, 6, 7,
        0, 1, 5, 4,
        1, 2, 6, 5,
        2, 3, 7, 6,
        3, 0, 4, 7,
    ]

    for index in range(COLLISION_SEGMENTS):
        angle_0 = 2.0 * math.pi * index / COLLISION_SEGMENTS
        angle_1 = 2.0 * math.pi * (index + 1) / COLLISION_SEGMENTS

        def point(radius: float, angle: float, z: float) -> Gf.Vec3f:
            return Gf.Vec3f(radius * math.cos(angle), radius * math.sin(angle), z)

        points = [
            point(inner_vertex_radius, angle_0, -half_length),
            point(outer_vertex_radius, angle_0, -half_length),
            point(outer_vertex_radius, angle_1, -half_length),
            point(inner_vertex_radius, angle_1, -half_length),
            point(inner_vertex_radius, angle_0, half_length),
            point(outer_vertex_radius, angle_0, half_length),
            point(outer_vertex_radius, angle_1, half_length),
            point(inner_vertex_radius, angle_1, half_length),
        ]
        path = COLLISION_PATH.AppendChild(f"Segment_{index:02d}")
        collision = UsdGeom.Mesh.Define(stage, path)
        collision.CreatePointsAttr(points)
        collision.CreateFaceVertexCountsAttr(face_counts)
        collision.CreateFaceVertexIndicesAttr(face_indices)
        collision.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
        xyz = np.asarray(points, dtype=np.float32)
        extent_min = tuple(float(value) for value in xyz.min(axis=0))
        extent_max = tuple(float(value) for value in xyz.max(axis=0))
        collision.CreateExtentAttr(
            [Gf.Vec3f(*extent_min), Gf.Vec3f(*extent_max)]
        )
        collision.CreatePurposeAttr(UsdGeom.Tokens.guide)
        collision.CreateVisibilityAttr(UsdGeom.Tokens.invisible)
        UsdPhysics.CollisionAPI.Apply(
            collision.GetPrim()
        ).CreateCollisionEnabledAttr(True)
        mesh_api = UsdPhysics.MeshCollisionAPI.Apply(collision.GetPrim())
        mesh_api.CreateApproximationAttr().Set("convexHull")
        collision.GetPrim().CreateAttribute(
            "physxCollision:contactOffset", Sdf.ValueTypeNames.Float
        ).Set(CONTACT_OFFSET)
        collision.GetPrim().CreateAttribute(
            "physxCollision:restOffset", Sdf.ValueTypeNames.Float
        ).Set(0.0)


def build() -> None:
    if not SOURCE.is_file():
        raise FileNotFoundError(SOURCE)
    source_stage = Usd.Stage.Open(str(SOURCE))
    if source_stage is None:
        raise RuntimeError(f"Failed to open {SOURCE}")
    source_mesh_prim = source_stage.GetPrimAtPath(SOURCE_MESH_PATH)
    if not source_mesh_prim or not source_mesh_prim.IsA(UsdGeom.Mesh):
        raise RuntimeError(f"Missing source mesh {SOURCE_MESH_PATH}")

    points, source_inner, source_outer, source_length = _flatten_and_remap_points(
        source_stage, source_mesh_prim
    )
    source_mesh = UsdGeom.Mesh(source_mesh_prim)

    stage = Usd.Stage.CreateNew(str(OUTPUT))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, ROOT_PATH).GetPrim()
    stage.SetDefaultPrim(root)
    root.SetMetadata("kind", "component")

    rigid = UsdPhysics.RigidBodyAPI.Apply(root)
    rigid.CreateRigidBodyEnabledAttr(True)
    mass = UsdPhysics.MassAPI.Apply(root)
    mass.CreateMassAttr(TASK_MASS)
    mass.CreateCenterOfMassAttr(Gf.Vec3f(0.0, 0.0, 0.0))
    mass.CreateDiagonalInertiaAttr(
        Gf.Vec3f(INERTIA_TRANSVERSE, INERTIA_TRANSVERSE, INERTIA_AXIAL)
    )
    mass.CreatePrincipalAxesAttr(Gf.Quatf(1.0, Gf.Vec3f(0.0, 0.0, 0.0)))
    root.CreateAttribute(
        "physxRigidBody:disableGravity", Sdf.ValueTypeNames.Bool
    ).Set(False)
    root.CreateAttribute(
        "physxRigidBody:enableSpeculativeCCD", Sdf.ValueTypeNames.Bool
    ).Set(True)
    root.CreateAttribute(
        "physxRigidBody:solverPositionIterationCount", Sdf.ValueTypeNames.Int
    ).Set(16)
    root.CreateAttribute(
        "physxRigidBody:solverVelocityIterationCount", Sdf.ValueTypeNames.Int
    ).Set(4)
    root.CreateAttribute(
        "physxRigidBody:maxDepenetrationVelocity", Sdf.ValueTypeNames.Float
    ).Set(1.0)
    visual = UsdGeom.Mesh.Define(stage, VISUAL_PATH)
    visual.CreatePointsAttr(points)
    visual.CreateFaceVertexCountsAttr(source_mesh.GetFaceVertexCountsAttr().Get())
    visual.CreateFaceVertexIndicesAttr(source_mesh.GetFaceVertexIndicesAttr().Get())
    visual.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    visual.CreateExtentAttr(
        [
            Gf.Vec3f(-OUTER_RADIUS, -OUTER_RADIUS, -0.5 * TARGET_LENGTH),
            Gf.Vec3f(OUTER_RADIUS, OUTER_RADIUS, 0.5 * TARGET_LENGTH),
        ]
    )
    _copy_display_color(source_mesh, visual)

    # The source look is self-contained and reads the face-varying displayColor.
    Sdf.CopySpec(
        source_stage.GetRootLayer(),
        SOURCE_LOOKS_PATH,
        stage.GetRootLayer(),
        LOOKS_PATH,
    )
    material = UsdShade.Material.Get(
        stage, LOOKS_PATH.AppendChild("VertexColorMaterial")
    )
    if not material:
        raise RuntimeError("Copied source material is missing")
    UsdShade.MaterialBindingAPI.Apply(visual.GetPrim()).Bind(material)

    _create_simple_hollow_cylinder_collision(stage)
    root.CreateAttribute("task:innerDiameter", Sdf.ValueTypeNames.Double).Set(
        2.0 * INNER_RADIUS
    )
    root.CreateAttribute("task:outerDiameter", Sdf.ValueTypeNames.Double).Set(
        2.0 * OUTER_RADIUS
    )
    root.CreateAttribute("task:length", Sdf.ValueTypeNames.Double).Set(TARGET_LENGTH)
    root.CreateAttribute("task:collisionStyle", Sdf.ValueTypeNames.String).Set(
        "explicit 16-piece convex ring"
    )
    root.CreateAttribute("task:collisionInnerDiameter", Sdf.ValueTypeNames.Double).Set(
        2.0 * COLLISION_BORE_RADIUS
    )
    root.CreateAttribute("task:collisionOuterDiameter", Sdf.ValueTypeNames.Double).Set(
        2.0 * COLLISION_OUTER_RADIUS
    )
    root.CreateAttribute("task:physicsReference", Sdf.ValueTypeNames.String).Set(
        "leiguan.usd"
    )
    root.CreateAttribute("task:inertiaTransverse", Sdf.ValueTypeNames.Double).Set(
        INERTIA_TRANSVERSE
    )
    root.CreateAttribute("task:inertiaAxial", Sdf.ValueTypeNames.Double).Set(
        INERTIA_AXIAL
    )
    root.CreateAttribute("task:sourceInnerRadius", Sdf.ValueTypeNames.Double).Set(
        source_inner
    )
    root.CreateAttribute("task:sourceOuterRadius", Sdf.ValueTypeNames.Double).Set(
        source_outer
    )
    root.CreateAttribute("task:sourceLength", Sdf.ValueTypeNames.Double).Set(
        source_length
    )

    stage.GetRootLayer().Save()
    print(f"Built {OUTPUT}")
    print(
        f"source radial=[{source_inner:.6f}, {source_outer:.6f}] m, "
        f"length={source_length:.6f} m"
    )
    print(
        f"target ID={2 * INNER_RADIUS:.3f} m, "
        f"OD={2 * OUTER_RADIUS:.3f} m, L={TARGET_LENGTH:.6f} m"
    )
    print(
        f"mass={TASK_MASS:.3f} kg, inertia="
        f"({INERTIA_TRANSVERSE:.9f}, {INERTIA_TRANSVERSE:.9f}, "
        f"{INERTIA_AXIAL:.9f}) kg*m^2"
    )


if __name__ == "__main__":
    build()
