#!/usr/bin/env python3
"""Build the pre-loaded detonator assembly used by the wall-hole task.

The detonator shell and inserted rigid connector are authored as one PhysX
body.  The existing 26-link lead remains articulated and is fixed to the tail
of that compound body.  The output is portable and has no session layers.
"""

from __future__ import annotations

import math
from pathlib import Path

from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade


ASSEMBLE_ROOT = Path(__file__).resolve().parents[3]
OBJECT_ROOT = ASSEMBLE_ROOT / "assets-new" / "object"
LEIGUAN_SOURCE = OBJECT_ROOT / "leiguan.usd"
LEAD_SOURCE = OBJECT_ROOT / "chaganshengzi_standing.usd"
OUTPUT = OBJECT_ROOT / "loaded_detonator_assembly.usd"

ROOT_PATH = Sdf.Path("/LoadedDetonatorAssembly")
BODY_PATH = ROOT_PATH.AppendChild("Body")
LEIGUAN_MESH_PATH = BODY_PATH.AppendChild("LeiguanMesh")
CONNECTOR_MESH_PATH = BODY_PATH.AppendChild("ConnectorMesh")
ROPE_PATH = ROOT_PATH.AppendChild("Rope")

ROOT_MASS_KG = 0.10
LEIGUAN_OUTER_RADIUS_M = 0.0225
LEIGUAN_INNER_RADIUS_M = 0.01575
LEIGUAN_SOURCE_LENGTH_M = 0.35
CONNECTOR_RADIUS_M = 0.0105
CONNECTOR_LENGTH_M = 0.252
# Keep the shell only 10 mm longer than the internal connector.
LEIGUAN_LENGTH_M = CONNECTOR_LENGTH_M + 0.010
# The connector is flipped so its source +Z end points toward the detonator's
# rear opening (local -Z).  Leave that end 1 mm outside the shortened shell so
# the lead insertion point is visible instead of buried inside the casing.
CONNECTOR_PROTRUSION_M = 0.001
CONNECTOR_BASE_Z_M = (
    -0.5 * LEIGUAN_LENGTH_M - CONNECTOR_PROTRUSION_M + CONNECTOR_LENGTH_M
)
ROPE_ATTACHMENT_Z_M = CONNECTOR_BASE_Z_M - 0.25029654499407916


def _copy_spec(
    source_layer: Sdf.Layer,
    source_path: str,
    destination_layer: Sdf.Layer,
    destination_path: Sdf.Path,
) -> None:
    if not Sdf.CopySpec(
        source_layer,
        Sdf.Path(source_path),
        destination_layer,
        destination_path,
    ):
        raise RuntimeError(f"Failed to copy {source_path} to {destination_path}")


def _remap_relationship_targets(prim: Usd.Prim) -> None:
    old_rope = Sdf.Path("/World/rope_40cm")
    old_connector = Sdf.Path("/World/SmoothConnector")
    for descendant in Usd.PrimRange(prim):
        for relationship in descendant.GetRelationships():
            targets = relationship.GetTargets()
            if not targets:
                continue
            remapped = []
            changed = False
            for target in targets:
                if target.HasPrefix(old_rope):
                    target = target.ReplacePrefix(old_rope, ROPE_PATH)
                    changed = True
                elif target == old_connector:
                    target = BODY_PATH
                    changed = True
                remapped.append(target)
            if changed:
                relationship.SetTargets(remapped)


def _set_compound_mass(root: Usd.Prim) -> None:
    rigid = UsdPhysics.RigidBodyAPI.Apply(root)
    rigid.CreateRigidBodyEnabledAttr().Set(True)
    rigid.CreateKinematicEnabledAttr().Set(False)

    # Split the requested mass 75/25 between the hollow shell and connector,
    # then combine their analytic cylinder inertias.  Their centers are almost
    # coincident in the authored loaded pose, so the small parallel-axis term
    # is negligible compared with the transverse inertia.
    shell_mass = ROOT_MASS_KG * 0.75
    connector_mass = ROOT_MASS_KG - shell_mass
    shell_axial = 0.5 * shell_mass * (
        LEIGUAN_OUTER_RADIUS_M**2 + LEIGUAN_INNER_RADIUS_M**2
    )
    shell_transverse = shell_mass * (
        3.0 * (LEIGUAN_OUTER_RADIUS_M**2 + LEIGUAN_INNER_RADIUS_M**2)
        + LEIGUAN_LENGTH_M**2
    ) / 12.0
    connector_axial = 0.5 * connector_mass * CONNECTOR_RADIUS_M**2
    connector_transverse = connector_mass * (
        3.0 * CONNECTOR_RADIUS_M**2 + CONNECTOR_LENGTH_M**2
    ) / 12.0

    mass = UsdPhysics.MassAPI.Apply(root)
    mass.CreateMassAttr().Set(ROOT_MASS_KG)
    mass.CreateCenterOfMassAttr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
    mass.CreateDiagonalInertiaAttr().Set(
        Gf.Vec3f(
            shell_transverse + connector_transverse,
            shell_transverse + connector_transverse,
            shell_axial + connector_axial,
        )
    )
    mass.CreatePrincipalAxesAttr().Set(
        Gf.Quatf(1.0, Gf.Vec3f(0.0, 0.0, 0.0))
    )


def _configure_mesh(
    prim: Usd.Prim,
    approximation: str,
) -> None:
    collision = UsdPhysics.CollisionAPI.Apply(prim)
    collision.CreateCollisionEnabledAttr().Set(True)
    mesh_collision = UsdPhysics.MeshCollisionAPI.Apply(prim)
    mesh_collision.CreateApproximationAttr().Set(approximation)


def build() -> None:
    for path in (LEIGUAN_SOURCE, LEAD_SOURCE):
        if not path.is_file():
            raise FileNotFoundError(path)

    leiguan_stage = Usd.Stage.Open(str(LEIGUAN_SOURCE), Usd.Stage.LoadAll)
    lead_stage = Usd.Stage.Open(str(LEAD_SOURCE), Usd.Stage.LoadAll)
    if leiguan_stage is None or lead_stage is None:
        raise RuntimeError("Unable to open source detonator assets")

    temporary = OUTPUT.with_name(f".{OUTPUT.stem}.tmp{OUTPUT.suffix}")
    temporary.unlink(missing_ok=True)
    stage = Usd.Stage.CreateNew(str(temporary))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, ROOT_PATH).GetPrim()
    body = UsdGeom.Xform.Define(stage, BODY_PATH).GetPrim()
    stage.SetDefaultPrim(root)

    # Flatten only in memory so referenced geometry (notably the connector
    # mesh) is available to Sdf.CopySpec without retaining source references.
    leiguan_layer = leiguan_stage.Flatten()
    lead_layer = lead_stage.Flatten()

    _copy_spec(
        leiguan_layer,
        "/World/HollowCylinder/Mesh",
        stage.GetRootLayer(),
        LEIGUAN_MESH_PATH,
    )
    _copy_spec(
        leiguan_layer,
        "/World/Looks",
        stage.GetRootLayer(),
        ROOT_PATH.AppendChild("Looks"),
    )
    leiguan_mesh = stage.GetPrimAtPath(LEIGUAN_MESH_PATH)
    leiguan_xform = UsdGeom.Xformable(leiguan_mesh)
    leiguan_xform.ClearXformOpOrder()
    leiguan_xform.AddScaleOp().Set(
        Gf.Vec3f(
            0.45,
            0.45,
            LEIGUAN_LENGTH_M / LEIGUAN_SOURCE_LENGTH_M,
        )
    )
    UsdShade.MaterialBindingAPI.Apply(leiguan_mesh).Bind(
        UsdShade.Material(
            stage.GetPrimAtPath(ROOT_PATH.AppendPath("Looks/OmniPBR"))
        )
    )
    _configure_mesh(leiguan_mesh, "convexDecomposition")

    _copy_spec(
        lead_layer,
        "/World/SmoothConnector/Mesh",
        stage.GetRootLayer(),
        CONNECTOR_MESH_PATH,
    )
    _copy_spec(
        lead_layer,
        "/World/SmoothConnector/Material",
        stage.GetRootLayer(),
        ROOT_PATH.AppendChild("ConnectorMaterial"),
    )
    connector_mesh = stage.GetPrimAtPath(CONNECTOR_MESH_PATH)
    connector_xform = UsdGeom.Xformable(connector_mesh)
    connector_xform.ClearXformOpOrder()
    connector_xform.AddTranslateOp().Set(
        Gf.Vec3d(0.0, 0.0, CONNECTOR_BASE_Z_M)
    )
    connector_xform.AddOrientOp().Set(
        Gf.Quatf(0.0, Gf.Vec3f(1.0, 0.0, 0.0))
    )
    UsdShade.MaterialBindingAPI.Apply(connector_mesh).Bind(
        UsdShade.Material(
            stage.GetPrimAtPath(ROOT_PATH.AppendChild("ConnectorMaterial"))
        )
    )
    _configure_mesh(connector_mesh, "convexHull")

    _copy_spec(
        lead_layer,
        "/World/rope_40cm",
        stage.GetRootLayer(),
        ROPE_PATH,
    )
    rope = stage.GetPrimAtPath(ROPE_PATH)
    rope_xform = UsdGeom.Xformable(rope)
    translate_op = next(
        op
        for op in rope_xform.GetOrderedXformOps()
        if op.GetOpName() == "xformOp:translate"
    )
    translate_op.Set(Gf.Vec3d(0.26179010520261914, 0.0, ROPE_ATTACHMENT_Z_M))
    _remap_relationship_targets(rope)

    fixed_joint = UsdPhysics.FixedJoint(
        stage.GetPrimAtPath(ROPE_PATH.AppendPath("Links/Link_00/FixedJoint"))
    )
    if not fixed_joint:
        raise RuntimeError("Copied rope is missing its root FixedJoint")
    fixed_joint.GetBody0Rel().SetTargets([BODY_PATH])
    fixed_joint.GetLocalPos0Attr().Set(
        Gf.Vec3f(0.013475498, 0.0, ROPE_ATTACHMENT_Z_M)
    )
    fixed_joint.GetLocalRot0Attr().Set(
        Gf.Quatf(1.0, Gf.Vec3f(0.0, 0.0, 0.0))
    )
    fixed_joint.GetLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
    fixed_joint.GetLocalRot1Attr().Set(
        Gf.Quatf(1.0, Gf.Vec3f(0.0, 0.0, 0.0))
    )

    # The reusable object must not bring its own PhysicsScene into the task.
    stage.RemovePrim(ROOT_PATH.AppendChild("PhysicsScene"))
    _set_compound_mass(body)
    stage.GetRootLayer().Save()

    reopened = Usd.Stage.Open(str(temporary), Usd.Stage.LoadAll)
    if reopened is None:
        raise RuntimeError(f"Failed to reopen generated asset: {temporary}")
    rigid_count = sum(
        1
        for prim in reopened.Traverse()
        if prim.HasAPI(UsdPhysics.RigidBodyAPI)
    )
    joint_count = sum(
        1 for prim in reopened.Traverse() if prim.IsA(UsdPhysics.Joint)
    )
    collider_count = sum(
        1
        for prim in reopened.Traverse()
        if prim.HasAPI(UsdPhysics.CollisionAPI)
    )
    if (rigid_count, joint_count, collider_count) != (27, 26, 28):
        raise RuntimeError(
            "Unexpected generated topology: "
            f"rigid={rigid_count}, joints={joint_count}, "
            f"colliders={collider_count}"
        )

    temporary.replace(OUTPUT)
    print(
        f"Built {OUTPUT}: {rigid_count} rigid bodies, "
        f"{joint_count} joints, {collider_count} colliders"
    )


if __name__ == "__main__":
    build()
