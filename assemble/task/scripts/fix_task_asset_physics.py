#!/usr/bin/env python3
"""Repair arm colliders and task-object mass properties for the drilling task."""

from __future__ import annotations

from pathlib import Path

from pxr import Usd, UsdGeom, UsdPhysics


SCRIPT_DIR = Path(__file__).resolve().parent
TASK_ROOT = SCRIPT_DIR.parent / "w1_mobile_drilling_task"
W1_ASSET = (
    TASK_ROOT
    / "assets"
    / "robot"
    / "Forklift_W1_Final"
    / "components"
    / "W1_UpperBody"
    / "components"
    / "W1.usd"
)
TASK_OBJECT_ASSET = (
    TASK_ROOT / "assets" / "objects" / "gasoline_powered_post_driver.usd"
)

ARM_SIDES = ("left", "right")
ARM_LINK_NUMBERS = range(1, 7)
TASK_OBJECT_MASS_KG = 5.0


def _open_stage(path: Path) -> Usd.Stage:
    stage = Usd.Stage.Open(str(path))
    if stage is None:
        raise RuntimeError(f"Could not open USD asset: {path}")
    return stage


def _arm_link_rigid_bodies(stage: Usd.Stage) -> dict[str, Usd.Prim]:
    wanted = {
        f"{side}_link{number}"
        for side in ARM_SIDES
        for number in ARM_LINK_NUMBERS
    }
    links = {
        prim.GetName(): prim
        for prim in stage.Traverse()
        if prim.GetName() in wanted and prim.HasAPI(UsdPhysics.RigidBodyAPI)
    }
    missing = sorted(wanted - set(links))
    if missing:
        raise RuntimeError(f"Missing W1 arm rigid bodies: {missing}")
    return links


def configure_arm_colliders() -> int:
    stage = _open_stage(W1_ASSET)
    links = _arm_link_rigid_bodies(stage)
    source_paths = set()
    configured = 0

    for link_name, rigid_body in sorted(links.items()):
        visual = rigid_body.GetChild(link_name)
        if not visual:
            raise RuntimeError(f"Missing visual instance below {rigid_body.GetPath()}")

        references = visual.GetMetadata("references")
        if references is None:
            raise RuntimeError(f"Visual prim has no reference: {visual.GetPath()}")
        reference_items = references.GetAddedOrExplicitItems()
        if len(reference_items) != 1 or reference_items[0].assetPath:
            raise RuntimeError(
                f"Expected one internal visual reference at {visual.GetPath()}"
            )

        source_path = reference_items[0].primPath
        if source_path in source_paths:
            raise RuntimeError(f"Arm links unexpectedly share prototype {source_path}")
        source_paths.add(source_path)

        # The flattened prototype source is authored as an ``over`` and is not
        # expanded by PrimRange.  Its single mesh uses the rigid-link name.
        mesh = stage.GetPrimAtPath(source_path.AppendChild(link_name))
        if not mesh or not mesh.IsA(UsdGeom.Mesh):
            raise RuntimeError(
                f"Expected mesh {source_path.AppendChild(link_name)}"
            )
        collision = (
            UsdPhysics.CollisionAPI(mesh)
            if mesh.HasAPI(UsdPhysics.CollisionAPI)
            else UsdPhysics.CollisionAPI.Apply(mesh)
        )
        collision.CreateCollisionEnabledAttr().Set(True)
        mesh_collision = (
            UsdPhysics.MeshCollisionAPI(mesh)
            if mesh.HasAPI(UsdPhysics.MeshCollisionAPI)
            else UsdPhysics.MeshCollisionAPI.Apply(mesh)
        )
        mesh_collision.CreateApproximationAttr().Set(
            UsdPhysics.Tokens.convexHull
        )
        configured += 1

    if configured != 12:
        raise RuntimeError(f"Expected to configure 12 arm colliders, got {configured}")
    stage.GetRootLayer().Save()
    return configured


def configure_task_object() -> tuple[float, int]:
    stage = _open_stage(TASK_OBJECT_ASSET)
    root = stage.GetDefaultPrim()
    if not root or not root.HasAPI(UsdPhysics.RigidBodyAPI):
        raise RuntimeError("Task object default prim is not a rigid body")

    mass = (
        UsdPhysics.MassAPI(root)
        if root.HasAPI(UsdPhysics.MassAPI)
        else UsdPhysics.MassAPI.Apply(root)
    )
    mass.CreateMassAttr().Set(TASK_OBJECT_MASS_KG)
    mass.CreateDensityAttr().Set(0.0)

    collider_count = 0
    for prim in stage.Traverse():
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        collision = UsdPhysics.CollisionAPI(prim)
        collision.CreateCollisionEnabledAttr().Set(True)
        if prim.IsA(UsdGeom.Mesh):
            mesh_collision = (
                UsdPhysics.MeshCollisionAPI(prim)
                if prim.HasAPI(UsdPhysics.MeshCollisionAPI)
                else UsdPhysics.MeshCollisionAPI.Apply(prim)
            )
            mesh_collision.CreateApproximationAttr().Set(
                UsdPhysics.Tokens.convexDecomposition
            )
        collider_count += 1

    if collider_count == 0:
        raise RuntimeError("Task object has no collision geometry")
    stage.GetRootLayer().Save()
    return TASK_OBJECT_MASS_KG, collider_count


def main() -> None:
    arm_count = configure_arm_colliders()
    object_mass, object_colliders = configure_task_object()
    print(f"Configured W1 arm colliders: {arm_count}")
    print(
        "Configured task object: "
        f"mass={object_mass:.3f} kg, colliders={object_colliders}, "
        "approximation=convexDecomposition"
    )


if __name__ == "__main__":
    main()
