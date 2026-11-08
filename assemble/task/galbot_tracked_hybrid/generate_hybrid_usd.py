#!/usr/bin/env python3
"""Build a tracked-base robot with the Galbot One Golf upper body."""

import os
from pathlib import Path

from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics


HERE = Path(__file__).resolve().parent
TRACKED_G1 = Path(
    os.environ.get("BENCH_ASSEMBLE_ROOT", str(Path(__file__).resolve().parents[2])) + "/assets-new/robot/"
    "Tracked_G1_UpperBody_SolidBoom/Tracked_G1_UpperBody_SolidBoom.usd"
)
GALBOT = Path(
    os.environ.get("BENCH_ASSEMBLE_ROOT", str(Path(__file__).resolve().parents[2])) + "/assets-new/robot/"
    "galbot_one_golf/usd/galbot_one_golf.usda"
)
OUTPUT = HERE / "Tracked_Galbot_UpperBody_SolidBoom.usda"

# The mount frame is copied from the working tracked-G1 fixed joint.
MOUNT_POSITION = Gf.Vec3f(-1.1624116, 0.0, 0.96743286)

# Galbot's upper-body root is leg_link5.  Rotate the Galbot root by 180 deg
# around Z so its forward direction matches the original G1, then translate
# leg_link5's body origin exactly onto the existing tracked-carriage mount.
GALBOT_ROOT_TRANSLATION = Gf.Vec3d(-1.05546936, 0.0, 0.41376818)
GALBOT_ROOT_ORIENTATION = Gf.Quatd(0.0, Gf.Vec3d(0.0, 0.0, 1.0))

# World orientation of leg_link5 after applying the 180-degree root turn.
MOUNT_FRAME_ORIENTATION = Gf.Quatf(
    0.49999815,
    Gf.Vec3f(0.50000167, -0.50000173, -0.49999812),
)

UPPER_BODY_PREFIXES = (
    "leg_link5",
    "head_link",
    "left_arm_link",
    "right_arm_link",
    "left_gripper_",
    "right_gripper_",
)


def is_kept_body(path: Sdf.Path, galbot_root: Sdf.Path) -> bool:
    try:
        relative = path.MakeRelativePath(galbot_root).pathString
    except Exception:
        return False
    top_name = relative.split("/", 1)[0]
    return any(top_name.startswith(prefix) for prefix in UPPER_BODY_PREFIXES)


def main() -> None:
    if not TRACKED_G1.is_file():
        raise FileNotFoundError(TRACKED_G1)
    if not GALBOT.is_file():
        raise FileNotFoundError(GALBOT)
    if OUTPUT.exists():
        OUTPUT.unlink()

    stage = Usd.Stage.CreateNew(str(OUTPUT))
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)

    assembly = UsdGeom.Xform.Define(stage, "/Assembly").GetPrim()
    assembly.GetReferences().AddReference(str(TRACKED_G1))
    stage.SetDefaultPrim(assembly)
    assembly.SetMetadata(
        "comment",
        "Tracked base and solid boom from the G1 assembly; upper body from Galbot One Golf.",
    )

    # Remove the complete Unitree body, including its root joint and physics.
    stage.OverridePrim("/Assembly/G1UpperBody").SetActive(False)

    galbot = UsdGeom.Xform.Define(stage, "/Assembly/GalbotUpperBody").GetPrim()
    galbot.GetReferences().AddReference(str(GALBOT))
    galbot.GetAttribute("xformOp:translate").Set(GALBOT_ROOT_TRANSLATION)
    galbot.GetAttribute("xformOp:orient").Set(GALBOT_ROOT_ORIENTATION)
    galbot.GetAttribute("xformOp:scale").Set(Gf.Vec3d(1.0))
    galbot.GetAttribute("xformOpOrder").Set(
        ["xformOp:translate", "xformOp:orient", "xformOp:scale"]
    )

    # Keep only Galbot's torso (leg_link5), head, arms, grippers and materials.
    # All Galbot mobile-base, wheel and lower telescopic-leg bodies are removed.
    for child in list(galbot.GetChildren()):
        name = child.GetName()
        if name in {"joints", "Looks", "VisualMaterials"}:
            continue
        if any(name.startswith(prefix) for prefix in UPPER_BODY_PREFIXES):
            continue
        stage.OverridePrim(child.GetPath()).SetActive(False)

    # Remove joints whose body endpoints are not both retained upper-body links.
    joints_path = galbot.GetPath().AppendChild("joints")
    joints = stage.GetPrimAtPath(joints_path)
    for joint_prim in list(joints.GetChildren()):
        if not joint_prim.IsA(UsdPhysics.Joint):
            continue
        joint = UsdPhysics.Joint(joint_prim)
        targets = joint.GetBody0Rel().GetTargets() + joint.GetBody1Rel().GetTargets()
        if not targets or not all(is_kept_body(path, galbot.GetPath()) for path in targets):
            stage.OverridePrim(joint_prim.GetPath()).SetActive(False)

        # The imported two-finger grippers are closed mechanical linkages.  Their
        # follower joints are therefore represented as PhysX loop constraints,
        # not internal articulation joints.  PhysxMimicJointAPI only accepts an
        # internal articulation joint and produces ten load-time errors here.
        # Removing the redundant mimic layer keeps both driven gripper joints and
        # all physical linkage constraints intact.
        if joint_prim.HasProperty("physxMimicJoint:rotZ:referenceJoint"):
            override = stage.OverridePrim(joint_prim.GetPath())
            api_schemas = Sdf.TokenListOp()
            api_schemas.deletedItems = ["PhysxMimicJointAPI:rotZ"]
            override.SetMetadata("apiSchemas", api_schemas)

    # Connect Galbot's retained root rigid body to the tracked lift carriage.
    mount = UsdPhysics.FixedJoint.Define(stage, "/Assembly/GalbotMountJoint")
    mount.CreateBody0Rel().SetTargets([Sdf.Path("/Assembly/LiftCarriage")])
    mount.CreateBody1Rel().SetTargets(
        [Sdf.Path("/Assembly/GalbotUpperBody/leg_link5")]
    )
    mount.CreateLocalPos0Attr(MOUNT_POSITION)
    mount.CreateLocalRot0Attr(MOUNT_FRAME_ORIENTATION)
    mount.CreateLocalPos1Attr(Gf.Vec3f(0.0))
    mount.CreateLocalRot1Attr(Gf.Quatf(1.0, Gf.Vec3f(0.0)))
    mount.CreateCollisionEnabledAttr(False)
    UsdPhysics.ArticulationRootAPI.Apply(mount.GetPrim())

    stage.GetRootLayer().customLayerData = {
        "sourceTrackedBase": str(TRACKED_G1),
        "sourceUpperBody": str(GALBOT),
        "upperBodyRoot": "/Assembly/GalbotUpperBody/leg_link5",
        "mountBody": "/Assembly/LiftCarriage",
    }
    stage.GetRootLayer().Save()
    print(OUTPUT)


if __name__ == "__main__":
    main()
