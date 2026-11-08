#!/usr/bin/env python3
"""Bake Galbot joint-state positions into the USD's default rigid-body transforms.

The source Galbot USD stores every rigid link as a direct child of the robot root.
PhysicsJointStateAPI is only evaluated after physics starts, so changing joint state
alone still leaves the stage in its zero/T pose while editing.  This script walks
the active joint tree from leg_link5 and authors matching default-frame transforms.
"""

from pathlib import Path

from pxr import Gf, Usd, UsdGeom, UsdPhysics


HERE = Path(__file__).resolve().parent
USD_PATH = HERE / "Tracked_Galbot_UpperBody_SolidBoom.usda"
ROBOT = "/Assembly/GalbotUpperBody"
ROOT_BODY = f"{ROBOT}/leg_link5"


def vec3(value) -> Gf.Vec3d:
    return Gf.Vec3d(float(value[0]), float(value[1]), float(value[2]))


def quat(value) -> Gf.Quatd:
    imaginary = value.GetImaginary()
    return Gf.Quatd(
        float(value.GetReal()),
        Gf.Vec3d(float(imaginary[0]), float(imaginary[1]), float(imaginary[2])),
    ).GetNormalized()


def pose(prim):
    return (
        vec3(prim.GetAttribute("xformOp:translate").Get()),
        quat(prim.GetAttribute("xformOp:orient").Get()),
    )


def joint_frame(joint: UsdPhysics.Joint, suffix: str):
    pos = joint.GetPrim().GetAttribute(f"physics:localPos{suffix}").Get()
    rot = joint.GetPrim().GetAttribute(f"physics:localRot{suffix}").Get()
    return vec3(pos), quat(rot)


def rotate(q: Gf.Quatd, value: Gf.Vec3d) -> Gf.Vec3d:
    return vec3(q.Transform(value))


def child_pose(parent_pose, joint: UsdPhysics.Joint, position: float):
    parent_pos, parent_rot = parent_pose
    frame0_pos, frame0_rot = joint_frame(joint, "0")
    frame1_pos, frame1_rot = joint_frame(joint, "1")
    joint_prim = joint.GetPrim()

    motion_rot = Gf.Quatd(1.0, Gf.Vec3d(0.0))
    motion_offset = Gf.Vec3d(0.0)
    if joint_prim.IsA(UsdPhysics.RevoluteJoint):
        axis_name = UsdPhysics.RevoluteJoint(joint_prim).GetAxisAttr().Get()
        axis = {
            UsdPhysics.Tokens.x: Gf.Vec3d(1.0, 0.0, 0.0),
            UsdPhysics.Tokens.y: Gf.Vec3d(0.0, 1.0, 0.0),
            UsdPhysics.Tokens.z: Gf.Vec3d(0.0, 0.0, 1.0),
        }[axis_name]
        motion_rot = Gf.Rotation(axis, position).GetQuat()
    elif joint_prim.IsA(UsdPhysics.PrismaticJoint):
        axis_name = UsdPhysics.PrismaticJoint(joint_prim).GetAxisAttr().Get()
        axis = {
            UsdPhysics.Tokens.x: Gf.Vec3d(1.0, 0.0, 0.0),
            UsdPhysics.Tokens.y: Gf.Vec3d(0.0, 1.0, 0.0),
            UsdPhysics.Tokens.z: Gf.Vec3d(0.0, 0.0, 1.0),
        }[axis_name]
        motion_offset = axis * position

    joint_world_rot = (parent_rot * frame0_rot).GetNormalized()
    result_rot = (joint_world_rot * motion_rot * frame1_rot.GetInverse()).GetNormalized()
    joint_world_pos = parent_pos + rotate(parent_rot, frame0_pos)
    result_pos = (
        joint_world_pos
        + rotate(joint_world_rot, motion_offset)
        - rotate(result_rot, frame1_pos)
    )
    return result_pos, result_rot


def angular_state(joint_prim) -> float:
    attr = joint_prim.GetAttribute("state:angular:physics:position")
    if attr and attr.HasAuthoredValueOpinion():
        return float(attr.Get())
    return 0.0


def main() -> None:
    stage = Usd.Stage.Open(str(USD_PATH))
    if not stage:
        raise RuntimeError(f"Could not open {USD_PATH}")

    root = stage.GetPrimAtPath(ROOT_BODY)
    if not root or not root.IsActive():
        raise RuntimeError(f"Missing active root body: {ROOT_BODY}")

    joints = []
    joints_root = stage.GetPrimAtPath(f"{ROBOT}/joints")
    for prim in joints_root.GetChildren():
        if not prim.IsActive() or not prim.IsA(UsdPhysics.Joint):
            continue
        joint = UsdPhysics.Joint(prim)
        body0 = joint.GetBody0Rel().GetTargets()
        body1 = joint.GetBody1Rel().GetTargets()
        if len(body0) != 1 or len(body1) != 1:
            continue
        if not str(body0[0]).startswith(ROBOT) or not str(body1[0]).startswith(ROBOT):
            continue
        if not stage.GetPrimAtPath(body0[0]).IsActive() or not stage.GetPrimAtPath(body1[0]).IsActive():
            continue
        joints.append((joint, str(body0[0]), str(body1[0])))

    solved = {ROOT_BODY: pose(root)}
    pending = list(joints)
    while pending:
        progressed = False
        for item in list(pending):
            joint, body0, body1 = item
            if body0 not in solved:
                continue
            solved[body1] = child_pose(solved[body0], joint, angular_state(joint.GetPrim()))
            pending.remove(item)
            progressed = True
        if not progressed:
            unresolved = [joint.GetPrim().GetPath().pathString for joint, _, _ in pending]
            raise RuntimeError(f"Could not solve active joint tree: {unresolved}")

    root_layer = stage.GetRootLayer()
    stage.SetEditTarget(root_layer)
    for path, (position, orientation) in solved.items():
        if path == ROOT_BODY:
            continue
        prim = stage.OverridePrim(path)
        prim.GetAttribute("xformOp:translate").Set(position)
        prim.GetAttribute("xformOp:orient").Set(orientation)

    root_layer.Save()
    print(f"baked_bodies={len(solved) - 1}")
    print(USD_PATH)


if __name__ == "__main__":
    main()
