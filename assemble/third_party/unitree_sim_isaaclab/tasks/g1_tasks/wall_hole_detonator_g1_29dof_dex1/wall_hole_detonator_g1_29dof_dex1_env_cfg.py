"""Tracked G1 task for loading a pre-assembled detonator into a wall hole."""

from pathlib import Path
import math
import os

import torch
from pxr import Gf, PhysxSchema, Usd, UsdGeom, UsdPhysics

import isaaclab.envs.mdp as base_mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim.spawners.from_files.from_files import spawn_from_usd
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass

from tasks.common_config import CameraBaseCfg, CameraPresets
from tasks.common_event.event_manager import SimpleEvent, SimpleEventManager
from tasks.common_observations.camera_state import get_camera_image
from tasks.common_observations.g1_29dof_state import get_robot_boy_joint_states
from tasks.common_observations.gripper_state import get_robot_gipper_joint_states
from tasks.g1_tasks.pick_place_cylinder_g1_29dof_dex1 import mdp
from tasks.g1_tasks.tabletop_grasp_g1_29dof_dex1.tabletop_grasp_g1_29dof_dex1_env_cfg import (
    ASSETS_ROOT,
    G1_MOUNT_LOCAL_ORIENTATION,
    G1_MOUNT_LOCAL_POSITION,
    TRACKED_G1_ASSEMBLY,
    WORKTABLE_NATIVE_TOP_Z_CM,
    spawn_combined_tracked_g1,
    spawn_static_worktable,
)


ASSEMBLE_ROOT = Path(__file__).resolve().parents[5]
TASK_ROOT = ASSEMBLE_ROOT / "task" / "g1_wall_hole_detonator_task"
ENVIRONMENT = (
    ASSETS_ROOT
    / "scene/RealScene04/real2sim_intermediate/realscene04_hole.usd"
)
WORKTABLE = ASSETS_ROOT / "object/TableTrolley_B/TableTrolley_B01_01.usd"
LOADED_DETONATOR_OBJECT = ASSETS_ROOT / "object/loaded_detonator_assembly.usd"
SOURCE_HOLE_MESH_PATH = "/World/leiguan/HollowCylinder/Mesh"

CAPTURED_TRACKED_BASE_X = -3.0386274360916987
CAPTURED_TRACKED_BASE_Y = 3.223047496135861
CAPTURED_TRACKED_BASE_YAW = -0.6097617191696184
# The tracked base's physical forward direction is its local -X direction,
# i.e. toward the wall (opposite the authored base yaw).  Start 10 cm closer
# to the wall while preserving the calibrated yaw.
ROBOT_FORWARD_SHIFT_M = float(
    os.environ.get("G1_WALL_HOLE_ROBOT_FORWARD_SHIFT_M", "0.10")
)
TABLE_OUTWARD_M = float(
    os.environ.get("G1_WALL_HOLE_TABLE_OUTWARD_M", "0.85")
)
TABLE_RIGHT_M = float(
    os.environ.get("G1_WALL_HOLE_TABLE_RIGHT_M", "0.65")
)
TABLE_OBJECT_LEFT_SHIFT_M = float(
    os.environ.get("G1_WALL_HOLE_TABLE_OBJECT_LEFT_SHIFT_M", "0.20")
)
TABLE_ONLY_RIGHT_SHIFT_M = float(
    os.environ.get("G1_WALL_HOLE_TABLE_ONLY_RIGHT_SHIFT_M", "0.05")
)
TABLE_ONLY_FORWARD_SHIFT_M = float(
    os.environ.get("G1_WALL_HOLE_TABLE_ONLY_FORWARD_SHIFT_M", "0.10")
)
OBJECT_OUTWARD_M = float(
    os.environ.get("G1_WALL_HOLE_OBJECT_OUTWARD_M", "0.85")
)
OBJECT_RIGHT_M = float(
    os.environ.get("G1_WALL_HOLE_OBJECT_RIGHT_M", "0.40")
)
OBJECT_WALL_BACKOFF_M = float(
    os.environ.get("G1_WALL_HOLE_OBJECT_WALL_BACKOFF_M", "0.15")
)
OBJECT_RANDOM_XY_LIMIT = float(
    os.environ.get("G1_WALL_HOLE_OBJECT_RANDOM_XY_M", "0.05")
)
TASK_TRACKED_LIFT_HEIGHT = float(
    os.environ.get("G1_WALL_HOLE_LIFT_HEIGHT_M", "0.058038")
)
TABLE_TOP_Z = 1.09
# Task 2 uses 0.01 to convert the trolley's authored centimetres to metres.
# This task intentionally uses half that XY scale: a roughly 0.37 x 0.61 m
# footprint instead of 0.74 x 1.22 m.  Height remains unchanged.
WORKTABLE_XY_SCALE = float(
    os.environ.get("G1_WALL_HOLE_TABLE_XY_SCALE", "0.005")
)
WORKTABLE_NATIVE_SHORT_CM = 73.88791275024414
WORKTABLE_NATIVE_LONG_CM = 122.0152645111084
TABLE_EDGE_MARGIN_M = 0.002
# After the requested clockwise quarter-turn, the trolley's long edge is
# outward/inward and its short edge is along the robot's left/right axis.
TABLE_OUTWARD_HALF_EXTENT = (
    0.5 * WORKTABLE_NATIVE_LONG_CM * WORKTABLE_XY_SCALE
)
TABLE_RIGHT_HALF_EXTENT = (
    0.5 * WORKTABLE_NATIVE_SHORT_CM * WORKTABLE_XY_SCALE
)
DETONATOR_OUTER_RADIUS = 0.0225
DETONATOR_HALF_LENGTH = 0.131
# Effective radius after the runtime bore-only expansion below.
HOLE_INNER_RADIUS = 0.03346875
HOLE_HALF_LENGTH = 0.30625
INSERTION_CONTACT_OFFSET = 0.0005
# The reconstructed target mesh has a 0.035 local-unit bore radius and a
# 0.050 local-unit outer radius.  Enlarge only the bore at spawn time to make
# teleoperation insertion robust; the visible/collision outer diameter stays
# exactly as authored.
HOLE_BORE_RADIUS_SCALE = 1.25
HOLE_MIN_WALL_THICKNESS_LOCAL = 0.0075


def _read_authored_hole_frame():
    """Read the hole center and outward axis before the task is spawned."""

    stage = Usd.Stage.Open(str(ENVIRONMENT), Usd.Stage.LoadAll)
    if stage is None:
        raise RuntimeError(f"Failed to open wall-hole scene: {ENVIRONMENT}")
    mesh = stage.GetPrimAtPath(SOURCE_HOLE_MESH_PATH)
    if not mesh or not mesh.IsA(UsdGeom.Mesh):
        raise RuntimeError(
            f"Wall-hole scene is missing {SOURCE_HOLE_MESH_PATH}"
        )
    transform = UsdGeom.XformCache().GetLocalToWorldTransform(mesh)
    center_source = transform.Transform(Gf.Vec3d(0.0, 0.0, 0.0))
    axis_endpoint = transform.Transform(Gf.Vec3d(0.0, 0.0, 1.0))
    axis_source = axis_endpoint - center_source

    # The task composes the environment with an outer 180 degree X rotation,
    # matching the coordinate convention used by the other RealScene04 tasks.
    center = Gf.Vec3d(
        center_source[0],
        -center_source[1],
        -center_source[2],
    )
    outward = Gf.Vec3d(
        axis_source[0],
        -axis_source[1],
        -axis_source[2],
    )
    length = outward.GetLength()
    if length <= 1.0e-8:
        raise RuntimeError("Wall-hole axis has zero length")
    outward /= length
    return (
        tuple(float(value) for value in center),
        tuple(float(value) for value in outward),
    )


HOLE_CENTER, HOLE_OUTWARD_AXIS = _read_authored_hole_frame()
_outward_xy_length = math.hypot(HOLE_OUTWARD_AXIS[0], HOLE_OUTWARD_AXIS[1])
if _outward_xy_length <= 1.0e-8:
    raise RuntimeError("Wall-hole axis has no horizontal component")
HOLE_OUTWARD_XY = (
    HOLE_OUTWARD_AXIS[0] / _outward_xy_length,
    HOLE_OUTWARD_AXIS[1] / _outward_xy_length,
)
HOLE_INWARD_XY = (-HOLE_OUTWARD_XY[0], -HOLE_OUTWARD_XY[1])
ROBOT_RIGHT_XY = (HOLE_INWARD_XY[1], -HOLE_INWARD_XY[0])
G1_FORWARD_YAW = math.atan2(HOLE_INWARD_XY[1], HOLE_INWARD_XY[0])
TASK_TRACKED_BASE_X = float(
    os.environ.get("G1_WALL_HOLE_BASE_X_M", str(CAPTURED_TRACKED_BASE_X))
)
TASK_TRACKED_BASE_Y = float(
    os.environ.get("G1_WALL_HOLE_BASE_Y_M", str(CAPTURED_TRACKED_BASE_Y))
)
TASK_TRACKED_BASE_YAW = float(
    os.environ.get("G1_WALL_HOLE_BASE_YAW_RAD", str(CAPTURED_TRACKED_BASE_YAW))
)
# Move along the robot's real forward axis, not along its yaw/+X axis.
TASK_TRACKED_BASE_X -= math.cos(TASK_TRACKED_BASE_YAW) * ROBOT_FORWARD_SHIFT_M
TASK_TRACKED_BASE_Y -= math.sin(TASK_TRACKED_BASE_YAW) * ROBOT_FORWARD_SHIFT_M

_task_base_cosine = math.cos(TASK_TRACKED_BASE_YAW)
_task_base_sine = math.sin(TASK_TRACKED_BASE_YAW)
TASK_G1_MOUNT_POSITION = (
    TASK_TRACKED_BASE_X
    + _task_base_cosine * G1_MOUNT_LOCAL_POSITION[0]
    - _task_base_sine * G1_MOUNT_LOCAL_POSITION[1],
    TASK_TRACKED_BASE_Y
    + _task_base_sine * G1_MOUNT_LOCAL_POSITION[0]
    + _task_base_cosine * G1_MOUNT_LOCAL_POSITION[1],
    G1_MOUNT_LOCAL_POSITION[2] + TASK_TRACKED_LIFT_HEIGHT,
)


def _multiply_quaternions_wxyz(left, right):
    lw, lx, ly, lz = left
    rw, rx, ry, rz = right
    return (
        lw * rw - lx * rx - ly * ry - lz * rz,
        lw * rx + lx * rw + ly * rz - lz * ry,
        lw * ry - lx * rz + ly * rw + lz * rx,
        lw * rz + lx * ry - ly * rx + lz * rw,
    )


TASK_G1_MOUNT_ORIENTATION = _multiply_quaternions_wxyz(
    (
        math.cos(0.5 * TASK_TRACKED_BASE_YAW),
        0.0,
        0.0,
        math.sin(0.5 * TASK_TRACKED_BASE_YAW),
    ),
    G1_MOUNT_LOCAL_ORIENTATION,
)

_TABLE_POSITION_BEFORE_GROUP_ROTATION = (
    HOLE_CENTER[0]
    + HOLE_OUTWARD_XY[0] * TABLE_OUTWARD_M
    + ROBOT_RIGHT_XY[0] * TABLE_RIGHT_M,
    HOLE_CENTER[1]
    + HOLE_OUTWARD_XY[1] * TABLE_OUTWARD_M
    + ROBOT_RIGHT_XY[1] * TABLE_RIGHT_M,
    0.0,
)
_TABLE_OBJECT_GROUP_POSITION = (
    _TABLE_POSITION_BEFORE_GROUP_ROTATION[0]
    - ROBOT_RIGHT_XY[0] * TABLE_OBJECT_LEFT_SHIFT_M,
    _TABLE_POSITION_BEFORE_GROUP_ROTATION[1]
    - ROBOT_RIGHT_XY[1] * TABLE_OBJECT_LEFT_SHIFT_M,
    0.0,
)
TABLE_POSITION = (
    _TABLE_OBJECT_GROUP_POSITION[0]
    + ROBOT_RIGHT_XY[0] * TABLE_ONLY_RIGHT_SHIFT_M
    + HOLE_INWARD_XY[0] * TABLE_ONLY_FORWARD_SHIFT_M,
    _TABLE_OBJECT_GROUP_POSITION[1]
    + ROBOT_RIGHT_XY[1] * TABLE_ONLY_RIGHT_SHIFT_M
    + HOLE_INWARD_XY[1] * TABLE_ONLY_FORWARD_SHIFT_M,
    0.0,
)
TABLE_YAW = G1_FORWARD_YAW - 0.5 * math.pi
LAYOUT_CLOCKWISE_YAW = -0.5 * math.pi
WORKTABLE_YAW = TABLE_YAW - 0.5 * math.pi + LAYOUT_CLOCKWISE_YAW
WORKTABLE_ORIENTATION = (
    math.cos(0.5 * WORKTABLE_YAW),
    0.0,
    0.0,
    math.sin(0.5 * WORKTABLE_YAW),
)
WORKTABLE_Z_SCALE = TABLE_TOP_Z / WORKTABLE_NATIVE_TOP_Z_CM

_LOADED_POSITION_BEFORE_GROUP_ROTATION = (
    HOLE_CENTER[0]
    + HOLE_OUTWARD_XY[0] * OBJECT_OUTWARD_M
    + ROBOT_RIGHT_XY[0] * OBJECT_RIGHT_M,
    HOLE_CENTER[1]
    + HOLE_OUTWARD_XY[1] * OBJECT_OUTWARD_M
    + ROBOT_RIGHT_XY[1] * OBJECT_RIGHT_M,
    0.0,
)
_object_relative_x = (
    _LOADED_POSITION_BEFORE_GROUP_ROTATION[0]
    - _TABLE_POSITION_BEFORE_GROUP_ROTATION[0]
)
_object_relative_y = (
    _LOADED_POSITION_BEFORE_GROUP_ROTATION[1]
    - _TABLE_POSITION_BEFORE_GROUP_ROTATION[1]
)
_layout_cosine = math.cos(LAYOUT_CLOCKWISE_YAW)
_layout_sine = math.sin(LAYOUT_CLOCKWISE_YAW)
_LOADED_POSITION_AFTER_GROUP_ROTATION = (
    _TABLE_OBJECT_GROUP_POSITION[0]
    + _layout_cosine * _object_relative_x
    - _layout_sine * _object_relative_y,
    _TABLE_OBJECT_GROUP_POSITION[1]
    + _layout_sine * _object_relative_x
    + _layout_cosine * _object_relative_y,
)
# Move only the loaded detonator away from the wall.  The table's position is
# deliberately independent of this backoff.
LOADED_DETONATOR_POSITION = (
    _LOADED_POSITION_AFTER_GROUP_ROTATION[0]
    + HOLE_OUTWARD_XY[0] * OBJECT_WALL_BACKOFF_M,
    _LOADED_POSITION_AFTER_GROUP_ROTATION[1]
    + HOLE_OUTWARD_XY[1] * OBJECT_WALL_BACKOFF_M,
    TABLE_TOP_Z + DETONATOR_OUTER_RADIUS + 0.002,
)


def _matrix_to_quaternion_wxyz(matrix):
    """Convert a row-major 3x3 rotation matrix to a normalized quaternion."""

    m00, m01, m02 = matrix[0]
    m10, m11, m12 = matrix[1]
    m20, m21, m22 = matrix[2]
    trace = m00 + m11 + m22
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * scale
        x = (m21 - m12) / scale
        y = (m02 - m20) / scale
        z = (m10 - m01) / scale
    elif m00 > m11 and m00 > m22:
        scale = math.sqrt(1.0 + m00 - m11 - m22) * 2.0
        w = (m21 - m12) / scale
        x = 0.25 * scale
        y = (m01 + m10) / scale
        z = (m02 + m20) / scale
    elif m11 > m22:
        scale = math.sqrt(1.0 + m11 - m00 - m22) * 2.0
        w = (m02 - m20) / scale
        x = (m01 + m10) / scale
        y = 0.25 * scale
        z = (m12 + m21) / scale
    else:
        scale = math.sqrt(1.0 + m22 - m00 - m11) * 2.0
        w = (m10 - m01) / scale
        x = (m02 + m20) / scale
        y = (m12 + m21) / scale
        z = 0.25 * scale
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    return (w / norm, x / norm, y / norm, z / norm)


# Columns are the desired world directions of local +X, +Y, +Z.  The complete
# tabletop layout is rotated clockwise by 90 degrees: local +X lays the lead
# outward along the trolley's long edge and local +Z points robot-right.
LOADED_DETONATOR_ORIENTATION = _matrix_to_quaternion_wxyz(
    (
        (HOLE_OUTWARD_XY[0], 0.0, ROBOT_RIGHT_XY[0]),
        (HOLE_OUTWARD_XY[1], 0.0, ROBOT_RIGHT_XY[1]),
        (0.0, -1.0, 0.0),
    )
)

for asset_path in (
    ENVIRONMENT,
    WORKTABLE,
    LOADED_DETONATOR_OBJECT,
    TRACKED_G1_ASSEMBLY,
):
    if not asset_path.is_file():
        raise FileNotFoundError(asset_path)


def zero_reward(env) -> torch.Tensor:
    return torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)


def never_terminate(env) -> torch.Tensor:
    return torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)


def spawn_wall_hole_environment(
    prim_path: str,
    cfg: UsdFileCfg,
    translation=None,
    orientation=None,
    **kwargs,
) -> Usd.Prim:
    root = spawn_from_usd(
        prim_path,
        cfg,
        translation=translation,
        orientation=orientation,
        **kwargs,
    )
    hole_body = None
    hole_collider = None
    ground = None
    scene_collision_root = None
    for prim in Usd.PrimRange(root):
        if prim.GetName() == "HollowCylinder":
            hole_body = prim
        elif (
            prim.GetName() == "Mesh"
            and hole_body is not None
            and prim.GetPath().HasPrefix(hole_body.GetPath())
        ):
            hole_collider = prim
        elif prim.GetName() == "GroundPlane":
            ground = prim
        elif prim.GetName() == "Xform" and prim != root:
            scene_collision_root = prim

    if hole_body is None or hole_collider is None:
        raise RuntimeError("Wall-hole scene is missing the fixed hollow collider")

    # The visual bore looked adequately larger than the detonator, but its
    # collision clearance was too small for the grasp/teleop jitter.  Move
    # only the vertices belonging to the inner cylindrical wall radially
    # outward.  The outer ring vertices are deliberately untouched, so the
    # collar's outside diameter and its placement do not change.
    hole_mesh = UsdGeom.Mesh(hole_collider)
    points_attr = hole_mesh.GetPointsAttr()
    points = points_attr.Get()
    radial_distances = [
        math.hypot(float(point[0]), float(point[1])) for point in points
    ]
    inner_radius = min(radial_distances)
    outer_radius = max(radial_distances)
    enlarged_inner_radius = min(
        inner_radius * HOLE_BORE_RADIUS_SCALE,
        outer_radius - HOLE_MIN_WALL_THICKNESS_LOCAL,
    )
    if enlarged_inner_radius <= inner_radius:
        raise RuntimeError("Wall-hole bore could not be enlarged safely")
    inner_vertex_threshold = 0.5 * (inner_radius + outer_radius)
    bore_scale = enlarged_inner_radius / inner_radius
    points_attr.Set(
        [
            Gf.Vec3f(
                point[0] * bore_scale,
                point[1] * bore_scale,
                point[2],
            )
            if math.hypot(float(point[0]), float(point[1]))
            < inner_vertex_threshold
            else point
            for point in points
        ]
    )

    # PhysX reported that ScaleOrientation on this kinematic rigid body is
    # unsupported.  The source target uses rotated, non-uniform scale on both
    # the rigid-body Xform and the collision Mesh.  Bake those two local
    # transforms into the mesh vertices and clear the transforms before
    # applying the rigid-body API.  This preserves the visible wall-hole
    # shape/center exactly while giving PhysX an unscaled rigid body and an
    # unscaled triangle-mesh collider, so its cooked collider matches what is
    # rendered.
    hole_body_xform = UsdGeom.Xformable(hole_body)
    hole_mesh_xform = UsdGeom.Xformable(hole_collider)
    baked_local_transform = (
        hole_mesh_xform.GetLocalTransformation()
        * hole_body_xform.GetLocalTransformation()
    )
    points_attr.Set(
        [
            Gf.Vec3f(
                baked_local_transform.Transform(
                    Gf.Vec3d(float(point[0]), float(point[1]), float(point[2]))
                )
            )
            for point in points_attr.Get()
        ]
    )
    hole_mesh_xform.ClearXformOpOrder()
    hole_body_xform.ClearXformOpOrder()

    rigid = UsdPhysics.RigidBodyAPI.Apply(hole_body)
    rigid.CreateRigidBodyEnabledAttr().Set(True)
    rigid.CreateKinematicEnabledAttr().Set(True)
    PhysxSchema.PhysxRigidBodyAPI.Apply(hole_body).CreateDisableGravityAttr().Set(
        True
    )

    collision = UsdPhysics.CollisionAPI.Apply(hole_collider)
    collision.CreateCollisionEnabledAttr().Set(True)
    mesh_collision = UsdPhysics.MeshCollisionAPI.Apply(hole_collider)
    # A convex decomposition cannot faithfully represent this hollow tube:
    # the convex pieces bridge the bore at their seams, producing invisible
    # centerline blockers.  The target is kinematic/static, so triangle-mesh
    # simplification is supported and preserves the continuous inner bore.
    mesh_collision.CreateApproximationAttr().Set("meshSimplification")
    physx_collision = PhysxSchema.PhysxCollisionAPI.Apply(hole_collider)
    physx_collision.CreateContactOffsetAttr().Set(INSERTION_CONTACT_OFFSET)
    physx_collision.CreateRestOffsetAttr().Set(0.0)

    # Match the target collar to the warm gray reconstructed wall instead of
    # retaining the source asset's dark brown material.
    hole_shader = root.GetStage().GetPrimAtPath(
        root.GetPath().AppendPath("leiguan/Looks/OmniPBR/Shader")
    )
    diffuse = hole_shader.GetAttribute("inputs:diffuse_color_constant")
    if not diffuse:
        raise RuntimeError("Wall-hole collider material shader is missing")
    diffuse.Set(Gf.Vec3f(0.50, 0.48, 0.44))
    hole_mesh.CreateDisplayColorAttr().Set(
        [Gf.Vec3f(0.50, 0.48, 0.44)]
    )

    if ground is None:
        raise RuntimeError("Wall-hole scene is missing collision-only GroundPlane")
    # The reconstructed scene is spawned with a 180-degree X rotation.  Put
    # the cube in the source frame at z=+0.10 m so its transformed top face is
    # exactly task-world z=0 (the robot and trolley floor datum).
    ground_xform = UsdGeom.Xformable(ground)
    translate_op = None
    scale_op = None
    for op in ground_xform.GetOrderedXformOps():
        if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
            translate_op = op
        elif op.GetOpType() == UsdGeom.XformOp.TypeScale:
            scale_op = op
    if translate_op is None:
        translate_op = ground_xform.AddTranslateOp()
    if scale_op is None:
        scale_op = ground_xform.AddScaleOp()
    translate_op.Set(Gf.Vec3d(0.0, 0.0, 0.10))
    scale_op.Set(Gf.Vec3d(100.0, 100.0, 0.10))
    UsdPhysics.CollisionAPI.Apply(ground).CreateCollisionEnabledAttr().Set(True)
    UsdGeom.Imageable(ground).MakeInvisible()
    UsdGeom.Imageable(ground).CreatePurposeAttr().Set(UsdGeom.Tokens.guide)

    if scene_collision_root is not None:
        UsdPhysics.CollisionAPI.Apply(
            scene_collision_root
        ).CreateCollisionEnabledAttr().Set(False)

    print(
        "[wall-hole-detonator] Environment verified: collision-only ground "
        "at task-world z=0, "
        "visual wall collision disabled, hollow target fixed kinematically, "
        f"with bore radius {inner_radius:.4f}->{enlarged_inner_radius:.4f} "
        "(outer radius unchanged), and colored to match the wall."
    )
    return root


def spawn_shifted_tracked_g1(
    prim_path: str,
    cfg: UsdFileCfg,
    translation=None,
    orientation=None,
    **kwargs,
) -> Usd.Prim:
    g1_upper = spawn_combined_tracked_g1(
        prim_path,
        cfg,
        translation=translation,
        orientation=orientation,
        **kwargs,
    )
    root = g1_upper.GetStage().GetPrimAtPath("/World/TrackedBase")
    if not root:
        raise RuntimeError("Tracked base root was not created")
    xform = UsdGeom.Xformable(root)
    translate_op = next(
        (op for op in xform.GetOrderedXformOps() if op.GetOpName() == "xformOp:translate"),
        None,
    )
    if translate_op is None:
        translate_op = xform.AddTranslateOp()
    translate_op.Set((TASK_TRACKED_BASE_X, TASK_TRACKED_BASE_Y, 0.0))
    orient_op = next(
        (op for op in xform.GetOrderedXformOps() if op.GetOpName() == "xformOp:orient"),
        None,
    )
    if orient_op is None:
        orient_op = xform.AddOrientOp()
    orient_op.Set(
        Gf.Quatd(
            math.cos(0.5 * TASK_TRACKED_BASE_YAW),
            Gf.Vec3d(0.0, 0.0, math.sin(0.5 * TASK_TRACKED_BASE_YAW)),
        )
    )

    carriage = root.GetStage().GetPrimAtPath("/World/TrackedBase/LiftCarriage")
    carriage_xform = UsdGeom.Xformable(carriage)
    carriage_translate = next(
        (
            op
            for op in carriage_xform.GetOrderedXformOps()
            if op.GetOpName() == "xformOp:translate"
        ),
        None,
    )
    if carriage_translate is None:
        carriage_translate = carriage_xform.AddTranslateOp()
    carriage_translate.Set((0.0, 0.0, TASK_TRACKED_LIFT_HEIGHT))

    g1_xform = UsdGeom.Xformable(g1_upper)
    g1_translate = next(
        (
            op
            for op in g1_xform.GetOrderedXformOps()
            if op.GetOpName() == "xformOp:translate"
        ),
        None,
    )
    if g1_translate is None:
        g1_translate = g1_xform.AddTranslateOp()
    g1_translate.Set(
        Gf.Vec3d(
            G1_MOUNT_LOCAL_POSITION[0],
            G1_MOUNT_LOCAL_POSITION[1],
            G1_MOUNT_LOCAL_POSITION[2] + TASK_TRACKED_LIFT_HEIGHT,
        )
    )
    print(
        "[wall-hole-detonator] G1 placed at captured base pose: "
        f"base=({TASK_TRACKED_BASE_X:.3f}, {TASK_TRACKED_BASE_Y:.3f}), "
        f"yaw={TASK_TRACKED_BASE_YAW:.3f}, lift={TASK_TRACKED_LIFT_HEIGHT:.3f}."
    )
    return g1_upper


def make_shifted_g1_upper_body_cfg():
    from tasks.g1_tasks.tabletop_grasp_g1_29dof_dex1.tabletop_grasp_g1_29dof_dex1_env_cfg import (
        make_embedded_g1_upper_body_cfg,
    )

    cfg = make_embedded_g1_upper_body_cfg()
    cfg.init_state.pos = TASK_G1_MOUNT_POSITION
    cfg.init_state.rot = TASK_G1_MOUNT_ORIENTATION
    cfg.spawn = UsdFileCfg(
        usd_path=str(TRACKED_G1_ASSEMBLY),
        func=spawn_shifted_tracked_g1,
    )
    return cfg


def spawn_loaded_detonator(
    prim_path: str,
    cfg: UsdFileCfg,
    translation=None,
    orientation=None,
    **kwargs,
) -> Usd.Prim:
    root = spawn_from_usd(
        prim_path,
        cfg,
        translation=translation,
        orientation=orientation,
        **kwargs,
    )
    rigid_count = 0
    collider_count = 0
    for prim in Usd.PrimRange(root):
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            physx = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
            physx.CreateEnableSpeculativeCCDAttr().Set(True)
            physx.CreateSolverPositionIterationCountAttr().Set(16)
            physx.CreateSolverVelocityIterationCountAttr().Set(4)
            physx.CreateMaxDepenetrationVelocityAttr().Set(1.0)
            rigid_count += 1
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            collision = PhysxSchema.PhysxCollisionAPI.Apply(prim)
            collision.CreateContactOffsetAttr().Set(INSERTION_CONTACT_OFFSET)
            collision.CreateRestOffsetAttr().Set(0.0)
            collider_count += 1
    if rigid_count != 27 or collider_count != 28:
        raise RuntimeError(
            "Unexpected loaded-detonator topology after spawn: "
            f"rigid={rigid_count}, colliders={collider_count}"
        )
    print(
        "[wall-hole-detonator] Loaded assembly configured: "
        f"{rigid_count} rigid bodies, {collider_count} colliders, "
        "speculative CCD enabled."
    )
    return root


@configclass
class WallHoleDetonatorSceneCfg(InteractiveSceneCfg):
    environment = AssetBaseCfg(
        prim_path="/World/envs/env_.*/Environment",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0),
            rot=(0.0, 1.0, 0.0, 0.0),
        ),
        spawn=UsdFileCfg(
            usd_path=str(ENVIRONMENT),
            func=spawn_wall_hole_environment,
        ),
    )

    worktable = AssetBaseCfg(
        prim_path="/World/envs/env_.*/Worktable",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=TABLE_POSITION,
            rot=WORKTABLE_ORIENTATION,
        ),
        spawn=UsdFileCfg(
            usd_path=str(WORKTABLE),
            func=spawn_static_worktable,
            scale=(WORKTABLE_XY_SCALE, WORKTABLE_XY_SCALE, WORKTABLE_Z_SCALE),
        ),
    )

    loaded_detonator = AssetBaseCfg(
        prim_path="/World/envs/env_.*/LoadedDetonator",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=LOADED_DETONATOR_POSITION,
            rot=LOADED_DETONATOR_ORIENTATION,
        ),
        spawn=UsdFileCfg(
            usd_path=str(LOADED_DETONATOR_OBJECT),
            func=spawn_loaded_detonator,
        ),
    )

    robot = make_shifted_g1_upper_body_cfg()

    front_camera = CameraPresets.g1_front_camera().replace(
        prim_path="/World/TrackedBase/G1UpperBody/d435_link/front_cam"
    )
    left_wrist_camera = CameraPresets.left_gripper_wrist_camera().replace(
        prim_path="/World/TrackedBase/G1UpperBody/left_hand_base_link/left_wrist_camera"
    )
    right_wrist_camera = CameraPresets.right_gripper_wrist_camera().replace(
        prim_path="/World/TrackedBase/G1UpperBody/right_hand_base_link/right_wrist_camera"
    )
    world_camera = CameraBaseCfg.get_camera_config(
        prim_path="/World/PerspectiveCamera",
        pos_offset=(-2.0, 4.15, 1.9),
        rot_offset=(-0.00617, 0.00617, 0.70708, -0.70708),
        focal_length=16.5,
    )
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(
            color=(0.9, 0.9, 0.9),
            intensity=1000.0,
        ),
    )


@configclass
class ActionsCfg:
    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        scale=1.0,
        use_default_offset=True,
    )


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        robot_joint_state = ObsTerm(func=get_robot_boy_joint_states)
        robot_gripper_state = ObsTerm(func=get_robot_gipper_joint_states)
        camera_image = ObsTerm(func=get_camera_image)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()


@configclass
class TerminationsCfg:
    task_end = DoneTerm(func=never_terminate)


@configclass
class RewardsCfg:
    task_reward = RewTerm(func=zero_reward, weight=0.0)


@configclass
class EventCfg:
    pass


def _runtime_controller(env):
    controller = getattr(env, "_wall_hole_detonator_runtime_controller", None)
    if controller is None:
        raise RuntimeError(
            "Wall-hole detonator runtime controller is not initialized"
        )
    return controller


def reset_task_objects(env) -> None:
    _runtime_controller(env).reset_task_objects()


def reset_everything(env) -> None:
    _runtime_controller(env).reset_everything()


@configclass
class WallHoleDetonatorG129Dex1EnvCfg(ManagerBasedRLEnvCfg):
    scene: WallHoleDetonatorSceneCfg = WallHoleDetonatorSceneCfg(
        num_envs=1,
        env_spacing=2.5,
        replicate_physics=True,
    )
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    rewards: RewardsCfg = RewardsCfg()
    events: EventCfg = EventCfg()
    commands = None
    curriculum = None

    def create_runtime_controller(self, env):
        from .wall_hole_detonator_controller import WallHoleDetonatorController

        controller = WallHoleDetonatorController(env)
        env._wall_hole_detonator_runtime_controller = controller
        return controller

    def __post_init__(self):
        self.decimation = 2
        self.episode_length_s = 1.0e9
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024
        self.sim.physx.friction_correlation_distance = 0.001
        self.sim.physx.gpu_constraint_solver_heavy_spring_enabled = True
        self.sim.physx.num_substeps = 4
        self.sim.physx.num_position_iterations = 16
        self.sim.physx.num_velocity_iterations = 8

        self.event_manager = SimpleEventManager()
        self.event_manager.register(
            "reset_object_self", SimpleEvent(func=reset_task_objects)
        )
        self.event_manager.register(
            "reset_all_self", SimpleEvent(func=reset_everything)
        )
