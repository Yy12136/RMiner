"""Tracked G1 task for placing a randomized metal component into a bin."""

from pathlib import Path
import math

import torch
from pxr import Gf, PhysxSchema, Usd, UsdGeom, UsdPhysics

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
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
    ENVIRONMENT,
    G1_MOUNT_LOCAL_ORIENTATION,
    G1_MOUNT_LOCAL_POSITION,
    TRACKED_G1_ASSEMBLY,
    WORKTABLE_NATIVE_TOP_Z_CM,
    WORKTABLE_XY_SCALE,
    make_embedded_g1_upper_body_cfg,
    spawn_combined_tracked_g1,
    spawn_static_worktable,
)


ASSEMBLE_ROOT = Path(__file__).resolve().parents[5]
TASK_ROOT = ASSEMBLE_ROOT / "task" / "g1_bin_pick_place_task"
WORKTABLE = ASSETS_ROOT / "object/TableTrolley_B/TableTrolley_B01_01.usd"
BLUE_BIN_OBJECT = ASSETS_ROOT / "object/BlueSortingBin/blue_sorting_bin.usd"
TOTE_BIN_OBJECT = (
    ASSETS_ROOT
    / "object/Tote_B08/Tote_B08_60x40x15cm_PR_V_NVD_01.usd"
)
CONTAINER_BIN_OBJECT = (
    ASSETS_ROOT
    / "object/Container_F26/Container_F26_116x80x52cm_PR_V_NVD_01.usd"
)
SAM3D_NOZZLE_OBJECT = (
    ASSETS_ROOT
    / "object/SAM3DMetalNozzle/SAM3D_Metal_Nozzle_GraspReady.usd"
)
SAM3D_COMPONENT_ROOT = ASSETS_ROOT / "object/SAM3DRandomMetalComponents"
SAM3D_COMPONENT_OBJECTS = tuple(
    SAM3D_COMPONENT_ROOT / f"SAM3D_Metal_Component_{index:02d}_GraspReady.usd"
    for index in range(1, 5)
)

# Live-captured from the running bin-pick-place task on 2026-08-23. Keep the
# existing table fixed below so only the robot's reset pose follows this new
# calibration.
TASK_TRACKED_BASE_X = -1.7623459807346942
TASK_TRACKED_BASE_Y = 2.1922098299307424
TASK_TRACKED_BASE_YAW = -0.5241686803753723
TASK_TRACKED_LIFT_HEIGHT = 0.20
TASK_FORWARD_YAW = TASK_TRACKED_BASE_YAW + math.pi
TABLE_YAW = TASK_FORWARD_YAW - 0.5 * math.pi
WORKTABLE_YAW = TABLE_YAW - 0.5 * math.pi


def _yaw_quaternion(yaw: float) -> tuple[float, float, float, float]:
    return (math.cos(0.5 * yaw), 0.0, 0.0, math.sin(0.5 * yaw))


TABLE_ORIENTATION = _yaw_quaternion(TABLE_YAW)
WORKTABLE_ORIENTATION = _yaw_quaternion(WORKTABLE_YAW)
BIN_ORIENTATION = WORKTABLE_ORIENTATION
TOTE_BIN_ORIENTATION = TABLE_ORIENTATION
CONTAINER_BIN_ORIENTATION = TABLE_ORIENTATION
OBJECT_ORIENTATION = TABLE_ORIENTATION
TABLE_RIGHT_VECTOR = (math.cos(TABLE_YAW), math.sin(TABLE_YAW))
TASK_FORWARD_VECTOR = (
    math.cos(TASK_FORWARD_YAW),
    math.sin(TASK_FORWARD_YAW),
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


_base_cosine = math.cos(TASK_TRACKED_BASE_YAW)
_base_sine = math.sin(TASK_TRACKED_BASE_YAW)
TASK_G1_MOUNT_POSITION = (
    TASK_TRACKED_BASE_X
    + _base_cosine * G1_MOUNT_LOCAL_POSITION[0]
    - _base_sine * G1_MOUNT_LOCAL_POSITION[1],
    TASK_TRACKED_BASE_Y
    + _base_sine * G1_MOUNT_LOCAL_POSITION[0]
    + _base_cosine * G1_MOUNT_LOCAL_POSITION[1],
    G1_MOUNT_LOCAL_POSITION[2] + TASK_TRACKED_LIFT_HEIGHT,
)
TASK_G1_MOUNT_ORIENTATION = _multiply_quaternions_wxyz(
    _yaw_quaternion(TASK_TRACKED_BASE_YAW), G1_MOUNT_LOCAL_ORIENTATION
)

# Preserve the table's current world pose while adopting the newly captured
# robot reset. It is now approximately 0.509 m in front of the G1 mount.
TABLE_FORWARD_DISTANCE_M = 0.508745
TABLE_POSITION = (-3.415838607865756, 3.159638243885426, 0.0)


def _table_local_to_world(local_right: float, local_forward: float, z: float):
    return (
        TABLE_POSITION[0]
        + TABLE_RIGHT_VECTOR[0] * local_right
        + TASK_FORWARD_VECTOR[0] * local_forward,
        TABLE_POSITION[1]
        + TABLE_RIGHT_VECTOR[1] * local_right
        + TASK_FORWARD_VECTOR[1] * local_forward,
        z,
    )


TABLE_TOP_HEIGHT_M = 1.09
WORKTABLE_Z_SCALE = TABLE_TOP_HEIGHT_M / WORKTABLE_NATIVE_TOP_Z_CM
BIN_SCALE = 1.30
TOTE_BIN_SCALE = 0.005
CONTAINER_BIN_SCALE = 0.0035
# Shift both randomized task regions 4 cm toward the robot's right arm. Reset
# offsets are sampled around these centers, so every randomized pose follows
# the same task-frame translation.
BIN_POSITION = _table_local_to_world(-0.18, -0.12, TABLE_TOP_HEIGHT_M + 0.002)
BIN_PARKING_POSITION = (BIN_POSITION[0], BIN_POSITION[1], -10.0)

# Physical dimensions below are in the composed task stage. The two migrated
# DigitalTwin assets are authored in centimeters, hence the 0.005/0.0035
# scales. All three openings comfortably contain the 0.22 x 0.073 m nozzle,
# while retaining visibly different sizes for reset-time data augmentation.
BIN_VARIANTS = (
    {
        "name": "blue_sorting_bin",
        "prim_path": "/World/envs/env_0/BlueSortingBin",
        "orientation": BIN_ORIENTATION,
        "inner_half_x_m": 0.115,
        "inner_half_y_m": 0.148,
        "floor_top_m": 0.002,
        "rim_height_m": 0.05046751761 * BIN_SCALE,
        "outer_size_m": (0.2605, 0.3250, 0.0656),
    },
    {
        "name": "tote_b08",
        "prim_path": "/World/envs/env_0/ToteB08",
        "orientation": TOTE_BIN_ORIENTATION,
        "inner_half_x_m": 0.135,
        "inner_half_y_m": 0.085,
        "floor_top_m": 0.002,
        "rim_height_m": 0.075,
        "outer_size_m": (0.300, 0.200, 0.075),
    },
    {
        "name": "container_f26",
        "prim_path": "/World/envs/env_0/ContainerF26",
        "orientation": CONTAINER_BIN_ORIENTATION,
        "inner_half_x_m": 0.180,
        "inner_half_y_m": 0.115,
        "floor_top_m": 0.050,
        "rim_height_m": 0.1827,
        "outer_size_m": (0.4054, 0.2785, 0.1827),
    },
)

# Muted industrial colors commonly encountered on underground storage and
# transport equipment. Albedo textures are desaturated before tinting so the
# original surface detail/normal/roughness remains visible without leaking the
# source asset's blue or gray hue into the selected color.
BIN_COLOR_PALETTE = (
    {"name": "industrial_blue", "rgb": (0.18, 0.42, 0.78)},
    {"name": "safety_yellow", "rgb": (0.95, 0.72, 0.10)},
    {"name": "warning_orange", "rgb": (0.92, 0.34, 0.08)},
    {"name": "graphite_gray", "rgb": (0.32, 0.35, 0.38)},
    {"name": "equipment_green", "rgb": (0.20, 0.50, 0.30)},
)
OBJECT_SURFACE_CLEARANCE_M = 0.0005
# Move every randomized grasp object 0.10 m toward the robot while preserving
# its left/right placement and existing +/-5 cm reset randomization.
OBJECT_BASE_POSITION = _table_local_to_world(0.29, -0.22, 0.0)
OBJECT_VARIANTS = (
    {
        "name": "metal_nozzle",
        "scene_key": "task_object_nozzle",
        "prim_name": "SAM3DMetalNozzle",
        "prim_path": "/World/envs/env_0/SAM3DMetalNozzle",
        "usd_path": SAM3D_NOZZLE_OBJECT,
        "dimensions_m": (0.220, 0.07300438732, 0.07531812042),
    },
    {
        "name": "metal_component_01",
        "scene_key": "task_object_component_01",
        "prim_name": "SAM3DMetalComponent01",
        "prim_path": "/World/envs/env_0/SAM3DMetalComponent01",
        "usd_path": SAM3D_COMPONENT_OBJECTS[0],
        "dimensions_m": (0.17939721, 0.07931019, 0.08000000),
    },
    {
        "name": "metal_component_02",
        "scene_key": "task_object_component_02",
        "prim_name": "SAM3DMetalComponent02",
        "prim_path": "/World/envs/env_0/SAM3DMetalComponent02",
        "usd_path": SAM3D_COMPONENT_OBJECTS[1],
        "dimensions_m": (0.22000000, 0.07942990, 0.07932955),
    },
    {
        "name": "metal_component_03",
        "scene_key": "task_object_component_03",
        "prim_name": "SAM3DMetalComponent03",
        "prim_path": "/World/envs/env_0/SAM3DMetalComponent03",
        "usd_path": SAM3D_COMPONENT_OBJECTS[2],
        "dimensions_m": (0.22000000, 0.07172577, 0.07300301),
    },
    {
        "name": "metal_component_04",
        "scene_key": "task_object_component_04",
        "prim_name": "SAM3DMetalComponent04",
        "prim_path": "/World/envs/env_0/SAM3DMetalComponent04",
        "usd_path": SAM3D_COMPONENT_OBJECTS[3],
        "dimensions_m": (0.21866709, 0.08000000, 0.07975534),
    },
)


def _object_table_position(dimensions_m):
    return (
        OBJECT_BASE_POSITION[0],
        OBJECT_BASE_POSITION[1],
        TABLE_TOP_HEIGHT_M
        + 0.5 * float(dimensions_m[2])
        + OBJECT_SURFACE_CLEARANCE_M,
    )


for index, variant in enumerate(OBJECT_VARIANTS):
    variant["table_position"] = _object_table_position(variant["dimensions_m"])
    variant["parking_position"] = (
        20.0 + 2.0 * index,
        20.0,
        2.0,
    )

OBJECT_MASS_KG = 0.25
OBJECT_CONTACT_OFFSET_M = 0.0010
BIN_CONTACT_OFFSET_M = 0.0010
BIN_FLOOR_COLLIDER_THICKNESS_M = 0.012
BIN_COMPOUND_COLLIDER_COUNT = 5
BIN_COLLIDER_ROOT_SUFFIX = "_TaskCollision"
OBJECT_LINEAR_DAMPING = 0.8
OBJECT_ANGULAR_DAMPING = 1.5
LEFT_ARM_SIDE_RAISE_Q = {
    "left_shoulder_pitch_joint": 0.15,
    "left_shoulder_roll_joint": 0.675,
    "left_shoulder_yaw_joint": 1.50,
    "left_elbow_joint": 0.0,
    "left_wrist_roll_joint": 0.0,
    "left_wrist_pitch_joint": 0.0,
    "left_wrist_yaw_joint": 0.0,
}

for asset_path in (
    ENVIRONMENT,
    WORKTABLE,
    BLUE_BIN_OBJECT,
    TOTE_BIN_OBJECT,
    CONTAINER_BIN_OBJECT,
    SAM3D_NOZZLE_OBJECT,
    *SAM3D_COMPONENT_OBJECTS,
    TRACKED_G1_ASSEMBLY,
):
    if not asset_path.is_file():
        raise FileNotFoundError(asset_path)


def zero_reward(env) -> torch.Tensor:
    return torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)


def never_terminate(env) -> torch.Tensor:
    return torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)


def spawn_task_tracked_g1(
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
    stage = g1_upper.GetStage()
    root = stage.GetPrimAtPath("/World/TrackedBase")
    carriage = stage.GetPrimAtPath("/World/TrackedBase/LiftCarriage")
    if not root or not carriage:
        raise RuntimeError("Tracked base or lift carriage was not created")

    root_xform = UsdGeom.Xformable(root)
    translate = next(
        (
            op
            for op in root_xform.GetOrderedXformOps()
            if op.GetOpName() == "xformOp:translate"
        ),
        None,
    ) or root_xform.AddTranslateOp()
    orient = next(
        (
            op
            for op in root_xform.GetOrderedXformOps()
            if op.GetOpName() == "xformOp:orient"
        ),
        None,
    ) or root_xform.AddOrientOp()
    translate.Set((TASK_TRACKED_BASE_X, TASK_TRACKED_BASE_Y, 0.0))
    orient.Set(
        Gf.Quatd(
            math.cos(0.5 * TASK_TRACKED_BASE_YAW),
            Gf.Vec3d(0.0, 0.0, math.sin(0.5 * TASK_TRACKED_BASE_YAW)),
        )
    )

    carriage_xform = UsdGeom.Xformable(carriage)
    carriage_translate = next(
        (
            op
            for op in carriage_xform.GetOrderedXformOps()
            if op.GetOpName() == "xformOp:translate"
        ),
        None,
    ) or carriage_xform.AddTranslateOp()
    carriage_translate.Set((0.0, 0.0, TASK_TRACKED_LIFT_HEIGHT))

    g1_xform = UsdGeom.Xformable(g1_upper)
    g1_translate = next(
        (
            op
            for op in g1_xform.GetOrderedXformOps()
            if op.GetOpName() == "xformOp:translate"
        ),
        None,
    ) or g1_xform.AddTranslateOp()
    g1_translate.Set(
        (
            G1_MOUNT_LOCAL_POSITION[0],
            G1_MOUNT_LOCAL_POSITION[1],
            G1_MOUNT_LOCAL_POSITION[2] + TASK_TRACKED_LIFT_HEIGHT,
        )
    )
    print(
        "[bin-pick-place] Robot reset pose applied: "
        f"base=({TASK_TRACKED_BASE_X:.6f}, {TASK_TRACKED_BASE_Y:.6f}), "
        f"yaw={TASK_TRACKED_BASE_YAW:.6f}, lift={TASK_TRACKED_LIFT_HEIGHT:.3f}."
    )
    return g1_upper


def make_task_g1_cfg():
    cfg = make_embedded_g1_upper_body_cfg()
    cfg.init_state.pos = TASK_G1_MOUNT_POSITION
    cfg.init_state.rot = TASK_G1_MOUNT_ORIENTATION
    cfg.spawn.func = spawn_task_tracked_g1
    hand_drive = cfg.actuators["hands"]
    hand_drive.stiffness = 400.0
    hand_drive.damping = 10.0
    hand_drive.effort_limit_sim = 40.0
    hand_drive.velocity_limit_sim = 0.20
    cfg.init_state.joint_pos.update(LEFT_ARM_SIDE_RAISE_Q)
    return cfg


def _set_collision_offsets(prim: Usd.Prim, contact_offset: float) -> None:
    api = PhysxSchema.PhysxCollisionAPI.Apply(prim)
    api.CreateContactOffsetAttr().Set(contact_offset)
    api.CreateRestOffsetAttr().Set(0.0)


def _bin_variant_for_prim_path(prim_path: str):
    prim_name = str(prim_path).rstrip("/").rsplit("/", 1)[-1]
    for variant in BIN_VARIANTS:
        if variant["prim_path"].rsplit("/", 1)[-1] == prim_name:
            return variant
    raise RuntimeError(f"No compound-collider specification for {prim_path}")


def _create_static_box_collider(
    stage: Usd.Stage,
    prim_path: str,
    center_m,
    size_m,
) -> Usd.Prim:
    """Create an invisible collision cube authored directly in stage metres."""
    cube = UsdGeom.Cube.Define(stage, prim_path)
    cube.CreateSizeAttr(1.0)
    imageable = UsdGeom.Imageable(cube.GetPrim())
    imageable.CreateVisibilityAttr().Set(UsdGeom.Tokens.invisible)

    xformable = UsdGeom.Xformable(cube.GetPrim())
    xformable.ClearXformOpOrder()
    xformable.AddTranslateOp().Set(Gf.Vec3d(*(float(value) for value in center_m)))
    xformable.AddScaleOp().Set(Gf.Vec3f(*(float(value) for value in size_m)))

    collider = cube.GetPrim()
    UsdPhysics.CollisionAPI.Apply(collider).CreateCollisionEnabledAttr().Set(True)
    _set_collision_offsets(collider, BIN_CONTACT_OFFSET_M)
    return collider


def _create_open_bin_compound_colliders(
    root: Usd.Prim,
    variant,
) -> tuple[Usd.Prim, list[Usd.Prim]]:
    """Build an independently transformed floor plus four walls in metres.

    The migrated bins use three different authoring units and root scales.  A
    collider parented below those scaled references can be displayed by the
    debug renderer yet still be cooked at an unexpected transform.  Keeping
    the task collision rig as an unscaled sibling removes that ambiguity.
    """
    outer_x, outer_y, outer_z = (
        float(value) for value in variant["outer_size_m"]
    )
    inner_x = 2.0 * float(variant["inner_half_x_m"])
    inner_y = 2.0 * float(variant["inner_half_y_m"])
    wall_x = 0.5 * (outer_x - inner_x)
    wall_y = 0.5 * (outer_y - inner_y)
    if min(wall_x, wall_y) <= 0.0:
        raise RuntimeError(
            f"Invalid open-bin dimensions for {variant['name']}: "
            f"outer={variant['outer_size_m']}, inner=({inner_x}, {inner_y})"
        )

    floor_top = float(variant["floor_top_m"])
    floor_thickness = BIN_FLOOR_COLLIDER_THICKNESS_M
    wall_center_z = 0.5 * outer_z
    collider_root_path = f"{root.GetPath()}{BIN_COLLIDER_ROOT_SUFFIX}"
    collider_root = UsdGeom.Xform.Define(root.GetStage(), collider_root_path)
    collider_xform = UsdGeom.Xformable(collider_root.GetPrim())
    collider_xform.ClearXformOpOrder()
    collider_xform.AddTranslateOp().Set(Gf.Vec3d(*BIN_POSITION))
    orientation = variant["orientation"]
    collider_xform.AddOrientOp().Set(
        Gf.Quatf(
            float(orientation[0]),
            Gf.Vec3f(
                float(orientation[1]),
                float(orientation[2]),
                float(orientation[3]),
            ),
        )
    )

    # A kinematic rigid-body root makes reset-time bin translations explicit
    # to PhysX. It remains fixed during normal simulation and is not affected
    # by gravity or contacts.
    rigid = UsdPhysics.RigidBodyAPI.Apply(collider_root.GetPrim())
    rigid.CreateRigidBodyEnabledAttr().Set(True)
    rigid.CreateKinematicEnabledAttr().Set(True)
    specs = (
        (
            "Floor",
            (0.0, 0.0, floor_top - 0.5 * floor_thickness),
            (outer_x, outer_y, floor_thickness),
        ),
        (
            "WallNegativeX",
            (-0.25 * (outer_x + inner_x), 0.0, wall_center_z),
            (wall_x, outer_y, outer_z),
        ),
        (
            "WallPositiveX",
            (0.25 * (outer_x + inner_x), 0.0, wall_center_z),
            (wall_x, outer_y, outer_z),
        ),
        (
            "WallNegativeY",
            (0.0, -0.25 * (outer_y + inner_y), wall_center_z),
            (outer_x, wall_y, outer_z),
        ),
        (
            "WallPositiveY",
            (0.0, 0.25 * (outer_y + inner_y), wall_center_z),
            (outer_x, wall_y, outer_z),
        ),
    )
    colliders = [
        _create_static_box_collider(
            root.GetStage(),
            f"{collider_root_path}/{name}",
            center,
            size,
        )
        for name, center, size in specs
    ]
    floor = UsdGeom.Cube(colliders[0])
    floor_size = floor.GetPrim().GetAttribute("xformOp:scale").Get()
    floor_center = floor.GetPrim().GetAttribute("xformOp:translate").Get()
    composed_floor_top = float(floor_center[2]) + 0.5 * float(floor_size[2])
    if not math.isclose(
        composed_floor_top, floor_top, rel_tol=0.0, abs_tol=1.0e-9
    ):
        raise RuntimeError(
            f"Floor collider height mismatch for {variant['name']}: "
            f"expected {floor_top}, got {composed_floor_top}"
        )
    return collider_root.GetPrim(), colliders


def spawn_static_bin(
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
    variant = _bin_variant_for_prim_path(prim_path)
    root_scale_values = tuple(float(value) for value in cfg.scale)
    if not all(
        math.isclose(value, root_scale_values[0], rel_tol=0.0, abs_tol=1.0e-9)
        for value in root_scale_values[1:]
    ):
        raise RuntimeError(
            f"Bin compound colliders require uniform scale, got {cfg.scale}"
        )

    disabled_source_colliders = 0
    removed_bodies = 0
    for prim in Usd.PrimRange(root):
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            prim.RemoveAPI(UsdPhysics.RigidBodyAPI)
            removed_bodies += 1
        if prim.HasAPI(UsdPhysics.MassAPI):
            prim.RemoveAPI(UsdPhysics.MassAPI)
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            UsdPhysics.CollisionAPI(prim).CreateCollisionEnabledAttr().Set(False)
            disabled_source_colliders += 1

    collider_root, compound_colliders = _create_open_bin_compound_colliders(
        root, variant
    )
    if len(compound_colliders) != BIN_COMPOUND_COLLIDER_COUNT:
        raise RuntimeError(
            f"Expected {BIN_COMPOUND_COLLIDER_COUNT} compound colliders, "
            f"found {len(compound_colliders)}"
        )
    print(
        "[bin-pick-place] Static open-bin compound collision: "
        f"variant={variant['name']}, scale={cfg.scale}, "
        f"collision_root={collider_root.GetPath()}, "
        f"boxes={len(compound_colliders)}, "
        f"disabled source colliders={disabled_source_colliders}, "
        f"removed rigid bodies={removed_bodies}."
    )
    return root


def spawn_task_object(
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
    rigid = UsdPhysics.RigidBodyAPI.Apply(root)
    rigid.CreateRigidBodyEnabledAttr().Set(True)
    rigid.CreateKinematicEnabledAttr().Set(False)
    mass = UsdPhysics.MassAPI.Apply(root)
    mass.CreateMassAttr().Set(OBJECT_MASS_KG)
    physx = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    physx.CreateDisableGravityAttr().Set(False)
    physx.CreateLinearDampingAttr().Set(OBJECT_LINEAR_DAMPING)
    physx.CreateAngularDampingAttr().Set(OBJECT_ANGULAR_DAMPING)
    physx.CreateEnableSpeculativeCCDAttr().Set(True)
    physx.CreateSolverPositionIterationCountAttr().Set(16)
    physx.CreateSolverVelocityIterationCountAttr().Set(4)
    physx.CreateMaxDepenetrationVelocityAttr().Set(1.0)
    collider_count = 0
    for prim in Usd.PrimRange(root):
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            _set_collision_offsets(prim, OBJECT_CONTACT_OFFSET_M)
            collider_count += 1
    if collider_count != 1:
        raise RuntimeError(
            f"Expected one SAM3D object collider, found {collider_count}"
        )
    print(
        f"[bin-pick-place] SAM3D object physics ready at {prim_path}: "
        f"mass=0.25 kg, native gravity, damping="
        f"{OBJECT_LINEAR_DAMPING}/{OBJECT_ANGULAR_DAMPING}, "
        "speculative CCD, solver=16/4, convex-hull collider."
    )
    return root


def _make_task_object_cfg(variant) -> RigidObjectCfg:
    return RigidObjectCfg(
        prim_path=f"/World/envs/env_.*/{variant['prim_name']}",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=variant["parking_position"],
            rot=OBJECT_ORIENTATION,
        ),
        spawn=UsdFileCfg(
            usd_path=str(variant["usd_path"]),
            func=spawn_task_object,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                disable_gravity=False,
                linear_damping=OBJECT_LINEAR_DAMPING,
                angular_damping=OBJECT_ANGULAR_DAMPING,
                max_linear_velocity=2.0,
                max_angular_velocity=360.0,
                max_depenetration_velocity=1.0,
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=4,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=OBJECT_MASS_KG),
            collision_props=sim_utils.CollisionPropertiesCfg(
                contact_offset=OBJECT_CONTACT_OFFSET_M,
                rest_offset=0.0,
            ),
        ),
    )


@configclass
class BinPickPlaceSceneCfg(InteractiveSceneCfg):
    environment = AssetBaseCfg(
        prim_path="/World/envs/env_.*/Environment",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0),
            rot=(0.0, 1.0, 0.0, 0.0),
        ),
        spawn=UsdFileCfg(usd_path=str(ENVIRONMENT)),
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

    # Hidden support for the four inactive randomized rigid bodies. Keeping
    # native gravity enabled on every variant avoids runtime PhysX gravity
    # toggles; parked bodies simply settle here, far outside the task scene.
    object_parking_shelf = AssetBaseCfg(
        prim_path="/World/envs/env_.*/ObjectParkingShelf",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(24.0, 20.0, 1.91)),
        spawn=sim_utils.CuboidCfg(
            size=(12.0, 2.0, 0.10),
            visible=False,
            collision_props=sim_utils.CollisionPropertiesCfg(
                contact_offset=0.001,
                rest_offset=0.0,
            ),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.0,
                dynamic_friction=0.8,
                restitution=0.0,
            ),
        ),
    )

    blue_bin = AssetBaseCfg(
        prim_path="/World/envs/env_.*/BlueSortingBin",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=BIN_POSITION,
            rot=BIN_ORIENTATION,
        ),
        spawn=UsdFileCfg(
            usd_path=str(BLUE_BIN_OBJECT),
            func=spawn_static_bin,
            scale=(BIN_SCALE, BIN_SCALE, BIN_SCALE),
        ),
    )

    tote_bin = AssetBaseCfg(
        prim_path="/World/envs/env_.*/ToteB08",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=BIN_PARKING_POSITION,
            rot=TOTE_BIN_ORIENTATION,
        ),
        spawn=UsdFileCfg(
            usd_path=str(TOTE_BIN_OBJECT),
            func=spawn_static_bin,
            scale=(TOTE_BIN_SCALE, TOTE_BIN_SCALE, TOTE_BIN_SCALE),
        ),
    )

    container_bin = AssetBaseCfg(
        prim_path="/World/envs/env_.*/ContainerF26",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=BIN_PARKING_POSITION,
            rot=CONTAINER_BIN_ORIENTATION,
        ),
        spawn=UsdFileCfg(
            usd_path=str(CONTAINER_BIN_OBJECT),
            func=spawn_static_bin,
            scale=(
                CONTAINER_BIN_SCALE,
                CONTAINER_BIN_SCALE,
                CONTAINER_BIN_SCALE,
            ),
        ),
    )

    task_object_nozzle = _make_task_object_cfg(OBJECT_VARIANTS[0])
    task_object_component_01 = _make_task_object_cfg(OBJECT_VARIANTS[1])
    task_object_component_02 = _make_task_object_cfg(OBJECT_VARIANTS[2])
    task_object_component_03 = _make_task_object_cfg(OBJECT_VARIANTS[3])
    task_object_component_04 = _make_task_object_cfg(OBJECT_VARIANTS[4])

    robot = make_task_g1_cfg()

    front_camera = CameraPresets.g1_front_camera().replace(
        prim_path="/World/TrackedBase/G1UpperBody/d435_link/front_cam"
    )
    left_wrist_camera = CameraPresets.left_gripper_wrist_camera().replace(
        prim_path=(
            "/World/TrackedBase/G1UpperBody/left_hand_base_link/left_wrist_camera"
        )
    )
    right_wrist_camera = CameraPresets.right_gripper_wrist_camera().replace(
        prim_path=(
            "/World/TrackedBase/G1UpperBody/right_hand_base_link/right_wrist_camera"
        )
    )
    world_camera = CameraBaseCfg.get_camera_config(
        prim_path="/World/PerspectiveCamera",
        pos_offset=(-2.75, 3.25, 1.95),
        rot_offset=(-0.00617, 0.00617, 0.70708, -0.70708),
        focal_length=16.5,
    )
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(
            color=(0.75, 0.75, 0.75),
            intensity=1200.0,
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
    # Runtime reset callbacks select and place all five object variants after
    # IsaacLab has restored their default parked states.
    pass


def _runtime_controller(env):
    controller = getattr(env, "_bin_pick_place_runtime_controller", None)
    if controller is None:
        raise RuntimeError("Bin pick-place runtime controller is not initialized")
    return controller


def reset_task_object(env) -> None:
    _runtime_controller(env).reset_task_object()


def reset_everything(env) -> None:
    _runtime_controller(env).reset_everything()


@configclass
class BinPickPlaceG129Dex1EnvCfg(ManagerBasedRLEnvCfg):
    scene: BinPickPlaceSceneCfg = BinPickPlaceSceneCfg(
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
        from .bin_pick_place_controller import BinPickPlaceController

        controller = BinPickPlaceController(env)
        env._bin_pick_place_runtime_controller = controller
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

        # sim_main routes the Quest recorder's object/all reset categories
        # through these task-specific callbacks.
        self.event_manager = SimpleEventManager()
        self.event_manager.register(
            "reset_object_self", SimpleEvent(func=reset_task_object)
        )
        self.event_manager.register(
            "reset_all_self", SimpleEvent(func=reset_everything)
        )
