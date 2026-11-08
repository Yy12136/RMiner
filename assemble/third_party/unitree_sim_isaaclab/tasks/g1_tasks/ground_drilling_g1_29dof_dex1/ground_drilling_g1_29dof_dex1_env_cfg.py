"""Tracked G1 task for lifting a pneumatic drill and pressing it to ground."""

from __future__ import annotations

import math
from pathlib import Path

import torch
from pxr import Gf, PhysxSchema, Usd, UsdGeom, UsdPhysics

import isaaclab.envs.mdp as base_mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
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
    G1_FORWARD_YAW,
    G1_MOUNT_LOCAL_ORIENTATION,
    G1_MOUNT_LOCAL_POSITION,
    TRACKED_G1_ASSEMBLY,
    make_embedded_g1_upper_body_cfg,
    spawn_combined_tracked_g1,
)


ASSEMBLE_ROOT = Path(__file__).resolve().parents[5]
TASK_ROOT = ASSEMBLE_ROOT / "task" / "g1_ground_drilling_task"
TASK_ID = "Isaac-GroundDrilling-G129-Dex1-Joint"
DRILL_OBJECT = (
    ASSETS_ROOT
    / "object/PneumaticRockDrill/PneumaticRockDrill_Optimized.usd"
)

# Preserve Task 2's calibrated base pose, but lower its carriage for a
# floor-level grasp. Task 2 itself remains at +0.20 m.
TASK_TRACKED_BASE_X = -1.5534816454092946
TASK_TRACKED_BASE_Y = 2.0714633528394826
TASK_TRACKED_BASE_YAW = -0.5241686803753723
TASK_TRACKED_LIFT_HEIGHT = -0.60

DRILL_FORWARD_DISTANCE = 0.35
DRILL_RANDOM_XY_LIMIT = 0.05
DRILL_WORKING_TIP_LOCAL_Z = -0.25008774
DRILL_TIP_CLEARANCE = 0.001
# Viewed from the robot's left side, rotate the formerly upright drill 90
# degrees counter-clockwise: local -Z (the working end) then points along the
# robot's world-forward direction. The rotated object's lowest rendered point
# comes from its local minimum X extent.
DRILL_FORWARD_PITCH_RAD = -0.5 * math.pi
DRILL_SIDE_LOCAL_MIN_X = -0.14634959220886232
DRILL_ROOT_Z = -DRILL_SIDE_LOCAL_MIN_X + DRILL_TIP_CLEARANCE
DRILL_MASS = 1.0
DRILL_CENTER_OF_MASS = (0.0, 0.0, -0.12)
DRILL_COLLIDER_COUNT = 12
DRILL_CONTACT_OFFSET = 0.001

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
DRILL_POSITION = (
    TASK_G1_MOUNT_POSITION[0]
    + math.cos(G1_FORWARD_YAW) * DRILL_FORWARD_DISTANCE,
    TASK_G1_MOUNT_POSITION[1]
    + math.sin(G1_FORWARD_YAW) * DRILL_FORWARD_DISTANCE,
    DRILL_ROOT_Z,
)
DRILL_ORIENTATION = _multiply_quaternions_wxyz(
    (
        math.cos(0.5 * G1_FORWARD_YAW),
        0.0,
        0.0,
        math.sin(0.5 * G1_FORWARD_YAW),
    ),
    (
        math.cos(0.5 * DRILL_FORWARD_PITCH_RAD),
        0.0,
        math.sin(0.5 * DRILL_FORWARD_PITCH_RAD),
        0.0,
    ),
)

for asset_path in (ENVIRONMENT, DRILL_OBJECT, TRACKED_G1_ASSEMBLY):
    if not asset_path.is_file():
        raise FileNotFoundError(asset_path)


def zero_reward(env) -> torch.Tensor:
    return torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)


def never_terminate(env) -> torch.Tensor:
    return torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)


def spawn_ground_drilling_tracked_g1(
    prim_path: str,
    cfg: UsdFileCfg,
    translation=None,
    orientation=None,
    **kwargs,
) -> Usd.Prim:
    """Spawn the shared tracked G1 at this task's low carriage height."""

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
        raise RuntimeError("Tracked base hierarchy was not created")

    def set_op(prim, name, value):
        xform = UsdGeom.Xformable(prim)
        op = next(
            (item for item in xform.GetOrderedXformOps() if item.GetOpName() == name),
            None,
        )
        if op is None:
            op = xform.AddTranslateOp() if name == "xformOp:translate" else xform.AddOrientOp()
        op.Set(value)

    set_op(root, "xformOp:translate", (TASK_TRACKED_BASE_X, TASK_TRACKED_BASE_Y, 0.0))
    set_op(
        root,
        "xformOp:orient",
        Gf.Quatd(
            math.cos(0.5 * TASK_TRACKED_BASE_YAW),
            Gf.Vec3d(0.0, 0.0, math.sin(0.5 * TASK_TRACKED_BASE_YAW)),
        ),
    )
    set_op(
        carriage,
        "xformOp:translate",
        (0.0, 0.0, TASK_TRACKED_LIFT_HEIGHT),
    )
    set_op(
        g1_upper,
        "xformOp:translate",
        (
            G1_MOUNT_LOCAL_POSITION[0],
            G1_MOUNT_LOCAL_POSITION[1],
            G1_MOUNT_LOCAL_POSITION[2] + TASK_TRACKED_LIFT_HEIGHT,
        ),
    )
    print(
        "[ground-drilling] Tracked G1 initialized at Task 2 base pose; "
        f"floor-work lift={TASK_TRACKED_LIFT_HEIGHT:.2f} m."
    )
    return g1_upper


def make_ground_drilling_g1_cfg():
    cfg = make_embedded_g1_upper_body_cfg()
    cfg.init_state.pos = TASK_G1_MOUNT_POSITION
    cfg.init_state.rot = TASK_G1_MOUNT_ORIENTATION
    cfg.spawn.func = spawn_ground_drilling_tracked_g1
    hand_drive = cfg.actuators["hands"]
    hand_drive.stiffness = 400.0
    hand_drive.damping = 10.0
    hand_drive.effort_limit_sim = 40.0
    hand_drive.velocity_limit_sim = 0.20
    return cfg


def spawn_task_drill(
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
    tip = root.GetAttribute("drill:workingTipLocalZ")
    count = root.GetAttribute("drill:colliderCount")
    if not tip or abs(float(tip.Get()) - DRILL_WORKING_TIP_LOCAL_Z) > 1.0e-6:
        raise RuntimeError("Optimized drill working-tip metadata is invalid")
    if not count or int(count.Get()) != DRILL_COLLIDER_COUNT:
        raise RuntimeError("Optimized drill collider metadata is invalid")

    rigid = UsdPhysics.RigidBodyAPI.Apply(root)
    rigid.CreateRigidBodyEnabledAttr().Set(True)
    rigid.CreateKinematicEnabledAttr().Set(False)
    mass = UsdPhysics.MassAPI.Apply(root)
    mass.CreateMassAttr().Set(DRILL_MASS)
    mass.CreateCenterOfMassAttr().Set(Gf.Vec3f(*DRILL_CENTER_OF_MASS))
    physx = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    physx.CreateDisableGravityAttr().Set(False)
    physx.CreateLinearDampingAttr().Set(0.2)
    physx.CreateAngularDampingAttr().Set(0.5)
    physx.CreateEnableSpeculativeCCDAttr().Set(True)
    physx.CreateSolverPositionIterationCountAttr().Set(16)
    physx.CreateSolverVelocityIterationCountAttr().Set(4)
    physx.CreateMaxDepenetrationVelocityAttr().Set(1.0)

    enabled_colliders = 0
    for prim in Usd.PrimRange(root):
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        enabled = prim.GetAttribute("physics:collisionEnabled")
        if enabled and enabled.Get() is False:
            continue
        collision = PhysxSchema.PhysxCollisionAPI.Apply(prim)
        collision.CreateContactOffsetAttr().Set(DRILL_CONTACT_OFFSET)
        collision.CreateRestOffsetAttr().Set(0.0)
        enabled_colliders += 1
    if enabled_colliders != DRILL_COLLIDER_COUNT:
        raise RuntimeError(
            f"Expected {DRILL_COLLIDER_COUNT} drill colliders, found {enabled_colliders}"
        )

    material = root.GetStage().GetPrimAtPath(
        root.GetPath().AppendPath("PhysicsMaterials/HighGrip")
    )
    if material:
        physx_material = PhysxSchema.PhysxMaterialAPI.Apply(material)
        physx_material.CreateFrictionCombineModeAttr().Set("max")
        physx_material.CreateRestitutionCombineModeAttr().Set("min")
    print(
        "[ground-drilling] Drill configured: mass=1.00 kg, gravity on, "
        f"compound colliders={enabled_colliders}, CCD, solver=16/4."
    )
    return root


@configclass
class GroundDrillingSceneCfg(InteractiveSceneCfg):
    environment = AssetBaseCfg(
        prim_path="/World/envs/env_.*/Environment",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0),
            rot=(0.0, 1.0, 0.0, 0.0),
        ),
        spawn=UsdFileCfg(usd_path=str(ENVIRONMENT)),
    )
    drill = RigidObjectCfg(
        prim_path="/World/envs/env_.*/PneumaticRockDrill",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=DRILL_POSITION,
            rot=DRILL_ORIENTATION,
        ),
        spawn=UsdFileCfg(
            usd_path=str(DRILL_OBJECT),
            func=spawn_task_drill,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                kinematic_enabled=False,
                disable_gravity=False,
                linear_damping=0.2,
                angular_damping=0.5,
                max_depenetration_velocity=1.0,
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=4,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=DRILL_MASS),
            collision_props=sim_utils.CollisionPropertiesCfg(
                contact_offset=DRILL_CONTACT_OFFSET,
                rest_offset=0.0,
            ),
        ),
    )
    robot = make_ground_drilling_g1_cfg()

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
        pos_offset=(-2.0, 4.15, 1.6),
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
    reset_drill = EventTermCfg(
        func=base_mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("drill"),
        },
    )


def _runtime_controller(env):
    controller = getattr(env, "_ground_drilling_runtime_controller", None)
    if controller is None:
        raise RuntimeError("Ground-drilling runtime controller is not initialized")
    return controller


def reset_task_objects(env) -> None:
    _runtime_controller(env).reset_task_objects()


def reset_everything(env) -> None:
    _runtime_controller(env).reset_everything()


@configclass
class GroundDrillingG129Dex1EnvCfg(ManagerBasedRLEnvCfg):
    scene: GroundDrillingSceneCfg = GroundDrillingSceneCfg(
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
        from .ground_drilling_controller import GroundDrillingController

        controller = GroundDrillingController(env)
        env._ground_drilling_runtime_controller = controller
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
