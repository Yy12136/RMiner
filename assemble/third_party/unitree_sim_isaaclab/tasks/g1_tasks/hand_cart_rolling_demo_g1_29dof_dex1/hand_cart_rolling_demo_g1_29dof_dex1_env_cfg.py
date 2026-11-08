"""Kinematic teleoperation scene for the articulated rolling hand-cart asset.

The cart is deliberately an animation prop in this demo: the tracked base moves
it by the identical planar displacement and the two wheels are rotated from the
actual left/right track travel.  Cart contacts and its authored articulation are
disabled here, so this task does not claim that the robot physically pushes it.
"""

from pathlib import Path
import math
import os

import torch
from pxr import Gf, Usd, UsdGeom, UsdPhysics

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
    ENVIRONMENT,
    G1_MOUNT_LOCAL_ORIENTATION,
    G1_MOUNT_LOCAL_POSITION,
    TRACKED_G1_ASSEMBLY,
    make_embedded_g1_upper_body_cfg,
    spawn_combined_tracked_g1,
)


ASSEMBLE_ROOT = Path(__file__).resolve().parents[5]
TASK_ROOT = ASSEMBLE_ROOT / "task" / "g1_hand_cart_rolling_demo_task"
HAND_CART_OBJECT = ASSEMBLE_ROOT / "objects" / "two_wheel_steel_hand_cart_original_handles_only.usd"
TASK_ID = "Isaac-HandCartRollingDemo-G129-Dex1-Joint"

# The demo intentionally uses a slower speed than the push-evaluation task.
DEMO_MAX_TRACK_SPEED_MPS = float(os.environ.get("G1_HAND_CART_DEMO_MAX_SPEED_MPS", "0.18"))
DEMO_MAX_TRACK_ACCEL_MPS2 = float(os.environ.get("G1_HAND_CART_DEMO_MAX_ACCEL_MPS2", "0.30"))
WHEEL_RADIUS_M = 0.1954552055698982

REFERENCE_BASE_X = -1.5534816454092946
REFERENCE_BASE_Y = 2.0714633528394826
TASK_TRACKED_BASE_YAW = -0.5241686803753723
ROBOT_BACK_OFFSET_M = float(os.environ.get("G1_HAND_CART_ROBOT_BACK_OFFSET_M", "1.8"))
TASK_FORWARD_YAW = TASK_TRACKED_BASE_YAW + math.pi
TASK_FORWARD_X = math.cos(TASK_FORWARD_YAW)
TASK_FORWARD_Y = math.sin(TASK_FORWARD_YAW)
TASK_TRACKED_BASE_X = REFERENCE_BASE_X - TASK_FORWARD_X * ROBOT_BACK_OFFSET_M
TASK_TRACKED_BASE_Y = REFERENCE_BASE_Y - TASK_FORWARD_Y * ROBOT_BACK_OFFSET_M
TASK_TRACKED_LIFT_HEIGHT = float(os.environ.get("G1_HAND_CART_INITIAL_LIFT_M", "-0.45"))


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
    TASK_TRACKED_BASE_X + _base_cosine * G1_MOUNT_LOCAL_POSITION[0] - _base_sine * G1_MOUNT_LOCAL_POSITION[1],
    TASK_TRACKED_BASE_Y + _base_sine * G1_MOUNT_LOCAL_POSITION[0] + _base_cosine * G1_MOUNT_LOCAL_POSITION[1],
    G1_MOUNT_LOCAL_POSITION[2] + TASK_TRACKED_LIFT_HEIGHT,
)
TASK_G1_MOUNT_ORIENTATION = _multiply_quaternions_wxyz(
    (math.cos(0.5 * TASK_TRACKED_BASE_YAW), 0.0, 0.0, math.sin(0.5 * TASK_TRACKED_BASE_YAW)),
    G1_MOUNT_LOCAL_ORIENTATION,
)

CART_DISTANCE_FROM_G1_M = float(os.environ.get("G1_HAND_CART_DISTANCE_M", "0.82"))
CART_POSITION = (
    TASK_G1_MOUNT_POSITION[0] + TASK_FORWARD_X * CART_DISTANCE_FROM_G1_M,
    TASK_G1_MOUNT_POSITION[1] + TASK_FORWARD_Y * CART_DISTANCE_FROM_G1_M,
    0.2930,
)
_cart_world_yaw = TASK_FORWARD_YAW + 0.5 * math.pi
CART_ORIENTATION = _multiply_quaternions_wxyz(
    (math.cos(0.5 * _cart_world_yaw), 0.0, 0.0, math.sin(0.5 * _cart_world_yaw)),
    (math.sqrt(0.5), math.sqrt(0.5), 0.0, 0.0),
)

for asset_path in (ENVIRONMENT, TRACKED_G1_ASSEMBLY, HAND_CART_OBJECT):
    if not asset_path.is_file():
        raise FileNotFoundError(asset_path)
if DEMO_MAX_TRACK_SPEED_MPS <= 0.0 or DEMO_MAX_TRACK_ACCEL_MPS2 <= 0.0:
    raise ValueError("G1_HAND_CART_DEMO_MAX_SPEED_MPS and acceleration must be positive")


def zero_reward(env) -> torch.Tensor:
    return torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)


def never_terminate(env) -> torch.Tensor:
    return torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)


def _get_or_add_op(xform: UsdGeom.Xformable, kind: str):
    name = f"xformOp:{kind}"
    found = next((op for op in xform.GetOrderedXformOps() if op.GetOpName() == name), None)
    if found is not None:
        return found
    if kind == "translate":
        return xform.AddTranslateOp()
    if kind == "orient":
        return xform.AddOrientOp(UsdGeom.XformOp.PrecisionDouble)
    raise ValueError(kind)


def spawn_rolling_demo_cart(prim_path: str, cfg: UsdFileCfg, translation=None, orientation=None, **kwargs) -> Usd.Prim:
    """Spawn the handles-only cart and enable its analytic handle colliders."""

    root = spawn_from_usd(prim_path, cfg, translation=translation, orientation=orientation, **kwargs)
    root_xform = UsdGeom.Xformable(root)
    _get_or_add_op(root_xform, "translate").Set(Gf.Vec3d(*(translation or CART_POSITION)))
    q = orientation or CART_ORIENTATION
    _get_or_add_op(root_xform, "orient").Set(Gf.Quatd(float(q[0]), Gf.Vec3d(*q[1:])))

    # The asset already authors one kinematic rigid body at its root.  Avoid
    # adding or removing rigid-body schemas after spawn because PhysX may have
    # created tensor views from the referenced USD by this point.
    if not root.HasAPI(UsdPhysics.RigidBodyAPI):
        raise RuntimeError("Handles-only hand cart must author a rigid body on its root")
    body = UsdPhysics.RigidBodyAPI(root)
    if not body.GetRigidBodyEnabledAttr().Get() or not body.GetKinematicEnabledAttr().Get():
        raise RuntimeError("Handles-only hand cart root must be an enabled kinematic body")
    collider_count = 0
    enabled_handle_count = 0
    for prim in Usd.PrimRange(root):
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            is_handle = prim.GetName().startswith(("Handle_Left_", "Handle_Right_"))
            is_enabled = bool(
                UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get()
            )
            if is_enabled != is_handle:
                raise RuntimeError(
                    f"Unexpected collision state on {prim.GetPath()}: {is_enabled}"
                )
            collider_count += 1
            enabled_handle_count += int(is_enabled)
    if collider_count != 6 or enabled_handle_count != 6:
        raise RuntimeError(
            "Unexpected handles-only collision profile: "
            f"{enabled_handle_count}/{collider_count} enabled"
        )
    root.SetCustomDataByKey("handCartDemoMode", "kinematic synchronized visual with handle-only collision; wheel angle derives from base track travel")
    root.SetCustomDataByKey("handCartVisualSource", str(HAND_CART_OBJECT))
    return root


def spawn_hand_cart_tracked_g1(prim_path: str, cfg: UsdFileCfg, translation=None, orientation=None, **kwargs) -> Usd.Prim:
    g1_upper = spawn_combined_tracked_g1(prim_path, cfg, translation=translation, orientation=orientation, **kwargs)
    stage = g1_upper.GetStage()
    root = stage.GetPrimAtPath("/World/TrackedBase")
    carriage = stage.GetPrimAtPath("/World/TrackedBase/LiftCarriage")
    if not root or not carriage:
        raise RuntimeError("Tracked base hierarchy is incomplete")
    root_xform = UsdGeom.Xformable(root)
    _get_or_add_op(root_xform, "translate").Set((TASK_TRACKED_BASE_X, TASK_TRACKED_BASE_Y, 0.0))
    _get_or_add_op(root_xform, "orient").Set(Gf.Quatd(math.cos(0.5 * TASK_TRACKED_BASE_YAW), Gf.Vec3d(0.0, 0.0, math.sin(0.5 * TASK_TRACKED_BASE_YAW))))
    _get_or_add_op(UsdGeom.Xformable(carriage), "translate").Set((0.0, 0.0, TASK_TRACKED_LIFT_HEIGHT))
    _get_or_add_op(UsdGeom.Xformable(g1_upper), "translate").Set((G1_MOUNT_LOCAL_POSITION[0], G1_MOUNT_LOCAL_POSITION[1], G1_MOUNT_LOCAL_POSITION[2] + TASK_TRACKED_LIFT_HEIGHT))
    return g1_upper


def make_hand_cart_g1_cfg():
    cfg = make_embedded_g1_upper_body_cfg()
    cfg.init_state.pos = TASK_G1_MOUNT_POSITION
    cfg.init_state.rot = TASK_G1_MOUNT_ORIENTATION
    cfg.spawn.func = spawn_hand_cart_tracked_g1
    return cfg


@configclass
class HandCartRollingDemoSceneCfg(InteractiveSceneCfg):
    environment = AssetBaseCfg(
        prim_path="/World/envs/env_.*/Environment",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0), rot=(0.0, 1.0, 0.0, 0.0)),
        spawn=UsdFileCfg(usd_path=str(ENVIRONMENT)),
    )
    hand_cart = AssetBaseCfg(
        prim_path="/World/envs/env_.*/HandCart",
        init_state=AssetBaseCfg.InitialStateCfg(pos=CART_POSITION, rot=CART_ORIENTATION),
        spawn=UsdFileCfg(usd_path=str(HAND_CART_OBJECT), func=spawn_rolling_demo_cart),
    )
    robot = make_hand_cart_g1_cfg()
    front_camera = CameraPresets.g1_front_camera().replace(prim_path="/World/TrackedBase/G1UpperBody/d435_link/front_cam")
    left_wrist_camera = CameraPresets.left_gripper_wrist_camera().replace(prim_path="/World/TrackedBase/G1UpperBody/left_hand_base_link/left_wrist_camera")
    right_wrist_camera = CameraPresets.right_gripper_wrist_camera().replace(prim_path="/World/TrackedBase/G1UpperBody/right_hand_base_link/right_wrist_camera")
    world_camera = CameraBaseCfg.get_camera_config(prim_path="/World/PerspectiveCamera", pos_offset=(-0.7, 3.8, 2.3), rot_offset=(0.105, 0.205, 0.675, -0.701), focal_length=18.0)
    light = AssetBaseCfg(prim_path="/World/light", spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=1200.0))


@configclass
class ActionsCfg:
    joint_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=[".*"], scale=1.0, use_default_offset=True)


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
    """No IsaacLab reset terms; the demo controller owns cart reset state."""


def _runtime_controller(env):
    controller = getattr(env, "_hand_cart_rolling_demo_runtime_controller", None)
    if controller is None:
        raise RuntimeError("Hand-cart rolling-demo runtime controller is not initialized")
    return controller


def reset_task_objects(env) -> None:
    _runtime_controller(env).reset_cart()


def reset_everything(env) -> None:
    _runtime_controller(env).reset_everything()


@configclass
class HandCartRollingDemoG129Dex1EnvCfg(ManagerBasedRLEnvCfg):
    scene: HandCartRollingDemoSceneCfg = HandCartRollingDemoSceneCfg(num_envs=1, env_spacing=2.5, replicate_physics=True)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    rewards: RewardsCfg = RewardsCfg()
    events: EventCfg = EventCfg()
    commands = None
    curriculum = None

    def create_runtime_controller(self, env):
        from .hand_cart_rolling_demo_controller import HandCartRollingDemoController

        controller = HandCartRollingDemoController(env)
        env._hand_cart_rolling_demo_runtime_controller = controller
        return controller

    def __post_init__(self):
        self.decimation = 2
        self.episode_length_s = 1.0e9
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.enable_stabilization = True
        self.sim.physx.num_substeps = 4
        self.sim.physx.num_position_iterations = 16
        self.sim.physx.num_velocity_iterations = 8
        self.event_manager = SimpleEventManager()
        self.event_manager.register("reset_object_self", SimpleEvent(func=reset_task_objects))
        self.event_manager.register("reset_all_self", SimpleEvent(func=reset_everything))
