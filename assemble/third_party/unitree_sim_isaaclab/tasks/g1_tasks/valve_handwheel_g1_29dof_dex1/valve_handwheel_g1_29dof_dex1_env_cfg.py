"""Tracked G1 task for rotating an installed three-way-valve handwheel."""

from pathlib import Path
import math

import torch
from pxr import Gf, Usd, UsdGeom, UsdPhysics

import isaaclab.envs.mdp as base_mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs.mdp.actions.joint_actions import JointPositionAction
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
    G1_MOUNT_LOCAL_ORIENTATION,
    G1_MOUNT_LOCAL_POSITION,
    TRACKED_BASE_YAW as REFERENCE_TRACKED_BASE_YAW,
    TRACKED_G1_ASSEMBLY,
    make_embedded_g1_upper_body_cfg,
    spawn_combined_tracked_g1,
)


ASSEMBLE_ROOT = Path(__file__).resolve().parents[5]
TASK_ROOT = ASSEMBLE_ROOT / "task" / "g1_valve_handwheel_task"
HANDWHEEL_OBJECT = (
    ASSETS_ROOT / "object/valve_handwheel_task/blue_valve_handwheel_task.usd"
)
VALVE_OBJECT = (
    ASSETS_ROOT / "object/valve_handwheel_task/three_way_ball_valve_task.usd"
)
# Install the assembly on the wall that was behind the previous robot pose.
# The contact point and inward normal were measured against the RealScene04
# collision mesh.  The local +Z face that formerly rested on the floor now
# points out of the wall, so the handwheel's revolute axis is horizontal.
WALL_SURFACE_POSITION = (1.186, 1.077, 1.25)
WALL_CLEARANCE = 0.005
ROBOT_TO_WALL_DISTANCE = 0.85
# Captured from the live LiftCarriage transform on 2026-08-22.  Keep this
# task-local so changing the wall task does not alter any other G1 task.
TASK_TRACKED_LIFT_HEIGHT = 0.1218405273555287


def _wrap_angle(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def _multiply_quaternions_wxyz(left, right):
    lw, lx, ly, lz = left
    rw, rx, ry, rz = right
    return (
        lw * rw - lx * rx - ly * ry - lz * rz,
        lw * rx + lx * rw + ly * rz - lz * ry,
        lw * ry - lx * rz + ly * rw + lz * rx,
        lw * rz + lx * ry - ly * rx + lz * rw,
    )


TASK_TRACKED_BASE_YAW = _wrap_angle(REFERENCE_TRACKED_BASE_YAW + math.pi)
TASK_G1_FORWARD_YAW = _wrap_angle(TASK_TRACKED_BASE_YAW + math.pi)
WALL_INWARD_NORMAL = (
    -math.cos(TASK_G1_FORWARD_YAW),
    -math.sin(TASK_G1_FORWARD_YAW),
    0.0,
)
TASK_G1_MOUNT_POSITION = (
    WALL_SURFACE_POSITION[0] + WALL_INWARD_NORMAL[0] * ROBOT_TO_WALL_DISTANCE,
    WALL_SURFACE_POSITION[1] + WALL_INWARD_NORMAL[1] * ROBOT_TO_WALL_DISTANCE,
    G1_MOUNT_LOCAL_POSITION[2] + TASK_TRACKED_LIFT_HEIGHT,
)
_task_base_cosine = math.cos(TASK_TRACKED_BASE_YAW)
_task_base_sine = math.sin(TASK_TRACKED_BASE_YAW)
TASK_TRACKED_BASE_X = (
    TASK_G1_MOUNT_POSITION[0]
    - _task_base_cosine * G1_MOUNT_LOCAL_POSITION[0]
    + _task_base_sine * G1_MOUNT_LOCAL_POSITION[1]
)
TASK_TRACKED_BASE_Y = (
    TASK_G1_MOUNT_POSITION[1]
    - _task_base_sine * G1_MOUNT_LOCAL_POSITION[0]
    - _task_base_cosine * G1_MOUNT_LOCAL_POSITION[1]
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


def _read_task_metadata(path: Path, expected_root: str) -> dict[str, float | str]:
    stage = Usd.Stage.Open(str(path))
    if stage is None:
        raise RuntimeError(f"Failed to open task asset: {path}")
    root = stage.GetDefaultPrim()
    if not root or str(root.GetPath()) != expected_root:
        raise RuntimeError(
            f"Unexpected default prim in {path}: {root.GetPath() if root else None}"
        )
    if UsdGeom.GetStageUpAxis(stage) != UsdGeom.Tokens.z:
        raise RuntimeError(f"Task asset must be Z-up: {path}")
    if abs(float(UsdGeom.GetStageMetersPerUnit(stage)) - 1.0) > 1.0e-9:
        raise RuntimeError(f"Task asset must use meters: {path}")
    return {
        attribute.GetName()[5:]: attribute.Get()
        for attribute in root.GetAttributes()
        if attribute.GetName().startswith("task:")
    }


for asset_path in (
    ENVIRONMENT,
    TRACKED_G1_ASSEMBLY,
    HANDWHEEL_OBJECT,
    VALVE_OBJECT,
):
    if not asset_path.is_file():
        raise FileNotFoundError(asset_path)

HANDWHEEL_METADATA = _read_task_metadata(HANDWHEEL_OBJECT, "/ValveHandwheel")
VALVE_METADATA = _read_task_metadata(VALVE_OBJECT, "/ThreeWayBallValve")
HANDWHEEL_OUTER_RADIUS = float(HANDWHEEL_METADATA["outerDiameter"]) / 2.0
HANDWHEEL_BORE_RADIUS = float(HANDWHEEL_METADATA["collisionBoreDiameter"]) / 2.0
HANDWHEEL_THICKNESS = float(HANDWHEEL_METADATA["thickness"])
HANDWHEEL_HUB_HEIGHT = float(HANDWHEEL_METADATA["hubCollisionHeight"])
VALVE_STEM_RADIUS = float(VALVE_METADATA["stemDiameter"]) / 2.0
VALVE_STEM_BASE_Z = float(VALVE_METADATA["stemBaseZ"])
VALVE_STEM_TIP_Z = float(VALVE_METADATA["stemTipZ"])
VALVE_SIZE_X = float(VALVE_METADATA["sizeX"])
VALVE_SIZE_Y = float(VALVE_METADATA["sizeY"])
VALVE_SIZE_Z = float(VALVE_METADATA["sizeZ"])

TASK_SIZE_SCALE = 1.8
if not (
    0.215 * TASK_SIZE_SCALE
    <= 2.0 * HANDWHEEL_OUTER_RADIUS
    <= 0.225 * TASK_SIZE_SCALE
):
    raise RuntimeError("Task handwheel outer diameter is outside scaled tolerance")
if not (
    0.0255 * TASK_SIZE_SCALE
    <= 2.0 * HANDWHEEL_BORE_RADIUS
    <= 0.0265 * TASK_SIZE_SCALE
):
    raise RuntimeError("Task handwheel collision bore is outside scaled tolerance")
if not (
    0.0155 * TASK_SIZE_SCALE
    <= 2.0 * VALVE_STEM_RADIUS
    <= 0.0165 * TASK_SIZE_SCALE
):
    raise RuntimeError("Task valve stem diameter is outside scaled tolerance")
if HANDWHEEL_BORE_RADIUS <= VALVE_STEM_RADIUS:
    raise RuntimeError("Task valve stem cannot fit through the handwheel bore")


# Put the valve's former floor-contacting local Z=0 face against the wall. The
# wheel begins fully seated and its rotation axis follows the wall normal.
VALVE_POSITION = (
    WALL_SURFACE_POSITION[0] + WALL_INWARD_NORMAL[0] * WALL_CLEARANCE,
    WALL_SURFACE_POSITION[1] + WALL_INWARD_NORMAL[1] * WALL_CLEARANCE,
    WALL_SURFACE_POSITION[2],
)
VALVE_WALL_YAW = TASK_G1_FORWARD_YAW - 0.5 * math.pi
VALVE_ORIENTATION = _multiply_quaternions_wxyz(
    (
        math.cos(0.5 * VALVE_WALL_YAW),
        0.0,
        0.0,
        math.sin(0.5 * VALVE_WALL_YAW),
    ),
    (math.cos(0.25 * math.pi), math.sin(0.25 * math.pi), 0.0, 0.0),
)
HANDWHEEL_POSITION = (
    VALVE_POSITION[0] + WALL_INWARD_NORMAL[0] * VALVE_STEM_BASE_Z,
    VALVE_POSITION[1] + WALL_INWARD_NORMAL[1] * VALVE_STEM_BASE_Z,
    VALVE_POSITION[2],
)
HANDWHEEL_ORIENTATION = VALVE_ORIENTATION
HANDWHEEL_JOINT_ANCHOR_Z = VALVE_STEM_BASE_Z + 0.5 * HANDWHEEL_HUB_HEIGHT


def spawn_valve_with_handwheel_joint(
    prim_path: str,
    cfg: UsdFileCfg,
    translation=None,
    orientation=None,
    **kwargs,
) -> Usd.Prim:
    """Spawn the fixed valve and constrain the installed wheel to Z rotation."""

    valve_root = spawn_from_usd(
        prim_path,
        cfg,
        translation=translation,
        orientation=orientation,
        **kwargs,
    )
    handwheel_path = valve_root.GetPath().GetParentPath().AppendChild("Handwheel")
    handwheel_root = valve_root.GetStage().GetPrimAtPath(handwheel_path)
    if not handwheel_root:
        raise RuntimeError(
            f"Cannot create valve joint; handwheel is missing at {handwheel_path}"
        )

    joint = UsdPhysics.RevoluteJoint.Define(
        valve_root.GetStage(), valve_root.GetPath().AppendChild("HandwheelJoint")
    )
    joint.CreateBody0Rel().SetTargets([valve_root.GetPath()])
    joint.CreateBody1Rel().SetTargets([handwheel_path])
    joint.CreateAxisAttr(UsdGeom.Tokens.z)
    joint.CreateCollisionEnabledAttr(False)
    joint.CreateLocalPos0Attr(
        Gf.Vec3f(0.0, 0.0, float(HANDWHEEL_JOINT_ANCHOR_Z))
    )
    joint.CreateLocalPos1Attr(
        Gf.Vec3f(0.0, 0.0, float(0.5 * HANDWHEEL_HUB_HEIGHT))
    )
    identity = Gf.Quatf(1.0, Gf.Vec3f(0.0))
    joint.CreateLocalRot0Attr(identity)
    joint.CreateLocalRot1Attr(identity)
    print(
        "[valve-handwheel] Installed handwheel revolute joint created: "
        "axis=Z, connected-body collision disabled."
    )
    return valve_root


def _set_xform_op(prim: Usd.Prim, op_name: str, value) -> None:
    xform = UsdGeom.Xformable(prim)
    op = next(
        (candidate for candidate in xform.GetOrderedXformOps() if candidate.GetOpName() == op_name),
        None,
    )
    if op is None:
        if op_name == "xformOp:translate":
            op = xform.AddTranslateOp()
        elif op_name == "xformOp:orient":
            op = xform.AddOrientOp()
        else:
            raise ValueError(f"Unsupported xform op: {op_name}")
    op.Set(value)


def spawn_wall_task_tracked_g1(
    prim_path: str,
    cfg: UsdFileCfg,
    translation=None,
    orientation=None,
    **kwargs,
) -> Usd.Prim:
    """Spawn the shared tracked G1, then apply this task's flipped wall pose."""

    g1_upper = spawn_combined_tracked_g1(
        prim_path,
        cfg,
        translation=translation,
        orientation=orientation,
        **kwargs,
    )
    tracked_root = g1_upper.GetStage().GetPrimAtPath("/World/TrackedBase")
    if not tracked_root:
        raise RuntimeError("Tracked G1 root is missing after spawn")
    _set_xform_op(
        tracked_root,
        "xformOp:translate",
        Gf.Vec3d(TASK_TRACKED_BASE_X, TASK_TRACKED_BASE_Y, 0.0),
    )
    _set_xform_op(
        tracked_root,
        "xformOp:orient",
        Gf.Quatd(
            math.cos(0.5 * TASK_TRACKED_BASE_YAW),
            Gf.Vec3d(0.0, 0.0, math.sin(0.5 * TASK_TRACKED_BASE_YAW)),
        ),
    )
    carriage = g1_upper.GetStage().GetPrimAtPath("/World/TrackedBase/LiftCarriage")
    if not carriage:
        raise RuntimeError("Tracked G1 lift carriage is missing after spawn")
    _set_xform_op(
        carriage,
        "xformOp:translate",
        Gf.Vec3d(0.0, 0.0, TASK_TRACKED_LIFT_HEIGHT),
    )
    _set_xform_op(
        g1_upper,
        "xformOp:translate",
        Gf.Vec3d(
            G1_MOUNT_LOCAL_POSITION[0],
            G1_MOUNT_LOCAL_POSITION[1],
            G1_MOUNT_LOCAL_POSITION[2] + TASK_TRACKED_LIFT_HEIGHT,
        ),
    )
    print(
        "[valve-handwheel] Tracked G1 flipped toward rear wall: "
        f"base=({TASK_TRACKED_BASE_X:.4f}, {TASK_TRACKED_BASE_Y:.4f}), "
        f"yaw={math.degrees(TASK_TRACKED_BASE_YAW):.1f} deg, "
        f"lift={TASK_TRACKED_LIFT_HEIGHT:.4f} m."
    )
    return g1_upper


def make_wall_task_g1_cfg() -> ArticulationCfg:
    cfg = make_embedded_g1_upper_body_cfg()
    cfg.init_state.pos = TASK_G1_MOUNT_POSITION
    cfg.init_state.rot = TASK_G1_MOUNT_ORIENTATION
    cfg.spawn = UsdFileCfg(
        usd_path=str(TRACKED_G1_ASSEMBLY),
        func=spawn_wall_task_tracked_g1,
    )
    return cfg


def zero_reward(env) -> torch.Tensor:
    return torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)


def never_terminate(env) -> torch.Tensor:
    return torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)


@configclass
class ValveHandwheelSceneCfg(InteractiveSceneCfg):
    environment = AssetBaseCfg(
        prim_path="/World/envs/env_.*/Environment",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0),
            rot=(0.0, 1.0, 0.0, 0.0),
        ),
        spawn=UsdFileCfg(usd_path=str(ENVIRONMENT)),
    )

    handwheel = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Handwheel",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=HANDWHEEL_POSITION,
            rot=HANDWHEEL_ORIENTATION,
        ),
        spawn=UsdFileCfg(
            usd_path=str(HANDWHEEL_OBJECT),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                kinematic_enabled=False,
                disable_gravity=False,
                max_depenetration_velocity=1.0,
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=4,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.5),
        ),
    )

    valve = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Valve",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=VALVE_POSITION,
            rot=VALVE_ORIENTATION,
        ),
        spawn=UsdFileCfg(
            usd_path=str(VALVE_OBJECT),
            func=spawn_valve_with_handwheel_joint,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                kinematic_enabled=True,
                disable_gravity=True,
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=4,
            ),
        ),
    )

    robot: ArticulationCfg = make_wall_task_g1_cfg()

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
        pos_offset=(-1.75, 4.4, 1.9),
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


class ValveHandwheelJointPositionAction(JointPositionAction):
    """Apply DDS joint targets while task-locally overriding waist pitch."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        try:
            self._waist_pitch_action_index = self._joint_names.index(
                "waist_pitch_joint"
            )
        except ValueError as error:
            raise RuntimeError(
                "Tracked G1 action does not contain waist_pitch_joint"
            ) from error

    def process_actions(self, actions: torch.Tensor) -> None:
        super().process_actions(actions)
        controller = getattr(
            self._env, "_valve_handwheel_runtime_controller", None
        )
        if controller is not None:
            self._processed_actions[:, self._waist_pitch_action_index] = (
                controller.waist_pitch_target
            )


@configclass
class ActionsCfg:
    joint_pos = mdp.JointPositionActionCfg(
        class_type=ValveHandwheelJointPositionAction,
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
    reset_handwheel = EventTermCfg(
        func=base_mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("handwheel"),
        },
    )


def _runtime_controller(env):
    controller = getattr(env, "_valve_handwheel_runtime_controller", None)
    if controller is None:
        raise RuntimeError("Valve-handwheel runtime controller is not initialized")
    return controller


def reset_task_objects(env) -> None:
    _runtime_controller(env).reset_task_objects()


def reset_everything(env) -> None:
    _runtime_controller(env).reset_everything()


@configclass
class ValveHandwheelG129Dex1EnvCfg(ManagerBasedRLEnvCfg):
    scene: ValveHandwheelSceneCfg = ValveHandwheelSceneCfg(
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
        from .valve_handwheel_controller import ValveHandwheelController

        controller = ValveHandwheelController(env)
        env._valve_handwheel_runtime_controller = controller
        return controller

    def __post_init__(self):
        self.decimation = 2
        self.episode_length_s = 1.0e9
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625
        self.sim.physx.gpu_constraint_solver_heavy_spring_enabled = True
        self.sim.physx.num_substeps = 2
        self.sim.physx.num_position_iterations = 12
        self.sim.physx.num_velocity_iterations = 4

        self.event_manager = SimpleEventManager()
        self.event_manager.register(
            "reset_object_self", SimpleEvent(func=reset_task_objects)
        )
        self.event_manager.register(
            "reset_all_self", SimpleEvent(func=reset_everything)
        )
