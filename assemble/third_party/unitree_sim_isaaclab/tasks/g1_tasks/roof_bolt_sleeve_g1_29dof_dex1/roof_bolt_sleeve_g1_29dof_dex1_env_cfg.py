"""Tracked G1 task for installing a coupling sleeve over a fixed roof bolt."""

from pathlib import Path
import math
import os

import torch
from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics

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
    TABLE_ORIENTATION,
    TRACKED_G1_ASSEMBLY,
    WORKTABLE_NATIVE_TOP_Z_CM,
    WORKTABLE_ORIENTATION,
    WORKTABLE_XY_SCALE,
    make_embedded_g1_upper_body_cfg,
    spawn_combined_tracked_g1,
    spawn_static_worktable,
)


ASSEMBLE_ROOT = Path(__file__).resolve().parents[5]
TASK_ROOT = ASSEMBLE_ROOT / "task" / "g1_roof_bolt_sleeve_task"
WORKTABLE = (
    ASSETS_ROOT
    / "object/TableTrolley_B/TableTrolley_B01_01.usd"
)
SLEEVE_OBJECT = (
    ASSETS_ROOT
    / "object/steel_mining_roof_bolt_coupling_sleeve_ID53_OD55_COD65_L250.usd"
)
ROOF_BOLT_OBJECT = ASSETS_ROOT / "object/M20_Bolt_Chain_D45_L1450.usda"

# Mirrored about the robot's forward axis: the fixed roof bolt is on the
# operator's left, while the worktable and graspable sleeve are on the right.
# Move the entire task group another 0.50 m toward the robot's left arm so the
# new worktable clears the furnishing already present on the scene's right.
# This can be tuned without editing code before launch if the scene changes.
GROUP_LEFT_SHIFT_M = float(
    os.environ.get("G1_ROOF_BOLT_GROUP_LEFT_SHIFT_M", "0.50")
)
GROUP_SHIFT_X = -math.sin(G1_FORWARD_YAW) * GROUP_LEFT_SHIFT_M
GROUP_SHIFT_Y = math.cos(G1_FORWARD_YAW) * GROUP_LEFT_SHIFT_M

# Fine-tune only the manipulation layout after moving the complete task group:
# keep the robot fixed and move the table plus both objects 0.20 m toward the
# robot's right arm.
OBJECT_RIGHT_SHIFT_M = float(
    os.environ.get("G1_ROOF_BOLT_OBJECT_RIGHT_SHIFT_M", "0.20")
)
OBJECT_SHIFT_X = math.sin(G1_FORWARD_YAW) * OBJECT_RIGHT_SHIFT_M
OBJECT_SHIFT_Y = -math.cos(G1_FORWARD_YAW) * OBJECT_RIGHT_SHIFT_M

# Captured from the live simulator on 2026-08-21.  Preserve support for the
# group-shift override by applying only its delta from the captured 0.50 m
# layout; with default settings these are the exact live base coordinates.
_CAPTURED_GROUP_LEFT_SHIFT_M = 0.50
_CAPTURED_TRACKED_BASE_X = -1.5534816454092946
_CAPTURED_TRACKED_BASE_Y = 2.0714633528394826
TASK_TRACKED_BASE_YAW = -0.5241686803753723
# Captured from the live Task 2 DDS state on 2026-08-21. The G1 root was at
# world Z=1.1904679537 m and the authored mount is at Z=0.9904679868 m.
TASK_TRACKED_LIFT_HEIGHT = 0.20
_GROUP_SHIFT_DELTA_M = GROUP_LEFT_SHIFT_M - _CAPTURED_GROUP_LEFT_SHIFT_M
TASK_TRACKED_BASE_X = (
    _CAPTURED_TRACKED_BASE_X
    - math.sin(G1_FORWARD_YAW) * _GROUP_SHIFT_DELTA_M
)
TASK_TRACKED_BASE_Y = (
    _CAPTURED_TRACKED_BASE_Y
    + math.cos(G1_FORWARD_YAW) * _GROUP_SHIFT_DELTA_M
)

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

_MIRRORED_TABLE_POSITION = (-2.814060147015613, 3.966824536771499, 0.0)
# The 0.250 m sleeve stays seated on the 1.09 m tabletop, placing its center
# at 1.215 m.
_MIRRORED_SLEEVE_POSITION = (-2.7978224747940406, 3.495405044174062, 1.215)
_MIRRORED_ROOF_BOLT_POSITION = (-3.0778315693665093, 3.3107569817506888, 0.575)


def _shift_xy(position):
    return (
        position[0] + GROUP_SHIFT_X + OBJECT_SHIFT_X,
        position[1] + GROUP_SHIFT_Y + OBJECT_SHIFT_Y,
        position[2],
    )


_SHIFTED_TABLE_POSITION = _shift_xy(_MIRRORED_TABLE_POSITION)
TABLE_RIGHT_SHIFT_M = float(
    os.environ.get("G1_ROOF_BOLT_TABLE_RIGHT_SHIFT_M", "0.03")
)
TABLE_POSITION = (
    _SHIFTED_TABLE_POSITION[0]
    + math.sin(G1_FORWARD_YAW) * TABLE_RIGHT_SHIFT_M,
    _SHIFTED_TABLE_POSITION[1]
    - math.cos(G1_FORWARD_YAW) * TABLE_RIGHT_SHIFT_M,
    _SHIFTED_TABLE_POSITION[2],
)
SLEEVE_POSITION = _shift_xy(_MIRRORED_SLEEVE_POSITION)
_SHIFTED_ROOF_BOLT_POSITION = _shift_xy(_MIRRORED_ROOF_BOLT_POSITION)
ROOF_BOLT_TOWARD_ROBOT_M = float(
    os.environ.get("G1_ROOF_BOLT_TOWARD_ROBOT_M", "0.15")
)
ROOF_BOLT_POSITION = (
    _SHIFTED_ROOF_BOLT_POSITION[0]
    - math.cos(G1_FORWARD_YAW) * ROOF_BOLT_TOWARD_ROBOT_M,
    _SHIFTED_ROOF_BOLT_POSITION[1]
    - math.sin(G1_FORWARD_YAW) * ROOF_BOLT_TOWARD_ROBOT_M,
    _SHIFTED_ROOF_BOLT_POSITION[2],
)
SLEEVE_ORIENTATION = TABLE_ORIENTATION
ROOF_BOLT_ORIENTATION = TABLE_ORIENTATION
WORKTABLE_Z_SCALE = 1.09 / WORKTABLE_NATIVE_TOP_Z_CM

SLEEVE_MASS = 0.2834361495612364
SLEEVE_INERTIA_TRANSVERSE = 0.00294677466
SLEEVE_INERTIA_AXIAL = 0.000106728127
SLEEVE_COLLIDER_COUNT = 16
SLEEVE_CONTACT_OFFSET = 0.0005
INSERTION_CONTACT_OFFSET = 0.0005

for asset_path in (
    ENVIRONMENT,
    WORKTABLE,
    SLEEVE_OBJECT,
    ROOF_BOLT_OBJECT,
    TRACKED_G1_ASSEMBLY,
):
    if not asset_path.is_file():
        raise FileNotFoundError(asset_path)


def zero_reward(env) -> torch.Tensor:
    return torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)


def never_terminate(env) -> torch.Tensor:
    return torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)


def spawn_shifted_tracked_g1(
    prim_path: str,
    cfg: UsdFileCfg,
    translation=None,
    orientation=None,
    **kwargs,
) -> Usd.Prim:
    """Spawn the shared robot asset, then shift this task's assembly only."""

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
        (
            op
            for op in xform.GetOrderedXformOps()
            if op.GetOpName() == "xformOp:translate"
        ),
        None,
    )
    if translate_op is None:
        translate_op = xform.AddTranslateOp()
    translate_op.Set((TASK_TRACKED_BASE_X, TASK_TRACKED_BASE_Y, 0.0))

    orient_op = next(
        (
            op
            for op in xform.GetOrderedXformOps()
            if op.GetOpName() == "xformOp:orient"
        ),
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

    carriage = g1_upper.GetStage().GetPrimAtPath("/World/TrackedBase/LiftCarriage")
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
        (
            G1_MOUNT_LOCAL_POSITION[0],
            G1_MOUNT_LOCAL_POSITION[1],
            G1_MOUNT_LOCAL_POSITION[2] + TASK_TRACKED_LIFT_HEIGHT,
        )
    )
    print(
        "[roof-bolt-sleeve] Task group shifted toward the robot's left "
        f"by {GROUP_LEFT_SHIFT_M:.2f} m."
    )
    return g1_upper


def make_shifted_g1_upper_body_cfg():
    cfg = make_embedded_g1_upper_body_cfg()
    cfg.init_state.pos = TASK_G1_MOUNT_POSITION
    cfg.init_state.rot = TASK_G1_MOUNT_ORIENTATION
    cfg.spawn.func = spawn_shifted_tracked_g1
    # The stock Dex1 drive is very stiff and has no explicit force/velocity
    # cap. On a 3 mm sleeve wall it can keep driving after first contact and
    # force the fingers through the collider. These limits are local to this
    # task and retain ample grip for the 0.5 kg sleeve under gravity.
    hand_drive = cfg.actuators["hands"]
    hand_drive.stiffness = 400.0
    hand_drive.damping = 10.0
    hand_drive.effort_limit_sim = 40.0
    hand_drive.velocity_limit_sim = 0.20
    return cfg


def _configure_collision_tree(root: Usd.Prim, contact_offset: float) -> int:
    count = 0
    for prim in Usd.PrimRange(root):
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        enabled = prim.GetAttribute("physics:collisionEnabled")
        if enabled and enabled.Get() is False:
            continue
        api = PhysxSchema.PhysxCollisionAPI.Apply(prim)
        api.CreateContactOffsetAttr().Set(contact_offset)
        api.CreateRestOffsetAttr().Set(0.0)
        count += 1
    return count


def spawn_task_sleeve(
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
    mass = UsdPhysics.MassAPI.Apply(root)
    mass.CreateMassAttr().Set(SLEEVE_MASS)
    mass.CreateCenterOfMassAttr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
    mass.CreateDiagonalInertiaAttr().Set(
        Gf.Vec3f(
            SLEEVE_INERTIA_TRANSVERSE,
            SLEEVE_INERTIA_TRANSVERSE,
            SLEEVE_INERTIA_AXIAL,
        )
    )
    mass.CreatePrincipalAxesAttr().Set(
        Gf.Quatf(1.0, Gf.Vec3f(0.0, 0.0, 0.0))
    )
    physx = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    physx.CreateDisableGravityAttr().Set(False)
    physx.CreateEnableSpeculativeCCDAttr().Set(True)
    physx.CreateSolverPositionIterationCountAttr().Set(16)
    physx.CreateSolverVelocityIterationCountAttr().Set(4)
    physx.CreateMaxDepenetrationVelocityAttr().Set(1.0)
    collider_count = _configure_collision_tree(root, SLEEVE_CONTACT_OFFSET)
    if collider_count != SLEEVE_COLLIDER_COUNT:
        raise RuntimeError(
            f"Expected {SLEEVE_COLLIDER_COUNT} sleeve collision mesh, "
            f"found {collider_count}"
        )
    print(
        f"[roof-bolt-sleeve] Sleeve uses Leiguan physics: mass={SLEEVE_MASS:.6f} kg, "
        "matching inertia, gravity/default damping/default material, "
        f"speculative CCD, solver=16/4, contact offset=0.5 mm, "
        f"explicit convex ring segments={collider_count}."
    )
    return root


def spawn_fixed_roof_bolt(
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
    # The reusable chain asset disables each referenced segment's source
    # rigid body, but IsaacLab requires those APIs to be absent (not merely
    # disabled) when resolving a managed RigidObject. Author API removals in
    # this task's composed stage; the source asset remains unchanged.
    removed_segment_bodies = 0
    for prim in Usd.PrimRange(root):
        if prim == root or not prim.HasAPI(UsdPhysics.RigidBodyAPI):
            continue
        if prim.RemoveAPI(UsdPhysics.RigidBodyAPI):
            removed_segment_bodies += 1
    rigid = UsdPhysics.RigidBodyAPI.Apply(root)
    rigid.CreateRigidBodyEnabledAttr().Set(True)
    rigid.CreateKinematicEnabledAttr().Set(True)
    physx = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    physx.CreateDisableGravityAttr().Set(True)
    collider_count = _configure_collision_tree(root, INSERTION_CONTACT_OFFSET)
    if collider_count != 1:
        raise RuntimeError(
            f"Expected one continuous roof-bolt collider, found {collider_count}"
        )
    print(
        "[roof-bolt-sleeve] Roof bolt fixed kinematically with "
        f"{collider_count} continuous collider; removed "
        f"{removed_segment_bodies} disabled segment rigid-body APIs."
    )
    return root


@configclass
class RoofBoltSleeveSceneCfg(InteractiveSceneCfg):
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
            # Preserve the raised task's calibrated 1.09 m tabletop.
            scale=(WORKTABLE_XY_SCALE, WORKTABLE_XY_SCALE, WORKTABLE_Z_SCALE),
        ),
    )

    sleeve = RigidObjectCfg(
        prim_path="/World/envs/env_.*/CouplingSleeve",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=SLEEVE_POSITION,
            rot=SLEEVE_ORIENTATION,
        ),
        spawn=UsdFileCfg(
            usd_path=str(SLEEVE_OBJECT),
            func=spawn_task_sleeve,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                disable_gravity=False,
                max_depenetration_velocity=1.0,
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=4,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=SLEEVE_MASS),
            collision_props=sim_utils.CollisionPropertiesCfg(
                contact_offset=SLEEVE_CONTACT_OFFSET,
                rest_offset=0.0,
            ),
        ),
    )

    roof_bolt = RigidObjectCfg(
        prim_path="/World/envs/env_.*/RoofBolt",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=ROOF_BOLT_POSITION,
            rot=ROOF_BOLT_ORIENTATION,
        ),
        spawn=UsdFileCfg(
            usd_path=str(ROOF_BOLT_OBJECT),
            func=spawn_fixed_roof_bolt,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                kinematic_enabled=True,
                disable_gravity=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                contact_offset=INSERTION_CONTACT_OFFSET,
                rest_offset=0.0,
            ),
        ),
    )

    robot = make_shifted_g1_upper_body_cfg()

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
        pos_offset=(-2.0, 4.15, 1.9),
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
    reset_sleeve = EventTermCfg(
        func=base_mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("sleeve"),
        },
    )
    # The roof bolt is kinematic and is reset deterministically by the runtime
    # BenchmarkProtocol controller. The generic event writes a velocity and
    # triggers a PhysX error on CPU physics, so it must not own this body.


def _runtime_controller(env):
    controller = getattr(env, "_roof_bolt_sleeve_runtime_controller", None)
    if controller is None:
        raise RuntimeError("Roof-bolt sleeve runtime controller is not initialized")
    return controller


def reset_task_objects(env) -> None:
    _runtime_controller(env).reset_task_objects()


def reset_everything(env) -> None:
    _runtime_controller(env).reset_everything()


@configclass
class RoofBoltSleeveG129Dex1EnvCfg(ManagerBasedRLEnvCfg):
    scene: RoofBoltSleeveSceneCfg = RoofBoltSleeveSceneCfg(
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
        from .roof_bolt_sleeve_controller import RoofBoltSleeveController

        controller = RoofBoltSleeveController(env)
        env._roof_bolt_sleeve_runtime_controller = controller
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
