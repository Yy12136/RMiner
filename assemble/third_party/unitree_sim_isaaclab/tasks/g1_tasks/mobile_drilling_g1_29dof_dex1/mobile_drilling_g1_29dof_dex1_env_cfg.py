"""G1-29 + Dex1 in the reconstructed mobile-drilling task scene.

Only the scene composition differs from the standard Unitree Dex1 example.
The robot, DDS action path, cameras, and official XR IK/controller remain the
same so the existing Quest 3/3S workflow can be used without a second teleop
implementation.
"""

from pathlib import Path
import math
import os

import torch
from pxr import Gf, Usd, UsdGeom, UsdPhysics

import isaaclab.envs.mdp as base_mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
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
from isaaclab.sim.utils import clone
from isaaclab.utils import configclass

from tasks.common_config import CameraBaseCfg, CameraPresets, G1RobotPresets
from tasks.common_event.event_manager import SimpleEvent, SimpleEventManager
from tasks.common_observations.camera_state import get_camera_image
from tasks.common_observations.g1_29dof_state import get_robot_boy_joint_states
from tasks.common_observations.gripper_state import get_robot_gipper_joint_states
from tasks.g1_tasks.pick_place_cylinder_g1_29dof_dex1 import mdp


ASSEMBLE_ROOT = Path(__file__).resolve().parents[5]
TASK_ROOT = ASSEMBLE_ROOT / "task" / "w1_mobile_drilling_task"
ASSETS_ROOT = ASSEMBLE_ROOT / "assets-new"
ENVIRONMENT = (
    TASK_ROOT
    / "assets/environment/RealScene04/real2sim_intermediate/RealScene04_Orin.usd"
)
DRILLING_OBJECT = TASK_ROOT / "assets/objects/gasoline_powered_post_driver.usd"
TRACKED_G1_ASSEMBLY = (
    ASSETS_ROOT
    / "robot/Tracked_G1_UpperBody_SolidBoom/Tracked_G1_UpperBody_SolidBoom.usd"
)

# Preserve the tracked W1's saved base pose, lift height, and authored upper-
# body mount. The G1 pelvis is synchronized to this exact mounting frame.
TRACKED_BASE_X = -0.9818309171279677
TRACKED_BASE_Y = 1.868526199382671
TRACKED_BASE_YAW = -0.5241687167665526
TRACKED_LIFT_HEIGHT = -0.4541666666666663


def _read_authored_g1_mount() -> tuple[
    tuple[float, float, float], tuple[float, float, float, float]
]:
    """Read the user-adjustable G1 mount directly from the combined USD."""

    stage = Usd.Stage.Open(str(TRACKED_G1_ASSEMBLY))
    if stage is None:
        raise RuntimeError(f"Failed to open combined robot USD: {TRACKED_G1_ASSEMBLY}")
    prim = stage.GetPrimAtPath("/Assembly/G1UpperBody")
    if not prim:
        raise RuntimeError("Combined robot USD is missing /Assembly/G1UpperBody")
    transform = Gf.Transform(UsdGeom.Xformable(prim).GetLocalTransformation())
    scale = transform.GetScale()
    if any(abs(float(value) - 1.0) > 1.0e-5 for value in scale):
        raise RuntimeError(
            "G1UpperBody scale must remain (1, 1, 1); adjust Translate/Orient only. "
            f"Current scale: {tuple(scale)}"
        )
    translation = transform.GetTranslation()
    quaternion = transform.GetRotation().GetQuat()
    imaginary = quaternion.GetImaginary()
    return (
        tuple(float(value) for value in translation),
        (
            float(quaternion.GetReal()),
            float(imaginary[0]),
            float(imaginary[1]),
            float(imaginary[2]),
        ),
    )


G1_MOUNT_LOCAL_POSITION, G1_MOUNT_LOCAL_ORIENTATION = _read_authored_g1_mount()
W1_UPPER_MOUNT_LOCAL = G1_MOUNT_LOCAL_POSITION


def _multiply_quaternions_wxyz(left, right):
    lw, lx, ly, lz = left
    rw, rx, ry, rz = right
    return (
        lw * rw - lx * rx - ly * ry - lz * rz,
        lw * rx + lx * rw + ly * rz - lz * ry,
        lw * ry - lx * rz + ly * rw + lz * rx,
        lw * rz + lx * ry - ly * rx + lz * rw,
    )


_base_cosine = math.cos(TRACKED_BASE_YAW)
_base_sine = math.sin(TRACKED_BASE_YAW)
G1_MOUNT_POSITION = (
    TRACKED_BASE_X
    + _base_cosine * G1_MOUNT_LOCAL_POSITION[0]
    - _base_sine * G1_MOUNT_LOCAL_POSITION[1],
    TRACKED_BASE_Y
    + _base_sine * G1_MOUNT_LOCAL_POSITION[0]
    + _base_cosine * G1_MOUNT_LOCAL_POSITION[1],
    G1_MOUNT_LOCAL_POSITION[2] + TRACKED_LIFT_HEIGHT,
)
G1_MOUNT_ORIENTATION = _multiply_quaternions_wxyz(
    (
        math.cos(0.5 * TRACKED_BASE_YAW),
        0.0,
        0.0,
        math.sin(0.5 * TRACKED_BASE_YAW),
    ),
    G1_MOUNT_LOCAL_ORIENTATION,
)

for asset_path in (
    ENVIRONMENT,
    DRILLING_OBJECT,
    TRACKED_G1_ASSEMBLY,
):
    if not asset_path.is_file():
        raise FileNotFoundError(asset_path)


def zero_reward(env) -> torch.Tensor:
    return torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)


def never_terminate(env) -> torch.Tensor:
    return torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)


def spawn_combined_tracked_g1(
    prim_path: str,
    cfg: UsdFileCfg,
    translation=None,
    orientation=None,
    **kwargs,
) -> Usd.Prim:
    """Spawn the combined assembly and return its embedded G1 prim."""

    if os.environ.get("G1_TRACKED_DISABLE_BASE", "0") == "1":
        sim_utils.create_prim(
            prim_path,
            prim_type="Xform",
            translation=translation,
            orientation=orientation,
        )
        return sim_utils.get_current_stage().GetPrimAtPath(prim_path)

    # ArticulationCfg calls this function for the embedded G1 path. Spawn the
    # complete USD at its assembly parent instead, then let IsaacLab bind the
    # articulation that already exists inside it.
    assembly_path = "/World/TrackedBase"
    root = spawn_from_usd(
        assembly_path,
        cfg,
        translation=(TRACKED_BASE_X, TRACKED_BASE_Y, 0.0),
        orientation=(
            math.cos(0.5 * TRACKED_BASE_YAW),
            0.0,
            0.0,
            math.sin(0.5 * TRACKED_BASE_YAW),
        ),
        **kwargs,
    )
    carriage = root.GetStage().GetPrimAtPath(f"{root.GetPath()}/LiftCarriage")
    if not carriage:
        raise RuntimeError(f"Tracked W1 lift carriage is missing under {root.GetPath()}")
    xform = UsdGeom.Xformable(carriage)
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
    translate_op.Set((0.0, 0.0, TRACKED_LIFT_HEIGHT))

    # The stored mount is authored for lift height zero. Align its initial
    # stage pose with the saved lift height.
    g1_upper = root.GetStage().GetPrimAtPath(f"{root.GetPath()}/G1UpperBody")
    if not g1_upper:
        raise RuntimeError(f"Combined G1 upper body is missing under {root.GetPath()}")
    g1_xform = UsdGeom.Xformable(g1_upper)
    g1_translate_op = next(
        (
            op
            for op in g1_xform.GetOrderedXformOps()
            if op.GetOpName() == "xformOp:translate"
        ),
        None,
    )
    if g1_translate_op is None:
        g1_translate_op = g1_xform.AddTranslateOp()
    g1_translate_op.Set(
        Gf.Vec3d(
            G1_MOUNT_LOCAL_POSITION[0],
            G1_MOUNT_LOCAL_POSITION[1],
            G1_MOUNT_LOCAL_POSITION[2] + TRACKED_LIFT_HEIGHT,
        )
    )
    root_joint = UsdPhysics.Joint(
        root.GetStage().GetPrimAtPath(f"{root.GetPath()}/G1UpperBody/root_joint")
    )
    if not root_joint:
        raise RuntimeError("Combined G1 root_joint is missing")
    # PhysX articulation roots cannot use a separate rigid body as body0.
    # Restore the official fixed-base root-joint topology here; the runtime
    # controller synchronizes the G1 root with the carriage kinematically.
    root_joint.GetBody0Rel().SetTargets([])
    root_joint.GetLocalPos0Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
    root_joint.GetLocalRot0Attr().Set(
        Gf.Quatf(1.0, Gf.Vec3f(0.0, 0.0, 0.0))
    )
    print(
        "[tracked-g1] Loaded combined USD: "
        f"{TRACKED_G1_ASSEMBLY}; authored mount="
        f"{G1_MOUNT_LOCAL_POSITION}"
    )
    return g1_upper


def make_embedded_g1_upper_body_cfg() -> ArticulationCfg:
    cfg = G1RobotPresets.g1_29dof_dex1_base_fix(
        init_pos=G1_MOUNT_POSITION,
        init_rot=G1_MOUNT_ORIENTATION,
    )
    cfg.prim_path = "/World/TrackedBase/G1UpperBody"
    cfg.spawn = UsdFileCfg(
        usd_path=str(TRACKED_G1_ASSEMBLY),
        func=spawn_combined_tracked_g1,
    )
    return cfg


@configclass
class MobileDrillingSceneCfg(InteractiveSceneCfg):
    # Load the authored WorldComposer composition directly. It already
    # contains the NuRec visual layer, invisible collision mesh, and their
    # coordinate-system transforms. The authored composition uses -90 deg X,
    # while the saved task placements use the original task's +90 deg X frame;
    # an outer 180 deg X rotation preserves those calibrated world positions.
    environment = AssetBaseCfg(
        prim_path="/World/envs/env_.*/Environment",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0),
            rot=(0.0, 1.0, 0.0, 0.0),
        ),
        spawn=UsdFileCfg(usd_path=str(ENVIRONMENT)),
    )

    # Keep the drilling asset at the placement saved by the original task.
    object = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Object",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(-3.2129466534, 3.0915591717, 0.3627696514),
            rot=(-0.4541494846, 0.0928831920, -0.0104347719, 0.8860092163),
        ),
        spawn=UsdFileCfg(
            usd_path=str(DRILLING_OBJECT),
            scale=(0.9, 0.9, 0.9),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
    )

    # Wrap the articulation already present inside the combined USD. No second
    # G1 asset is spawned at runtime.
    robot: ArticulationCfg = make_embedded_g1_upper_body_cfg()

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
    """Native reset hook required while IsaacLab initializes its managers."""

    reset_object = EventTermCfg(
        func=base_mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("object"),
        },
    )


@configclass
class MobileDrillingG129Dex1EnvCfg(ManagerBasedRLEnvCfg):
    scene: MobileDrillingSceneCfg = MobileDrillingSceneCfg(
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
        from .tracked_g1_base_controller import TrackedG1BaseController

        return TrackedG1BaseController(env)

    def __post_init__(self):
        self.decimation = 2
        self.episode_length_s = 1.0e9
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625

        self.event_manager = SimpleEventManager()
        self.event_manager.register(
            "reset_object_self",
            SimpleEvent(
                func=lambda env: base_mdp.reset_root_state_uniform(
                    env,
                    torch.arange(env.num_envs, device=env.device),
                    pose_range={},
                    velocity_range={},
                    asset_cfg=SceneEntityCfg("object"),
                )
            ),
        )
        self.event_manager.register(
            "reset_all_self",
            SimpleEvent(
                func=lambda env: base_mdp.reset_scene_to_default(
                    env,
                    torch.arange(env.num_envs, device=env.device),
                )
            ),
        )
