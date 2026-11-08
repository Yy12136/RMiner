"""Tracked G1 task for lifting a horizontal roof bolt and standing it upright."""

from __future__ import annotations

import math
from pathlib import Path

import torch
from pxr import Gf, PhysxSchema, Usd, UsdPhysics, UsdShade

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
from tasks.g1_tasks.roof_bolt_sleeve_g1_29dof_dex1.roof_bolt_sleeve_g1_29dof_dex1_env_cfg import (
    ASSETS_ROOT,
    ENVIRONMENT,
    G1_FORWARD_YAW,
    TASK_G1_MOUNT_ORIENTATION,
    TASK_G1_MOUNT_POSITION,
    TASK_TRACKED_BASE_X,
    TASK_TRACKED_BASE_Y,
    TASK_TRACKED_BASE_YAW,
    TASK_TRACKED_LIFT_HEIGHT,
    TRACKED_G1_ASSEMBLY,
    make_shifted_g1_upper_body_cfg,
)


ASSEMBLE_ROOT = Path(__file__).resolve().parents[5]
TASK_ROOT = ASSEMBLE_ROOT / "task" / "g1_roof_bolt_upright_task"
TASK_ID = "Isaac-RoofBoltUpright-G129-Dex1-Joint"
ROD_OBJECT = ASSETS_ROOT / "object/M20_Bolt_Chain_D45_L1450.usda"

ROD_LENGTH = 1.45
ROD_HALF_LENGTH = 0.5 * ROD_LENGTH
ROD_RADIUS = 0.0225
ROD_MASS = 0.30
ROD_INERTIA_TRANSVERSE = 0.05260046875
ROD_INERTIA_AXIAL = 0.0000759375
ROD_CONTACT_OFFSET = 0.002

SUPPORT_SIZE = (0.12, 0.12, 0.15)
# Keep the invisible supports 20 cm above world Z=0.  They are kinematic
# colliders, so they can remain suspended while physically carrying the rod.
SUPPORT_BOTTOM_Z = 0.20
SUPPORT_CENTER_Z = SUPPORT_BOTTOM_Z + 0.5 * SUPPORT_SIZE[2]
SUPPORT_TOP_Z = SUPPORT_BOTTOM_Z + SUPPORT_SIZE[2]
SUPPORT_AXIS_OFFSET = 0.55
ROD_BOTTOM_Z = SUPPORT_TOP_Z
ROD_CENTER_Z = ROD_BOTTOM_Z + ROD_RADIUS
ROD_FORWARD_DISTANCE = 0.65
RESET_RANDOM_XY_LIMIT = 0.03

TASK_FORWARD_X = math.cos(G1_FORWARD_YAW)
TASK_FORWARD_Y = math.sin(G1_FORWARD_YAW)
TASK_LEFT_X = -TASK_FORWARD_Y
TASK_LEFT_Y = TASK_FORWARD_X

ROD_POSITION = (
    TASK_G1_MOUNT_POSITION[0] + TASK_FORWARD_X * ROD_FORWARD_DISTANCE,
    TASK_G1_MOUNT_POSITION[1] + TASK_FORWARD_Y * ROD_FORWARD_DISTANCE,
    ROD_CENTER_Z,
)

# Rotate the source asset's local +Z long axis onto the robot's left direction.
_half_turn = math.sqrt(0.5)
ROD_ORIENTATION = (
    _half_turn,
    -TASK_LEFT_Y * _half_turn,
    TASK_LEFT_X * _half_turn,
    0.0,
)
SUPPORT_ORIENTATION = (1.0, 0.0, 0.0, 0.0)
LEFT_SUPPORT_POSITION = (
    ROD_POSITION[0] + TASK_LEFT_X * SUPPORT_AXIS_OFFSET,
    ROD_POSITION[1] + TASK_LEFT_Y * SUPPORT_AXIS_OFFSET,
    SUPPORT_CENTER_Z,
)
RIGHT_SUPPORT_POSITION = (
    ROD_POSITION[0] - TASK_LEFT_X * SUPPORT_AXIS_OFFSET,
    ROD_POSITION[1] - TASK_LEFT_Y * SUPPORT_AXIS_OFFSET,
    SUPPORT_CENTER_Z,
)

for asset_path in (ENVIRONMENT, ROD_OBJECT, TRACKED_G1_ASSEMBLY):
    if not asset_path.is_file():
        raise FileNotFoundError(asset_path)


def zero_reward(env) -> torch.Tensor:
    return torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)


def never_terminate(env) -> torch.Tensor:
    return torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)


def spawn_dynamic_roof_bolt(
    prim_path: str,
    cfg: UsdFileCfg,
    translation=None,
    orientation=None,
    **kwargs,
) -> Usd.Prim:
    """Spawn the reusable visual assembly as one lightweight dynamic body."""

    root = spawn_from_usd(
        prim_path,
        cfg,
        translation=translation,
        orientation=orientation,
        **kwargs,
    )
    removed_segment_bodies = 0
    removed_visual_colliders = 0
    for prim in Usd.PrimRange(root):
        if prim != root and prim.HasAPI(UsdPhysics.RigidBodyAPI):
            if prim.RemoveAPI(UsdPhysics.RigidBodyAPI):
                removed_segment_bodies += 1

        # The source bolt segment meshes carry CollisionAPI even though their
        # collisionEnabled attribute is false.  Once the segment bodies are
        # collapsed into one dynamic root, PhysX 5.x still attempts to cook
        # those triangle meshes.  Remove the APIs entirely and retain only the
        # purpose=guide continuous cylinder below.
        if prim == root or not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        enabled = prim.GetAttribute("physics:collisionEnabled")
        if enabled and enabled.Get() is False:
            if prim.HasAPI(PhysxSchema.PhysxCollisionAPI):
                prim.RemoveAPI(PhysxSchema.PhysxCollisionAPI)
            if prim.HasAPI(UsdPhysics.MeshCollisionAPI):
                prim.RemoveAPI(UsdPhysics.MeshCollisionAPI)
            if prim.RemoveAPI(UsdPhysics.CollisionAPI):
                removed_visual_colliders += 1

    rigid = UsdPhysics.RigidBodyAPI.Apply(root)
    rigid.CreateRigidBodyEnabledAttr().Set(True)
    rigid.CreateKinematicEnabledAttr().Set(False)
    mass = UsdPhysics.MassAPI.Apply(root)
    mass.CreateMassAttr().Set(ROD_MASS)
    mass.CreateCenterOfMassAttr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
    mass.CreateDiagonalInertiaAttr().Set(
        Gf.Vec3f(
            ROD_INERTIA_TRANSVERSE,
            ROD_INERTIA_TRANSVERSE,
            ROD_INERTIA_AXIAL,
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

    # UsdFileCfg does not accept a physics_material field in this IsaacLab
    # version.  Author and bind the intended rod material explicitly instead.
    material = UsdShade.Material.Define(
        root.GetStage(), root.GetPath().AppendPath("PhysicsMaterials/Rod")
    )
    material_api = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    material_api.CreateStaticFrictionAttr().Set(1.5)
    material_api.CreateDynamicFrictionAttr().Set(1.2)
    material_api.CreateRestitutionAttr().Set(0.0)
    physx_material = PhysxSchema.PhysxMaterialAPI.Apply(material.GetPrim())
    physx_material.CreateFrictionCombineModeAttr().Set("max")
    physx_material.CreateRestitutionCombineModeAttr().Set("min")

    enabled_colliders = 0
    for prim in Usd.PrimRange(root):
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        enabled = prim.GetAttribute("physics:collisionEnabled")
        if enabled and enabled.Get() is False:
            continue
        collision = PhysxSchema.PhysxCollisionAPI.Apply(prim)
        collision.CreateContactOffsetAttr().Set(ROD_CONTACT_OFFSET)
        collision.CreateRestOffsetAttr().Set(0.0)
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(
            material, materialPurpose="physics"
        )
        enabled_colliders += 1
    if enabled_colliders != 1:
        raise RuntimeError(
            f"Expected one continuous roof-bolt collider, found {enabled_colliders}"
        )
    print(
        "[roof-bolt-upright] Dynamic rod configured: "
        f"mass={ROD_MASS:.2f} kg, continuous colliders={enabled_colliders}, "
        f"removed segment rigid bodies={removed_segment_bodies}, "
        f"removed visual colliders={removed_visual_colliders}."
    )
    return root


def _support_spawn_cfg() -> sim_utils.CuboidCfg:
    return sim_utils.CuboidCfg(
        size=SUPPORT_SIZE,
        visible=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            kinematic_enabled=True,
            disable_gravity=True,
        ),
        mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
        collision_props=sim_utils.CollisionPropertiesCfg(
            collision_enabled=True,
            contact_offset=ROD_CONTACT_OFFSET,
            rest_offset=0.0,
        ),
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="max",
            restitution_combine_mode="min",
            static_friction=2.0,
            dynamic_friction=1.5,
            restitution=0.0,
        ),
    )


@configclass
class RoofBoltUprightSceneCfg(InteractiveSceneCfg):
    environment = AssetBaseCfg(
        prim_path="/World/envs/env_.*/Environment",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0),
            rot=(0.0, 1.0, 0.0, 0.0),
        ),
        spawn=UsdFileCfg(usd_path=str(ENVIRONMENT)),
    )

    rod = RigidObjectCfg(
        prim_path="/World/envs/env_.*/RoofBolt",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=ROD_POSITION,
            rot=ROD_ORIENTATION,
        ),
        spawn=UsdFileCfg(
            usd_path=str(ROD_OBJECT),
            func=spawn_dynamic_roof_bolt,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                kinematic_enabled=False,
                disable_gravity=False,
                linear_damping=0.05,
                angular_damping=0.05,
                max_linear_velocity=3.0,
                max_angular_velocity=720.0,
                max_depenetration_velocity=1.0,
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=4,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                contact_offset=ROD_CONTACT_OFFSET,
                rest_offset=0.0,
            ),
        ),
    )

    left_support = RigidObjectCfg(
        prim_path="/World/envs/env_.*/LeftRodSupport",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=LEFT_SUPPORT_POSITION,
            rot=SUPPORT_ORIENTATION,
        ),
        spawn=_support_spawn_cfg(),
    )
    right_support = RigidObjectCfg(
        prim_path="/World/envs/env_.*/RightRodSupport",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=RIGHT_SUPPORT_POSITION,
            rot=SUPPORT_ORIENTATION,
        ),
        spawn=_support_spawn_cfg(),
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
    reset_rod = EventTermCfg(
        func=base_mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("rod"),
        },
    )
    reset_left_support = EventTermCfg(
        func=base_mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("left_support"),
        },
    )
    reset_right_support = EventTermCfg(
        func=base_mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("right_support"),
        },
    )


def _runtime_controller(env):
    controller = getattr(env, "_roof_bolt_upright_runtime_controller", None)
    if controller is None:
        raise RuntimeError("Roof-bolt upright runtime controller is not initialized")
    return controller


def reset_task_objects(env) -> None:
    _runtime_controller(env).reset_task_objects()


def reset_everything(env) -> None:
    _runtime_controller(env).reset_everything()


@configclass
class RoofBoltUprightG129Dex1EnvCfg(ManagerBasedRLEnvCfg):
    scene: RoofBoltUprightSceneCfg = RoofBoltUprightSceneCfg(
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
        from .roof_bolt_upright_controller import RoofBoltUprightController

        controller = RoofBoltUprightController(env)
        env._roof_bolt_upright_runtime_controller = controller
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
