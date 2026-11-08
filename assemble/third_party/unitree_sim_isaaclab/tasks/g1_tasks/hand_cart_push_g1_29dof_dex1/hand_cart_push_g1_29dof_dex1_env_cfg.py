"""Tracked G1 task for pushing a non-articulated hand cart forward."""

from pathlib import Path
import math
import os

import torch
from pxr import Gf, PhysxSchema, Usd, UsdGeom, UsdPhysics, UsdShade

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
    ENVIRONMENT,
    G1_MOUNT_LOCAL_ORIENTATION,
    G1_MOUNT_LOCAL_POSITION,
    TRACKED_G1_ASSEMBLY,
    make_embedded_g1_upper_body_cfg,
    spawn_combined_tracked_g1,
)


ASSEMBLE_ROOT = Path(__file__).resolve().parents[5]
TASK_ROOT = ASSEMBLE_ROOT / "task" / "g1_hand_cart_push_task"
HAND_CART_OBJECT = ASSEMBLE_ROOT / "objects" / "two_wheel_steel_hand_cart.usd"
TASK_ID = "Isaac-HandCartPush-G129-Dex1-Joint"

# Captured roof-bolt task base pose, shifted 1.8 m opposite the robot's forward
# direction.  The long cart then has room both to fit and to travel 0.5 m.
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
    TASK_TRACKED_BASE_X
    + _base_cosine * G1_MOUNT_LOCAL_POSITION[0]
    - _base_sine * G1_MOUNT_LOCAL_POSITION[1],
    TASK_TRACKED_BASE_Y
    + _base_sine * G1_MOUNT_LOCAL_POSITION[0]
    + _base_cosine * G1_MOUNT_LOCAL_POSITION[1],
    G1_MOUNT_LOCAL_POSITION[2] + TASK_TRACKED_LIFT_HEIGHT,
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

# The source cart is Y-up.  +90 degrees around local X maps source Y to world
# Z.  The following world yaw maps the cart's source +Z (the tray direction)
# to the G1 forward direction, leaving the two handles facing the robot.
CART_DISTANCE_FROM_G1_M = float(os.environ.get("G1_HAND_CART_DISTANCE_M", "0.82"))
CART_POSITION = (
    TASK_G1_MOUNT_POSITION[0] + TASK_FORWARD_X * CART_DISTANCE_FROM_G1_M,
    TASK_G1_MOUNT_POSITION[1] + TASK_FORWARD_Y * CART_DISTANCE_FROM_G1_M,
    # The scanned floor's collision surface at this location sits about
    # 4.8 cm above its visual/world datum.  Starting at the raw asset value
    # made the SDF overlap the floor, so PhysX depenetrated the whole cart
    # sideways after every reset.  This contact-calibrated height keeps the
    # same XY layout while avoiding that artificial launch impulse.
    0.2930,
)
_cart_world_yaw = TASK_FORWARD_YAW + 0.5 * math.pi
CART_ORIENTATION = _multiply_quaternions_wxyz(
    (
        math.cos(0.5 * _cart_world_yaw),
        0.0,
        0.0,
        math.sin(0.5 * _cart_world_yaw),
    ),
    (math.sqrt(0.5), math.sqrt(0.5), 0.0, 0.0),
)

CART_MASS_KG = float(os.environ.get("G1_HAND_CART_MASS_KG", "12.0"))
CART_STATIC_FRICTION = float(os.environ.get("G1_HAND_CART_STATIC_FRICTION", "0.25"))
CART_DYNAMIC_FRICTION = float(os.environ.get("G1_HAND_CART_DYNAMIC_FRICTION", "0.20"))
# A slightly wider predictive contact shell is important for this asset: its
# two visual wheels are fused into one SDF and the scanned floor is uneven.
# The rest offset stays zero, so this improves contact generation without
# changing the visible resting gap.
CART_CONTACT_OFFSET = 0.010
CART_MAX_DEPENETRATION_VELOCITY = 0.20

# The source cart has one fused SDF collider.  Its roughly 24 mm diameter
# scanned handle tubes are too thin for that SDF to preserve reliably, which
# lets the position-controlled Dex1 fingers pass through the visible handles.
# These task-only capsules follow the two handle centerlines without changing
# the source asset or its appearance.  The 13 mm radius is deliberately only
# about 2--3 mm outside the scan and gives PhysX a continuous analytic surface.
CART_HANDLE_COLLIDER_RADIUS_M = float(
    os.environ.get("G1_HAND_CART_HANDLE_COLLIDER_RADIUS_M", "0.013")
)
CART_HANDLE_CONTACT_OFFSET_M = float(
    os.environ.get("G1_HAND_CART_HANDLE_CONTACT_OFFSET_M", "0.003")
)
CART_HANDLE_CENTERLINES = {
    "Left": (
        (0.232, 0.171, -0.493),
        (0.233, 0.196, -0.455),
        (0.235, 0.218, -0.420),
        (0.235, 0.218, -0.145),
    ),
    "Right": (
        (-0.239, 0.173, -0.493),
        (-0.240, 0.198, -0.455),
        (-0.236, 0.226, -0.415),
        (-0.228, 0.220, -0.145),
    ),
}

for asset_path in (ENVIRONMENT, TRACKED_G1_ASSEMBLY, HAND_CART_OBJECT):
    if not asset_path.is_file():
        raise FileNotFoundError(asset_path)


def zero_reward(env) -> torch.Tensor:
    return torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)


def never_terminate(env) -> torch.Tensor:
    return torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)


def spawn_hand_cart_tracked_g1(
    prim_path: str,
    cfg: UsdFileCfg,
    translation=None,
    orientation=None,
    **kwargs,
) -> Usd.Prim:
    """Spawn the shared tracked G1 at this task's reset pose."""

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
        raise RuntimeError("Tracked base hierarchy is incomplete")

    root_xform = UsdGeom.Xformable(root)
    root_translate = next(
        (op for op in root_xform.GetOrderedXformOps() if op.GetOpName() == "xformOp:translate"),
        None,
    ) or root_xform.AddTranslateOp()
    root_orient = next(
        (op for op in root_xform.GetOrderedXformOps() if op.GetOpName() == "xformOp:orient"),
        None,
    ) or root_xform.AddOrientOp()
    root_translate.Set((TASK_TRACKED_BASE_X, TASK_TRACKED_BASE_Y, 0.0))
    root_orient.Set(
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
        (op for op in g1_xform.GetOrderedXformOps() if op.GetOpName() == "xformOp:translate"),
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
        "[hand-cart-push] Tracked G1 placed "
        f"{ROBOT_BACK_OFFSET_M:.2f} m behind the reference pose; "
        f"lift={TASK_TRACKED_LIFT_HEIGHT:.2f} m."
    )
    return g1_upper


def make_hand_cart_g1_cfg():
    cfg = make_embedded_g1_upper_body_cfg()
    cfg.init_state.pos = TASK_G1_MOUNT_POSITION
    cfg.init_state.rot = TASK_G1_MOUNT_ORIENTATION
    cfg.spawn.func = spawn_hand_cart_tracked_g1
    return cfg


def _add_analytic_handle_colliders(root: Usd.Prim) -> int:
    """Attach invisible analytic colliders to the cart's two thin handles."""

    if CART_HANDLE_COLLIDER_RADIUS_M <= 0.0:
        raise ValueError("G1_HAND_CART_HANDLE_COLLIDER_RADIUS_M must be positive")
    if CART_HANDLE_CONTACT_OFFSET_M <= 0.0:
        raise ValueError("G1_HAND_CART_HANDLE_CONTACT_OFFSET_M must be positive")

    stage = root.GetStage()
    root_path = str(root.GetPath())
    collider_root = UsdGeom.Xform.Define(
        stage, f"{root_path}/TaskHandleColliders"
    )
    collider_root.GetPrim().SetCustomDataByKey(
        "handCartPurpose", "analytic collision for thin scanned handles"
    )

    physics_material = UsdShade.Material.Define(
        stage, f"{root_path}/TaskHandlePhysicsMaterial"
    )
    material_api = UsdPhysics.MaterialAPI.Apply(physics_material.GetPrim())
    material_api.CreateStaticFrictionAttr().Set(CART_STATIC_FRICTION)
    material_api.CreateDynamicFrictionAttr().Set(CART_DYNAMIC_FRICTION)
    material_api.CreateRestitutionAttr().Set(0.02)
    physx_material = PhysxSchema.PhysxMaterialAPI.Apply(
        physics_material.GetPrim()
    )
    physx_material.CreateFrictionCombineModeAttr().Set("max")
    physx_material.CreateRestitutionCombineModeAttr().Set("min")

    collider_count = 0
    for side, centerline in CART_HANDLE_CENTERLINES.items():
        for segment_index, (start_raw, end_raw) in enumerate(
            zip(centerline[:-1], centerline[1:]), start=1
        ):
            start = Gf.Vec3d(*start_raw)
            end = Gf.Vec3d(*end_raw)
            delta = end - start
            length = delta.GetLength()
            if length <= 1.0e-6:
                raise RuntimeError(
                    f"Degenerate {side} hand-cart handle segment {segment_index}"
                )

            capsule = UsdGeom.Capsule.Define(
                stage,
                f"{collider_root.GetPath()}/{side}Segment{segment_index}",
            )
            capsule.CreateAxisAttr(UsdGeom.Tokens.z)
            capsule.CreateRadiusAttr(CART_HANDLE_COLLIDER_RADIUS_M)
            capsule.CreateHeightAttr(length)
            capsule.CreateVisibilityAttr(UsdGeom.Tokens.invisible)
            capsule.AddTranslateOp().Set(0.5 * (start + end))
            capsule.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(
                Gf.Rotation(Gf.Vec3d(0.0, 0.0, 1.0), delta / length).GetQuat()
            )

            prim = capsule.GetPrim()
            collision = UsdPhysics.CollisionAPI.Apply(prim)
            collision.CreateCollisionEnabledAttr().Set(True)
            physx_collision = PhysxSchema.PhysxCollisionAPI.Apply(prim)
            physx_collision.CreateContactOffsetAttr().Set(
                CART_HANDLE_CONTACT_OFFSET_M
            )
            physx_collision.CreateRestOffsetAttr().Set(0.0)
            UsdShade.MaterialBindingAPI.Apply(prim).Bind(
                physics_material, materialPurpose="physics"
            )
            prim.SetCustomDataByKey("handCartHandleSide", side.lower())
            collider_count += 1

    return collider_count


def spawn_push_cart(
    prim_path: str,
    cfg: UsdFileCfg,
    translation=None,
    orientation=None,
    **kwargs,
) -> Usd.Prim:
    """Spawn the unchanged visual asset and apply task-only physics overrides."""

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
    mass.CreateMassAttr().Set(CART_MASS_KG)

    physx_body = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    physx_body.CreateDisableGravityAttr().Set(False)
    physx_body.CreateLinearDampingAttr().Set(0.10)
    physx_body.CreateAngularDampingAttr().Set(0.50)
    physx_body.CreateEnableSpeculativeCCDAttr().Set(True)
    physx_body.CreateSolverPositionIterationCountAttr().Set(16)
    physx_body.CreateSolverVelocityIterationCountAttr().Set(4)
    physx_body.CreateMaxDepenetrationVelocityAttr().Set(
        CART_MAX_DEPENETRATION_VELOCITY
    )
    # The source wheels are part of the same SDF rather than revolute bodies.
    # On the reconstructed floor, ordinary thresholds let small alternating
    # wheel contacts keep the cart awake and slowly creep.  These thresholds
    # park the cart at rest; contact/force from the robot wakes it normally.
    physx_body.CreateSleepThresholdAttr().Set(0.05)
    physx_body.CreateStabilizationThresholdAttr().Set(0.01)
    # Bit 0=X and bit 1=Y.  Keep yaw free while preventing the non-articulated
    # visual wheels from allowing the cart to pitch or roll.
    physx_body.CreateLockedRotAxisAttr().Set(3)

    collider_count = 0
    material_count = 0
    for prim in Usd.PrimRange(root):
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            collision = PhysxSchema.PhysxCollisionAPI.Apply(prim)
            collision.CreateContactOffsetAttr().Set(CART_CONTACT_OFFSET)
            collision.CreateRestOffsetAttr().Set(0.0)
            if prim.IsA(UsdGeom.Mesh):
                # The source layer authors the USD approximation token, but a
                # referenced dynamic mesh also needs the PhysX SDF schema on
                # the composed prim. Without it PhysX silently falls back to a
                # convex hull and closes the cart's open geometry.
                mesh_collision = UsdPhysics.MeshCollisionAPI.Apply(prim)
                mesh_collision.CreateApproximationAttr().Set("sdf")
                sdf_collision = PhysxSchema.PhysxSDFMeshCollisionAPI.Apply(prim)
                sdf_collision.CreateSdfResolutionAttr().Set(256)
            collider_count += 1
        if prim.HasAPI(UsdPhysics.MaterialAPI):
            material = UsdPhysics.MaterialAPI.Apply(prim)
            material.CreateStaticFrictionAttr().Set(CART_STATIC_FRICTION)
            material.CreateDynamicFrictionAttr().Set(CART_DYNAMIC_FRICTION)
            material.CreateRestitutionAttr().Set(0.02)
            physx_material = PhysxSchema.PhysxMaterialAPI.Apply(prim)
            # Keep the authored cart coefficients above, but do not let a
            # low/default material on the scanned floor reduce them further.
            # This is necessary for the fused non-rolling wheels to remain at
            # rest on the slightly sloped reconstruction.
            physx_material.CreateFrictionCombineModeAttr().Set("max")
            physx_material.CreateRestitutionCombineModeAttr().Set("min")
            material_count += 1
    if collider_count != 1:
        raise RuntimeError(f"Expected one hand-cart SDF collider, found {collider_count}")
    if material_count != 1:
        raise RuntimeError(f"Expected one hand-cart physics material, found {material_count}")
    analytic_handle_colliders = _add_analytic_handle_colliders(root)
    if analytic_handle_colliders != 6:
        raise RuntimeError(
            "Expected six analytic hand-cart handle capsules, found "
            f"{analytic_handle_colliders}"
        )
    print(
        "[hand-cart-push] Cart physics overridden: "
        f"mass={CART_MASS_KG:.1f} kg, friction="
        f"{CART_STATIC_FRICTION:.2f}/{CART_DYNAMIC_FRICTION:.2f}, "
        "gravity on, SDF collider retained, six analytic handle capsules "
        "added, roll/pitch locked."
    )
    return root


@configclass
class HandCartPushSceneCfg(InteractiveSceneCfg):
    environment = AssetBaseCfg(
        prim_path="/World/envs/env_.*/Environment",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0),
            rot=(0.0, 1.0, 0.0, 0.0),
        ),
        spawn=UsdFileCfg(usd_path=str(ENVIRONMENT)),
    )

    hand_cart = RigidObjectCfg(
        prim_path="/World/envs/env_.*/HandCart",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=CART_POSITION,
            rot=CART_ORIENTATION,
        ),
        spawn=UsdFileCfg(
            usd_path=str(HAND_CART_OBJECT),
            func=spawn_push_cart,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                disable_gravity=False,
                linear_damping=0.10,
                angular_damping=0.50,
                max_depenetration_velocity=CART_MAX_DEPENETRATION_VELOCITY,
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=4,
                sleep_threshold=0.05,
                stabilization_threshold=0.01,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=CART_MASS_KG),
            collision_props=sim_utils.CollisionPropertiesCfg(
                contact_offset=CART_CONTACT_OFFSET,
                rest_offset=0.0,
            ),
        ),
    )

    robot = make_hand_cart_g1_cfg()

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
        pos_offset=(-0.7, 3.8, 2.3),
        rot_offset=(0.105, 0.205, 0.675, -0.701),
        focal_length=18.0,
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
    reset_hand_cart = EventTermCfg(
        func=base_mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("hand_cart"),
        },
    )


def _runtime_controller(env):
    controller = getattr(env, "_hand_cart_push_runtime_controller", None)
    if controller is None:
        raise RuntimeError("Hand-cart push runtime controller is not initialized")
    return controller


def reset_task_objects(env) -> None:
    _runtime_controller(env).reset_task_objects()


def reset_everything(env) -> None:
    _runtime_controller(env).reset_everything()


@configclass
class HandCartPushG129Dex1EnvCfg(ManagerBasedRLEnvCfg):
    scene: HandCartPushSceneCfg = HandCartPushSceneCfg(
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
        from .hand_cart_push_controller import HandCartPushController

        controller = HandCartPushController(env)
        env._hand_cart_push_runtime_controller = controller
        return controller

    def __post_init__(self):
        self.decimation = 2
        self.episode_length_s = 1.0e9
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.enable_stabilization = True
        self.sim.physx.enable_external_forces_every_iteration = True
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024
        # Keep wheel/floor contact points in one friction patch.  The 1 mm
        # value used by insertion tasks fragmented this cart's broad SDF
        # contact and let it creep across the scanned floor.
        self.sim.physx.friction_correlation_distance = 0.025
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
