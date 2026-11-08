"""G1-29 + Dex1 tabletop grasping in the reconstructed task scene.

The robot, DDS action path, cameras, and official XR IK/controller are kept
identical to the mobile-drilling task.  This task adds a static worktable, an
upright cylindrical object, and a flexible 26-link rod-and-rope assembly.
"""

from pathlib import Path
import math
import os

import torch
from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics

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
from isaaclab.sim.spawners.sensors.sensors import spawn_camera
from isaaclab.sim.utils import clone
from isaaclab.utils import configclass

from tasks.common_config import CameraBaseCfg, CameraPresets, G1RobotPresets
from tasks.common_event.event_manager import SimpleEvent, SimpleEventManager
from tasks.common_observations.camera_state import get_camera_image
from tasks.common_observations.g1_29dof_state import get_robot_boy_joint_states
from tasks.common_observations.gripper_state import get_robot_gipper_joint_states
from tasks.g1_tasks.pick_place_cylinder_g1_29dof_dex1 import mdp
from .headlamp import preauthor_auxiliary_spotlights


ASSEMBLE_ROOT = Path(__file__).resolve().parents[5]
TASK_ROOT = ASSEMBLE_ROOT / "task" / "g1_tabletop_grasp_task"
ASSETS_ROOT = ASSEMBLE_ROOT / "assets-new"
ENVIRONMENT = (
    ASSETS_ROOT
    / "scene/RealScene04/real2sim_intermediate/RealScene04_Orin.usd"
)
WORKTABLE = (
    ASSETS_ROOT
    / "object/TableTrolley_B/TableTrolley_B01_01.usd"
)
LEIGUAN_OBJECT = ASSETS_ROOT / "object/leiguan.usd"
CHAGANSHENGZI_OBJECT = ASSETS_ROOT / "object/chaganshengzi_standing.usd"
CHAGANSHENGZI_HOLDER = ASSETS_ROOT / "object/chaganshengzi_holder.usd"
TRACKED_G1_ASSEMBLY = (
    ASSETS_ROOT
    / "robot/Tracked_G1_UpperBody_SolidBoom/Tracked_G1_UpperBody_SolidBoom.usd"
)


def _headlamp_env_float(name: str, default: float) -> float:
    value = float(os.environ.get(name, str(default)))
    if value < 0.0:
        raise ValueError(f"{name} must be non-negative; got {value}")
    return value


HEADLAMP_MODE = os.environ.get("G1_HEADLAMP", "off").strip().lower()
if HEADLAMP_MODE not in {"off", "on"}:
    raise ValueError(
        f"G1_HEADLAMP must be one of: off, on; got {HEADLAMP_MODE!r}"
    )
HEADLAMP_ENABLED = HEADLAMP_MODE == "on"


def _comma_floats(name: str, default: tuple[float, ...]) -> tuple[float, ...]:
    raw = os.environ.get(name)
    if raw is None:
        return default
    values = tuple(float(value.strip()) for value in raw.split(","))
    if len(values) != len(default):
        raise ValueError(f"{name} must contain {len(default)} comma-separated values")
    return values


THIRD_PERSON_ENABLED = os.environ.get("G1_ENABLE_THIRD_PERSON", "0").strip().lower() in {
    "1", "true", "yes", "on"
}
THIRD_PERSON_POSITION = _comma_floats(
    "G1_THIRD_PERSON_POSITION", (-1.75, 4.4, 1.9)
)
THIRD_PERSON_ROTATION = _comma_floats(
    "G1_THIRD_PERSON_ROTATION", (-0.00617, 0.00617, 0.70708, -0.70708)
)
THIRD_PERSON_FOCAL_LENGTH = _headlamp_env_float(
    "G1_THIRD_PERSON_FOCAL_LENGTH", 16.5
)
THIRD_PERSON_CONVENTION = os.environ.get("G1_THIRD_PERSON_CONVENTION", "ros").strip().lower()
HEADLAMP_INTENSITY = _headlamp_env_float(
    "G1_HEADLAMP_INTENSITY", 2000000.0
)
HEADLAMP_AMBIENT_INTENSITY = _headlamp_env_float(
    "G1_HEADLAMP_AMBIENT_INTENSITY", 80.0
)
HEADLAMP_CAMERA_EXPOSURE = float(
    os.environ.get("G1_HEADLAMP_CAMERA_EXPOSURE", "0.0")
)
if not -8.0 <= HEADLAMP_CAMERA_EXPOSURE <= 4.0:
    raise ValueError("G1_HEADLAMP_CAMERA_EXPOSURE must be in [-8, 4] EV")
HEADLAMP_NUREC_EMISSIVE_SCALE = _headlamp_env_float(
    "G1_HEADLAMP_NUREC_EMISSIVE_SCALE", 0.14
)
if HEADLAMP_NUREC_EMISSIVE_SCALE > 1.0:
    raise ValueError("G1_HEADLAMP_NUREC_EMISSIVE_SCALE must be in [0, 1]")
HEADLAMP_AIM_YAW_DEG = float(
    os.environ.get("G1_HEADLAMP_AIM_YAW_DEG", "0.0")
)
HEADLAMP_AIM_PITCH_DEG = float(
    os.environ.get("G1_HEADLAMP_AIM_PITCH_DEG", "0.0")
)
if not -30.0 <= HEADLAMP_AIM_YAW_DEG <= 30.0:
    raise ValueError("G1_HEADLAMP_AIM_YAW_DEG must be in [-30, 30]")
if not -20.0 <= HEADLAMP_AIM_PITCH_DEG <= 20.0:
    raise ValueError("G1_HEADLAMP_AIM_PITCH_DEG must be in [-20, 20]")

AUX_LIGHTS_MODE = os.environ.get("G1_AUX_LIGHTS", "off").strip().lower()
if AUX_LIGHTS_MODE not in {"off", "on"}:
    raise ValueError(
        f"G1_AUX_LIGHTS must be one of: off, on; got {AUX_LIGHTS_MODE!r}"
    )
AUX_LIGHTS_ENABLED = AUX_LIGHTS_MODE == "on"


def spawn_headlamp_fixed_exposure_camera(
    prim_path: str,
    cfg,
    translation=None,
    orientation=None,
    **kwargs,
) -> Usd.Prim:
    """Spawn the head camera with auto exposure disabled before RenderProduct creation."""

    prim = spawn_camera(
        prim_path,
        cfg,
        translation=translation,
        orientation=orientation,
        **kwargs,
    )
    for schema in (
        "OmniRtxCameraAutoExposureAPI_1",
        "OmniRtxCameraExposureAPI_1",
    ):
        if schema not in prim.GetAppliedSchemas():
            prim.AddAppliedSchema(schema)
    fixed_exposure = {
        "exposure": HEADLAMP_CAMERA_EXPOSURE,
        "exposure:fStop": 1.0,
        "exposure:iso": 0.0,
        "exposure:responsivity": 1.0,
        "exposure:time": 1.0,
    }
    for name, value in fixed_exposure.items():
        prim.CreateAttribute(name, Sdf.ValueTypeNames.Float).Set(value)
    prim.CreateAttribute(
        "omni:rtx:autoExposure:enabled", Sdf.ValueTypeNames.Bool
    ).Set(False)
    print(
        "[headlamp] Head camera auto exposure disabled before RenderProduct "
        f"creation: {prim_path}; exposure={HEADLAMP_CAMERA_EXPOSURE:.2f} EV"
    )
    return prim


def spawn_headlamp_environment(
    prim_path: str,
    cfg: UsdFileCfg,
    translation=None,
    orientation=None,
    **kwargs,
) -> Usd.Prim:
    """Spawn NuRec environment and dim its emissive field for the mine-dark preset."""

    root = spawn_from_usd(
        prim_path,
        cfg,
        translation=translation,
        orientation=orientation,
        **kwargs,
    )
    # Auxiliary-only demos also need the mine-dark environment; otherwise the
    # normal NuRec emissive background overwhelms the three scene spotlights.
    if not (HEADLAMP_ENABLED or AUX_LIGHTS_ENABLED):
        return root
    changed = 0
    for prim in Usd.PrimRange(root):
        role = prim.GetAttribute("fieldRole")
        if not role or role.Get() != "emissiveColor":
            continue
        scale = float(HEADLAMP_NUREC_EMISSIVE_SCALE)
        prim.CreateAttribute(
            "omni:nurec:ccmR", Sdf.ValueTypeNames.Float4
        ).Set(Gf.Vec4f(scale, 0.0, 0.0, 0.0))
        prim.CreateAttribute(
            "omni:nurec:ccmG", Sdf.ValueTypeNames.Float4
        ).Set(Gf.Vec4f(0.0, scale, 0.0, 0.0))
        prim.CreateAttribute(
            "omni:nurec:ccmB", Sdf.ValueTypeNames.Float4
        ).Set(Gf.Vec4f(0.0, 0.0, scale, 0.0))
        changed += 1
    if changed == 0:
        raise RuntimeError(
            f"No NuRec emissiveColor field found below {root.GetPath()}"
        )
    print(
        "[headlamp] Darkened NuRec emissive environment before camera "
        f"creation: scale={HEADLAMP_NUREC_EMISSIVE_SCALE:.3f}, "
        f"fields={changed}"
    )
    return root

# Robot/table geometry was captured from the running task on 2026-08-19, then
# the entire task group was translated +0.45 m along world Y toward the rear
# wall.  The saved G1 root is now (-2.501575708, 3.208633614, 0.912308514).
TRACKED_BASE_X = -1.2885240045627258
TRACKED_BASE_Y = 2.495828781902252
TRACKED_BASE_YAW = -0.5241687167665526
TRACKED_LIFT_HEIGHT = -0.07815947264447132

# Keep the already calibrated table and object layout fixed in the scene when
# the robot's reset pose changes.  These are the original layout-reference
# base coordinates, not the robot's new reset coordinates above.
TABLE_LAYOUT_BASE_X = -0.9818309171279677
TABLE_LAYOUT_BASE_Y = 2.318526199382671
TABLE_LAYOUT_BASE_YAW = -0.5241687167665526


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

# Place the table from the original calibrated robot reference, so saving a
# closer/higher robot reset pose does not move the task objects with it.
_layout_base_cosine = math.cos(TABLE_LAYOUT_BASE_YAW)
_layout_base_sine = math.sin(TABLE_LAYOUT_BASE_YAW)
TABLE_LAYOUT_G1_MOUNT_POSITION = (
    TABLE_LAYOUT_BASE_X
    + _layout_base_cosine * G1_MOUNT_LOCAL_POSITION[0]
    - _layout_base_sine * G1_MOUNT_LOCAL_POSITION[1],
    TABLE_LAYOUT_BASE_Y
    + _layout_base_sine * G1_MOUNT_LOCAL_POSITION[0]
    + _layout_base_cosine * G1_MOUNT_LOCAL_POSITION[1],
)
G1_FORWARD_YAW = TABLE_LAYOUT_BASE_YAW + math.pi
TABLE_YAW = G1_FORWARD_YAW - 0.5 * math.pi
TABLE_POSITION = (
    TABLE_LAYOUT_G1_MOUNT_POSITION[0] + math.cos(G1_FORWARD_YAW),
    TABLE_LAYOUT_G1_MOUNT_POSITION[1] + math.sin(G1_FORWARD_YAW),
    0.0,
)
TABLE_ORIENTATION = (
    math.cos(0.5 * TABLE_YAW),
    0.0,
    0.0,
    math.sin(0.5 * TABLE_YAW),
)

# The previous worktable's long edge was authored along local X, whereas the
# trolley's long edge is local Y. Rotate only the referenced table asset so its
# footprint continues to match the calibrated task layout. TABLE_YAW and
# TABLE_ORIENTATION deliberately remain unchanged because they define the
# semantic frame used by the task objects and reset randomization.
WORKTABLE_YAW = TABLE_YAW - 0.5 * math.pi
WORKTABLE_ORIENTATION = (
    math.cos(0.5 * WORKTABLE_YAW),
    0.0,
    0.0,
    math.sin(0.5 * WORKTABLE_YAW),
)
WORKTABLE_NATIVE_TOP_Z_CM = 85.37979125976562
WORKTABLE_XY_SCALE = 0.01
WORKTABLE_Z_SCALE = 0.84 / WORKTABLE_NATIVE_TOP_Z_CM


def _table_local_to_world(local_x: float, local_y: float, local_z: float):
    cosine = math.cos(TABLE_YAW)
    sine = math.sin(TABLE_YAW)
    return (
        TABLE_POSITION[0] + cosine * local_x - sine * local_y,
        TABLE_POSITION[1] + sine * local_x + cosine * local_y,
        TABLE_POSITION[2] + local_z,
    )


# The table top is at Z=0.84 m. The robot approaches from the table's local -Y
# side.  The loose 6 cm holder leaves the rod's bottom 5.2 cm in the socket,
# enough to initialize it upright without wedging it during a grasp.
LEIGUAN_POSITION = _table_local_to_world(0.24, -0.26, 1.020)
CHAGANSHENGZI_HOLDER_POSITION = _table_local_to_world(-0.05, -0.26, 0.840)
CHAGANSHENGZI_POSITION = _table_local_to_world(-0.05, -0.26, 0.848)
LEIGUAN_ORIENTATION = TABLE_ORIENTATION
CHAGANSHENGZI_HOLDER_ORIENTATION = TABLE_ORIENTATION
_chaganshengzi_yaw_orientation = (
    math.cos(0.5 * (TABLE_YAW + math.pi)),
    0.0,
    0.0,
    math.sin(0.5 * (TABLE_YAW + math.pi)),
)
CHAGANSHENGZI_ORIENTATION = _chaganshengzi_yaw_orientation

for asset_path in (
    ENVIRONMENT,
    WORKTABLE,
    LEIGUAN_OBJECT,
    CHAGANSHENGZI_OBJECT,
    CHAGANSHENGZI_HOLDER,
    TRACKED_G1_ASSEMBLY,
):
    if not asset_path.is_file():
        raise FileNotFoundError(asset_path)


def zero_reward(env) -> torch.Tensor:
    return torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)


def never_terminate(env) -> torch.Tensor:
    return torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)


# --- Demo: hide the robot's renderable geometry ---------------------------
# A pure-environment mining shot needs the head camera and the pre-authored
# Task01Headlamp but not the robot body.  Authoring visibility on a robot
# *ancestor* would propagate down the USD hierarchy and take the headlamp with
# it, so the hide is applied per renderable Gprim and never to a Camera or a
# light.  The default "on" leaves every collect / train / eval path untouched.
ROBOT_VISUALS_ENABLED = (
    os.environ.get("G1_ROBOT_VISUALS", "on").strip().lower() != "off"
)
DUMP_ROBOT_PRIMS = (
    os.environ.get("G1_DUMP_ROBOT_PRIMS", "0").strip().lower()
    in {"1", "true", "yes", "on"}
)


def _is_light_prim(prim) -> bool:
    return str(prim.GetTypeName()).endswith("Light") or prim.HasAPI(
        UsdLux.LightAPI
    )


def _assembly_root(g1_upper):
    """Return the tracked-assembly root that owns the whole robot.

    The tracked base, lift column and boom are siblings of the G1UpperBody
    articulation root, so hiding only G1UpperBody leaves them in frame.  The
    parent prim is the assembly root; the guard keeps a future re-parenting
    from silently turning this into "hide the entire scene".
    """
    parent = g1_upper.GetParent()
    if str(parent.GetPath()) in ("/", "/World"):
        return g1_upper
    return parent


def _hide_robot_visuals(g1_upper) -> None:
    """Hide only the robot's renderable geometry.

    Visibility is authored on leaf Gprim nodes and never on an ancestor Xform,
    so the xform chain stays intact and the head camera plus the Task01Headlamp
    that share it keep working.  Cameras and lights are skipped explicitly as a
    second guard -- which also lets the traversal start at the assembly root,
    a level above the camera and the lamp.
    """
    root = _assembly_root(g1_upper)
    hidden = []
    kept = []
    for prim in Usd.PrimRange(root):
        if prim.IsA(UsdGeom.Camera) or _is_light_prim(prim):
            kept.append(str(prim.GetPath()))
            continue
        if prim.IsA(UsdGeom.Gprim):
            UsdGeom.Imageable(prim).MakeInvisible()
            hidden.append(str(prim.GetPath()))
    print(
        f"[robot-visuals] root={root.GetPath()} "
        f"hid {len(hidden)} geometry prims, "
        f"kept {len(kept)} camera/light prims."
    )
    for path in hidden[:6]:
        print(f"[robot-visuals]   hidden: {path}")
    if len(hidden) > 6:
        print(f"[robot-visuals]   ... and {len(hidden) - 6} more")
    for path in kept:
        print(f"[robot-visuals]   kept:   {path}")


def _dump_robot_prims(g1_upper) -> None:
    """Print the robot subtree so the hide target set can be checked."""
    root = _assembly_root(g1_upper)
    root_str = str(root.GetPath())
    print(f"[robot-prims] ---- subtree of {root_str} ----")
    for prim in Usd.PrimRange(root):
        path_str = str(prim.GetPath())
        depth = path_str[len(root_str):].count("/")
        if prim.IsA(UsdGeom.Camera):
            flag = " [CAMERA]"
        elif _is_light_prim(prim):
            flag = " [LIGHT]"
        elif prim.IsA(UsdGeom.Gprim):
            flag = " [GPRIM]"
        else:
            flag = ""
        vis = ""
        if prim.IsA(UsdGeom.Imageable):
            attr = UsdGeom.Imageable(prim).GetVisibilityAttr()
            if attr and attr.HasAuthoredValueOpinion():
                vis = f" visibility={attr.Get()}"
        print(
            f"[robot-prims] {'  ' * depth}{prim.GetName()} "
            f"<{prim.GetTypeName()}>{flag}{vis}"
        )
    print("[robot-prims] ---- end ----")



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

    # Dex1 is position driven, so a finger can otherwise move across the thin
    # Leiguan wall between two discrete contact passes.  Speculative CCD is
    # supported on articulation links and creates contacts before that can
    # happen.  Keep the contact envelope small enough that it does not alter
    # normal grasping or block the Leiguan cavity.
    ccd_body_count = 0
    contact_shape_count = 0
    for prim in Usd.PrimRange(g1_upper):
        path = str(prim.GetPath()).lower()
        if "hand_" not in path:
            continue
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            rigid_api = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
            rigid_api.CreateEnableSpeculativeCCDAttr().Set(True)
            rigid_api.CreateSolverPositionIterationCountAttr().Set(16)
            rigid_api.CreateSolverVelocityIterationCountAttr().Set(4)
            ccd_body_count += 1
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            collision_api = PhysxSchema.PhysxCollisionAPI.Apply(prim)
            collision_api.CreateContactOffsetAttr().Set(0.003)
            collision_api.CreateRestOffsetAttr().Set(0.0)
            contact_shape_count += 1
    print(
        "[tracked-g1] Loaded combined USD: "
        f"{TRACKED_G1_ASSEMBLY}; authored mount="
        f"{G1_MOUNT_LOCAL_POSITION}"
    )
    print(
        "[tracked-g1] Dex1 anti-tunneling enabled on "
        f"{ccd_body_count} rigid bodies / {contact_shape_count} colliders."
    )
    stage = root.GetStage()
    if not ROBOT_VISUALS_ENABLED:
        _hide_robot_visuals(g1_upper)
    # Author the headlamp before camera render products are created.  The
    # off-screen sensor in this task does not reliably notice lights created
    # after renderer initialization, so this is intentionally part of the
    # scene-spawn path.  It shares the d435 link and the exact converted
    # OpenGL camera orientation, making its -Z light axis camera-aligned.
    if HEADLAMP_ENABLED:
        light_path = f"{g1_upper.GetPath()}/d435_link/Task01Headlamp"
        if stage.GetPrimAtPath(light_path).IsValid():
            stage.RemovePrim(light_path)
        light = UsdLux.SphereLight.Define(stage, light_path)
        radius = _headlamp_env_float("G1_HEADLAMP_RADIUS_M", 0.025)
        cone_angle = _headlamp_env_float(
            "G1_HEADLAMP_CONE_ANGLE_DEG", 44.0
        )
        cone_softness = _headlamp_env_float(
            "G1_HEADLAMP_CONE_SOFTNESS", 0.50
        )
        focus = _headlamp_env_float("G1_HEADLAMP_FOCUS", 2.0)
        color_temperature = _headlamp_env_float(
            "G1_HEADLAMP_COLOR_TEMP_K", 4100.0
        )
        light.CreateIntensityAttr(HEADLAMP_INTENSITY).Set(
            HEADLAMP_INTENSITY
        )
        light.CreateExposureAttr(0.0).Set(0.0)
        light.CreateRadiusAttr(radius).Set(radius)
        light.CreateNormalizeAttr(False).Set(False)
        light.CreateColorAttr(Gf.Vec3f(1.0, 1.0, 1.0)).Set(
            Gf.Vec3f(1.0, 1.0, 1.0)
        )
        light.CreateEnableColorTemperatureAttr(True).Set(True)
        light.CreateColorTemperatureAttr(color_temperature).Set(
            color_temperature
        )
        shaping = UsdLux.ShapingAPI.Apply(light.GetPrim())
        shaping.CreateShapingConeAngleAttr(cone_angle).Set(cone_angle)
        shaping.CreateShapingConeSoftnessAttr(cone_softness).Set(
            cone_softness
        )
        shaping.CreateShapingFocusAttr(focus).Set(focus)
        shaping.CreateShapingFocusTintAttr(
            Gf.Vec3f(0.0, 0.0, 0.0)
        ).Set(Gf.Vec3f(0.0, 0.0, 0.0))
        UsdLux.ShadowAPI.Apply(light.GetPrim()).CreateShadowEnableAttr(
            True
        ).Set(True)
        xform = UsdGeom.Xformable(light.GetPrim())
        xform.ClearXformOpOrder()
        xform.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(
            Gf.Vec3d(0.0, 0.035, -0.015)
        )
        camera_orientation = Gf.Quatd(
            0.5, Gf.Vec3d(-0.5, -0.5, 0.5)
        )
        yaw_offset = Gf.Rotation(
            Gf.Vec3d(0.0, 1.0, 0.0), HEADLAMP_AIM_YAW_DEG
        ).GetQuat()
        pitch_offset = Gf.Rotation(
            Gf.Vec3d(1.0, 0.0, 0.0), HEADLAMP_AIM_PITCH_DEG
        ).GetQuat()
        xform.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(
            camera_orientation * yaw_offset * pitch_offset
        )
        print(
            "[headlamp] Pre-authored before renderer initialization: "
            f"path={light_path}, intensity={HEADLAMP_INTENSITY:.1f}, "
            f"ambient={HEADLAMP_AMBIENT_INTENSITY:.1f}, "
            f"cone={cone_angle:.1f}deg, softness={cone_softness:.2f}, "
            f"focus={focus:.2f}, temperature={color_temperature:.0f}K, "
            f"aim_yaw={HEADLAMP_AIM_YAW_DEG:.2f}deg, "
            f"aim_pitch={HEADLAMP_AIM_PITCH_DEG:.2f}deg"
        )
    if AUX_LIGHTS_ENABLED:
        preauthor_auxiliary_spotlights(stage, TABLE_POSITION, TABLE_YAW)
    if DUMP_ROBOT_PRIMS:
        _dump_robot_prims(g1_upper)
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


def spawn_static_worktable(
    prim_path: str,
    cfg: UsdFileCfg,
    translation=None,
    orientation=None,
    **kwargs,
) -> Usd.Prim:
    """Spawn the visual worktable and author triangle-mesh static collision."""

    root = spawn_from_usd(
        prim_path,
        cfg,
        translation=translation,
        orientation=orientation,
        **kwargs,
    )
    collision_mesh_count = 0
    for prim in Usd.PrimRange(root):
        if not prim.IsA(UsdGeom.Mesh):
            continue
        UsdPhysics.CollisionAPI.Apply(prim)
        mesh_collision = UsdPhysics.MeshCollisionAPI.Apply(prim)
        mesh_collision.CreateApproximationAttr().Set("none")
        collision_mesh_count += 1
    if collision_mesh_count == 0:
        raise RuntimeError(f"No worktable meshes found below {root.GetPath()}")
    print(
        f"[tabletop-grasp] Static worktable collision applied to "
        f"{collision_mesh_count} mesh(es)."
    )
    return root


def spawn_stable_leiguan(
    prim_path: str,
    cfg: UsdFileCfg,
    translation=None,
    orientation=None,
    **kwargs,
) -> Usd.Prim:
    """Spawn Leiguan with per-body CCD while retaining its hollow collider."""

    root = spawn_from_usd(
        prim_path,
        cfg,
        translation=translation,
        orientation=orientation,
        **kwargs,
    )
    rigid_body_count = 0
    for prim in Usd.PrimRange(root):
        if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
            continue
        rigid_api = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
        # Standard sweep CCD is ignored by GPU PhysX.  Speculative CCD works
        # on the CUDA dynamics path used by this task and also covers motion
        # caused by a closing articulation link.
        rigid_api.CreateEnableSpeculativeCCDAttr().Set(True)
        rigid_api.CreateSolverPositionIterationCountAttr().Set(16)
        rigid_api.CreateSolverVelocityIterationCountAttr().Set(4)
        rigid_api.CreateMaxDepenetrationVelocityAttr().Set(1.0)
        rigid_body_count += 1
    if rigid_body_count == 0:
        raise RuntimeError(f"Leiguan has no rigid body below {root.GetPath()}")
    print(
        "[tabletop-grasp] Leiguan GPU speculative CCD enabled on "
        f"{rigid_body_count} rigid body."
    )
    return root


@configclass
class TabletopGraspSceneCfg(InteractiveSceneCfg):
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
        spawn=UsdFileCfg(
            usd_path=str(ENVIRONMENT),
            func=spawn_headlamp_environment,
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
            # Match the previous 0.84 m tabletop exactly while retaining the
            # trolley's native horizontal dimensions in meters.
            scale=(WORKTABLE_XY_SCALE, WORKTABLE_XY_SCALE, WORKTABLE_Z_SCALE),
        ),
    )

    leiguan = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Leiguan",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=LEIGUAN_POSITION,
            rot=LEIGUAN_ORIENTATION,
        ),
        spawn=UsdFileCfg(
            usd_path=str(LEIGUAN_OBJECT),
            func=spawn_stable_leiguan,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                max_depenetration_velocity=1.0,
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=4,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                contact_offset=0.003,
                rest_offset=0.0,
            ),
        ),
    )

    chaganshengzi_holder = RigidObjectCfg(
        prim_path="/World/envs/env_.*/ChaganShengziHolder",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=CHAGANSHENGZI_HOLDER_POSITION,
            rot=CHAGANSHENGZI_HOLDER_ORIENTATION,
        ),
        spawn=UsdFileCfg(
            usd_path=str(CHAGANSHENGZI_HOLDER),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                kinematic_enabled=True,
                disable_gravity=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
    )

    # The asset contains 27 rigid bodies and 26 joints, so it is deliberately
    # spawned as an unmanaged AssetBase.  The runtime controller captures and
    # restores every body for deterministic object resets.
    chaganshengzi = AssetBaseCfg(
        prim_path="/World/envs/env_.*/ChaganShengzi",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=CHAGANSHENGZI_POSITION,
            rot=CHAGANSHENGZI_ORIENTATION,
        ),
        spawn=UsdFileCfg(usd_path=str(CHAGANSHENGZI_OBJECT)),
    )

    # Wrap the articulation already present inside the combined USD. No second
    # G1 asset is spawned at runtime.
    robot: ArticulationCfg = make_embedded_g1_upper_body_cfg()

    front_camera = CameraPresets.g1_front_camera().replace(
        prim_path="/World/TrackedBase/G1UpperBody/d435_link/front_cam"
    )
    if HEADLAMP_ENABLED:
        front_camera.spawn.func = spawn_headlamp_fixed_exposure_camera
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
        pos_offset=THIRD_PERSON_POSITION,
        rot_offset=THIRD_PERSON_ROTATION,
        focal_length=THIRD_PERSON_FOCAL_LENGTH,
        convention=THIRD_PERSON_CONVENTION,
    )

    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(
            color=(0.75, 0.75, 0.75),
            intensity=(
                HEADLAMP_AMBIENT_INTENSITY
                if HEADLAMP_ENABLED
                else 1200.0
            ),
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

    reset_leiguan = EventTermCfg(
        func=base_mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("leiguan"),
        },
    )


def _runtime_controller(env):
    controller = getattr(env, "_tabletop_grasp_runtime_controller", None)
    if controller is None:
        raise RuntimeError("Tabletop grasp runtime controller is not initialized")
    return controller


def reset_task_objects(env) -> None:
    # The runtime controller resets Leiguan and the complete rod/rope/holder
    # assembly with independent +/-5 cm XY offsets.
    _runtime_controller(env).reset_task_objects()


def reset_everything(env) -> None:
    _runtime_controller(env).reset_everything()


@configclass
class TabletopGraspG129Dex1EnvCfg(ManagerBasedRLEnvCfg):
    scene: TabletopGraspSceneCfg = TabletopGraspSceneCfg(
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
        from .tabletop_grasp_controller import TabletopGraspController

        controller = TabletopGraspController(env)
        env._tabletop_grasp_runtime_controller = controller
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
        # GPU PhysX ignores standard scene CCD.  The spawn helpers use GPU-
        # compatible speculative CCD; substeps and extra iterations further
        # reduce penetration under the high-stiffness Dex1 position drives.
        self.sim.physx.gpu_constraint_solver_heavy_spring_enabled = True
        self.sim.physx.num_substeps = 2
        self.sim.physx.num_position_iterations = 12
        self.sim.physx.num_velocity_iterations = 4

        self.event_manager = SimpleEventManager()
        self.event_manager.register(
            "reset_object_self",
            SimpleEvent(func=reset_task_objects),
        )
        self.event_manager.register(
            "reset_all_self",
            SimpleEvent(func=reset_everything),
        )
