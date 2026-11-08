#!/usr/bin/env python3
"""Quest 3/3S teleoperation for the tracked upper-body robots.

A config-driven Quest teleoperation controller for the tracked Galbot One Golf
and Astribot S1 upper-body assemblies, following the W1/X7S teleop pattern in
this repository (Vuer WebXR QuestBridge + Pinocchio/Pink constrained IK).

Controls (Quest 3/3S):
  Hold left Grip    drive the left arm from the left controller
  Hold right Grip   drive the right arm from the right controller
  Left/Right Trigger  close the corresponding gripper (Galbot) or curl the
                      corresponding Wuji hand (Astribot)
  Left stick Y      head pitch (optional)
  Right B           reset arms to the home pose
  ESC               quit

Keyboard fallback (when no Quest):
  W/S/A/D           move left arm FK joint 2 / joint 1
  I/K/J/L           move right arm FK joint 2 / joint 1
  T/Y               close/open left gripper, U/O right gripper
  R                 reset to home
  P                 print joint state
  ESC               quit
"""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

import numpy as np
from isaacsim import SimulationApp


CONFIG_PATH = Path(os.environ.get("TRACKED_TELEOP_CONFIG", "")).resolve()
if not CONFIG_PATH.is_file():
    raise SystemExit(f"TRACKED_TELEOP_CONFIG is missing: {CONFIG_PATH}")
CONFIG = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))

SCRIPT_DIR = Path(__file__).resolve().parent
QUEST_DEPS = SCRIPT_DIR / ".quest_deps"
QUEST_SETUP = SCRIPT_DIR / "setup_w1_quest3s.sh"

ROBOT_USD = Path(CONFIG["usd_path"]).resolve()
if not ROBOT_USD.is_file():
    raise SystemExit(f"Robot USD is missing: {ROBOT_USD}")

ARTICULATION_PATH = CONFIG["articulation_path"]
ARTICULATION_ROOT_REF = CONFIG.get("articulation_root_ref", "/Assembly")
JOINT_PREFIX = CONFIG["joint_prefix"]
LEFT_ARM_NAMES = tuple(CONFIG["left_arm_joints"])
RIGHT_ARM_NAMES = tuple(CONFIG["right_arm_joints"])
HEAD_NAMES = tuple(CONFIG["head_joints"])
LEFT_GRIPPER_NAMES = tuple(CONFIG["left_gripper_joints"])
RIGHT_GRIPPER_NAMES = tuple(CONFIG["right_gripper_joints"])
LEFT_HAND_NAMES = tuple(CONFIG["left_hand_joints"])
RIGHT_HAND_NAMES = tuple(CONFIG["right_hand_joints"])
EE_PATHS = CONFIG["end_effector_paths"]
HEAD_CAMERA_PATH = CONFIG.get("head_camera_path", "/Assembly")
GRIPPER_TYPE = CONFIG.get("gripper_type", "two_finger")

HOME_ARM_DEG = {
    "left": np.asarray(CONFIG["home_arm_degrees"]["left"], dtype=np.float64),
    "right": np.asarray(CONFIG["home_arm_degrees"]["right"], dtype=np.float64),
}
GRIPPER_OPEN = math.radians(float(CONFIG.get("gripper_open_degrees", 0.0)))
GRIPPER_CLOSED = math.radians(float(CONFIG.get("gripper_closed_degrees", 92.0)))
GRIPPER_RATE = float(CONFIG.get("gripper_rate", 1.25))
ARM_STIFFNESS = float(CONFIG.get("arm_stiffness", 800.0))
ARM_DAMPING = float(CONFIG.get("arm_damping", 60.0))

HEADLESS = os.environ.get("TRACKED_HEADLESS", "0") == "1"
SMOKE_TEST = os.environ.get("TRACKED_SMOKE_TEST", "0") == "1"
SMOKE_TEST_FRAMES = int(os.environ.get("TRACKED_SMOKE_FRAMES", "90"))
MAX_FRAMES = int(os.environ.get("TRACKED_MAX_FRAMES", "0"))
QUEST_ENABLED = os.environ.get("TRACKED_QUEST_ENABLED", "1") != "0"
QUEST_PORT = int(os.environ.get("TRACKED_QUEST_PORT", "8012"))
QUEST_STREAM_FPS = int(os.environ.get("TRACKED_QUEST_STREAM_FPS", "20"))
QUEST_POSITION_SCALE = float(os.environ.get("TRACKED_QUEST_POSITION_SCALE", "0.7"))
SHOW_CAMERA_VIEWPORTS = os.environ.get("TRACKED_SHOW_CAMERA_VIEWPORTS", "1") != "0"
PHYSICS_SUBSTEPS = 1

# LW-BenchHub core/devices/vr/consts.py
GRD_YUP_TO_GRD_ZUP = np.asarray(
    [
        [0.0, 0.0, -1.0, 0.0],
        [-1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)
GRD_ZUP_TO_GRD_YUP = np.linalg.inv(GRD_YUP_TO_GRD_ZUP)

simulation_app = SimulationApp(
    {
        "headless": HEADLESS,
        "width": 1280,
        "height": 720,
        "renderer": "MinimalRendering" if HEADLESS else "RaytracedLighting",
        "minimal_shading_mode": 4 if HEADLESS else 0,
        "disable_viewport_updates": HEADLESS,
        "anti_aliasing": 0 if HEADLESS else 2,
        "multi_gpu": False,
    }
)

import carb.input
import omni.appwindow
import omni.usd
from isaacsim.core.api import World
from isaacsim.core.prims import Articulation, RigidPrim
from isaacsim.core.utils.stage import create_new_stage
from isaacsim.core.utils.viewports import set_camera_view
from isaacsim.sensors.camera import Camera
from omni.kit.viewport.utility import get_active_viewport
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics

try:
    from pxr import PhysxSchema
except ImportError:
    PhysxSchema = None

sys.path.insert(0, str(SCRIPT_DIR))
from w1_quest3s import QuestBridge  # noqa: E402
from tracked_upperbody_ik import TrackedBimanualIK  # noqa: E402


def _wait_for_stage(max_updates: int = 1200) -> None:
    from isaacsim.core.utils.stage import is_stage_loading

    for _ in range(max_updates):
        if not is_stage_loading():
            return
        simulation_app.update()
    raise RuntimeError("Stage did not finish loading")


def _add_reference(stage: Usd.Stage, asset: Path, prim_path: str) -> None:
    prim = stage.DefinePrim(prim_path, "Xform")
    prim.GetReferences().AddReference(str(asset))
    if not prim.IsValid():
        raise RuntimeError(f"Failed to reference {asset}")


def _add_ground_and_lights(stage: Usd.Stage) -> None:
    ground = UsdGeom.Cube.Define(stage, "/World/FallbackGround")
    ground.CreateSizeAttr(1.0)
    ground.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, -0.05))
    ground.AddScaleOp().Set(Gf.Vec3f(100.0, 100.0, 0.10))
    ground.CreateVisibilityAttr().Set(UsdGeom.Tokens.invisible)
    UsdPhysics.CollisionAPI.Apply(ground.GetPrim()).CreateCollisionEnabledAttr(True)

    dome = UsdLux.DomeLight.Define(stage, "/World/Lights/Dome")
    dome.CreateIntensityAttr(300.0)
    dome.CreateColorAttr(Gf.Vec3f(0.78, 0.82, 0.92))
    sun = UsdLux.DistantLight.Define(stage, "/World/Lights/Sun")
    sun.CreateIntensityAttr(900.0)
    sun.CreateAngleAttr(1.0)
    sun.AddRotateXYZOp().Set(Gf.Vec3f(48.0, -35.0, -42.0))


def _configure_drives(stage: Usd.Stage) -> int:
    """Prepare kinematic-style joints: no drive forces, high damping.

    The tracked assemblies are driven exactly like the G1 hand-cart demo's
    "kinematic presentation logic": joints are moved by writing positions
    directly every frame (RobotJointCommander.apply).  Drive forces are zeroed
    so PhysX never pushes the arms/grippers around, and the base is anchored
    so nothing drifts before teleop begins.
    """
    root = stage.GetPrimAtPath(ARTICULATION_ROOT_REF)
    if not root:
        raise RuntimeError(f"Robot articulation root missing: {ARTICULATION_ROOT_REF}")
    # Disable gravity exactly like the W1 task does; otherwise the whole
    # assembly falls through the ground plane.
    if PhysxSchema is not None:
        for prim in Usd.PrimRange(root):
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                PhysxSchema.PhysxRigidBodyAPI.Apply(prim).CreateDisableGravityAttr(True)

    # The W1 chassis has no joint between the static Vehicle and the LiftCarriage
    # elevator, so the upper-body articulation's drive reaction forces shove the
    # whole assembly around before teleop even begins.  Anchor the LiftCarriage
    # to the Vehicle with a fixed joint so the base cannot drift.
    mount = stage.GetPrimAtPath(f"{ARTICULATION_ROOT_REF}/VehicleCarriageMount")
    if not mount or not mount.IsA(UsdPhysics.Joint):
        anchor = UsdPhysics.FixedJoint.Define(
            stage, f"{ARTICULATION_ROOT_REF}/VehicleCarriageMount"
        )
        anchor.CreateBody0Rel().SetTargets(
            [Sdf.Path(f"{ARTICULATION_ROOT_REF}/Vehicle")]
        )
        anchor.CreateBody1Rel().SetTargets(
            [Sdf.Path(f"{ARTICULATION_ROOT_REF}/LiftCarriage")]
        )
        anchor.CreateCollisionEnabledAttr(False)

    configured = 0
    for prim in Usd.PrimRange(root):
        if not prim.IsA(UsdPhysics.Joint):
            continue
        drive_type = "linear" if prim.IsA(UsdPhysics.PrismaticJoint) else "angular"
        drive = UsdPhysics.DriveAPI(prim, drive_type)
        # Zero stiffness/max-force: joints only move through direct position
        # writes.  High damping absorbs the PhysX solver's one-step
        # reconciliation so the assembly stays frozen at its authored pose.
        drive.CreateStiffnessAttr(0.0)
        drive.CreateDampingAttr(5000.0)
        drive.CreateMaxForceAttr(0.0)
        drive.CreateTargetVelocityAttr(0.0)
        configured += 1
    return configured


def _add_head_camera(stage: Usd.Stage, path: str) -> str:
    camera = UsdGeom.Camera.Define(stage, f"{path}/QuestHeadCamera")
    camera.CreateFocalLengthAttr(19.3)
    camera.CreateHorizontalApertureAttr(36.0)
    camera.CreateVerticalApertureAttr(20.25)
    camera.CreateClippingRangeAttr(Gf.Vec2f(0.03, 100.0))
    camera.CreateFocusDistanceAttr(2.0)
    camera.CreateFStopAttr(0.0)
    return str(camera.GetPrim().GetPath())


def build_world() -> tuple[World, Usd.Stage, Articulation]:
    World.clear_instance()
    create_new_stage()
    simulation_app.update()
    world = World(
        stage_units_in_meters=1.0,
        physics_dt=1.0 / (60.0 * PHYSICS_SUBSTEPS),
        rendering_dt=1.0 / 60.0,
    )
    stage = omni.usd.get_context().get_stage()
    _add_reference(stage, ROBOT_USD, ARTICULATION_ROOT_REF)
    _wait_for_stage()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    # The assembled mount is a fixed joint, which PhysX does not register as an
    # articulation root.  Promote the upper-body root rigid body so Isaac Sim's
    # Articulation view can form the tensor articulation.
    root_prim = stage.GetPrimAtPath(ARTICULATION_PATH)
    if not root_prim:
        raise RuntimeError(f"Articulation root missing: {ARTICULATION_PATH}")
    if not root_prim.HasAPI(UsdPhysics.RigidBodyAPI):
        raise RuntimeError(f"Articulation root is not a rigid body: {ARTICULATION_PATH}")
    UsdPhysics.ArticulationRootAPI.Apply(root_prim)
    _add_ground_and_lights(stage)
    drive_count = _configure_drives(stage)
    camera_path = _add_head_camera(stage, HEAD_CAMERA_PATH)

    robot = world.scene.add(
        Articulation(
            prim_paths_expr=ARTICULATION_PATH,
            name=f"{CONFIG['robot']}_task_robot",
            reset_xform_properties=False,
        )
    )
    world.reset()
    print(
        f"[{CONFIG['robot']}] articulation DOFs={robot.num_dof} "
        f"drives_configured={drive_count}",
        flush=True,
    )
    return world, stage, robot


class RobotJointCommander:
    """Joint targets in articulation DOF order."""

    def __init__(self, robot: Articulation) -> None:
        self.robot = robot
        self.names = list(robot.dof_names)
        self.index = {name: index for index, name in enumerate(self.names)}
        expected = (
            set(LEFT_ARM_NAMES)
            | set(RIGHT_ARM_NAMES)
            | set(HEAD_NAMES)
            | set(LEFT_GRIPPER_NAMES)
            | set(RIGHT_GRIPPER_NAMES)
            | set(LEFT_HAND_NAMES)
            | set(RIGHT_HAND_NAMES)
        )
        missing = sorted(expected - set(self.names))
        if missing:
            raise RuntimeError(
                f"Missing articulation joints ({CONFIG['robot']}): {missing}"
            )
        self.limits = np.asarray(robot.get_dof_limits()[0], dtype=np.float64)
        self.max_velocities = np.full(robot.num_dof, 2.61, dtype=np.float32)
        for name, index in self.index.items():
            if name in set(LEFT_ARM_NAMES) | set(RIGHT_ARM_NAMES):
                self.max_velocities[index] = 2.175
            elif name in set(LEFT_HAND_NAMES) | set(RIGHT_HAND_NAMES):
                self.max_velocities[index] = 1.0
            elif "gripper" in name:
                self.max_velocities[index] = 0.03
        self.robot.set_max_joint_velocities(self.max_velocities[np.newaxis, :])

        # Do NOT teleport the robot to a scripted home pose on startup.  The
        # tracked assemblies are authored in their natural baked pose; forcing
        # all joints to zero/home (especially the un-commanded gripper finger
        # DOFs) makes the robot jump violently before teleop even begins.
        # Instead, hold the robot exactly at its current pose and remember it
        # as the reset home, exactly like the Unitree G1 controller does.
        current = np.asarray(robot.get_joint_positions()[0], dtype=np.float64)
        self.target = np.clip(current, self.limits[:, 0], self.limits[:, 1])
        self.home = self.target.copy()
        self.zero_velocity = np.zeros((1, robot.num_dof), dtype=np.float32)

        self.left_arm_indices = np.asarray(
            [self.index[name] for name in LEFT_ARM_NAMES], dtype=np.int64
        )
        self.right_arm_indices = np.asarray(
            [self.index[name] for name in RIGHT_ARM_NAMES], dtype=np.int64
        )
        self.head_indices = np.asarray(
            [self.index[name] for name in HEAD_NAMES], dtype=np.int64
        )
        self.left_gripper_indices = np.asarray(
            [self.index[name] for name in LEFT_GRIPPER_NAMES], dtype=np.int64
        )
        self.right_gripper_indices = np.asarray(
            [self.index[name] for name in RIGHT_GRIPPER_NAMES], dtype=np.int64
        )
        self.left_hand_indices = np.asarray(
            [self.index[name] for name in LEFT_HAND_NAMES], dtype=np.int64
        )
        self.right_hand_indices = np.asarray(
            [self.index[name] for name in RIGHT_HAND_NAMES], dtype=np.int64
        )
        # All gripper DOFs (main + closed-loop finger/knuckle links) so closing
        # moves the whole linkage, and opening stays exactly at the current
        # authored pose instead of snapping to zero when idle.
        self.gripper_dof_indices = {
            side: np.asarray(
                [
                    self.index[name]
                    for name in self.names
                    if name.startswith(side) and "gripper" in name
                ],
                dtype=np.int64,
            )
            for side in ("left", "right")
        }
        self.gripper_open_state = {
            side: self.target[inds].copy() for side, inds in self.gripper_dof_indices.items()
        }
        self.gripper_closed_state = {
            side: self.limits[inds, 0].copy() for side, inds in self.gripper_dof_indices.items()
        }
        self.reset()

    def clamp(self, name: str, value: float) -> float:
        index = self.index[name]
        low, high = self.limits[index]
        return float(np.clip(value, low, high))

    def reset(self) -> None:
        self.target[:] = self.home
        self.apply()

    def apply(self) -> None:
        if not np.all(np.isfinite(self.target)):
            raise RuntimeError("Refusing to apply non-finite joint targets")
        self.target = np.clip(self.target, self.limits[:, 0], self.limits[:, 1])
        # Kinematic-style control (same as the G1 hand-cart demo): write the
        # joint positions directly every frame.  Drive forces are disabled, so
        # PhysX never pushes the robot around; it holds perfectly still until a
        # target changes, and arms track the IK output exactly.
        self.robot.set_joint_positions(self.target[np.newaxis, :])
        self.robot.set_joint_velocities(self.zero_velocity)

    def set_arm_targets(self, left: np.ndarray, right: np.ndarray) -> None:
        if left.shape == (7,) and np.all(np.isfinite(left)):
            self.target[self.left_arm_indices] = left
        if right.shape == (7,) and np.all(np.isfinite(right)):
            self.target[self.right_arm_indices] = right

    def set_head(self, yaw_deg: float, pitch_deg: float) -> None:
        for name, value in zip(HEAD_NAMES, (yaw_deg, pitch_deg)):
            self.target[self.index[name]] = self.clamp(
                name, math.radians(value)
            )

    def set_grippers(self, left: float, right: float) -> None:
        for side, value in (("left", float(left)), ("right", float(right))):
            amount = float(np.clip(value / GRIPPER_CLOSED, 0.0, 1.0))
            inds = self.gripper_dof_indices[side]
            self.target[inds] = (
                self.gripper_open_state[side]
                + amount
                * (self.gripper_closed_state[side] - self.gripper_open_state[side])
            )

    def set_hand_curl(self, left: float, right: float) -> None:
        """Drive a coordinated open/close curl over the Wuji hand flex joints."""
        left = float(np.clip(left, 0.0, 1.0))
        right = float(np.clip(right, 0.0, 1.0))
        for index in self.left_hand_indices:
            name = self.names[index]
            if "abd" in name:
                target = 0.0
            else:
                # Open = 0 (neutral), closed = flex toward the positive limit.
                low, high = self.limits[index]
                target = 0.0 if low <= 0.0 <= high else low
                if high > 0.0:
                    target = high * left
                else:
                    target = low * left
            self.target[index] = self.clamp(name, target)
        for index in self.right_hand_indices:
            name = self.names[index]
            if "abd" in name:
                target = 0.0
            else:
                low, high = self.limits[index]
                if high > 0.0:
                    target = high * right
                else:
                    target = low * right
            self.target[index] = self.clamp(name, target)


def _valid_pose(matrix: np.ndarray) -> bool:
    return bool(
        matrix.shape == (4, 4)
        and np.all(np.isfinite(matrix))
        and abs(float(np.linalg.det(matrix[:3, :3]))) > 0.1
    )


def _matrix_to_pose_wxyz(matrix: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation

    quaternion_xyzw = Rotation.from_matrix(matrix[:3, :3]).as_quat()
    return np.asarray(
        [
            *matrix[:3, 3],
            quaternion_xyzw[3],
            quaternion_xyzw[0],
            quaternion_xyzw[1],
            quaternion_xyzw[2],
        ],
        dtype=np.float64,
    )


def _quat_wxyz_to_rotation_matrix(quaternion: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation

    w, x, y, z = (float(value) for value in quaternion)
    return Rotation.from_quat([x, y, z, w]).as_matrix()


def _clamp_norm(vector: np.ndarray, maximum: float) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= maximum:
        return vector
    return vector * (maximum / norm)


def _rotation_vector(rotation: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation

    return np.asarray(Rotation.from_matrix(rotation).as_rotvec(), dtype=np.float64)


def _smooth_pose(target: np.ndarray, current: np.ndarray | None) -> np.ndarray:
    if current is None or not np.all(np.isfinite(current)):
        return target.copy()
    from scipy.spatial.transform import Rotation

    rate = float(os.environ.get("TRACKED_SMOOTH_RATE", "0.5"))
    position = rate * target[:3, 3] + (1.0 - rate) * current[:3, 3]
    rotation = Rotation.from_matrix(target[:3, :3]).slerp(
        Rotation.from_matrix(current[:3, :3]), 1.0 - rate
    )
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = rotation.as_matrix()
    result[:3, 3] = position
    return result


class TrackedQuestController:
    def __init__(self, joints: RobotJointCommander, stage: Usd.Stage) -> None:
        self.joints = joints
        self.stage = stage
        self.bridge: QuestBridge | None = None
        self.camera: Camera | None = None
        self.stream_frame = 0
        self.stream_interval = max(1, round(60 / QUEST_STREAM_FPS))
        self.ik = TrackedBimanualIK(
            stage,
            [f"{JOINT_PREFIX}{name}" for name in LEFT_ARM_NAMES],
            [f"{JOINT_PREFIX}{name}" for name in RIGHT_ARM_NAMES],
            EE_PATHS,
            model_scale=1.0,
        )
        # World EE poses are read from the articulation rigid bodies, not the
        # Pinocchio model (which is rooted at the first joint frame).  This
        # matches the W1 controller and avoids model-frame/world-frame drift.
        self.eef_views = {
            side: RigidPrim(
                prim_paths_expr=EE_PATHS[side],
                name=f"tracked_{side}_eef",
                reset_xform_properties=False,
                prepare_contact_sensors=False,
            )
            for side in ("left", "right")
        }
        self.reset_requested = False
        self.active = False
        self.grip_was_pressed = {side: False for side in ("left", "right")}
        self.awaiting_grip_release = {side: True for side in ("left", "right")}
        self.arm_control_active = {side: False for side in ("left", "right")}
        self.last_grip_values = {side: 0.0 for side in ("left", "right")}
        self.controller_anchors = {side: None for side in ("left", "right")}
        self.controller_rotation_anchors = {side: None for side in ("left", "right")}
        self.robot_anchors = {side: None for side in ("left", "right")}
        self.robot_rotation_anchors = {side: None for side in ("left", "right")}
        self.target_positions = {side: None for side in ("left", "right")}
        self.target_orientations = {side: None for side in ("left", "right")}
        self.latest_pose = {side: None for side in ("left", "right")}
        self.robot = None

    @staticmethod
    def _convert_controller_pose(pose: np.ndarray) -> np.ndarray:
        return GRD_YUP_TO_GRD_ZUP @ pose @ GRD_ZUP_TO_GRD_YUP

    def _ee_world_pose(self, side: str) -> np.ndarray:
        view = self.eef_views[side]
        positions, orientations = view.get_world_poses()
        matrix = np.eye(4, dtype=np.float64)
        matrix[:3, 3] = np.asarray(positions[0], dtype=np.float64)
        matrix[:3, :3] = _quat_wxyz_to_rotation_matrix(
            np.asarray(orientations[0], dtype=np.float64)
        )
        return matrix

    def start(self) -> None:
        for side, view in self.eef_views.items():
            view.initialize()
            if not view.is_physics_handle_valid():
                raise RuntimeError(
                    f"Quest {side} end-effector view did not initialize: {EE_PATHS[side]}"
                )
        if HEADLESS or SMOKE_TEST or not QUEST_ENABLED:
            return
        if not (QUEST_DEPS / "vuer").is_dir():
            if not QUEST_SETUP.is_file():
                raise RuntimeError("Quest setup script missing")
            import subprocess

            subprocess.run([str(QUEST_SETUP)], check=True)
        self.camera = Camera(
            prim_path=f"{HEAD_CAMERA_PATH}/QuestHeadCamera",
            frequency=QUEST_STREAM_FPS,
            resolution=(640, 360),
        )
        self.camera.initialize()
        self.bridge = QuestBridge(
            image_height=360,
            eye_width=640,
            port=QUEST_PORT,
            stream_fps=QUEST_STREAM_FPS,
        )
        print("\n[QUEST] WebXR server is starting.", flush=True)
        print(f"[QUEST] 1. Accept the certificate: {self.bridge.certificate_url}", flush=True)
        print(f"[QUEST] 2. Open and press Enter VR: {self.bridge.quest_url}", flush=True)
        print(
            "[QUEST] Hold a controller Grip to drive that arm; trigger closes the gripper.",
            flush=True,
        )

    def reset(self) -> None:
        self.joints.reset()
        self.ik.reset()
        self.reset_requested = False
        self.active = False
        self.grip_was_pressed = {side: False for side in ("left", "right")}
        self.awaiting_grip_release = {side: True for side in ("left", "right")}
        self.arm_control_active = {side: False for side in ("left", "right")}
        self.controller_anchors = {side: None for side in ("left", "right")}
        self.controller_rotation_anchors = {side: None for side in ("left", "right")}
        self.robot_anchors = {side: None for side in ("left", "right")}
        self.robot_rotation_anchors = {side: None for side in ("left", "right")}

    def update(self, dt: float) -> None:
        if self.bridge is None:
            return
        states = {side: self.bridge.state(side) for side in ("left", "right")}
        self.last_grip_values = {
            side: float(states[side]["squeeze"]) for side in ("left", "right")
        }
        for side in ("left", "right"):
            pose = self.bridge.pose(side)
            if _valid_pose(pose):
                self.latest_pose[side] = self._convert_controller_pose(pose)

        reset_pressed = bool(states["right"]["b_button"])
        if reset_pressed and not getattr(self, "_reset_was_pressed", False):
            self.reset()
            print("[QUEST] Right B: arms reset to home", flush=True)
        self._reset_was_pressed = reset_pressed

        current_joints = np.asarray(
            self.robot_joint_positions(), dtype=np.float64
        )
        for side in ("left", "right"):
            self._update_arm(side, current_joints, states[side], dt)

        left_trigger = float(np.clip(states["left"]["trigger"], 0.0, 1.0))
        right_trigger = float(np.clip(states["right"]["trigger"], 0.0, 1.0))
        if GRIPPER_TYPE == "dexterous_hand":
            self.joints.set_hand_curl(left_trigger, right_trigger)
        else:
            self.joints.set_grippers(
                GRIPPER_CLOSED * left_trigger,
                GRIPPER_CLOSED * right_trigger,
            )

    def robot_joint_positions(self) -> np.ndarray:
        return np.asarray(self.robot.get_joint_positions()[0], dtype=np.float64)

    def _update_arm(self, side: str, current_joints: np.ndarray, state: dict, dt: float) -> None:
        indices = self.joints.left_arm_indices if side == "left" else self.joints.right_arm_indices
        current_ee = self._ee_world_pose(side)
        grip_pressed = float(state["squeeze"]) > 0.1
        pose = self.latest_pose.get(side)

        if not grip_pressed:
            self.awaiting_grip_release[side] = False
            self.arm_control_active[side] = False
            self.controller_anchors[side] = None
            self.controller_rotation_anchors[side] = None
            self.robot_anchors[side] = None
            self.robot_rotation_anchors[side] = None
            self.ik.reset(side)
        elif self.awaiting_grip_release[side]:
            self.arm_control_active[side] = False
        elif pose is not None and _valid_pose(pose):
            controller_position = pose[:3, 3].copy()
            if self.controller_anchors[side] is None:
                self.controller_anchors[side] = controller_position
                self.controller_rotation_anchors[side] = pose[:3, :3].copy()
                self.robot_anchors[side] = current_ee[:3, 3].copy()
                self.robot_rotation_anchors[side] = current_ee[:3, :3].copy()
                self.target_positions[side] = current_ee[:3, 3].copy()
                self.target_orientations[side] = current_ee[:3, :3].copy()
                self.ik.reset(side)
                print(f"[QUEST] {side} Grip anchored to current pose", flush=True)
            self.arm_control_active[side] = True
            # controller_to_world converts controller-frame deltas into the
            # robot world frame.  It is used both for 3-vectors (translation)
            # and 3x3 rotations, so it must be a 3x3 rotation matrix.
            controller_to_world = np.eye(3)
            displacement = controller_to_world @ (
                controller_position - self.controller_anchors[side]
            )
            anchor = self.robot_anchors[side]
            desired = anchor + QUEST_POSITION_SCALE * displacement
            self.target_positions[side] = np.clip(
                desired,
                anchor - np.asarray([0.50, 0.50, 0.45]),
                anchor + np.asarray([0.50, 0.50, 0.45]),
            )
            controller_rotation_delta = (
                pose[:3, :3] @ self.controller_rotation_anchors[side].T
            )
            world_rotation_delta = (
                controller_to_world @ controller_rotation_delta @ controller_to_world.T
            )
            self.target_orientations[side] = (
                world_rotation_delta @ self.robot_rotation_anchors[side]
            )
        else:
            self.arm_control_active[side] = False

        if self.arm_control_active[side]:
            position_error = _clamp_norm(
                self.target_positions[side] - current_ee[:3, 3],
                0.025,
            )
            rotation_error = _clamp_norm(
                _rotation_vector(
                    self.target_orientations[side] @ current_ee[:3, :3].T
                ),
                0.060,
            )
            target = self.ik.solve(
                side,
                current_joints[indices],
                current_ee[:3, :3],
                position_error,
                rotation_error,
                dt,
            )
            if target is not None:
                self.joints.target[indices] = np.clip(
                    target,
                    self.joints.limits[indices, 0],
                    self.joints.limits[indices, 1],
                )
        self.grip_was_pressed[side] = grip_pressed

    def stream_camera(self) -> None:
        if self.bridge is None or self.camera is None:
            return
        self.stream_frame += 1
        if self.stream_frame % self.stream_interval:
            return
        rgba = self.camera.get_rgba()
        if rgba is None or not rgba.size:
            return
        self.bridge.write_mono_image(np.asarray(rgba)[..., :3])

    def status(self) -> str:
        if self.bridge is None:
            return "disabled"
        left_updates, right_updates = self.bridge.update_counts
        return (
            f"updates=({left_updates},{right_updates}), "
            f"clutch=({self.arm_control_active['left']},{self.arm_control_active['right']})"
        )

    def close(self) -> None:
        if self.bridge is not None:
            self.bridge.close()
            self.bridge = None


HELP = f"""
{CONFIG['display_name']} — Quest 3/3S teleoperation

Quest:
  Hold left Grip    drive left arm from the left controller
  Hold right Grip   drive right arm from the right controller
  Trigger           close the corresponding gripper / curl the Wuji hand
  Right B           reset arms to home
  Left stick Y      head pitch (optional)

Keyboard:
  W/S, A/D          move left arm joint 2 / joint 1
  I/K, J/L          move right arm joint 2 / joint 1
  T/Y, U/O          close/open left / right gripper
  R                 reset
  P                 print joint state
  ESC               quit
""".strip()


def main() -> None:
    world = None
    quest = None
    subscription = None
    try:
        print(f"[{CONFIG['robot']}] Building stage...", flush=True)
        world, stage, robot = build_world()
        joints = RobotJointCommander(robot)
        quest = TrackedQuestController(joints, stage)
        quest.robot = robot
        for _ in range(20 * PHYSICS_SUBSTEPS):
            joints.apply()
            world.step(render=False)

        viewport = None
        if not HEADLESS:
            viewport = get_active_viewport()
            viewport.set_texture_resolution((1280, 720))
            set_camera_view(
                eye=[1.2, -1.8, 1.4],
                target=[0.0, 0.0, 1.0],
                camera_prim_path="/OmniverseKit_Persp",
            )
        quest.start()

        print(HELP, flush=True)
        print(f"\nLoaded {CONFIG['display_name']}:", flush=True)
        print(f"  robot: {ROBOT_USD}", flush=True)
        print(f"  articulation: {ARTICULATION_PATH}", flush=True)
        print(f"  IK: {quest.ik.version}", flush=True)
        print(f"  DOFs: {robot.num_dof}", flush=True)

        held: set[carb.input.KeyboardInput] = set()
        state = {"quit": False}
        if not HEADLESS and not SMOKE_TEST:
            input_interface = carb.input.acquire_input_interface()
            keyboard = omni.appwindow.get_default_app_window().get_keyboard()

            def on_key(event, *_args):
                key = event.input
                if event.type in (
                    carb.input.KeyboardEventType.KEY_PRESS,
                    carb.input.KeyboardEventType.KEY_REPEAT,
                ):
                    held.add(key)
                    if key == carb.input.KeyboardInput.ESCAPE:
                        state["quit"] = True
                    elif key == carb.input.KeyboardInput.R:
                        quest.reset()
                        print("Reset to home", flush=True)
                    elif key == carb.input.KeyboardInput.P:
                        print(
                            f"joints={np.round(joints.target, 4).tolist()}",
                            flush=True,
                        )
                elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
                    held.discard(key)
                return True

            subscription = input_interface.subscribe_to_keyboard_events(keyboard, on_key)

        initial_targets = joints.target.copy()
        frames = 0
        world.play()
        while simulation_app.is_running() and not state["quit"]:
            if SMOKE_TEST:
                if frames == 0:
                    joints.set_arm_targets(
                        HOME_ARM_DEG["left"] * np.pi / 180.0
                        + np.asarray([0.05, 0.04, 0.02, 0.01, 0.0, 0.0, 0.0]),
                        HOME_ARM_DEG["right"] * np.pi / 180.0
                        + np.asarray([0.05, -0.04, 0.02, -0.01, 0.0, 0.0, 0.0]),
                    )
                    if GRIPPER_TYPE == "dexterous_hand":
                        joints.set_hand_curl(0.5, 0.5)
                    else:
                        joints.set_grippers(GRIPPER_CLOSED * 0.5, GRIPPER_CLOSED * 0.5)
            else:
                # Keyboard arm fallback when Quest is disabled.
                left_j2 = float(
                    (carb.input.KeyboardInput.W in held)
                    - (carb.input.KeyboardInput.S in held)
                )
                left_j1 = float(
                    (carb.input.KeyboardInput.A in held)
                    - (carb.input.KeyboardInput.D in held)
                )
                right_j2 = float(
                    (carb.input.KeyboardInput.I in held)
                    - (carb.input.KeyboardInput.K in held)
                )
                right_j1 = float(
                    (carb.input.KeyboardInput.J in held)
                    - (carb.input.KeyboardInput.L in held)
                )
                rate = float(os.environ.get("TRACKED_KEY_ARM_RATE", "0.02"))
                arm_target = joints.target.copy()
                left_idx = joints.left_arm_indices
                right_idx = joints.right_arm_indices
                arm_target[left_idx[1]] = joints.clamp(
                    LEFT_ARM_NAMES[1],
                    arm_target[left_idx[1]] + rate * left_j2,
                )
                arm_target[left_idx[0]] = joints.clamp(
                    LEFT_ARM_NAMES[0],
                    arm_target[left_idx[0]] + rate * left_j1,
                )
                arm_target[right_idx[1]] = joints.clamp(
                    RIGHT_ARM_NAMES[1],
                    arm_target[right_idx[1]] + rate * right_j2,
                )
                arm_target[right_idx[0]] = joints.clamp(
                    RIGHT_ARM_NAMES[0],
                    arm_target[right_idx[0]] + rate * right_j1,
                )
                joints.target[:] = arm_target
                grip_l = float(
                    (carb.input.KeyboardInput.T in held)
                    - (carb.input.KeyboardInput.Y in held)
                )
                grip_r = float(
                    (carb.input.KeyboardInput.U in held)
                    - (carb.input.KeyboardInput.O in held)
                )
                if GRIPPER_TYPE == "dexterous_hand":
                    joints.set_hand_curl(
                        max(0.0, grip_l),
                        max(0.0, grip_r),
                    )
                else:
                    joints.set_grippers(
                        GRIPPER_CLOSED * max(0.0, grip_l),
                        GRIPPER_CLOSED * max(0.0, grip_r),
                    )
                try:
                    quest.update(1.0 / 60.0)
                except Exception:
                    import traceback

                    traceback.print_exc()
                    print(
                        "[tracked-teleop] Quest update failed; stopping cleanly "
                        "instead of exiting silently.",
                        flush=True,
                    )
                    break
                if quest.reset_requested:
                    quest.reset()
                    quest.reset_requested = False

            joints.apply()
            for _ in range(PHYSICS_SUBSTEPS):
                world.step(render=False)
            if not HEADLESS:
                world.render()
                quest.stream_camera()
            frames += 1
            if SMOKE_TEST and frames % 10 == 0:
                positions = np.asarray(robot.get_joint_positions()[0], dtype=np.float64)
                hand_target = joints.target[joints.left_hand_indices] if len(joints.left_hand_indices) else np.zeros(0)
                hand_pos = positions[joints.left_hand_indices] if len(joints.left_hand_indices) else np.zeros(0)
                hand_error = np.max(np.abs(hand_pos - hand_target)) if len(hand_target) else 0.0
                print(
                    f"smoke frame={frames}: "
                    f"max_arm_error={np.max(np.abs(positions[joints.left_arm_indices] - joints.target[joints.left_arm_indices])):.5f}, "
                    f"hand_error={hand_error:.4f}",
                    flush=True,
                )
            if quest.bridge is not None and frames % 300 == 0:
                print(f"[QUEST] {quest.status()}", flush=True)
            if SMOKE_TEST and frames >= SMOKE_TEST_FRAMES:
                break
            if MAX_FRAMES > 0 and frames >= MAX_FRAMES:
                break

        if SMOKE_TEST:
            positions = np.asarray(robot.get_joint_positions()[0], dtype=np.float64)
            if not np.all(np.isfinite(positions)):
                raise RuntimeError(f"{CONFIG['robot']} smoke test produced non-finite joints")
            print(
                f"[SMOKE] {CONFIG['robot']} OK: DOFs={robot.num_dof} "
                f"frames={frames}",
                flush=True,
            )
    finally:
        if quest is not None:
            quest.close()
        if world is not None:
            world.clear_instance()
        simulation_app.close()


if __name__ == "__main__":
    main()