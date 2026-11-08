#!/usr/bin/env python3
"""Shared scene assembly and Quest teleoperation for config-driven W1 tasks.

This script intentionally does not use DirectRLEnv.  It only composes the
environment, task object, and robot in one Isaac Sim world.  Keyboard input
drives the mobile base and lift while Quest 3S drives the W1 upper body.
"""

from __future__ import annotations

import json
import math
import os
import sys
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
from isaacsim import SimulationApp


SCRIPT_DIR = Path(__file__).resolve().parent
TASK_ROOT_ENV = os.environ.get("W1_TASK_ROOT")
if not TASK_ROOT_ENV:
    raise RuntimeError(
        "W1_TASK_ROOT is required; launch through a task package's run_task.sh"
    )
TASK_ROOT = Path(TASK_ROOT_ENV).resolve()
if not TASK_ROOT.is_dir():
    raise FileNotFoundError(f"W1 task package does not exist: {TASK_ROOT}")
CONFIG_DIR = TASK_ROOT / "config"
TASK_CONFIG_PATH = CONFIG_DIR / "task.json"
TASK_CONFIG = json.loads(TASK_CONFIG_PATH.read_text(encoding="utf-8"))
TASK_NAME = str(TASK_CONFIG.get("name", TASK_ROOT.name))


def _configured_asset(name: str) -> Path:
    try:
        relative_path = TASK_CONFIG["assets"][name]
    except (KeyError, TypeError) as error:
        raise RuntimeError(f"Missing asset '{name}' in {TASK_CONFIG_PATH}") from error
    path = (TASK_ROOT / relative_path).resolve()
    if not path.is_relative_to(TASK_ROOT):
        raise RuntimeError(f"Asset '{name}' escapes task package: {path}")
    return path


ROBOT_PACKAGE = TASK_ROOT / "assets" / "robot" / "Forklift_W1_Final"
ROBOT_TOOLS = ROBOT_PACKAGE / "tools"
PLACEMENT_PATH = TASK_ROOT / TASK_CONFIG["placement_config"]
RUNTIME_STATE_CAPTURE_PATH = CONFIG_DIR / "runtime_state_capture.json"

ENVIRONMENT_VISUAL_ASSET = _configured_asset("environment_visual")
ENVIRONMENT_COLLISION_ASSET = _configured_asset("environment_collision")
ROBOT_ASSET = _configured_asset("robot")
TASK_OBJECT_ASSET = _configured_asset("task_object")

for asset_path in (
    ENVIRONMENT_VISUAL_ASSET,
    ENVIRONMENT_COLLISION_ASSET,
    ROBOT_ASSET,
    TASK_OBJECT_ASSET,
):
    if not asset_path.is_file():
        raise FileNotFoundError(f"Task asset does not exist: {asset_path}")


def _load_placement() -> dict:
    if not PLACEMENT_PATH.is_file():
        raise FileNotFoundError(f"Task placement file is missing: {PLACEMENT_PATH}")
    data = json.loads(PLACEMENT_PATH.read_text(encoding="utf-8"))
    try:
        robot = data["robot"]
        task_object = data["task_object"]
        environment = data["environment"]
        camera = data["camera"]
        object_scale = tuple(
            map(float, task_object.get("scale_xyz", (1.0, 1.0, 1.0)))
        )
        required = (
            float(robot["x"]),
            float(robot["y"]),
            float(robot["yaw_radians"]),
            float(robot["lift_height_m"]),
            *map(float, task_object["position_xyz_m"]),
            *map(float, task_object["orientation_wxyz"]),
            *object_scale,
            *map(float, environment["position_xyz_m"]),
            float(environment["rotation_x_degrees"]),
            *map(float, camera["eye_xyz_m"]),
            *map(float, camera["target_xyz_m"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise RuntimeError(f"Invalid task placement file: {PLACEMENT_PATH}") from error
    if len(required) != 24 or any(value <= 0.0 for value in object_scale):
        raise RuntimeError(f"Invalid vector lengths in {PLACEMENT_PATH}")
    task_object["scale_xyz"] = list(object_scale)
    joint_state = data.get("robot_joint_state")
    if joint_state is not None:
        targets = joint_state.get("position_targets")
        if not isinstance(targets, dict) or not targets:
            raise RuntimeError(
                f"Invalid robot_joint_state.position_targets in {PLACEMENT_PATH}"
            )
        try:
            if not all(math.isfinite(float(value)) for value in targets.values()):
                raise ValueError("non-finite joint target")
        except (TypeError, ValueError) as error:
            raise RuntimeError(
                f"Invalid robot joint targets in {PLACEMENT_PATH}"
            ) from error
    return data


PLACEMENT = _load_placement()

HEADLESS = os.environ.get(
    "W1_DRILLING_HEADLESS", os.environ.get("TASK001_HEADLESS", "0")
) == "1"
SMOKE_TEST = os.environ.get(
    "W1_DRILLING_SMOKE_TEST", os.environ.get("TASK001_SMOKE_TEST", "0")
) == "1"
MAX_FRAMES = int(
    os.environ.get(
        "W1_DRILLING_MAX_FRAMES", os.environ.get("TASK001_MAX_FRAMES", "0")
    )
)
ENABLE_SCENE_COLLISION = os.environ.get(
    "W1_DRILLING_SCENE_COLLISION",
    os.environ.get("TASK001_SCENE_COLLISION", "1"),
) != "0"
STRICT_JOINT_HOLD = os.environ.get(
    "W1_DRILLING_STRICT_JOINT_HOLD",
    os.environ.get("TASK001_STRICT_JOINT_HOLD", "1"),
) != "0"
QUEST_ENABLED = os.environ.get("W1_QUEST_ENABLED", "1") != "0"
QUEST_PORT = int(os.environ.get("W1_QUEST_PORT", "8012"))
QUEST_STREAM_FPS = int(os.environ.get("W1_QUEST_STREAM_FPS", "20"))
QUEST_POSITION_SCALE = float(os.environ.get("W1_QUEST_POSITION_SCALE", "0.7"))
QUEST_IK_DAMPING = float(os.environ.get("W1_QUEST_IK_DAMPING", "0.08"))
QUEST_IK_ROTATION_WEIGHT = float(
    os.environ.get("W1_QUEST_IK_ROTATION_WEIGHT", "0.35")
)
QUEST_IK_BACKEND = os.environ.get("W1_QUEST_IK_BACKEND", "pinocchio").strip().lower()
if QUEST_IK_BACKEND not in ("pinocchio", "realman", "dls"):
    raise RuntimeError(
        "W1_QUEST_IK_BACKEND must be 'pinocchio', 'realman', or 'dls', got "
        f"{QUEST_IK_BACKEND!r}"
    )
REALMAN_MODEL_SCALE = float(os.environ.get("W1_REALMAN_MODEL_SCALE", "1.1"))
QUEST_HEAD_RATE = float(os.environ.get("W1_QUEST_HEAD_RATE", "0.35"))
QUEST_WAIST_RATE = float(os.environ.get("W1_QUEST_WAIST_RATE", "0.45"))
QUEST_GRIPPER_RATE = float(os.environ.get("W1_QUEST_GRIPPER_RATE", "1.25"))
SHOW_CAMERA_VIEWPORTS = os.environ.get("W1_SHOW_CAMERA_VIEWPORTS", "1") != "0"
KEYBOARD_UPPER_BODY = os.environ.get("W1_KEYBOARD_UPPER_BODY", "0") == "1"
KEYBOARD_ARM_RATE = float(os.environ.get("W1_KEYBOARD_ARM_RATE", "0.65"))
KEYBOARD_GRIPPER_RATE = float(os.environ.get("W1_KEYBOARD_GRIPPER_RATE", "1.25"))
KEYBOARD_HEAD_RATE = float(os.environ.get("W1_KEYBOARD_HEAD_RATE", "0.75"))
KEYBOARD_WAIST_RATE = float(os.environ.get("W1_KEYBOARD_WAIST_RATE", "0.45"))
GRIPPER_OPEN_POSITION = 0.0
GRIPPER_CLOSED_POSITION = math.radians(40.0)

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

# Isaac Sim modules must be imported after SimulationApp is created.
import carb.input
import omni.appwindow
import omni.usd
from isaacsim.core.api import World
from isaacsim.core.prims import Articulation, RigidPrim
from isaacsim.core.utils.stage import create_new_stage
from isaacsim.core.utils.viewports import set_camera_view
from isaacsim.sensors.camera import Camera
from omni.kit.viewport.utility import create_viewport_window, get_active_viewport
from omni.kit.viewport.utility.camera_state import ViewportCameraState
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics

sys.path.insert(0, str(ROBOT_TOOLS))
from common import (  # noqa: E402
    COMPOSITE_PATH,
    FRONT_CAMERA_PATH,
    LEFT_END_EFFECTOR_PATH,
    ROBOT_PATH,
    RIGHT_END_EFFECTOR_PATH,
    ROOT_JOINT_PATH,
    VEHICLE_PATH,
    JointCommander,
    MobileAssembly,
    make_w1_mobile,
    multiply_quat_wxyz,
    wait_for_stage,
)
from w1_quest3s import QuestBridge  # noqa: E402
from w1_realman_ik import RealManRM75IK  # noqa: E402
from w1_pinocchio_ik import W1PinocchioIK  # noqa: E402


ENVIRONMENT_VISUAL_PATH = "/World/EnvironmentVisual"
ENVIRONMENT_COLLISION_PATH = "/World/EnvironmentCollision"
TASK_OBJECT_PATH = "/World/TaskObject"
LIFT_CARRIAGE_PATH = f"{COMPOSITE_PATH}/LiftCarriage"
SUPPORT_PATH = f"{LIFT_CARRIAGE_PATH}/Support"
FIXED_CARRIAGE_VISUAL_PATH = f"{COMPOSITE_PATH}/Lift/CarriageVisual"
LEFT_WRIST_CAMERA_PATH = (
    LEFT_END_EFFECTOR_PATH.rsplit("/left_functional_gripper", maxsplit=1)[0]
    + "/benchmark_sensors/left_wrist_camera"
)
RIGHT_WRIST_CAMERA_PATH = (
    RIGHT_END_EFFECTOR_PATH.rsplit("/right_functional_gripper", maxsplit=1)[0]
    + "/benchmark_sensors/right_wrist_camera"
)
WAIST_PITCH_JOINT_PATH = f"{ROBOT_PATH}/Physics/waist_pitch_joint"
DISPLAY_CAMERA_PATHS = {
    "head": FRONT_CAMERA_PATH,
    "left_wrist": LEFT_WRIST_CAMERA_PATH,
    "right_wrist": RIGHT_WRIST_CAMERA_PATH,
}

ROBOT_SPAWN_XY_YAW = tuple(
    float(PLACEMENT["robot"][name]) for name in ("x", "y", "yaw_radians")
)
ROBOT_SPAWN_LIFT_HEIGHT_M = float(PLACEMENT["robot"]["lift_height_m"])
OBJECT_SPAWN_XYZ_M = tuple(map(float, PLACEMENT["task_object"]["position_xyz_m"]))
OBJECT_SPAWN_ORIENTATION_WXYZ = tuple(
    map(float, PLACEMENT["task_object"]["orientation_wxyz"])
)
OBJECT_SCALE_XYZ = tuple(map(float, PLACEMENT["task_object"]["scale_xyz"]))
ENVIRONMENT_SPAWN_XYZ_M = tuple(
    map(float, PLACEMENT["environment"]["position_xyz_m"])
)
REALSCENE_ROTATE_X_DEG = float(PLACEMENT["environment"]["rotation_x_degrees"])
CAMERA_EYE_XYZ_M = tuple(map(float, PLACEMENT["camera"]["eye_xyz_m"]))
CAMERA_TARGET_XYZ_M = tuple(map(float, PLACEMENT["camera"]["target_xyz_m"]))

UPPER_BODY_HELP = (
    """Keyboard upper body:
  1 ... 7     select arm joint
  Z / X       selected LEFT joint - / +
  C / V       selected RIGHT joint - / +
  N / M       left gripper close / open
  , / .       right gripper close / open
  G / H       both grippers close / open
  J / L       head yaw left / right
  I / K       head pitch up / down
  U / O       waist bend / straighten"""
    if KEYBOARD_UPPER_BODY
    else """Quest 3S upper body:
  Left/Right Grip       release once, then hold to clutch/move that arm
  Left/Right Trigger    close the corresponding gripper
  Left X / Y            bend / straighten waist
  Left stick horizontal head yaw left / right
  Right A / Stick click head pitch down / up
  Right B               reset arms, grippers, head, and task object"""
)

HELP = f"""
{TASK_NAME} {'keyboard teleoperation' if KEYBOARD_UPPER_BODY else 'keyboard + Quest 3S teleoperation'}

Tracked base:
  W / S       forward / reverse
  A / D       turn left / right

Lift carriage:
  T / Y       raise / lower W1 and support

{UPPER_BODY_HELP}

Other:
  R           reset robot pose and joint targets (the object is not reset)
  P           save robot/object placement and current viewport camera
  ESC         quit

Quest arm start:
  Enter VR with both Grips released. Put your arms in a comfortable spread
  pose, then press/hold each Grip. That instant anchors each controller to the
  W1 arm's current pose without an initial jump.
""".strip()


class StableMobileAssembly(MobileAssembly):
    """Drive the planar chassis and vertical W1 lift carriage.

    Vehicle and LiftCarriage are independent kinematic bodies. The asset's
    root_joint fixes the dynamic W1 base_link to LiftCarriage, so changing the
    carriage Z target raises the support and W1 as one assembly.
    """

    _ZERO_ROOT_VELOCITY = np.zeros((1, 6), dtype=np.float32)

    def __init__(
        self,
        stage: Usd.Stage,
        robot: Articulation,
        vehicle_body: RigidPrim,
        lift_carriage: RigidPrim,
    ) -> None:
        self.vehicle_body = vehicle_body
        self.lift_carriage = lift_carriage
        lift_prim = stage.GetPrimAtPath(LIFT_CARRIAGE_PATH)
        if not lift_prim:
            raise RuntimeError(f"Lift carriage is missing: {LIFT_CARRIAGE_PATH}")
        self.lift_attributes = {
            name: lift_prim.GetAttribute(f"liftDrive:{name}")
            for name in (
                "height",
                "targetHeight",
                "lowerLimit",
                "upperLimit",
                "maxSpeed",
            )
        }
        missing = [name for name, attr in self.lift_attributes.items() if not attr]
        if missing:
            raise RuntimeError(f"Missing lift-drive attributes: {missing}")
        lift_xform = UsdGeom.Xformable(lift_prim)
        self.lift_translate_op = next(
            (
                op
                for op in lift_xform.GetOrderedXformOps()
                if op.GetOpName() == "xformOp:translate"
            ),
            None,
        )
        if self.lift_translate_op is None:
            self.lift_translate_op = lift_xform.AddTranslateOp()
        support_prim = stage.GetPrimAtPath(SUPPORT_PATH)
        if not support_prim:
            raise RuntimeError(f"Lift support is missing: {SUPPORT_PATH}")
        self.support_prim = support_prim
        self.support_zero_height_z = float(
            UsdGeom.XformCache()
            .GetLocalToWorldTransform(support_prim)
            .ExtractTranslation()[2]
        )
        fixed_carriage_prim = stage.GetPrimAtPath(FIXED_CARRIAGE_VISUAL_PATH)
        if not fixed_carriage_prim:
            raise RuntimeError(
                f"Fixed carriage visual is missing: {FIXED_CARRIAGE_VISUAL_PATH}"
            )
        self.fixed_carriage_prim = fixed_carriage_prim
        self.fixed_carriage_zero_height_z = float(
            UsdGeom.XformCache()
            .GetLocalToWorldTransform(fixed_carriage_prim)
            .ExtractTranslation()[2]
        )
        self.lift_lower_limit = float(self.lift_attributes["lowerLimit"].Get())
        self.lift_upper_limit = float(self.lift_attributes["upperLimit"].Get())
        self.lift_max_speed = float(self.lift_attributes["maxSpeed"].Get())
        self.lift_height = float(self.lift_attributes["height"].Get())
        super().__init__(stage, robot, vehicle_body, lift_carriage)

    def _target_robot_pose(self) -> tuple[np.ndarray, np.ndarray]:
        half = 0.5 * self.yaw
        cosine = np.cos(self.yaw)
        sine = np.sin(self.yaw)
        authored = self.initial_robot_position
        position = np.asarray(
            [
                self.x + cosine * authored[0] - sine * authored[1],
                self.y + sine * authored[0] + cosine * authored[1],
                authored[2] + self.lift_height,
            ],
            dtype=np.float32,
        )
        yaw_orientation = np.asarray(
            [np.cos(half), 0.0, 0.0, np.sin(half)],
            dtype=np.float32,
        )
        orientation = multiply_quat_wxyz(
            yaw_orientation,
            self.initial_robot_orientation,
        )
        return position, orientation

    def apply(self) -> None:
        half = 0.5 * self.yaw
        orientation = np.asarray(
            [np.cos(half), 0.0, 0.0, np.sin(half)],
            dtype=np.float32,
        )
        self.translate_op.Set(Gf.Vec3d(self.x, self.y, 0.0))
        self.orient_op.Set(
            Gf.Quatf(
                float(orientation[0]),
                Gf.Vec3f(0.0, 0.0, float(orientation[3])),
            )
        )
        # The tensor API moves the kinematic actor in PhysX but does not author
        # the USD local transform inherited by non-rigid visual children. Keep
        # the carriage root transform in sync so both truss supports and the
        # visible carriage move with W1.
        self.lift_translate_op.Set(Gf.Vec3d(0.0, 0.0, self.lift_height))
        self.vehicle_body.set_world_poses(
            positions=np.asarray([[self.x, self.y, 0.0]], dtype=np.float32),
            orientations=orientation[np.newaxis, :],
        )
        self.lift_carriage.set_world_poses(
            positions=np.asarray(
                [[self.x, self.y, self.lift_height]],
                dtype=np.float32,
            ),
            orientations=orientation[np.newaxis, :],
        )
        # PhysX does not advance this externally mounted articulation root from
        # a moving kinematic parent reliably. Keep the asset fixed joint as the
        # persistent mount definition, and synchronize the dynamic W1 root to
        # the same target pose so no visual/physical gap can accumulate.
        robot_position, robot_orientation = self._target_robot_pose()
        self.robot.set_world_poses(
            positions=robot_position[np.newaxis, :],
            orientations=robot_orientation[np.newaxis, :],
        )
        self.lift_attributes["height"].Set(float(self.lift_height))
        self.lift_attributes["targetHeight"].Set(float(self.lift_height))

    def move_lift(self, delta: float) -> None:
        self.lift_height = float(
            np.clip(
                self.lift_height + delta,
                self.lift_lower_limit,
                self.lift_upper_limit,
            )
        )

    def reset_lift(self) -> None:
        self.lift_height = float(
            np.clip(0.0, self.lift_lower_limit, self.lift_upper_limit)
        )

    def set_pose(self, x: float, y: float, yaw: float) -> None:
        """Move both constrained bodies together for spawn/reset only."""
        self.x = float(x)
        self.y = float(y)
        self.yaw = float(yaw)
        self.apply()
        position, orientation = self._target_robot_pose()
        self.robot.set_world_poses(
            positions=position[np.newaxis, :],
            orientations=orientation[np.newaxis, :],
        )
        if self.robot.is_physics_handle_valid():
            self.robot.set_velocities(self._ZERO_ROOT_VELOCITY)


class StableJointCommander(JointCommander):
    """Deterministic position control for the benchmark's movable W1 joints."""

    def __init__(self, robot: Articulation):
        super().__init__(robot)
        # The task uses waist_pitch for Quest bending but continues to lock the
        # prismatic waist_updown joint.  Keep the original 18 benchmark output
        # channels unchanged while adding waist_pitch to runtime commands.
        self.waist_pitch_index = self.index["waist_pitch_joint"]
        self.locked_indices = np.asarray(
            [self.index["waist_updown_joint"]],
            dtype=np.int64,
        )
        self.command_indices = np.concatenate(
            (
                self.control_indices,
                np.asarray([self.waist_pitch_index], dtype=np.int64),
                self.locked_indices,
            )
        ).astype(np.int64)
        self.gripper_hold_exclusions = np.asarray(
            [self.index[name] for name in self.GRIPPER_NAMES],
            dtype=np.int64,
        )
        self.arm_hold_exclusions = (
            np.asarray(
                [
                    self.index[f"{side}_joint{joint}"]
                    for side in ("left", "right")
                    for joint in range(1, 8)
                ],
                dtype=np.int64,
            )
            if QUEST_IK_BACKEND == "pinocchio"
            else np.empty(0, dtype=np.int64)
        )
        rigid_hold_exclusions = set(
            np.concatenate(
                (self.gripper_hold_exclusions, self.arm_hold_exclusions)
            ).tolist()
        )
        self.rigid_hold_indices = np.asarray(
            [
                index
                for index in self.command_indices
                if index not in rigid_hold_exclusions
            ],
            dtype=np.int64,
        )
        self.zero_command_velocity = np.zeros(
            (1, len(self.command_indices)),
            dtype=np.float32,
        )
        self.zero_rigid_hold_velocity = np.zeros(
            (1, len(self.rigid_hold_indices)),
            dtype=np.float32,
        )
        self.snap_to_targets()
        self.home_target = self.target.copy()

    def _command_positions(self) -> np.ndarray:
        self.target[self.locked_indices] = 0.0
        return self.target[self.command_indices][np.newaxis, :]

    def apply(self) -> None:
        positions = self._command_positions()
        if STRICT_JOINT_HOLD:
            # The W1 articulation has a floating dynamic root in Isaac Sim 6.0.
            # Enforce head/waist directly so root constraint impulses cannot
            # produce visible drift. Grippers are always excluded so their
            # mimic links follow PhysX drives. Pinocchio-controlled arms are
            # also excluded: their bounded QP targets must be followed through
            # the drives instead of teleporting every rigid link each frame.
            self.robot.set_joint_positions(
                self.target[self.rigid_hold_indices][np.newaxis, :],
                joint_indices=self.rigid_hold_indices,
            )
            self.robot.set_joint_velocities(
                self.zero_rigid_hold_velocity,
                joint_indices=self.rigid_hold_indices,
            )
        self.robot.set_joint_position_targets(
            positions,
            joint_indices=self.command_indices,
        )
        self.robot.set_joint_velocity_targets(
            self.zero_command_velocity,
            joint_indices=self.command_indices,
        )

    def stop_motion(self) -> None:
        """Clear actual and target velocities without changing joint pose."""
        self.robot.set_joint_velocities(
            self.zero_command_velocity,
            joint_indices=self.command_indices,
        )
        self.apply()

    def snap_to_targets(self) -> None:
        """Initialize/reset exactly at the hold pose with no residual motion."""
        positions = self._command_positions()
        self.robot.set_joint_positions(
            positions,
            joint_indices=self.command_indices,
        )
        self.robot.set_joint_velocities(
            self.zero_command_velocity,
            joint_indices=self.command_indices,
        )
        self.apply()

    def capture_home(self) -> None:
        """Use the current commanded pose as the persistent reset pose."""
        self.home_target = self.target.copy()
        self.home_target[self.locked_indices] = 0.0

    def reset(self) -> None:
        self.target[:] = self.home_target
        self.target[self.locked_indices] = 0.0
        self.snap_to_targets()


def _damped_least_squares(
    jacobian: np.ndarray,
    error: np.ndarray,
    damping: float,
) -> np.ndarray:
    regularizer = (damping * damping) * np.eye(
        jacobian.shape[0],
        dtype=np.float64,
    )
    return jacobian.T @ np.linalg.solve(
        jacobian @ jacobian.T + regularizer,
        error,
    )


def _clamp_norm(vector: np.ndarray, maximum: float) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm > maximum and norm > 1.0e-9:
        return vector * (maximum / norm)
    return vector


def _quat_wxyz_to_rotation_matrix(quaternion: np.ndarray) -> np.ndarray:
    """Convert an Isaac wxyz quaternion to a normalized 3x3 rotation."""
    values = np.asarray(quaternion, dtype=np.float64)
    norm = float(np.linalg.norm(values))
    if values.shape != (4,) or norm < 1.0e-9:
        raise RuntimeError(f"Invalid end-effector quaternion: {values}")
    w, x, y, z = values / norm
    return np.asarray(
        [
            [
                1.0 - 2.0 * (y * y + z * z),
                2.0 * (x * y - z * w),
                2.0 * (x * z + y * w),
            ],
            [
                2.0 * (x * y + z * w),
                1.0 - 2.0 * (x * x + z * z),
                2.0 * (y * z - x * w),
            ],
            [
                2.0 * (x * z - y * w),
                2.0 * (y * z + x * w),
                1.0 - 2.0 * (x * x + y * y),
            ],
        ],
        dtype=np.float64,
    )


def _rotation_vector(rotation: np.ndarray) -> np.ndarray:
    """Return the shortest axis-angle vector represented by a 3x3 matrix."""
    matrix = np.asarray(rotation, dtype=np.float64)
    cosine = float(np.clip((np.trace(matrix) - 1.0) * 0.5, -1.0, 1.0))
    angle = math.acos(cosine)
    skew = np.asarray(
        [
            matrix[2, 1] - matrix[1, 2],
            matrix[0, 2] - matrix[2, 0],
            matrix[1, 0] - matrix[0, 1],
        ],
        dtype=np.float64,
    )
    if angle < 1.0e-6:
        return 0.5 * skew
    sine = math.sin(angle)
    if abs(sine) > 1.0e-5:
        return (angle / (2.0 * sine)) * skew
    axis = np.sqrt(np.maximum((np.diag(matrix) + 1.0) * 0.5, 0.0))
    if float(np.linalg.norm(axis)) < 1.0e-6:
        return np.zeros(3, dtype=np.float64)
    axis[0] = math.copysign(axis[0], skew[0] if abs(skew[0]) > 1.0e-9 else 1.0)
    axis[1] = math.copysign(axis[1], skew[1] if abs(skew[1]) > 1.0e-9 else 1.0)
    axis[2] = math.copysign(axis[2], skew[2] if abs(skew[2]) > 1.0e-9 else 1.0)
    return angle * axis / np.linalg.norm(axis)


class QuestUpperBodyController:
    """Quest relative 6D IK for both arms, grippers, and head joints."""

    SIDES = ("left", "right")

    def __init__(
        self,
        robot: Articulation,
        joints: StableJointCommander,
    ) -> None:
        self.robot = robot
        self.joints = joints
        self.bridge: QuestBridge | None = None
        self.quest_camera: Camera | None = None
        self.frame_index = 0
        self.stream_interval = max(1, round(60 / QUEST_STREAM_FPS))

        self.eef_views = {
            "left": RigidPrim(
                prim_paths_expr=LEFT_END_EFFECTOR_PATH,
                name="w1_task_left_eef",
                reset_xform_properties=False,
                prepare_contact_sensors=False,
            ),
            "right": RigidPrim(
                prim_paths_expr=RIGHT_END_EFFECTOR_PATH,
                name="w1_task_right_eef",
                reset_xform_properties=False,
                prepare_contact_sensors=False,
            ),
        }
        for side, view in self.eef_views.items():
            view.initialize()
            if not view.is_physics_handle_valid():
                raise RuntimeError(f"Quest {side} end-effector view did not initialize")

        dof_names = list(robot.dof_names)
        body_names = list(robot.body_names)
        dof_index = {name: index for index, name in enumerate(dof_names)}
        self.arm_indices = {
            side: np.asarray(
                [dof_index[f"{side}_joint{joint}"] for joint in range(1, 8)],
                dtype=np.int32,
            )
            for side in self.SIDES
        }
        self.gripper_indices = {
            side: dof_index[f"{side}_gripper_joint"] for side in self.SIDES
        }
        self.head_yaw_index = dof_index["head_yaw_joint"]
        self.head_pitch_index = dof_index["head_pitch_joint"]
        self.waist_pitch_index = dof_index["waist_pitch_joint"]
        eef_body_indices = {
            side: body_names.index(f"{side}_functional_gripper")
            for side in self.SIDES
        }

        jacobian_shape = tuple(robot.get_jacobian_shape())
        jacobian_dof_offset = jacobian_shape[2] - robot.num_dof
        if jacobian_dof_offset not in (0, 6):
            raise RuntimeError(
                "Unexpected Quest Jacobian DOF layout: "
                f"jacobian={jacobian_shape}, DOFs={robot.num_dof}"
            )
        self.arm_jacobian_indices = {
            side: indices + jacobian_dof_offset
            for side, indices in self.arm_indices.items()
        }
        if jacobian_shape[0] == len(body_names):
            self.eef_jacobian_indices = eef_body_indices
        elif jacobian_shape[0] == len(body_names) - 1:
            self.eef_jacobian_indices = {
                side: body_index - 1
                for side, body_index in eef_body_indices.items()
            }
        else:
            raise RuntimeError(
                "Unexpected Quest Jacobian/body layout: "
                f"jacobian={jacobian_shape}, bodies={len(body_names)}"
            )

        home_poses = self._eef_poses()
        self.home_positions = {
            side: pose[0].copy() for side, pose in home_poses.items()
        }
        self.target_positions = {
            side: value.copy() for side, value in self.home_positions.items()
        }
        self.target_orientations = {
            side: home_poses[side][1].copy() for side in self.SIDES
        }
        self.controller_anchors = {side: None for side in self.SIDES}
        self.controller_rotation_anchors = {side: None for side in self.SIDES}
        self.robot_anchors = {side: None for side in self.SIDES}
        self.robot_rotation_anchors = {side: None for side in self.SIDES}
        self.grip_was_pressed = {side: False for side in self.SIDES}
        # A Quest Grip is a clutch, not an absolute pose command.  Require one
        # released sample after entering WebXR or resetting, then anchor the
        # current controller pose to the robot's current end-effector pose on
        # the next press.  This prevents stale/partially initialized WebXR
        # state from choosing an arbitrary arm origin.
        self.awaiting_grip_release = {side: True for side in self.SIDES}
        self.arm_control_active = {side: False for side in self.SIDES}
        self.last_grip_values = {side: 0.0 for side in self.SIDES}
        self.reset_was_pressed = False
        self.gripper_open_position = GRIPPER_OPEN_POSITION
        self.gripper_closed_position = GRIPPER_CLOSED_POSITION
        self.pinocchio_ik: W1PinocchioIK | None = None
        self.realman_ik: RealManRM75IK | None = None
        if not KEYBOARD_UPPER_BODY and QUEST_IK_BACKEND == "pinocchio":
            try:
                self.pinocchio_ik = W1PinocchioIK(
                    omni.usd.get_context().get_stage(),
                    ROBOT_PATH,
                    {
                        "left": LEFT_END_EFFECTOR_PATH,
                        "right": RIGHT_END_EFFECTOR_PATH,
                    },
                    model_scale=REALMAN_MODEL_SCALE,
                )
            except RuntimeError as error:
                print(
                    f"[IK] Pinocchio backend unavailable; using legacy DLS: {error}",
                    flush=True,
                )
        elif not KEYBOARD_UPPER_BODY and QUEST_IK_BACKEND == "realman":
            try:
                self.realman_ik = RealManRM75IK(
                    model_scale=REALMAN_MODEL_SCALE,
                )
            except RuntimeError as error:
                print(
                    f"[IK] RealMan backend unavailable; using legacy DLS: {error}",
                    flush=True,
                )

        # WebXR axes: +X right, +Y up, -Z forward.  At chassis yaw zero the
        # W1 faces world -X, with its right side toward world +Y.
        self.webxr_to_robot_zero_yaw = np.asarray(
            [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            dtype=np.float64,
        )
        expected = {
            "forward": (
                np.asarray([0.0, 0.0, -1.0]),
                np.asarray([-1.0, 0.0, 0.0]),
            ),
            "right": (
                np.asarray([1.0, 0.0, 0.0]),
                np.asarray([0.0, 1.0, 0.0]),
            ),
            "up": (
                np.asarray([0.0, 1.0, 0.0]),
                np.asarray([0.0, 0.0, 1.0]),
            ),
        }
        for direction, (webxr_axis, robot_axis) in expected.items():
            mapped = self.webxr_to_robot_zero_yaw @ webxr_axis
            if not np.allclose(mapped, robot_axis):
                raise RuntimeError(
                    f"Invalid Quest {direction} axis mapping: {mapped} != {robot_axis}"
                )

        print(
            "  "
            f"{'Keyboard upper-body' if KEYBOARD_UPPER_BODY else 'Quest IK'}: "
            f"Jacobian={jacobian_shape}, "
            f"base offset={jacobian_dof_offset}",
            flush=True,
        )
        if not KEYBOARD_UPPER_BODY:
            if self.pinocchio_ik is not None:
                print(
                    "  Quest IK backend: constrained QP with "
                    f"{self.pinocchio_ik.version} (legacy DLS fallback enabled)",
                    flush=True,
                )
            elif self.realman_ik is not None:
                print(
                    "  Quest IK backend: RealMan RM75 official API "
                    f"v{self.realman_ik.version} (legacy DLS fallback enabled)",
                    flush=True,
                )
            else:
                print("  Quest IK backend: legacy DLS", flush=True)

    def _eef_poses(self) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        poses: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for side, view in self.eef_views.items():
            positions, orientations = view.get_world_poses()
            poses[side] = (
                np.asarray(positions[0], dtype=np.float64),
                _quat_wxyz_to_rotation_matrix(
                    np.asarray(orientations[0], dtype=np.float64)
                ),
            )
        return poses

    def _eef_positions(self) -> dict[str, np.ndarray]:
        return {
            side: pose[0] for side, pose in self._eef_poses().items()
        }

    def start(self) -> None:
        if not QUEST_ENABLED or HEADLESS or SMOKE_TEST:
            return
        current_stage = omni.usd.get_context().get_stage()
        if not current_stage.GetPrimAtPath(FRONT_CAMERA_PATH).IsValid():
            raise RuntimeError(f"Quest head camera is missing: {FRONT_CAMERA_PATH}")
        self.quest_camera = Camera(
            prim_path=FRONT_CAMERA_PATH,
            frequency=QUEST_STREAM_FPS,
            resolution=(640, 360),
        )
        self.quest_camera.initialize()
        self.bridge = QuestBridge(
            image_height=360,
            eye_width=640,
            port=QUEST_PORT,
            stream_fps=QUEST_STREAM_FPS,
        )
        print("\n[QUEST] W1 WebXR server is starting.", flush=True)
        print(
            f"[QUEST] 1. Accept the certificate: {self.bridge.certificate_url}",
            flush=True,
        )
        print(
            f"[QUEST] 2. Open and press Enter VR: {self.bridge.quest_url}",
            flush=True,
        )

    def reset(self) -> None:
        self.joints.reset()
        if self.pinocchio_ik is not None:
            self.pinocchio_ik.reset()
        current = self._eef_poses()
        self.home_positions = {
            side: pose[0].copy() for side, pose in current.items()
        }
        self.target_positions = {
            side: pose[0].copy() for side, pose in current.items()
        }
        self.target_orientations = {
            side: pose[1].copy() for side, pose in current.items()
        }
        self.controller_anchors = {side: None for side in self.SIDES}
        self.controller_rotation_anchors = {side: None for side in self.SIDES}
        self.robot_anchors = {side: None for side in self.SIDES}
        self.robot_rotation_anchors = {side: None for side in self.SIDES}
        self.grip_was_pressed = {side: False for side in self.SIDES}
        self.awaiting_grip_release = {side: True for side in self.SIDES}
        self.arm_control_active = {side: False for side in self.SIDES}
        self.last_grip_values = {side: 0.0 for side in self.SIDES}

    @staticmethod
    def _reset_task_object(
        task_object_body: RigidPrim,
        placement_state: dict,
    ) -> None:
        """Restore the task object pose and remove residual rigid-body motion."""
        saved_object = placement_state["task_object"]
        position = np.asarray(
            [saved_object["position_xyz_m"]],
            dtype=np.float32,
        )
        orientation = np.asarray(
            [saved_object["orientation_wxyz"]],
            dtype=np.float32,
        )
        task_object_body.set_world_poses(
            positions=position,
            orientations=orientation,
        )
        task_object_body.set_velocities(
            np.zeros((1, 6), dtype=np.float32),
        )

    def update(
        self,
        base: StableMobileAssembly,
        task_object_body: RigidPrim,
        placement_state: dict,
        dt: float,
    ) -> None:
        if self.bridge is None:
            return

        hand_poses = {side: self.bridge.pose(side) for side in self.SIDES}
        states = {side: self.bridge.state(side) for side in self.SIDES}
        self.last_grip_values = {
            side: float(states[side]["squeeze"]) for side in self.SIDES
        }
        reset_pressed = bool(states["right"]["b_button"])
        if reset_pressed and not self.reset_was_pressed:
            self.reset()
            self._reset_task_object(task_object_body, placement_state)
            print(
                "[QUEST] B: arms, grippers, head, and task object reset",
                flush=True,
            )
        self.reset_was_pressed = reset_pressed
        if reset_pressed:
            return

        current_joints = np.asarray(
            self.robot.get_joint_positions()[0],
            dtype=np.float64,
        )
        current_eef_poses = self._eef_poses()
        current_eef_positions = {
            side: pose[0] for side, pose in current_eef_poses.items()
        }
        current_eef_orientations = {
            side: pose[1] for side, pose in current_eef_poses.items()
        }

        # Match the reference controller: left X/Y bends/straightens the waist.
        # The unused left-stick horizontal axis retains full head-yaw control.
        waist_direction = int(bool(states["left"]["a_button"])) - int(
            bool(states["left"]["b_button"])
        )
        head_yaw_direction = float(states["left"]["thumbstick_y"])
        head_pitch_direction = int(bool(states["right"]["a_button"])) - int(
            bool(states["right"]["thumbstick"])
        )
        if waist_direction:
            index = self.waist_pitch_index
            self.joints.target[index] = self.joints.clamp(
                index,
                current_joints[index]
                + waist_direction * QUEST_WAIST_RATE * dt,
            )
            # Let unheld arm targets follow torso motion instead of causing IK
            # to counteract the requested bend.
            for side in self.SIDES:
                if float(states[side]["squeeze"]) <= 0.35:
                    self.target_positions[side] = current_eef_positions[side].copy()
                    self.target_orientations[side] = (
                        current_eef_orientations[side].copy()
                    )
                    self.controller_anchors[side] = None
                    self.controller_rotation_anchors[side] = None
                    self.robot_anchors[side] = None
                    self.robot_rotation_anchors[side] = None
        if abs(head_yaw_direction) > 0.12:
            index = self.head_yaw_index
            self.joints.target[index] = self.joints.clamp(
                index,
                current_joints[index]
                + head_yaw_direction * QUEST_HEAD_RATE * dt,
            )
        if head_pitch_direction:
            index = self.head_pitch_index
            self.joints.target[index] = self.joints.clamp(
                index,
                current_joints[index]
                + head_pitch_direction * QUEST_HEAD_RATE * dt,
            )

        cosine = math.cos(base.yaw)
        sine = math.sin(base.yaw)
        chassis_rotation = np.asarray(
            [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        controller_to_world = chassis_rotation @ self.webxr_to_robot_zero_yaw

        for side in self.SIDES:
            state = states[side]
            hand_pose = hand_poses[side]
            grip_pressed = float(state["squeeze"]) > 0.35
            pose_valid = bool(
                np.all(np.isfinite(hand_pose))
                and abs(float(np.linalg.det(hand_pose[:3, :3]))) > 0.1
            )
            if not grip_pressed:
                # Grip released: keep the last joint-space target.  The arm
                # then follows chassis/lift/waist motion naturally without an
                # uncommanded Cartesian IK update.
                self.awaiting_grip_release[side] = False
                self.arm_control_active[side] = False
                self.target_positions[side] = current_eef_positions[side].copy()
                self.target_orientations[side] = (
                    current_eef_orientations[side].copy()
                )
                self.controller_anchors[side] = None
                self.controller_rotation_anchors[side] = None
                self.robot_anchors[side] = None
                self.robot_rotation_anchors[side] = None
                if self.pinocchio_ik is not None:
                    self.pinocchio_ik.reset(side)
            elif self.awaiting_grip_release[side]:
                # Ignore a Grip that was already held while WebXR connected or
                # while B reset was performed.  A release and fresh press are
                # required before this arm can move.
                self.arm_control_active[side] = False
            elif pose_valid:
                controller_position = hand_pose[:3, 3].copy()
                if self.controller_anchors[side] is None:
                    self.controller_anchors[side] = controller_position
                    self.controller_rotation_anchors[side] = (
                        hand_pose[:3, :3].copy()
                    )
                    self.robot_anchors[side] = current_eef_positions[side].copy()
                    self.robot_rotation_anchors[side] = (
                        current_eef_orientations[side].copy()
                    )
                    self.target_positions[side] = current_eef_positions[side].copy()
                    self.target_orientations[side] = (
                        current_eef_orientations[side].copy()
                    )
                    if self.pinocchio_ik is not None:
                        self.pinocchio_ik.reset(side)
                    print(
                        f"[QUEST] {side} Grip anchored to the current W1 pose",
                        flush=True,
                    )
                self.arm_control_active[side] = True
                if self.controller_anchors[side] is not None:
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
                        hand_pose[:3, :3]
                        @ self.controller_rotation_anchors[side].T
                    )
                    world_rotation_delta = (
                        controller_to_world
                        @ controller_rotation_delta
                        @ controller_to_world.T
                    )
                    self.target_orientations[side] = (
                        world_rotation_delta @ self.robot_rotation_anchors[side]
                    )
            else:
                self.arm_control_active[side] = False
            self.grip_was_pressed[side] = grip_pressed

        jacobians = None
        if any(self.arm_control_active.values()):
            candidate_jacobians = np.asarray(
                self.robot.get_jacobians()[0], dtype=np.float64
            )
            if np.all(np.isfinite(candidate_jacobians)):
                jacobians = candidate_jacobians
        for side in self.SIDES:
            indices = self.arm_indices[side]
            if self.arm_control_active[side] and jacobians is not None:
                position_error = _clamp_norm(
                    self.target_positions[side] - current_eef_positions[side],
                    0.025,
                )
                rotation_error = _clamp_norm(
                    _rotation_vector(
                        self.target_orientations[side]
                        @ current_eef_orientations[side].T
                    ),
                    0.060,
                )
                position_jacobian = jacobians[
                    self.eef_jacobian_indices[side],
                    :3,
                ][:, self.arm_jacobian_indices[side]]
                rotation_jacobian = jacobians[
                    self.eef_jacobian_indices[side],
                    3:6,
                ][:, self.arm_jacobian_indices[side]]
                official_target = None
                if self.pinocchio_ik is not None:
                    official_target = self.pinocchio_ik.solve(
                        side,
                        current_joints[indices],
                        current_eef_orientations[side],
                        position_error,
                        rotation_error,
                        dt,
                    )
                    pin_arm = self.pinocchio_ik.arms[side]
                    if official_target is None and (
                        pin_arm.fallback_count == 1
                        or pin_arm.fallback_count % 600 == 0
                    ):
                        print(
                            "[IK] Pinocchio/Pink solution rejected; using legacy "
                            f"DLS for this frame: {pin_arm.last_failure}",
                            flush=True,
                        )
                elif self.realman_ik is not None:
                    official_target = self.realman_ik.solve(
                        current_joints[indices],
                        position_error,
                        rotation_error,
                        rotation_jacobian,
                    )
                    fallback_count = self.realman_ik.fallback_count
                    if official_target is None and (
                        fallback_count == 1 or fallback_count % 600 == 0
                    ):
                        print(
                            "[IK] RealMan solution rejected; using legacy DLS for "
                            f"this frame: {self.realman_ik.last_failure}",
                            flush=True,
                        )
                if official_target is not None:
                    delta = official_target - current_joints[indices]
                else:
                    # Preserved legacy solver: safe per-frame fallback whenever a
                    # constrained or vendor solver rejects the current target.
                    task_jacobian = np.vstack(
                        (
                            position_jacobian,
                            QUEST_IK_ROTATION_WEIGHT * rotation_jacobian,
                        )
                    )
                    task_error = np.concatenate(
                        (
                            position_error,
                            QUEST_IK_ROTATION_WEIGHT * rotation_error,
                        )
                    )
                    delta = _damped_least_squares(
                        task_jacobian,
                        task_error,
                        QUEST_IK_DAMPING,
                    )
                delta = np.clip(delta, -0.025, 0.025)
                candidate_target = np.clip(
                    current_joints[indices] + delta,
                    self.joints.limits[indices, 0],
                    self.joints.limits[indices, 1],
                )
                if np.all(np.isfinite(candidate_target)):
                    self.joints.target[indices] = candidate_target
            trigger = float(np.clip(states[side]["trigger"], 0.0, 1.0))
            gripper_index = self.gripper_indices[side]
            gripper_position = self.gripper_open_position + (
                self.gripper_closed_position - self.gripper_open_position
            ) * trigger
            # Match keyboard teleoperation's gradual joint motion.  Applying
            # the full 0..0.70 rad Trigger target in one physics frame makes
            # the Robotiq mimic links jump and can visibly destabilize them.
            current_target = float(self.joints.target[gripper_index])
            maximum_step = QUEST_GRIPPER_RATE * dt
            gripper_position = current_target + float(
                np.clip(
                    gripper_position - current_target,
                    -maximum_step,
                    maximum_step,
                )
            )
            self.joints.target[gripper_index] = self.joints.clamp(
                gripper_index,
                gripper_position,
            )

    def stream_camera(self) -> None:
        if self.bridge is None or self.quest_camera is None:
            return
        self.frame_index += 1
        if self.frame_index % self.stream_interval:
            return
        rgba = self.quest_camera.get_rgba()
        if rgba is None or not rgba.size:
            return
        self.bridge.write_mono_image(np.asarray(rgba)[..., :3])

    def status(self) -> str:
        if self.bridge is None:
            return "disabled"
        left_updates, right_updates = self.bridge.update_counts
        return (
            f"left_updates={left_updates}, right_updates={right_updates}, "
            f"grip=({self.last_grip_values['left']:.2f},"
            f"{self.last_grip_values['right']:.2f}), "
            "arm_clutch=("
            f"{'active' if self.arm_control_active['left'] else 'idle'},"
            f"{'active' if self.arm_control_active['right'] else 'idle'})"
        )

    def close(self) -> None:
        if self.bridge is not None:
            self.bridge.close()
            self.bridge = None


def _get_authored_robot_pose(stage: Usd.Stage) -> tuple[np.ndarray, np.ndarray]:
    """Read the W1 mounting pose before PhysX can move its floating root."""
    robot_prim = stage.GetPrimAtPath(ROBOT_PATH)
    if not robot_prim:
        raise RuntimeError(f"W1 robot root is missing: {ROBOT_PATH}")

    world_matrix = UsdGeom.XformCache().GetLocalToWorldTransform(robot_prim)
    position = np.asarray(world_matrix.ExtractTranslation(), dtype=np.float32)
    rotation = world_matrix.RemoveScaleShear().ExtractRotationQuat().GetNormalized()
    orientation = np.asarray(
        [rotation.GetReal(), *rotation.GetImaginary()],
        dtype=np.float32,
    )
    return position, orientation


def _configure_w1_joint_drives(stage: Usd.Stage) -> int:
    """Add damping floors and zero velocity targets in the task session layer."""
    damping_floor = {
        **{
            f"{side}_joint{joint}": damping
            for side in ("left", "right")
            for joint, damping in enumerate(
                (35.0, 28.0, 20.0, 8.0, 4.0, 1.5, 0.8),
                start=1,
            )
        },
        "left_gripper_joint": 1.0,
        "right_gripper_joint": 1.0,
        "head_yaw_joint": 2.5,
        "head_pitch_joint": 0.5,
        "waist_updown_joint": 50.0,
        "waist_pitch_joint": 20.0,
    }

    robot_root = stage.GetPrimAtPath(ROBOT_PATH)
    if not robot_root:
        raise RuntimeError(f"W1 robot root is missing: {ROBOT_PATH}")

    configured: set[str] = set()
    for prim in Usd.PrimRange(robot_root):
        name_override = prim.GetAttribute("isaac:nameOverride")
        name = (
            str(name_override.Get())
            if name_override and name_override.Get()
            else prim.GetName()
        )
        if name not in damping_floor:
            continue
        if prim.IsA(UsdPhysics.RevoluteJoint):
            drive_name = "angular"
        elif prim.IsA(UsdPhysics.PrismaticJoint):
            drive_name = "linear"
        else:
            raise RuntimeError(f"Expected a driven W1 joint at {prim.GetPath()}")

        drive = UsdPhysics.DriveAPI(prim, drive_name)
        damping_attr = drive.GetDampingAttr()
        if not damping_attr:
            damping_attr = drive.CreateDampingAttr()
        current_damping = damping_attr.Get()
        damping_attr.Set(
            max(
                float(current_damping) if current_damping is not None else 0.0,
                damping_floor[name],
            )
        )
        velocity_attr = drive.GetTargetVelocityAttr()
        if not velocity_attr:
            velocity_attr = drive.CreateTargetVelocityAttr()
        velocity_attr.Set(0.0)
        configured.add(name)

    missing = sorted(set(damping_floor) - configured)
    if missing:
        raise RuntimeError(f"W1 drive stabilization missed joints: {missing}")
    return len(configured)


def _enable_runtime_waist_pitch(stage: Usd.Stage) -> tuple[float, float]:
    """Restore the reference W1 bending range in this task session only."""
    prim = stage.GetPrimAtPath(WAIST_PITCH_JOINT_PATH)
    if not prim or not prim.IsA(UsdPhysics.RevoluteJoint):
        raise RuntimeError(f"W1 waist-pitch joint is missing: {WAIST_PITCH_JOINT_PATH}")
    joint = UsdPhysics.RevoluteJoint(prim)
    lower_degrees = 0.0
    upper_degrees = 89.95437622070312
    joint.CreateLowerLimitAttr(lower_degrees)
    joint.CreateUpperLimitAttr(upper_degrees)
    return lower_degrees, upper_degrees


def _set_asset_pose(
    stage: Usd.Stage,
    prim_path: str,
    translation: tuple[float, float, float],
    rotation_x_degrees: float | None = None,
    orientation_wxyz: tuple[float, float, float, float] | None = None,
    scale_xyz: tuple[float, float, float] | None = None,
) -> None:
    """Set a referenced asset's task-stage pose."""
    prim = stage.GetPrimAtPath(prim_path)
    if not prim:
        raise RuntimeError(f"Referenced asset root is missing: {prim_path}")

    xform = UsdGeom.Xformable(prim)
    xform.ClearXformOpOrder()
    xform.AddTranslateOp().Set(Gf.Vec3d(*translation))
    if orientation_wxyz is not None:
        w, x, y, z = orientation_wxyz
        xform.AddOrientOp().Set(Gf.Quatf(w, Gf.Vec3f(x, y, z)))
    elif rotation_x_degrees is not None:
        xform.AddRotateXOp().Set(float(rotation_x_degrees))
    else:
        raise ValueError("Asset pose requires an orientation")
    if scale_xyz is not None:
        xform.AddScaleOp().Set(Gf.Vec3f(*scale_xyz))


def _add_usd_reference(stage: Usd.Stage, asset_path: Path, prim_path: str) -> None:
    """Reference a local metre-scale USD without the legacy metrics assembler."""
    prim = stage.DefinePrim(prim_path, "Xform")
    if not prim.GetReferences().AddReference(str(asset_path)):
        raise RuntimeError(f"Failed to reference {asset_path} at {prim_path}")


def _enable_static_scene_collision(stage: Usd.Stage) -> int:
    """Use the reconstructed environment mesh as a static triangle collider."""
    root = stage.GetPrimAtPath(ENVIRONMENT_COLLISION_PATH)
    if not root:
        raise RuntimeError(f"Environment collision root is missing: {ENVIRONMENT_COLLISION_PATH}")

    # The mesh proxy is physics-only.  The colored NuRec volume is the visible
    # representation of the environment.
    UsdGeom.Imageable(root).CreateVisibilityAttr().Set(UsdGeom.Tokens.invisible)

    collider_count = 0
    for prim in Usd.PrimRange(root):
        if not prim.IsA(UsdGeom.Mesh):
            continue
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            UsdPhysics.CollisionAPI.Apply(prim)
        if not prim.HasAPI(UsdPhysics.MeshCollisionAPI):
            mesh_collision = UsdPhysics.MeshCollisionAPI.Apply(prim)
        else:
            mesh_collision = UsdPhysics.MeshCollisionAPI(prim)
        mesh_collision.CreateApproximationAttr().Set("meshSimplification")
        collider_count += 1
    return collider_count


def _count_nurec_volumes(stage: Usd.Stage) -> int:
    """Count referenced NuRec volume prims used for colored scene rendering."""
    root = stage.GetPrimAtPath(ENVIRONMENT_VISUAL_PATH)
    if not root:
        raise RuntimeError(f"Environment visual root is missing: {ENVIRONMENT_VISUAL_PATH}")
    return sum(
        1
        for prim in Usd.PrimRange(root)
        if prim.GetAttribute("omni:nurec:isNuRecVolume").Get() is True
    )


def _add_fallback_ground(stage: Usd.Stage) -> None:
    """Add an invisible floor at Z=0 so the task object cannot fall away."""
    ground = UsdGeom.Cube.Define(stage, "/World/FallbackGround")
    ground.CreateSizeAttr(1.0)
    ground.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, -0.05))
    ground.AddScaleOp().Set(Gf.Vec3f(100.0, 100.0, 0.10))
    ground.CreateVisibilityAttr().Set(UsdGeom.Tokens.invisible)
    UsdPhysics.CollisionAPI.Apply(ground.GetPrim()).CreateCollisionEnabledAttr(True)


def _add_lighting(stage: Usd.Stage) -> None:
    dome = UsdLux.DomeLight.Define(stage, "/World/Lights/Dome")
    dome.CreateIntensityAttr(300.0)
    dome.CreateColorAttr(Gf.Vec3f(0.78, 0.82, 0.92))

    sun = UsdLux.DistantLight.Define(stage, "/World/Lights/Sun")
    sun.CreateIntensityAttr(900.0)
    sun.CreateAngleAttr(1.0)
    sun.AddRotateXYZOp().Set(Gf.Vec3f(48.0, -35.0, -42.0))


def _ensure_wrist_cameras(stage: Usd.Stage) -> None:
    """Add non-physical wrist cameras to the task session layer."""
    camera_specs = {
        LEFT_WRIST_CAMERA_PATH: (
            Gf.Vec3d(0.0, 0.07881743898056037, 0.058011117340431184),
            Gf.Vec3f(88.4, -0.3, 0.0),
            "left_wrist",
        ),
        RIGHT_WRIST_CAMERA_PATH: (
            Gf.Vec3d(0.0, -0.07881743898056037, 0.058011117340431184),
            Gf.Vec3f(88.4, 0.3, 180.0),
            "right_wrist",
        ),
    }
    for camera_path, (translation, rotation, role) in camera_specs.items():
        path = Sdf.Path(camera_path)
        sensors_path = path.GetParentPath()
        link_path = sensors_path.GetParentPath()
        link_prim = stage.GetPrimAtPath(link_path)
        if not link_prim:
            raise RuntimeError(f"Wrist camera parent link is missing: {link_path}")
        # W1 link prims may arrive through instanceable references.  De-instance
        # only these two link7 parents so the session layer can add cameras.
        stage.OverridePrim(link_path).SetInstanceable(False)
        UsdGeom.Xform.Define(stage, sensors_path)
        camera = UsdGeom.Camera.Define(stage, path)
        camera.CreateFocalLengthAttr(18.0)
        camera.CreateHorizontalApertureAttr(20.955)
        camera.CreateVerticalApertureAttr(11.7871875)
        camera.CreateClippingRangeAttr(Gf.Vec2f(0.03, 30.0))
        camera.CreateFocusDistanceAttr(2.0)
        camera.CreateFStopAttr(0.0)
        camera.GetPrim().CreateAttribute(
            "w1:cameraRole",
            Sdf.ValueTypeNames.String,
            custom=True,
        ).Set(role)
        xform = UsdGeom.Xformable(camera)
        xform.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(translation)
        xform.AddRotateXYZOp(UsdGeom.XformOp.PrecisionFloat).Set(rotation)


def _open_robot_camera_viewports(stage: Usd.Stage) -> list:
    """Open head and dual-wrist live views on every interactive launch."""
    if HEADLESS or SMOKE_TEST or not SHOW_CAMERA_VIEWPORTS:
        return []
    titles = {
        "head": "W1 Head Camera",
        "left_wrist": "W1 Left Wrist Camera",
        "right_wrist": "W1 Right Wrist Camera",
    }
    windows = []
    for index, (role, camera_path) in enumerate(DISPLAY_CAMERA_PATHS.items()):
        if not stage.GetPrimAtPath(camera_path).IsValid():
            raise RuntimeError(f"Missing {role} display camera: {camera_path}")
        windows.append(
            create_viewport_window(
                name=titles[role],
                width=470,
                height=330,
                position_x=10 + index * 475,
                position_y=80,
                camera_path=Sdf.Path(camera_path),
            )
        )
    print("[VIEW] Head, left-wrist, and right-wrist camera windows opened.", flush=True)
    return windows


def build_task_world() -> tuple[
    World,
    Usd.Stage,
    Articulation,
    RigidPrim,
    RigidPrim,
    RigidPrim,
    np.ndarray,
    np.ndarray,
    int,
    int,
    int,
]:
    """Compose the environment, object, and W1 robot into one world."""
    World.clear_instance()
    create_new_stage()
    simulation_app.update()

    world = World(
        stage_units_in_meters=1.0,
        physics_dt=1.0 / 120.0,
        rendering_dt=1.0 / 60.0,
    )

    stage = omni.usd.get_context().get_stage()
    _add_usd_reference(stage, ENVIRONMENT_VISUAL_ASSET, ENVIRONMENT_VISUAL_PATH)
    _add_usd_reference(stage, ENVIRONMENT_COLLISION_ASSET, ENVIRONMENT_COLLISION_PATH)
    _add_usd_reference(stage, TASK_OBJECT_ASSET, TASK_OBJECT_PATH)
    _add_usd_reference(stage, ROBOT_ASSET, COMPOSITE_PATH)
    wait_for_stage(simulation_app)

    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    _set_asset_pose(
        stage,
        ENVIRONMENT_VISUAL_PATH,
        ENVIRONMENT_SPAWN_XYZ_M,
        REALSCENE_ROTATE_X_DEG,
    )
    _set_asset_pose(
        stage,
        ENVIRONMENT_COLLISION_PATH,
        ENVIRONMENT_SPAWN_XYZ_M,
        REALSCENE_ROTATE_X_DEG,
    )
    _set_asset_pose(
        stage,
        TASK_OBJECT_PATH,
        OBJECT_SPAWN_XYZ_M,
        orientation_wxyz=OBJECT_SPAWN_ORIENTATION_WXYZ,
        scale_xyz=OBJECT_SCALE_XYZ,
    )

    nurec_volume_count = _count_nurec_volumes(stage)
    if nurec_volume_count == 0:
        raise RuntimeError("Environment visual asset did not provide a NuRec volume")
    scene_collider_count = 0
    if ENABLE_SCENE_COLLISION:
        scene_collider_count = _enable_static_scene_collision(stage)
    _add_fallback_ground(stage)
    _add_lighting(stage)
    _ensure_wrist_cameras(stage)
    make_w1_mobile(stage)

    # make_w1_mobile releases legacy world-root joints. This asset repurposes
    # root_joint as an external LiftCarriage -> W1 mount, so keep it active.
    mount_joint_prim = stage.GetPrimAtPath(ROOT_JOINT_PATH)
    if not mount_joint_prim or not mount_joint_prim.IsA(UsdPhysics.FixedJoint):
        raise RuntimeError(f"Asset fixed mount is missing: {ROOT_JOINT_PATH}")
    mount_joint_prim.SetActive(True)

    vehicle_prim = stage.GetPrimAtPath(VEHICLE_PATH)
    if not vehicle_prim or not vehicle_prim.HasAPI(UsdPhysics.RigidBodyAPI):
        raise RuntimeError(f"Asset chassis rigid body is missing: {VEHICLE_PATH}")
    lift_carriage_prim = stage.GetPrimAtPath(LIFT_CARRIAGE_PATH)
    if not lift_carriage_prim or not lift_carriage_prim.HasAPI(
        UsdPhysics.RigidBodyAPI
    ):
        raise RuntimeError(f"Asset lift carriage is missing: {LIFT_CARRIAGE_PATH}")
    mount_joint = UsdPhysics.FixedJoint(mount_joint_prim)
    if mount_joint.GetBody0Rel().GetTargets() != [lift_carriage_prim.GetPath()]:
        raise RuntimeError("Asset fixed mount body0 must target LiftCarriage")
    UsdPhysics.RigidBodyAPI(vehicle_prim).CreateKinematicEnabledAttr(True)
    UsdPhysics.RigidBodyAPI(lift_carriage_prim).CreateKinematicEnabledAttr(True)

    # The articulation root stays dynamic and is constrained by the asset joint.
    robot_base_prim = stage.GetPrimAtPath(f"{ROBOT_PATH}/Geometry/base_link")
    if not robot_base_prim:
        raise RuntimeError("W1 base_link is missing")
    if mount_joint.GetBody1Rel().GetTargets() != [robot_base_prim.GetPath()]:
        raise RuntimeError("Asset fixed mount body1 must target W1 base_link")
    if mount_joint.GetExcludeFromArticulationAttr().Get() is not True:
        raise RuntimeError("Asset fixed mount must be excluded from articulation")
    authored_robot_position, authored_robot_orientation = _get_authored_robot_pose(stage)
    _enable_runtime_waist_pitch(stage)
    stabilized_joint_drive_count = _configure_w1_joint_drives(stage)
    UsdPhysics.RigidBodyAPI(robot_base_prim).CreateKinematicEnabledAttr(False)

    object_prim = stage.GetPrimAtPath(TASK_OBJECT_PATH)
    if not object_prim.HasAPI(UsdPhysics.RigidBodyAPI):
        raise RuntimeError("The configured task object did not load as a rigid body")

    robot = world.scene.add(
        Articulation(
            prim_paths_expr=ROBOT_PATH,
            name="w1_task_robot",
            reset_xform_properties=False,
        )
    )
    world.reset()

    # Do not register kinematic RigidPrim views with Scene before reset.
    # RigidPrim._on_post_reset writes default linear/angular velocities, which
    # PhysX rejects for kinematic actors.  The bodies are already present in
    # the physics scene; post-reset views provide pose control without that
    # invalid automatic velocity write.
    vehicle_body = RigidPrim(
        prim_paths_expr=VEHICLE_PATH,
        name="w1_task_tracked_chassis",
        reset_xform_properties=False,
    )
    lift_carriage = RigidPrim(
        prim_paths_expr=LIFT_CARRIAGE_PATH,
        name="w1_task_lift_carriage",
        reset_xform_properties=False,
    )
    task_object_body = RigidPrim(
        prim_paths_expr=TASK_OBJECT_PATH,
        name="w1_task_object",
        reset_xform_properties=False,
    )
    vehicle_body.initialize()
    lift_carriage.initialize()
    task_object_body.initialize()

    if robot.num_dof != 30:
        raise RuntimeError(f"Expected 30 W1 DOFs, found {robot.num_dof}")
    if not vehicle_body.is_physics_handle_valid():
        raise RuntimeError("Asset chassis rigid body did not initialize")
    if not lift_carriage.is_physics_handle_valid():
        raise RuntimeError("Asset lift carriage rigid body did not initialize")
    if not task_object_body.is_physics_handle_valid():
        raise RuntimeError("Task object rigid body did not initialize")
    return (
        world,
        stage,
        robot,
        vehicle_body,
        lift_carriage,
        task_object_body,
        authored_robot_position,
        authored_robot_orientation,
        scene_collider_count,
        nurec_volume_count,
        stabilized_joint_drive_count,
    )


def _read_active_view_camera(viewport) -> tuple[np.ndarray, np.ndarray]:
    """Read the active viewport camera's world-space eye and look-at target."""
    if viewport is None:
        raise RuntimeError("No active viewport is available for camera saving")
    camera_path = str(viewport.get_active_camera())
    camera_state = ViewportCameraState(camera_path, viewport)
    eye = np.asarray(tuple(camera_state.position_world), dtype=np.float64)
    target = np.asarray(tuple(camera_state.target_world), dtype=np.float64)
    if eye.shape != (3,) or target.shape != (3,):
        raise RuntimeError(
            f"Invalid viewport camera vectors: eye={eye.shape}, target={target.shape}"
        )
    if not (np.all(np.isfinite(eye)) and np.all(np.isfinite(target))):
        raise RuntimeError("Viewport camera pose contains non-finite values")
    if float(np.linalg.norm(target - eye)) < 1.0e-6:
        raise RuntimeError("Viewport camera eye and target are coincident")
    return eye, target


def _restore_saved_joint_state(
    joints: StableJointCommander,
    placement_state: dict,
) -> int:
    """Set the task's initial/reset joint pose from placement configuration."""
    # The Robotiq 2F-140 is fully open at zero radians.
    for name in ("left_gripper_joint", "right_gripper_joint"):
        index = joints.index[name]
        joints.target[index] = joints.clamp(index, GRIPPER_OPEN_POSITION)

    targets = placement_state.get("robot_joint_state", {}).get(
        "position_targets",
        {},
    )
    command_indices = set(map(int, joints.command_indices))
    # A task started before joint persistence was added may still have its old
    # P-key handler in memory.  If that live process rewrites placement.json,
    # retain this captured pose as a safe one-time compatibility fallback.
    if not targets and RUNTIME_STATE_CAPTURE_PATH.is_file():
        capture = json.loads(
            RUNTIME_STATE_CAPTURE_PATH.read_text(encoding="utf-8")
        )
        names = capture.get("joint_names", [])
        values = capture.get("joint_targets", [])
        if not isinstance(names, list) or not isinstance(values, list):
            raise RuntimeError(
                f"Invalid runtime joint capture: {RUNTIME_STATE_CAPTURE_PATH}"
            )
        if len(names) != len(values):
            raise RuntimeError(
                f"Mismatched runtime joint capture: {RUNTIME_STATE_CAPTURE_PATH}"
            )
        targets = {
            name: value
            for name, value in zip(names, values)
            if name in joints.index and joints.index[name] in command_indices
        }
    restored = 0
    for name, value in targets.items():
        if name not in joints.index:
            raise RuntimeError(f"Saved W1 joint does not exist: {name}")
        index = joints.index[name]
        if index not in command_indices:
            raise RuntimeError(f"Saved W1 joint is not commandable: {name}")
        joints.target[index] = joints.clamp(index, float(value))
        restored += 1
    joints.target[joints.locked_indices] = 0.0
    joints.snap_to_targets()
    joints.capture_home()
    return restored


def _save_current_placement(
    placement_state: dict,
    base: StableMobileAssembly,
    joints: StableJointCommander,
    task_object_body: RigidPrim,
    viewport,
) -> None:
    positions, orientations = task_object_body.get_world_poses()
    if positions is None or orientations is None:
        raise RuntimeError("Could not read the current task-object pose")
    object_position = np.asarray(positions[0], dtype=float)
    object_orientation = np.asarray(orientations[0], dtype=float)
    if not (
        np.all(np.isfinite(object_position))
        and np.all(np.isfinite(object_orientation))
    ):
        raise RuntimeError("Task-object pose contains non-finite values")
    camera_eye, camera_target = _read_active_view_camera(viewport)

    placement_state["robot"] = {
        "x": float(base.x),
        "y": float(base.y),
        "yaw_radians": float(base.yaw),
        "lift_height_m": float(base.lift_height),
    }
    placement_state["task_object"] = {
        "position_xyz_m": object_position.tolist(),
        "orientation_wxyz": object_orientation.tolist(),
        "scale_xyz": list(map(float, placement_state["task_object"]["scale_xyz"])),
    }
    placement_state["camera"] = {
        "eye_xyz_m": camera_eye.tolist(),
        "target_xyz_m": camera_target.tolist(),
    }
    placement_state["robot_joint_state"] = {
        "position_targets": {
            joints.names[int(index)]: float(joints.target[int(index)])
            for index in joints.command_indices
        }
    }
    placement_state["updated_at"] = datetime.now().astimezone().isoformat()
    temporary_path = PLACEMENT_PATH.with_suffix(".json.tmp")
    temporary_path.write_text(
        json.dumps(placement_state, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(PLACEMENT_PATH)
    joints.capture_home()
    print(f"Saved persistent placement for {TASK_NAME}: {PLACEMENT_PATH}", flush=True)
    print(
        "  robot (x, y, yaw, lift): "
        f"({base.x:.6f}, {base.y:.6f}, {base.yaw:.6f}, "
        f"{base.lift_height:.6f})",
        flush=True,
    )
    print(
        "  task object position: "
        f"{object_position.tolist()}",
        flush=True,
    )
    print(
        "  viewport camera eye/target: "
        f"{camera_eye.tolist()} -> {camera_target.tolist()}",
        flush=True,
    )
    print(
        f"  robot joint targets: {len(joints.command_indices)} saved",
        flush=True,
    )


def main() -> None:
    (
        world,
        stage,
        robot,
        vehicle_body,
        lift_carriage,
        task_object_body,
        authored_robot_position,
        authored_robot_orientation,
        scene_collider_count,
        nurec_volume_count,
        stabilized_joint_drive_count,
    ) = build_task_world()
    # Drive the two persistent kinematic bodies authored inside the robot asset.
    base = StableMobileAssembly(
        stage,
        robot,
        vehicle_body,
        lift_carriage,
    )
    base.initial_robot_position = authored_robot_position.copy()
    base.initial_robot_orientation = authored_robot_orientation.copy()
    base.lift_height = float(
        np.clip(
            ROBOT_SPAWN_LIFT_HEIGHT_M,
            base.lift_lower_limit,
            base.lift_upper_limit,
        )
    )
    base.set_pose(*ROBOT_SPAWN_XY_YAW)
    joints = StableJointCommander(robot)
    placement_state = json.loads(json.dumps(PLACEMENT))
    restored_joint_count = _restore_saved_joint_state(joints, placement_state)

    # Warm up PhysX while continuously enforcing the asset mount joint.  Do
    # not render an intermediate frame in which the dynamic root has drifted.
    joints.apply()
    for _ in range(24):
        base.apply()
        joints.stop_motion()
        world.step(render=False)
        base.apply()
        joints.stop_motion()

    quest = QuestUpperBodyController(robot, joints)
    viewport = None
    if not HEADLESS:
        viewport = get_active_viewport()
        viewport.set_texture_resolution((1280, 720))
        set_camera_view(
            eye=list(CAMERA_EYE_XYZ_M),
            target=list(CAMERA_TARGET_XYZ_M),
            camera_prim_path="/OmniverseKit_Persp",
        )
    quest.start()
    camera_viewports = _open_robot_camera_viewports(stage)

    print(HELP, flush=True)
    print(f"\nLoaded {TASK_NAME} assets:", flush=True)
    print(f"  visual environment:    {ENVIRONMENT_VISUAL_ASSET}", flush=True)
    print(f"  collision environment: {ENVIRONMENT_COLLISION_ASSET}", flush=True)
    print(f"  task object: {TASK_OBJECT_ASSET}", flush=True)
    print(f"  robot:       {ROBOT_ASSET}", flush=True)
    print(f"  NuRec visual volumes: {nurec_volume_count}", flush=True)
    print(f"  scene colliders: {scene_collider_count}", flush=True)
    print(f"  stabilized W1 joint drives: {stabilized_joint_drive_count}", flush=True)
    print(f"  asset chassis body: {VEHICLE_PATH}", flush=True)
    print(f"  asset lift carriage: {LIFT_CARRIAGE_PATH}", flush=True)
    print(f"  asset mount joint: fixed_joint {ROOT_JOINT_PATH}", flush=True)
    print(
        "  lift range/speed: "
        f"{base.lift_lower_limit:.2f}..{base.lift_upper_limit:.2f} m / "
        f"{base.lift_max_speed:.2f} m/s",
        flush=True,
    )
    print(
        "  W1 joint control mode: "
        + (
            "hybrid_pd_arms_strict_head_waist"
            if STRICT_JOINT_HOLD and QUEST_IK_BACKEND == "pinocchio"
            else "strict_position_hold"
            if STRICT_JOINT_HOLD
            else "dynamic_pd"
        ),
        flush=True,
    )
    print(
        "  authored W1 mount pose: "
        f"position={authored_robot_position.tolist()}, "
        f"orientation_wxyz={authored_robot_orientation.tolist()}",
        flush=True,
    )
    print(f"  robot spawn (x, y, yaw): {ROBOT_SPAWN_XY_YAW}", flush=True)
    print(f"  robot spawn lift height: {ROBOT_SPAWN_LIFT_HEIGHT_M}", flush=True)
    print(f"  object spawn (x, y, z):  {OBJECT_SPAWN_XYZ_M}", flush=True)
    print(f"  object uniform scale:    {OBJECT_SCALE_XYZ}", flush=True)
    print(f"  persistent placement:    {PLACEMENT_PATH}", flush=True)
    print(f"  restored joint targets: {restored_joint_count}", flush=True)
    print(
        "  upper-body control:      "
        + (
            "keyboard"
            if KEYBOARD_UPPER_BODY
            else (
                "Quest 3S WebXR"
                if quest.bridge is not None
                else "disabled for this run"
            )
        ),
        flush=True,
    )
    print(
        f"  robot camera viewports: {len(camera_viewports)}",
        flush=True,
    )
    if not HEADLESS:
        print("\nReady. Click the viewport once, then use the keys above.", flush=True)

    held: set[carb.input.KeyboardInput] = set()
    state = {"quit": False, "joint": 1}
    input_iface = None
    keyboard = None
    subscription = None

    if not HEADLESS and not SMOKE_TEST:
        input_iface = carb.input.acquire_input_interface()
        keyboard = omni.appwindow.get_default_app_window().get_keyboard()
        joint_keys = {
            carb.input.KeyboardInput.KEY_1: 1,
            carb.input.KeyboardInput.KEY_2: 2,
            carb.input.KeyboardInput.KEY_3: 3,
            carb.input.KeyboardInput.KEY_4: 4,
            carb.input.KeyboardInput.KEY_5: 5,
            carb.input.KeyboardInput.KEY_6: 6,
            carb.input.KeyboardInput.KEY_7: 7,
        }

        def on_key(event, *_args):
            key = event.input
            if event.type in (
                carb.input.KeyboardEventType.KEY_PRESS,
                carb.input.KeyboardEventType.KEY_REPEAT,
            ):
                held.add(key)
                if KEYBOARD_UPPER_BODY and key in joint_keys:
                    state["joint"] = joint_keys[key]
                    print(f"Selected arm joint {state['joint']}", flush=True)
                elif key == carb.input.KeyboardInput.R:
                    saved_robot = placement_state["robot"]
                    base.lift_height = float(
                        np.clip(
                            saved_robot["lift_height_m"],
                            base.lift_lower_limit,
                            base.lift_upper_limit,
                        )
                    )
                    base.set_pose(
                        saved_robot["x"],
                        saved_robot["y"],
                        saved_robot["yaw_radians"],
                    )
                    base.reset_tracks()
                    quest.reset()
                    print("Robot pose, lift height, and joint targets reset", flush=True)
                elif (
                    key == carb.input.KeyboardInput.P
                    and event.type == carb.input.KeyboardEventType.KEY_PRESS
                ):
                    _save_current_placement(
                        placement_state,
                        base,
                        joints,
                        task_object_body,
                        viewport,
                    )
                elif key == carb.input.KeyboardInput.ESCAPE:
                    state["quit"] = True
            elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
                held.discard(key)
            return True

        subscription = input_iface.subscribe_to_keyboard_events(keyboard, on_key)

    dt = 1.0 / 60.0
    track_speed = base.track_drive.max_track_speed
    lift_speed = base.lift_max_speed
    frames = 0
    smoke_initial_lift_height = base.lift_height
    smoke_initial_waist_pitch = float(
        joints.target[joints.waist_pitch_index]
    )
    smoke_lift_direction = (
        1.0 if base.lift_upper_limit - base.lift_height >= 0.08 else -1.0
    )

    try:
        world.play()
        while simulation_app.is_running() and not state["quit"]:
            if SMOKE_TEST:
                left_track = 0.0
                right_track = 0.0
                if frames < 20:
                    left_track = 0.25
                    right_track = 0.35
                    joints.move_head("yaw", 0.15 * dt)
                    waist_index = joints.waist_pitch_index
                    joints.target[waist_index] = joints.clamp(
                        waist_index,
                        joints.target[waist_index] + 0.20 * dt,
                    )
                    base.move_lift(
                        smoke_lift_direction * base.lift_max_speed * dt
                    )
            else:
                throttle = float(
                    (carb.input.KeyboardInput.W in held)
                    - (carb.input.KeyboardInput.S in held)
                )
                steering = float(
                    (carb.input.KeyboardInput.A in held)
                    - (carb.input.KeyboardInput.D in held)
                )
                left_axis = throttle - steering
                right_axis = throttle + steering
                normalization = max(1.0, abs(left_axis), abs(right_axis))
                left_track = track_speed * left_axis / normalization
                right_track = track_speed * right_axis / normalization
                base.move_lift(
                    lift_speed
                    * dt
                    * (
                        (carb.input.KeyboardInput.T in held)
                        - (carb.input.KeyboardInput.Y in held)
                    )
                )

                if KEYBOARD_UPPER_BODY:
                    joint = state["joint"]
                    joints.move_arm_joint(
                        "left",
                        joint,
                        KEYBOARD_ARM_RATE
                        * dt
                        * (
                            (carb.input.KeyboardInput.X in held)
                            - (carb.input.KeyboardInput.Z in held)
                        ),
                    )
                    joints.move_arm_joint(
                        "right",
                        joint,
                        KEYBOARD_ARM_RATE
                        * dt
                        * (
                            (carb.input.KeyboardInput.V in held)
                            - (carb.input.KeyboardInput.C in held)
                        ),
                    )
                    joints.move_gripper(
                        "left",
                        KEYBOARD_GRIPPER_RATE
                        * dt
                        * (
                            (carb.input.KeyboardInput.N in held)
                            - (carb.input.KeyboardInput.M in held)
                            + (carb.input.KeyboardInput.G in held)
                            - (carb.input.KeyboardInput.H in held)
                        ),
                    )
                    joints.move_gripper(
                        "right",
                        KEYBOARD_GRIPPER_RATE
                        * dt
                        * (
                            (carb.input.KeyboardInput.COMMA in held)
                            - (carb.input.KeyboardInput.PERIOD in held)
                            + (carb.input.KeyboardInput.G in held)
                            - (carb.input.KeyboardInput.H in held)
                        ),
                    )
                    joints.move_head(
                        "yaw",
                        KEYBOARD_HEAD_RATE
                        * dt
                        * (
                            (carb.input.KeyboardInput.J in held)
                            - (carb.input.KeyboardInput.L in held)
                        ),
                    )
                    joints.move_head(
                        "pitch",
                        KEYBOARD_HEAD_RATE
                        * dt
                        * (
                            (carb.input.KeyboardInput.I in held)
                            - (carb.input.KeyboardInput.K in held)
                        ),
                    )
                    waist_index = joints.waist_pitch_index
                    joints.target[waist_index] = joints.clamp(
                        waist_index,
                        joints.target[waist_index]
                        + KEYBOARD_WAIST_RATE
                        * dt
                        * (
                            (carb.input.KeyboardInput.U in held)
                            - (carb.input.KeyboardInput.O in held)
                        ),
                    )
                else:
                    quest.update(base, task_object_body, placement_state, dt)

            base.integrate_tracks(left_track, right_track, dt)
            joints.apply()
            # Let the asset-authored fixed joint carry W1 before rendering.
            world.step(render=False)
            base.apply()
            joints.apply()
            if not HEADLESS:
                world.render()
                quest.stream_camera()
            frames += 1

            if quest.bridge is not None and frames % 600 == 0:
                print(f"[QUEST] {quest.status()}", flush=True)

            if SMOKE_TEST and frames >= 30:
                break
            if MAX_FRAMES > 0 and frames >= MAX_FRAMES:
                break

        if SMOKE_TEST:
            robot_position = np.asarray(robot.get_world_poses()[0][0], dtype=float)
            expected_position, _ = base._target_robot_pose()
            mount_position_error = float(
                np.linalg.norm(robot_position - expected_position.astype(float))
            )
            support_z = float(
                UsdGeom.XformCache()
                .GetLocalToWorldTransform(base.support_prim)
                .ExtractTranslation()[2]
            )
            support_lift_delta = support_z - base.support_zero_height_z
            fixed_carriage_z = float(
                UsdGeom.XformCache()
                .GetLocalToWorldTransform(base.fixed_carriage_prim)
                .ExtractTranslation()[2]
            )
            fixed_carriage_vertical_delta = (
                fixed_carriage_z - base.fixed_carriage_zero_height_z
            )
            lift_motion = base.lift_height - smoke_initial_lift_height
            final_joint_positions = np.asarray(
                robot.get_joint_positions()[0],
                dtype=float,
            )
            waist_pitch_motion = float(
                final_joint_positions[joints.waist_pitch_index]
                - smoke_initial_waist_pitch
            )
            planar_displacement = float(
                np.hypot(
                    base.x - ROBOT_SPAWN_XY_YAW[0],
                    base.y - ROBOT_SPAWN_XY_YAW[1],
                )
            )
            print(
                f"{TASK_NAME} smoke result: "
                f"lift_height={base.lift_height:.6f} m, "
                f"lift_motion={lift_motion:.6f} m, "
                f"waist_pitch_motion={waist_pitch_motion:.6f} rad, "
                f"planar_displacement={planar_displacement:.6f} m, "
                f"support_lift_delta={support_lift_delta:.6f} m, "
                "fixed_carriage_vertical_delta="
                f"{fixed_carriage_vertical_delta:.6f} m, "
                f"w1_mount_position_error={mount_position_error:.6f} m",
                flush=True,
            )
            if abs(lift_motion) < 0.07:
                raise RuntimeError("Smoke test lift did not move")
            if abs(waist_pitch_motion) < 0.03:
                raise RuntimeError("Smoke test waist pitch did not move")
            if planar_displacement < 0.05:
                raise RuntimeError("Smoke test tracked base did not move")
            if mount_position_error > 0.03:
                raise RuntimeError("W1 separated from LiftCarriage during smoke test")
            if abs(support_lift_delta - base.lift_height) > 1e-4:
                raise RuntimeError("Support trusses did not follow LiftCarriage")
            if abs(fixed_carriage_vertical_delta) > 1e-4:
                raise RuntimeError("Fixed CarriageVisual moved vertically")
    finally:
        quest.close()
        if subscription is not None:
            input_iface.unsubscribe_to_keyboard_events(keyboard, subscription)
        print(f"{TASK_NAME} stopped after {frames} control frames.", flush=True)
        world.stop()


try:
    main()
except BaseException:
    # SimulationApp.close() may terminate Kit before Python's default exception
    # hook runs, so report task failures explicitly while stderr is available.
    traceback.print_exc()
    raise
finally:
    simulation_app.close()
