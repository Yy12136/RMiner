#!/usr/bin/env python3
"""X7S + Quest 3S controller for the reconstructed drilling task.

The X7S robot, initial joint configuration, VR pose mapping, input mapping,
end-effector smoothing, and bimanual IK contract follow LW-BenchHub's X7S-Abs
teleoperation configuration.  The existing task WebXR bridge stays in place so
the Quest certificate URL, Vuer page, MotionControllers node, and port remain
unchanged.
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
    raise RuntimeError("W1_TASK_ROOT is required; launch through run_task.sh")
TASK_ROOT = Path(TASK_ROOT_ENV).resolve()
CONFIG_PATH = TASK_ROOT / "config" / "x7s_placement.json"
PLACEMENT = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))

ENVIRONMENT_VISUAL_ASSET = (
    TASK_ROOT
    / "assets/environment/RealScene04/real2sim_intermediate/RealScene04.usdz"
)
ENVIRONMENT_COLLISION_ASSET = (
    TASK_ROOT
    / "assets/environment/RealScene04/real2sim_intermediate/RealScene04_mesh.usd"
)
TASK_OBJECT_ASSET = TASK_ROOT / "assets/objects/gasoline_powered_post_driver.usd"
X7S_ROOT = TASK_ROOT / "assets/robot/X7S"
ROBOT_ASSET = X7S_ROOT / "x7s.usd"
X7S_URDF = X7S_ROOT / "urdf/X7S.urdf"
for path in (
    ENVIRONMENT_VISUAL_ASSET,
    ENVIRONMENT_COLLISION_ASSET,
    TASK_OBJECT_ASSET,
    ROBOT_ASSET,
    X7S_URDF,
):
    if not path.is_file():
        raise FileNotFoundError(path)

HEADLESS = os.environ.get("W1_DRILLING_HEADLESS", "0") == "1"
SMOKE_TEST = os.environ.get("W1_DRILLING_SMOKE_TEST", "0") == "1"
SMOKE_TEST_FRAMES = int(os.environ.get("X7S_SMOKE_TEST_FRAMES", "90"))
SMOKE_BASE_MOTION = os.environ.get("X7S_SMOKE_BASE_MOTION", "1") != "0"
SMOKE_BODY_MOTION = os.environ.get("X7S_SMOKE_BODY_MOTION", "1") != "0"
MAX_FRAMES = int(os.environ.get("W1_DRILLING_MAX_FRAMES", "0"))
QUEST_ENABLED = os.environ.get("W1_QUEST_ENABLED", "1") != "0"
QUEST_PORT = int(os.environ.get("W1_QUEST_PORT", "8012"))
QUEST_STREAM_FPS = int(os.environ.get("W1_QUEST_STREAM_FPS", "20"))
SHOW_CAMERA_VIEWPORTS = os.environ.get("W1_SHOW_CAMERA_VIEWPORTS", "1") != "0"
ENABLE_SCENE_COLLISION = os.environ.get("W1_DRILLING_SCENE_COLLISION", "1") != "0"
VALIDATE_KINEMATICS = os.environ.get("X7S_VALIDATE_KINEMATICS", "0") == "1"
ARM_STIFFNESS = float(os.environ.get("X7S_ARM_STIFFNESS", "400"))
ARM_DAMPING = float(os.environ.get("X7S_ARM_DAMPING", "80"))
PHYSICS_SUBSTEPS = 1

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
from isaacsim.core.utils.stage import create_new_stage, is_stage_loading
from isaacsim.core.utils.viewports import set_camera_view
from isaacsim.sensors.camera import Camera
from omni.kit.viewport.utility import create_viewport_window, get_active_viewport
from omni.kit.viewport.utility.camera_state import ViewportCameraState
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics
from scipy.spatial.transform import Rotation

from w1_quest3s import QuestBridge
from x7s_bimanual_ik import create_x7s_ik


ENVIRONMENT_VISUAL_PATH = "/World/EnvironmentVisual"
ENVIRONMENT_COLLISION_PATH = "/World/EnvironmentCollision"
TASK_OBJECT_PATH = "/World/TaskObject"
ROBOT_PATH = "/World/Robot"
HEAD_CAMERA_PATH = f"{ROBOT_PATH}/link4/first_person_camera"
LEFT_CAMERA_PATH = f"{ROBOT_PATH}/left_hand_link/left_hand_camera"
RIGHT_CAMERA_PATH = f"{ROBOT_PATH}/right_hand_link/right_hand_camera"

LEFT_ARM_NAMES = (
    "left_shoulder_y",
    "left_shoulder_x",
    "left_shoulder_z",
    "left_elbow_y",
    "left_elbow_x",
    "left_wrist_y",
    "left_wrist_z",
)
RIGHT_ARM_NAMES = (
    "right_shoulder_y",
    "right_shoulder_x",
    "right_shoulder_z",
    "right_elbow_y",
    "right_elbow_x",
    "right_wrist_y",
    "right_wrist_z",
)

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

# Teleoperation-ready pose derived from the LW-BenchHub X7S URDF.  The
# upstream all-zero arm pose folds both forearms close to the chassis.  This
# symmetric pose starts with the hands slightly raised, forward, and apart so
# a Quest user has room to calibrate and begin manipulating immediately.
# Forward-kinematics positions of the ready joint configurations below.  Using
# their exact Cartesian anchors prevents the first B calibration from pulling
# either arm toward an approximate rounded target.
X7S_LEFT_HOME = np.asarray(
    [0.35127807, 0.23962249, 0.33677017], dtype=np.float64
)
X7S_RIGHT_HOME = np.asarray(
    [0.35127259, -0.23962586, 0.33675855], dtype=np.float64
)
X7S_LEFT_READY_Q = np.asarray(
    [0.0513, -0.2732, -0.2523, 0.4505, -0.3086, -0.4492, 0.1101],
    dtype=np.float64,
)
X7S_RIGHT_READY_Q = np.asarray(
    [-0.0513, 0.2721, 0.2513, -0.4514, 0.3075, 0.4497, -0.1094],
    dtype=np.float64,
)


def _wait_for_stage(max_updates: int = 1200) -> None:
    for _ in range(max_updates):
        simulation_app.update()
        if not is_stage_loading():
            return
    raise TimeoutError("USD stage did not finish loading")


def _add_reference(stage: Usd.Stage, asset: Path, prim_path: str) -> None:
    prim = stage.DefinePrim(prim_path, "Xform")
    if not prim.GetReferences().AddReference(str(asset)):
        raise RuntimeError(f"Could not reference {asset} at {prim_path}")


def _set_pose(
    stage: Usd.Stage,
    prim_path: str,
    position: list[float],
    orientation_wxyz: list[float] | None = None,
    rotation_x_degrees: float | None = None,
    scale: list[float] | None = None,
) -> None:
    prim = stage.GetPrimAtPath(prim_path)
    if not prim:
        raise RuntimeError(f"Missing referenced prim: {prim_path}")
    xform = UsdGeom.Xformable(prim)
    xform.ClearXformOpOrder()
    xform.AddTranslateOp().Set(Gf.Vec3d(*map(float, position)))
    if orientation_wxyz is not None:
        w, x, y, z = map(float, orientation_wxyz)
        xform.AddOrientOp().Set(Gf.Quatf(w, Gf.Vec3f(x, y, z)))
    elif rotation_x_degrees is not None:
        xform.AddRotateXOp().Set(float(rotation_x_degrees))
    if scale is not None:
        xform.AddScaleOp().Set(Gf.Vec3f(*map(float, scale)))


def _configure_scene_collision(stage: Usd.Stage) -> int:
    root = stage.GetPrimAtPath(ENVIRONMENT_COLLISION_PATH)
    UsdGeom.Imageable(root).CreateVisibilityAttr().Set(UsdGeom.Tokens.invisible)
    count = 0
    for prim in Usd.PrimRange(root):
        if not prim.IsA(UsdGeom.Mesh):
            continue
        collision = (
            UsdPhysics.CollisionAPI(prim)
            if prim.HasAPI(UsdPhysics.CollisionAPI)
            else UsdPhysics.CollisionAPI.Apply(prim)
        )
        collision.CreateCollisionEnabledAttr(True)
        mesh_collision = (
            UsdPhysics.MeshCollisionAPI(prim)
            if prim.HasAPI(UsdPhysics.MeshCollisionAPI)
            else UsdPhysics.MeshCollisionAPI.Apply(prim)
        )
        mesh_collision.CreateApproximationAttr().Set("meshSimplification")
        count += 1
    return count


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


def _configure_x7s_physics(stage: Usd.Stage) -> int:
    """Apply the actuator and rigid-body values from X7_CFG."""
    root = stage.GetPrimAtPath(ROBOT_PATH)
    if not root:
        raise RuntimeError("X7S reference did not compose")
    root.CreateAttribute(
        "physxArticulation:enabledSelfCollisions",
        Sdf.ValueTypeNames.Bool,
    ).Set(False)
    root.CreateAttribute(
        "physxArticulation:solverPositionIterationCount",
        Sdf.ValueTypeNames.Int,
    ).Set(8)
    root.CreateAttribute(
        "physxArticulation:solverVelocityIterationCount",
        Sdf.ValueTypeNames.Int,
    ).Set(0)

    # The USD adds three virtual planar-base bodies that are absent from the
    # arm URDF.  They have no colliders or mass API, so raw PhysX assigns a
    # negative placeholder mass and their drives inject impulses into the arm.
    # Isaac Lab masks this when loading its ArticulationCfg; author explicit,
    # small positive inertias for the direct Isaac Core path used here.
    for body_name in ("world", "base_x_link", "base_y_link", "base_yaw_link"):
        body = stage.GetPrimAtPath(f"{ROBOT_PATH}/{body_name}")
        if not body or not body.HasAPI(UsdPhysics.RigidBodyAPI):
            raise RuntimeError(f"Missing X7S virtual base body: {body_name}")
        mass = (
            UsdPhysics.MassAPI(body)
            if body.HasAPI(UsdPhysics.MassAPI)
            else UsdPhysics.MassAPI.Apply(body)
        )
        mass.CreateMassAttr(1.0)
        mass.CreateCenterOfMassAttr(Gf.Vec3f(0.0, 0.0, 0.0))
        mass.CreateDiagonalInertiaAttr(Gf.Vec3f(0.01, 0.01, 0.01))
        mass.CreatePrincipalAxesAttr(Gf.Quatf(1.0, Gf.Vec3f(0.0)))

    for prim in Usd.PrimRange(root):
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            prim.CreateAttribute(
                "physxRigidBody:disableGravity",
                Sdf.ValueTypeNames.Bool,
            ).Set(True)

    configured = 0
    for prim in Usd.PrimRange(root):
        if not prim.IsA(UsdPhysics.Joint):
            continue
        name = prim.GetName()
        if name in {"FixedJoint"}:
            continue
        if name.startswith("base_"):
            stiffness, damping, max_force, velocity_limit = 1.0e6, 1.0e4, 1.0e5, 1000.0
        elif name.startswith("body_"):
            stiffness, damping, max_force, velocity_limit = 1.0e6, 1.0e4, 1.0e5, 1000.0
        elif "shoulder" in name:
            stiffness, damping, max_force, velocity_limit = ARM_STIFFNESS, ARM_DAMPING, 87.0, 2.175
        elif "elbow" in name or "wrist" in name:
            stiffness, damping, max_force, velocity_limit = ARM_STIFFNESS, ARM_DAMPING, 12.0, 2.61
        elif "gripper" in name:
            stiffness, damping, max_force, velocity_limit = 3000.0, 500.0, 600.0, 0.03
        elif name.startswith("head_"):
            stiffness, damping, max_force, velocity_limit = 400.0, 80.0, 12.0, 2.61
        else:
            continue
        drive_type = "linear" if prim.IsA(UsdPhysics.PrismaticJoint) else "angular"
        drive = UsdPhysics.DriveAPI(prim, drive_type)
        # Angular USD drive gains are authored per degree, while the tensor API
        # and LW-BenchHub's ImplicitActuatorCfg values are per radian.
        gain_scale = 1.0 if drive_type == "linear" else math.pi / 180.0
        drive.CreateStiffnessAttr(stiffness * gain_scale)
        drive.CreateDampingAttr(damping * gain_scale)
        drive.CreateMaxForceAttr(max_force)
        drive.CreateTargetVelocityAttr(0.0)
        # Isaac Lab's implicit actuator enforces these X7_CFG limits outside
        # PhysX.  This task uses raw Isaac Core, so author the equivalent joint
        # velocity cap directly; the source USD otherwise allows ~1000 rad/s.
        authored_velocity_limit = (
            velocity_limit
            if drive_type == "linear"
            else math.degrees(velocity_limit)
        )
        prim.CreateAttribute(
            "physxJoint:maxJointVelocity",
            Sdf.ValueTypeNames.Float,
        ).Set(float(authored_velocity_limit))
        configured += 1
    return configured


def _add_camera(
    stage: Usd.Stage,
    path: str,
    translation: tuple[float, float, float],
    orientation_wxyz: tuple[float, float, float, float],
    role: str,
) -> None:
    if not stage.GetPrimAtPath(Sdf.Path(path).GetParentPath()):
        raise RuntimeError(f"X7S camera parent is missing: {path}")
    camera = UsdGeom.Camera.Define(stage, path)
    camera.CreateFocalLengthAttr(19.3)
    camera.CreateHorizontalApertureAttr(36.0)
    camera.CreateVerticalApertureAttr(20.25)
    camera.CreateClippingRangeAttr(Gf.Vec2f(0.03, 100.0))
    camera.CreateFocusDistanceAttr(2.0)
    camera.CreateFStopAttr(0.0)
    camera.GetPrim().CreateAttribute(
        "x7s:cameraRole",
        Sdf.ValueTypeNames.String,
        custom=True,
    ).Set(role)
    xform = UsdGeom.Xformable(camera)
    xform.ClearXformOpOrder()
    xform.AddTranslateOp().Set(Gf.Vec3d(*translation))
    w, x, y, z = orientation_wxyz
    xform.AddOrientOp().Set(Gf.Quatf(w, Gf.Vec3f(x, y, z)))


def _add_x7s_cameras(stage: Usd.Stage) -> None:
    # Exact camera offsets from X7SEnvCfg.
    _add_camera(
        stage,
        HEAD_CAMERA_PATH,
        (0.1, 0.025, 0.065),
        (0.64086, 0.29884, -0.29884, -0.64086),
        "head",
    )
    _add_camera(
        stage,
        LEFT_CAMERA_PATH,
        (0.1, 0.0, 0.12),
        (0.57923, 0.40558, -0.40558, -0.57923),
        "left_wrist",
    )
    _add_camera(
        stage,
        RIGHT_CAMERA_PATH,
        (0.1, 0.0, 0.12),
        (0.57923, 0.40558, -0.40558, -0.57923),
        "right_wrist",
    )


class X7SJointCommander:
    """Joint targets in the exact X7S-Abs action order."""

    def __init__(self, robot: Articulation, placement: dict) -> None:
        self.robot = robot
        self.names = list(robot.dof_names)
        self.index = {name: index for index, name in enumerate(self.names)}
        expected = {
            "base_x_joint",
            "base_y_joint",
            "base_yaw_link",
            "body_z_joint",
            "body_y_joint",
            "head_z_joint",
            "head_y_joint",
            "left_gripper1",
            "left_gripper2",
            "right_gripper1",
            "right_gripper2",
            *LEFT_ARM_NAMES,
            *RIGHT_ARM_NAMES,
        }
        missing = sorted(expected - set(self.names))
        if missing or robot.num_dof != 25:
            raise RuntimeError(
                f"Invalid X7S articulation: DOFs={robot.num_dof}, missing={missing}"
            )
        self.limits = np.asarray(robot.get_dof_limits()[0], dtype=np.float64)
        self.max_velocities = np.full(robot.num_dof, 2.61, dtype=np.float32)
        for name, index in self.index.items():
            if name.startswith("base_") or name.startswith("body_"):
                self.max_velocities[index] = 1000.0
            elif "shoulder" in name:
                self.max_velocities[index] = 2.175
            elif "gripper" in name:
                self.max_velocities[index] = 0.03
        # USD authoring alone is not reliably transferred into the already
        # initialized PhysX tensor view.  Apply X7_CFG's actuator limits to the
        # live articulation as Isaac Lab does.
        self.robot.set_max_joint_velocities(
            self.max_velocities[np.newaxis, :]
        )
        self.target = np.zeros(robot.num_dof, dtype=np.float64)
        saved = placement["robot"]
        for name, value in (
            ("base_x_joint", saved["x"]),
            ("base_y_joint", saved["y"]),
            ("base_yaw_link", saved["yaw_radians"]),
            ("body_z_joint", saved["body_z_m"]),
            ("body_y_joint", saved["body_pitch_radians"]),
            ("head_z_joint", saved.get("head_yaw_radians", 0.0)),
            ("head_y_joint", saved.get("head_pitch_radians", 0.0)),
            ("left_gripper1", 0.044),
            ("left_gripper2", 0.044),
            ("right_gripper1", 0.0),
            ("right_gripper2", 0.0),
        ):
            self.target[self.index[name]] = self.clamp(name, float(value))
        self.left_arm_indices = np.asarray(
            [self.index[name] for name in LEFT_ARM_NAMES], dtype=np.int64
        )
        self.right_arm_indices = np.asarray(
            [self.index[name] for name in RIGHT_ARM_NAMES], dtype=np.int64
        )
        self.target[self.left_arm_indices] = X7S_LEFT_READY_Q
        self.target[self.right_arm_indices] = X7S_RIGHT_READY_Q
        self.home = self.target.copy()
        self.base_indices = np.asarray(
            [
                self.index["base_x_joint"],
                self.index["base_y_joint"],
                self.index["base_yaw_link"],
            ],
            dtype=np.int64,
        )
        self._last_written_base_target = self.target[self.base_indices].copy()
        self.zero_velocity = np.zeros((1, robot.num_dof), dtype=np.float32)
        self.reset()

    def clamp(self, name: str, value: float) -> float:
        index = self.index[name]
        low, high = self.limits[index]
        return float(np.clip(value, low, high))

    def reset(self) -> None:
        self.target[:] = self.home
        self.robot.set_joint_positions(self.target[np.newaxis, :])
        self.robot.set_joint_velocities(self.zero_velocity)
        self._last_written_base_target = self.target[self.base_indices].copy()
        self.apply()

    def apply(self) -> None:
        if not np.all(np.isfinite(self.target)):
            raise RuntimeError("Refusing to apply non-finite X7S joint targets")
        self.target = np.clip(
            self.target,
            self.limits[:, 0],
            self.limits[:, 1],
        )
        self.robot.set_joint_position_targets(self.target[np.newaxis, :])
        self.robot.set_joint_velocity_targets(self.zero_velocity)
        # Preserve the task's deliberately kinematic planar-base behavior.
        # Compare against the last command written, not the simulated joint
        # value: comparing against PhysX feedback caused a full articulation
        # state rewrite every frame and continuously excited the arm joints.
        if (
            np.max(
                np.abs(
                    self.target[self.base_indices]
                    - self._last_written_base_target
                )
            )
            > 1.0e-7
        ):
            # The planar base is intentionally kinematic.  Move it with one
            # coherent articulation state: preserving lagging physical arm
            # positions while teleporting the base injects large constraint
            # velocities and leaves both arms oscillating after the base stops.
            self.robot.set_joint_positions(self.target[np.newaxis, :])
            self.robot.set_joint_velocities(self.zero_velocity)
            self._last_written_base_target = self.target[self.base_indices].copy()

    def set_arm_targets(self, left: np.ndarray, right: np.ndarray) -> None:
        if left.shape == (7,) and np.all(np.isfinite(left)):
            self.target[self.left_arm_indices] = left
        if right.shape == (7,) and np.all(np.isfinite(right)):
            self.target[self.right_arm_indices] = right

    def set_grippers(self, left_closed: bool, right_closed: bool) -> None:
        left = 0.0 if left_closed else 0.044
        right = -0.04 if right_closed else 0.0
        for name in ("left_gripper1", "left_gripper2"):
            self.target[self.index[name]] = self.clamp(name, left)
        for name in ("right_gripper1", "right_gripper2"):
            self.target[self.index[name]] = self.clamp(name, right)

    def move_base_local(self, forward: float, lateral: float, yaw: float) -> None:
        yaw_now = float(self.target[self.index["base_yaw_link"]])
        cosine, sine = math.cos(yaw_now), math.sin(yaw_now)
        self.target[self.index["base_x_joint"]] += cosine * forward - sine * lateral
        self.target[self.index["base_y_joint"]] += sine * forward + cosine * lateral
        self.target[self.index["base_yaw_link"]] += yaw

    def move_body(self, height_delta: float, pitch_delta: float) -> None:
        for name, delta in (
            ("body_z_joint", height_delta),
            ("body_y_joint", pitch_delta),
        ):
            self.target[self.index[name]] = self.clamp(
                name,
                self.target[self.index[name]] + delta,
            )


def _valid_pose(matrix: np.ndarray) -> bool:
    return bool(
        matrix.shape == (4, 4)
        and np.all(np.isfinite(matrix))
        and abs(float(np.linalg.det(matrix[:3, :3]))) > 0.1
    )


def _matrix_to_pose_wxyz(matrix: np.ndarray) -> np.ndarray:
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


def _usd_pose_matrix(stage: Usd.Stage, prim_path: str) -> np.ndarray:
    matrix = UsdGeom.XformCache().GetLocalToWorldTransform(
        stage.GetPrimAtPath(prim_path)
    )
    quaternion = Gf.Transform(matrix).GetRotation().GetQuat()
    result = np.eye(4, dtype=np.float64)
    # Gf matrices use row-vector convention; SciPy/Pinocchio use columns.
    result[:3, :3] = np.asarray(Gf.Matrix3d(quaternion), dtype=np.float64).T
    result[:3, 3] = np.asarray(matrix.ExtractTranslation(), dtype=np.float64)
    return result


def _validate_usd_urdf_kinematics(
    world: World,
    stage: Usd.Stage,
    robot: Articulation,
    joints: "X7SJointCommander",
    ik,
) -> None:
    """Numerically compare X7S USD links with the IK URDF at equal q."""
    samples = (
        ("zero", np.zeros(7), np.zeros(7)),
        ("ready", X7S_LEFT_READY_Q, X7S_RIGHT_READY_Q),
        (
            "asymmetric",
            np.asarray([-0.2, -0.35, 0.25, 0.3, -0.2, 0.15, -0.1]),
            np.asarray([0.25, 0.3, -0.2, -0.25, 0.15, -0.1, 0.2]),
        ),
    )
    print("[KINEMATICS] USD versus URDF end-effector validation", flush=True)
    for label, left_q, right_q in samples:
        full = np.zeros(robot.num_dof, dtype=np.float64)
        full[joints.left_arm_indices] = left_q
        full[joints.right_arm_indices] = right_q
        robot.set_joint_positions(full[np.newaxis, :])
        robot.set_joint_velocities(np.zeros((1, robot.num_dof), dtype=np.float32))
        robot.set_joint_position_targets(full[np.newaxis, :])
        for _ in range(3):
            world.step(render=False)

        base_world = _usd_pose_matrix(stage, f"{ROBOT_PATH}/base_link")
        for side, q in (("left", left_q), ("right", right_q)):
            usd_world = _usd_pose_matrix(stage, f"{ROBOT_PATH}/{side}_hand_link")
            usd_relative = np.linalg.inv(base_world) @ usd_world
            urdf_pose = getattr(ik, side).forward_pose(q)
            position_error = float(
                np.linalg.norm(usd_relative[:3, 3] - urdf_pose[:3, 3])
            )
            rotation_error = float(
                np.linalg.norm(
                    Rotation.from_matrix(
                        urdf_pose[:3, :3].T @ usd_relative[:3, :3]
                    ).as_rotvec()
                )
            )
            print(
                f"[KINEMATICS] {label} {side}: "
                f"usd_xyz={usd_relative[:3, 3].round(6).tolist()} "
                f"urdf_xyz={urdf_pose[:3, 3].round(6).tolist()} "
                f"position_error={position_error:.8f} m "
                f"rotation_error={rotation_error:.8f} rad",
                flush=True,
            )


def _smooth_pose(
    target: np.ndarray,
    previous: np.ndarray | None,
    alpha: float = 0.25,
    max_position_delta: float = 0.012,
    max_rotation_delta: float = 0.08,
) -> np.ndarray:
    if previous is None:
        return target.copy()
    result = previous.copy()
    position = alpha * target[:3, 3] + (1.0 - alpha) * previous[:3, 3]
    delta_position = position - previous[:3, 3]
    norm = float(np.linalg.norm(delta_position))
    if norm > max_position_delta:
        delta_position *= max_position_delta / norm
    result[:3, 3] = previous[:3, 3] + delta_position

    relative = target[:3, :3] @ previous[:3, :3].T
    rotation_vector = Rotation.from_matrix(relative).as_rotvec() * alpha
    angle = float(np.linalg.norm(rotation_vector))
    if angle > max_rotation_delta:
        rotation_vector *= max_rotation_delta / angle
    result[:3, :3] = (
        Rotation.from_rotvec(rotation_vector).as_matrix() @ previous[:3, :3]
    )
    return result


class X7SQuestController:
    """LW-BenchHub VRController pose/input semantics on the existing bridge."""

    def __init__(self, joints: X7SJointCommander) -> None:
        self.joints = joints
        self.bridge: QuestBridge | None = None
        self.camera: Camera | None = None
        self.stream_frame = 0
        self.stream_interval = max(1, round(60 / QUEST_STREAM_FPS))
        self.ik, self.ik_backend = create_x7s_ik(
            urdf_path=X7S_URDF,
            package_dirs=[str(X7S_ROOT.parent)],
            scripts_dir=SCRIPT_DIR,
            vendor_dir=X7S_ROOT / "vendor",
            max_iters=50,
            tol=1.0e-4,
        )
        self.active = False
        self.paused = False
        # Match LW-BenchHub: arm motion remains disabled until Right B has
        # been pressed and released.  If that happens before both poses arrive,
        # activation_pending remembers the explicit request.
        self.activation_pending = False
        self.last_b = False
        self.last_a = False
        self.reset_chord_latched = False
        self.reset_requested = False
        self.last_solved_counts: tuple[int, int] | None = None
        self.controller_anchor = {"left": None, "right": None}
        self.target_anchor = {
            "left": self._home_target(X7S_LEFT_HOME),
            "right": self._home_target(X7S_RIGHT_HOME),
        }
        self.desired = {
            side: target.copy() for side, target in self.target_anchor.items()
        }
        self.smoothed = {"left": None, "right": None}
        self.latest_pose = {"left": None, "right": None}
        self.last_states = {
            side: {"trigger": 0.0, "squeeze": 0.0}
            for side in ("left", "right")
        }

    @staticmethod
    def _home_target(position: np.ndarray) -> np.ndarray:
        target = np.eye(4, dtype=np.float64)
        target[:3, 3] = position
        return target

    @staticmethod
    def _convert_controller_pose(pose: np.ndarray) -> np.ndarray:
        return GRD_YUP_TO_GRD_ZUP @ pose @ GRD_ZUP_TO_GRD_YUP

    def start(self) -> None:
        if HEADLESS or SMOKE_TEST or not QUEST_ENABLED:
            return
        self.camera = Camera(
            prim_path=HEAD_CAMERA_PATH,
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
        print("\n[QUEST] X7S WebXR server is starting.", flush=True)
        print(f"[QUEST] 1. Accept the certificate: {self.bridge.certificate_url}", flush=True)
        print(f"[QUEST] 2. Open and press Enter VR: {self.bridge.quest_url}", flush=True)
        print(
            "[QUEST] Hold both controllers comfortably, then press and "
            "release Right B to calibrate and start the arms.",
            flush=True,
        )

    def reset(self) -> None:
        self.joints.reset()
        self.ik.reset()
        self.active = False
        self.paused = False
        self.activation_pending = False
        self.last_b = False
        self.last_a = False
        self.reset_chord_latched = False
        self.reset_requested = False
        self.last_solved_counts = None
        self.controller_anchor = {"left": None, "right": None}
        self.target_anchor = {
            "left": self._home_target(X7S_LEFT_HOME),
            "right": self._home_target(X7S_RIGHT_HOME),
        }
        self.desired = {
            side: target.copy() for side, target in self.target_anchor.items()
        }
        self.smoothed = {"left": None, "right": None}

    def _anchor(self, use_current_target: bool) -> bool:
        if any(pose is None for pose in self.latest_pose.values()):
            return False
        for side in ("left", "right"):
            self.controller_anchor[side] = self.latest_pose[side].copy()
            if use_current_target:
                self.target_anchor[side] = self.desired[side].copy()
            else:
                home = X7S_LEFT_HOME if side == "left" else X7S_RIGHT_HOME
                self.target_anchor[side] = self._home_target(home)
                self.desired[side] = self.target_anchor[side].copy()
                self.smoothed[side] = self.desired[side].copy()
        self.ik.reset()
        self.last_solved_counts = None
        return True

    def _update_desired(self) -> None:
        for side in ("left", "right"):
            current = self.latest_pose[side]
            anchor = self.controller_anchor[side]
            if current is None or anchor is None:
                continue
            delta_rotation = current[:3, :3] @ anchor[:3, :3].T
            target = np.eye(4, dtype=np.float64)
            target[:3, 3] = (
                self.target_anchor[side][:3, 3]
                + current[:3, 3]
                - anchor[:3, 3]
            )
            target[:3, :3] = (
                delta_rotation @ self.target_anchor[side][:3, :3]
            )
            self.desired[side] = target
            self.smoothed[side] = _smooth_pose(target, self.smoothed[side])

    def solve_current_targets(self) -> tuple[bool, bool]:
        left_pose = _matrix_to_pose_wxyz(self.smoothed["left"])
        right_pose = _matrix_to_pose_wxyz(self.smoothed["right"])
        # Seed from the currently commanded arm configuration.  The X7S is
        # redundant (7 DOF), so restarting every solve from zero can select a
        # different elbow branch even for a nearby hand target.
        left_warm_start = self.joints.target[
            self.joints.left_arm_indices
        ][np.newaxis, :].copy()
        right_warm_start = self.joints.target[
            self.joints.right_arm_indices
        ][np.newaxis, :].copy()
        left_q, left_ok, right_q, right_ok = self.ik.solve_pose_to_joints(
            left_pose[np.newaxis, :],
            right_pose[np.newaxis, :],
            left_warm_start=left_warm_start,
            right_warm_start=right_warm_start,
        )
        self.joints.set_arm_targets(left_q[0], right_q[0])
        return bool(left_ok[0]), bool(right_ok[0])

    def update(self) -> None:
        if self.bridge is None:
            return
        states = {side: self.bridge.state(side) for side in ("left", "right")}
        self.last_states = states
        for side in ("left", "right"):
            pose = self.bridge.pose(side)
            if _valid_pose(pose):
                self.latest_pose[side] = self._convert_controller_pose(pose)

        # Only complete a previously requested B activation here.  Merely
        # receiving controller poses must never start arm motion.
        if self.activation_pending and not self.active:
            left_updates, right_updates = self.bridge.update_counts
            if (
                left_updates >= 5
                and right_updates >= 5
                and self._anchor(use_current_target=False)
            ):
                self.activation_pending = False
                self.active = True
                self.paused = False
                print(
                    "[QUEST] Requested B activation completed; X7S arm "
                    "teleoperation calibrated and started",
                    flush=True,
                )

        right_b = bool(states["right"]["b_button"])
        if right_b and not self.last_b:
            print("[QUEST] Right B pressed", flush=True)
        if self.last_b and not right_b:
            print("[QUEST] Right B released", flush=True)
            if self.active:
                # Re-anchor in place without toggling teleoperation off.  A
                # toggle made repeated B presses look like all Quest input had
                # stopped working.
                if self._anchor(use_current_target=True):
                    self.paused = False
                    self.activation_pending = False
                    print(
                        "[QUEST] X7S arm teleoperation recalibrated (still active)",
                        flush=True,
                    )
            elif self._anchor(use_current_target=False):
                self.active = True
                self.paused = False
                self.activation_pending = False
                print("[QUEST] X7S arm teleoperation started and calibrated", flush=True)
            else:
                self.activation_pending = True
                print(
                    "[QUEST] Start requested; waiting for valid left/right "
                    "controller poses",
                    flush=True,
                )
        self.last_b = right_b

        right_a = bool(states["right"]["a_button"])
        if self.active and right_a and not self.last_a:
            self.paused = True
            print("[QUEST] X7S arm pose paused", flush=True)
        elif self.active and self.last_a and not right_a:
            if self._anchor(use_current_target=True):
                self.paused = False
                print("[QUEST] X7S arm pose resumed from a new controller origin", flush=True)
        self.last_a = right_a

        # WebXR calls the physical left X/Y buttons bButton/aButton.  Require
        # both together so a benchmark reset cannot be triggered accidentally.
        reset_chord = bool(
            states["left"]["a_button"] and states["left"]["b_button"]
        )
        if reset_chord and not self.reset_chord_latched:
            self.reset_requested = True
            print(
                "[QUEST] Left X+Y reset requested for robot and task object",
                flush=True,
            )
        self.reset_chord_latched = reset_chord

        if self.active and not self.paused:
            update_counts = self.bridge.update_counts
            # WebXR normally streams at about 30 Hz while Isaac renders at
            # 60 Hz.  Solving twice for the same pose wastes CPU and can walk
            # a redundant 7-DOF solution between equivalent elbow branches.
            # Advance smoothing and IK exactly once per fresh controller pose.
            if update_counts != self.last_solved_counts:
                self._update_desired()
                left_ok, right_ok = self.solve_current_targets()
                self.last_solved_counts = update_counts
            else:
                left_ok = right_ok = True
            if not left_ok and self.ik.left.failure_count % 300 == 1:
                print(
                    "[IK] X7S left target gated; "
                    f"position_error={self.ik.left.last_position_error:.4f} m, "
                    f"orientation_error={self.ik.left.last_orientation_error:.4f} rad",
                    flush=True,
                )
            if not right_ok and self.ik.right.failure_count % 300 == 1:
                print(
                    "[IK] X7S right target gated; "
                    f"position_error={self.ik.right.last_position_error:.4f} m, "
                    f"orientation_error={self.ik.right.last_orientation_error:.4f} rad",
                    flush=True,
                )

        # X7S repository controller mapping and action scales.  Locomotion and
        # body controls must remain available even before arm calibration.
        right_x = float(states["right"]["thumbstick_x"])
        right_y = float(states["right"]["thumbstick_y"])
        if float(states["right"]["squeeze"]) > 0.5:
            self.joints.move_base_local(0.005 * right_x, 0.0, 0.005 * right_y)
        else:
            self.joints.move_base_local(0.005 * right_x, 0.005 * right_y, 0.0)
        left_x = float(states["left"]["thumbstick_x"])
        left_y = float(states["left"]["thumbstick_y"])
        self.joints.move_body(0.00125 * left_x, 0.00625 * left_y)

        self.joints.set_grippers(
            left_closed=float(states["left"]["trigger"]) > 0.5,
            right_closed=float(states["right"]["trigger"]) > 0.5,
        )

    def stream_camera(self) -> None:
        if self.bridge is None or self.camera is None:
            return
        self.stream_frame += 1
        if self.stream_frame % self.stream_interval:
            return
        rgba = self.camera.get_rgba()
        if rgba is not None and rgba.size:
            self.bridge.write_mono_image(np.asarray(rgba)[..., :3])

    def status(self) -> str:
        if self.bridge is None:
            return "disabled"
        left_updates, right_updates = self.bridge.update_counts
        controller_delta = []
        for side in ("left", "right"):
            pose = self.latest_pose[side]
            anchor = self.controller_anchor[side]
            controller_delta.append(
                0.0
                if pose is None or anchor is None
                else float(np.linalg.norm(pose[:3, 3] - anchor[:3, 3]))
            )
        return (
            f"updates=({left_updates},{right_updates}), "
            f"active={self.active}, paused={self.paused}, "
            f"buttons=(A:{int(bool(self.last_states['right']['a_button']))},"
            f"B:{int(bool(self.last_states['right']['b_button']))}), "
            f"hand_delta=({controller_delta[0]:.3f},{controller_delta[1]:.3f})m, "
            f"IK_success=({self.ik.left.success_count},{self.ik.right.success_count}), "
            f"IK_gated=({self.ik.left.failure_count},{self.ik.right.failure_count}), "
            f"pos_error=({self.ik.left.last_position_error:.3f},"
            f"{self.ik.right.last_position_error:.3f})m, "
            f"rot_error=({self.ik.left.last_orientation_error:.3f},"
            f"{self.ik.right.last_orientation_error:.3f})rad"
        )

    def close(self) -> None:
        if self.bridge is not None:
            self.bridge.close()
            self.bridge = None
        close_ik = getattr(self.ik, "close", None)
        if close_ik is not None:
            close_ik()


def _open_camera_viewports(stage: Usd.Stage) -> list:
    if HEADLESS or SMOKE_TEST or not SHOW_CAMERA_VIEWPORTS:
        return []
    specs = (
        ("X7S Head Camera", HEAD_CAMERA_PATH),
        ("X7S Left Wrist Camera", LEFT_CAMERA_PATH),
        ("X7S Right Wrist Camera", RIGHT_CAMERA_PATH),
    )
    windows = []
    for index, (title, path) in enumerate(specs):
        if not stage.GetPrimAtPath(path):
            raise RuntimeError(f"Missing X7S display camera: {path}")
        windows.append(
            create_viewport_window(
                name=title,
                width=470,
                height=330,
                position_x=10 + index * 475,
                position_y=80,
                camera_path=Sdf.Path(path),
            )
        )
    print("[VIEW] X7S head, left-wrist, and right-wrist windows opened.", flush=True)
    return windows


def _read_view_camera(viewport) -> tuple[np.ndarray, np.ndarray]:
    state = ViewportCameraState(str(viewport.get_active_camera()), viewport)
    eye = np.asarray(tuple(state.position_world), dtype=np.float64)
    target = np.asarray(tuple(state.target_world), dtype=np.float64)
    if not np.all(np.isfinite(np.concatenate((eye, target)))):
        raise RuntimeError("Viewport camera contains a non-finite value")
    return eye, target


def _save_placement(
    placement: dict,
    joints: X7SJointCommander,
    task_object: RigidPrim,
    viewport,
) -> None:
    object_position, object_orientation = task_object.get_world_poses()
    eye, target = _read_view_camera(viewport)
    placement["robot"] = {
        "x": float(joints.target[joints.index["base_x_joint"]]),
        "y": float(joints.target[joints.index["base_y_joint"]]),
        "yaw_radians": float(joints.target[joints.index["base_yaw_link"]]),
        "root_z_m": float(placement["robot"].get("root_z_m", 0.0)),
        "body_z_m": float(joints.target[joints.index["body_z_joint"]]),
        "body_pitch_radians": float(joints.target[joints.index["body_y_joint"]]),
        "head_yaw_radians": float(joints.target[joints.index["head_z_joint"]]),
        "head_pitch_radians": float(joints.target[joints.index["head_y_joint"]]),
    }
    placement["task_object"]["position_xyz_m"] = np.asarray(
        object_position[0], dtype=float
    ).tolist()
    placement["task_object"]["orientation_wxyz"] = np.asarray(
        object_orientation[0], dtype=float
    ).tolist()
    placement["camera"] = {
        "eye_xyz_m": eye.tolist(),
        "target_xyz_m": target.tolist(),
    }
    placement["updated_at"] = datetime.now().astimezone().isoformat()
    CONFIG_PATH.write_text(
        json.dumps(placement, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Saved X7S/object/camera placement to {CONFIG_PATH}", flush=True)


def build_world() -> tuple[World, Usd.Stage, Articulation, RigidPrim, int, int]:
    World.clear_instance()
    create_new_stage()
    simulation_app.update()
    world = World(
        stage_units_in_meters=1.0,
        physics_dt=1.0 / (60.0 * PHYSICS_SUBSTEPS),
        rendering_dt=1.0 / 60.0,
    )
    stage = omni.usd.get_context().get_stage()
    _add_reference(stage, ENVIRONMENT_VISUAL_ASSET, ENVIRONMENT_VISUAL_PATH)
    _add_reference(stage, ENVIRONMENT_COLLISION_ASSET, ENVIRONMENT_COLLISION_PATH)
    _add_reference(stage, TASK_OBJECT_ASSET, TASK_OBJECT_PATH)
    _add_reference(stage, ROBOT_ASSET, ROBOT_PATH)
    _wait_for_stage()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    _set_pose(
        stage,
        ENVIRONMENT_VISUAL_PATH,
        PLACEMENT["environment"]["position_xyz_m"],
        rotation_x_degrees=PLACEMENT["environment"]["rotation_x_degrees"],
    )
    _set_pose(
        stage,
        ENVIRONMENT_COLLISION_PATH,
        PLACEMENT["environment"]["position_xyz_m"],
        rotation_x_degrees=PLACEMENT["environment"]["rotation_x_degrees"],
    )
    _set_pose(
        stage,
        TASK_OBJECT_PATH,
        PLACEMENT["task_object"]["position_xyz_m"],
        orientation_wxyz=PLACEMENT["task_object"]["orientation_wxyz"],
        scale=PLACEMENT["task_object"]["scale_xyz"],
    )
    _set_pose(
        stage,
        ROBOT_PATH,
        [0.0, 0.0, float(PLACEMENT["robot"].get("root_z_m", 0.0))],
    )
    scene_colliders = _configure_scene_collision(stage) if ENABLE_SCENE_COLLISION else 0
    _add_ground_and_lights(stage)
    drive_count = _configure_x7s_physics(stage)
    _add_x7s_cameras(stage)

    robot = world.scene.add(
        Articulation(
            prim_paths_expr=ROBOT_PATH,
            name="x7s_task_robot",
            reset_xform_properties=False,
        )
    )
    task_object = RigidPrim(
        prim_paths_expr=TASK_OBJECT_PATH,
        name="x7s_task_object",
        reset_xform_properties=False,
    )
    world.reset()
    task_object.initialize()
    if not task_object.is_physics_handle_valid():
        raise RuntimeError("Task object rigid body did not initialize")
    return world, stage, robot, task_object, scene_colliders, drive_count


HELP = """
X7S Mobile Drilling Task — LW-BenchHub Quest controls

Quest 3S:
  Right B release       start arm control after both controllers are tracked
  Right B again         recalibrate arm control in the current hand pose
  Move both controllers command both 7-DOF arms (no Grip clutch)
  Left/Right Trigger    close the corresponding X7S gripper
  Right A hold/release  pause arms, then resume from a new controller origin
  Right stick           translate base; hold Right Grip for forward/yaw mode
  Left stick            body lift / body pitch
  Left X + Y together   reset X7S and task object

Keyboard fallback:
  W/S, A/D              base forward/reverse and turn
  T/Y                    body lift up/down
  U/O                    body pitch
  R                      reset X7S and task object
  P                      save X7S/object/viewport placement
  ESC                    quit
""".strip()


def main() -> None:
    world = None
    quest = None
    subscription = None
    try:
        print("[X7S] Building drilling task stage...", flush=True)
        world, stage, robot, task_object, scene_colliders, drive_count = build_world()
        joints = X7SJointCommander(robot, PLACEMENT)
        for _ in range(20 * PHYSICS_SUBSTEPS):
            joints.apply()
            world.step(render=False)
        quest = X7SQuestController(joints)
        if VALIDATE_KINEMATICS:
            _validate_usd_urdf_kinematics(world, stage, robot, joints, quest.ik)
            return

        viewport = None
        if not HEADLESS:
            viewport = get_active_viewport()
            viewport.set_texture_resolution((1280, 720))
            set_camera_view(
                eye=list(map(float, PLACEMENT["camera"]["eye_xyz_m"])),
                target=list(map(float, PLACEMENT["camera"]["target_xyz_m"])),
                camera_prim_path="/OmniverseKit_Persp",
            )
        quest.start()
        camera_windows = _open_camera_viewports(stage)

        print(HELP, flush=True)
        print("\nLoaded X7S drilling assets:", flush=True)
        print(f"  environment: {ENVIRONMENT_VISUAL_ASSET}", flush=True)
        print(f"  task object: {TASK_OBJECT_ASSET}", flush=True)
        print(f"  X7S robot:   {ROBOT_ASSET}", flush=True)
        print(f"  X7S URDF IK: {X7S_URDF}", flush=True)
        print(f"  X7S DOFs/drives: {robot.num_dof}/{drive_count}", flush=True)
        print(f"  scene colliders: {scene_colliders}", flush=True)
        print(f"  IK: {quest.ik_backend}", flush=True)
        print(f"  camera viewports: {len(camera_windows)}", flush=True)
        print(f"  placement: {CONFIG_PATH}", flush=True)
        if not HEADLESS:
            print(
                "\nReady. Enter VR, hold both controllers comfortably, then "
                "press and release Right B to start arm control.",
                flush=True,
            )

        held: set[carb.input.KeyboardInput] = set()
        state = {"quit": False, "reset": False, "save": False}
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
                    if key == carb.input.KeyboardInput.R:
                        state["reset"] = True
                    elif key == carb.input.KeyboardInput.P:
                        state["save"] = True
                    elif key == carb.input.KeyboardInput.ESCAPE:
                        state["quit"] = True
                elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
                    held.discard(key)
                return True

            subscription = input_interface.subscribe_to_keyboard_events(keyboard, on_key)

        initial_targets = joints.target.copy()
        frames = 0
        world.play()
        while simulation_app.is_running() and not state["quit"]:
            if state["reset"]:
                quest.reset()
                saved = PLACEMENT["task_object"]
                task_object.set_world_poses(
                    positions=np.asarray([saved["position_xyz_m"]], dtype=np.float32),
                    orientations=np.asarray([saved["orientation_wxyz"]], dtype=np.float32),
                )
                task_object.set_velocities(np.zeros((1, 6), dtype=np.float32))
                state["reset"] = False
                print("Reset X7S and drilling object", flush=True)
            if state["save"] and viewport is not None:
                _save_placement(PLACEMENT, joints, task_object, viewport)
                state["save"] = False

            if SMOKE_TEST:
                if frames == 0:
                    quest.smoothed["left"] = quest._home_target(
                        X7S_LEFT_HOME + np.asarray([0.025, 0.012, 0.015])
                    )
                    quest.smoothed["right"] = quest._home_target(
                        X7S_RIGHT_HOME + np.asarray([0.025, -0.012, 0.015])
                    )
                if frames < 20:
                    quest.solve_current_targets()
                    if SMOKE_BASE_MOTION:
                        joints.move_base_local(0.004, 0.0, 0.002)
                    if SMOKE_BODY_MOTION:
                        joints.move_body(0.001, -0.001)
            else:
                quest.update()
                if quest.reset_requested:
                    state["reset"] = True
                    quest.reset_requested = False
                forward = float(
                    (carb.input.KeyboardInput.W in held)
                    - (carb.input.KeyboardInput.S in held)
                )
                turn = float(
                    (carb.input.KeyboardInput.A in held)
                    - (carb.input.KeyboardInput.D in held)
                )
                joints.move_base_local(0.30 / 60.0 * forward, 0.0, 0.35 / 60.0 * turn)
                joints.move_body(
                    0.075 / 60.0
                    * float(
                        (carb.input.KeyboardInput.T in held)
                        - (carb.input.KeyboardInput.Y in held)
                    ),
                    0.35 / 60.0
                    * float(
                        (carb.input.KeyboardInput.U in held)
                        - (carb.input.KeyboardInput.O in held)
                    ),
                )

            joints.apply()
            for _ in range(PHYSICS_SUBSTEPS):
                world.step(render=False)
            if not HEADLESS:
                world.render()
                quest.stream_camera()
            frames += 1
            if SMOKE_TEST and frames % 10 == 0:
                smoke_positions = np.asarray(
                    robot.get_joint_positions()[0], dtype=np.float64
                )
                smoke_velocities = np.asarray(
                    robot.get_joint_velocities()[0], dtype=np.float64
                )
                smoke_arm_indices = np.concatenate(
                    (joints.left_arm_indices, joints.right_arm_indices)
                )
                print(
                    f"X7S arm tracking frame={frames}: "
                    f"max_error={np.max(np.abs(smoke_positions[smoke_arm_indices] - joints.target[smoke_arm_indices])):.6f}, "
                    f"max_velocity={np.max(np.abs(smoke_velocities[smoke_arm_indices])):.6f}",
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
                raise RuntimeError("X7S smoke test produced non-finite joints")
            base_motion = float(
                np.linalg.norm(
                    positions[
                        [joints.index["base_x_joint"], joints.index["base_y_joint"]]
                    ]
                    - initial_targets[
                        [joints.index["base_x_joint"], joints.index["base_y_joint"]]
                    ]
                )
            )
            arm_motion = float(
                max(
                    np.max(np.abs(positions[joints.left_arm_indices])),
                    np.max(np.abs(positions[joints.right_arm_indices])),
                )
            )
            arm_tracking_error = float(
                max(
                    np.max(
                        np.abs(
                            positions[joints.left_arm_indices]
                            - joints.target[joints.left_arm_indices]
                        )
                    ),
                    np.max(
                        np.abs(
                            positions[joints.right_arm_indices]
                            - joints.target[joints.right_arm_indices]
                        )
                    ),
                )
            )
            print(
                "X7S drilling smoke result: "
                f"base_motion={base_motion:.6f} m, arm_motion={arm_motion:.6f} rad, "
                f"arm_tracking_error={arm_tracking_error:.6f} rad, "
                f"IK_success=({quest.ik.left.success_count},{quest.ik.right.success_count}), "
                f"IK_gated=({quest.ik.left.failure_count},{quest.ik.right.failure_count})",
                flush=True,
            )
            print(
                "X7S arm smoke actual/target/error: "
                f"left={positions[joints.left_arm_indices].round(6).tolist()}/"
                f"{joints.target[joints.left_arm_indices].round(6).tolist()}/"
                f"{(positions[joints.left_arm_indices] - joints.target[joints.left_arm_indices]).round(6).tolist()}, "
                f"right={positions[joints.right_arm_indices].round(6).tolist()}/"
                f"{joints.target[joints.right_arm_indices].round(6).tolist()}/"
                f"{(positions[joints.right_arm_indices] - joints.target[joints.right_arm_indices]).round(6).tolist()}",
                flush=True,
            )
            final_velocities = np.asarray(
                robot.get_joint_velocities()[0], dtype=np.float64
            )
            print(
                "X7S arm smoke velocities: "
                f"left={final_velocities[joints.left_arm_indices].round(6).tolist()}, "
                f"right={final_velocities[joints.right_arm_indices].round(6).tolist()}",
                flush=True,
            )
            arm_kps, arm_kds = robot.get_gains(
                joint_indices=joints.left_arm_indices
            )
            print(
                "X7S left arm gains/effort: "
                f"kp={np.asarray(arm_kps)[0].round(6).tolist()}, "
                f"kd={np.asarray(arm_kds)[0].round(6).tolist()}, "
                f"max_effort={np.asarray(robot.get_max_efforts(joint_indices=joints.left_arm_indices))[0].round(6).tolist()}",
                flush=True,
            )
            print(
                "X7S base smoke positions/targets: "
                f"actual={positions[:3].tolist()}, "
                f"initial={initial_targets[:3].tolist()}, "
                f"target={joints.target[:3].tolist()}",
                flush=True,
            )
            applied = robot.get_applied_actions()
            kps, kds = robot.get_gains(joint_indices=np.asarray([0, 1, 2]))
            print(
                "X7S base smoke drive state: "
                f"applied_target={np.asarray(applied.joint_positions)[0, :3].tolist()}, "
                f"kp={np.asarray(kps)[0].tolist()}, kd={np.asarray(kds)[0].tolist()}, "
                f"max_effort={np.asarray(robot.get_max_efforts(joint_indices=np.asarray([0, 1, 2])))[0].tolist()}, "
                f"effort={np.asarray(robot.get_applied_joint_efforts(joint_indices=np.asarray([0, 1, 2])))[0].tolist()}",
                flush=True,
            )
            if (
                (SMOKE_BASE_MOTION and base_motion < 0.02)
                or arm_motion < 0.005
                or arm_tracking_error > 0.5
            ):
                raise RuntimeError("X7S smoke test did not produce expected motion")
        print(f"X7S Mobile Drilling Task stopped after {frames} frames.", flush=True)
    finally:
        if subscription is not None:
            subscription = None
        if quest is not None:
            quest.close()
        if world is not None:
            world.stop()


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        traceback.print_exc()
        simulation_app.close()
        raise
    else:
        simulation_app.close()
