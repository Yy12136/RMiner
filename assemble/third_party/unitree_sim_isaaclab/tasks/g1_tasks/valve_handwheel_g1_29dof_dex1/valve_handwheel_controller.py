"""Tracked-base controls, fixed assembly reset, and rotation success checking."""

from __future__ import annotations

import json
import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from pxr import Gf, UsdGeom, UsdPhysics

import carb.input
import omni.usd
import isaaclab.envs.mdp as base_mdp
from isaacsim.core.prims import RigidPrim

try:
    import omni.appwindow as omni_appwindow
except ModuleNotFoundError:
    omni_appwindow = None

from tasks.g1_tasks.tabletop_grasp_g1_29dof_dex1.tabletop_grasp_g1_29dof_dex1_env_cfg import (
    ASSETS_ROOT,
    G1_MOUNT_LOCAL_ORIENTATION,
    W1_UPPER_MOUNT_LOCAL,
)
from tasks.g1_tasks.benchmark_protocol import BenchmarkProtocol
from .valve_handwheel_g1_29dof_dex1_env_cfg import (
    HANDWHEEL_JOINT_ANCHOR_Z,
    HANDWHEEL_POSITION,
    TASK_TRACKED_BASE_X as TRACKED_BASE_X,
    TASK_TRACKED_BASE_Y as TRACKED_BASE_Y,
    TASK_TRACKED_BASE_YAW as TRACKED_BASE_YAW,
    TASK_TRACKED_LIFT_HEIGHT as TRACKED_LIFT_HEIGHT,
    VALVE_POSITION,
    WALL_INWARD_NORMAL,
)


W1_TOOLS = ASSETS_ROOT / "robot/Forklift_W1_Final/tools"
SUCCESS_MARKER_PATH = Path(
    os.environ.get(
        "G1_TASK_SUCCESS_FILE",
        f"/tmp/g1_valve_handwheel_{os.getuid()}.success",
    )
)
FAILURE_MARKER_PATH = Path(
    os.environ.get(
        "G1_TASK_FAILURE_FILE",
        f"/tmp/g1_valve_handwheel_{os.getuid()}.failure",
    )
)

REQUIRED_ROTATION = math.radians(60.0)
RESET_SETTLE_SECONDS = 1.0
# The wheel hangs on a frictionless revolute joint, so the reset teleport can
# leave residual solver spin. The reset handshake therefore stays open while
# the wheel drifts: its velocity is re-zeroed each frame until it has been
# still for SETTLE_STILL_FRAMES, after which any rotation is attributable to
# the policy.
SETTLE_STILL_THRESHOLD_RAD_S = 0.05
SETTLE_STILL_FRAMES = 15
# Stay below the bridge's 25 s reset timeout so a wheel that never settles
# surfaces as an infrastructure error and the evaluator retries the seed.
SETTLE_HANDSHAKE_BUDGET_SECONDS = 20.0
WAIST_PITCH_MIN = 0.0
WAIST_PITCH_MAX = math.radians(25.0)
WAIST_PITCH_SPEED = math.radians(60.0)
RESET_RANDOM_WALL_LIMIT = 0.05
WALL_LATERAL_AXIS = np.asarray(
    (-WALL_INWARD_NORMAL[1], WALL_INWARD_NORMAL[0], 0.0),
    dtype=np.float64,
)

if str(W1_TOOLS) not in sys.path:
    sys.path.insert(0, str(W1_TOOLS))
from differential_drive import DifferentialTrackKinematics  # noqa: E402


class ValveHandwheelController:
    """Preserve existing G1 controls and manage the valve-handwheel task."""

    ROOT_PATH = "/World/TrackedBase"
    VEHICLE_PATH = f"{ROOT_PATH}/Vehicle"
    CARRIAGE_PATH = f"{ROOT_PATH}/LiftCarriage"

    def __init__(self, env) -> None:
        self.env = env
        self.robot = env.scene["robot"]
        self.handwheel = env.scene["handwheel"]
        self.valve = env.scene["valve"]
        self.device = env.device
        self.stage = omni.usd.get_context().get_stage()

        self.x = float(TRACKED_BASE_X)
        self.y = float(TRACKED_BASE_Y)
        self.yaw = float(TRACKED_BASE_YAW)
        self.lift_height = float(TRACKED_LIFT_HEIGHT)
        self.waist_pitch_target = WAIST_PITCH_MIN
        self.initial = (self.x, self.y, self.yaw, self.lift_height)
        self.success_latched = False
        # reset_task_objects() clears terminal state during construction, so
        # initialize failure state before that first reset.
        self._failure_latched = False
        self._failure_reason = None
        self.success_rotation = 0.0
        self._last_twist_angle: float | None = None
        self._accumulated_rotation = 0.0
        seed = int(getattr(getattr(env, "cfg", None), "seed", 0) or 0)
        self._object_rng = np.random.default_rng(seed)
        self._benchmark = BenchmarkProtocol()
        self.held: set[carb.input.KeyboardInput] = set()
        self._input = None
        self._keyboard = None
        self._subscription = None

        root = self.stage.GetPrimAtPath(self.ROOT_PATH)
        vehicle = self.stage.GetPrimAtPath(self.VEHICLE_PATH)
        carriage = self.stage.GetPrimAtPath(self.CARRIAGE_PATH)
        if not root or not vehicle or not carriage:
            raise RuntimeError(
                "Tracked base hierarchy is incomplete: "
                f"root={bool(root)}, vehicle={bool(vehicle)}, "
                f"carriage={bool(carriage)}"
            )

        self.root_translate_op = self._get_or_add_op(
            UsdGeom.Xformable(root), "translate"
        )
        self.root_orient_op = self._get_or_add_op(
            UsdGeom.Xformable(root), "orient"
        )
        self.carriage_translate_op = self._get_or_add_op(
            UsdGeom.Xformable(carriage), "translate"
        )

        self.track_attributes = {
            name: vehicle.GetAttribute(f"trackDrive:{name}")
            for name in (
                "leftVelocity",
                "rightVelocity",
                "actualLeftVelocity",
                "actualRightVelocity",
                "leftDistance",
                "rightDistance",
                "trackSeparation",
                "maxTrackSpeed",
                "maxTrackAcceleration",
                "headingOffsetRadians",
            )
        }
        missing = [name for name, attr in self.track_attributes.items() if not attr]
        if missing:
            raise RuntimeError(f"Tracked-base attributes are missing: {missing}")

        self.lift_attributes = {
            name: carriage.GetAttribute(f"liftDrive:{name}")
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
            raise RuntimeError(f"Tracked-lift attributes are missing: {missing}")

        self.lift_lower = float(self.lift_attributes["lowerLimit"].Get())
        self.lift_upper = float(self.lift_attributes["upperLimit"].Get())
        self.lift_speed = float(self.lift_attributes["maxSpeed"].Get())
        self.track_drive = DifferentialTrackKinematics(
            track_separation=float(self.track_attributes["trackSeparation"].Get()),
            max_track_speed=float(self.track_attributes["maxTrackSpeed"].Get()),
            max_track_acceleration=float(
                self.track_attributes["maxTrackAcceleration"].Get()
            ),
            heading_offset=float(
                self.track_attributes["headingOffsetRadians"].Get()
            ),
        )

        self.vehicle_body = RigidPrim(
            prim_paths_expr=self.VEHICLE_PATH,
            name="valve_handwheel_tracked_vehicle",
            reset_xform_properties=False,
        )
        self.carriage_body = RigidPrim(
            prim_paths_expr=self.CARRIAGE_PATH,
            name="valve_handwheel_lift_carriage",
            reset_xform_properties=False,
        )
        self.vehicle_body.initialize()
        self.carriage_body.initialize()
        if not self.vehicle_body.is_physics_handle_valid():
            raise RuntimeError("Tracked vehicle physics handle did not initialize")
        if not self.carriage_body.is_physics_handle_valid():
            raise RuntimeError("Tracked lift physics handle did not initialize")

        self._subscribe_keyboard()
        self.apply()
        self.reset_task_objects()
        print(
            "[valve-handwheel] Controller ready. W/S drive, A/D turn, "
            "T/Y lift, I bends forward, K returns upright, "
            "R resets robot, handwheel, and valve."
        )

    @staticmethod
    def _get_or_add_op(xform: UsdGeom.Xformable, kind: str):
        name = f"xformOp:{kind}"
        found = next(
            (op for op in xform.GetOrderedXformOps() if op.GetOpName() == name),
            None,
        )
        if found is not None:
            return found
        if kind == "translate":
            return xform.AddTranslateOp()
        if kind == "orient":
            return xform.AddOrientOp()
        raise ValueError(kind)

    def _subscribe_keyboard(self) -> None:
        try:
            if omni_appwindow is None:
                print(
                    "[valve-handwheel] Keyboard input disabled in headless mode."
                )
                return
            window = omni_appwindow.get_default_app_window()
            if window is None:
                return
            self._input = carb.input.acquire_input_interface()
            self._keyboard = window.get_keyboard()

            def on_key(event, *_args):
                key = event.input
                if event.type in (
                    carb.input.KeyboardEventType.KEY_PRESS,
                    carb.input.KeyboardEventType.KEY_REPEAT,
                ):
                    self.held.add(key)
                    if (
                        key == carb.input.KeyboardInput.R
                        and event.type == carb.input.KeyboardEventType.KEY_PRESS
                    ):
                        self.reset_everything()
                elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
                    self.held.discard(key)
                return True

            self._subscription = self._input.subscribe_to_keyboard_events(
                self._keyboard, on_key
            )
        except Exception as error:
            print(f"[valve-handwheel] Keyboard input unavailable: {error}")

    def _write_track_state(self, left: float, right: float) -> None:
        values = {
            "leftVelocity": left,
            "rightVelocity": right,
            "actualLeftVelocity": self.track_drive.left_velocity,
            "actualRightVelocity": self.track_drive.right_velocity,
            "leftDistance": self.track_drive.left_distance,
            "rightDistance": self.track_drive.right_distance,
        }
        for name, value in values.items():
            self.track_attributes[name].Set(float(value))

    def _robot_pose(self) -> tuple[np.ndarray, np.ndarray]:
        mount_x, mount_y, mount_z = W1_UPPER_MOUNT_LOCAL
        cosine = math.cos(self.yaw)
        sine = math.sin(self.yaw)
        position = np.asarray(
            [
                self.x + cosine * mount_x - sine * mount_y,
                self.y + sine * mount_x + cosine * mount_y,
                mount_z + self.lift_height,
            ],
            dtype=np.float32,
        )
        bw, bx, by, bz = (
            math.cos(0.5 * self.yaw),
            0.0,
            0.0,
            math.sin(0.5 * self.yaw),
        )
        mw, mx, my, mz = G1_MOUNT_LOCAL_ORIENTATION
        orientation = np.asarray(
            [
                bw * mw - bx * mx - by * my - bz * mz,
                bw * mx + bx * mw + by * mz - bz * my,
                bw * my - bx * mz + by * mw + bz * mx,
                bw * mz + bx * my - by * mx + bz * mw,
            ],
            dtype=np.float32,
        )
        return position, orientation

    def apply(self) -> None:
        orientation = np.asarray(
            [math.cos(0.5 * self.yaw), 0.0, 0.0, math.sin(0.5 * self.yaw)],
            dtype=np.float32,
        )
        self.root_translate_op.Set(Gf.Vec3d(self.x, self.y, 0.0))
        self.root_orient_op.Set(
            Gf.Quatd(
                float(orientation[0]),
                Gf.Vec3d(0.0, 0.0, float(orientation[3])),
            )
        )
        self.carriage_translate_op.Set(Gf.Vec3d(0.0, 0.0, self.lift_height))
        self.vehicle_body.set_world_poses(
            positions=torch.tensor(
                [[self.x, self.y, 0.0]], dtype=torch.float32, device=self.device
            ),
            orientations=torch.as_tensor(
                orientation[np.newaxis, :], dtype=torch.float32, device=self.device
            ),
        )
        self.carriage_body.set_world_poses(
            positions=torch.tensor(
                [[self.x, self.y, self.lift_height]],
                dtype=torch.float32,
                device=self.device,
            ),
            orientations=torch.as_tensor(
                orientation[np.newaxis, :], dtype=torch.float32, device=self.device
            ),
        )
        robot_position, robot_orientation = self._robot_pose()
        root_pose = torch.as_tensor(
            np.concatenate((robot_position, robot_orientation))[np.newaxis, :],
            dtype=torch.float32,
            device=self.device,
        )
        self.robot.write_root_pose_to_sim(root_pose)
        self.robot.write_root_velocity_to_sim(
            torch.zeros((1, 6), dtype=torch.float32, device=self.device)
        )
        self.lift_attributes["height"].Set(float(self.lift_height))
        self.lift_attributes["targetHeight"].Set(float(self.lift_height))

    def update(self, dt: float) -> None:
        if self.success_latched:
            return
        if self._benchmark.active:
            self.held.clear()
        dt = float(np.clip(dt, 1.0e-4, 0.1))
        throttle = float(
            (carb.input.KeyboardInput.W in self.held)
            - (carb.input.KeyboardInput.S in self.held)
        )
        steering = float(
            (carb.input.KeyboardInput.A in self.held)
            - (carb.input.KeyboardInput.D in self.held)
        )
        left_axis = throttle - steering
        right_axis = throttle + steering
        normalization = max(1.0, abs(left_axis), abs(right_axis))
        left_command = self.track_drive.max_track_speed * left_axis / normalization
        right_command = self.track_drive.max_track_speed * right_axis / normalization
        self.x, self.y, self.yaw = self.track_drive.step(
            self.x,
            self.y,
            self.yaw,
            left_command,
            right_command,
            dt,
        )
        self._write_track_state(left_command, right_command)
        lift_direction = float(
            (carb.input.KeyboardInput.T in self.held)
            - (carb.input.KeyboardInput.Y in self.held)
        )
        self.lift_height = float(
            np.clip(
                self.lift_height + lift_direction * self.lift_speed * dt,
                self.lift_lower,
                self.lift_upper,
            )
        )
        waist_direction = float(
            (carb.input.KeyboardInput.I in self.held)
            - (carb.input.KeyboardInput.K in self.held)
        )
        self.waist_pitch_target = float(
            np.clip(
                self.waist_pitch_target
                + waist_direction * WAIST_PITCH_SPEED * dt,
                WAIST_PITCH_MIN,
                WAIST_PITCH_MAX,
            )
        )
        self.apply()
        self._update_settle_gate()
        self._update_success_checker(dt)
        self._benchmark.tick()

    def reset(self) -> None:
        self.x, self.y, self.yaw, self.lift_height = self.initial
        self.waist_pitch_target = WAIST_PITCH_MIN
        self.track_drive.reset()
        self._write_track_state(0.0, 0.0)
        self.apply()
        print("[valve-handwheel] Base pose, lift, and waist pitch reset.")

    def reset_everything(self) -> None:
        # Do not call reset_scene_to_default here: it attempts to wake the
        # kinematic valve body and produces a PhysX error. The bridge has
        # already driven the arm home; the runtime reset below owns the base,
        # handwheel, valve and joint anchor deterministically.
        self.reset()
        self.reset_task_objects()
        print(
            "[valve-handwheel] R reset complete: robot, base/lift, "
            "handwheel, and fixed valve restored."
        )

    def _reset_managed_object(
        self, asset, position_offset, env_ids, *, clear_velocity: bool = True
    ) -> None:
        root_state = asset.data.default_root_state[env_ids].clone()
        root_state[:, :3] += self.env.scene.env_origins[env_ids]
        root_state[:, :3] += torch.as_tensor(
            position_offset,
            dtype=torch.float32,
            device=self.device,
        )
        asset.write_root_pose_to_sim(root_state[:, :7], env_ids=env_ids)
        if clear_velocity:
            asset.write_root_velocity_to_sim(
                torch.zeros(
                    (len(env_ids), 6), dtype=torch.float32, device=self.device
                ),
                env_ids=env_ids,
            )

    def _reset_kinematic_object(self, asset, position_offset, env_ids) -> None:
        root_state = asset.data.default_root_state[env_ids].clone()
        root_state[:, :3] += self.env.scene.env_origins[env_ids]
        root_state[:, :3] += torch.as_tensor(
            position_offset, dtype=torch.float32, device=self.device
        )
        root_pose_wxyz = root_state[:, :7]
        if str(self.device).startswith("cpu"):
            # PhysX CPU implements true kinematic targets. Convert IsaacLab's
            # WXYZ quaternion to the tensor API's XYZW convention.
            root_pose_xyzw = torch.cat(
                (root_pose_wxyz[:, :3], root_pose_wxyz[:, 4:7], root_pose_wxyz[:, 3:4]),
                dim=-1,
            )
            asset.root_physx_view.set_kinematic_targets(root_pose_xyzw, indices=env_ids)
        else:
            # Isaac Sim 5.1 does not implement kinematic targets in its GPU
            # rigid-body backend. Preserve manual CUDA simulator compatibility.
            asset.write_root_pose_to_sim(root_pose_wxyz, env_ids=env_ids)

    def _set_handwheel_joint_anchor_offset(
        self, lateral_offset: float, vertical_offset: float
    ) -> None:
        for env_index in range(self.env.num_envs):
            joint_path = (
                f"/World/envs/env_{env_index}/Valve/HandwheelJoint"
            )
            joint = UsdPhysics.Joint(self.stage.GetPrimAtPath(joint_path))
            if not joint:
                raise RuntimeError(f"Handwheel joint is missing: {joint_path}")
            joint.GetLocalPos0Attr().Set(
                Gf.Vec3f(
                    float(lateral_offset),
                    float(vertical_offset),
                    float(HANDWHEEL_JOINT_ANCHOR_Z),
                )
            )

    def reset_task_objects(self) -> None:
        self._clear_success()
        request = self._benchmark.consume_request()
        if request is not None:
            self._object_rng = np.random.default_rng(int(request["seed"]))
        env_ids = torch.arange(self.env.num_envs, device=self.device)
        lateral_offset, vertical_offset = self._object_rng.uniform(
            -RESET_RANDOM_WALL_LIMIT,
            RESET_RANDOM_WALL_LIMIT,
            size=2,
        )
        position_offset = (
            WALL_LATERAL_AXIS * lateral_offset
            + np.asarray((0.0, 0.0, vertical_offset), dtype=np.float64)
        )
        # The live PhysX joint keeps its initial wall anchor when its kinematic
        # valve body is teleported. Shift the body-0 joint frame by the same
        # wall-local lateral/vertical offset so the constrained wheel follows
        # the complete randomized assembly instead of snapping back.
        self._set_handwheel_joint_anchor_offset(lateral_offset, vertical_offset)
        self._reset_kinematic_object(self.valve, position_offset, env_ids)
        self._reset_managed_object(self.handwheel, position_offset, env_ids)
        print(
            "[valve-handwheel] Wall assembly reset randomization: "
            f"left/right={100.0 * lateral_offset:+.1f} cm, "
            f"up/down={100.0 * vertical_offset:+.1f} cm; "
            f"world_offset=({position_offset[0]:+.4f}, "
            f"{position_offset[1]:+.4f}, {position_offset[2]:+.4f}) m."
        )
        if request is not None:
            # The handshake is held open by _update_settle_gate until the
            # wheel has stopped drifting; the bridge's reset timeout turns a
            # never-still wheel into an infrastructure error for the
            # evaluator to retry on the same seed.
            self._pending_settle = (
                request,
                {
                    "wall_lateral_offset_m": float(lateral_offset),
                    "wall_vertical_offset_m": float(vertical_offset),
                    "world_offset_xyz_m": position_offset.astype(float).tolist(),
                    "success_metric": "success_60deg",
                    "required_rotation_deg": 60.0,
                    "protocol": "settled_rotation_v3",
                },
            )
            self._settle_started_at = time.monotonic()

    @staticmethod
    def _quaternion_multiply(left: np.ndarray, right: np.ndarray) -> np.ndarray:
        lw, lx, ly, lz = left
        rw, rx, ry, rz = right
        return np.asarray(
            [
                lw * rw - lx * rx - ly * ry - lz * rz,
                lw * rx + lx * rw + ly * rz - lz * ry,
                lw * ry - lx * rz + ly * rw + lz * rx,
                lw * rz + lx * ry - ly * rx + lz * rw,
            ],
            dtype=np.float64,
        )

    @classmethod
    def _pose_relative_to(
        cls, reference_pose: np.ndarray, target_pose: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        reference_position = reference_pose[:3].astype(np.float64)
        reference_orientation = reference_pose[3:7].astype(np.float64)
        target_position = target_pose[:3].astype(np.float64)
        target_orientation = target_pose[3:7].astype(np.float64)
        reference_inverse = reference_orientation.copy()
        reference_inverse[1:] *= -1.0
        reference_inverse /= np.dot(reference_orientation, reference_orientation)
        delta = target_position - reference_position
        relative_position = cls._quaternion_multiply(
            cls._quaternion_multiply(
                reference_inverse, np.concatenate(([0.0], delta))
            ),
            reference_orientation,
        )[1:]
        relative_orientation = cls._quaternion_multiply(
            reference_inverse, target_orientation
        )
        relative_orientation /= np.linalg.norm(relative_orientation)
        return relative_position, relative_orientation

    @staticmethod
    def _twist_angle_about_z(quaternion: np.ndarray) -> float:
        norm = math.hypot(float(quaternion[0]), float(quaternion[3]))
        if norm < 1.0e-9:
            return 0.0
        w = float(quaternion[0]) / norm
        z = float(quaternion[3]) / norm
        angle = 2.0 * math.atan2(z, w)
        return (angle + math.pi) % (2.0 * math.pi) - math.pi

    def _current_relative_twist(self) -> float:
        valve_pose = (
            self.valve.data.root_pose_w[0]
            .detach()
            .cpu()
            .numpy()
            .astype(np.float64)
        )
        wheel_pose = (
            self.handwheel.data.root_pose_w[0]
            .detach()
            .cpu()
            .numpy()
            .astype(np.float64)
        )
        _, wheel_orientation = self._pose_relative_to(
            valve_pose, wheel_pose
        )
        return self._twist_angle_about_z(wheel_orientation)

    @staticmethod
    def _rotate_vector(quaternion: np.ndarray, vector: np.ndarray) -> np.ndarray:
        quaternion = np.asarray(quaternion, dtype=np.float64)
        quaternion = quaternion / max(np.linalg.norm(quaternion), 1.0e-12)
        q_xyz = quaternion[1:]
        vector = np.asarray(vector, dtype=np.float64)
        return vector + 2.0 * (
            quaternion[0] * np.cross(q_xyz, vector)
            + np.cross(q_xyz, np.cross(q_xyz, vector))
        )

    def _handwheel_axis_and_center(self) -> tuple[np.ndarray, np.ndarray]:
        valve_pose = (
            self.valve.data.root_pose_w[0].detach().cpu().numpy().astype(np.float64)
        )
        wheel_pose = (
            self.handwheel.data.root_pose_w[0].detach().cpu().numpy().astype(np.float64)
        )
        axis = self._rotate_vector(valve_pose[3:7], np.asarray((0.0, 0.0, 1.0)))
        axis /= max(np.linalg.norm(axis), 1.0e-12)
        return axis, wheel_pose[:3]

    def _wheel_axis_angular_speed(self) -> float:
        """Signed-axis magnitude of the wheel's world angular velocity."""
        axis, _ = self._handwheel_axis_and_center()
        velocity = (
            self.handwheel.data.root_vel_w[0]
            .detach()
            .cpu()
            .numpy()
            .astype(np.float64)
        )
        return float(abs(np.dot(velocity[3:6], axis)))

    def _update_settle_gate(self) -> None:
        """Hold the reset handshake until the wheel stops drifting.

        The frictionless revolute joint picks up residual solver velocity
        during the reset teleport, which used to rotate the wheel before the
        policy ever acted. While a settle is pending, the wheel velocity is
        re-zeroed every frame until it has stayed under the still threshold
        for SETTLE_STILL_FRAMES consecutive frames; only then is the settle
        handshake handed to BenchmarkProtocol. If the wheel never settles
        within the budget, the handshake simply never completes and the
        bridge's reset timeout surfaces the episode as an infrastructure
        error for the evaluator to retry on the same seed.
        """
        if self._pending_settle is None:
            return
        request, details = self._pending_settle
        elapsed = time.monotonic() - self._settle_started_at
        speed = self._wheel_axis_angular_speed()
        self._settle_last_speed = speed
        if speed > SETTLE_STILL_THRESHOLD_RAD_S:
            self.handwheel.write_root_velocity_to_sim(
                torch.zeros(
                    (self.env.num_envs, 6),
                    dtype=torch.float32,
                    device=self.device,
                )
            )
            self._settle_zeroings += 1
            self._settle_still_frames = 0
        elif elapsed >= RESET_SETTLE_SECONDS:
            self._settle_still_frames += 1
        if elapsed - self._settle_last_diag >= 5.0:
            self._settle_last_diag = elapsed
            print(
                "[valve-handwheel] Settle gate: "
                f"t={elapsed:.1f}s wheel_speed={speed:.3f} rad/s, "
                f"still_frames={self._settle_still_frames}/{SETTLE_STILL_FRAMES}, "
                f"zeroings={self._settle_zeroings}."
            )
            if elapsed >= SETTLE_HANDSHAKE_BUDGET_SECONDS and not self._settle_budget_warned:
                self._settle_budget_warned = True
                print(
                    "[valve-handwheel] Settle budget exceeded while the wheel "
                    "keeps drifting; the reset will time out and retry."
                )
            return
        if speed > SETTLE_STILL_THRESHOLD_RAD_S or elapsed < RESET_SETTLE_SECONDS:
            return
        if self._settle_still_frames < SETTLE_STILL_FRAMES:
            return
        payload = dict(details)
        payload["settle_zeroings"] = int(self._settle_zeroings)
        payload["settle_still_seconds"] = float(elapsed)
        self._benchmark.begin_settle(request, payload, steps=1)
        self._pending_settle = None
        print(
            "[valve-handwheel] Settle gate passed: wheel still for "
            f"{SETTLE_STILL_FRAMES} frames after {elapsed:.2f}s with "
            f"{self._settle_zeroings} velocity zeroings."
        )

    def _update_success_checker_legacy(self) -> None:
        twist_angle = self._current_relative_twist()
        if self._last_twist_angle is None:
            self._last_twist_angle = twist_angle
            return
        delta = (twist_angle - self._last_twist_angle + math.pi) % (2.0 * math.pi) - math.pi
        self._accumulated_rotation += delta
        self._last_twist_angle = twist_angle
        self.success_rotation = self._accumulated_rotation
        if abs(self._accumulated_rotation) < REQUIRED_ROTATION:
            return
        self.success_latched = True
        self._phase = "success"
        payload = {
            "success": True,
            "rotation_deg": math.degrees(self._accumulated_rotation),
            "protocol": "legacy_manual",
            "timestamp": datetime.now().astimezone().isoformat(),
        }
        self._write_terminal_marker(SUCCESS_MARKER_PATH, payload)

    def _write_terminal_marker(self, path: Path, payload: dict) -> None:
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, path)

    def _mark_success(self) -> None:
        if self.success_latched or self._failure_latched:
            return
        self.success_latched = True
        self._phase = "success"
        self.success_rotation = self._accumulated_rotation
        payload = {
            "success": True,
            "protocol": "settled_rotation_v3",
            "rotation_deg": math.degrees(self._accumulated_rotation),
            "ignored_pre_settle_rotation_deg": math.degrees(self._settle_ignored_rotation),
            "counting_seconds": float(self._counting_elapsed),
            "settle_zeroings": int(self._settle_zeroings),
            "settle_still_seconds": float(self._settle_still_seconds),
            "timestamp": datetime.now().astimezone().isoformat(),
        }
        self._write_terminal_marker(SUCCESS_MARKER_PATH, payload)
        print("\n" + "=" * 72)
        print("SUCCESS")
        print(
            "[valve-handwheel] Wheel rotated "
            f"{math.degrees(self._accumulated_rotation):.1f} deg after a "
            "verified-still reset (settled_rotation_v3)."
        )
        print("=" * 72 + "\n")

    def _update_success_checker(self, dt: float) -> None:
        if not self._benchmark.active:
            self._update_success_checker_legacy()
            return
        dt = float(np.clip(dt, 1.0e-4, 0.2))
        twist_angle = self._current_relative_twist()
        if self._last_twist_angle is None:
            self._last_twist_angle = twist_angle
            return
        delta = (twist_angle - self._last_twist_angle + math.pi) % (2.0 * math.pi) - math.pi
        self._last_twist_angle = twist_angle
        self._last_wheel_angular_speed = abs(delta) / max(dt, 1.0e-6)
        if self._phase == "reset_settling":
            self._settle_elapsed += dt
            self._settle_ignored_rotation += abs(delta)
            if self._pending_settle is None:
                # The settle gate has verified a still wheel: restart from a
                # zero-twist baseline and attribute every following degree of
                # wheel motion to the policy.
                self._phase = "counting"
                self._accumulated_rotation = 0.0
                self._counting_elapsed = 0.0
                self._last_twist_angle = twist_angle
                print(
                    "[valve-handwheel] Verified-still reset complete; "
                    "counting wheel rotation (settled_rotation_v3)."
                )
            return
        self._counting_elapsed += dt
        self._accumulated_rotation += delta
        self.success_rotation = self._accumulated_rotation
        if abs(self._accumulated_rotation) >= REQUIRED_ROTATION:
            self._mark_success()

    @property
    def is_success_frozen(self) -> bool:
        return self.success_latched

    @property
    def is_terminal_frozen(self) -> bool:
        return self.success_latched or self._failure_latched

    def _clear_success(self) -> None:
        was_latched = self.success_latched or self._failure_latched
        self.success_latched = False
        self._failure_latched = False
        self._failure_reason = None
        self.success_rotation = 0.0
        self._phase = "reset_settling"
        self._settle_elapsed = 0.0
        self._settle_ignored_rotation = 0.0
        self._accumulated_rotation = 0.0
        self._counting_elapsed = 0.0
        self._pending_settle = None
        self._settle_started_at = None
        self._settle_still_frames = 0
        self._settle_zeroings = 0
        self._settle_last_speed = 0.0
        self._settle_still_seconds = 0.0
        self._settle_last_diag = 0.0
        self._settle_budget_warned = False
        self._last_twist_angle = None
        self._last_wheel_angular_speed = 0.0
        SUCCESS_MARKER_PATH.unlink(missing_ok=True)
        FAILURE_MARKER_PATH.unlink(missing_ok=True)
        for marker in (SUCCESS_MARKER_PATH, FAILURE_MARKER_PATH):
            marker.with_name(f".{marker.name}.{os.getpid()}.tmp").unlink(missing_ok=True)
        if was_latched:
            print("[valve-handwheel] Terminal state cleared by reset.")

    def get_recording_state(self) -> dict:
        """Expose the fixed valve and rotating handwheel for replay setup."""
        valve_pose = self.valve.data.root_pose_w[0].detach().cpu().numpy()
        handwheel_pose = self.handwheel.data.root_pose_w[0].detach().cpu().numpy()
        objects = {
            "valve": {"root_pose_wxyz": valve_pose.astype(np.float64).tolist()},
            "handwheel": {"root_pose_wxyz": handwheel_pose.astype(np.float64).tolist()},
        }
        return {
            "schema": "valve_handwheel_recording_state_v1",
            "task_id": "Isaac-ValveHandwheel-G129-Dex1-Joint",
            "timestamp_ns": time.time_ns(),
            "replay_initial_state": {"objects": objects},
            "objects": objects,
        }

    def close(self) -> None:
        self._clear_success()
        if self._input is not None and self._subscription is not None:
            try:
                self._input.unsubscribe_to_keyboard_events(
                    self._keyboard, self._subscription
                )
            except Exception:
                pass
        self._subscription = None
        self.held.clear()
