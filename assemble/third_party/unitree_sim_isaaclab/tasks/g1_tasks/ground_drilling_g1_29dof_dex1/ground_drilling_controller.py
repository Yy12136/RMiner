"""Tracked-base controls, drill reset, and ground-contact success checking."""

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
from pxr import Gf, UsdGeom

import carb.input
import omni.appwindow
import omni.usd
import isaaclab.envs.mdp as base_mdp
from isaacsim.core.prims import RigidPrim

from tasks.g1_tasks.tabletop_grasp_g1_29dof_dex1.tabletop_grasp_g1_29dof_dex1_env_cfg import (
    ASSETS_ROOT,
    G1_MOUNT_LOCAL_ORIENTATION,
    W1_UPPER_MOUNT_LOCAL,
)
from .ground_drilling_g1_29dof_dex1_env_cfg import (
    DRILL_ORIENTATION,
    DRILL_POSITION,
    DRILL_RANDOM_XY_LIMIT,
    DRILL_ROOT_Z,
    DRILL_WORKING_TIP_LOCAL_Z,
    TASK_TRACKED_BASE_X,
    TASK_TRACKED_BASE_Y,
    TASK_TRACKED_BASE_YAW,
    TASK_TRACKED_LIFT_HEIGHT,
)


W1_TOOLS = ASSETS_ROOT / "robot/Forklift_W1_Final/tools"
SUCCESS_MARKER_PATH = Path(
    os.environ.get(
        "G1_TASK_SUCCESS_FILE",
        f"/tmp/g1_ground_drilling_{os.getuid()}.success",
    )
)
SUCCESS_STATUS_PATH = SUCCESS_MARKER_PATH.with_suffix(".status.json")

LIFTED_ROOT_DELTA_Z = 0.12
MIN_SUCCESS_TIP_Z = -0.002
MAX_SUCCESS_TIP_Z = 0.020
MAX_AXIS_TILT_DEG = 15.0
SUCCESS_HOLD_SECONDS = 0.30

if str(W1_TOOLS) not in sys.path:
    sys.path.insert(0, str(W1_TOOLS))
from differential_drive import DifferentialTrackKinematics  # noqa: E402


class GroundDrillingController:
    """Keep Task 2 controls while managing the floor-level drill task."""

    ROOT_PATH = "/World/TrackedBase"
    VEHICLE_PATH = f"{ROOT_PATH}/Vehicle"
    CARRIAGE_PATH = f"{ROOT_PATH}/LiftCarriage"

    def __init__(self, env) -> None:
        self.env = env
        self.robot = env.scene["robot"]
        self.drill = env.scene["drill"]
        self.device = env.device
        self.stage = omni.usd.get_context().get_stage()

        self.x = float(TASK_TRACKED_BASE_X)
        self.y = float(TASK_TRACKED_BASE_Y)
        self.yaw = float(TASK_TRACKED_BASE_YAW)
        self.lift_height = float(TASK_TRACKED_LIFT_HEIGHT)
        self.initial = (self.x, self.y, self.yaw, self.lift_height)
        self.object_random_xy_limit = DRILL_RANDOM_XY_LIMIT
        self._object_rng = np.random.default_rng()
        self.success_latched = False
        self.drill_has_been_lifted = False
        self.success_hold_time = 0.0
        self.success_tip_z = float("inf")
        self.success_axis_tilt_deg = 180.0
        self._last_success_diagnostic_time = 0.0
        SUCCESS_STATUS_PATH.unlink(missing_ok=True)
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
            name="ground_drilling_tracked_vehicle",
            reset_xform_properties=False,
        )
        self.carriage_body = RigidPrim(
            prim_paths_expr=self.CARRIAGE_PATH,
            name="ground_drilling_lift_carriage",
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
            "[ground-drilling] Controller ready. W/S drive, A/D turn, "
            "T/Y lift, R resets robot and drill."
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
            window = omni.appwindow.get_default_app_window()
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
            print(f"[ground-drilling] Keyboard input unavailable: {error}")

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
        base = (
            math.cos(0.5 * self.yaw),
            0.0,
            0.0,
            math.sin(0.5 * self.yaw),
        )
        bw, bx, by, bz = base
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
        half_yaw = 0.5 * self.yaw
        orientation = np.asarray(
            [math.cos(half_yaw), 0.0, 0.0, math.sin(half_yaw)],
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
        self.apply()
        self._update_success_checker(dt)

    def reset(self) -> None:
        self.x, self.y, self.yaw, self.lift_height = self.initial
        self.track_drive.reset()
        self._write_track_state(0.0, 0.0)
        self.apply()
        print("[ground-drilling] Base pose and lift reset.")

    def reset_everything(self) -> None:
        env_ids = torch.arange(self.env.num_envs, device=self.device)
        base_mdp.reset_scene_to_default(self.env, env_ids)
        self.reset()
        self.reset_task_objects()
        print(
            "[ground-drilling] R reset complete: robot, base/lift, and drill restored."
        )

    def reset_task_objects(self) -> None:
        self._clear_success()
        env_ids = torch.arange(self.env.num_envs, device=self.device)
        offset = self._object_rng.uniform(
            -self.object_random_xy_limit,
            self.object_random_xy_limit,
            size=2,
        )
        drill_state = self.drill.data.default_root_state[env_ids].clone()
        drill_state[:, :3] += self.env.scene.env_origins[env_ids]
        drill_state[:, 0] = (
            self.env.scene.env_origins[env_ids, 0]
            + DRILL_POSITION[0]
            + float(offset[0])
        )
        drill_state[:, 1] = (
            self.env.scene.env_origins[env_ids, 1]
            + DRILL_POSITION[1]
            + float(offset[1])
        )
        drill_state[:, 2] = (
            self.env.scene.env_origins[env_ids, 2] + DRILL_POSITION[2]
        )
        drill_state[:, 3:7] = torch.tensor(
            DRILL_ORIENTATION, dtype=torch.float32, device=self.device
        )
        self.drill.write_root_pose_to_sim(drill_state[:, :7], env_ids=env_ids)
        self.drill.write_root_velocity_to_sim(
            torch.zeros((len(env_ids), 6), dtype=torch.float32, device=self.device),
            env_ids=env_ids,
        )
        print(
            "[ground-drilling] Drill reset horizontal with tip forward: "
            f"dx={100.0 * offset[0]:+.1f} cm, "
            f"dy={100.0 * offset[1]:+.1f} cm."
        )

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
    def _rotate_vector_by_quaternion(
        cls, quaternion: np.ndarray, vector: np.ndarray
    ) -> np.ndarray:
        quaternion = quaternion / np.linalg.norm(quaternion)
        conjugate = quaternion.copy()
        conjugate[1:] *= -1.0
        return cls._quaternion_multiply(
            cls._quaternion_multiply(
                quaternion, np.concatenate(([0.0], vector))
            ),
            conjugate,
        )[1:]

    def _drill_geometry(self) -> tuple[float, float, float]:
        pose = (
            self.drill.data.root_pose_w[0]
            .detach()
            .cpu()
            .numpy()
            .astype(np.float64)
        )
        axis = self._rotate_vector_by_quaternion(
            pose[3:7], np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
        )
        axis_z = float(np.clip(axis[2], -1.0, 1.0))
        axis_tilt_deg = math.degrees(math.acos(axis_z))
        working_tip = pose[:3] + DRILL_WORKING_TIP_LOCAL_Z * axis
        return axis_tilt_deg, float(working_tip[2]), float(pose[2])

    def _update_success_checker(self, dt: float) -> None:
        axis_tilt_deg, tip_z, root_z = self._drill_geometry()
        self.success_axis_tilt_deg = axis_tilt_deg
        self.success_tip_z = tip_z
        # The drill now starts horizontally with its working end already above
        # the floor. Require the whole object to be lifted 12 cm above that
        # reset height before an eventual vertical ground press can succeed.
        if root_z >= DRILL_ROOT_Z + LIFTED_ROOT_DELTA_Z:
            self.drill_has_been_lifted = True

        tilt_ok = axis_tilt_deg <= MAX_AXIS_TILT_DEG
        tip_ok = MIN_SUCCESS_TIP_Z <= tip_z <= MAX_SUCCESS_TIP_Z
        candidate = self.drill_has_been_lifted and tilt_ok and tip_ok
        self.success_hold_time = (
            self.success_hold_time + dt if candidate else 0.0
        )
        success = self.success_hold_time >= SUCCESS_HOLD_SECONDS

        now = time.monotonic()
        if now - self._last_success_diagnostic_time >= 0.5:
            self._last_success_diagnostic_time = now
            SUCCESS_STATUS_PATH.write_text(
                json.dumps(
                    {
                        "success": bool(success),
                        "drill_has_been_lifted": bool(self.drill_has_been_lifted),
                        "working_tip_z_m": tip_z,
                        "drill_root_z_m": root_z,
                        "lifted_root_threshold_m": (
                            DRILL_ROOT_Z + LIFTED_ROOT_DELTA_Z
                        ),
                        "minimum_success_tip_z_m": MIN_SUCCESS_TIP_Z,
                        "maximum_success_tip_z_m": MAX_SUCCESS_TIP_Z,
                        "axis_tilt_deg": axis_tilt_deg,
                        "maximum_axis_tilt_deg": MAX_AXIS_TILT_DEG,
                        "hold_seconds": self.success_hold_time,
                        "required_hold_seconds": SUCCESS_HOLD_SECONDS,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            if self.drill_has_been_lifted:
                print(
                    "[ground-drilling] Success check: "
                    f"lifted={'YES' if self.drill_has_been_lifted else 'NO'}, "
                    f"tip_z={tip_z:.3f} m "
                    f"({'OK' if tip_ok else 'OUT'}), "
                    f"tilt={axis_tilt_deg:.1f}/{MAX_AXIS_TILT_DEG:.1f} deg "
                    f"({'OK' if tilt_ok else 'HIGH'}), "
                    f"hold={self.success_hold_time:.2f}/{SUCCESS_HOLD_SECONDS:.2f} s."
                )
        if not success or self.success_latched:
            return

        self.success_latched = True
        SUCCESS_MARKER_PATH.write_text(
            json.dumps(
                {
                    "success": True,
                    "working_tip_z_m": tip_z,
                    "axis_tilt_deg": axis_tilt_deg,
                    "hold_seconds": self.success_hold_time,
                    "timestamp": datetime.now().astimezone().isoformat(),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print("\n" + "=" * 72)
        print("SUCCESS")
        print(
            "[ground-drilling] Drill was lifted and returned upright to ground: "
            f"tip_z={tip_z:.4f} m, axis_tilt={axis_tilt_deg:.2f} deg, "
            f"stable={self.success_hold_time:.2f} s."
        )
        print(
            "[ground-drilling] Simulation, commands, and new camera frames "
            "are frozen. Press right [A] to save or terminal [s] to discard."
        )
        print("=" * 72 + "\n")

    @property
    def is_success_frozen(self) -> bool:
        return self.success_latched

    def _clear_success(self) -> None:
        was_latched = self.success_latched
        self.success_latched = False
        self.drill_has_been_lifted = False
        self.success_hold_time = 0.0
        self.success_tip_z = float("inf")
        self.success_axis_tilt_deg = 180.0
        SUCCESS_MARKER_PATH.unlink(missing_ok=True)
        SUCCESS_STATUS_PATH.unlink(missing_ok=True)
        if was_latched:
            print("[ground-drilling] Success freeze cleared by reset.")

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
