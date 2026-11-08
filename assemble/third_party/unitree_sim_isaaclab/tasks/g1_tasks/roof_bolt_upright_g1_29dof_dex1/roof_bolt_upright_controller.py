"""Physical-target base/lift control, synchronized reset, and success checking."""

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
import isaaclab.envs.mdp as base_mdp
import omni.appwindow
import omni.usd
from isaacsim.core.prims import RigidPrim

from tasks.g1_tasks.tabletop_grasp_g1_29dof_dex1.tabletop_grasp_g1_29dof_dex1_env_cfg import (
    ASSETS_ROOT,
    G1_MOUNT_LOCAL_ORIENTATION,
    W1_UPPER_MOUNT_LOCAL,
)
from .roof_bolt_upright_g1_29dof_dex1_env_cfg import (
    LEFT_SUPPORT_POSITION,
    RESET_RANDOM_XY_LIMIT,
    RIGHT_SUPPORT_POSITION,
    ROD_HALF_LENGTH,
    ROD_ORIENTATION,
    ROD_POSITION,
    SUPPORT_ORIENTATION,
    TASK_FORWARD_X,
    TASK_FORWARD_Y,
    TASK_ID,
    TASK_TRACKED_BASE_X,
    TASK_TRACKED_BASE_Y,
    TASK_TRACKED_BASE_YAW,
    TASK_TRACKED_LIFT_HEIGHT,
)


W1_TOOLS = ASSETS_ROOT / "robot/Forklift_W1_Final/tools"
SUCCESS_MARKER_PATH = Path(
    os.environ.get(
        "G1_TASK_SUCCESS_FILE",
        f"/tmp/g1_roof_bolt_upright_{os.getuid()}.success",
    )
)
SUCCESS_STATUS_PATH = SUCCESS_MARKER_PATH.with_suffix(".status.json")

GROUND_Z = 0.0
# The original layout required the endpoint to rise from 0.1725 m to 0.25 m.
# Preserve that 7.75 cm clearance after lifting the rod/support group by 20 cm.
LIFTED_LOWER_ENDPOINT_Z = ROD_POSITION[2] + 0.0775
MAX_AXIS_TILT_DEG = 10.0
MIN_SUCCESS_TIP_Z = -0.001
MAX_SUCCESS_TIP_Z = 0.030
SUCCESS_HOLD_SECONDS = 0.30

if str(W1_TOOLS) not in sys.path:
    sys.path.insert(0, str(W1_TOOLS))
from differential_drive import DifferentialTrackKinematics  # noqa: E402


class RoofBoltUprightController:
    """Drive the tracked mount as PhysX kinematic bodies and manage the task."""

    ROOT_PATH = "/World/TrackedBase"
    VEHICLE_PATH = f"{ROOT_PATH}/Vehicle"
    CARRIAGE_PATH = f"{ROOT_PATH}/LiftCarriage"

    def __init__(self, env) -> None:
        self.env = env
        self.robot = env.scene["robot"]
        self.rod = env.scene["rod"]
        self.left_support = env.scene["left_support"]
        self.right_support = env.scene["right_support"]
        self.device = env.device
        self.stage = omni.usd.get_context().get_stage()

        self.x = float(TASK_TRACKED_BASE_X)
        self.y = float(TASK_TRACKED_BASE_Y)
        self.yaw = float(TASK_TRACKED_BASE_YAW)
        self.lift_height = float(TASK_TRACKED_LIFT_HEIGHT)
        self.initial = (self.x, self.y, self.yaw, self.lift_height)

        seed = int(getattr(getattr(env, "cfg", None), "seed", 0) or 0)
        self._object_rng = np.random.default_rng(seed)
        self.held: set[carb.input.KeyboardInput] = set()
        self._input = None
        self._keyboard = None
        self._subscription = None
        # Non-interactive validation hook; normal launches always resolve to 0.
        self._validation_lift_direction = float(
            os.environ.get("G1_UPRIGHT_VALIDATE_LIFT_DIRECTION", "0")
        )
        if self._validation_lift_direction not in (-1.0, 0.0, 1.0):
            raise ValueError(
                "G1_UPRIGHT_VALIDATE_LIFT_DIRECTION must be -1, 0, or 1"
            )

        self.success_latched = False
        self.rod_has_been_lifted = False
        self.success_hold_time = 0.0
        self.success_axis_tilt_deg = 90.0
        self.success_lower_tip_z = float("inf")
        self._last_success_diagnostic_time = 0.0
        SUCCESS_MARKER_PATH.unlink(missing_ok=True)
        SUCCESS_STATUS_PATH.unlink(missing_ok=True)

        self._telemetry_sequence = 0
        self._left_target_velocity = 0.0
        self._right_target_velocity = 0.0
        self._lift_target_velocity = 0.0
        self._recording_last_time = time.monotonic()
        self._recording_last_lift = self.lift_height
        self._recording_lift_velocity = 0.0

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
            name="roof_bolt_upright_vehicle",
            reset_xform_properties=False,
        )
        self.carriage_body = RigidPrim(
            prim_paths_expr=self.CARRIAGE_PATH,
            name="roof_bolt_upright_lift_carriage",
            reset_xform_properties=False,
        )
        self.vehicle_body.initialize()
        self.carriage_body.initialize()
        for label, body in (
            ("vehicle", self.vehicle_body),
            ("lift carriage", self.carriage_body),
        ):
            if not body.is_physics_handle_valid():
                raise RuntimeError(f"Tracked {label} physics handle did not initialize")

        self._subscribe_keyboard()
        self._teleport_base_and_lift()
        self.reset_task_objects()
        print(
            "[roof-bolt-upright] Controller ready. W/S drive, A/D turn, "
            "T/Y synchronized lift, R resets robot, rod, and supports."
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
                    if event.type == carb.input.KeyboardEventType.KEY_PRESS:
                        if key == carb.input.KeyboardInput.T:
                            if self.lift_height >= self.lift_upper - 1.0e-5:
                                print(
                                    "[roof-bolt-upright] T: lift is already at "
                                    f"the upper limit ({self.lift_upper:+.2f} m)."
                                )
                            else:
                                print("[roof-bolt-upright] T: lift moving up.")
                        elif key == carb.input.KeyboardInput.Y:
                            if self.lift_height <= self.lift_lower + 1.0e-5:
                                print(
                                    "[roof-bolt-upright] Y: lift is already at "
                                    f"the lower limit ({self.lift_lower:+.2f} m)."
                                )
                            else:
                                print("[roof-bolt-upright] Y: lift moving down.")
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
            print(f"[roof-bolt-upright] Keyboard input unavailable: {error}")

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

    def _base_orientation(self) -> np.ndarray:
        return np.asarray(
            [math.cos(0.5 * self.yaw), 0.0, 0.0, math.sin(0.5 * self.yaw)],
            dtype=np.float32,
        )

    def _teleport_base_and_lift(self) -> None:
        """Set reset state immediately; never call this during normal motion."""

        orientation = self._base_orientation()
        self.root_translate_op.Set(Gf.Vec3d(self.x, self.y, 0.0))
        self.root_orient_op.Set(
            Gf.Quatd(
                float(orientation[0]),
                Gf.Vec3d(0.0, 0.0, float(orientation[3])),
            )
        )
        self.carriage_translate_op.Set(Gf.Vec3d(0.0, 0.0, self.lift_height))
        orientation_tensor = torch.as_tensor(
            orientation[np.newaxis, :], dtype=torch.float32, device=self.device
        )
        self.vehicle_body.set_world_poses(
            positions=torch.tensor(
                [[self.x, self.y, 0.0]], dtype=torch.float32, device=self.device
            ),
            orientations=orientation_tensor,
        )
        self.carriage_body.set_world_poses(
            positions=torch.tensor(
                [[self.x, self.y, self.lift_height]],
                dtype=torch.float32,
                device=self.device,
            ),
            orientations=orientation_tensor,
        )
        self.vehicle_body.set_velocities(
            torch.zeros((1, 6), dtype=torch.float32, device=self.device)
        )
        self.carriage_body.set_velocities(
            torch.zeros((1, 6), dtype=torch.float32, device=self.device)
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
        self._recording_last_time = time.monotonic()
        self._recording_last_lift = self.lift_height
        self._recording_lift_velocity = 0.0

    def _apply_continuous_motion(self, dt: float) -> None:
        """Synchronize the tracked base, lift carriage, and fixed-base G1.

        The embedded official G1 remains a fixed-base articulation.  During
        asset spawning its root joint cannot stay attached to LiftCarriage:
        PhysX does not permit an articulation's fixed root to use an external
        rigid body as body0.  Consequently moving only LiftCarriage leaves G1
        behind.  Apply the same smoothly integrated target to all three roots
        and publish their physical velocity so contacts with a grasped object
        see the commanded base/lift motion instead of a zero-velocity jump.
        """

        orientation = self._base_orientation()
        self.root_translate_op.Set(Gf.Vec3d(self.x, self.y, 0.0))
        self.root_orient_op.Set(
            Gf.Quatd(
                float(orientation[0]),
                Gf.Vec3d(0.0, 0.0, float(orientation[3])),
            )
        )
        self.carriage_translate_op.Set(Gf.Vec3d(0.0, 0.0, self.lift_height))
        self.lift_attributes["targetHeight"].Set(float(self.lift_height))

        orientation_tensor = torch.as_tensor(
            orientation[np.newaxis, :], dtype=torch.float32, device=self.device
        )
        self.vehicle_body.set_world_poses(
            positions=torch.tensor(
                [[self.x, self.y, 0.0]], dtype=torch.float32, device=self.device
            ),
            orientations=orientation_tensor,
        )
        self.carriage_body.set_world_poses(
            positions=torch.tensor(
                [[self.x, self.y, self.lift_height]],
                dtype=torch.float32,
                device=self.device,
            ),
            orientations=orientation_tensor,
        )

        robot_position, robot_orientation = self._robot_pose()
        root_pose = torch.as_tensor(
            np.concatenate((robot_position, robot_orientation))[np.newaxis, :],
            dtype=torch.float32,
            device=self.device,
        )
        self.robot.write_root_pose_to_sim(root_pose)

        forward_speed = 0.5 * (
            self.track_drive.left_velocity + self.track_drive.right_velocity
        )
        yaw_rate = (
            self.track_drive.right_velocity - self.track_drive.left_velocity
        ) / self.track_drive.track_separation
        linear_x = forward_speed * math.cos(
            self.yaw + self.track_drive.heading_offset
        )
        linear_y = forward_speed * math.sin(
            self.yaw + self.track_drive.heading_offset
        )
        lift_velocity = self._lift_target_velocity
        common_velocity = torch.tensor(
            [[linear_x, linear_y, 0.0, 0.0, 0.0, yaw_rate]],
            dtype=torch.float32,
            device=self.device,
        )
        carriage_velocity = common_velocity.clone()
        carriage_velocity[0, 2] = lift_velocity
        self.vehicle_body.set_velocities(common_velocity)
        self.carriage_body.set_velocities(carriage_velocity)
        self.robot.write_root_velocity_to_sim(carriage_velocity)

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
        self._left_target_velocity = (
            self.track_drive.max_track_speed * left_axis / normalization
        )
        self._right_target_velocity = (
            self.track_drive.max_track_speed * right_axis / normalization
        )
        self.x, self.y, self.yaw = self.track_drive.step(
            self.x,
            self.y,
            self.yaw,
            self._left_target_velocity,
            self._right_target_velocity,
            dt,
        )
        self._write_track_state(
            self._left_target_velocity, self._right_target_velocity
        )

        lift_direction = float(
            (carb.input.KeyboardInput.T in self.held)
            - (carb.input.KeyboardInput.Y in self.held)
        ) + self._validation_lift_direction
        lift_direction = float(np.clip(lift_direction, -1.0, 1.0))
        self._lift_target_velocity = lift_direction * self.lift_speed
        previous_lift = self.lift_height
        self.lift_height = float(
            np.clip(
                self.lift_height + self._lift_target_velocity * dt,
                self.lift_lower,
                self.lift_upper,
            )
        )
        if self.lift_height == previous_lift:
            self._lift_target_velocity = 0.0

        self._apply_continuous_motion(dt)
        self._telemetry_sequence += 1
        self._update_success_checker(dt)

    def reset(self) -> None:
        self.x, self.y, self.yaw, self.lift_height = self.initial
        self.track_drive.reset()
        self._left_target_velocity = 0.0
        self._right_target_velocity = 0.0
        self._lift_target_velocity = 0.0
        self._write_track_state(0.0, 0.0)
        self._teleport_base_and_lift()
        print("[roof-bolt-upright] Base pose and lift reset.")

    def reset_everything(self) -> None:
        env_ids = torch.arange(self.env.num_envs, device=self.device)
        base_mdp.reset_scene_to_default(self.env, env_ids)
        self.reset()
        self.reset_task_objects()
        print(
            "[roof-bolt-upright] R reset complete: robot, base/lift, rod, "
            "and both invisible supports restored."
        )

    @staticmethod
    def _pose_tensor(position, orientation, device) -> torch.Tensor:
        return torch.tensor(
            [[*position, *orientation]], dtype=torch.float32, device=device
        )

    def reset_task_objects(self) -> None:
        self._clear_success()
        offset = self._object_rng.uniform(
            -RESET_RANDOM_XY_LIMIT,
            RESET_RANDOM_XY_LIMIT,
            size=2,
        )
        env_ids = torch.arange(self.env.num_envs, device=self.device)

        def shifted(position):
            return (
                float(position[0] + offset[0]),
                float(position[1] + offset[1]),
                float(position[2]),
            )

        self.rod.write_root_pose_to_sim(
            self._pose_tensor(shifted(ROD_POSITION), ROD_ORIENTATION, self.device),
            env_ids=env_ids,
        )
        self.rod.write_root_velocity_to_sim(
            torch.zeros((len(env_ids), 6), dtype=torch.float32, device=self.device),
            env_ids=env_ids,
        )
        for support, position in (
            (self.left_support, LEFT_SUPPORT_POSITION),
            (self.right_support, RIGHT_SUPPORT_POSITION),
        ):
            support.write_root_pose_to_sim(
                self._pose_tensor(
                    shifted(position), SUPPORT_ORIENTATION, self.device
                ),
                env_ids=env_ids,
            )
            support.write_root_velocity_to_sim(
                torch.zeros(
                    (len(env_ids), 6), dtype=torch.float32, device=self.device
                ),
                env_ids=env_ids,
            )
        print(
            "[roof-bolt-upright] Object reset: rod/support shared "
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

    def _rod_geometry(self) -> tuple[float, float]:
        pose = self.rod.data.root_pose_w[0].detach().cpu().numpy().astype(np.float64)
        axis = self._rotate_vector_by_quaternion(
            pose[3:7], np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
        )
        axis_z = float(np.clip(abs(axis[2]), 0.0, 1.0))
        axis_tilt_deg = math.degrees(math.acos(axis_z))
        lower_tip_z = float(pose[2] - ROD_HALF_LENGTH * axis_z - GROUND_Z)
        return axis_tilt_deg, lower_tip_z

    def _update_success_checker(self, dt: float) -> None:
        axis_tilt_deg, lower_tip_z = self._rod_geometry()
        self.success_axis_tilt_deg = axis_tilt_deg
        self.success_lower_tip_z = lower_tip_z
        if lower_tip_z >= LIFTED_LOWER_ENDPOINT_Z:
            self.rod_has_been_lifted = True

        tilt_ok = axis_tilt_deg <= MAX_AXIS_TILT_DEG
        tip_ok = MIN_SUCCESS_TIP_Z <= lower_tip_z <= MAX_SUCCESS_TIP_Z
        candidate = self.rod_has_been_lifted and tilt_ok and tip_ok
        self.success_hold_time = (
            self.success_hold_time + dt if candidate else 0.0
        )
        success = self.success_hold_time >= SUCCESS_HOLD_SECONDS

        now = time.monotonic()
        if now - self._last_success_diagnostic_time >= 0.5:
            self._last_success_diagnostic_time = now
            carriage_position, _ = self.carriage_body.get_world_poses()
            actual_lift_z = float(carriage_position[0, 2].item())
            actual_robot_root_z = float(
                self.robot.data.root_pos_w[0, 2].detach().cpu().item()
            )
            SUCCESS_STATUS_PATH.write_text(
                json.dumps(
                    {
                        "success": bool(success),
                        "rod_has_been_lifted": bool(self.rod_has_been_lifted),
                        "lower_tip_z_m": lower_tip_z,
                        "minimum_success_tip_z_m": MIN_SUCCESS_TIP_Z,
                        "maximum_success_tip_z_m": MAX_SUCCESS_TIP_Z,
                        "axis_tilt_deg": axis_tilt_deg,
                        "maximum_axis_tilt_deg": MAX_AXIS_TILT_DEG,
                        "hold_seconds": self.success_hold_time,
                        "required_hold_seconds": SUCCESS_HOLD_SECONDS,
                        "commanded_lift_height_m": self.lift_height,
                        "actual_lift_carriage_z_m": actual_lift_z,
                        "actual_robot_root_z_m": actual_robot_root_z,
                        "lift_velocity_target_mps": self._lift_target_velocity,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
        if not success or self.success_latched:
            return

        self.success_latched = True
        SUCCESS_MARKER_PATH.write_text(
            json.dumps(
                {
                    "success": True,
                    "lower_tip_z_m": lower_tip_z,
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
            "[roof-bolt-upright] Rod is upright on the ground: "
            f"lower_tip_z={lower_tip_z:.4f} m, "
            f"axis_tilt={axis_tilt_deg:.2f} deg, "
            f"stable={self.success_hold_time:.2f} s."
        )
        print(
            "[roof-bolt-upright] Simulation, commands, and new camera frames "
            "are frozen. Press right [A] to save or terminal [s] to discard."
        )
        print("=" * 72 + "\n")

    @staticmethod
    def _yaw_from_quaternion(quaternion_wxyz: np.ndarray) -> float:
        w, x, y, z = quaternion_wxyz
        return math.atan2(
            2.0 * (w * z + x * y),
            1.0 - 2.0 * (y * y + z * z),
        )

    def get_recording_state(self) -> dict:
        """Return 7 state and 3 action values for LeRobot mobile telemetry."""

        vehicle_position, vehicle_orientation = self.vehicle_body.get_world_poses()
        carriage_position, _ = self.carriage_body.get_world_poses()
        actual_x = float(vehicle_position[0, 0].item())
        actual_y = float(vehicle_position[0, 1].item())
        actual_yaw = self._yaw_from_quaternion(
            vehicle_orientation[0].detach().cpu().numpy().astype(np.float64)
        )
        actual_lift = float(carriage_position[0, 2].item())

        now = time.monotonic()
        elapsed = max(1.0e-4, now - self._recording_last_time)
        self._recording_lift_velocity = (
            actual_lift - self._recording_last_lift
        ) / elapsed
        self._recording_last_time = now
        self._recording_last_lift = actual_lift

        dx = actual_x - self.initial[0]
        dy = actual_y - self.initial[1]
        forward = dx * TASK_FORWARD_X + dy * TASK_FORWARD_Y
        lateral = dx * -TASK_FORWARD_Y + dy * TASK_FORWARD_X
        yaw_delta = math.atan2(
            math.sin(actual_yaw - self.initial[2]),
            math.cos(actual_yaw - self.initial[2]),
        )
        return {
            "schema": "tracked_base_v1",
            "task_id": TASK_ID,
            "sequence": int(self._telemetry_sequence),
            "timestamp_ns": time.time_ns(),
            "state": {
                "base_forward_position_m": float(forward),
                "base_lateral_position_m": float(lateral),
                "base_yaw_delta_rad": float(yaw_delta),
                "lift_height_m": actual_lift,
                "left_track_velocity_mps": float(self.track_drive.left_velocity),
                "right_track_velocity_mps": float(self.track_drive.right_velocity),
                "lift_velocity_mps": float(self._recording_lift_velocity),
            },
            "action": {
                "left_track_velocity_target_mps": float(
                    self._left_target_velocity
                ),
                "right_track_velocity_target_mps": float(
                    self._right_target_velocity
                ),
                "lift_velocity_target_mps": float(self._lift_target_velocity),
            },
        }

    @property
    def is_success_frozen(self) -> bool:
        return self.success_latched

    def _clear_success(self) -> None:
        was_latched = self.success_latched
        self.success_latched = False
        self.rod_has_been_lifted = False
        self.success_hold_time = 0.0
        self.success_axis_tilt_deg = 90.0
        self.success_lower_tip_z = float("inf")
        SUCCESS_MARKER_PATH.unlink(missing_ok=True)
        SUCCESS_STATUS_PATH.unlink(missing_ok=True)
        if was_latched:
            print("[roof-bolt-upright] Success freeze cleared by reset.")

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
