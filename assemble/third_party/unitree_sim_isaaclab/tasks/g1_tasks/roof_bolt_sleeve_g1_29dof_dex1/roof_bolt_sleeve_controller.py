"""Tracked-base controls, object reset, and insertion success checking."""

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
from tasks.g1_tasks.benchmark_protocol import BenchmarkProtocol
from .roof_bolt_sleeve_g1_29dof_dex1_env_cfg import (
    TASK_TRACKED_BASE_X,
    TASK_TRACKED_BASE_Y,
    TASK_TRACKED_BASE_YAW,
    TASK_TRACKED_LIFT_HEIGHT,
)


W1_TOOLS = ASSETS_ROOT / "robot/Forklift_W1_Final/tools"
SUCCESS_MARKER_PATH = Path(
    os.environ.get(
        "G1_TASK_SUCCESS_FILE",
        f"/tmp/g1_roof_bolt_sleeve_{os.getuid()}.success",
    )
)
SUCCESS_STATUS_PATH = SUCCESS_MARKER_PATH.with_suffix(".status.json")

SLEEVE_INNER_RADIUS = 0.0265
# The task collision bore is intentionally 55 mm even though the visible bore
# is 53 mm. Use that physical bore for the success check as well.
SLEEVE_COLLISION_INNER_RADIUS = 0.0275
SLEEVE_HALF_LENGTH = 0.250 / 2.0
ROOF_BOLT_RADIUS = 0.0225
ROOF_BOLT_HALF_LENGTH = 0.725
# The sleeve was shortened from the original 437.9 mm design to 250 mm. Using
# 0.25 m here required mathematically perfect 100% overlap and could miss on
# normal pose/float error even when the installation was visually complete.
# Require 80% engagement while retaining the independent cavity/radial check.
MIN_INSERTION_DEPTH = 0.20
RADIAL_TOLERANCE = 0.0010
MAX_AXIS_TILT_DEG = 12.0

if str(W1_TOOLS) not in sys.path:
    sys.path.insert(0, str(W1_TOOLS))
from differential_drive import DifferentialTrackKinematics  # noqa: E402


class RoofBoltSleeveController:
    """Keep the existing base controls and manage the two new task objects."""

    ROOT_PATH = "/World/TrackedBase"
    VEHICLE_PATH = f"{ROOT_PATH}/Vehicle"
    CARRIAGE_PATH = f"{ROOT_PATH}/LiftCarriage"

    def __init__(self, env) -> None:
        self.env = env
        self.robot = env.scene["robot"]
        self.sleeve = env.scene["sleeve"]
        self.roof_bolt = env.scene["roof_bolt"]
        self.device = env.device
        self.stage = omni.usd.get_context().get_stage()

        self.x = float(TASK_TRACKED_BASE_X)
        self.y = float(TASK_TRACKED_BASE_Y)
        self.yaw = float(TASK_TRACKED_BASE_YAW)
        self.lift_height = float(TASK_TRACKED_LIFT_HEIGHT)
        self.initial = (self.x, self.y, self.yaw, self.lift_height)
        self.object_random_xy_limit = 0.05
        self.roof_bolt_random_xy_limit = 0.03
        self._object_rng = np.random.default_rng()
        self._benchmark = BenchmarkProtocol()
        self.success_latched = False
        self.success_insertion_depth = 0.0
        self.success_max_radial_error = float("inf")
        self.success_axis_tilt_deg = float("inf")
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
            name="roof_bolt_tracked_vehicle",
            reset_xform_properties=False,
        )
        self.carriage_body = RigidPrim(
            prim_paths_expr=self.CARRIAGE_PATH,
            name="roof_bolt_lift_carriage",
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
            "[roof-bolt-sleeve] Controller ready. W/S drive, A/D turn, "
            "T/Y lift, R resets robot and both objects."
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
            print(f"[roof-bolt-sleeve] Keyboard input unavailable: {error}")

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
        self.apply()
        self._update_success_checker()
        self._benchmark.tick()

    def reset(self) -> None:
        self.x, self.y, self.yaw, self.lift_height = self.initial
        self.track_drive.reset()
        self._write_track_state(0.0, 0.0)
        self.apply()
        print("[roof-bolt-sleeve] Base pose and lift reset.")

    def reset_everything(self) -> None:
        # The bridge has already driven the arms home. Avoid the generic scene
        # reset because it writes velocity to the fixed kinematic roof bolt.
        self.reset()
        self.reset_task_objects()
        print(
            "[roof-bolt-sleeve] R reset complete: robot, base/lift, sleeve, "
            "and fixed roof bolt restored."
        )

    def reset_task_objects(self) -> None:
        self._clear_success()
        request = self._benchmark.consume_request()
        if request is not None:
            self._object_rng = np.random.default_rng(int(request["seed"]))
        env_ids = torch.arange(self.env.num_envs, device=self.device)
        sleeve_offset = self._object_rng.uniform(
            -self.object_random_xy_limit,
            self.object_random_xy_limit,
            size=2,
        )
        roof_bolt_offset = self._object_rng.uniform(
            -self.roof_bolt_random_xy_limit,
            self.roof_bolt_random_xy_limit,
            size=2,
        )
        sleeve_state = self.sleeve.data.default_root_state[env_ids].clone()
        sleeve_state[:, :3] += self.env.scene.env_origins[env_ids]
        sleeve_state[:, 0] += float(sleeve_offset[0])
        sleeve_state[:, 1] += float(sleeve_offset[1])
        self.sleeve.write_root_pose_to_sim(sleeve_state[:, :7], env_ids=env_ids)
        self.sleeve.write_root_velocity_to_sim(
            torch.zeros((len(env_ids), 6), dtype=torch.float32, device=self.device),
            env_ids=env_ids,
        )

        bolt_state = self.roof_bolt.data.default_root_state[env_ids].clone()
        bolt_state[:, :3] += self.env.scene.env_origins[env_ids]
        bolt_state[:, 0] += float(roof_bolt_offset[0])
        bolt_state[:, 1] += float(roof_bolt_offset[1])
        self.roof_bolt.write_root_pose_to_sim(bolt_state[:, :7], env_ids=env_ids)
        print(
            "[roof-bolt-sleeve] Object reset: sleeve randomization "
            f"dx={100.0 * sleeve_offset[0]:+.1f} cm, "
            f"dy={100.0 * sleeve_offset[1]:+.1f} cm; roof-bolt randomization "
            f"dx={100.0 * roof_bolt_offset[0]:+.1f} cm, "
            f"dy={100.0 * roof_bolt_offset[1]:+.1f} cm."
        )
        self._benchmark.begin_settle(
            request,
            {
                "sleeve_offset_xy_m": sleeve_offset.astype(float).tolist(),
                "roof_bolt_offset_xy_m": roof_bolt_offset.astype(float).tolist(),
                "sample_attempts": 1,
            },
            steps=10,
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

    @classmethod
    def _rotate_vector_by_quaternion(
        cls, quaternion: np.ndarray, vector: np.ndarray
    ) -> np.ndarray:
        conjugate = quaternion.copy()
        conjugate[1:] *= -1.0
        return cls._quaternion_multiply(
            cls._quaternion_multiply(
                quaternion, np.concatenate(([0.0], vector))
            ),
            conjugate,
        )[1:]

    def _success_geometry(self) -> tuple[bool, float, float, float]:
        sleeve_pose = (
            self.sleeve.data.root_pose_w[0]
            .detach()
            .cpu()
            .numpy()
            .astype(np.float64)
        )
        bolt_pose = (
            self.roof_bolt.data.root_pose_w[0]
            .detach()
            .cpu()
            .numpy()
            .astype(np.float64)
        )
        bolt_origin, bolt_orientation = self._pose_relative_to(
            sleeve_pose, bolt_pose
        )
        bolt_axis = self._rotate_vector_by_quaternion(
            bolt_orientation,
            np.asarray([0.0, 0.0, 1.0], dtype=np.float64),
        )
        axis_z = float(bolt_axis[2])
        if abs(axis_z) < 1.0e-8:
            return False, 0.0, float("inf"), 90.0

        first = (-SLEEVE_HALF_LENGTH - bolt_origin[2]) / axis_z
        second = (SLEEVE_HALF_LENGTH - bolt_origin[2]) / axis_z
        inserted_start = max(-ROOF_BOLT_HALF_LENGTH, min(first, second))
        inserted_end = min(ROOF_BOLT_HALF_LENGTH, max(first, second))
        insertion_depth = max(0.0, inserted_end - inserted_start)
        if insertion_depth <= 0.0:
            return False, 0.0, float("inf"), math.degrees(
                math.acos(float(np.clip(abs(axis_z), 0.0, 1.0)))
            )

        # Checking both overlap endpoints made the detector much stricter than
        # the actual physics: a few degrees of normal hand-held tilt amplified
        # into a large endpoint error over 20 cm. The collision solver already
        # prevents the solid bolt from crossing the sleeve wall, so determine
        # cavity occupancy at the overlap midpoint and bound tilt separately.
        overlap_midpoint = bolt_origin + (
            0.5 * (inserted_start + inserted_end) * bolt_axis
        )
        centerline_radial_error = float(np.linalg.norm(overlap_midpoint[:2]))
        axis_tilt_deg = math.degrees(
            math.acos(float(np.clip(abs(axis_z), 0.0, 1.0)))
        )
        inside_cavity = (
            centerline_radial_error + ROOF_BOLT_RADIUS
            <= SLEEVE_COLLISION_INNER_RADIUS + RADIAL_TOLERANCE
            and axis_tilt_deg <= MAX_AXIS_TILT_DEG
        )
        return (
            inside_cavity and insertion_depth >= MIN_INSERTION_DEPTH,
            insertion_depth,
            centerline_radial_error,
            axis_tilt_deg,
        )

    def _update_success_checker(self) -> None:
        (
            success,
            insertion_depth,
            centerline_radial_error,
            axis_tilt_deg,
        ) = self._success_geometry()
        self.success_insertion_depth = insertion_depth
        self.success_max_radial_error = centerline_radial_error
        self.success_axis_tilt_deg = axis_tilt_deg

        now = time.monotonic()
        if now - self._last_success_diagnostic_time >= 0.5:
            self._last_success_diagnostic_time = now
            depth_ok = insertion_depth >= MIN_INSERTION_DEPTH
            radial_limit = (
                SLEEVE_COLLISION_INNER_RADIUS
                + RADIAL_TOLERANCE
                - ROOF_BOLT_RADIUS
            )
            radial_ok = centerline_radial_error <= radial_limit
            tilt_ok = axis_tilt_deg <= MAX_AXIS_TILT_DEG
            status = {
                "success": bool(success),
                "insertion_depth_m": insertion_depth,
                "minimum_insertion_depth_m": MIN_INSERTION_DEPTH,
                "centerline_radial_error_m": centerline_radial_error,
                "maximum_centerline_radial_error_m": radial_limit,
                "axis_tilt_deg": axis_tilt_deg,
                "maximum_axis_tilt_deg": MAX_AXIS_TILT_DEG,
                "depth_ok": bool(depth_ok),
                "radial_ok": bool(radial_ok),
                "tilt_ok": bool(tilt_ok),
            }
            SUCCESS_STATUS_PATH.write_text(
                json.dumps(status, indent=2), encoding="utf-8"
            )
            if insertion_depth >= 0.10:
                print(
                    "[roof-bolt-sleeve] Success check: "
                    f"depth={insertion_depth:.3f}/{MIN_INSERTION_DEPTH:.3f} m "
                    f"({'OK' if depth_ok else 'LOW'}), "
                    f"radial={1000.0 * centerline_radial_error:.1f}/"
                    f"{1000.0 * radial_limit:.1f} mm "
                    f"({'OK' if radial_ok else 'OUT'}), "
                    f"tilt={axis_tilt_deg:.1f}/{MAX_AXIS_TILT_DEG:.1f} deg "
                    f"({'OK' if tilt_ok else 'HIGH'})."
                )
        if not success or self.success_latched:
            return
        self.success_latched = True
        SUCCESS_MARKER_PATH.write_text(
            json.dumps(
                {
                    "success": True,
                    "insertion_depth_m": insertion_depth,
                    "centerline_radial_error_m": centerline_radial_error,
                    "axis_tilt_deg": axis_tilt_deg,
                    "timestamp": datetime.now().astimezone().isoformat(),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print("\n" + "=" * 72)
        print("SUCCESS")
        print(
            "[roof-bolt-sleeve] Sleeve installed over roof bolt: "
            f"insertion_depth={insertion_depth:.4f} m, "
            f"centerline_radial_error="
            f"{1000.0 * centerline_radial_error:.2f} mm, "
            f"axis_tilt={axis_tilt_deg:.2f} deg."
        )
        print(
            "[roof-bolt-sleeve] Simulation, commands, and new camera frames "
            "are frozen. Press right [A] to save or terminal [s] to discard."
        )
        print("=" * 72 + "\n")

    @property
    def is_success_frozen(self) -> bool:
        return self.success_latched

    def _clear_success(self) -> None:
        was_latched = self.success_latched
        self.success_latched = False
        self.success_insertion_depth = 0.0
        self.success_max_radial_error = float("inf")
        self.success_axis_tilt_deg = float("inf")
        SUCCESS_MARKER_PATH.unlink(missing_ok=True)
        SUCCESS_STATUS_PATH.unlink(missing_ok=True)
        if was_latched:
            print("[roof-bolt-sleeve] Success freeze cleared by reset.")

    def get_recording_state(self) -> dict:
        """Expose the two replay-critical object poses at episode start."""
        sleeve_pose = self.sleeve.data.root_pose_w[0].detach().cpu().numpy()
        bolt_pose = self.roof_bolt.data.root_pose_w[0].detach().cpu().numpy()
        return {
            "schema": "roof_bolt_sleeve_recording_state_v1",
            "task_id": "Isaac-RoofBoltSleeve-G129-Dex1-Joint",
            "timestamp_ns": time.time_ns(),
            "replay_initial_state": {
                "objects": {
                    "coupling_sleeve": {"root_pose_wxyz": sleeve_pose.astype(np.float64).tolist()},
                    "roof_bolt": {"root_pose_wxyz": bolt_pose.astype(np.float64).tolist()},
                }
            },
            "objects": {
                "coupling_sleeve": {"root_pose_wxyz": sleeve_pose.astype(np.float64).tolist()},
                "roof_bolt": {"root_pose_wxyz": bolt_pose.astype(np.float64).tolist()},
            },
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
