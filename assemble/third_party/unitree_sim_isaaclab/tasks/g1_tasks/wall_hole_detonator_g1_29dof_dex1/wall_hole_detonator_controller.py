"""Tracked-base controls, loaded-assembly reset, and wall insertion success."""

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
from .wall_hole_detonator_g1_29dof_dex1_env_cfg import (
    DETONATOR_HALF_LENGTH,
    DETONATOR_OUTER_RADIUS,
    HOLE_CENTER,
    HOLE_HALF_LENGTH,
    HOLE_OUTWARD_AXIS,
    OBJECT_RANDOM_XY_LIMIT,
    ROBOT_RIGHT_XY,
    TABLE_EDGE_MARGIN_M,
    TABLE_OUTWARD_HALF_EXTENT,
    TABLE_POSITION,
    TABLE_RIGHT_HALF_EXTENT,
    TASK_TRACKED_BASE_X,
    TASK_TRACKED_BASE_Y,
    TASK_TRACKED_BASE_YAW,
    TASK_TRACKED_LIFT_HEIGHT,
)


W1_TOOLS = ASSETS_ROOT / "robot/Forklift_W1_Final/tools"
TASK_ID = "Isaac-WallHoleDetonator-G129-Dex1-Joint"
REPLAY_STATE_SCHEMA = "wall_hole_detonator_replay_v1"
SUCCESS_MARKER_PATH = Path(
    os.environ.get(
        "G1_TASK_SUCCESS_FILE",
        f"/tmp/g1_wall_hole_detonator_{os.getuid()}.success",
    )
)
SUCCESS_STATUS_PATH = SUCCESS_MARKER_PATH.with_suffix(".status.json")
# The detonator is 2 * DETONATOR_HALF_LENGTH long. Success needs five eighths
# of that length inside the bore, with a centerline offset strictly smaller
# than one complete detonator diameter.
MIN_INSERTION_DEPTH = 2.0 * DETONATOR_HALF_LENGTH * 5.0 / 8.0
MAX_CENTERLINE_RADIAL_ERROR = 2.0 * DETONATOR_OUTER_RADIUS
OBJECT_GRIPPER_CLEARANCE = 0.03
MAX_OBJECT_SAMPLE_ATTEMPTS = 1000
LEAD_OUTWARD_OVERHANG_M = 0.17

if str(W1_TOOLS) not in sys.path:
    sys.path.insert(0, str(W1_TOOLS))
from differential_drive import DifferentialTrackKinematics  # noqa: E402


class WallHoleDetonatorController:
    """Control the tracked G1 and manage the loaded detonator assembly."""

    ROOT_PATH = "/World/TrackedBase"
    VEHICLE_PATH = f"{ROOT_PATH}/Vehicle"
    CARRIAGE_PATH = f"{ROOT_PATH}/LiftCarriage"
    ASSET_PATH = "/World/envs/env_0/LoadedDetonator"
    ASSEMBLY_PATH = f"{ASSET_PATH}/Body"
    ROPE_PATH_EXPR = f"{ASSET_PATH}/Rope/Links/Link_.*"

    def __init__(self, env) -> None:
        self.env = env
        self.robot = env.scene["robot"]
        self.device = env.device
        self.stage = omni.usd.get_context().get_stage()

        self.x = float(TASK_TRACKED_BASE_X)
        self.y = float(TASK_TRACKED_BASE_Y)
        self.yaw = float(TASK_TRACKED_BASE_YAW)
        self.lift_height = float(TASK_TRACKED_LIFT_HEIGHT)
        self.initial = (self.x, self.y, self.yaw, self.lift_height)
        self.object_random_xy_limit = float(OBJECT_RANDOM_XY_LIMIT)
        self._object_rng = np.random.default_rng()
        self._benchmark = BenchmarkProtocol()
        self.reset_count = 0
        self._active_object_initial_pose = np.zeros(7, dtype=np.float64)
        self._reset_settle_steps_remaining = 0
        self.success_latched = False
        self.success_insertion_depth = 0.0
        self.success_max_radial_error = float("inf")
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
            name="wall_hole_tracked_vehicle",
            reset_xform_properties=False,
        )
        self.carriage_body = RigidPrim(
            prim_paths_expr=self.CARRIAGE_PATH,
            name="wall_hole_lift_carriage",
            reset_xform_properties=False,
        )
        self.assembly_body = RigidPrim(
            prim_paths_expr=self.ASSEMBLY_PATH,
            name="loaded_detonator_root",
            reset_xform_properties=False,
        )
        self.rope_bodies = RigidPrim(
            prim_paths_expr=self.ROPE_PATH_EXPR,
            name="loaded_detonator_rope_links",
            reset_xform_properties=False,
        )
        self._task_object_views = (self.assembly_body, self.rope_bodies)
        for view in (
            self.vehicle_body,
            self.carriage_body,
            self.assembly_body,
            self.rope_bodies,
        ):
            view.initialize()
            if not view.is_physics_handle_valid():
                raise RuntimeError(
                    f"Physics handle did not initialize for {view.name}"
                )
        if self.assembly_body.count != 1 or self.rope_bodies.count != 26:
            raise RuntimeError(
                "Unexpected loaded assembly body count: "
                f"root={self.assembly_body.count}, rope={self.rope_bodies.count}"
            )

        self._task_object_initial_poses = [
            view.get_world_poses(clone=True) for view in self._task_object_views
        ]
        initial_root_positions, initial_root_orientations = (
            self._task_object_initial_poses[0]
        )
        self._active_object_initial_pose = np.concatenate(
            (
                initial_root_positions[0].detach().cpu().numpy(),
                initial_root_orientations[0].detach().cpu().numpy(),
            )
        ).astype(np.float64)
        self._gripper_body_indices, _ = self.robot.find_bodies(
            ["left_hand_.*", "right_hand_.*"],
            preserve_order=True,
        )
        if not self._gripper_body_indices:
            raise RuntimeError("No G1 gripper bodies found for reset clearance")

        self._subscribe_keyboard()
        self.apply()
        self._home_gripper_guard_points = self._read_gripper_guard_points()
        self.reset_task_objects()
        print(
            "[wall-hole-detonator] Controller ready. W/S drive, A/D turn, "
            "T/Y lift, R resets robot and loaded assembly."
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
            print(f"[wall-hole-detonator] Keyboard input unavailable: {error}")

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
        if self._reset_settle_steps_remaining > 0:
            self._zero_task_velocities()
            self._reset_settle_steps_remaining -= 1
        self._benchmark.tick()

    def reset(self) -> None:
        self.x, self.y, self.yaw, self.lift_height = self.initial
        self.track_drive.reset()
        self._write_track_state(0.0, 0.0)
        self.apply()
        print("[wall-hole-detonator] Base pose and lift reset.")


    def reset_everything(self) -> None:
        env_ids = torch.arange(self.env.num_envs, device=self.device)
        base_mdp.reset_scene_to_default(self.env, env_ids)
        self.reset()
        self.reset_task_objects()
        print(
            "[wall-hole-detonator] Full reset complete: robot, base/lift, "
            "loaded detonator, and all flexible lead links restored."
        )

    def _read_gripper_guard_points(self) -> np.ndarray:
        points = (
            self.robot.data.body_pos_w[0, self._gripper_body_indices]
            .detach()
            .cpu()
            .numpy()
            .astype(np.float64)
        )
        return points[np.all(np.isfinite(points), axis=1)]

    def _combined_gripper_guard_points(self) -> np.ndarray:
        current = self._read_gripper_guard_points()
        if current.size == 0:
            return self._home_gripper_guard_points.copy()
        return np.concatenate((self._home_gripper_guard_points, current), axis=0)

    def _candidate_position_sets(self, offset: np.ndarray) -> list[np.ndarray]:
        positions = []
        delta = np.asarray([offset[0], offset[1], 0.0], dtype=np.float64)
        for initial_positions, _ in self._task_object_initial_poses:
            positions.append(
                initial_positions.detach().cpu().numpy().astype(np.float64)
                + delta[None, :]
            )
        return positions

    def _candidate_is_safe(
        self,
        offset: np.ndarray,
        gripper_points: np.ndarray,
    ) -> bool:
        body_positions, rope_positions = self._candidate_position_sets(offset)
        positions = np.concatenate((body_positions, rope_positions), axis=0)
        outward = np.asarray(
            [HOLE_OUTWARD_AXIS[0], HOLE_OUTWARD_AXIS[1]], dtype=np.float64
        )
        outward /= np.linalg.norm(outward)
        right = np.asarray(ROBOT_RIGHT_XY, dtype=np.float64)

        body_relative = (
            body_positions[:, :2] - np.asarray(TABLE_POSITION[:2])[None, :]
        )
        body_outward = body_relative @ outward
        body_right = body_relative @ right
        if np.any(
            np.abs(body_outward)
            > TABLE_OUTWARD_HALF_EXTENT
            - DETONATOR_OUTER_RADIUS
            - TABLE_EDGE_MARGIN_M
        ):
            return False
        if np.any(
            np.abs(body_right)
            > TABLE_RIGHT_HALF_EXTENT
            - DETONATOR_HALF_LENGTH
            - TABLE_EDGE_MARGIN_M
        ):
            return False

        # The table-only forward adjustment intentionally leaves the far end
        # of the flexible lead hanging slightly over the outward edge.  Keep
        # the body on the tabletop and permit only this bounded lead overhang.
        rope_relative = (
            rope_positions[:, :2] - np.asarray(TABLE_POSITION[:2])[None, :]
        )
        rope_outward = rope_relative @ outward
        rope_right = rope_relative @ right
        if np.any(
            rope_outward
            < -TABLE_OUTWARD_HALF_EXTENT + 0.012 + TABLE_EDGE_MARGIN_M
        ) or np.any(
            rope_outward
            > TABLE_OUTWARD_HALF_EXTENT
            + LEAD_OUTWARD_OVERHANG_M
            - 0.012
            - TABLE_EDGE_MARGIN_M
        ):
            return False
        if np.any(
            np.abs(rope_right)
            > TABLE_RIGHT_HALF_EXTENT - 0.004 - TABLE_EDGE_MARGIN_M
        ):
            return False
        if gripper_points.size:
            distances = np.linalg.norm(
                positions[:, None, :] - gripper_points[None, :, :],
                axis=2,
            )
            if float(np.min(distances)) < OBJECT_GRIPPER_CLEARANCE:
                return False
        return True

    def _sample_object_offset(self) -> tuple[np.ndarray, int]:
        gripper_points = self._combined_gripper_guard_points()
        for attempt in range(1, MAX_OBJECT_SAMPLE_ATTEMPTS + 1):
            offset = self._object_rng.uniform(
                -self.object_random_xy_limit,
                self.object_random_xy_limit,
                size=2,
            )
            if self._candidate_is_safe(offset, gripper_points):
                return offset, attempt
        raise RuntimeError(
            "Unable to sample a table-safe, gripper-clear loaded detonator "
            f"pose after {MAX_OBJECT_SAMPLE_ATTEMPTS} attempts"
        )

    def _zero_task_velocities(self) -> None:
        for view in self._task_object_views:
            view.set_velocities(
                torch.zeros(
                    (view.count, 6),
                    dtype=torch.float32,
                    device=self.device,
                )
            )

    def reset_task_objects(self) -> None:
        self._clear_success()
        request = self._benchmark.consume_request()
        if request is not None:
            self._object_rng = np.random.default_rng(int(request["seed"]))
        offset, attempts = self._sample_object_offset()
        offset_tensor = torch.tensor(
            [float(offset[0]), float(offset[1]), 0.0],
            dtype=torch.float32,
            device=self.device,
        )
        for view, (positions, orientations) in zip(
            self._task_object_views,
            self._task_object_initial_poses,
        ):
            view.set_world_poses(
                positions=positions + offset_tensor.unsqueeze(0),
                orientations=orientations,
            )
        root_positions, root_orientations = self._task_object_initial_poses[0]
        self._active_object_initial_pose = np.concatenate(
            (
                (root_positions[0] + offset_tensor).detach().cpu().numpy(),
                root_orientations[0].detach().cpu().numpy(),
            )
        ).astype(np.float64)
        self.reset_count += 1
        self._zero_task_velocities()
        self._reset_settle_steps_remaining = 10
        print(
            "[wall-hole-detonator] Loaded assembly reset: "
            f"reset={self.reset_count}, "
            f"dx={100.0 * offset[0]:+.1f} cm, "
            f"dy={100.0 * offset[1]:+.1f} cm "
            f"(sample attempts={attempts})."
        )
        self._benchmark.begin_settle(
            request,
            {
                "object_offset_xy_m": offset.astype(float).tolist(),
                "object_initial_pose_wxyz": self._active_object_initial_pose.astype(float).tolist(),
                "sample_attempts": int(attempts),
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
    def _rotate_vector_by_quaternion(
        cls,
        quaternion: np.ndarray,
        vector: np.ndarray,
    ) -> np.ndarray:
        quaternion = quaternion / np.linalg.norm(quaternion)
        conjugate = quaternion.copy()
        conjugate[1:] *= -1.0
        return cls._quaternion_multiply(
            cls._quaternion_multiply(
                quaternion,
                np.concatenate(([0.0], vector)),
            ),
            conjugate,
        )[1:]

    def _success_geometry(self) -> tuple[bool, float, float]:
        positions, orientations = self.assembly_body.get_world_poses(clone=True)
        detonator_center = (
            positions[0].detach().cpu().numpy().astype(np.float64)
        )
        detonator_orientation = (
            orientations[0].detach().cpu().numpy().astype(np.float64)
        )
        detonator_axis = self._rotate_vector_by_quaternion(
            detonator_orientation,
            np.asarray([0.0, 0.0, 1.0], dtype=np.float64),
        )
        detonator_axis /= np.linalg.norm(detonator_axis)
        hole_center = np.asarray(HOLE_CENTER, dtype=np.float64)
        hole_inward = -np.asarray(HOLE_OUTWARD_AXIS, dtype=np.float64)
        hole_inward /= np.linalg.norm(hole_inward)

        alignment = float(np.dot(detonator_axis, hole_inward))
        delta = detonator_center - hole_center
        center_t = float(np.dot(delta, hole_inward))
        endpoint_t = (
            center_t - DETONATOR_HALF_LENGTH * alignment,
            center_t + DETONATOR_HALF_LENGTH * alignment,
        )
        segment_min = min(endpoint_t)
        segment_max = max(endpoint_t)
        overlap_start = max(-HOLE_HALF_LENGTH, segment_min)
        overlap_end = min(HOLE_HALF_LENGTH, segment_max)
        insertion_depth = max(0.0, overlap_end - overlap_start)

        radial_error = float("inf")
        if insertion_depth > 0.0 and abs(alignment) > 1.0e-8:
            overlap_midpoint_t = 0.5 * (overlap_start + overlap_end)
            detonator_parameter = (
                overlap_midpoint_t - center_t
            ) / alignment
            detonator_point = (
                detonator_center + detonator_parameter * detonator_axis
            )
            hole_point = hole_center + overlap_midpoint_t * hole_inward
            radial_error = float(np.linalg.norm(detonator_point - hole_point))

        radial_limit = MAX_CENTERLINE_RADIAL_ERROR
        success = (
            insertion_depth >= MIN_INSERTION_DEPTH
            and radial_error < radial_limit
        )
        return success, insertion_depth, radial_error

    def _update_success_checker(self) -> None:
        (
            success,
            insertion_depth,
            radial_error,
        ) = self._success_geometry()
        self.success_insertion_depth = insertion_depth
        self.success_max_radial_error = radial_error
        radial_limit = MAX_CENTERLINE_RADIAL_ERROR

        now = time.monotonic()
        if now - self._last_success_diagnostic_time >= 0.5:
            self._last_success_diagnostic_time = now
            status = {
                "success": bool(success),
                "insertion_depth_m": insertion_depth,
                "minimum_insertion_depth_m": MIN_INSERTION_DEPTH,
                "centerline_radial_error_m": radial_error,
                "maximum_centerline_radial_error_m": radial_limit,
            }
            SUCCESS_STATUS_PATH.write_text(
                json.dumps(status, indent=2), encoding="utf-8"
            )
            if insertion_depth >= 0.10:
                print(
                    "[wall-hole-detonator] Success check: "
                    f"depth={insertion_depth:.3f}/{MIN_INSERTION_DEPTH:.3f} m, "
                    f"radial={1000.0 * radial_error:.1f}/"
                    f"<{1000.0 * radial_limit:.1f} mm."
                )

        if not success or self.success_latched:
            return
        self.success_latched = True
        payload = {
            "success": True,
            "insertion_depth_m": insertion_depth,
            "centerline_radial_error_m": radial_error,
            "timestamp": datetime.now().astimezone().isoformat(),
        }
        SUCCESS_MARKER_PATH.write_text(
            json.dumps(payload, indent=2), encoding="utf-8"
        )
        SUCCESS_STATUS_PATH.write_text(
            json.dumps(payload, indent=2), encoding="utf-8"
        )
        print("\n" + "=" * 72)
        print("SUCCESS")
        print(
            "[wall-hole-detonator] Loaded detonator inserted into wall hole: "
            f"depth={insertion_depth:.4f} m, "
            f"radial={1000.0 * radial_error:.2f} mm."
        )
        print(
            "[wall-hole-detonator] Simulation and new frames are frozen. "
            "Press right [A] to save or terminal [s] to discard."
        )
        print("=" * 72 + "\n")

    def get_recording_state(self) -> dict:
        """Expose the exact loaded-detonator pose for episode metadata."""

        positions, orientations = self.assembly_body.get_world_poses(clone=True)
        object_pose = np.concatenate(
            (
                positions[0].detach().cpu().numpy(),
                orientations[0].detach().cpu().numpy(),
            )
        ).astype(np.float64)
        robot_position, robot_orientation = self._robot_pose()
        return {
            "schema": REPLAY_STATE_SCHEMA,
            "task_id": TASK_ID,
            "timestamp_ns": time.time_ns(),
            "reset_count": int(self.reset_count),
            "active_object": "loaded_detonator_assembly",
            "tracked_base_pose": [self.x, self.y, self.yaw],
            "lift_height_m": self.lift_height,
            "object_root_pose_wxyz": object_pose.tolist(),
            "replay_initial_state": {
                "robot": {
                    "tracked_base_pose_xy_yaw": [self.x, self.y, self.yaw],
                    "lift_height_m": self.lift_height,
                    "g1_root_pose_wxyz": [
                        *robot_position.astype(np.float64).tolist(),
                        *robot_orientation.astype(np.float64).tolist(),
                    ],
                },
                "object": {
                    "name": "loaded_detonator_assembly",
                    "root_pose_wxyz": object_pose.tolist(),
                    "authored_reset_pose_wxyz": (
                        self._active_object_initial_pose.tolist()
                    ),
                },
            },
        }

    @property
    def is_success_frozen(self) -> bool:
        return self.success_latched

    def _clear_success(self) -> None:
        was_latched = self.success_latched
        self.success_latched = False
        self.success_insertion_depth = 0.0
        self.success_max_radial_error = float("inf")
        SUCCESS_MARKER_PATH.unlink(missing_ok=True)
        SUCCESS_STATUS_PATH.unlink(missing_ok=True)
        if was_latched:
            print("[wall-hole-detonator] Success freeze cleared by reset.")

    def close(self) -> None:
        self._clear_success()
        if self._input is not None and self._subscription is not None:
            try:
                self._input.unsubscribe_to_keyboard_events(
                    self._keyboard,
                    self._subscription,
                )
            except Exception:
                pass
        self._subscription = None
