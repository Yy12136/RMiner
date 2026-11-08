"""Tracked-base control, randomized bin/object reset, and success check."""

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
from pxr import Gf, Usd, UsdGeom, UsdPhysics

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
from .bin_pick_place_g1_29dof_dex1_env_cfg import (
    BIN_COLLIDER_ROOT_SUFFIX,
    BIN_COLOR_PALETTE,
    BIN_PARKING_POSITION,
    BIN_POSITION,
    BIN_VARIANTS,
    OBJECT_ORIENTATION,
    OBJECT_VARIANTS,
    TABLE_RIGHT_VECTOR,
    TASK_FORWARD_VECTOR,
    TASK_TRACKED_BASE_X,
    TASK_TRACKED_BASE_Y,
    TASK_TRACKED_BASE_YAW,
    TASK_TRACKED_LIFT_HEIGHT,
)


W1_TOOLS = ASSETS_ROOT / "robot/Forklift_W1_Final/tools"
SUCCESS_MARKER_PATH = Path(
    os.environ.get(
        "G1_TASK_SUCCESS_FILE",
        f"/tmp/g1_bin_pick_place_{os.getuid()}.success",
    )
)
SUCCESS_STATUS_PATH = SUCCESS_MARKER_PATH.with_suffix(".status.json")

BIN_FLOOR_PENETRATION_TOLERANCE_M = 0.003
BIN_FLOOR_SUPPORT_TOLERANCE_M = 0.006
BIN_CENTER_INTERIOR_MARGIN_M = 0.003
RESET_LATERAL_LIMIT_M = 0.05
RESET_TOWARD_ROBOT_LIMIT_M = 0.05
MAX_LINEAR_SPEED_MPS = 0.08
MAX_ANGULAR_SPEED_RADPS = 0.50
SUCCESS_HOLD_SECONDS = 0.50
TASK_ID = "Isaac-BinPickPlace-G129-Dex1-Joint"
REPLAY_STATE_SCHEMA = "g1_bin_pick_place_replay_v1"

if str(W1_TOOLS) not in sys.path:
    sys.path.insert(0, str(W1_TOOLS))
from differential_drive import DifferentialTrackKinematics  # noqa: E402


class BinPickPlaceController:
    """Control the tracked base and manage the pick-place task lifecycle."""

    ROOT_PATH = "/World/TrackedBase"
    VEHICLE_PATH = f"{ROOT_PATH}/Vehicle"
    CARRIAGE_PATH = f"{ROOT_PATH}/LiftCarriage"

    def __init__(self, env) -> None:
        self.env = env
        self.robot = env.scene["robot"]
        self.task_object = None
        self.device = env.device
        self.stage = omni.usd.get_context().get_stage()

        self.x = float(TASK_TRACKED_BASE_X)
        self.y = float(TASK_TRACKED_BASE_Y)
        self.yaw = float(TASK_TRACKED_BASE_YAW)
        self.lift_height = float(TASK_TRACKED_LIFT_HEIGHT)
        self.initial = (self.x, self.y, self.yaw, self.lift_height)
        self._object_rng = np.random.default_rng()
        self._bin_rng = np.random.default_rng()
        self._benchmark = BenchmarkProtocol()
        self.reset_count = 0
        self.success_latched = False
        self.success_hold_time = 0.0
        self._last_success_diagnostic_time = 0.0
        SUCCESS_MARKER_PATH.unlink(missing_ok=True)
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
            name="bin_pick_place_tracked_vehicle",
            reset_xform_properties=False,
        )
        self.carriage_body = RigidPrim(
            prim_paths_expr=self.CARRIAGE_PATH,
            name="bin_pick_place_lift_carriage",
            reset_xform_properties=False,
        )
        self.vehicle_body.initialize()
        self.carriage_body.initialize()
        if not self.vehicle_body.is_physics_handle_valid():
            raise RuntimeError("Tracked vehicle physics handle did not initialize")
        if not self.carriage_body.is_physics_handle_valid():
            raise RuntimeError("Tracked lift physics handle did not initialize")

        self._bin_instances = self._initialize_bin_instances()
        self._object_instances = self._initialize_object_instances()
        self.active_bin_name = ""
        self.active_bin_color = ""
        self._active_bin_color_rgb = (0.0, 0.0, 0.0)
        self.active_object_name = ""
        self._object_local_collision_points = np.zeros((0, 3), dtype=np.float64)
        self._bin_pose = np.zeros(7, dtype=np.float64)
        self._bin_inner_half_x_m = 0.0
        self._bin_inner_half_y_m = 0.0
        self._bin_floor_top_m = 0.0
        self._bin_rim_height_m = 0.0
        self._bin_reset_lateral_m = 0.0
        self._bin_reset_toward_m = 0.0
        self._object_reset_lateral_m = 0.0
        self._object_reset_toward_m = 0.0
        self._object_reset_yaw_rad = 0.0
        self._active_object_initial_pose = np.zeros(7, dtype=np.float64)
        self._active_object_asset_path = ""
        self._active_bin_prim_path = ""
        self._subscribe_keyboard()
        self.apply()
        self.reset_task_object()
        print(
            "[bin-pick-place] Controller ready. W/S drive, A/D turn, "
            "T/Y lift, R resets robot, object, and randomized bin."
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
            print(f"[bin-pick-place] Keyboard input unavailable: {error}")

    def _initialize_bin_instances(self):
        instances = []
        expected_collider_names = (
            "Floor",
            "WallNegativeX",
            "WallPositiveX",
            "WallNegativeY",
            "WallPositiveY",
        )
        for variant in BIN_VARIANTS:
            prim = self.stage.GetPrimAtPath(variant["prim_path"])
            if not prim:
                raise RuntimeError(
                    f"Randomized bin prim is missing: {variant['prim_path']}"
                )
            collider_root_path = (
                f"{variant['prim_path']}{BIN_COLLIDER_ROOT_SUFFIX}"
            )
            collider_root = self.stage.GetPrimAtPath(collider_root_path)
            if not collider_root:
                raise RuntimeError(
                    f"Task collision root is missing: {collider_root_path}"
                )
            colliders = {}
            tint_attributes = []
            desaturation_attributes = []
            for child in Usd.PrimRange(prim):
                material_path = str(child.GetPath()).lower()
                is_main_plastic = (
                    "plastic_bin" in material_path
                    or "opaque__plastic__blue_a" in material_path
                )
                if is_main_plastic:
                    tint = child.GetAttribute("inputs:diffuse_tint")
                    if tint:
                        tint_attributes.append(tint)
                    desaturation = child.GetAttribute(
                        "inputs:albedo_desaturation"
                    )
                    if desaturation:
                        desaturation_attributes.append(desaturation)
            for child in Usd.PrimRange(collider_root):
                if child != collider_root and child.HasAPI(
                    UsdPhysics.CollisionAPI
                ):
                    colliders[child.GetName()] = child
            if tuple(sorted(colliders)) != tuple(sorted(expected_collider_names)):
                raise RuntimeError(
                    f"Incomplete compound collision for {variant['name']}: "
                    f"found={sorted(colliders)}, "
                    f"expected={list(expected_collider_names)}"
                )
            floor = colliders["Floor"]
            floor_center = floor.GetAttribute("xformOp:translate").Get()
            floor_size = floor.GetAttribute("xformOp:scale").Get()
            if floor_center is None or floor_size is None:
                raise RuntimeError(
                    f"Floor transform is missing for {variant['name']}"
                )
            measured_floor_top = float(floor_center[2]) + 0.5 * float(
                floor_size[2]
            )
            if not math.isclose(
                measured_floor_top,
                float(variant["floor_top_m"]),
                rel_tol=0.0,
                abs_tol=1.0e-6,
            ):
                raise RuntimeError(
                    f"Floor top mismatch for {variant['name']}: "
                    f"authored={measured_floor_top}, "
                    f"configured={variant['floor_top_m']}"
                )
            if not tint_attributes or not desaturation_attributes:
                raise RuntimeError(
                    f"Color-randomization inputs are missing for "
                    f"{variant['name']}: tint={len(tint_attributes)}, "
                    f"desaturation={len(desaturation_attributes)}"
                )
            instances.append(
                {
                    **variant,
                    "prim": prim,
                    "imageable": UsdGeom.Imageable(prim),
                    "translate_op": self._get_or_add_op(
                        UsdGeom.Xformable(prim), "translate"
                    ),
                    "collider_translate_op": self._get_or_add_op(
                        UsdGeom.Xformable(collider_root), "translate"
                    ),
                    "collision_attributes": [
                        UsdPhysics.CollisionAPI(colliders[name])
                        .GetCollisionEnabledAttr()
                        for name in expected_collider_names
                    ],
                    "floor_prim": floor,
                    "tint_attributes": tint_attributes,
                    "desaturation_attributes": desaturation_attributes,
                }
            )
        return instances

    @staticmethod
    def _sample_table_reset_offset(rng) -> tuple[float, float, np.ndarray]:
        """Sample lateral +/-5 cm and toward-robot-only 0..5 cm."""
        lateral = float(
            rng.uniform(-RESET_LATERAL_LIMIT_M, RESET_LATERAL_LIMIT_M)
        )
        toward = float(rng.uniform(0.0, RESET_TOWARD_ROBOT_LIMIT_M))
        world_offset = np.asarray(
            [
                TABLE_RIGHT_VECTOR[0] * lateral
                - TASK_FORWARD_VECTOR[0] * toward,
                TABLE_RIGHT_VECTOR[1] * lateral
                - TASK_FORWARD_VECTOR[1] * toward,
                0.0,
            ],
            dtype=np.float64,
        )
        return lateral, toward, world_offset

    def _select_random_bin(self) -> tuple[float, float]:
        selected_index = int(self._bin_rng.integers(0, len(self._bin_instances)))
        selected = self._bin_instances[selected_index]
        color_index = int(self._bin_rng.integers(0, len(BIN_COLOR_PALETTE)))
        color = BIN_COLOR_PALETTE[color_index]
        lateral, toward, world_offset = self._sample_table_reset_offset(
            self._bin_rng
        )
        selected_position = np.asarray(BIN_POSITION, dtype=np.float64) + world_offset
        for index, instance in enumerate(self._bin_instances):
            is_selected = index == selected_index
            position = selected_position if is_selected else BIN_PARKING_POSITION
            instance["translate_op"].Set(Gf.Vec3d(*position))
            instance["collider_translate_op"].Set(Gf.Vec3d(*position))
            if is_selected:
                instance["imageable"].MakeVisible()
            else:
                instance["imageable"].MakeInvisible()
            for attribute in instance["collision_attributes"]:
                attribute.Set(is_selected)

        self.active_bin_name = selected["name"]
        self.active_bin_color = color["name"]
        self._active_bin_color_rgb = tuple(float(value) for value in color["rgb"])
        for attribute in selected["desaturation_attributes"]:
            attribute.Set(1.0)
        tint = Gf.Vec3f(*color["rgb"])
        for attribute in selected["tint_attributes"]:
            attribute.Set(tint)
        self._bin_pose = np.asarray(
            (*selected_position, *selected["orientation"]), dtype=np.float64
        )
        self._bin_inner_half_x_m = float(selected["inner_half_x_m"])
        self._bin_inner_half_y_m = float(selected["inner_half_y_m"])
        self._bin_floor_top_m = float(selected["floor_top_m"])
        self._bin_rim_height_m = float(selected["rim_height_m"])
        self._bin_reset_lateral_m = lateral
        self._bin_reset_toward_m = toward
        self._active_bin_prim_path = str(selected["prim_path"])
        print(
            "[bin-pick-place] Randomized bin selected: "
            f"{self.active_bin_name}, color={self.active_bin_color}, "
            f"outer_size={selected['outer_size_m']} m, "
            f"lateral={100.0 * lateral:+.1f} cm, "
            f"toward_robot={100.0 * toward:.1f} cm; "
            "floor + four wall colliders verified."
        )
        return lateral, toward

    def _initialize_object_instances(self):
        instances = []
        for variant in OBJECT_VARIANTS:
            root = self.stage.GetPrimAtPath(variant["prim_path"])
            if not root:
                raise RuntimeError(
                    f"Randomized object prim is missing: {variant['prim_path']}"
                )
            collision_attributes = []
            for child in Usd.PrimRange(root):
                if child.HasAPI(UsdPhysics.CollisionAPI):
                    collision_attributes.append(
                        UsdPhysics.CollisionAPI(child).GetCollisionEnabledAttr()
                    )
            if len(collision_attributes) != 1:
                raise RuntimeError(
                    f"Expected one collider for {variant['name']}, "
                    f"found {len(collision_attributes)}"
                )
            collider_prim = next(
                child
                for child in Usd.PrimRange(root)
                if child.HasAPI(UsdPhysics.CollisionAPI)
            )
            if not collider_prim.IsA(UsdGeom.Mesh):
                raise RuntimeError(
                    f"Expected a mesh collider for {variant['name']}, "
                    f"got {collider_prim.GetTypeName()}"
                )
            points = np.asarray(
                UsdGeom.Mesh(collider_prim).GetPointsAttr().Get(),
                dtype=np.float64,
            )
            if points.ndim != 2 or points.shape[1] != 3 or len(points) < 4:
                raise RuntimeError(
                    f"Invalid collision points for {variant['name']}: "
                    f"shape={points.shape}"
                )
            xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
            collider_to_world = np.asarray(
                xform_cache.GetLocalToWorldTransform(collider_prim),
                dtype=np.float64,
            )
            root_to_world = np.asarray(
                xform_cache.GetLocalToWorldTransform(root), dtype=np.float64
            )
            homogeneous = np.concatenate(
                (points, np.ones((len(points), 1), dtype=np.float64)), axis=1
            )
            local_collision_points = (
                homogeneous @ collider_to_world @ np.linalg.inv(root_to_world)
            )[:, :3]
            instances.append(
                {
                    **variant,
                    "root": root,
                    "imageable": UsdGeom.Imageable(root),
                    "rigid_object": self.env.scene[variant["scene_key"]],
                    "collision_attributes": collision_attributes,
                    "local_collision_points": local_collision_points,
                }
            )
        return instances

    def _select_random_object(self, offset, yaw_delta: float, env_ids) -> None:
        selected_index = int(
            self._object_rng.integers(0, len(self._object_instances))
        )
        zero_velocity = torch.zeros(
            (len(env_ids), 6), dtype=torch.float32, device=self.device
        )

        # Park every variant first. All colliders and native-gravity settings
        # remain authored and stable; inactive objects are hidden and settle
        # on the far-away parking shelf instead of falling indefinitely.
        for instance in self._object_instances:
            instance["imageable"].MakeInvisible()
            rigid_object = instance["rigid_object"]
            state = rigid_object.data.default_root_state[env_ids].clone()
            state[:, :3] += self.env.scene.env_origins[env_ids]
            rigid_object.write_root_pose_to_sim(state[:, :7], env_ids=env_ids)
            rigid_object.write_root_velocity_to_sim(
                zero_velocity, env_ids=env_ids
            )
            zero_wrench = torch.zeros(
                (len(env_ids), 1, 3), dtype=torch.float32, device=self.device
            )
            rigid_object.permanent_wrench_composer.set_forces_and_torques(
                forces=zero_wrench,
                torques=zero_wrench,
                env_ids=env_ids,
                is_global=True,
            )

        selected = self._object_instances[selected_index]
        rigid_object = selected["rigid_object"]
        state = rigid_object.data.default_root_state[env_ids].clone()
        state[:, :3] += self.env.scene.env_origins[env_ids]
        state[:, 0] = (
            float(selected["table_position"][0] + offset[0])
            + self.env.scene.env_origins[env_ids, 0]
        )
        state[:, 1] = (
            float(selected["table_position"][1] + offset[1])
            + self.env.scene.env_origins[env_ids, 1]
        )
        state[:, 2] = (
            float(selected["table_position"][2])
            + self.env.scene.env_origins[env_ids, 2]
        )
        yaw_quaternion = np.asarray(
            [math.cos(0.5 * yaw_delta), 0.0, 0.0, math.sin(0.5 * yaw_delta)],
            dtype=np.float64,
        )
        randomized_orientation = self._quaternion_multiply(
            yaw_quaternion, np.asarray(OBJECT_ORIENTATION, dtype=np.float64)
        )
        state[:, 3:7] = torch.as_tensor(
            randomized_orientation,
            dtype=torch.float32,
            device=self.device,
        ).repeat(len(env_ids), 1)
        rigid_object.write_root_pose_to_sim(state[:, :7], env_ids=env_ids)
        rigid_object.write_root_velocity_to_sim(zero_velocity, env_ids=env_ids)
        zero_wrench = torch.zeros(
            (len(env_ids), 1, 3), dtype=torch.float32, device=self.device
        )
        rigid_object.permanent_wrench_composer.set_forces_and_torques(
            forces=zero_wrench,
            torques=zero_wrench,
            env_ids=env_ids,
            is_global=True,
        )
        selected["imageable"].MakeVisible()

        self.task_object = rigid_object
        self.active_object_name = selected["name"]
        self._object_reset_yaw_rad = float(yaw_delta)
        self._active_object_initial_pose = (
            state[0, :7].detach().cpu().numpy().astype(np.float64)
        )
        self._active_object_asset_path = str(selected["usd_path"])
        self._object_local_collision_points = selected[
            "local_collision_points"
        ]
        print(
            "[bin-pick-place] Randomized object selected: "
            f"{self.active_object_name}, "
            f"size={selected['dimensions_m']} m, "
            f"collision_points={len(self._object_local_collision_points)}."
        )

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
        orientation = self._quaternion_multiply(
            np.asarray(base, dtype=np.float64),
            np.asarray(G1_MOUNT_LOCAL_ORIENTATION, dtype=np.float64),
        ).astype(np.float32)
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
        self._update_success_checker(dt)
        self._benchmark.tick()

    def reset(self) -> None:
        self.x, self.y, self.yaw, self.lift_height = self.initial
        self.track_drive.reset()
        self._write_track_state(0.0, 0.0)
        self.apply()
        print("[bin-pick-place] Base pose and lift reset.")

    def reset_everything(self) -> None:
        env_ids = torch.arange(self.env.num_envs, device=self.device)
        base_mdp.reset_scene_to_default(self.env, env_ids)
        self.reset()
        self.reset_task_object()
        print(
            "[bin-pick-place] R reset complete: robot, base/lift, "
            "randomized bin, and randomized metal component restored."
        )

    def reset_task_object(self) -> None:
        self._clear_success()
        request = self._benchmark.consume_request()
        if request is not None:
            object_seed, bin_seed = np.random.SeedSequence(int(request["seed"])).spawn(2)
            self._object_rng = np.random.default_rng(object_seed)
            self._bin_rng = np.random.default_rng(bin_seed)
        self._select_random_bin()
        env_ids = torch.arange(self.env.num_envs, device=self.device)
        object_lateral, object_toward, object_world_offset = (
            self._sample_table_reset_offset(self._object_rng)
        )
        object_yaw = float(self._object_rng.uniform(-math.pi, math.pi))
        self._object_reset_lateral_m = object_lateral
        self._object_reset_toward_m = object_toward
        self._select_random_object(
            object_world_offset[:2], object_yaw, env_ids
        )
        self.reset_count += 1
        print(
            f"[bin-pick-place] Object reset #{self.reset_count}: "
            f"object lateral={100.0 * object_lateral:+.1f} cm, "
            f"toward_robot={100.0 * object_toward:.1f} cm, "
            f"horizontal_yaw={math.degrees(object_yaw):+.1f} deg; "
            f"bin lateral={100.0 * self._bin_reset_lateral_m:+.1f} cm, "
            f"toward_robot={100.0 * self._bin_reset_toward_m:.1f} cm; "
            f"active object={self.active_object_name}, "
            f"active bin={self.active_bin_name}, "
            f"color={self.active_bin_color}."
        )
        self._benchmark.begin_settle(
            request,
            {
                "active_object": self.active_object_name,
                "active_bin": self.active_bin_name,
                "bin_color": self.active_bin_color,
                "object_lateral_m": float(object_lateral),
                "object_toward_robot_m": float(object_toward),
                "object_yaw_rad": float(object_yaw),
                "bin_lateral_m": float(self._bin_reset_lateral_m),
                "bin_toward_robot_m": float(self._bin_reset_toward_m),
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
    def _rotate_vectors_by_quaternion(
        cls, quaternion: np.ndarray, vectors: np.ndarray
    ) -> np.ndarray:
        quaternion = quaternion / np.linalg.norm(quaternion)
        q_xyz = quaternion[1:]
        qw = quaternion[0]
        cross_once = np.cross(np.broadcast_to(q_xyz, vectors.shape), vectors)
        cross_twice = np.cross(np.broadcast_to(q_xyz, vectors.shape), cross_once)
        return vectors + 2.0 * (qw * cross_once + cross_twice)

    @classmethod
    def _world_points_in_reference(
        cls,
        reference_pose: np.ndarray,
        world_points: np.ndarray,
    ) -> np.ndarray:
        inverse = reference_pose[3:7].copy()
        inverse[1:] *= -1.0
        return cls._rotate_vectors_by_quaternion(
            inverse, world_points - reference_pose[:3]
        )

    def _success_geometry(self):
        pose = (
            self.task_object.data.root_pose_w[0]
            .detach()
            .cpu()
            .numpy()
            .astype(np.float64)
        )
        velocity = (
            self.task_object.data.root_vel_w[0]
            .detach()
            .cpu()
            .numpy()
            .astype(np.float64)
        )
        world_collision_points = pose[:3] + self._rotate_vectors_by_quaternion(
            pose[3:7], self._object_local_collision_points
        )
        bin_collision_points = self._world_points_in_reference(
            self._bin_pose, world_collision_points
        )
        bin_center = self._world_points_in_reference(
            self._bin_pose, pose[:3][np.newaxis, :]
        )[0]

        # The task requires bottom support and stability. Requiring every OBB
        # corner to fit rejected valid diagonal placements of the long parts,
        # so containment is based on the object's center being in the opening.
        xy_inside = bool(
            abs(bin_center[0])
            <= self._bin_inner_half_x_m - BIN_CENTER_INTERIOR_MARGIN_M
            and abs(bin_center[1])
            <= self._bin_inner_half_y_m - BIN_CENTER_INTERIOR_MARGIN_M
        )
        object_bottom_z = float(bin_collision_points[:, 2].min())
        floor_gap = object_bottom_z - self._bin_floor_top_m
        floor_contact = bool(
            -BIN_FLOOR_PENETRATION_TOLERANCE_M
            <= floor_gap
            <= BIN_FLOOR_SUPPORT_TOLERANCE_M
        )
        z_inside = bool(floor_contact and bin_center[2] <= self._bin_rim_height_m)
        linear_speed = float(np.linalg.norm(velocity[:3]))
        angular_speed = float(np.linalg.norm(velocity[3:]))
        stable = bool(
            linear_speed < MAX_LINEAR_SPEED_MPS
            and angular_speed < MAX_ANGULAR_SPEED_RADPS
        )
        return (
            xy_inside,
            z_inside,
            stable,
            bin_center,
            object_bottom_z,
            floor_gap,
            floor_contact,
            linear_speed,
            angular_speed,
        )

    def _update_success_checker(self, dt: float) -> None:
        (
            xy_inside,
            z_inside,
            stable,
            bin_center,
            object_bottom_z,
            floor_gap,
            floor_contact,
            linear_speed,
            angular_speed,
        ) = self._success_geometry()
        candidate = xy_inside and z_inside and stable
        self.success_hold_time = self.success_hold_time + dt if candidate else 0.0
        success = self.success_hold_time >= SUCCESS_HOLD_SECONDS

        now = time.monotonic()
        if now - self._last_success_diagnostic_time >= 0.5:
            self._last_success_diagnostic_time = now
            status = {
                "success": bool(success),
                "reset_count": self.reset_count,
                "active_object": self.active_object_name,
                "active_bin": self.active_bin_name,
                "active_bin_color": self.active_bin_color,
                "xy_inside": bool(xy_inside),
                "z_inside": bool(z_inside),
                "floor_contact": bool(floor_contact),
                "stable": bool(stable),
                "hold_seconds": self.success_hold_time,
                "required_hold_seconds": SUCCESS_HOLD_SECONDS,
                "object_center_in_bin_m": bin_center.tolist(),
                "object_bottom_in_bin_m": object_bottom_z,
                "bin_floor_top_m": self._bin_floor_top_m,
                "floor_gap_m": floor_gap,
                "bin_random_lateral_m": self._bin_reset_lateral_m,
                "bin_random_toward_robot_m": self._bin_reset_toward_m,
                "object_random_lateral_m": self._object_reset_lateral_m,
                "object_random_toward_robot_m": self._object_reset_toward_m,
                "object_random_yaw_rad": self._object_reset_yaw_rad,
                "linear_speed_mps": linear_speed,
                "angular_speed_radps": angular_speed,
                "tracked_base_pose": [self.x, self.y, self.yaw],
                "lift_height_m": self.lift_height,
            }
            SUCCESS_STATUS_PATH.write_text(
                json.dumps(status, indent=2), encoding="utf-8"
            )
            # Keep normal operation quiet. Detailed live diagnostics become
            # useful only once the object has actually crossed the bin's XY
            # opening; z_inside alone is also true while it rests on the table.
            if xy_inside:
                print(
                    "[bin-pick-place] Success check: "
                    f"xy={'OK' if xy_inside else 'OUT'}, "
                    f"z={'OK' if z_inside else 'OUT'}, "
                    f"floor_contact={'YES' if floor_contact else 'NO'}, "
                    f"bottom={object_bottom_z:.3f}/"
                    f"{self._bin_floor_top_m:.3f} m, "
                    f"gap={1000.0 * floor_gap:+.1f} mm, "
                    f"linear={linear_speed:.3f}/{MAX_LINEAR_SPEED_MPS:.3f} m/s, "
                    f"angular={angular_speed:.3f}/{MAX_ANGULAR_SPEED_RADPS:.3f} rad/s, "
                    f"hold={self.success_hold_time:.2f}/{SUCCESS_HOLD_SECONDS:.2f} s."
                )

        if not success or self.success_latched:
            return
        self.success_latched = True
        SUCCESS_MARKER_PATH.write_text(
            json.dumps(
                {
                    "success": True,
                    "reset_count": self.reset_count,
                    "active_object": self.active_object_name,
                    "active_bin": self.active_bin_name,
                    "active_bin_color": self.active_bin_color,
                    "object_center_in_bin_m": bin_center.tolist(),
                    "object_bottom_in_bin_m": object_bottom_z,
                    "bin_floor_top_m": self._bin_floor_top_m,
                    "floor_gap_m": floor_gap,
                    "floor_contact": True,
                    "linear_speed_mps": linear_speed,
                    "angular_speed_radps": angular_speed,
                    "stable_seconds": self.success_hold_time,
                    "timestamp": datetime.now().astimezone().isoformat(),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print("\n" + "=" * 72)
        print("SUCCESS")
        print(
            f"[bin-pick-place] {self.active_object_name} is inside "
            f"{self.active_bin_color} {self.active_bin_name}, touching its "
            "physical floor, "
            f"and stable for {self.success_hold_time:.2f} s."
        )
        print(
            "[bin-pick-place] Simulation, commands, and new camera frames "
            "are frozen. Press right [A] to save or terminal [s] to discard."
        )
        print("=" * 72 + "\n")

    @property
    def is_success_frozen(self) -> bool:
        return self.success_latched

    def get_recording_state(self) -> dict:
        """Expose a complete reset snapshot for deterministic episode replay."""
        robot_position, robot_orientation = self._robot_pose()
        if self.task_object is None:
            object_pose = self._active_object_initial_pose.copy()
            object_velocity = np.zeros(6, dtype=np.float64)
        else:
            object_pose = (
                self.task_object.data.root_pose_w[0]
                .detach()
                .cpu()
                .numpy()
                .astype(np.float64)
            )
            object_velocity = (
                self.task_object.data.root_vel_w[0]
                .detach()
                .cpu()
                .numpy()
                .astype(np.float64)
            )
        return {
            "schema": REPLAY_STATE_SCHEMA,
            "task_id": TASK_ID,
            "timestamp_ns": time.time_ns(),
            "reset_count": int(self.reset_count),
            "tracked_base_pose": [self.x, self.y, self.yaw],
            "lift_height_m": self.lift_height,
            "active_object": self.active_object_name,
            "active_bin": self.active_bin_name,
            "active_bin_color": self.active_bin_color,
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
                    "name": self.active_object_name,
                    "asset_path": self._active_object_asset_path,
                    "root_pose_wxyz": object_pose.tolist(),
                    "root_velocity_linear_angular": object_velocity.tolist(),
                    "authored_reset_pose_wxyz": (
                        self._active_object_initial_pose.tolist()
                    ),
                    "random_lateral_m": self._object_reset_lateral_m,
                    "random_toward_robot_m": self._object_reset_toward_m,
                    "random_yaw_rad": self._object_reset_yaw_rad,
                },
                "bin": {
                    "name": self.active_bin_name,
                    "prim_path": self._active_bin_prim_path,
                    "color": self.active_bin_color,
                    "color_rgb": list(self._active_bin_color_rgb),
                    "root_pose_wxyz": self._bin_pose.tolist(),
                    "random_lateral_m": self._bin_reset_lateral_m,
                    "random_toward_robot_m": self._bin_reset_toward_m,
                    "inner_half_extents_xy_m": [
                        self._bin_inner_half_x_m,
                        self._bin_inner_half_y_m,
                    ],
                    "floor_top_m": self._bin_floor_top_m,
                    "rim_height_m": self._bin_rim_height_m,
                },
                "object_catalog": [
                    {
                        "name": str(variant["name"]),
                        "asset_path": str(variant["usd_path"]),
                        "dimensions_m": [
                            float(value) for value in variant["dimensions_m"]
                        ],
                    }
                    for variant in OBJECT_VARIANTS
                ],
                "bin_catalog": [
                    {
                        "name": str(variant["name"]),
                        "prim_path": str(variant["prim_path"]),
                    }
                    for variant in BIN_VARIANTS
                ],
            },
        }

    def _clear_success(self) -> None:
        was_latched = self.success_latched
        self.success_latched = False
        self.success_hold_time = 0.0
        SUCCESS_MARKER_PATH.unlink(missing_ok=True)
        SUCCESS_STATUS_PATH.unlink(missing_ok=True)
        if was_latched:
            print("[bin-pick-place] Success freeze cleared by reset.")

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
