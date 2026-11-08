"""Slow teleoperation controller with a synchronized animated rolling cart."""

from __future__ import annotations

import json
import math
import os
import sys
import threading
import time

import numpy as np
import torch
from pxr import Gf, UsdGeom

import carb
import carb.input
import omni.appwindow
import omni.usd
from isaacsim.core.prims import RigidPrim
from unitree_sdk2py.core.channel import ChannelSubscriber
from unitree_sdk2py.idl.std_msgs.msg.dds_ import String_

import isaaclab.envs.mdp as base_mdp
from tasks.g1_tasks.tabletop_grasp_g1_29dof_dex1.tabletop_grasp_g1_29dof_dex1_env_cfg import (
    ASSETS_ROOT,
    G1_MOUNT_LOCAL_ORIENTATION,
    W1_UPPER_MOUNT_LOCAL,
)
from .hand_cart_rolling_demo_g1_29dof_dex1_env_cfg import (
    CART_ORIENTATION,
    CART_POSITION,
    DEMO_MAX_TRACK_ACCEL_MPS2,
    DEMO_MAX_TRACK_SPEED_MPS,
    TASK_FORWARD_X,
    TASK_FORWARD_Y,
    TASK_ID,
    TASK_TRACKED_BASE_X,
    TASK_TRACKED_BASE_Y,
    TASK_TRACKED_BASE_YAW,
    TASK_TRACKED_LIFT_HEIGHT,
    WHEEL_RADIUS_M,
)


W1_TOOLS = ASSETS_ROOT / "robot/Forklift_W1_Final/tools"
if str(W1_TOOLS) not in sys.path:
    sys.path.insert(0, str(W1_TOOLS))
from differential_drive import DifferentialTrackKinematics  # noqa: E402


QUEST_BASE_COMMAND_TOPIC = "rt/hand_cart_base/cmd"
QUEST_BASE_COMMAND_TIMEOUT_S = float(os.environ.get("G1_HAND_CART_QUEST_COMMAND_TIMEOUT_S", "0.25"))


class HandCartRollingDemoController:
    """Move the base and cart together while rotating only the cart's wheels.

    This is intentionally kinematic presentation logic.  It does not model a
    robot-to-handle contact force, does not use a success condition, and does
    not write a velocity or pose into the source physical cart asset.
    """

    ROOT_PATH = "/World/TrackedBase"
    VEHICLE_PATH = f"{ROOT_PATH}/Vehicle"
    CARRIAGE_PATH = f"{ROOT_PATH}/LiftCarriage"
    CART_PATH = "/World/envs/env_0/HandCart"
    CART_BODY_PATH = CART_PATH

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
        self._cart_position = np.asarray(CART_POSITION, dtype=np.float64)
        self._cart_orientation = np.asarray(CART_ORIENTATION, dtype=np.float64)
        self._wheel_angles_rad = {"Wheel_Left": 0.0, "Wheel_Right": 0.0}
        self._telemetry_sequence = 0
        self._left_target_velocity = 0.0
        self._right_target_velocity = 0.0
        self._lift_target_velocity = 0.0
        self.held: set[carb.input.KeyboardInput] = set()
        self._input = None
        self._keyboard = None
        self._subscription = None
        self._quest_base_command_lock = threading.Lock()
        self._quest_forward_axis = 0.0
        self._quest_command_received_at = 0.0

        root = self.stage.GetPrimAtPath(self.ROOT_PATH)
        vehicle = self.stage.GetPrimAtPath(self.VEHICLE_PATH)
        carriage = self.stage.GetPrimAtPath(self.CARRIAGE_PATH)
        cart = self.stage.GetPrimAtPath(self.CART_PATH)
        if not root or not vehicle or not carriage or not cart:
            raise RuntimeError(
                "Rolling-demo hierarchy is incomplete: "
                f"base={bool(root)}, vehicle={bool(vehicle)}, carriage={bool(carriage)}, cart={bool(cart)}"
            )
        self.root_translate_op = self._get_or_add_op(UsdGeom.Xformable(root), "translate")
        self.root_orient_op = self._get_or_add_op(UsdGeom.Xformable(root), "orient")
        self.carriage_translate_op = self._get_or_add_op(UsdGeom.Xformable(carriage), "translate")
        self.cart_body = RigidPrim(
            prim_paths_expr=self.CART_BODY_PATH,
            name="rolling_demo_cart_chassis",
            reset_xform_properties=False,
        )
        self.cart_body.initialize()
        if not self.cart_body.is_physics_handle_valid():
            raise RuntimeError("Rolling-demo cart chassis physics handle did not initialize")
        self.wheel_orient_ops = {}
        for wheel_name in self._wheel_angles_rad:
            wheel = self.stage.GetPrimAtPath(f"{self.CART_PATH}/{wheel_name}")
            if not wheel:
                raise RuntimeError(f"Rolling-demo cart is missing {wheel_name}")
            self.wheel_orient_ops[wheel_name] = self._get_or_add_op(
                UsdGeom.Xformable(wheel), "orient"
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
            for name in ("height", "targetHeight", "lowerLimit", "upperLimit", "maxSpeed")
        }
        missing = [name for name, attr in self.lift_attributes.items() if not attr]
        if missing:
            raise RuntimeError(f"Tracked-lift attributes are missing: {missing}")

        self.lift_lower = float(self.lift_attributes["lowerLimit"].Get())
        self.lift_upper = float(self.lift_attributes["upperLimit"].Get())
        self.lift_speed = float(self.lift_attributes["maxSpeed"].Get())
        self.track_drive = DifferentialTrackKinematics(
            track_separation=float(self.track_attributes["trackSeparation"].Get()),
            max_track_speed=min(DEMO_MAX_TRACK_SPEED_MPS, float(self.track_attributes["maxTrackSpeed"].Get())),
            max_track_acceleration=min(DEMO_MAX_TRACK_ACCEL_MPS2, float(self.track_attributes["maxTrackAcceleration"].Get())),
            heading_offset=float(self.track_attributes["headingOffsetRadians"].Get()),
        )
        self.vehicle_body = RigidPrim(prim_paths_expr=self.VEHICLE_PATH, name="rolling_demo_tracked_vehicle", reset_xform_properties=False)
        self.carriage_body = RigidPrim(prim_paths_expr=self.CARRIAGE_PATH, name="rolling_demo_lift_carriage", reset_xform_properties=False)
        self.vehicle_body.initialize()
        self.carriage_body.initialize()
        if not self.vehicle_body.is_physics_handle_valid() or not self.carriage_body.is_physics_handle_valid():
            raise RuntimeError("Tracked base physics handles did not initialize")

        if QUEST_BASE_COMMAND_TIMEOUT_S <= 0.0:
            raise ValueError("G1_HAND_CART_QUEST_COMMAND_TIMEOUT_S must be positive")
        self._quest_base_subscriber = ChannelSubscriber(QUEST_BASE_COMMAND_TOPIC, String_)
        self._quest_base_subscriber.Init(self._on_quest_base_command, 1)
        self._subscribe_keyboard()
        self.apply()
        self._write_cart_animation()
        print(
            "[hand-cart-rolling-demo] Ready: W/S drive, A/D turn, T/Y lift, R reset. "
            f"Cart sync is kinematic; wheel radius={WHEEL_RADIUS_M:.4f} m; max speed={self.track_drive.max_track_speed:.2f} m/s."
        )

    @staticmethod
    def _get_or_add_op(xform: UsdGeom.Xformable, kind: str):
        name = f"xformOp:{kind}"
        found = next((op for op in xform.GetOrderedXformOps() if op.GetOpName() == name), None)
        if found is not None:
            return found
        if kind == "translate":
            return xform.AddTranslateOp()
        if kind == "orient":
            return xform.AddOrientOp(UsdGeom.XformOp.PrecisionDouble)
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
                if event.type in (carb.input.KeyboardEventType.KEY_PRESS, carb.input.KeyboardEventType.KEY_REPEAT):
                    self.held.add(key)
                    if key == carb.input.KeyboardInput.R and event.type == carb.input.KeyboardEventType.KEY_PRESS:
                        self.reset_everything()
                elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
                    self.held.discard(key)
                return True

            self._subscription = self._input.subscribe_to_keyboard_events(self._keyboard, on_key)
        except Exception as error:
            print(f"[hand-cart-rolling-demo] Keyboard input unavailable: {error}")

    def _on_quest_base_command(self, message: String_) -> None:
        try:
            payload = json.loads(message.data)
            if payload.get("schema") != "hand_cart_base_command_v1" or payload.get("task_id") != TASK_ID:
                return
            forward_axis = float(payload["forward_axis"])
            if not math.isfinite(forward_axis):
                return
            with self._quest_base_command_lock:
                self._quest_forward_axis = float(np.clip(forward_axis, -1.0, 1.0))
                self._quest_command_received_at = time.monotonic()
        except (AttributeError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            return

    def _fresh_quest_forward_axis(self) -> float:
        with self._quest_base_command_lock:
            if time.monotonic() - self._quest_command_received_at > QUEST_BASE_COMMAND_TIMEOUT_S:
                return 0.0
            return self._quest_forward_axis

    def _clear_quest_base_command(self) -> None:
        with self._quest_base_command_lock:
            self._quest_forward_axis = 0.0
            self._quest_command_received_at = 0.0

    def _robot_pose(self) -> tuple[np.ndarray, np.ndarray]:
        mount_x, mount_y, mount_z = W1_UPPER_MOUNT_LOCAL
        cosine, sine = math.cos(self.yaw), math.sin(self.yaw)
        position = np.asarray([
            self.x + cosine * mount_x - sine * mount_y,
            self.y + sine * mount_x + cosine * mount_y,
            mount_z + self.lift_height,
        ], dtype=np.float32)
        base = (math.cos(0.5 * self.yaw), 0.0, 0.0, math.sin(0.5 * self.yaw))
        bw, bx, by, bz = base
        mw, mx, my, mz = G1_MOUNT_LOCAL_ORIENTATION
        orientation = np.asarray([
            bw * mw - bx * mx - by * my - bz * mz,
            bw * mx + bx * mw + by * mz - bz * my,
            bw * my - bx * mz + by * mw + bz * mx,
            bw * mz + bx * my - by * mx + bz * mw,
        ], dtype=np.float32)
        return position, orientation

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

    def apply(self) -> None:
        half_yaw = 0.5 * self.yaw
        orientation = np.asarray([math.cos(half_yaw), 0.0, 0.0, math.sin(half_yaw)], dtype=np.float32)
        self.root_translate_op.Set(Gf.Vec3d(self.x, self.y, 0.0))
        self.root_orient_op.Set(Gf.Quatd(float(orientation[0]), Gf.Vec3d(0.0, 0.0, float(orientation[3]))) )
        self.carriage_translate_op.Set(Gf.Vec3d(0.0, 0.0, self.lift_height))
        self.vehicle_body.set_world_poses(positions=torch.tensor([[self.x, self.y, 0.0]], dtype=torch.float32, device=self.device), orientations=torch.as_tensor(orientation[np.newaxis, :], dtype=torch.float32, device=self.device))
        self.carriage_body.set_world_poses(positions=torch.tensor([[self.x, self.y, self.lift_height]], dtype=torch.float32, device=self.device), orientations=torch.as_tensor(orientation[np.newaxis, :], dtype=torch.float32, device=self.device))
        robot_position, robot_orientation = self._robot_pose()
        root_pose = torch.as_tensor(np.concatenate((robot_position, robot_orientation))[np.newaxis, :], dtype=torch.float32, device=self.device)
        self.robot.write_root_pose_to_sim(root_pose)
        self.robot.write_root_velocity_to_sim(torch.zeros((1, 6), dtype=torch.float32, device=self.device))
        self.lift_attributes["height"].Set(float(self.lift_height))
        self.lift_attributes["targetHeight"].Set(float(self.lift_height))

    def _write_cart_animation(self) -> None:
        # The cart root is the only kinematic rigid body.  Its visual chassis,
        # handle colliders and visual wheels therefore share one world pose.
        self.cart_body.set_world_poses(
            positions=torch.as_tensor(
                self._cart_position[np.newaxis, :], dtype=torch.float32, device=self.device
            ),
            orientations=torch.as_tensor(
                self._cart_orientation[np.newaxis, :], dtype=torch.float32, device=self.device
            ),
        )
        for wheel_name, angle in self._wheel_angles_rad.items():
            wrapped = math.atan2(math.sin(angle), math.cos(angle))
            self.wheel_orient_ops[wheel_name].Set(
                Gf.Quatd(math.cos(0.5 * wrapped), Gf.Vec3d(math.sin(0.5 * wrapped), 0.0, 0.0))
            )

    def update(self, dt: float) -> None:
        dt = float(np.clip(dt, 1.0e-4, 0.1))
        keyboard_throttle = float((carb.input.KeyboardInput.W in self.held) - (carb.input.KeyboardInput.S in self.held))
        throttle = float(np.clip(keyboard_throttle + self._fresh_quest_forward_axis(), -1.0, 1.0))
        steering = float((carb.input.KeyboardInput.A in self.held) - (carb.input.KeyboardInput.D in self.held))
        left_axis, right_axis = throttle - steering, throttle + steering
        normalization = max(1.0, abs(left_axis), abs(right_axis))
        self._left_target_velocity = self.track_drive.max_track_speed * left_axis / normalization
        self._right_target_velocity = self.track_drive.max_track_speed * right_axis / normalization

        previous_x, previous_y = self.x, self.y
        previous_left_distance = self.track_drive.left_distance
        previous_right_distance = self.track_drive.right_distance
        self.x, self.y, self.yaw = self.track_drive.step(self.x, self.y, self.yaw, self._left_target_velocity, self._right_target_velocity, dt)
        # The cart shares the base's exact planar displacement; wheel spin uses
        # each physical track path, including opposite rotation during turns.
        self._cart_position[0] += self.x - previous_x
        self._cart_position[1] += self.y - previous_y
        # Keep the validated wheel convention used by this demo.  Positive
        # track travel maps to positive local-X rotation for both wheels.
        self._wheel_angles_rad["Wheel_Left"] += (self.track_drive.left_distance - previous_left_distance) / WHEEL_RADIUS_M
        self._wheel_angles_rad["Wheel_Right"] += (self.track_drive.right_distance - previous_right_distance) / WHEEL_RADIUS_M
        self._write_track_state(self._left_target_velocity, self._right_target_velocity)

        lift_direction = float((carb.input.KeyboardInput.T in self.held) - (carb.input.KeyboardInput.Y in self.held))
        self._lift_target_velocity = lift_direction * self.lift_speed
        self.lift_height = float(np.clip(self.lift_height + self._lift_target_velocity * dt, self.lift_lower, self.lift_upper))
        self.apply()
        self._write_cart_animation()
        self._telemetry_sequence += 1

    def reset_cart(self) -> None:
        self._cart_position = np.asarray(CART_POSITION, dtype=np.float64)
        self._cart_orientation = np.asarray(CART_ORIENTATION, dtype=np.float64)
        self._wheel_angles_rad = {"Wheel_Left": 0.0, "Wheel_Right": 0.0}
        self._write_cart_animation()
        print("[hand-cart-rolling-demo] Cart reset: synchronized pose and wheel angles restored.")

    def reset(self) -> None:
        self._clear_quest_base_command()
        self.x, self.y, self.yaw, self.lift_height = self.initial
        self.track_drive.reset()
        self._left_target_velocity = 0.0
        self._right_target_velocity = 0.0
        self._lift_target_velocity = 0.0
        self._write_track_state(0.0, 0.0)
        self.apply()

    def reset_everything(self) -> None:
        env_ids = torch.arange(self.env.num_envs, device=self.device)
        base_mdp.reset_scene_to_default(self.env, env_ids)
        self.reset()
        self.reset_cart()
        print("[hand-cart-rolling-demo] R reset complete: robot, base, cart and wheels restored.")

    def get_recording_state(self) -> dict:
        dx, dy = self.x - self.initial[0], self.y - self.initial[1]
        forward = dx * TASK_FORWARD_X + dy * TASK_FORWARD_Y
        lateral = dx * -TASK_FORWARD_Y + dy * TASK_FORWARD_X
        yaw_delta = math.atan2(math.sin(self.yaw - self.initial[2]), math.cos(self.yaw - self.initial[2]))
        return {
            "schema": "tracked_base_v1",
            "task_id": TASK_ID,
            "sequence": int(self._telemetry_sequence),
            "timestamp_ns": time.time_ns(),
            "replay_initial_state": {
                "objects": {
                    "hand_cart": {
                        "root_pose_wxyz": [
                            float(self._cart_position[0]),
                            float(self._cart_position[1]),
                            float(self._cart_position[2]),
                            float(self._cart_orientation[0]),
                            float(self._cart_orientation[1]),
                            float(self._cart_orientation[2]),
                            float(self._cart_orientation[3]),
                        ]
                    }
                },
                "robot": {
                    "tracked_base_pose_xy_yaw": [float(self.x), float(self.y), float(self.yaw)],
                    "lift_height_m": float(self.lift_height),
                },
            },
            "state": {
                "base_forward_position_m": float(forward),
                "base_lateral_position_m": float(lateral),
                "base_yaw_delta_rad": float(yaw_delta),
                "lift_height_m": float(self.lift_height),
                "left_track_velocity_mps": float(self.track_drive.left_velocity),
                "right_track_velocity_mps": float(self.track_drive.right_velocity),
                "lift_velocity_mps": float(self._lift_target_velocity),
                "cart_world_x_m": float(self._cart_position[0]),
                "cart_world_y_m": float(self._cart_position[1]),
                "cart_world_z_m": float(self._cart_position[2]),
                "cart_world_qw": float(self._cart_orientation[0]),
                "cart_world_qx": float(self._cart_orientation[1]),
                "cart_world_qy": float(self._cart_orientation[2]),
                "cart_world_qz": float(self._cart_orientation[3]),
                "wheel_left_angle_rad": float(self._wheel_angles_rad["Wheel_Left"]),
                "wheel_right_angle_rad": float(self._wheel_angles_rad["Wheel_Right"]),
            },
            "action": {
                "left_track_velocity_target_mps": float(self._left_target_velocity),
                "right_track_velocity_target_mps": float(self._right_target_velocity),
                "lift_velocity_target_mps": float(self._lift_target_velocity),
            },
            "demo": {
                "cart_mode": "kinematic_synchronized_visual",
                "wheel_radius_m": WHEEL_RADIUS_M,
                "distance_success_check": False,
            },
        }

    @property
    def is_terminal_frozen(self) -> bool:
        return False

    def close(self) -> None:
        self._clear_quest_base_command()
        if self._quest_base_subscriber is not None:
            try:
                self._quest_base_subscriber.Close()
            except Exception:
                pass
            self._quest_base_subscriber = None
        if self._input is not None and self._subscription is not None:
            try:
                self._input.unsubscribe_to_keyboard_events(self._keyboard, self._subscription)
            except Exception:
                pass
        self._subscription = None
        self.held.clear()
