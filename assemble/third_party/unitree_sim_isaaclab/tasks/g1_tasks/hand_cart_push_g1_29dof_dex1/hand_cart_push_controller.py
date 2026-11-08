"""Tracked-base controls, cart reset, telemetry, and success checking."""

from __future__ import annotations

import json
import math
import os
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from pxr import Gf, UsdGeom

import carb
import carb.input
import omni.appwindow
import omni.usd
import isaaclab.envs.mdp as base_mdp
from isaacsim.core.prims import RigidPrim
from omni.physx import get_physx_scene_query_interface
from unitree_sdk2py.core.channel import ChannelSubscriber
from unitree_sdk2py.idl.std_msgs.msg.dds_ import String_

from tasks.g1_tasks.tabletop_grasp_g1_29dof_dex1.tabletop_grasp_g1_29dof_dex1_env_cfg import (
    ASSETS_ROOT,
    G1_MOUNT_LOCAL_ORIENTATION,
    W1_UPPER_MOUNT_LOCAL,
)
from tasks.g1_tasks.benchmark_protocol import BenchmarkProtocol
from .hand_cart_push_g1_29dof_dex1_env_cfg import (
    CART_ORIENTATION,
    CART_POSITION,
    TASK_FORWARD_X,
    TASK_FORWARD_Y,
    TASK_ID,
    TASK_ROOT,
    TASK_TRACKED_BASE_X,
    TASK_TRACKED_BASE_Y,
    TASK_TRACKED_BASE_YAW,
    TASK_TRACKED_LIFT_HEIGHT,
)


W1_TOOLS = ASSETS_ROOT / "robot/Forklift_W1_Final/tools"
SUCCESS_MARKER_PATH = Path(
    os.environ.get(
        "G1_TASK_SUCCESS_FILE",
        f"/tmp/g1_hand_cart_push_{os.getuid()}.success",
    )
)
FAILURE_MARKER_PATH = Path(
    os.environ.get(
        "G1_TASK_FAILURE_FILE",
        f"/tmp/g1_hand_cart_push_{os.getuid()}.failure",
    )
)
MIN_FORWARD_PROGRESS_M = 0.50
MAX_LATERAL_DRIFT_M = 0.20
MIN_UPRIGHT_DOT = math.cos(math.radians(15.0))
POST_PUSH_HOLD_SECONDS = 0.50
GRIPPER_MAX_DISTAL_GAP_M = float(
    os.environ.get("G1_HAND_CART_GRIPPER_MAX_GAP_M", "0.055")
)
GRIPPER_CENTER_PROBE_RADIUS_M = float(
    os.environ.get("G1_HAND_CART_GRASP_PROBE_RADIUS_M", "0.004")
)
DUAL_GRASP_CONFIRM_FRAMES = int(
    os.environ.get("G1_HAND_CART_GRASP_CONFIRM_FRAMES", "5")
)
GRASP_DEBUG_INTERVAL_S = float(
    os.environ.get("G1_HAND_CART_GRASP_DEBUG_INTERVAL_S", "1.0")
)
GRASP_DIAGNOSTICS_PATH = Path(
    os.environ.get(
        "G1_HAND_CART_GRASP_DIAGNOSTICS_FILE",
        f"/tmp/g1_hand_cart_push_{os.getuid()}_grasp.jsonl",
    )
)
DUAL_GRASP_TEMPLATE_PATH = Path(
    os.environ.get(
        "G1_HAND_CART_GRASP_TEMPLATE_FILE",
        str(TASK_ROOT / "config/dual_grasp_template.json"),
    )
)
QUEST_BASE_COMMAND_TOPIC = "rt/hand_cart_base/cmd"
QUEST_BASE_COMMAND_TIMEOUT_S = float(
    os.environ.get("G1_HAND_CART_QUEST_COMMAND_TIMEOUT_S", "0.25")
)

if str(W1_TOOLS) not in sys.path:
    sys.path.insert(0, str(W1_TOOLS))
from differential_drive import DifferentialTrackKinematics  # noqa: E402


class HandCartPushController:
    """Keep the existing base controls and manage the sliding hand cart."""

    ROOT_PATH = "/World/TrackedBase"
    VEHICLE_PATH = f"{ROOT_PATH}/Vehicle"
    CARRIAGE_PATH = f"{ROOT_PATH}/LiftCarriage"

    def __init__(self, env) -> None:
        self.env = env
        self.robot = env.scene["robot"]
        self.hand_cart = env.scene["hand_cart"]
        self.device = env.device
        self.stage = omni.usd.get_context().get_stage()

        self.x = float(TASK_TRACKED_BASE_X)
        self.y = float(TASK_TRACKED_BASE_Y)
        self.yaw = float(TASK_TRACKED_BASE_YAW)
        self.lift_height = float(TASK_TRACKED_LIFT_HEIGHT)
        self.initial = (self.x, self.y, self.yaw, self.lift_height)
        self.cart_random_xy_limit = float(
            os.environ.get("G1_HAND_CART_RANDOM_XY_M", "0.05")
        )
        if self.cart_random_xy_limit < 0.0:
            raise ValueError("G1_HAND_CART_RANDOM_XY_M must be non-negative")
        self.cart_random_yaw_limit = math.radians(3.0)
        seed = int(getattr(getattr(env, "cfg", None), "seed", 0) or 0)
        self._object_rng = np.random.default_rng(seed)
        self._benchmark = BenchmarkProtocol()
        self._cart_reset_position = np.asarray(CART_POSITION, dtype=np.float64)
        self._cart_reset_orientation = np.asarray(CART_ORIENTATION, dtype=np.float64)

        self.success_latched = False
        self.success_forward_progress = 0.0
        self.success_lateral_drift = 0.0
        self.held: set[carb.input.KeyboardInput] = set()
        self._input = None
        self._keyboard = None
        self._subscription = None
        self._telemetry_sequence = 0
        self._left_target_velocity = 0.0
        self._right_target_velocity = 0.0
        self._lift_target_velocity = 0.0
        replay_aux_path = os.environ.get("G1_REPLAY_AUX_ACTION_FILE", "").strip()
        self._replay_aux_action_file = Path(replay_aux_path) if replay_aux_path else None
        self._last_replay_aux_sequence = -1
        # The tracked base is pose-driven while the source cart has no real
        # wheel joints.  Keep a controlled cart pose and, during forward base
        # motion, apply exactly the same XY displacement to both bodies.  This
        # preserves their relative distance and prevents gripper/cart
        # tunnelling caused by two different motion models.
        self._cart_sync_position = self._cart_reset_position.copy()
        self._cart_sync_orientation = self._cart_reset_orientation.copy()
        self._cart_forward_sync_active = False
        self._phase = "acquiring_grasp"
        self._failure_latched = False
        self._failure_reason = None
        self._dual_grasp_valid_now = False
        self._dual_grasp_frames = 0
        self._push_started_at = None
        self._post_push_hold_elapsed = 0.0
        self._gripper_hold_targets = None
        self._last_upright_dot = 1.0

        if GRIPPER_MAX_DISTAL_GAP_M <= 0.0:
            raise ValueError("G1_HAND_CART_GRIPPER_MAX_GAP_M must be positive")
        if GRIPPER_CENTER_PROBE_RADIUS_M <= 0.0:
            raise ValueError("G1_HAND_CART_GRASP_PROBE_RADIUS_M must be positive")
        if DUAL_GRASP_CONFIRM_FRAMES < 1:
            raise ValueError("G1_HAND_CART_GRASP_CONFIRM_FRAMES must be at least 1")
        if GRASP_DEBUG_INTERVAL_S < 0.0:
            raise ValueError("G1_HAND_CART_GRASP_DEBUG_INTERVAL_S must be non-negative")
        if QUEST_BASE_COMMAND_TIMEOUT_S <= 0.0:
            raise ValueError("G1_HAND_CART_QUEST_COMMAND_TIMEOUT_S must be positive")

        # A contact event is not enough to call the cart grasped.  For each
        # hand, sample the corridor halfway between the two opposing fingers
        # and require cart collision geometry to occupy that corridor while
        # both actual prismatic finger joints are closed.  Link2/Link3 give us
        # several stations along the finger pads, so a handle may be held near
        # the palm, pad center, or fingertip without accepting one-sided touch.
        body_names = tuple(self.robot.data.body_names)
        joint_names = tuple(self.robot.data.joint_names)
        self._gripper_body_indices = {
            side: tuple(
                body_names.index(f"{side}_hand_Link{branch}_{segment}")
                for segment in (2, 3)
                for branch in (1, 2)
            )
            for side in ("left", "right")
        }
        self._gripper_joint_indices = {
            side: tuple(
                joint_names.index(f"{side}_hand_Joint{branch}_1")
                for branch in (1, 2)
            )
            for side in ("left", "right")
        }
        self._scene_query = get_physx_scene_query_interface()
        self._dual_grasp_template = self._load_dual_grasp_template()
        self._dual_grasp_frames = 0
        self._left_grasp_status = self._empty_grasp_status()
        self._right_grasp_status = self._empty_grasp_status()
        self._last_grasp_debug_time = 0.0
        self._quest_base_command_lock = threading.Lock()
        self._quest_forward_axis = 0.0
        self._quest_command_received_at = 0.0
        self._quest_command_sequence = -1
        self._quest_base_subscriber = ChannelSubscriber(
            QUEST_BASE_COMMAND_TOPIC, String_
        )
        self._quest_base_subscriber.Init(self._on_quest_base_command, 1)

        root = self.stage.GetPrimAtPath(self.ROOT_PATH)
        vehicle = self.stage.GetPrimAtPath(self.VEHICLE_PATH)
        carriage = self.stage.GetPrimAtPath(self.CARRIAGE_PATH)
        if not root or not vehicle or not carriage:
            raise RuntimeError(
                "Tracked base hierarchy is incomplete: "
                f"root={bool(root)}, vehicle={bool(vehicle)}, carriage={bool(carriage)}"
            )

        self.root_translate_op = self._get_or_add_op(UsdGeom.Xformable(root), "translate")
        self.root_orient_op = self._get_or_add_op(UsdGeom.Xformable(root), "orient")
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
            name="hand_cart_tracked_vehicle",
            reset_xform_properties=False,
        )
        self.carriage_body = RigidPrim(
            prim_paths_expr=self.CARRIAGE_PATH,
            name="hand_cart_lift_carriage",
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
            "[hand-cart-push] Controller ready. W/S drive, A/D turn, "
            "T/Y lift, Quest left Y/X drive forward/reverse, "
            "R resets robot and cart."
        )
        print(
            "[hand-cart-push] Dual-handle grasp template loaded: "
            f"{DUAL_GRASP_TEMPLATE_PATH}"
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
            print(f"[hand-cart-push] Keyboard input unavailable: {error}")

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

    def _on_quest_base_command(self, message: String_) -> None:
        """Receive task-scoped Quest base commands on a DDS callback thread."""

        try:
            payload = json.loads(message.data)
            if payload.get("schema") != "hand_cart_base_command_v1":
                return
            if payload.get("task_id") != TASK_ID:
                return
            forward_axis = float(payload["forward_axis"])
            sequence = int(payload["sequence"])
            if not math.isfinite(forward_axis):
                return
            with self._quest_base_command_lock:
                self._quest_forward_axis = float(
                    np.clip(forward_axis, -1.0, 1.0)
                )
                self._quest_command_received_at = time.monotonic()
                self._quest_command_sequence = sequence
        except (AttributeError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            return

    def _fresh_quest_forward_axis(self) -> float:
        with self._quest_base_command_lock:
            age = time.monotonic() - self._quest_command_received_at
            if age > QUEST_BASE_COMMAND_TIMEOUT_S:
                return 0.0
            return self._quest_forward_axis

    def _clear_quest_base_command(self) -> None:
        with self._quest_base_command_lock:
            self._quest_forward_axis = 0.0
            self._quest_command_received_at = 0.0

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
            [math.cos(half_yaw), 0.0, 0.0, math.sin(half_yaw)], dtype=np.float32
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

    def _read_replay_aux_action(self):
        """Read Task04 track/lift targets written by the replay publisher."""

        if self._replay_aux_action_file is None:
            return None
        try:
            payload = json.loads(
                self._replay_aux_action_file.read_text(encoding="utf-8")
            )
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            return None
        sequence = int(payload.get("sequence", -1))
        values = payload.get("values")
        if not isinstance(values, list) or len(values) != 3:
            return None
        vector = np.asarray(values, dtype=np.float64)
        if not np.all(np.isfinite(vector)):
            return None
        self._last_replay_aux_sequence = max(self._last_replay_aux_sequence, sequence)
        vector[0:2] = np.clip(
            vector[0:2],
            -self.track_drive.max_track_speed,
            self.track_drive.max_track_speed,
        )
        vector[2] = np.clip(vector[2], -self.lift_speed, self.lift_speed)
        return tuple(float(value) for value in vector)

    def update(self, dt: float) -> None:
        if self.is_terminal_frozen:
            return
        benchmark_active = self._benchmark.active
        if benchmark_active:
            self.held.clear()
        dt = float(np.clip(dt, 1.0e-4, 0.1))
        if benchmark_active:
            dual_grasp_confirmed = self._update_dual_grasp_status()
            if self._phase in ("pushing", "post_push_hold") and not self._dual_grasp_valid_now:
                _, forward, lateral, upright_dot = self._success_geometry()
                self.success_forward_progress = forward
                self.success_lateral_drift = lateral
                self._last_upright_dot = upright_dot
                self._mark_failure(
                    "grasp_lost_during_push"
                    if self._phase == "pushing"
                    else "grasp_lost_during_post_hold"
                )
                self._benchmark.tick()
                return
        else:
            dual_grasp_confirmed = False
        replay_aux = self._read_replay_aux_action()
        if replay_aux is not None:
            self._left_target_velocity, self._right_target_velocity, replay_lift = replay_aux
            throttle = 0.5 * (
                self._left_target_velocity + self._right_target_velocity
            ) / max(self.track_drive.max_track_speed, 1.0e-6)
        else:
            keyboard_throttle = float(
                (carb.input.KeyboardInput.W in self.held)
                - (carb.input.KeyboardInput.S in self.held)
            )
            quest_throttle = self._fresh_quest_forward_axis()
            throttle = float(np.clip(keyboard_throttle + quest_throttle, -1.0, 1.0))
            if benchmark_active:
                if self._phase == "acquiring_grasp":
                    if throttle > 0.0 and not dual_grasp_confirmed:
                        throttle = 0.0
                    elif throttle > 0.0:
                        self._phase = "pushing"
                        self._push_started_at = time.monotonic()
                elif self._phase != "pushing":
                    throttle = 0.0
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
        base_x_before = self.x
        base_y_before = self.y
        self.x, self.y, self.yaw = self.track_drive.step(
            self.x,
            self.y,
            self.yaw,
            self._left_target_velocity,
            self._right_target_velocity,
            dt,
        )
        forward_command_active = throttle > 0.0
        if forward_command_active and not self._cart_forward_sync_active:
            self._cart_forward_sync_active = True
            print(
                "[hand-cart-push] Forward synchronization active: "
                "cart follows the base with identical XY displacement."
            )
        forward_velocity = 0.5 * (
            self.track_drive.left_velocity + self.track_drive.right_velocity
        )
        continue_forward_sync = self._cart_forward_sync_active and (
            forward_command_active or forward_velocity > 1.0e-5
        )
        if continue_forward_sync:
            self._cart_sync_position[0] += self.x - base_x_before
            self._cart_sync_position[1] += self.y - base_y_before
        elif self._cart_forward_sync_active:
            self._cart_forward_sync_active = False
            print("[hand-cart-push] Forward synchronization stopped.")
        self._write_track_state(
            self._left_target_velocity, self._right_target_velocity
        )
        if replay_aux is not None:
            self._lift_target_velocity = replay_lift
        else:
            lift_direction = float(
                (carb.input.KeyboardInput.T in self.held)
                - (carb.input.KeyboardInput.Y in self.held)
            )
            self._lift_target_velocity = lift_direction * self.lift_speed
        self.lift_height = float(
            np.clip(
                self.lift_height + self._lift_target_velocity * dt,
                self.lift_lower,
                self.lift_upper,
            )
        )
        self.apply()
        self._write_synchronized_cart_pose()
        self._telemetry_sequence += 1
        if not benchmark_active:
            dual_grasp_confirmed = self._update_dual_grasp_status()
        self._update_success_checker(dt, benchmark_active)
        self._benchmark.tick()

    def reset(self) -> None:
        self._clear_quest_base_command()
        self.x, self.y, self.yaw, self.lift_height = self.initial
        self.track_drive.reset()
        self._left_target_velocity = 0.0
        self._right_target_velocity = 0.0
        self._lift_target_velocity = 0.0
        self._write_track_state(0.0, 0.0)
        self.apply()
        print("[hand-cart-push] Base pose and lift reset.")

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
    def _rotate_vector(cls, quaternion: np.ndarray, vector: np.ndarray) -> np.ndarray:
        conjugate = quaternion.copy()
        conjugate[1:] *= -1.0
        return cls._quaternion_multiply(
            cls._quaternion_multiply(quaternion, np.concatenate(([0.0], vector))),
            conjugate,
        )[1:]

    def reset_task_objects(self) -> None:
        self._clear_success()
        request = self._benchmark.consume_request()
        if request is not None:
            self._object_rng = np.random.default_rng(int(request["seed"]))
        env_ids = torch.arange(self.env.num_envs, device=self.device)
        xy_offset = self._object_rng.uniform(
            -self.cart_random_xy_limit,
            self.cart_random_xy_limit,
            size=2,
        )
        yaw_offset = float(
            self._object_rng.uniform(
                -self.cart_random_yaw_limit, self.cart_random_yaw_limit
            )
        )
        cart_state = self.hand_cart.data.default_root_state[env_ids].clone()
        cart_state[:, :3] += self.env.scene.env_origins[env_ids]
        cart_state[:, 0] += float(xy_offset[0])
        cart_state[:, 1] += float(xy_offset[1])
        yaw_delta = np.asarray(
            [math.cos(0.5 * yaw_offset), 0.0, 0.0, math.sin(0.5 * yaw_offset)],
            dtype=np.float64,
        )
        orientation = self._quaternion_multiply(
            yaw_delta, np.asarray(CART_ORIENTATION, dtype=np.float64)
        )
        orientation /= np.linalg.norm(orientation)
        cart_state[:, 3:7] = torch.as_tensor(
            orientation[np.newaxis, :], dtype=torch.float32, device=self.device
        )
        self.hand_cart.write_root_pose_to_sim(cart_state[:, :7], env_ids=env_ids)
        self.hand_cart.write_root_velocity_to_sim(
            torch.zeros((len(env_ids), 6), dtype=torch.float32, device=self.device),
            env_ids=env_ids,
        )
        self._cart_reset_position = cart_state[0, :3].detach().cpu().numpy().astype(np.float64)
        self._cart_reset_orientation = orientation.copy()
        self._cart_sync_position = self._cart_reset_position.copy()
        self._cart_sync_orientation = self._cart_reset_orientation.copy()
        self._cart_forward_sync_active = False
        print(
            "[hand-cart-push] Cart reset: "
            f"dx={100.0 * xy_offset[0]:+.1f} cm, "
            f"dy={100.0 * xy_offset[1]:+.1f} cm, "
            f"dyaw={math.degrees(yaw_offset):+.1f} deg."
        )
        self._benchmark.begin_settle(
            request,
            {
                "cart_offset_xy_m": xy_offset.astype(float).tolist(),
                "cart_yaw_offset_rad": float(yaw_offset),
                "cart_initial_pose_wxyz": np.concatenate(
                    (self._cart_reset_position, self._cart_reset_orientation)
                ).astype(float).tolist(),
                "sample_attempts": 1,
            },
            steps=10,
        )

    def _write_synchronized_cart_pose(self) -> None:
        """Write the held/synchronized cart pose and clear residual velocity."""

        env_ids = torch.arange(self.env.num_envs, device=self.device)
        pose = np.concatenate(
            (self._cart_sync_position, self._cart_sync_orientation)
        )[np.newaxis, :]
        self.hand_cart.write_root_pose_to_sim(
            torch.as_tensor(pose, dtype=torch.float32, device=self.device),
            env_ids=env_ids,
        )
        self.hand_cart.write_root_velocity_to_sim(
            torch.zeros((len(env_ids), 6), dtype=torch.float32, device=self.device),
            env_ids=env_ids,
        )

    def reset_everything(self) -> None:
        env_ids = torch.arange(self.env.num_envs, device=self.device)
        base_mdp.reset_scene_to_default(self.env, env_ids)
        self.reset()
        self.reset_task_objects()
        print(
            "[hand-cart-push] R reset complete: robot, base/lift, and cart restored."
        )

    @staticmethod
    def _empty_grasp_status() -> dict:
        return {
            "valid": False,
            "cart_between_fingers": False,
            "fingers_closed": False,
            "joint_positions_m": [],
            "dex1_motor_positions_rad": [],
            "min_joint_position_m": float("-inf"),
            "max_joint_position_m": float("-inf"),
            "distal_gap_m": float("inf"),
            "probe_hits": [False, False, False, False],
            "finger_points_cart_local_m": [],
            "finger_orientations_cart_local_wxyz": [],
            "proximal_midpoint_cart_local_m": [],
            "distal_midpoint_cart_local_m": [],
            "centerline_match": False,
            "centerline_rms_error_m": float("inf"),
            "centerline_max_error_m": float("inf"),
            "handle_axis_match": False,
            "handle_crossline_rms_error_m": float("inf"),
            "handle_crossline_max_error_m": float("inf"),
            "handle_max_lateral_x_error_m": float("inf"),
            "handle_longitudinal_offset_m": float("inf"),
            "geometry_match": False,
            "geometry_rms_error_m": float("inf"),
            "geometry_max_point_error_m": float("inf"),
        }

    @staticmethod
    def _load_dual_grasp_template() -> dict:
        try:
            template = json.loads(DUAL_GRASP_TEMPLATE_PATH.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise RuntimeError(
                f"Failed to load dual-grasp template {DUAL_GRASP_TEMPLATE_PATH}: {error}"
            ) from error
        if template.get("schema") != "hand_cart_dual_grasp_template_v1":
            raise RuntimeError("Unsupported hand-cart dual-grasp template schema")
        if template.get("task_id") != TASK_ID:
            raise RuntimeError("Hand-cart dual-grasp template task_id mismatch")
        for side in ("left", "right"):
            points = np.asarray(
                template.get(side, {}).get("finger_points_cart_local_m"),
                dtype=np.float64,
            )
            if points.shape != (4, 3) or not np.isfinite(points).all():
                raise RuntimeError(
                    f"Invalid {side} finger points in dual-grasp template"
                )
        matching = template.get("matching", {})
        if float(matching.get("rms_point_error_m", 0.0)) <= 0.0:
            raise RuntimeError("Dual-grasp RMS tolerance must be positive")
        if float(matching.get("max_point_error_m", 0.0)) <= 0.0:
            raise RuntimeError("Dual-grasp max-point tolerance must be positive")
        if float(matching.get("centerline_rms_error_m", 0.0)) <= 0.0:
            raise RuntimeError("Dual-grasp centerline RMS tolerance must be positive")
        if float(matching.get("centerline_max_error_m", 0.0)) <= 0.0:
            raise RuntimeError("Dual-grasp centerline max tolerance must be positive")
        if float(matching.get("handle_crossline_rms_error_m", 0.0)) <= 0.0:
            raise RuntimeError("Handle crossline RMS tolerance must be positive")
        if float(matching.get("handle_crossline_max_error_m", 0.0)) <= 0.0:
            raise RuntimeError("Handle crossline max tolerance must be positive")
        if float(matching.get("handle_max_lateral_x_error_m", 0.0)) <= 0.0:
            raise RuntimeError("Handle lateral-X tolerance must be positive")
        longitudinal_range = matching.get("handle_longitudinal_offset_range_m", [])
        if (
            not isinstance(longitudinal_range, list)
            or len(longitudinal_range) != 2
            or not float(longitudinal_range[0]) < float(longitudinal_range[1])
        ):
            raise RuntimeError("Invalid handle longitudinal offset range")
        return template

    @staticmethod
    def _query_hit_path(hit, field: str) -> str:
        value = hit.get(field) if isinstance(hit, dict) else getattr(hit, field, None)
        return "" if value is None else str(value)

    def _cart_overlaps_probe(self, point: np.ndarray) -> bool:
        """Return true only when cart collision occupies a finger-center probe."""

        found_cart = False

        def report_hit(hit):
            nonlocal found_cart
            rigid_body = self._query_hit_path(hit, "rigid_body")
            collision = self._query_hit_path(hit, "collision")
            if "/HandCart" in rigid_body or "/HandCart" in collision:
                found_cart = True
                return False
            return True

        self._scene_query.overlap_sphere(
            GRIPPER_CENTER_PROBE_RADIUS_M,
            carb.Float3(float(point[0]), float(point[1]), float(point[2])),
            report_hit,
            False,
        )
        return found_cart

    def _gripper_capture_status(self, side: str) -> dict:
        # Stored order: Link1_2, Link2_2, Link1_3, Link2_3.
        indices = self._gripper_body_indices[side]
        body_positions = (
            self.robot.data.body_pos_w[0, list(indices)]
            .detach()
            .cpu()
            .numpy()
            .astype(np.float64)
        )
        body_orientations = (
            self.robot.data.body_quat_w[0, list(indices)]
            .detach()
            .cpu()
            .numpy()
            .astype(np.float64)
        )
        proximal_midpoint = 0.5 * (body_positions[0] + body_positions[1])
        distal_midpoint = 0.5 * (body_positions[2] + body_positions[3])
        finger_axis = distal_midpoint - proximal_midpoint
        # The final point reaches the center/tip portion of the distal pad;
        # the link origin itself lies near the start of that pad.
        probe_points = (
            proximal_midpoint,
            0.5 * (proximal_midpoint + distal_midpoint),
            distal_midpoint,
            distal_midpoint + 0.9 * finger_axis,
        )
        probe_hits = [self._cart_overlaps_probe(point) for point in probe_points]
        cart_between_fingers = any(probe_hits)

        joint_indices = self._gripper_joint_indices[side]
        joint_positions = (
            self.robot.data.joint_pos[0, list(joint_indices)]
            .detach()
            .cpu()
            .numpy()
            .astype(np.float64)
        )
        min_joint_position = float(np.min(joint_positions))
        max_joint_position = float(np.max(joint_positions))
        # Isaac joint: -0.020 m=open, +0.024 m=closed.  The official Dex1
        # controller reports the inverse motor convention: 5.4 rad=open,
        # 0 rad=closed.  Keep both forms in diagnostics to avoid ambiguity.
        dex1_motor_positions = np.clip(
            5.4 * (0.024 - joint_positions) / 0.044, 0.0, 5.4
        )
        distal_gap = float(np.linalg.norm(body_positions[2] - body_positions[3]))
        cart_orientation_inverse = self._cart_sync_orientation.copy()
        cart_orientation_inverse[1:] *= -1.0
        finger_points_cart_local = np.asarray(
            [
                self._rotate_vector(
                    cart_orientation_inverse,
                    point - self._cart_sync_position,
                )
                for point in body_positions
            ],
            dtype=np.float64,
        )
        finger_orientations_cart_local = np.asarray(
            [
                self._quaternion_multiply(
                    cart_orientation_inverse, orientation
                )
                for orientation in body_orientations
            ],
            dtype=np.float64,
        )
        finger_orientations_cart_local /= np.maximum(
            np.linalg.norm(finger_orientations_cart_local, axis=1, keepdims=True),
            1.0e-12,
        )
        proximal_midpoint_cart_local = 0.5 * (
            finger_points_cart_local[0] + finger_points_cart_local[1]
        )
        distal_midpoint_cart_local = 0.5 * (
            finger_points_cart_local[2] + finger_points_cart_local[3]
        )
        template_points = np.asarray(
            self._dual_grasp_template[side]["finger_points_cart_local_m"],
            dtype=np.float64,
        )
        point_errors = np.linalg.norm(
            finger_points_cart_local - template_points, axis=1
        )
        geometry_rms_error = float(np.sqrt(np.mean(np.square(point_errors))))
        geometry_max_point_error = float(np.max(point_errors))
        matching = self._dual_grasp_template["matching"]
        geometry_match = (
            geometry_rms_error <= float(matching["rms_point_error_m"])
            and geometry_max_point_error <= float(matching["max_point_error_m"])
        )
        template_midpoints = np.asarray(
            [
                0.5 * (template_points[0] + template_points[1]),
                0.5 * (template_points[2] + template_points[3]),
            ],
            dtype=np.float64,
        )
        current_midpoints = np.asarray(
            [proximal_midpoint_cart_local, distal_midpoint_cart_local],
            dtype=np.float64,
        )
        centerline_errors = np.linalg.norm(
            current_midpoints - template_midpoints, axis=1
        )
        centerline_rms_error = float(
            np.sqrt(np.mean(np.square(centerline_errors)))
        )
        centerline_max_error = float(np.max(centerline_errors))
        centerline_match = (
            centerline_rms_error
            <= float(matching["centerline_rms_error_m"])
            and centerline_max_error
            <= float(matching["centerline_max_error_m"])
        )
        # The calibrated samples were taken near one end of each long handle.
        # A valid grasp may slide along the handle's local Z direction while
        # remaining tightly centered across it.  Remove the common local-Z
        # displacement before measuring cross-handle error, but constrain that
        # displacement to the actual usable handle length.  This accepts
        # different grasp points without turning the check into a large sphere.
        centerline_delta = current_midpoints - template_midpoints
        handle_longitudinal_offset = float(np.mean(centerline_delta[:, 2]))
        handle_crossline_delta = centerline_delta.copy()
        handle_crossline_delta[:, 2] -= handle_longitudinal_offset
        handle_crossline_errors = np.linalg.norm(
            handle_crossline_delta, axis=1
        )
        handle_crossline_rms_error = float(
            np.sqrt(np.mean(np.square(handle_crossline_errors)))
        )
        handle_crossline_max_error = float(
            np.max(handle_crossline_errors)
        )
        handle_max_lateral_x_error = float(
            np.max(np.abs(handle_crossline_delta[:, 0]))
        )
        longitudinal_min, longitudinal_max = (
            float(value)
            for value in matching["handle_longitudinal_offset_range_m"]
        )
        handle_axis_match = (
            handle_crossline_rms_error
            <= float(matching["handle_crossline_rms_error_m"])
            and handle_crossline_max_error
            <= float(matching["handle_crossline_max_error_m"])
            and handle_max_lateral_x_error
            <= float(matching["handle_max_lateral_x_error_m"])
            and longitudinal_min
            <= handle_longitudinal_offset
            <= longitudinal_max
        )
        # Contact with an analytic handle can stop the two symmetric Dex1
        # joints at slightly different positions.  The live accepted pushes
        # included a valid handle enclosure with one joint at 3--6 mm, so a
        # per-joint lower bound produced false negatives.  The actual distal
        # separation robustly distinguishes an enclosed handle (3--36 mm)
        # from an open gripper (about 90 mm); keep the individual positions in
        # diagnostics but do not use them as a hard decision boundary.
        fingers_closed = distal_gap <= GRIPPER_MAX_DISTAL_GAP_M
        return {
            # Opposing finger positions move inward as Dex1 closes, even when
            # the hand stays centered on the same handle.  Match the two
            # gripper centerline stations to the user-confirmed handle pose,
            # then use closure only as a broad guard against an open hand that
            # happens to hover around the handle.  The unreliable fused-SDF
            # overlap probe and exact four-point spacing remain diagnostics.
            "valid": bool(handle_axis_match and fingers_closed),
            "cart_between_fingers": bool(cart_between_fingers),
            "fingers_closed": bool(fingers_closed),
            "joint_positions_m": [float(value) for value in joint_positions],
            "dex1_motor_positions_rad": [
                float(value) for value in dex1_motor_positions
            ],
            "min_joint_position_m": min_joint_position,
            "max_joint_position_m": max_joint_position,
            "distal_gap_m": distal_gap,
            "probe_hits": [bool(value) for value in probe_hits],
            "finger_points_cart_local_m": finger_points_cart_local.tolist(),
            "finger_orientations_cart_local_wxyz": (
                finger_orientations_cart_local.tolist()
            ),
            "proximal_midpoint_cart_local_m": (
                proximal_midpoint_cart_local.tolist()
            ),
            "distal_midpoint_cart_local_m": distal_midpoint_cart_local.tolist(),
            "centerline_match": bool(centerline_match),
            "centerline_rms_error_m": centerline_rms_error,
            "centerline_max_error_m": centerline_max_error,
            "handle_axis_match": bool(handle_axis_match),
            "handle_crossline_rms_error_m": handle_crossline_rms_error,
            "handle_crossline_max_error_m": handle_crossline_max_error,
            "handle_max_lateral_x_error_m": handle_max_lateral_x_error,
            "handle_longitudinal_offset_m": handle_longitudinal_offset,
            "geometry_match": bool(geometry_match),
            "geometry_rms_error_m": geometry_rms_error,
            "geometry_max_point_error_m": geometry_max_point_error,
        }

    def _update_dual_grasp_status(self) -> bool:
        self._left_grasp_status = self._gripper_capture_status("left")
        self._right_grasp_status = self._gripper_capture_status("right")
        self._dual_grasp_valid_now = bool(
            self._left_grasp_status["valid"] and self._right_grasp_status["valid"]
        )
        if self._dual_grasp_valid_now:
            self._dual_grasp_frames = min(
                self._dual_grasp_frames + 1, DUAL_GRASP_CONFIRM_FRAMES
            )
        else:
            # Keep the acquisition evidence tolerant of one contact-solver
            # fluctuation, but use _dual_grasp_valid_now for strict push checks.
            self._dual_grasp_frames = max(self._dual_grasp_frames - 1, 0)
        return self._dual_grasp_frames >= DUAL_GRASP_CONFIRM_FRAMES

    def _success_geometry(self) -> tuple[bool, float, float, float]:
        pose = self.hand_cart.data.root_pose_w[0].detach().cpu().numpy().astype(np.float64)
        displacement = pose[:3] - self._cart_reset_position
        forward = float(
            displacement[0] * TASK_FORWARD_X + displacement[1] * TASK_FORWARD_Y
        )
        lateral = float(
            displacement[0] * -TASK_FORWARD_Y + displacement[1] * TASK_FORWARD_X
        )
        orientation = pose[3:7]
        orientation /= max(np.linalg.norm(orientation), 1.0e-12)
        # The source asset's authored up axis is local +Y.
        cart_up = self._rotate_vector(
            orientation, np.asarray([0.0, 1.0, 0.0], dtype=np.float64)
        )
        upright_dot = float(cart_up[2])
        success = (
            forward >= MIN_FORWARD_PROGRESS_M
            and abs(lateral) <= MAX_LATERAL_DRIFT_M
            and upright_dot >= MIN_UPRIGHT_DOT
        )
        return success, forward, lateral, upright_dot

    def _stop_base_for_terminal_phase(self) -> None:
        self._clear_quest_base_command()
        self._left_target_velocity = 0.0
        self._right_target_velocity = 0.0
        self._lift_target_velocity = 0.0
        self._cart_forward_sync_active = False
        # Do not allow the kinematic acceleration ramp to carry the robot past
        # the 0.50 m target or during a terminal failure/post-push hold.
        self.track_drive.left_velocity = 0.0
        self.track_drive.right_velocity = 0.0
        self._write_track_state(0.0, 0.0)

    def _write_terminal_marker(self, path: Path, payload: dict) -> None:
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)

    def _capture_gripper_hold_targets(self) -> None:
        positions = self.robot.data.joint_pos[0].detach().cpu()
        indices = sorted(
            set(self._gripper_joint_indices["left"])
            | set(self._gripper_joint_indices["right"])
        )
        self._gripper_hold_targets = {
            int(index): float(positions[index].item()) for index in indices
        }

    def override_action(self, action):
        """Hold the current gripper joint targets during post-push confirmation."""

        if self._gripper_hold_targets is None or not torch.is_tensor(action):
            return action
        result = action.clone()
        indices = sorted(self._gripper_hold_targets)
        values = torch.as_tensor(
            [self._gripper_hold_targets[index] for index in indices],
            dtype=result.dtype,
            device=result.device,
        )
        index_tensor = torch.as_tensor(indices, dtype=torch.long, device=result.device)
        if result.ndim == 1:
            result.index_copy_(0, index_tensor, values)
        else:
            shape = (*result.shape[:-1], len(indices))
            result.index_copy_(-1, index_tensor, values.expand(shape))
        return result

    def _mark_failure(self, reason: str) -> None:
        if self._failure_latched or self.success_latched:
            return
        failed_phase = self._phase
        self._failure_latched = True
        self._failure_reason = str(reason)
        self._phase = "failure"
        self._stop_base_for_terminal_phase()
        push_elapsed = None
        if self._push_started_at is not None:
            push_elapsed = max(0.0, time.monotonic() - self._push_started_at)
        payload = {
            "success": False,
            "failure_reason": self._failure_reason,
            "failed_phase": failed_phase,
            "forward_progress_m": float(self.success_forward_progress),
            "lateral_drift_m": float(self.success_lateral_drift),
            "push_elapsed_seconds": push_elapsed,
            "dual_grasp_valid_now": bool(self._dual_grasp_valid_now),
            "dual_grasp_confirm_frames": int(self._dual_grasp_frames),
            "left_grasp": self._left_grasp_status,
            "right_grasp": self._right_grasp_status,
            "timestamp": datetime.now().astimezone().isoformat(),
        }
        self._write_terminal_marker(FAILURE_MARKER_PATH, payload)
        print(
            "[hand-cart-push] FAILURE: "
            f"{self._failure_reason}, progress={self.success_forward_progress:.3f} m."
        )

    def _mark_success(self) -> None:
        if self.success_latched or self._failure_latched:
            return
        self.success_latched = True
        self._phase = "success"
        self._stop_base_for_terminal_phase()
        payload = {
            "success": True,
            "forward_progress_m": self.success_forward_progress,
            "lateral_drift_m": self.success_lateral_drift,
            "upright_dot": self._last_upright_dot,
            "dual_grasp_confirmed": True,
            "dual_grasp_valid_through_push": True,
            "dual_grasp_confirm_frames": self._dual_grasp_frames,
            "post_push_hold_seconds": self._post_push_hold_elapsed,
            "left_grasp": self._left_grasp_status,
            "right_grasp": self._right_grasp_status,
            "timestamp": datetime.now().astimezone().isoformat(),
        }
        self._write_terminal_marker(SUCCESS_MARKER_PATH, payload)
        print(
            "[hand-cart-push] SUCCESS: cart pushed forward with both grippers "
            f"held for {self._post_push_hold_elapsed:.2f} s."
        )

    def _update_success_checker(self, dt: float, benchmark_active: bool) -> None:
        motion_success, forward, lateral, upright_dot = self._success_geometry()
        self.success_forward_progress = forward
        self.success_lateral_drift = lateral
        self._last_upright_dot = upright_dot
        dual_grasp_confirmed = (
            self._dual_grasp_frames >= DUAL_GRASP_CONFIRM_FRAMES
        )
        self._maybe_print_grasp_diagnostics(
            motion_success, forward, lateral, upright_dot, dual_grasp_confirmed
        )
        if not benchmark_active:
            if motion_success and dual_grasp_confirmed:
                self._mark_success()
            return
        if self._phase == "acquiring_grasp":
            return
        if self._phase == "pushing":
            if not self._dual_grasp_valid_now:
                self._mark_failure("grasp_lost_during_push")
                return
            if motion_success:
                self._phase = "post_push_hold"
                self._post_push_hold_elapsed = 0.0
                self._capture_gripper_hold_targets()
                self._stop_base_for_terminal_phase()
            return
        if self._phase != "post_push_hold":
            return
        self._stop_base_for_terminal_phase()
        if not self._dual_grasp_valid_now:
            self._mark_failure("grasp_lost_during_post_hold")
            return
        self._post_push_hold_elapsed += float(dt)
        if self._post_push_hold_elapsed >= POST_PUSH_HOLD_SECONDS:
            self._mark_success()

    def _maybe_print_grasp_diagnostics(
        self,
        motion_success: bool,
        forward: float,
        lateral: float,
        upright_dot: float,
        dual_grasp_confirmed: bool,
    ) -> None:
        if GRASP_DEBUG_INTERVAL_S == 0.0:
            return
        now = time.monotonic()
        if now - self._last_grasp_debug_time < GRASP_DEBUG_INTERVAL_S:
            return
        self._last_grasp_debug_time = now

        def summarize(side: str, status: dict) -> str:
            joints = ",".join(
                f"{value:+.4f}" for value in status["joint_positions_m"]
            )
            motors = ",".join(
                f"{value:.3f}" for value in status["dex1_motor_positions_rad"]
            )
            hits = "".join("1" if value else "0" for value in status["probe_hits"])
            return (
                f"{side}:valid={int(status['valid'])},"
                f"axis={int(status['handle_axis_match'])},"
                f"cross_rms={status['handle_crossline_rms_error_m']:.4f},"
                f"cross_max={status['handle_crossline_max_error_m']:.4f},"
                f"lateral_x={status['handle_max_lateral_x_error_m']:.4f},"
                f"along={status['handle_longitudinal_offset_m']:+.4f},"
                f"center={int(status['centerline_match'])},"
                f"center_rms={status['centerline_rms_error_m']:.4f},"
                f"center_max={status['centerline_max_error_m']:.4f},"
                f"geom={int(status['geometry_match'])},"
                f"geom_rms={status['geometry_rms_error_m']:.4f},"
                f"geom_max={status['geometry_max_point_error_m']:.4f},"
                f"closed={int(status['fingers_closed'])},"
                f"between={int(status['cart_between_fingers'])},"
                f"joint_m=[{joints}],motor_rad=[{motors}],"
                f"gap={status['distal_gap_m']:.4f},probe={hits}"
            )

        diagnostic_line = (
            "[grasp-debug] "
            f"motion={int(motion_success)},progress={forward:.3f},"
            f"lateral={lateral:+.3f},upright={upright_dot:.3f}; "
            f"{summarize('L', self._left_grasp_status)}; "
            f"{summarize('R', self._right_grasp_status)}; "
            f"dual_frames={self._dual_grasp_frames}/{DUAL_GRASP_CONFIRM_FRAMES},"
            f"confirmed={int(dual_grasp_confirmed)}"
        )
        print(diagnostic_line)
        try:
            GRASP_DIAGNOSTICS_PATH.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "timestamp": datetime.now().astimezone().isoformat(),
                "motion_success": bool(motion_success),
                "forward_progress_m": float(forward),
                "lateral_drift_m": float(lateral),
                "upright_dot": float(upright_dot),
                "left": self._left_grasp_status,
                "right": self._right_grasp_status,
                "dual_confirm_frames": int(self._dual_grasp_frames),
                "dual_confirmed": bool(dual_grasp_confirmed),
            }
            with GRASP_DIAGNOSTICS_PATH.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(payload, ensure_ascii=False, allow_nan=False))
                stream.write("\n")
        except OSError as error:
            print(f"[grasp-debug] Failed to append diagnostics: {error}")

    def get_recording_state(self) -> dict:
        """Return the opt-in mobile-base sample appended to rt/sim_state."""

        dx = self.x - self.initial[0]
        dy = self.y - self.initial[1]
        forward = dx * TASK_FORWARD_X + dy * TASK_FORWARD_Y
        lateral = dx * -TASK_FORWARD_Y + dy * TASK_FORWARD_X
        yaw_delta = math.atan2(
            math.sin(self.yaw - self.initial[2]),
            math.cos(self.yaw - self.initial[2]),
        )
        cart_pose = [
            float(self._cart_sync_position[0]),
            float(self._cart_sync_position[1]),
            float(self._cart_sync_position[2]),
            float(self._cart_sync_orientation[0]),
            float(self._cart_sync_orientation[1]),
            float(self._cart_sync_orientation[2]),
            float(self._cart_sync_orientation[3]),
        ]
        return {
            "schema": "tracked_base_v1",
            "task_id": TASK_ID,
            "sequence": int(self._telemetry_sequence),
            "timestamp_ns": time.time_ns(),
            "replay_initial_state": {
                "objects": {"hand_cart": {"root_pose_wxyz": cart_pose}}
            },
            "state": {
                "base_forward_position_m": float(forward),
                "base_lateral_position_m": float(lateral),
                "base_yaw_delta_rad": float(yaw_delta),
                "lift_height_m": float(self.lift_height),
                "left_track_velocity_mps": float(self.track_drive.left_velocity),
                "right_track_velocity_mps": float(self.track_drive.right_velocity),
                "lift_velocity_mps": float(self._lift_target_velocity),
                "cart_world_x_m": float(self._cart_sync_position[0]),
                "cart_world_y_m": float(self._cart_sync_position[1]),
                "cart_world_z_m": float(self._cart_sync_position[2]),
                "cart_world_qw": float(self._cart_sync_orientation[0]),
                "cart_world_qx": float(self._cart_sync_orientation[1]),
                "cart_world_qy": float(self._cart_sync_orientation[2]),
                "cart_world_qz": float(self._cart_sync_orientation[3]),
            },
            "action": {
                "left_track_velocity_target_mps": float(self._left_target_velocity),
                "right_track_velocity_target_mps": float(self._right_target_velocity),
                "lift_velocity_target_mps": float(self._lift_target_velocity),
            },
            "grasp_status": {
                "left": self._left_grasp_status,
                "right": self._right_grasp_status,
                "dual_confirm_frames": int(self._dual_grasp_frames),
                "dual_confirmed": bool(
                    self._dual_grasp_frames >= DUAL_GRASP_CONFIRM_FRAMES
                ),
            },
        }

    @property
    def is_success_frozen(self) -> bool:
        return self.success_latched

    @property
    def is_terminal_frozen(self) -> bool:
        return self.success_latched or self._failure_latched

    def _clear_success(self) -> None:
        was_latched = self.success_latched
        self.success_latched = False
        self.success_forward_progress = 0.0
        self.success_lateral_drift = 0.0
        self._last_upright_dot = 1.0
        self._dual_grasp_frames = 0
        self._dual_grasp_valid_now = False
        self._phase = "acquiring_grasp"
        self._failure_latched = False
        self._failure_reason = None
        self._push_started_at = None
        self._post_push_hold_elapsed = 0.0
        self._gripper_hold_targets = None
        self._left_grasp_status = self._empty_grasp_status()
        self._right_grasp_status = self._empty_grasp_status()
        SUCCESS_MARKER_PATH.unlink(missing_ok=True)
        FAILURE_MARKER_PATH.unlink(missing_ok=True)
        for marker in (SUCCESS_MARKER_PATH, FAILURE_MARKER_PATH):
            marker.with_name(f".{marker.name}.{os.getpid()}.tmp").unlink(missing_ok=True)
        if was_latched:
            print("[hand-cart-push] Success freeze cleared by reset.")

    def close(self) -> None:
        self._clear_success()
        self._clear_quest_base_command()
        if self._quest_base_subscriber is not None:
            try:
                self._quest_base_subscriber.Close()
            except Exception:
                pass
            self._quest_base_subscriber = None
        if self._input is not None and self._subscription is not None:
            try:
                self._input.unsubscribe_to_keyboard_events(
                    self._keyboard, self._subscription
                )
            except Exception:
                pass
        self._subscription = None
        self.held.clear()
