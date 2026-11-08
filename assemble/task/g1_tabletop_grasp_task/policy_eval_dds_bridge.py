#!/usr/bin/env python3
"""Local-only DDS bridge for safe 16-DoF simulated G1 policy control."""

from __future__ import annotations

import argparse
import json
import math
import os
import signal
import sys
import time
import uuid
import xml.etree.ElementTree as ET
from multiprocessing.connection import Listener
from pathlib import Path

import numpy as np

XR_TELEOPERATE_ROOT = (
    Path(__file__).resolve().parents[2] / "third_party" / "xr_teleoperate"
)
if str(XR_TELEOPERATE_ROOT) not in sys.path:
    sys.path.insert(0, str(XR_TELEOPERATE_ROOT))

from unitree_sdk2py.core.channel import (
    ChannelFactoryInitialize,
    ChannelPublisher,
    ChannelSubscriber,
)
from unitree_sdk2py.idl.default import unitree_go_msg_dds__MotorCmd_
from unitree_sdk2py.idl.std_msgs.msg.dds_ import String_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import MotorCmds_, MotorStates_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_

from teleop.robot_control.robot_arm import G1_29_ArmController


ARM_JOINT_NAMES = [
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]
DEFAULT_URDF = Path(
    os.environ.get("XR_TELEOP_ROOT")
    or (Path(__file__).resolve().parents[2] / "third_party" / "xr_teleoperate")
) / "assets/g1/g1_body29_hand14.urdf"


def write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def load_arm_limits(urdf_path: Path) -> tuple[np.ndarray, np.ndarray]:
    root = ET.parse(urdf_path).getroot()
    limits = {}
    for joint in root.findall("joint"):
        limit = joint.find("limit")
        if limit is None or "lower" not in limit.attrib or "upper" not in limit.attrib:
            continue
        limits[joint.attrib["name"]] = (
            float(limit.attrib["lower"]),
            float(limit.attrib["upper"]),
        )
    missing = [name for name in ARM_JOINT_NAMES if name not in limits]
    if missing:
        raise RuntimeError(f"G1 URDF is missing arm joint limits: {missing}")
    lower = np.asarray([limits[name][0] for name in ARM_JOINT_NAMES], dtype=np.float64)
    upper = np.asarray([limits[name][1] for name in ARM_JOINT_NAMES], dtype=np.float64)
    return lower, upper


def sanitize_action(
    requested: np.ndarray,
    previous: np.ndarray,
    arm_lower: np.ndarray,
    arm_upper: np.ndarray,
    arm_velocity_limit: float,
    dt: float,
    gripper_max_delta: float,
) -> np.ndarray:
    """Validate and apply all policy-side G1/Dex1 safety constraints."""

    requested = np.asarray(requested, dtype=np.float64).reshape(-1)
    previous = np.asarray(previous, dtype=np.float64).reshape(-1)
    if requested.shape != (16,) or not np.isfinite(requested).all():
        raise ValueError("Policy action must be a finite 16-dimensional vector")
    if previous.shape != (16,) or not np.isfinite(previous).all():
        raise ValueError("Previous safe action must be a finite 16-dimensional vector")
    dt = float(np.clip(dt, 1.0 / 250.0, 0.1))
    arm_target = np.clip(requested[:14], arm_lower, arm_upper)
    arm_delta = float(arm_velocity_limit) * dt
    arm_target = np.clip(
        arm_target,
        previous[:14] - arm_delta,
        previous[:14] + arm_delta,
    )
    arm_target = np.clip(arm_target, arm_lower, arm_upper)
    gripper_target = np.clip(requested[14:], 0.0, 5.4)
    gripper_target = np.clip(
        gripper_target,
        previous[14:] - float(gripper_max_delta),
        previous[14:] + float(gripper_max_delta),
    )
    return np.concatenate((arm_target, gripper_target))


class DirectDex1Controller:
    """Publish absolute Dex1 motor targets and track feedback timestamps."""

    def __init__(self) -> None:
        self._states = np.zeros(2, dtype=np.float64)
        self._timestamps = np.zeros(2, dtype=np.float64)
        self._left_pub = ChannelPublisher("rt/dex1/left/cmd", MotorCmds_)
        self._right_pub = ChannelPublisher("rt/dex1/right/cmd", MotorCmds_)
        self._left_pub.Init()
        self._right_pub.Init()
        self._left_sub = ChannelSubscriber("rt/dex1/left/state", MotorStates_)
        self._right_sub = ChannelSubscriber("rt/dex1/right/state", MotorStates_)
        self._left_sub.Init(lambda msg: self._on_state(0, msg), 1)
        self._right_sub.Init(lambda msg: self._on_state(1, msg), 1)
        self._messages = [self._make_message(), self._make_message()]

    @staticmethod
    def _make_message() -> MotorCmds_:
        message = MotorCmds_()
        message.cmds = [unitree_go_msg_dds__MotorCmd_()]
        command = message.cmds[0]
        command.dq = 0.0
        command.tau = 0.0
        command.kp = 5.0
        command.kd = 0.05
        return message

    def _on_state(self, index: int, message: MotorStates_) -> None:
        if message.states:
            self._states[index] = float(message.states[0].q)
            self._timestamps[index] = time.monotonic()

    def command(self, target: np.ndarray) -> None:
        target = np.asarray(target, dtype=np.float64).reshape(2)
        self._messages[0].cmds[0].q = float(target[0])
        self._messages[1].cmds[0].q = float(target[1])
        self._left_pub.Write(self._messages[0])
        self._right_pub.Write(self._messages[1])

    def state(self) -> tuple[np.ndarray, float]:
        newest_common = float(np.min(self._timestamps))
        age = math.inf if newest_common <= 0.0 else time.monotonic() - newest_common
        return self._states.copy(), age


class EvaluationBridge:
    def __init__(self, args: argparse.Namespace) -> None:
        ChannelFactoryInitialize(1, networkInterface=args.network_interface)
        self.args = args
        self.arm_lower, self.arm_upper = load_arm_limits(args.urdf)
        self._last_lowstate_time = 0.0
        self._health_sub = ChannelSubscriber("rt/lowstate", LowState_)
        self._health_sub.Init(self._on_lowstate, 1)
        self.arm = G1_29_ArmController(simulation_mode=True)
        self.arm.set_arm_velocity_limit(args.arm_velocity_limit)
        self.gripper = DirectDex1Controller()
        self.reset_publisher = ChannelPublisher("rt/reset_pose/cmd", String_)
        self.reset_publisher.Init()
        current_arm = self.arm.get_current_dual_arm_q().astype(np.float64)
        current_gripper, _ = self.gripper.state()
        self.last_safe_target = np.concatenate((current_arm, current_gripper))
        self.last_command_time = time.monotonic()

    def _on_lowstate(self, _message: LowState_) -> None:
        self._last_lowstate_time = time.monotonic()

    def state(self) -> tuple[np.ndarray, float]:
        arm_state = self.arm.get_current_dual_arm_q().astype(np.float64)
        gripper_state, gripper_age = self.gripper.state()
        arm_age = (
            math.inf
            if self._last_lowstate_time <= 0.0
            else time.monotonic() - self._last_lowstate_time
        )
        state = np.concatenate((arm_state, gripper_state))
        if not np.isfinite(state).all():
            raise RuntimeError("DDS state contains NaN or infinity")
        return state, max(arm_age, gripper_age)

    def hold(self) -> None:
        self.arm.ctrl_dual_arm(self.last_safe_target[:14], np.zeros(14))
        self.gripper.command(self.last_safe_target[14:])

    def command(
        self, requested: np.ndarray, control_dt: float | None = None
    ) -> np.ndarray:
        now = time.monotonic()
        if control_dt is None:
            effective_dt = now - self.last_command_time
        else:
            effective_dt = float(control_dt)
            if not math.isfinite(effective_dt) or effective_dt <= 0.0:
                raise ValueError("control_dt must be a positive finite value")
        try:
            safe = sanitize_action(
                requested=requested,
                previous=self.last_safe_target,
                arm_lower=self.arm_lower,
                arm_upper=self.arm_upper,
                arm_velocity_limit=self.args.arm_velocity_limit,
                dt=effective_dt,
                gripper_max_delta=self.args.gripper_max_delta,
            )
        except ValueError:
            self.hold()
            raise
        self.arm.ctrl_dual_arm(safe[:14], np.zeros(14))
        self.gripper.command(safe[14:])
        self.last_safe_target = safe
        self.last_command_time = now
        return safe.copy()

    def _command_initial_gripper(self) -> None:
        target = np.asarray([self.args.initial_gripper] * 2, dtype=np.float64)
        for _ in range(36):
            next_target = np.clip(
                target,
                self.last_safe_target[14:] - self.args.gripper_max_delta,
                self.last_safe_target[14:] + self.args.gripper_max_delta,
            )
            self.gripper.command(next_target)
            self.last_safe_target[14:] = next_target
            if np.allclose(next_target, target, atol=1.0e-6):
                break
            time.sleep(1.0 / 30.0)

    def reset_episode(self, episode_index: int, seed: int) -> dict:
        self.arm.ctrl_dual_arm_go_home()
        self.last_safe_target[:14] = 0.0
        self._command_initial_gripper()
        reset_id = uuid.uuid4().hex
        request = {
            "reset_id": reset_id,
            "episode_index": int(episode_index),
            "seed": int(seed),
            "requested_at": time.time(),
        }
        self.args.reset_status_file.unlink(missing_ok=True)
        write_json_atomic(self.args.reset_request_file, request)
        self.reset_publisher.Write(String_(data="2"))

        deadline = time.monotonic() + self.args.reset_timeout
        status = None
        while time.monotonic() < deadline:
            self.arm.ctrl_dual_arm(np.zeros(14), np.zeros(14))
            self.gripper.command(np.asarray([self.args.initial_gripper] * 2))
            if self.args.reset_status_file.is_file():
                try:
                    candidate = json.loads(
                        self.args.reset_status_file.read_text(encoding="utf-8")
                    )
                    if candidate.get("reset_id") == reset_id and candidate.get("settled"):
                        status = candidate
                        break
                except (OSError, json.JSONDecodeError):
                    pass
            time.sleep(0.05)
        if status is None:
            self.hold()
            raise TimeoutError(
                f"Simulator did not acknowledge settled reset {reset_id} within "
                f"{self.args.reset_timeout:.1f}s"
            )
        self.last_safe_target = np.concatenate(
            (np.zeros(14), np.asarray([self.args.initial_gripper] * 2))
        )
        self.last_command_time = time.monotonic()
        return status

    def shutdown(self) -> None:
        try:
            self.arm.ctrl_dual_arm_go_home()
            self.last_safe_target[:14] = 0.0
            self._command_initial_gripper()
        except Exception as error:
            print(f"[eval-bridge] Safe shutdown warning: {error}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--socket", type=Path, required=True)
    parser.add_argument("--authkey", required=True)
    parser.add_argument("--reset-request-file", type=Path, required=True)
    parser.add_argument("--reset-status-file", type=Path, required=True)
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    parser.add_argument("--initial-gripper", type=float, default=4.0)
    parser.add_argument("--arm-velocity-limit", type=float, default=3.0)
    parser.add_argument("--gripper-max-delta", type=float, default=0.18)
    parser.add_argument("--reset-timeout", type=float, default=20.0)
    parser.add_argument("--network-interface", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.initial_gripper <= 5.4:
        raise ValueError("--initial-gripper must be in [0, 5.4]")
    args.socket.unlink(missing_ok=True)
    stopping = False

    def request_stop(_signum, _frame) -> None:
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    bridge = EvaluationBridge(args)
    listener = Listener(str(args.socket), family="AF_UNIX", authkey=args.authkey.encode())
    os.chmod(args.socket, 0o600)
    print(f"[eval-bridge] READY socket={args.socket}", flush=True)
    try:
        while not stopping:
            connection = listener.accept()
            try:
                while not stopping:
                    request = connection.recv()
                    command = request.get("command")
                    if command == "get_state":
                        state, age = bridge.state()
                        response = {"ok": True, "state": state.tolist(), "dds_age": age}
                    elif command == "action":
                        action = bridge.command(
                            request["action"], control_dt=request.get("control_dt")
                        )
                        state, age = bridge.state()
                        response = {
                            "ok": True,
                            "action": action.tolist(),
                            "state": state.tolist(),
                            "dds_age": age,
                        }
                    elif command == "hold":
                        bridge.hold()
                        response = {"ok": True}
                    elif command == "reset":
                        status = bridge.reset_episode(
                            request["episode_index"], request["seed"]
                        )
                        response = {"ok": True, "reset": status}
                    elif command == "shutdown":
                        response = {"ok": True}
                        connection.send(response)
                        stopping = True
                        break
                    else:
                        raise ValueError(f"Unknown bridge command: {command!r}")
                    connection.send(response)
            except EOFError:
                pass
            except Exception as error:
                try:
                    connection.send(
                        {"ok": False, "error": f"{type(error).__name__}: {error}"}
                    )
                except Exception:
                    pass
            finally:
                connection.close()
    finally:
        bridge.shutdown()
        listener.close()
        args.socket.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
