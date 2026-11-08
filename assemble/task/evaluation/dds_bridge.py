#!/usr/bin/env python3
"""Task-aware local DDS bridge with joint safety, locks, and telemetry checks."""

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

from task_specs import ARM_NAMES, MOBILE_STATE_NAMES, TaskSpec, get_spec
from safety import apply_right_only_lock, project_mobile_action, sanitize_joint_action
from utils import write_json_atomic

_BENCH_ASSEMBLE = Path(__file__).resolve().parents[2]
XR_ROOT = Path(os.environ.get("XR_TELEOP_ROOT") or (_BENCH_ASSEMBLE / "third_party" / "xr_teleoperate"))
sys.path.insert(0, str(XR_ROOT))

from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber
from unitree_sdk2py.idl.default import unitree_go_msg_dds__MotorCmd_
from unitree_sdk2py.idl.std_msgs.msg.dds_ import String_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import MotorCmds_, MotorStates_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_
from teleop.robot_control.robot_arm import G1_29_ArmController

DEFAULT_URDF = XR_ROOT / "assets/g1/g1_body29_hand14.urdf"


def load_arm_limits(path: Path) -> tuple[np.ndarray, np.ndarray]:
    root = ET.parse(path).getroot()
    limits = {}
    for joint in root.findall("joint"):
        limit = joint.find("limit")
        if limit is not None and "lower" in limit.attrib and "upper" in limit.attrib:
            limits[joint.attrib["name"]] = (float(limit.attrib["lower"]), float(limit.attrib["upper"]))
    missing = [name for name in ARM_NAMES if name not in limits]
    if missing:
        raise RuntimeError(f"URDF is missing arm limits: {missing}")
    return (
        np.asarray([limits[name][0] for name in ARM_NAMES]),
        np.asarray([limits[name][1] for name in ARM_NAMES]),
    )


class DirectDex1Controller:
    def __init__(self) -> None:
        self._states, self._timestamps = np.zeros(2), np.zeros(2)
        self._pubs = [ChannelPublisher("rt/dex1/left/cmd", MotorCmds_), ChannelPublisher("rt/dex1/right/cmd", MotorCmds_)]
        for publisher in self._pubs:
            publisher.Init()
        self._subs = [ChannelSubscriber("rt/dex1/left/state", MotorStates_), ChannelSubscriber("rt/dex1/right/state", MotorStates_)]
        for index, subscriber in enumerate(self._subs):
            subscriber.Init(lambda message, i=index: self._state(i, message), 1)
        self._messages = [self._message(), self._message()]

    @staticmethod
    def _message():
        result = MotorCmds_(); result.cmds = [unitree_go_msg_dds__MotorCmd_()]
        result.cmds[0].dq = result.cmds[0].tau = 0.0
        result.cmds[0].kp, result.cmds[0].kd = 5.0, 0.05
        return result

    def _state(self, index, message) -> None:
        if message.states:
            self._states[index], self._timestamps[index] = float(message.states[0].q), time.monotonic()

    def command(self, target) -> None:
        for index, value in enumerate(np.asarray(target).reshape(2)):
            self._messages[index].cmds[0].q = float(value)
            self._pubs[index].Write(self._messages[index])

    def state(self):
        stamp = float(np.min(self._timestamps))
        return self._states.copy(), math.inf if stamp <= 0 else time.monotonic() - stamp


class SimTelemetry:
    def __init__(self) -> None:
        self.payload: dict | None = None
        self.received = 0.0
        self.subscriber = ChannelSubscriber("rt/sim_state", String_)
        self.subscriber.Init(self._on_message, 1)

    def _on_message(self, message) -> None:
        try:
            payload = json.loads(message.data)
            runtime = payload.get("runtime_state")
            if isinstance(runtime, dict):
                self.payload, self.received = runtime, time.monotonic()
        except Exception:
            pass

    def get(self, spec: TaskSpec) -> tuple[list[float], float, dict]:
        if spec.telemetry_schema is None:
            return [], 0.0, {}
        if not self.payload:
            raise RuntimeError("Task telemetry is unavailable")
        if self.payload.get("schema") != spec.telemetry_schema:
            raise RuntimeError(f"Telemetry schema mismatch: {self.payload.get('schema')!r}")
        age = time.monotonic() - self.received
        if spec.key == "task04":
            state = self.payload.get("state", {})
            values = [float(state[name]) for name in MOBILE_STATE_NAMES]
        elif spec.key == "task06":
            values = [float(value) for value in self.payload["object_root_pose_wxyz"]]
        else:
            values = []
        if not np.isfinite(values).all():
            raise RuntimeError("Task telemetry contains NaN or infinity")
        return values, age, self.payload


class EvaluationBridge:
    def __init__(self, args, spec: TaskSpec) -> None:
        ChannelFactoryInitialize(1, networkInterface=args.network_interface)
        self.args, self.spec = args, spec
        self.lower, self.upper = load_arm_limits(args.urdf)
        self._lowstate_time = 0.0
        self._health = ChannelSubscriber("rt/lowstate", LowState_)
        self._health.Init(lambda _message: setattr(self, "_lowstate_time", time.monotonic()), 1)
        lowstate_deadline = time.monotonic() + 45.0
        while self._lowstate_time <= 0.0 and time.monotonic() < lowstate_deadline:
            time.sleep(0.1)
        if self._lowstate_time <= 0.0:
            raise TimeoutError("rt/lowstate did not become available within 45 seconds")
        self.arm = G1_29_ArmController(simulation_mode=True)
        self.arm.set_arm_velocity_limit(args.arm_velocity_limit)
        self.gripper, self.telemetry = DirectDex1Controller(), SimTelemetry()
        self.reset_pub = ChannelPublisher("rt/reset_pose/cmd", String_); self.reset_pub.Init()
        self.mobile_pub = ChannelPublisher("rt/hand_cart_base/cmd", String_); self.mobile_pub.Init()
        grip, _ = self.gripper.state()
        self.last_safe = np.concatenate((self.arm.get_current_dual_arm_q().astype(float), grip))
        self.last_command = time.monotonic()
        self.locked_left = np.concatenate((np.asarray(spec.left_arm_reset_target), [spec.initial_grippers[0]]))
        self.projection_count = 0
        self.mobile_sequence = 0

    def _publish_mobile(self, forward_axis: float) -> None:
        """Publish one protocol-complete command with a monotonic sequence number."""
        self.mobile_sequence += 1
        payload = {
            "schema": "hand_cart_base_command_v1",
            "task_id": self.spec.sim_task,
            "forward_axis": float(forward_axis),
            "sequence": self.mobile_sequence,
            "timestamp_ns": time.time_ns(),
        }
        self.mobile_pub.Write(String_(data=json.dumps(payload)))

    def _mobile_stop(self) -> None:
        if self.spec.key == "task04":
            self._publish_mobile(0.0)

    def state(self):
        arm = self.arm.get_current_dual_arm_q().astype(float)
        grip, grip_age = self.gripper.state()
        arm_age = math.inf if self._lowstate_time <= 0 else time.monotonic() - self._lowstate_time
        extra, telemetry_age, telemetry = self.telemetry.get(self.spec)
        state = np.concatenate((arm, grip, np.asarray(extra, dtype=float)))
        if state.shape != (self.spec.state_dim,) or not np.isfinite(state).all():
            raise RuntimeError(f"Invalid task state {state.shape}")
        return state, max(arm_age, grip_age, telemetry_age), telemetry

    def hold(self) -> None:
        self.arm.ctrl_dual_arm(self.last_safe[:14], np.zeros(14))
        self.gripper.command(self.last_safe[14:])
        self._mobile_stop()

    def _apply_locks(self, action: np.ndarray) -> np.ndarray:
        if not self.spec.right_arm_only:
            return action
        return apply_right_only_lock(action, self.locked_left)

    def command(self, requested):
        requested = np.asarray(requested, dtype=float).reshape(-1)
        if requested.shape != (self.spec.action_dim,) or not np.isfinite(requested).all():
            raise ValueError(f"Expected finite {self.spec.action_dim}-vector")
        joint_request = self._apply_locks(requested[:16])
        now = time.monotonic()
        safe = sanitize_joint_action(joint_request, self.last_safe, self.lower, self.upper,
                                     self.args.arm_velocity_limit, now - self.last_command,
                                     self.args.gripper_max_delta)
        self.arm.ctrl_dual_arm(safe[:14], np.zeros(14)); self.gripper.command(safe[14:])
        executed = safe
        projected = False
        if self.spec.key == "task04":
            mobile, projected = project_mobile_action(requested[16:])
            if projected:
                self.projection_count += 1
            forward = 1.0 if mobile[0] > 0.0 else 0.0
            self._publish_mobile(forward)
            executed = np.concatenate((safe, mobile))
        self.last_safe, self.last_command = safe, now
        return executed, projected

    def _approach_initial(self) -> None:
        arm_target = np.zeros(14)
        arm_target[:7] = np.asarray(self.spec.left_arm_reset_target)
        grip_target = np.asarray(self.spec.initial_grippers)
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            self.arm.ctrl_dual_arm(arm_target, np.zeros(14))
            current = self.last_safe[14:]
            current = np.clip(grip_target, current - self.args.gripper_max_delta, current + self.args.gripper_max_delta)
            self.gripper.command(current); self.last_safe[14:] = current
            time.sleep(1 / 30)

    def reset(self, episode_index: int, seed: int) -> dict:
        self._mobile_stop(); self._approach_initial()
        reset_id = uuid.uuid4().hex
        request = {"reset_id": reset_id, "episode_index": int(episode_index), "seed": int(seed),
                   "requested_at": time.time(), "task": self.spec.key}
        self.args.reset_status_file.unlink(missing_ok=True)
        write_json_atomic(self.args.reset_request_file, request)
        self.reset_pub.Write(String_(data="2"))
        deadline, status = time.monotonic() + self.args.reset_timeout, None
        arm_target = np.zeros(14); arm_target[:7] = self.spec.left_arm_reset_target
        while time.monotonic() < deadline:
            self.arm.ctrl_dual_arm(arm_target, np.zeros(14)); self.gripper.command(self.spec.initial_grippers)
            if self.args.reset_status_file.is_file():
                try:
                    candidate = json.loads(self.args.reset_status_file.read_text())
                    if candidate.get("reset_id") == reset_id and candidate.get("settled"):
                        status = candidate; break
                except (OSError, json.JSONDecodeError):
                    pass
            time.sleep(0.05)
        if status is None:
            self.hold(); raise TimeoutError(f"Reset {reset_id} did not settle")
        actual_arm = self.arm.get_current_dual_arm_q().astype(float)
        actual_grip, _ = self.gripper.state()
        self.last_safe = np.concatenate((actual_arm, actual_grip))
        if self.spec.right_arm_only:
            self.locked_left = np.concatenate((actual_arm[:7], [actual_grip[0]]))
        self.last_command, self.projection_count = time.monotonic(), 0
        status["locked_left_arm_q"] = self.locked_left[:7].tolist() if self.spec.right_arm_only else None
        return status

    def shutdown(self) -> None:
        try:
            self._mobile_stop(); self.arm.ctrl_dual_arm_go_home()
            for _ in range(30):
                self.gripper.command(self.spec.initial_grippers); time.sleep(1 / 60)
        except Exception as error:
            print(f"[eval-bridge] shutdown warning: {error}", flush=True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True); parser.add_argument("--socket", type=Path, required=True)
    parser.add_argument("--authkey", required=True); parser.add_argument("--reset-request-file", type=Path, required=True)
    parser.add_argument("--reset-status-file", type=Path, required=True); parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    parser.add_argument("--arm-velocity-limit", type=float, default=3.0)
    parser.add_argument("--gripper-max-delta", type=float, default=0.18)
    parser.add_argument("--reset-timeout", type=float, default=25.0)
    parser.add_argument("--network-interface", default=None)
    return parser.parse_args()


def main() -> None:
    args, spec = parse_args(), None
    spec = get_spec(args.task)
    args.socket.unlink(missing_ok=True)
    stopping = False
    def stop(_signal, _frame):
        nonlocal stopping; stopping = True
    signal.signal(signal.SIGINT, stop); signal.signal(signal.SIGTERM, stop)
    bridge = EvaluationBridge(args, spec)
    listener = Listener(str(args.socket), family="AF_UNIX", authkey=args.authkey.encode())
    os.chmod(args.socket, 0o600)
    print(f"[eval-bridge] READY task={spec.key} socket={args.socket}", flush=True)
    try:
        while not stopping:
            connection = listener.accept()
            try:
                while not stopping:
                    try:
                        request = connection.recv(); command = request.get("command")
                        if command == "get_state":
                            state, age, telemetry = bridge.state()
                            response = {"ok": True, "state": state.tolist(), "dds_age": age, "telemetry": telemetry}
                        elif command == "action":
                            action, projected = bridge.command(request["action"])
                            state, age, telemetry = bridge.state()
                            response = {"ok": True, "action": action.tolist(), "dds_age": age,
                                        "telemetry": telemetry, "mobile_projected": projected,
                                        "mobile_projection_count": bridge.projection_count}
                        elif command == "hold":
                            bridge.hold(); response = {"ok": True}
                        elif command == "reset":
                            response = {"ok": True, "reset": bridge.reset(request["episode_index"], request["seed"])}
                        elif command == "shutdown":
                            connection.send({"ok": True}); stopping = True; break
                        else:
                            raise ValueError(f"Unknown command {command!r}")
                        connection.send(response)
                    except Exception as error:
                        # A reset timeout is an infrastructure error for the
                        # current attempt, not a reason to invalidate the
                        # persistent evaluator/bridge connection.  Keep the
                        # connection alive so the evaluator's one permitted
                        # retry can issue a fresh reset for the same seed.
                        connection.send({"ok": False, "error": f"{type(error).__name__}: {error}"})
            except EOFError:
                pass
            except (BrokenPipeError, ConnectionResetError, OSError):
                pass
            finally:
                connection.close()
    finally:
        bridge.shutdown(); listener.close(); args.socket.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
