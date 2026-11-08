#!/usr/bin/env python3
"""Run automatic LeRobot policy evaluation against the tabletop Isaac Sim task."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import select
import sys
import termios
import threading
import time
import traceback
from collections import deque
from contextlib import nullcontext
from datetime import datetime
from multiprocessing.connection import Client
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import cv2
import numpy as np
import torch

_EVALUATION_DIR = Path(__file__).resolve().parent.parent / "evaluation"
if str(_EVALUATION_DIR) not in sys.path:
    sys.path.insert(0, str(_EVALUATION_DIR))

from benchmark_camera import CAMERA_KEYS, CameraReader
from lerobot.common.control_utils import predict_action
from lerobot.configs import PreTrainedConfig
from lerobot.policies import prepare_observation_for_inference
from lerobot.policies.factory import get_policy_class, make_pre_post_processors
from perturbation import (
    PERTURBATION_VARIANTS,
    TRAINING_AUGMENTATION_SEED,
    TRAINING_PARTICLE_COUNT,
    EpisodePerturber,
    choose_variant,
)


TASK_INSTRUCTION = "Insert the detonator lead wire into the detonator opening."
STATE_ACTION_NAMES = [
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
    "left_gripper_joint",
    "right_gripper_joint",
]
EXPECTED_INPUTS = {
    "observation.state": (16,),
    "observation.images.head": (3, 480, 640),
    "observation.images.left_wrist": (3, 480, 640),
    "observation.images.right_wrist": (3, 480, 640),
}
EXPECTED_OUTPUTS = {"action": (16,)}


class IllegalPolicyAction(RuntimeError):
    pass


def write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def feature_shapes(features: dict) -> dict[str, tuple[int, ...]]:
    return {key: tuple(int(value) for value in feature.shape) for key, feature in features.items()}


# Checkpoints trained with renamed camera keys: X-VLA used
# head -> image, left_wrist -> image2, right_wrist -> image3.
CHECKPOINT_CAMERA_ALIASES: dict[str, dict[str, str]] = {
    "xvla": {
        "observation.images.head": "observation.images.image",
        "observation.images.left_wrist": "observation.images.image2",
        "observation.images.right_wrist": "observation.images.image3",
    },
}


def camera_key_map(policy_type: str) -> dict[str, str] | None:
    return CHECKPOINT_CAMERA_ALIASES.get(policy_type)


def validate_checkpoint_features(config: PreTrainedConfig) -> None:
    inputs = feature_shapes(config.input_features)
    outputs = feature_shapes(config.output_features)
    expected_inputs = EXPECTED_INPUTS
    key_map = camera_key_map(config.type)
    if key_map is not None:
        expected_inputs = {key_map.get(key, key): value for key, value in expected_inputs.items()}
    if inputs != expected_inputs:
        raise ValueError(
            "Checkpoint input features do not match this benchmark. "
            f"Expected {expected_inputs}, got {inputs}."
        )
    if outputs != EXPECTED_OUTPUTS:
        raise ValueError(
            "Checkpoint output features do not match this benchmark. "
            f"Expected {EXPECTED_OUTPUTS}, got {outputs}."
        )


def load_remote_checkpoint_config(checkpoint: Path) -> SimpleNamespace:
    """Validate checkpoint metadata without importing its policy implementation locally."""

    payload = json.loads((checkpoint / "config.json").read_text(encoding="utf-8"))
    if payload.get("type") != "lingbot_va":
        raise ValueError(
            f"Remote evaluation currently supports lingbot_va, got {payload.get('type')!r}"
        )
    inputs = {
        key: tuple(int(value) for value in feature["shape"])
        for key, feature in payload.get("input_features", {}).items()
    }
    outputs = {
        key: tuple(int(value) for value in feature["shape"])
        for key, feature in payload.get("output_features", {}).items()
    }
    expected_lingbot_inputs = {
        key: shape for key, shape in EXPECTED_INPUTS.items() if "images" in key
    }
    if inputs != expected_lingbot_inputs:
        raise ValueError(
            "Remote LingBot checkpoint input features do not match this benchmark. "
            f"Expected {expected_lingbot_inputs}, got {inputs}."
        )
    if outputs != EXPECTED_OUTPUTS:
        raise ValueError(
            "Remote LingBot checkpoint output features do not match this benchmark. "
            f"Expected {EXPECTED_OUTPUTS}, got {outputs}."
        )
    return SimpleNamespace(type="lingbot_va")


class SyncPolicyRunner:
    def __init__(self, policy, preprocessor, postprocessor, device: torch.device, key_map: dict[str, str] | None = None) -> None:
        self.policy = policy
        self.preprocessor = preprocessor
        self.postprocessor = postprocessor
        self.device = device
        self.key_map = key_map
        self.compat = None
        # Relative chunks must be converted to absolute actions exactly once, using
        # the observation that produced the chunk.  Keeping a local absolute FIFO
        # prevents later control ticks from re-anchoring cached relative actions.
        self._absolute_action_queue: deque[torch.Tensor] = deque()

    def reset(self) -> None:
        self._absolute_action_queue.clear()
        self.policy.reset()
        self.preprocessor.reset()
        self.postprocessor.reset()

    def _rename_observation(self, observation: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        if not self.key_map:
            return observation
        return {self.key_map.get(key, key): value for key, value in observation.items()}

    def action(self, observation: dict[str, np.ndarray]) -> tuple[np.ndarray | None, float]:
        started = time.perf_counter()
        observation = self._rename_observation(observation)
        if bool(getattr(self.policy.config, "use_relative_actions", False)):
            if not self._absolute_action_queue:
                with (
                    torch.inference_mode(),
                    torch.autocast(device_type=self.device.type)
                    if self.device.type == "cuda" and bool(self.policy.config.use_amp)
                    else nullcontext(),
                ):
                    prepared = prepare_observation_for_inference(
                        dict(observation),
                        self.device,
                        TASK_INSTRUCTION,
                        "unitree_g1_29_dex1",
                    )
                    processed = self.preprocessor(prepared)
                    chunk = self.policy.predict_action_chunk(processed)
                    n_action_steps = int(self.policy.config.n_action_steps)
                    absolute_chunk = self.postprocessor(chunk[:, :n_action_steps])
                if absolute_chunk.ndim != 3 or absolute_chunk.shape[0] != 1:
                    raise IllegalPolicyAction(
                        "Relative policy must return one action chunk with shape (1, T, A); "
                        f"got {tuple(absolute_chunk.shape)}"
                    )
                self._absolute_action_queue.extend(absolute_chunk.transpose(0, 1))
            action = self._absolute_action_queue.popleft()
            latency = time.perf_counter() - started
            return action.squeeze(0).detach().cpu().numpy(), latency

        action = predict_action(
            observation=observation,
            policy=self.policy,
            device=self.device,
            preprocessor=self.preprocessor,
            postprocessor=self.postprocessor,
            use_amp=bool(self.policy.config.use_amp),
            task=TASK_INSTRUCTION,
            robot_type="unitree_g1_29_dex1",
        )
        latency = time.perf_counter() - started
        # X-VLA inference runs in bfloat16; numpy has no bf16 dtype.
        return action.squeeze(0).detach().float().cpu().numpy(), latency

    def close(self) -> None:
        if self.compat is not None:
            self.compat.close()
            self.compat = None
        return


class RemotePolicyRunner:
    """LingBot inference client connected through a loopback SSH tunnel."""

    def __init__(self, endpoint: str, authkey_file: Path, jpeg_quality: int) -> None:
        host, separator, port = endpoint.rpartition(":")
        if not separator or not host or not port.isdigit():
            raise ValueError("--remote-policy must have the form HOST:PORT")
        if host not in ("127.0.0.1", "localhost", "::1"):
            raise ValueError("Remote policy endpoint must be loopback; use an SSH tunnel")
        mode = authkey_file.stat().st_mode & 0o777
        if mode & 0o077:
            raise PermissionError(
                f"Remote auth-key file must be mode 0600: {authkey_file} mode={mode:o}"
            )
        self.connection = Client(
            (host, int(port)), authkey=authkey_file.read_bytes().strip()
        )
        self.jpeg_quality = jpeg_quality
        self.needs_images = True
        reply = self._request({"op": "hello"})
        if reply.get("policy") != "lingbot_va" or reply.get("output_shape") != [16]:
            raise RuntimeError(f"Unexpected remote policy metadata: {reply}")
        self.needs_images = bool(reply.get("needs_images", True))

    def _request(self, payload: dict) -> dict:
        self.connection.send(payload)
        reply = self.connection.recv()
        if not reply.get("ok"):
            detail = reply.get("traceback") or reply.get("error") or repr(reply)
            raise RuntimeError(f"Remote inference failed:\n{detail}")
        return reply

    def reset(self, seed: int) -> None:
        reply = self._request({"op": "reset", "seed": int(seed)})
        if int(reply.get("seed", -1)) != int(seed):
            raise RuntimeError(
                f"Remote policy did not acknowledge episode seed {seed}: {reply}"
            )
        self.needs_images = bool(reply.get("needs_images", True))

    def action(self, observation: dict[str, np.ndarray]) -> tuple[np.ndarray, float]:
        started = time.perf_counter()
        payload: dict[str, Any] = {
            "op": "action",
            "task": TASK_INSTRUCTION,
            "state": np.asarray(observation["observation.state"], dtype=np.float32).tolist(),
        }
        if self.needs_images:
            images: dict[str, bytes] = {}
            for key in CAMERA_KEYS:
                rgb = np.asarray(observation[key], dtype=np.uint8)
                bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                ok, encoded = cv2.imencode(
                    ".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality]
                )
                if not ok:
                    raise RuntimeError(f"Could not JPEG-encode {key}")
                images[key] = encoded.tobytes()
            payload["images"] = images
        reply = self._request(payload)
        self.needs_images = bool(reply.get("needs_images", True))
        action = np.asarray(reply["action"], dtype=np.float32)
        return action, time.perf_counter() - started

    def close(self) -> None:
        try:
            self._request({"op": "close"})
        except Exception:
            pass
        self.connection.close()


class SimulationPauseController:
    """Handshake with Isaac Sim so chunk inference consumes no physics time."""

    def __init__(self, request_path: Path, status_path: Path, timeout: float) -> None:
        self.request_path = request_path
        self.status_path = status_path
        self.timeout = float(timeout)
        self._active_token: str | None = None

    def freeze(self) -> None:
        if self._active_token is not None:
            raise RuntimeError("Simulation physics is already paused")
        token = f"{os.getpid()}-{time.monotonic_ns()}"
        self.status_path.unlink(missing_ok=True)
        self.request_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.request_path.with_name(
            f".{self.request_path.name}.{os.getpid()}.tmp"
        )
        temporary.write_text(token + "\n", encoding="utf-8")
        os.replace(temporary, self.request_path)
        deadline = time.monotonic() + self.timeout
        while time.monotonic() < deadline:
            try:
                if self.status_path.read_text(encoding="utf-8").strip() == token:
                    self._active_token = token
                    return
            except FileNotFoundError:
                pass
            time.sleep(0.005)
        self.request_path.unlink(missing_ok=True)
        raise TimeoutError(
            f"Isaac Sim did not acknowledge inference pause within {self.timeout:.1f}s"
        )

    def resume(self) -> None:
        if self._active_token is None:
            self.request_path.unlink(missing_ok=True)
            return
        token = self._active_token
        self.request_path.unlink(missing_ok=True)
        deadline = time.monotonic() + self.timeout
        while time.monotonic() < deadline:
            if not self.status_path.exists():
                self._active_token = None
                return
            time.sleep(0.005)
        raise TimeoutError(
            f"Isaac Sim did not release inference pause {token} within {self.timeout:.1f}s"
        )

    def close(self) -> None:
        self.request_path.unlink(missing_ok=True)
        self.status_path.unlink(missing_ok=True)
        self._active_token = None


class RTCPolicyRunner:
    """Asynchronous action-chunk runner for policies declaring RTC support."""

    def __init__(self, policy, preprocessor, postprocessor, device: torch.device) -> None:
        from lerobot.rollout.inference.rtc import supports_rtc_inference

        if not supports_rtc_inference(policy):
            raise ValueError(
                f"Policy {policy.config.type!r} does not declare compatible RTC inference; "
                "use --inference sync."
            )
        self.policy = policy
        self.preprocessor = preprocessor
        self.postprocessor = postprocessor
        self.device = device
        self._queue: deque[np.ndarray] = deque()
        self._observation: dict[str, np.ndarray] | None = None
        self._condition = threading.Condition()
        self._model_lock = threading.Lock()
        self._stopping = False
        self._epoch = 0
        self._error: str | None = None
        self._last_latency = math.nan
        self._thread = threading.Thread(target=self._worker, daemon=True, name="PolicyRTC")
        self._thread.start()

    def reset(self) -> None:
        with self._model_lock, self._condition:
            self.policy.reset()
            self.preprocessor.reset()
            self.postprocessor.reset()
            self._queue.clear()
            self._observation = None
            self._error = None
            self._epoch += 1

    def _worker(self) -> None:
        from lerobot.policies.utils import prepare_observation_for_inference

        while True:
            with self._condition:
                self._condition.wait_for(
                    lambda: self._stopping
                    or self._error is not None
                    or (self._observation is not None and len(self._queue) <= 10)
                )
                if self._stopping:
                    return
                observation = self._observation
                epoch = self._epoch
            try:
                started = time.perf_counter()
                with self._model_lock, torch.inference_mode():
                    batch = prepare_observation_for_inference(
                        observation,
                        self.device,
                        TASK_INSTRUCTION,
                        "unitree_g1_29_dex1",
                    )
                    batch = self.preprocessor(batch)
                    chunk = self.policy.predict_action_chunk(
                        batch,
                        inference_delay=0,
                        prev_chunk_left_over=None,
                    )
                    chunk = self.postprocessor(chunk).squeeze(0).detach().cpu().numpy()
                latency = time.perf_counter() - started
                if chunk.ndim != 2 or chunk.shape[1] != 16:
                    raise ValueError(f"RTC policy returned invalid chunk shape {chunk.shape}")
                with self._condition:
                    if epoch == self._epoch:
                        self._queue.extend(np.asarray(row, dtype=np.float32) for row in chunk)
                        self._last_latency = latency
            except Exception:
                with self._condition:
                    self._error = traceback.format_exc()

    def action(self, observation: dict[str, np.ndarray]) -> tuple[np.ndarray | None, float]:
        with self._condition:
            if self._error:
                raise RuntimeError(f"RTC inference failed:\n{self._error}")
            self._observation = observation
            self._condition.notify_all()
            if not self._queue:
                return None, math.nan
            return self._queue.popleft(), self._last_latency

    def close(self) -> None:
        with self._condition:
            self._stopping = True
            self._condition.notify_all()
        self._thread.join(timeout=3.0)


class RawKeyboard:
    def __init__(self) -> None:
        self.fd = sys.stdin.fileno()
        self.enabled = sys.stdin.isatty()
        self.settings = termios.tcgetattr(self.fd) if self.enabled else None

    def __enter__(self) -> "RawKeyboard":
        import tty

        if self.enabled:
            tty.setcbreak(self.fd)
        return self

    def poll(self) -> str | None:
        if not self.enabled:
            return None
        readable, _, _ = select.select([sys.stdin], [], [], 0.0)
        return sys.stdin.read(1).lower() if readable else None

    def __exit__(self, *_args) -> None:
        if self.enabled and self.settings is not None:
            termios.tcsetattr(self.fd, termios.TCSADRAIN, self.settings)


class BridgeClient:
    def __init__(self, socket_path: Path, authkey: str) -> None:
        self.connection = Client(
            str(socket_path), family="AF_UNIX", authkey=authkey.encode()
        )

    def request(self, command: str, **payload) -> dict:
        self.connection.send({"command": command, **payload})
        response = self.connection.recv()
        if not response.get("ok"):
            raise RuntimeError(response.get("error", "Unknown DDS bridge error"))
        return response

    def close(self) -> None:
        try:
            self.request("shutdown")
        except Exception:
            pass
        self.connection.close()


def try_hold(bridge: BridgeClient) -> None:
    """Best-effort hold that does not hide the original infrastructure error."""

    try:
        bridge.request("hold")
    except Exception:
        pass


def load_policy(checkpoint: Path, inference: str):
    # Legacy checkpoints can carry config fields the current config class no
    # longer accepts (e.g. rtc_training_max_delay). CompatibleCheckpoint
    # presents a symlinked view with those fields stripped, without ever
    # mutating the original config.json.
    from policy_runtime import CompatibleCheckpoint

    compat = CompatibleCheckpoint(checkpoint)
    try:
        config = PreTrainedConfig.from_pretrained(str(compat.path))
        validate_checkpoint_features(config)
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required but torch.cuda.is_available() is false")
        config.device = "cuda"
        policy_class = get_policy_class(config.type)
        policy = policy_class.from_pretrained(
            pretrained_name_or_path=str(compat.path), config=config
        )
        policy.to("cuda")
        policy.eval()
        preprocessor, postprocessor = make_pre_post_processors(
            policy_cfg=config,
            pretrained_path=str(compat.path),
            preprocessor_overrides={"device_processor": {"device": "cuda"}},
        )
        runner_class = SyncPolicyRunner if inference == "sync" else RTCPolicyRunner
        if runner_class is SyncPolicyRunner:
            runner = runner_class(
                policy, preprocessor, postprocessor, torch.device("cuda"),
                key_map=camera_key_map(config.type),
            )
        else:
            runner = runner_class(policy, preprocessor, postprocessor, torch.device("cuda"))
    except Exception:
        compat.close()
        raise
    runner.compat = compat
    return config, runner


def make_observation(state: np.ndarray, frames: dict) -> dict[str, np.ndarray]:
    observation: dict[str, np.ndarray] = {
        "observation.state": np.asarray(state, dtype=np.float32)
    }
    for key, frame in frames.items():
        observation[key] = cv2.cvtColor(frame.bgr, cv2.COLOR_BGR2RGB)
    return observation


def validate_dds_state(response: dict, stale_timeout: float) -> np.ndarray:
    age = float(response["dds_age"])
    if age > stale_timeout:
        raise ConnectionError(f"DDS state is stale ({age:.3f}s)")
    state = np.asarray(response["state"], dtype=np.float32)
    if state.shape != (16,) or not np.isfinite(state).all():
        raise ConnectionError(f"Invalid DDS state: shape={state.shape}")
    return state


def validate_camera_frames(frames: dict | None, stale_timeout: float) -> dict:
    if frames is None or set(frames) != set(CAMERA_KEYS):
        raise ConnectionError("One or more camera streams are unavailable")
    oldest_age = max(frame.age_seconds for frame in frames.values())
    if oldest_age > stale_timeout:
        raise ConnectionError(f"Camera stream is stale ({oldest_age:.3f}s)")
    return frames


def validate_policy_action(action: np.ndarray) -> np.ndarray:
    action = np.asarray(action, dtype=np.float32).reshape(-1)
    if action.shape != (16,) or not np.isfinite(action).all():
        raise IllegalPolicyAction(
            f"Policy produced illegal action shape={action.shape}"
        )
    return action


def wait_for_live_remote_observation(
    bridge: BridgeClient,
    camera: CameraReader,
    stale_timeout: float,
    wait_timeout: float,
    previous_timestamps: dict[str, int] | None,
) -> None:
    """Wait for a live frame after the prior keyframe before freezing physics."""

    deadline = time.monotonic() + wait_timeout
    last_dds_age = math.inf
    last_camera_age = math.inf
    while time.monotonic() < deadline:
        response = bridge.request("get_state")
        validate_dds_state(response, math.inf)
        last_dds_age = float(response["dds_age"])
        frames = validate_camera_frames(camera.read_all(), math.inf)
        last_camera_age = max(frame.age_seconds for frame in frames.values())
        timestamps = {key: frame.timestamp_ms for key, frame in frames.items()}
        advanced = previous_timestamps is None or all(
            timestamps[key] > previous_timestamps[key] for key in CAMERA_KEYS
        )
        if (
            last_dds_age <= stale_timeout
            and last_camera_age <= stale_timeout
            and advanced
        ):
            return
        time.sleep(0.01)
    raise ConnectionError(
        "Remote-evaluation sensors did not advance before the next keyframe "
        f"(dds_age={last_dds_age:.3f}s, camera_age={last_camera_age:.3f}s)"
    )


def percentile(values: list[float], q: float) -> float | None:
    finite = np.asarray([value for value in values if np.isfinite(value)], dtype=np.float64)
    return None if finite.size == 0 else float(np.percentile(finite, q))


def wilson_interval(successes: int, total: int) -> list[float] | None:
    if total == 0:
        return None
    z = 1.959963984540054
    p = successes / total
    denominator = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denominator
    radius = z * math.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total)) / denominator
    return [max(0.0, center - radius), min(1.0, center + radius)]


def summarize(records: list[dict], all_latencies: list[float]) -> dict:
    valid = [record for record in records if record.get("valid")]
    successes = [record for record in valid if record["status"] == "success"]
    timeouts = [record for record in valid if record["status"] == "timeout"]
    success_times = [float(record["elapsed_seconds"]) for record in successes]
    summary = {
        "success_rate": len(successes) / len(valid) if valid else None,
        "wilson_95_interval": wilson_interval(len(successes), len(valid)),
        "successful_episodes": len(successes),
        "timeout_episodes": len(timeouts),
        "valid_episodes": len(valid),
        "invalid_episodes": len(records) - len(valid),
        "mean_success_time_seconds": float(np.mean(success_times)) if success_times else None,
        "inference_latency_ms_p50": (
            None if percentile(all_latencies, 50) is None else 1000.0 * percentile(all_latencies, 50)
        ),
        "inference_latency_ms_p95": (
            None if percentile(all_latencies, 95) is None else 1000.0 * percentile(all_latencies, 95)
        ),
        "updated_at": datetime.now().astimezone().isoformat(),
    }
    breakdown: dict[str, dict] = {}
    for record in records:
        if not record.get("valid") or record.get("perturbation") is None:
            continue
        slot = breakdown.setdefault(
            str(record["perturbation"]), {"valid_episodes": 0, "successful_episodes": 0}
        )
        slot["valid_episodes"] += 1
        slot["successful_episodes"] += int(record["status"] == "success")
    if breakdown:
        for slot in breakdown.values():
            slot["success_rate"] = slot["successful_episodes"] / slot["valid_episodes"]
        summary["perturbation_breakdown"] = breakdown
    return summary


def checkpoint_step(checkpoint: Path) -> str:
    for part in reversed(checkpoint.parts):
        match = re.fullmatch(r"(?:checkpoint-)?(\d{3,})", part)
        if match:
            return match.group(1)
    return "unknown"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--socket", type=Path, required=True)
    parser.add_argument("--authkey", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--success-marker-file", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--frequency", type=float, default=30.0)
    parser.add_argument("--initial-gripper", type=float, default=4.0)
    parser.add_argument("--inference", choices=("sync", "rtc"), default="sync")
    parser.add_argument("--stale-timeout", type=float, default=1.0)
    parser.add_argument(
        "--remote-policy",
        help="Loopback HOST:PORT of an SSH-tunnelled LingBot inference server",
    )
    parser.add_argument("--remote-authkey-file", type=Path)
    parser.add_argument("--remote-jpeg-quality", type=int, default=90)
    parser.add_argument(
        "--sim-pause-request-file",
        type=Path,
        default=Path(f"/tmp/g1_tabletop_grasp_{os.getuid()}.lingbot_pause_request"),
    )
    parser.add_argument(
        "--sim-pause-status-file",
        type=Path,
        default=Path(f"/tmp/g1_tabletop_grasp_{os.getuid()}.lingbot_pause_status"),
    )
    parser.add_argument("--sim-pause-timeout", type=float, default=10.0)
    parser.add_argument(
        "--auto-start", action="store_true", help="Start evaluation without an interactive prompt"
    )
    parser.add_argument(
        "--perturb",
        default="none",
        choices=PERTURBATION_VARIANTS + ("random",),
        help="Training-matched visual perturbation: a fixed variant, a "
        "per-episode random draw (perturbation or none), or off (default).",
    )
    parser.add_argument(
        "--perturb-seed",
        type=int,
        default=0,
        help="Salt for the random per-episode variant draw.",
    )
    parser.add_argument(
        "--perturb-particle-count",
        type=int,
        default=TRAINING_PARTICLE_COUNT,
        help="Dust particle count for *_dust variants (training default 5000).",
    )
    parser.add_argument(
        "--timeout-steps",
        type=int,
        default=0,
        help="End an episode as timeout after this many executed steps "
        "instead of the wall-clock budget; 0 keeps the wall-clock "
        "behaviour. Use the measured clean-baseline step cap so perturbed "
        "episodes get identical control-time budgets.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.checkpoint = args.checkpoint.expanduser().resolve()
    if not (args.checkpoint / "config.json").is_file():
        raise FileNotFoundError(f"Invalid pretrained_model checkpoint: {args.checkpoint}")
    if args.episodes <= 0 or args.timeout <= 0 or args.frequency <= 0:
        raise ValueError("episodes, timeout, and frequency must be positive")

    if args.remote_policy:
        if args.inference != "sync":
            raise ValueError("Remote LingBot evaluation currently requires --inference sync")
        if args.remote_authkey_file is None:
            raise ValueError("--remote-authkey-file is required with --remote-policy")
        if not 50 <= args.remote_jpeg_quality <= 100:
            raise ValueError("--remote-jpeg-quality must be between 50 and 100")
        policy_config = load_remote_checkpoint_config(args.checkpoint)
        runner = RemotePolicyRunner(
            args.remote_policy,
            args.remote_authkey_file.expanduser().resolve(),
            args.remote_jpeg_quality,
        )
        pause_controller = SimulationPauseController(
            args.sim_pause_request_file,
            args.sim_pause_status_file,
            args.sim_pause_timeout,
        )
    else:
        policy_config, runner = load_policy(args.checkpoint, args.inference)
        pause_controller = None
    policy_name = str(policy_config.type)
    run_name = (
        f"{policy_name}_{checkpoint_step(args.checkpoint)}_"
        f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    output_dir = args.output_root / run_name
    rollout_dir = output_dir / "rollouts"
    output_dir.mkdir(parents=True, exist_ok=False)

    xr_root = Path(
        os.environ.get("XR_TELEOP_ROOT")
        or (Path(__file__).resolve().parents[2] / "third_party" / "xr_teleoperate")
    )
    sys.path.insert(0, str(xr_root))
    from teleop.utils.lerobot_v30_writer import LeRobotV30Writer

    writer = LeRobotV30Writer(
        root=rollout_dir,
        fps=args.frequency,
        instruction=TASK_INSTRUCTION,
        state_names=STATE_ACTION_NAMES,
        action_names=STATE_ACTION_NAMES,
        camera_keys=list(CAMERA_KEYS),
        resume=False,
    )
    config_payload = {
        "checkpoint": str(args.checkpoint),
        "policy": policy_name,
        "dataset": str(rollout_dir),
        "source_training_dataset": (
            f"{Path(__file__).resolve().parent}/"
            "data/lerobot_v30/detonator_wire_insertion"
        ),
        "instruction": TASK_INSTRUCTION,
        "seed": args.seed,
        "episodes": args.episodes,
        "timeout_seconds": args.timeout,
        "timeout_steps": args.timeout_steps,
        "timeout_basis": (
            "executed_control_time"
            if args.remote_policy
            else (
                f"steps:{args.timeout_steps} (matched to clean baseline measured cap)"
                if args.timeout_steps > 0
                else "wall_clock"
            )
        ),
        "frequency_hz": args.frequency,
        "initial_gripper_rad": args.initial_gripper,
        "inference": args.inference,
        "remote_policy": args.remote_policy,
        "remote_jpeg_quality": args.remote_jpeg_quality if args.remote_policy else None,
        "remote_temporal_consistency": (
            {
                "mode": "pause_physics_during_fresh_chunk_inference",
                "fixed_control_dt_seconds": 1.0 / args.frequency,
                "episode_policy_seeded": True,
            }
            if args.remote_policy
            else None
        ),
        "stale_timeout_seconds": args.stale_timeout,
        "object_randomization": {
            "profile": "collision_safe_v2",
            "world_xy_limit_m": 0.05,
            "leiguan_min_table_local_depth_offset_m": -0.02,
        },
        "action_safety": {
            "dimension": 16,
            "arm_joint_limits": "official G1 URDF",
            "arm_velocity_limit_rad_s": 3.0,
            "gripper_range_rad": [0.0, 5.4],
            "gripper_max_delta_rad_per_tick": 0.18,
        },
        "perturbation": (
            {
                "mode": args.perturb,
                "selection_seed_salt": args.perturb_seed,
                "particle_count": args.perturb_particle_count,
                "training_augmentation_seed": TRAINING_AUGMENTATION_SEED,
                "variants": list(PERTURBATION_VARIANTS),
                "applied_to": ["policy_observation", "rollout_videos"],
            }
            if args.perturb != "none"
            else None
        ),
        "created_at": datetime.now().astimezone().isoformat(),
    }
    write_json_atomic(output_dir / "config.json", config_payload)
    records: list[dict] = []
    all_latencies: list[float] = []
    episodes_path = output_dir / "episodes.jsonl"

    print(f"Checkpoint loaded: {policy_name} | {args.checkpoint}")
    print(f"Evaluation output: {output_dir}")
    if not args.auto_start:
        print("Press [r] then Enter once to start automatic evaluation; [s] aborts one episode; [q] stops.")
        while True:
            choice = input("eval> ").strip().lower()
            if choice == "r":
                break
            if choice == "q":
                writer.close()
                runner.close()
                return

    bridge = BridgeClient(args.socket, args.authkey)
    camera = CameraReader()
    valid_count = 0
    infra_failures_for_slot = 0
    stop_requested = False
    try:
        with RawKeyboard() as keyboard:
            while valid_count < args.episodes and not stop_requested:
                episode_seed = args.seed + valid_count
                episode_number = valid_count
                perturber = None
                perturbation_name = "none"
                if args.perturb != "none":
                    perturbation_name = (
                        choose_variant(episode_seed, salt=args.perturb_seed)
                        if args.perturb == "random"
                        else args.perturb
                    )
                    if perturbation_name != "none":
                        perturber = EpisodePerturber(
                            perturbation_name,
                            episode_seed,
                            particle_count=args.perturb_particle_count,
                        )
                try:
                    reset = bridge.request(
                        "reset", episode_index=episode_number, seed=episode_seed
                    )["reset"]
                except Exception as error:
                    try_hold(bridge)
                    infra_failures_for_slot += 1
                    record = {
                        "attempt_index": len(records),
                        "episode_index": episode_number,
                        "seed": episode_seed,
                        "perturbation": perturbation_name,
                        "status": "infrastructure_error",
                        "valid": False,
                        "elapsed_seconds": 0.0,
                        "steps": 0,
                        "rollout_episode_index": None,
                        "rollout_frames": 0,
                        "leiguan_offset_xy_m": None,
                        "rod_offset_xy_m": None,
                        "insertion_depth_m": 0.0,
                        "max_radial_error_m": None,
                        "inference_latency_ms_p50": None,
                        "inference_latency_ms_p95": None,
                        "error": f"{type(error).__name__}: {error}",
                    }
                    records.append(record)
                    with episodes_path.open("a", encoding="utf-8") as file:
                        file.write(json.dumps(record, ensure_ascii=False) + "\n")
                        file.flush()
                        os.fsync(file.fileno())
                    write_json_atomic(
                        output_dir / "summary.json", summarize(records, all_latencies)
                    )
                    if infra_failures_for_slot >= 2:
                        print("Episode reset infrastructure failed twice; stopping evaluation.")
                        stop_requested = True
                    else:
                        print("Episode reset failed; automatically retrying once.")
                        time.sleep(0.5)
                    continue
                args.success_marker_file.unlink(missing_ok=True)
                if args.remote_policy:
                    runner.reset(episode_seed)
                else:
                    runner.reset()
                writer.start_episode()
                started = time.monotonic()
                step = 0
                episode_latencies: list[float] = []
                episode_pause_count = 0
                episode_paused_seconds = 0.0
                last_remote_image_timestamps: dict[str, int] | None = None
                status = "timeout"
                error_message = None
                success_payload: dict[str, Any] = {}
                print(
                    f"\nEpisode {valid_count + 1}/{args.episodes} "
                    f"seed={episode_seed} perturbation={perturbation_name} "
                    f"offsets={reset['leiguan_offset_xy_m']}/"
                    f"{reset['rod_offset_xy_m']}"
                )
                try:
                    while (
                        (step < args.timeout_steps if args.timeout_steps > 0 else True)
                        and (
                            step / args.frequency
                            if args.remote_policy
                            else time.monotonic() - started
                        )
                        < (
                            4.0 * args.timeout
                            if args.timeout_steps > 0
                            else args.timeout
                        )
                    ):
                        tick_started = time.monotonic()
                        key = keyboard.poll()
                        if key == "q":
                            status = "manual_stop"
                            stop_requested = True
                            break
                        if key == "s":
                            status = "manual_abort"
                            break

                        pause_for_inference = bool(
                            args.remote_policy and runner.needs_images
                        )
                        pause_started = None
                        if pause_for_inference:
                            wait_for_live_remote_observation(
                                bridge,
                                camera,
                                args.stale_timeout,
                                args.sim_pause_timeout,
                                last_remote_image_timestamps,
                            )
                            pause_started = time.monotonic()
                            pause_controller.freeze()
                        response = None
                        try:
                            state_response = bridge.request("get_state")
                            state = validate_dds_state(
                                state_response,
                                math.inf if args.remote_policy else args.stale_timeout,
                            )
                            frames = validate_camera_frames(
                                camera.read_all(),
                                math.inf if args.remote_policy else args.stale_timeout,
                            )
                            if perturber is not None:
                                frames = perturber.apply(frames)
                            if pause_for_inference:
                                last_remote_image_timestamps = {
                                    key: frame.timestamp_ms
                                    for key, frame in frames.items()
                                }
                            observation = make_observation(state, frames)
                            action, latency = runner.action(observation)
                            # Publish the new keyframe action while physics is
                            # still frozen. The first resumed simulation step
                            # therefore cannot consume the previous target.
                            if pause_for_inference and action is not None:
                                action = validate_policy_action(action)
                                response = bridge.request(
                                    "action",
                                    action=action.tolist(),
                                    control_dt=1.0 / args.frequency,
                                )
                        finally:
                            if pause_for_inference:
                                pause_controller.resume()
                                episode_pause_count += 1
                                episode_paused_seconds += (
                                    time.monotonic() - pause_started
                                )
                        if action is None:
                            bridge.request("hold")
                        else:
                            action = validate_policy_action(action)
                            if response is None:
                                action_payload: dict[str, Any] = {
                                    "action": action.tolist()
                                }
                                if args.remote_policy:
                                    action_payload["control_dt"] = 1.0 / args.frequency
                                response = bridge.request("action", **action_payload)
                            if (
                                not args.remote_policy
                                and float(response["dds_age"]) > args.stale_timeout
                            ):
                                raise ConnectionError(
                                    f"DDS state became stale ({response['dds_age']:.3f}s)"
                                )
                            executed = np.asarray(response["action"], dtype=np.float32)
                            writer.add_frame(
                                state=state,
                                action=executed,
                                images={key: value.bgr for key, value in frames.items()},
                            )
                            if np.isfinite(latency):
                                episode_latencies.append(float(latency))
                                all_latencies.append(float(latency))
                            step += 1

                        if args.success_marker_file.is_file():
                            bridge.request("hold")
                            success_payload = json.loads(
                                args.success_marker_file.read_text(encoding="utf-8")
                            )
                            status = "success"
                            break
                        remaining = 1.0 / args.frequency - (time.monotonic() - tick_started)
                        if remaining > 0:
                            time.sleep(remaining)
                except (ConnectionError, EOFError, BrokenPipeError) as error:
                    try_hold(bridge)
                    status = "infrastructure_error"
                    error_message = f"{type(error).__name__}: {error}"
                except IllegalPolicyAction as error:
                    try_hold(bridge)
                    status = "policy_error"
                    error_message = f"{type(error).__name__}: {error}"
                    stop_requested = True
                except Exception as error:
                    try_hold(bridge)
                    status = "infrastructure_error"
                    error_message = f"{type(error).__name__}: {error}"

                elapsed = time.monotonic() - started
                control_elapsed = step / args.frequency
                valid = status in ("success", "timeout")
                if valid:
                    rollout_episode, rollout_frames = writer.save_episode()
                    valid_count += 1
                    infra_failures_for_slot = 0
                else:
                    rollout_episode = None
                    rollout_frames = writer.discard_episode()
                record = {
                    "attempt_index": len(records),
                    "episode_index": episode_number,
                    "seed": episode_seed,
                    "perturbation": perturbation_name,
                    "status": status,
                    "valid": valid,
                    "elapsed_seconds": elapsed,
                    "control_elapsed_seconds": control_elapsed,
                    "steps": step,
                    "inference_pause_count": episode_pause_count,
                    "inference_paused_wall_seconds": episode_paused_seconds,
                    "rollout_episode_index": rollout_episode,
                    "rollout_frames": rollout_frames,
                    "leiguan_offset_xy_m": reset["leiguan_offset_xy_m"],
                    "leiguan_table_local_depth_offset_m": reset.get(
                        "leiguan_table_local_depth_offset_m"
                    ),
                    "leiguan_sample_attempts": reset.get(
                        "leiguan_sample_attempts", 1
                    ),
                    "rod_offset_xy_m": reset["rod_offset_xy_m"],
                    "insertion_depth_m": success_payload.get("insertion_depth_m", 0.0),
                    "max_radial_error_m": success_payload.get("max_radial_error_m"),
                    "inference_latency_ms_p50": (
                        None if percentile(episode_latencies, 50) is None else 1000.0 * percentile(episode_latencies, 50)
                    ),
                    "inference_latency_ms_p95": (
                        None if percentile(episode_latencies, 95) is None else 1000.0 * percentile(episode_latencies, 95)
                    ),
                    "error": error_message,
                }
                records.append(record)
                with episodes_path.open("a", encoding="utf-8") as file:
                    file.write(json.dumps(record, ensure_ascii=False) + "\n")
                    file.flush()
                    os.fsync(file.fileno())
                write_json_atomic(
                    output_dir / "summary.json", summarize(records, all_latencies)
                )
                print(
                    f"Episode result: {status}, elapsed={elapsed:.2f}s, "
                    f"control_elapsed={control_elapsed:.2f}s, steps={step}, valid={valid}"
                )
                if status == "infrastructure_error":
                    infra_failures_for_slot += 1
                    if infra_failures_for_slot >= 2:
                        print("Infrastructure failed twice for the same episode; stopping evaluation.")
                        stop_requested = True
                    else:
                        print("Infrastructure error; automatically retrying this episode once.")
                elif status == "manual_abort":
                    infra_failures_for_slot = 0
                if status == "policy_error":
                    print("Illegal policy action; evaluation stopped with the last safe target held.")
    finally:
        writer.close()
        runner.close()
        if pause_controller is not None:
            pause_controller.close()
        camera.close()
        bridge.close()
        write_json_atomic(output_dir / "summary.json", summarize(records, all_latencies))
        print(f"Evaluation finished. Results: {output_dir}")


if __name__ == "__main__":
    main()
