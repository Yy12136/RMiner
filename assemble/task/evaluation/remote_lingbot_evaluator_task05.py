#!/usr/bin/env python3
"""Run the shared evaluator with a remote LingBot-VA policy client.

This wrapper keeps the development machine free of the 5B model.  The shared
evaluator still owns Task05 reset, success detection, cameras, videos, and DDS.
"""
from __future__ import annotations

import os
import runpy
import sys
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from collections import deque
import types
from multiprocessing.connection import Client
from pathlib import Path

import cv2
import numpy as np

import camera as _camera


class FreshCameraReader(_camera.CameraReader):
    """Wait for a newly rendered frame after reset before remote inference."""

    def __init__(self) -> None:
        super().__init__()
        self._last_timestamps: dict[str, int] = {}

    def read_all(self, attempts: int = 8):
        # Task05 reset can temporarily block Isaac camera publication while
        # the PhysX joint settles.  Do not fall back to an old frame: wait long
        # enough for a genuinely new frame, otherwise the attempt is invalid.
        deadline = time.monotonic() + 60.0
        while time.monotonic() < deadline:
            frames = super().read_all(attempts)
            if frames is not None:
                fresh = all(frame.age_seconds <= 1.0 for frame in frames.values())
                advanced = (
                    not self._last_timestamps
                    or any(frame.timestamp_ms != self._last_timestamps.get(key)
                           for key, frame in frames.items())
                )
                if fresh and advanced:
                    self._last_timestamps = {
                        key: frame.timestamp_ms for key, frame in frames.items()
                    }
                    return frames
            time.sleep(0.02)
        raise RuntimeError("camera streams did not publish a fresh frame after reset")


class IllegalPolicyAction(RuntimeError):
    pass


class _Compat:
    def close(self) -> None:
        return


class _Config:
    type = "lingbot_va"


class RemoteRunner:
    def __init__(self, endpoint: str, auth_file: Path, spec) -> None:
        host, sep, port = endpoint.rpartition(":")
        if not sep or not host or not port.isdigit():
            raise ValueError("LINGBOT_REMOTE_ENDPOINT must be HOST:PORT")
        mode = auth_file.stat().st_mode & 0o777
        if mode & 0o077:
            raise PermissionError(f"Remote auth key must be mode 0600: {auth_file}")
        self.connection = Client((host, int(port)), authkey=auth_file.read_bytes().strip())
        hello = self._request({"op": "hello"})
        if hello.get("policy") != "lingbot_va" or hello.get("output_shape") != [spec.action_dim]:
            raise RuntimeError(f"Unexpected remote policy metadata: {hello}")
        self.spec = spec
        self.seed = int(os.environ.get("LINGBOT_REMOTE_SEED", "42"))
        self.reset_count = 0
        self.pause_request = Path(os.environ["G1_INFERENCE_PAUSE_REQUEST_FILE"])
        self.pause_status = Path(os.environ["G1_INFERENCE_PAUSE_STATUS_FILE"])

    def _request(self, payload: dict) -> dict:
        self.connection.send(payload)
        reply = self.connection.recv()
        if not reply.get("ok"):
            raise RuntimeError(reply.get("error", "remote inference failed"))
        return reply

    def reset(self) -> None:
        # evaluator calls reset once per valid episode; seeds are sequential.
        self._request({"op": "reset", "seed": self.seed + self.reset_count})
        self.reset_count += 1

    def action(self, observation: dict[str, np.ndarray]) -> tuple[np.ndarray, float]:
        token = f"{os.getpid()}-{time.time_ns()}"
        self.pause_request.parent.mkdir(parents=True, exist_ok=True)
        self.pause_request.write_text(token + "\n", encoding="utf-8")
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline:
            try:
                if self.pause_status.read_text(encoding="utf-8").strip() == token:
                    break
            except FileNotFoundError:
                pass
            time.sleep(0.005)
        else:
            self.pause_request.unlink(missing_ok=True)
            raise RuntimeError("simulator did not acknowledge inference pause")

        encoded: dict[str, bytes] = {}
        for key in ("observation.images.head", "observation.images.left_wrist", "observation.images.right_wrist"):
            rgb = np.asarray(observation[key])
            ok, buf = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 90])
            if not ok:
                self.pause_request.unlink(missing_ok=True)
                raise RuntimeError(f"failed to encode {key}")
            encoded[key] = buf.tobytes()
        started = time.perf_counter()
        try:
            reply = self._request({
                "op": "action", "images": encoded,
                "state": np.asarray(observation["observation.state"], dtype=np.float32).tolist(),
                "task": self.spec.instruction,
            })
        finally:
            self.pause_request.unlink(missing_ok=True)
            release_deadline = time.monotonic() + 10.0
            while time.monotonic() < release_deadline and self.pause_status.exists():
                time.sleep(0.005)
        action = np.asarray(reply["action"], dtype=np.float32)
        if action.shape != (self.spec.action_dim,) or not np.isfinite(action).all():
            raise IllegalPolicyAction(f"illegal remote action shape={action.shape}")
        return action, time.perf_counter() - started

    def close(self) -> None:
        try:
            self._request({"op": "close"})
        except Exception:
            pass
        try:
            self.connection.close()
        except Exception:
            pass


class AsyncChunkRunner:
    """Remote LingBot runner with a non-blocking action-chunk queue.

    The first chunk is fetched synchronously.  Subsequent chunks are requested
    by a background worker while the evaluator keeps sending the last queued
    target at the control frequency.  The simulator is never paused during a
    slow model call.  Keyframes are sent in the same stride used by the native
    LingBot streaming policy; if a chunk is not ready yet, the fallback action
    is explicitly counted in the episode record.
    """

    def __init__(self, endpoint: str, auth_file: Path, spec) -> None:
        host, sep, port = endpoint.rpartition(":")
        if not sep or not host or not port.isdigit():
            raise ValueError("LINGBOT_REMOTE_ENDPOINT must be HOST:PORT")
        mode = auth_file.stat().st_mode & 0o777
        if mode & 0o077:
            raise PermissionError(f"Remote auth key must be mode 0600: {auth_file}")
        self.connection = Client((host, int(port)), authkey=auth_file.read_bytes().strip())
        self.lock = threading.Lock()
        hello = self._request({"op": "hello"})
        if hello.get("policy") != "lingbot_va" or hello.get("output_shape") != [spec.action_dim]:
            raise RuntimeError(f"Unexpected remote policy metadata: {hello}")
        if not hello.get("supports_action_chunks"):
            raise RuntimeError("Remote LingBot server does not support chunk mode")
        self.spec = spec
        self.seed = int(os.environ.get("LINGBOT_REMOTE_SEED", "42"))
        self.reset_count = 0
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="lingbot-chunk")
        self.future: Future | None = None
        self.actions: deque[np.ndarray] = deque()
        self.last_action = np.zeros(spec.action_dim, dtype=np.float32)
        self.keyframes: list[dict] = []
        self.step_in_chunk = 0
        self.chunk_index = 0
        self.started = False
        self.starved_steps = 0
        self.total_actions = 0
        self.keyframe_stride = int(os.environ.get("LINGBOT_KEYFRAME_STRIDE", "4"))

    def needs_observation(self) -> bool:
        """Whether this control tick must acquire a new camera keyframe.

        LingBot consumes one visual keyframe every ``keyframe_stride`` action
        ticks.  Intermediate ticks can reuse the most recent frames while the
        control loop continues at its requested rate.
        """
        return (not self.started) or (self.step_in_chunk % self.keyframe_stride == 0)

    def _request(self, payload: dict) -> dict:
        with self.lock:
            self.connection.send(payload)
            reply = self.connection.recv()
        if not reply.get("ok"):
            raise RuntimeError(reply.get("error", "remote inference failed"))
        return reply

    @staticmethod
    def _encode_observation(observation: dict[str, np.ndarray], spec) -> dict:
        encoded: dict[str, bytes] = {}
        for key in ("observation.images.head", "observation.images.left_wrist", "observation.images.right_wrist"):
            rgb = np.asarray(observation[key])
            ok, buf = cv2.imencode(
                ".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 90]
            )
            if not ok:
                raise RuntimeError(f"failed to encode {key}")
            encoded[key] = buf.tobytes()
        return {
            "images": encoded,
            "state": np.asarray(observation["observation.state"], dtype=np.float32).tolist(),
            "task": spec.instruction,
        }

    @staticmethod
    def _decode_actions(reply: dict, spec) -> list[np.ndarray]:
        actions = [np.asarray(a, dtype=np.float32) for a in reply.get("actions", [])]
        if not actions or any(a.shape != (spec.action_dim,) or not np.isfinite(a).all() for a in actions):
            raise RuntimeError(f"Illegal remote action chunk: {reply.get('action_shape')}")
        return actions

    def reset(self) -> None:
        if self.future is not None:
            self.future.cancel()
            self.future = None
        self._request({"op": "reset", "seed": self.seed + self.reset_count})
        self.reset_count += 1
        self.actions.clear()
        self.keyframes.clear()
        self.last_action.fill(0.0)
        self.step_in_chunk = 0
        self.chunk_index = 0
        self.started = False
        self.starved_steps = 0
        self.total_actions = 0

    def _fetch_next(self, keyframes: list[dict]) -> list[np.ndarray]:
        reply = self._request({"op": "chunk_next", "keyframes": keyframes, "task": self.spec.instruction})
        return self._decode_actions(reply, self.spec)

    def _poll_future(self) -> None:
        if self.future is None or not self.future.done():
            return
        future, self.future = self.future, None
        actions = future.result()
        self.actions.extend(actions)
        self.keyframes.clear()
        self.step_in_chunk = 0
        self.chunk_index += 1

    def _start_next_if_ready(self) -> None:
        if self.future is not None:
            # Once the complete keyframe set is available, request the next
            # chunk immediately.  The worker owns the socket; the evaluator
            # keeps issuing queued actions while inference runs.
            return
        required = 4 if self.chunk_index == 0 else 8
        if len(self.keyframes) < required:
            return
        snapshot = list(self.keyframes)
        self.future = self.executor.submit(self._fetch_next, snapshot)

    def action(self, observation: dict[str, np.ndarray]) -> tuple[np.ndarray, float]:
        self._poll_future()
        # The observation at offsets 4, 8, ... is the post-action keyframe
        # consumed by LingBot's streaming feedback path.
        if self.total_actions > 0 and self.step_in_chunk % self.keyframe_stride == 0:
            self.keyframes.append(self._encode_observation(observation, self.spec))

        if not self.started:
            started = time.perf_counter()
            payload = self._encode_observation(observation, self.spec)
            reply = self._request({"op": "chunk_start", **payload})
            self.actions.extend(self._decode_actions(reply, self.spec))
            self.keyframes.clear()
            self.step_in_chunk = 0
            self.started = True
            latency = time.perf_counter() - started
        else:
            latency = 0.0

        self._start_next_if_ready()
        if self.actions:
            action = self.actions.popleft()
            self.last_action = action.copy()
        else:
            action = self.last_action.copy()
            self.starved_steps += 1
        self.step_in_chunk += 1
        self.total_actions += 1
        return action, latency

    def close(self) -> None:
        if self.future is not None:
            self.future.cancel()
            self.future = None
        # Do not send the protocol close while the worker may still be
        # receiving a chunk; waiting keeps the single socket message order
        # deterministic and prevents a stale inference thread on shutdown.
        self.executor.shutdown(wait=True, cancel_futures=True)
        try:
            self._request({"op": "close"})
        except Exception:
            pass
        try:
            self.connection.close()
        except Exception:
            pass


def remote_load_policy(checkpoint: Path, spec):
    endpoint = os.environ["LINGBOT_REMOTE_ENDPOINT"]
    auth_file = Path(os.environ["LINGBOT_REMOTE_AUTHKEY_FILE"])
    mode = os.environ.get("LINGBOT_REMOTE_INFERENCE_MODE", "sync").lower()
    runner = (
        AsyncChunkRunner(endpoint, auth_file, spec)
        if mode == "async"
        else RemoteRunner(endpoint, auth_file, spec)
    )
    return _Config(), runner, _Compat()


def main() -> None:
    evaluation_dir = Path(__file__).resolve().parent
    sys.path.insert(0, str(evaluation_dir))
    _camera.CameraReader = FreshCameraReader
    fake = types.ModuleType("policy_runtime")
    fake.IllegalPolicyAction = IllegalPolicyAction
    fake.load_policy = remote_load_policy
    sys.modules["policy_runtime"] = fake
    sys.argv[0] = str(evaluation_dir / "evaluator.py")
    runpy.run_path(str(evaluation_dir / "evaluator.py"), run_name="__main__")


if __name__ == "__main__":
    main()
