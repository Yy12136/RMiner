#!/usr/bin/env python3
"""Automatic, task-configured DP/PI0.5 evaluation loop."""

from __future__ import annotations

import argparse
import json
import os
import signal
import time
from datetime import datetime
from multiprocessing.connection import Client
from pathlib import Path

import cv2
import numpy as np

from camera import CAMERA_KEYS, CameraReader
from perturbation import (
    PERTURBATION_VARIANTS,
    TRAINING_AUGMENTATION_SEED,
    TRAINING_PARTICLE_COUNT,
    EpisodePerturber,
    choose_variant,
)
from policy_runtime import IllegalPolicyAction, load_policy
from safety import synchronized_displacement_error
from task_specs import get_spec
from utils import checkpoint_step, model_label, percentile, summarize, write_json_atomic


class WallClockVideoRecorder:
    """Write a sidecar video whose duration follows wall-clock episode time.

    LeRobotV30Writer intentionally writes one frame per evaluator step at a
    declared FPS.  That is appropriate for a normally paced controller, but
    a slow remote policy can produce only a handful of frames during a long
    timeout.  This recorder repeats the most recent camera frame to cover
    inference gaps, so the resulting MP4 is useful for diagnosing pauses and
    has the same duration as the episode.  The original LeRobot rollout is
    left untouched.
    """

    def __init__(self, root: Path, fps: float, camera_keys: tuple[str, ...], enabled: bool = True):
        self.root = root
        self.fps = float(fps)
        self.camera_keys = tuple(camera_keys)
        self.enabled = enabled
        self._writers: dict[str, cv2.VideoWriter] = {}
        self._written = 0
        self._episode_start: float | None = None
        self._episode_index: int | None = None

    def start_episode(self, episode_index: int, started: float) -> None:
        self.finish_episode()
        self._episode_index = int(episode_index)
        self._episode_start = float(started)
        self._written = 0

    def _ensure_writers(self, frames) -> None:
        if not self.enabled or self._writers:
            return
        if self._episode_index is None:
            return
        for key in self.camera_keys:
            frame = frames[key].bgr
            height, width = frame.shape[:2]
            path = (
                self.root
                / "wallclock_videos"
                / key
                / "chunk-000"
                / f"file-{self._episode_index:03d}.mp4"
            )
            path.parent.mkdir(parents=True, exist_ok=True)
            writer = cv2.VideoWriter(
                str(path), cv2.VideoWriter_fourcc(*"mp4v"), self.fps, (width, height)
            )
            if not writer.isOpened():
                raise RuntimeError(f"Could not open wall-clock video writer: {path}")
            self._writers[key] = writer

    def add(self, frames, now: float) -> None:
        if not self.enabled or self._episode_start is None:
            return
        self._ensure_writers(frames)
        if not self._writers:
            return
        target = max(self._written + 1, int(round(max(0.0, now - self._episode_start) * self.fps)) + 1)
        repeat = target - self._written
        for _ in range(repeat):
            for key in self.camera_keys:
                self._writers[key].write(frames[key].bgr)
        self._written = target

    def finish_episode(self) -> None:
        for writer in self._writers.values():
            writer.release()
        self._writers.clear()
        self._episode_start = None
        self._episode_index = None
        self._written = 0

    def close(self) -> None:
        self.finish_episode()


class BridgeClient:
    def __init__(self, path: Path, authkey: str, timeout: float = 600.0) -> None:
        deadline = time.monotonic() + timeout
        while True:
            try:
                self.connection = Client(str(path), family="AF_UNIX", authkey=authkey.encode()); break
            except (FileNotFoundError, ConnectionRefusedError):
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"DDS bridge socket was not ready: {path}")
                time.sleep(0.2)

    def request(self, command: str, **payload) -> dict:
        self.connection.send({"command": command, **payload})
        response = self.connection.recv()
        if not response.get("ok"):
            raise RuntimeError(response.get("error", "Unknown bridge error"))
        return response

    def close(self) -> None:
        try: self.request("shutdown")
        except Exception: pass
        try: self.connection.close()
        except Exception: pass


def validate_state(response: dict, dimension: int, stale: float) -> np.ndarray:
    age = float(response["dds_age"])
    if age > stale:
        raise ConnectionError(f"DDS/simulator state is stale ({age:.3f}s)")
    state = np.asarray(response["state"], dtype=np.float32)
    if state.shape != (dimension,) or not np.isfinite(state).all():
        raise ConnectionError(f"Invalid observation state {state.shape}")
    return state


def validate_frames(frames, stale: float):
    if frames is None or set(frames) != set(CAMERA_KEYS):
        raise ConnectionError("One or more camera streams are unavailable")
    age = max(frame.age_seconds for frame in frames.values())
    if age > stale:
        raise ConnectionError(f"Camera stream is stale ({age:.3f}s)")
    return frames


def make_observation(state, frames):
    result = {"observation.state": np.asarray(state, dtype=np.float32)}
    for key, frame in frames.items():
        result[key] = cv2.cvtColor(frame.bgr, cv2.COLOR_BGR2RGB)
    return result


def canonical_reset(payload: dict) -> dict:
    ignored = {"reset_id", "episode_index", "settled", "settled_at", "locked_left_arm_q"}
    # settle_* keys carry runtime settle-gate telemetry (velocity zeroings,
    # wall-clock still time) that legitimately varies between resets; they
    # are not seed-dependent reset parameters.
    return {
        key: value
        for key, value in payload.items()
        if key not in ignored and not key.startswith("settle_")
    }


def task_frame_invariants(task: str, state: np.ndarray, response: dict, previous: np.ndarray | None) -> None:
    if task == "task06":
        pose = np.asarray(response["telemetry"]["object_root_pose_wxyz"], dtype=np.float32)
        if np.max(np.abs(state[-7:] - pose)) > 1e-6:
            raise ConnectionError("Task06 object pose differs from rt/sim_state payload")
    if task == "task04" and previous is not None:
        base_delta = np.linalg.norm(state[16:18] - previous[16:18])
        cart_delta = np.linalg.norm(state[23:25] - previous[23:25])
        error = synchronized_displacement_error(previous, state)
        if max(base_delta, cart_delta) > 1e-6 and error > 1e-3:
            raise ConnectionError(
                f"Task04 synchronized displacement mismatch: base={base_delta:.6f}, cart={cart_delta:.6f}"
            )


def task04_control_preflight(bridge: BridgeClient, seed: int) -> dict:
    """Exercise the real stop/forward DDS path before policy control."""
    bridge.request("reset", episode_index=-2, seed=seed)
    start_response = bridge.request("get_state")
    start = validate_state(start_response, 30, 1.0)
    for _ in range(45):
        action = np.concatenate((start[:16], [1.2, 1.2, 0.0]))
        response = bridge.request("action", action=action.tolist())
        if response["action"][-3:] != [1.2, 1.2, 0.0]:
            raise RuntimeError("Task04 legal forward action was not preserved")
        time.sleep(1 / 30)
    moving = validate_state(bridge.request("get_state"), 30, 1.0)
    base_distance = float(np.linalg.norm(moving[16:18] - start[16:18]))
    cart_distance = float(np.linalg.norm(moving[23:25] - start[23:25]))
    sync_error = abs(base_distance - cart_distance)
    # Task04's grasp-safety protocol intentionally blocks mobile motion until
    # both handles are confirmed grasped. The legacy preflight sends a
    # forward command before grasp acquisition, so an exactly zero,
    # synchronized displacement is the expected safety response rather than a
    # broken DDS path. Continue to the policy evaluation in that case.
    if base_distance < 1e-6 and cart_distance < 1e-6 and sync_error <= 1e-6:
        return {
            "base_distance_m": base_distance, "cart_distance_m": cart_distance,
            "synchronization_error_m": sync_error,
            "stop_base_drift_m": 0.0, "stop_cart_drift_m": 0.0,
            "allowed_actions": [[0, 0, 0], [1.2, 1.2, 0]],
            "forward_blocked_before_grasp": True,
        }
    if base_distance < 0.05 or sync_error > 1e-3:
        raise RuntimeError(
            f"Task04 controlled forward check failed: base={base_distance:.6f}, "
            f"cart={cart_distance:.6f}, error={sync_error:.6f}"
        )
    stop_action = np.concatenate((moving[:16], [0.0, 0.0, 0.0]))
    for _ in range(30):
        response = bridge.request("action", action=stop_action.tolist())
        if response["action"][-3:] != [0.0, 0.0, 0.0]:
            raise RuntimeError("Task04 stop action was not preserved")
        time.sleep(1 / 30)
    stopped = validate_state(bridge.request("get_state"), 30, 1.0)
    for _ in range(15):
        bridge.request("hold"); time.sleep(1 / 30)
    stopped_later = validate_state(bridge.request("get_state"), 30, 1.0)
    stop_base_drift = float(np.linalg.norm(stopped_later[16:18] - stopped[16:18]))
    stop_cart_drift = float(np.linalg.norm(stopped_later[23:25] - stopped[23:25]))
    if max(stop_base_drift, stop_cart_drift) > 1e-3:
        raise RuntimeError(
            f"Task04 stop drift exceeds 1 mm: base={stop_base_drift:.6f}, cart={stop_cart_drift:.6f}"
        )
    return {
        "base_distance_m": base_distance, "cart_distance_m": cart_distance,
        "synchronization_error_m": sync_error, "stop_base_drift_m": stop_base_drift,
        "stop_cart_drift_m": stop_cart_drift, "allowed_actions": [[0, 0, 0], [1.2, 1.2, 0]],
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True); parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--socket", type=Path, required=True); parser.add_argument("--authkey", required=True)
    parser.add_argument("--output-root", type=Path, required=True); parser.add_argument("--success-marker-file", type=Path, required=True)
    parser.add_argument("--failure-marker-file", type=Path, required=True)
    parser.add_argument("--ready-file", type=Path, required=True); parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--timeout", type=float, default=60.0); parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--frequency", type=float, default=30.0)
    parser.add_argument("--inference", choices=("sync", "async"), default="sync")
    parser.add_argument(
        "--no-wallclock-video",
        action="store_true",
        help="Disable real-time sidecar videos; keep the standard rollout videos.",
    )
    parser.add_argument("--stale-timeout", type=float, default=1.0)
    parser.add_argument("--skip-reset-preflight", action="store_true")
    parser.add_argument(
        "--perturb",
        default="none",
        choices=PERTURBATION_VARIANTS + ("random",),
        help="Training-matched visual perturbation: a fixed variant, a "
        "per-episode random draw (perturbation or none), or off (default).",
    )
    parser.add_argument("--perturb-seed", type=int, default=0)
    parser.add_argument("--perturb-particle-count", type=int, default=TRAINING_PARTICLE_COUNT)
    parser.add_argument(
        "--timeout-steps",
        type=int,
        default=0,
        help="End an episode as timeout after this many executed steps "
        "instead of the wall-clock budget; 0 keeps the wall-clock behaviour.",
    )
    return parser.parse_args()


def main() -> None:
    args, stopping = parse_args(), False
    spec = get_spec(args.task); checkpoint = args.checkpoint.expanduser().resolve()
    if not (checkpoint / "config.json").is_file():
        raise FileNotFoundError(f"Invalid pretrained_model directory: {checkpoint}")
    if min(args.episodes, args.timeout, args.frequency) <= 0:
        raise ValueError("episodes, timeout and frequency must be positive")
    def stop(_signal, _frame):
        nonlocal stopping; stopping = True
    signal.signal(signal.SIGINT, stop); signal.signal(signal.SIGTERM, stop)

    config, runner, compat = load_policy(checkpoint, spec)
    step, label = checkpoint_step(checkpoint), model_label(checkpoint)
    run_name = f"{config.type}_{step}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir = args.output_root / label / step / run_name
    output_dir.mkdir(parents=True, exist_ok=False)
    args.ready_file.parent.mkdir(parents=True, exist_ok=True); args.ready_file.write_text(str(output_dir) + "\n")
    write_json_atomic(output_dir / "config.json", {
        "task": spec.key, "checkpoint": str(checkpoint), "policy": str(config.type),
        "instruction": spec.instruction, "source_training_dataset": str(spec.dataset),
        "state_dim": spec.state_dim, "action_dim": spec.action_dim,
        "seed_range": [args.seed, args.seed + args.episodes - 1], "episodes": args.episodes,
        "timeout_seconds": args.timeout,
        "timeout_steps": args.timeout_steps,
        "timeout_basis": (
            f"steps:{args.timeout_steps} (matched to clean baseline measured cap)"
            if args.timeout_steps > 0
            else "wall_clock"
        ),
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
        ), "frequency_hz": args.frequency,
        "inference": args.inference, "initial_grippers_rad": list(spec.initial_grippers),
        "wallclock_video": {
            "enabled": not args.no_wallclock_video,
            "fps": args.frequency,
            "path": "rollouts/wallclock_videos",
            "method": "repeat_last_observation_during_inference_gaps",
        },
        "right_arm_only": spec.right_arm_only,
        "sim_physics_device": os.environ.get("G1_EVAL_SIM_DEVICE", "unspecified"),
        "camera_render_device": "GPU/Vulkan",
        "relative_chunk_anchor": "whole chunk postprocessed exactly once",
        "valid_terminal_statuses": ["success", "failure", "timeout"],
        "task04_grasp_protocol": "continuous_grasp_then_0.5s_post_push_hold" if spec.key == "task04" else None,
        "task05_rotation_protocol": "settled_rotation_v3" if spec.key == "task05" else None,
        "task05_rotation_thresholds": ({
            "required_rotation_deg": 60.0,
            "reset_settle_seconds": 1.0,
            "settle_still_threshold_rad_s": 0.05,
            "settle_still_frames": 15,
            "settle_handshake_budget_seconds": 20.0,
        } if spec.key == "task05" else None),
        "created_at": datetime.now().astimezone().isoformat(),
    })
    print(f"[evaluator] POLICY_READY task={spec.key} policy={config.type} output={output_dir}", flush=True)

    xr_root = Path(os.environ.get("XR_TELEOP_ROOT")
                    or (Path(__file__).resolve().parents[2] / "third_party" / "xr_teleoperate"))
    import sys
    sys.path.insert(0, str(xr_root))
    from teleop.utils.lerobot_v30_writer import LeRobotV30Writer
    writer = LeRobotV30Writer(root=output_dir / "rollouts", fps=args.frequency,
                             instruction=spec.instruction, state_names=list(spec.state_names),
                             action_names=list(spec.action_names), camera_keys=list(CAMERA_KEYS), resume=False)
    wallclock_video = WallClockVideoRecorder(
        output_dir / "rollouts", args.frequency, tuple(CAMERA_KEYS), enabled=not args.no_wallclock_video
    )
    bridge, camera = None, CameraReader()
    records, latencies = [], []
    episodes_path = output_dir / "episodes.jsonl"
    valid_count = 0; infra_for_slot = 0
    try:
        bridge = BridgeClient(args.socket, args.authkey)
        if not args.skip_reset_preflight:
            first = bridge.request("reset", episode_index=-1, seed=args.seed)["reset"]
            second = bridge.request("reset", episode_index=-1, seed=args.seed)["reset"]
            different = bridge.request("reset", episode_index=-1, seed=args.seed + 1)["reset"]
            first_key, second_key, different_key = map(canonical_reset, (first, second, different))
            if first_key != second_key:
                raise RuntimeError("Reset reproducibility failed: identical seeds produced different parameters")
            if first_key == different_key:
                raise RuntimeError("Reset diversity failed: adjacent seeds produced identical parameters")
            write_json_atomic(output_dir / "reset_reproducibility.json", {
                "same_seed": args.seed, "same_seed_equal": True,
                "different_seed": args.seed + 1, "different_seed_differs": True,
                "parameters": {"same_seed": first_key, "different_seed": different_key},
            })
            print(f"[evaluator] RESET_PREFLIGHT_OK seed={args.seed}/{args.seed + 1}", flush=True)
        if spec.key == "task04":
            mobile_check = task04_control_preflight(bridge, args.seed)
            write_json_atomic(output_dir / "task04_mobile_preflight.json", mobile_check)
            print(
                f"[evaluator] TASK04_MOBILE_PREFLIGHT_OK sync_error="
                f"{mobile_check['synchronization_error_m']:.6f}m",
                flush=True,
            )
        while valid_count < args.episodes and not stopping:
            episode_index, seed = valid_count, args.seed + valid_count
            perturber = None
            perturbation_name = "none"
            if args.perturb != "none":
                perturbation_name = (
                    choose_variant(seed, salt=args.perturb_seed)
                    if args.perturb == "random"
                    else args.perturb
                )
                if perturbation_name != "none":
                    perturber = EpisodePerturber(
                        perturbation_name,
                        seed,
                        particle_count=args.perturb_particle_count,
                    )
            reset, reset_error = None, None
            try:
                reset = bridge.request("reset", episode_index=episode_index, seed=seed)["reset"]
            except Exception as error:
                reset_error = error
            if reset_error is not None:
                try: bridge.request("hold")
                except Exception: pass
                status, valid, elapsed, steps, rollout, frames_saved = "infrastructure_error", False, 0.0, 0, None, 0
                success_payload, failure_payload, episode_latencies, projection_count = {}, {}, [], 0
                error_message = f"{type(reset_error).__name__}: {reset_error}"
            else:
                args.success_marker_file.unlink(missing_ok=True)
                args.failure_marker_file.unlink(missing_ok=True)
                runner.reset(); writer.start_episode()
                started = time.monotonic(); steps = 0; status = "timeout"; error_message = None
                episode_latencies, success_payload, failure_payload, projection_count = [], {}, {}, 0
                previous_state = None
                frames = None
                wallclock_video.start_episode(episode_index, started)
                try:
                    step_budget = args.timeout_steps if args.timeout_steps > 0 else None
                    wall_deadline = started + (
                        4.0 * args.timeout if step_budget is not None else args.timeout
                    )
                    while (
                        not stopping
                        and (step_budget is None or steps < step_budget)
                        and time.monotonic() < wall_deadline
                    ):
                        tick = time.monotonic()
                        response = bridge.request("get_state")
                        state = validate_state(response, spec.state_dim, args.stale_timeout)
                        task_frame_invariants(spec.key, state, response, previous_state)
                        previous_state = state.copy()
                        need_fresh_frames = (
                            frames is None
                            or not hasattr(runner, "needs_observation")
                            or bool(runner.needs_observation())
                        )
                        if need_fresh_frames:
                            frames = validate_frames(camera.read_all(), args.stale_timeout)
                            if perturber is not None:
                                # Apply exactly once per freshly read frame so
                                # reused observations are never double-perturbed.
                                frames = perturber.apply(frames)
                        action, latency = runner.action(make_observation(state, frames))
                        response = bridge.request("action", action=action.tolist())
                        if float(response["dds_age"]) > args.stale_timeout:
                            raise ConnectionError("State became stale after action")
                        executed = np.asarray(response["action"], dtype=np.float32)
                        if executed.shape != (spec.action_dim,):
                            raise ConnectionError(f"Bridge executed invalid action {executed.shape}")
                        projection_count = int(response.get("mobile_projection_count", 0))
                        writer.add_frame(state=state, action=executed,
                                         images={key: value.bgr for key, value in frames.items()})
                        wallclock_video.add(frames, time.monotonic())
                        episode_latencies.append(latency); latencies.append(latency); steps += 1
                        if args.success_marker_file.is_file():
                            bridge.request("hold")
                            success_payload = json.loads(args.success_marker_file.read_text())
                            status = "success"; break
                        if args.failure_marker_file.is_file():
                            bridge.request("hold")
                            failure_payload = json.loads(args.failure_marker_file.read_text())
                            status = "failure"
                            error_message = failure_payload.get("failure_reason", "task_failure")
                            break
                        remainder = 1 / args.frequency - (time.monotonic() - tick)
                        if remainder > 0: time.sleep(remainder)
                    if stopping: status = "manual_stop"
                except IllegalPolicyAction as error:
                    status, error_message, stopping = "policy_error", f"{type(error).__name__}: {error}", True
                except Exception as error:
                    status, error_message = "infrastructure_error", f"{type(error).__name__}: {error}"
                try: bridge.request("hold")
                except Exception: pass
                elapsed = time.monotonic() - started
                if frames is not None:
                    wallclock_video.add(frames, started + elapsed)
                wallclock_video.finish_episode()
                valid = status in ("success", "failure", "timeout")
                if valid:
                    rollout, frames_saved = writer.save_episode(); valid_count += 1; infra_for_slot = 0
                else:
                    rollout, frames_saved = None, writer.discard_episode()
            if status == "infrastructure_error": infra_for_slot += 1
            p50, p95 = percentile(episode_latencies, 50), percentile(episode_latencies, 95)
            record = {
                "attempt_index": len(records), "episode_index": episode_index, "seed": seed,
                "perturbation": perturbation_name,
                "status": status, "valid": valid, "elapsed_seconds": elapsed, "steps": steps,
                "rollout_episode_index": rollout, "rollout_frames": frames_saved,
                "reset": reset, "success": success_payload, "failure": failure_payload,
                "mobile_projection_count": projection_count,
                "inference_latency_ms_p50": None if p50 is None else 1000 * p50,
                "inference_latency_ms_p95": None if p95 is None else 1000 * p95,
                "error": error_message,
            }
            if hasattr(runner, "starved_steps"):
                record["async_chunk_metrics"] = {
                    "starved_steps": int(getattr(runner, "starved_steps", 0)),
                    "chunks_completed": int(getattr(runner, "chunk_index", 0)),
                    "total_actions": int(getattr(runner, "total_actions", 0)),
                }
            records.append(record)
            with episodes_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n"); stream.flush(); os.fsync(stream.fileno())
            write_json_atomic(output_dir / "summary.json", summarize(records, latencies))
            print(f"[evaluator] episode={episode_index} seed={seed} perturbation={perturbation_name} status={status} steps={steps}", flush=True)
            if infra_for_slot >= 2:
                print("[evaluator] same valid slot failed infrastructure twice; stopping", flush=True); break
    finally:
        if bridge is not None: bridge.close()
        wallclock_video.close()
        camera.close(); writer.close(); runner.close(); compat.close()
        args.ready_file.unlink(missing_ok=True)
        args.failure_marker_file.unlink(missing_ok=True)
        write_json_atomic(output_dir / "summary.json", summarize(records, latencies))
        print(f"[evaluator] FINISHED output={output_dir}", flush=True)


if __name__ == "__main__":
    main()
