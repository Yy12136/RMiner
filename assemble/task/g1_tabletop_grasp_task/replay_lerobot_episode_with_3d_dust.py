#!/usr/bin/env python3
"""Replay a LeRobot v3 arm/gripper episode through Isaac Sim DDS.

All output images come from simulator cameras.  Dust is selected by
``G1_DUST_MODE``; source MP4 files are never decoded or modified.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np
import pyarrow.parquet as pq

from benchmark_camera import CAMERA_KEYS, CameraReader
from unitree_sdk2py.core.channel import (
    ChannelFactoryInitialize,
    ChannelPublisher,
    ChannelSubscriber,
)
from unitree_sdk2py.idl.default import (
    unitree_go_msg_dds__MotorCmd_,
    unitree_hg_msg_dds__LowCmd_,
)
from unitree_sdk2py.idl.unitree_go.msg.dds_ import MotorCmds_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import MotorStates_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
from unitree_sdk2py.utils.crc import CRC


ARM_MOTOR_INDICES = tuple(range(15, 29))
REPLAY_CAMERA_KEYS = dict(CAMERA_KEYS)
if os.environ.get("G1_ENABLE_THIRD_PERSON", "0").strip().lower() in {"1", "true", "yes", "on"}:
    REPLAY_CAMERA_KEYS["observation.images.third_person"] = "third"
CAMERA_SHORT_NAMES = tuple(REPLAY_CAMERA_KEYS.values())


class VideoEncoder:
    def __init__(self, path: Path, width: int, height: int, fps: float) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.process = subprocess.Popen(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "bgr24",
                "-video_size",
                f"{width}x{height}",
                "-framerate",
                f"{fps:.8f}",
                "-i",
                "pipe:0",
                "-an",
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-crf",
                "19",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(path),
            ],
            stdin=subprocess.PIPE,
        )

    def write(self, frame: np.ndarray) -> None:
        if self.process.stdin is None or self.process.poll() is not None:
            raise RuntimeError(f"Encoder stopped early: {self.path}")
        self.process.stdin.write(np.ascontiguousarray(frame).tobytes())

    def finish(self) -> None:
        if self.process.stdin is not None:
            self.process.stdin.close()
        code = self.process.wait(timeout=120)
        if code != 0:
            raise RuntimeError(f"ffmpeg exited with status {code}: {self.path}")

    def abort(self) -> None:
        if self.process.stdin is not None and not self.process.stdin.closed:
            self.process.stdin.close()
        if self.process.poll() is None:
            self.process.terminate()
            self.process.wait(timeout=5)


class G1Dex1ReplayPublisher:
    def __init__(self, timeout: float) -> None:
        ChannelFactoryInitialize(1)
        self.lowcmd_publisher = ChannelPublisher("rt/lowcmd", LowCmd_)
        self.lowcmd_publisher.Init()
        self.lowstate_subscriber = ChannelSubscriber("rt/lowstate", LowState_)
        self.lowstate_subscriber.Init()
        deadline = time.monotonic() + timeout
        state = None
        while time.monotonic() < deadline:
            state = self.lowstate_subscriber.Read()
            if state is not None:
                break
            time.sleep(0.01)
        if state is None:
            raise TimeoutError("Timed out waiting for G1 rt/lowstate")

        self.command = unitree_hg_msg_dds__LowCmd_()
        self.command.mode_pr = 0
        self.command.mode_machine = state.mode_machine
        for index, motor in enumerate(self.command.motor_cmd):
            motor.mode = 1
            motor.dq = 0.0
            motor.tau = 0.0
            if index < len(state.motor_state):
                motor.q = float(state.motor_state[index].q)
            motor.kp = 40.0 if index in ARM_MOTOR_INDICES else 100.0
            motor.kd = 1.5 if index in ARM_MOTOR_INDICES else 3.0
        self.crc = CRC()

        self.left_publisher = ChannelPublisher(
            "rt/dex1/left/cmd", MotorCmds_
        )
        self.right_publisher = ChannelPublisher(
            "rt/dex1/right/cmd", MotorCmds_
        )
        self.left_publisher.Init()
        self.right_publisher.Init()
        self.left_state_subscriber = ChannelSubscriber(
            "rt/dex1/left/state", MotorStates_
        )
        self.right_state_subscriber = ChannelSubscriber(
            "rt/dex1/right/state", MotorStates_
        )
        self.left_state_subscriber.Init()
        self.right_state_subscriber.Init()
        self.left_command = self._make_gripper_command()
        self.right_command = self._make_gripper_command()

    @staticmethod
    def _make_gripper_command() -> MotorCmds_:
        message = MotorCmds_()
        motor = unitree_go_msg_dds__MotorCmd_()
        motor.dq = 0.0
        motor.tau = 0.0
        motor.kp = 5.0
        motor.kd = 0.05
        message.cmds = [motor]
        return message

    def publish(self, values: np.ndarray) -> None:
        if values.shape != (16,) or not np.all(np.isfinite(values)):
            raise ValueError("Replay action must be a finite 16-vector")
        for motor_index, target in zip(ARM_MOTOR_INDICES, values[:14]):
            self.command.motor_cmd[motor_index].q = float(target)
        self.command.crc = self.crc.Crc(self.command)
        self.lowcmd_publisher.Write(self.command)

        self.left_command.cmds[0].q = float(values[14])
        self.right_command.cmds[0].q = float(values[15])
        self.left_publisher.Write(self.left_command)
        self.right_publisher.Write(self.right_command)

    def read_state(self, fallback: np.ndarray) -> np.ndarray:
        """Read actual arm/gripper feedback, falling back only for a stale tick."""

        result = np.asarray(fallback, dtype=np.float32).reshape(16).copy()
        lowstate = self.lowstate_subscriber.Read()
        if lowstate is not None and len(lowstate.motor_state) > ARM_MOTOR_INDICES[-1]:
            result[:14] = [
                float(lowstate.motor_state[index].q) for index in ARM_MOTOR_INDICES
            ]
        left = self.left_state_subscriber.Read()
        right = self.right_state_subscriber.Read()
        if left is not None and left.states:
            result[14] = float(left.states[0].q)
        if right is not None and right.states:
            result[15] = float(right.states[0].q)
        return result


def load_episode(dataset: Path, episode_index: int):
    episode_row = None
    for path in sorted((dataset / "meta/episodes").glob("chunk-*/*.parquet")):
        table = pq.read_table(path)
        for row in table.to_pylist():
            if int(row["episode_index"]) == episode_index:
                episode_row = row
                break
        if episode_row is not None:
            break
    if episode_row is None:
        raise IndexError(f"Episode {episode_index} does not exist in {dataset}")
    data_path = dataset / (
        f"data/chunk-{int(episode_row['data/chunk_index']):03d}/"
        f"file-{int(episode_row['data/file_index']):03d}.parquet"
    )
    table = pq.read_table(data_path)
    actions = np.asarray(table["action"].to_pylist(), dtype=np.float32)
    states = np.asarray(table["observation.state"].to_pylist(), dtype=np.float32)
    if (
        actions.ndim != 2
        or actions.shape[1] < 16
        or states.ndim != 2
        or states.shape[0] != actions.shape[0]
        or states.shape[1] < 16
    ):
        raise ValueError(f"Unexpected episode arrays: state={states.shape}, action={actions.shape}")
    return states, actions, data_path


def wait_for_cameras(reader: CameraReader, timeout: float):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        frames = reader.read_all()
        if frames is not None and all(frame.age_seconds < 2.0 for frame in frames.values()):
            return frames
        time.sleep(0.02)
    raise TimeoutError("Timed out waiting for all three simulator cameras")


def wait_for_next_cameras(
    reader: CameraReader,
    previous_timestamps: dict[str, int],
    timeout: float = 2.0,
    max_skew_ms: int = 100,
):
    """Wait for a strictly newer, mutually synchronized camera triplet."""

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        frames = reader.read_all()
        if frames is None:
            time.sleep(0.002)
            continue
        timestamps = {key: frame.timestamp_ms for key, frame in frames.items()}
        if not all(
            timestamps[key] > previous_timestamps.get(key, -1)
            for key in timestamps
        ):
            time.sleep(0.002)
            continue
        if max(timestamps.values()) - min(timestamps.values()) > max_skew_ms:
            time.sleep(0.002)
            continue
        if all(frame.age_seconds < 2.0 for frame in frames.values()):
            return frames, timestamps
        time.sleep(0.002)
    raise TimeoutError("Timed out waiting for a new synchronized camera triplet")


def write_aux_action(path: Path | None, sequence: int, action: np.ndarray) -> None:
    if path is None:
        return
    values = (
        [float(value) for value in action[16:19]]
        if action.shape[0] >= 19
        else [0.0, 0.0, 0.0]
    )
    payload = {"sequence": int(sequence), "values": values, "timestamp_ns": time.time_ns()}
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload), encoding="utf-8")
    os.replace(temporary, path)


def make_grid(inputs: list[Path], output: Path) -> None:
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(inputs[0]),
            "-i",
            str(inputs[1]),
            "-i",
            str(inputs[2]),
            "-filter_complex",
            "[0:v]scale=320:240[h];[1:v]scale=320:240[l];"
            "[2:v]scale=320:240[r];[h][l][r]hstack=inputs=3[grid]",
            "-map",
            "[grid]",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "20",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(output),
        ],
        check=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optionally replay only the first N frames of the episode.",
    )
    parser.add_argument(
        "--snapshot-frame",
        type=int,
        default=None,
        help="Request a native USD snapshot after this zero-based replay frame.",
    )
    parser.add_argument("--dds-timeout", type=float, default=60.0)
    parser.add_argument("--camera-timeout", type=float, default=120.0)
    args = parser.parse_args()

    dataset = args.dataset.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    states, actions, data_path = load_episode(dataset, args.episode)
    source_episode_frames = len(actions)
    if args.max_frames is not None:
        if args.max_frames <= 0:
            raise ValueError("--max-frames must be positive when provided")
        states = states[: args.max_frames]
        actions = actions[: args.max_frames]
    if len(actions) == 0:
        raise RuntimeError("The requested replay contains no frames")
    snapshot_request_value = os.environ.get("G1_STATIC_SNAPSHOT_REQUEST_FILE", "").strip()
    snapshot_status_value = os.environ.get("G1_STATIC_SNAPSHOT_STATUS_FILE", "").strip()
    snapshot_request_path = Path(snapshot_request_value) if snapshot_request_value else None
    snapshot_status_path = Path(snapshot_status_value) if snapshot_status_value else None
    snapshot_frame = args.snapshot_frame
    if snapshot_frame is None and snapshot_request_path is not None:
        snapshot_frame = int(os.environ.get("G1_STATIC_SNAPSHOT_FRAME", "900"))
    if snapshot_frame is not None and snapshot_frame < 0:
        raise ValueError("--snapshot-frame must be non-negative")
    if snapshot_request_path is not None and snapshot_status_path is None:
        raise ValueError("G1_STATIC_SNAPSHOT_STATUS_FILE is required with snapshot requests")
    publisher = G1Dex1ReplayPublisher(args.dds_timeout)
    reader = CameraReader(include_third=("third" in CAMERA_SHORT_NAMES))
    encoders: dict[str, VideoEncoder] = {}
    marker_value = os.environ.get("G1_DUST_RECORDING_MARKER")
    recording_marker = Path(marker_value) if marker_value else None
    if recording_marker is not None:
        recording_marker.unlink(missing_ok=True)
    replay_wall_seconds = 0.0
    actual_states: list[np.ndarray] = []
    camera_timestamp_rows: list[dict[str, int]] = []
    aux_action_value = os.environ.get("G1_REPLAY_AUX_ACTION_FILE", "").strip()
    aux_action_path = Path(aux_action_value) if aux_action_value else None
    output_paths = {
        name: output_dir / f"observation.images.{name}.mp4"
        for name in CAMERA_SHORT_NAMES
    }
    try:
        initial_frames = wait_for_cameras(reader, args.camera_timeout)
        previous_timestamps = {
            key: frame.timestamp_ms for key, frame in initial_frames.items()
        }
        # Establish the recorded initial arm/gripper state before replaying
        # commands; this is a simulator action, not use of the source video.
        # The first 16 state values are the arm/gripper state consumed by DDS;
        # newer datasets may append object-pose telemetry after those values.
        publisher.publish(states[0][:16])
        time.sleep(1.0)
        for feature_key, short_name in REPLAY_CAMERA_KEYS.items():
            frame = initial_frames[feature_key].bgr
            height, width = frame.shape[:2]
            encoders[short_name] = VideoEncoder(
                output_paths[short_name], width, height, args.fps
            )

        period = 1.0 / args.fps
        if recording_marker is not None:
            recording_marker.touch()
        start = time.monotonic()
        for frame_index, action in enumerate(actions):
            publisher.publish(action[:16])
            write_aux_action(aux_action_path, frame_index, action)
            target = start + (frame_index + 1) * period
            while time.monotonic() < target:
                time.sleep(min(0.002, max(0.0, target - time.monotonic())))
            latest, previous_timestamps = wait_for_next_cameras(
                reader, previous_timestamps
            )
            for feature_key, short_name in REPLAY_CAMERA_KEYS.items():
                encoders[short_name].write(latest[feature_key].bgr)
            actual = states[frame_index].copy()
            actual[:16] = publisher.read_state(states[frame_index][:16])
            actual_states.append(actual)
            camera_timestamp_rows.append(
                {key: int(value) for key, value in previous_timestamps.items()}
            )
            if snapshot_frame is not None and frame_index == snapshot_frame:
                if snapshot_request_path is None:
                    raise ValueError("--snapshot-frame requires G1_STATIC_SNAPSHOT_REQUEST_FILE")
                snapshot_request_path.parent.mkdir(parents=True, exist_ok=True)
                token = f"episode-{args.episode}-frame-{frame_index}-{time.time_ns()}"
                output_value = os.environ.get("G1_STATIC_SNAPSHOT_OUTPUT", "").strip()
                if not output_value:
                    raise ValueError("G1_STATIC_SNAPSHOT_OUTPUT is required for a snapshot")
                payload = {
                    "token": token,
                    "output": output_value,
                    "episode": int(args.episode),
                    "frame": int(frame_index),
                }
                temporary = snapshot_request_path.with_name(
                    f".{snapshot_request_path.name}.{os.getpid()}.tmp"
                )
                temporary.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
                os.replace(temporary, snapshot_request_path)
                deadline = time.monotonic() + float(
                    os.environ.get("G1_STATIC_SNAPSHOT_TIMEOUT", "180")
                )
                while time.monotonic() < deadline:
                    if snapshot_status_path.is_file():
                        status = json.loads(snapshot_status_path.read_text(encoding="utf-8"))
                        if status.get("token") == token:
                            if not status.get("ok", False):
                                raise RuntimeError(f"Static snapshot failed: {status}")
                            print(
                                f"[replay] native static snapshot ready: {status.get('output')}",
                                flush=True,
                            )
                            break
                    time.sleep(0.05)
                else:
                    raise TimeoutError("Timed out waiting for Isaac static snapshot export")
                break
            if (frame_index + 1) % 100 == 0:
                print(
                    f"[3d-dust-replay] {frame_index + 1}/{len(actions)} frames",
                    flush=True,
                )
        replay_wall_seconds = time.monotonic() - start
        if recording_marker is not None:
            recording_marker.unlink(missing_ok=True)
        for encoder in encoders.values():
            encoder.finish()
        grid_path = output_dir / "three_camera_grid.mp4"
        make_grid([output_paths[name] for name in CAMERA_KEYS.values()], grid_path)
        actual_state_path = output_dir / "actual_replay_state.npz"
        np.savez_compressed(
            actual_state_path,
            observation_state=np.asarray(actual_states, dtype=np.float32),
        )
        camera_timestamp_path = output_dir / "camera_timestamps.jsonl"
        camera_timestamp_path.write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in camera_timestamp_rows),
            encoding="utf-8",
        )
        max_camera_skew_ms = max(
            (
                max(row.values()) - min(row.values())
                for row in camera_timestamp_rows
            ),
            default=0,
        )
        manifest = {
            "source_dataset": str(dataset),
            "source_data_parquet": str(data_path),
            "episode_index": args.episode,
            "frames": len(actions),
            "source_episode_frames": source_episode_frames,
            "max_frames": args.max_frames,
            "fps": args.fps,
            "rendering": "Isaac Sim native replay with deterministic lighting",
            "dust_mode": os.environ.get("G1_DUST_MODE", "legacy"),
            "dust_profile": os.environ.get("G1_DUST_PROFILE", "train"),
            "dust_preset": os.environ.get("G1_DUST_PRESET", "medium"),
            "dust_ambient_multiplier": float(
                os.environ.get("G1_DUST_AMBIENT_MULTIPLIER", "1.0")
            ),
            "dust_seed": int(os.environ.get("G1_DUST_SEED", "260822")),
            "object_reset_seed": (
                int(os.environ["G1_OBJECT_RESET_SEED"])
                if "G1_OBJECT_RESET_SEED" in os.environ
                else None
            ),
            "replay_wall_seconds": replay_wall_seconds,
            "effective_replay_fps": (
                len(actions) / replay_wall_seconds
                if replay_wall_seconds > 0.0
                else 0.0
            ),
            "source_videos_decoded": False,
            "source_dataset_modified": False,
            "robot_replay_fields": (
                "14 arm joints + 2 Dex1 grippers + 2 track and 1 lift target"
                if actions.shape[1] >= 19
                else "14 arm joints + 2 Dex1 grippers"
            ),
            "initial_state_metadata": os.environ.get("G1_REPLAY_INITIAL_STATE_FILE"),
            "initial_state_replay": bool(os.environ.get("G1_REPLAY_INITIAL_STATE_FILE")),
            "actual_state_file": str(actual_state_path),
            "camera_timestamp_file": str(camera_timestamp_path),
            "camera_timestamps_strictly_increasing": True,
            "max_camera_skew_ms": int(max_camera_skew_ms),
            "object_state_replay": bool(os.environ.get("G1_REPLAY_INITIAL_STATE_FILE")),
            "object_state_note": (
                "The source LeRobot v3 episode does not contain per-frame object, "
                "rope-link, tracked-base, or lift transforms. Their motion is newly "
                "simulated by PhysX and is not guaranteed to match the source episode."
            ),
            "videos": {name: str(path) for name, path in output_paths.items()},
            "comparison_grid": str(grid_path),
        }
        (output_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        print(f"[3d-dust-replay] Saved simulator replay to {output_dir}")
    except Exception:
        for encoder in encoders.values():
            encoder.abort()
        raise
    finally:
        if recording_marker is not None:
            recording_marker.unlink(missing_ok=True)
        reader.close()
        if aux_action_path is not None:
            aux_action_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
