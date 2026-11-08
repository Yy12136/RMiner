"""Small, dependency-light LeRobot v2.1 dataset writer.

This writer intentionally only depends on numpy, pyarrow and ffmpeg so it can
run in Unitree's ``tv`` conda environment without installing the full LeRobot
training stack.  An episode is only added to the dataset when ``save_episode``
is called.  ``discard_episode`` removes every staged video and sample.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Mapping, Sequence

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


CODEBASE_VERSION = "v2.1"
DEFAULT_CHUNK_SIZE = 1000


def _json_dump_atomic(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as file:
            json.dump(value, file, ensure_ascii=False, indent=2)
            file.write("\n")
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _jsonl_dump_atomic(path: Path, rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as file:
            for row in rows:
                file.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
                file.write("\n")
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def _feature_stats(values: np.ndarray) -> dict:
    values = np.asarray(values)
    if values.ndim == 1:
        values = values[:, None]
    return {
        "min": values.min(axis=0).astype(np.float32).tolist(),
        "max": values.max(axis=0).astype(np.float32).tolist(),
        "mean": values.mean(axis=0).astype(np.float32).tolist(),
        "std": values.std(axis=0).astype(np.float32).tolist(),
        "count": [int(values.shape[0])],
    }


class _VideoEncoder:
    def __init__(self, output_path: Path, width: int, height: int, fps: float):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_path = output_path
        self.log_path = output_path.with_suffix(".ffmpeg.log")
        self._log = self.log_path.open("wb")
        command = [
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
            str(fps),
            "-i",
            "pipe:0",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(output_path),
        ]
        self._process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=self._log,
        )

    def write(self, image_bgr: np.ndarray) -> None:
        if self._process.stdin is None or self._process.poll() is not None:
            raise RuntimeError(f"ffmpeg stopped while writing {self.output_path}")
        try:
            self._process.stdin.write(np.ascontiguousarray(image_bgr).tobytes())
        except BrokenPipeError as error:
            raise RuntimeError(f"ffmpeg pipe closed while writing {self.output_path}") from error

    def finish(self) -> None:
        if self._process.stdin is not None and not self._process.stdin.closed:
            self._process.stdin.close()
        return_code = self._process.wait(timeout=30)
        self._log.close()
        if return_code != 0:
            details = self.log_path.read_text(encoding="utf-8", errors="replace")
            raise RuntimeError(f"ffmpeg failed for {self.output_path}: {details.strip()}")
        self.log_path.unlink(missing_ok=True)

    def abort(self) -> None:
        try:
            if self._process.stdin is not None and not self._process.stdin.closed:
                self._process.stdin.close()
            if self._process.poll() is None:
                self._process.terminate()
                try:
                    self._process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    self._process.kill()
                    self._process.wait(timeout=3)
        finally:
            if not self._log.closed:
                self._log.close()


class LeRobotV21Writer:
    """Write synchronized low-dimensional data and RGB videos as LeRobot v2.1."""

    def __init__(
        self,
        root: str | Path,
        fps: float,
        instruction: str,
        state_names: Sequence[str],
        action_names: Sequence[str],
        camera_keys: Sequence[str],
        robot_type: str = "unitree_g1_29_dex1",
        chunk_size: int = DEFAULT_CHUNK_SIZE,
    ):
        self.root = Path(root).expanduser().resolve()
        self.meta_dir = self.root / "meta"
        self.staging_root = self.root / ".staging"
        self.fps = float(fps)
        self.instruction = instruction.strip()
        self.state_names = list(state_names)
        self.action_names = list(action_names)
        self.camera_keys = list(camera_keys)
        self.robot_type = robot_type
        self.chunk_size = int(chunk_size)
        if not self.instruction:
            raise ValueError("The task instruction must not be empty.")
        if not self.camera_keys:
            raise ValueError("At least one camera is required.")
        if len(set(self.camera_keys)) != len(self.camera_keys):
            raise ValueError("Camera keys must be unique.")

        self.meta_dir.mkdir(parents=True, exist_ok=True)
        self.staging_root.mkdir(parents=True, exist_ok=True)
        # Anything in staging is from an interrupted or deliberately discarded episode.
        for stale in self.staging_root.iterdir():
            if stale.is_dir():
                shutil.rmtree(stale)
            else:
                stale.unlink()

        self._episodes = _read_jsonl(self.meta_dir / "episodes.jsonl")
        self._episode_stats = _read_jsonl(self.meta_dir / "episodes_stats.jsonl")
        self._tasks = _read_jsonl(self.meta_dir / "tasks.jsonl")
        self._camera_shapes: dict[str, tuple[int, int, int]] = {}
        self._load_and_validate_existing_dataset()

        self._active = False
        self._staging_dir: Path | None = None
        self._encoders: dict[str, _VideoEncoder] = {}
        self._states: list[np.ndarray] = []
        self._actions: list[np.ndarray] = []

    @property
    def is_recording(self) -> bool:
        return self._active

    @property
    def total_episodes(self) -> int:
        return len(self._episodes)

    @property
    def total_frames(self) -> int:
        return sum(int(row["length"]) for row in self._episodes)

    def _load_and_validate_existing_dataset(self) -> None:
        info_path = self.meta_dir / "info.json"
        if info_path.exists():
            with info_path.open("r", encoding="utf-8") as file:
                info = json.load(file)
            if info.get("codebase_version") != CODEBASE_VERSION:
                raise ValueError(
                    f"Existing dataset is {info.get('codebase_version')}, expected {CODEBASE_VERSION}."
                )
            if abs(float(info.get("fps", self.fps)) - self.fps) > 1e-6:
                raise ValueError("Existing dataset FPS does not match --frequency.")
            features = info.get("features", {})
            if features.get("observation.state", {}).get("names") != self.state_names:
                raise ValueError("Existing dataset state schema does not match this task.")
            if features.get("action", {}).get("names") != self.action_names:
                raise ValueError("Existing dataset action schema does not match this task.")
            for key in self.camera_keys:
                feature = features.get(key)
                if feature is None:
                    raise ValueError(f"Existing dataset is missing camera feature {key}.")
                channels, height, width = feature["shape"]
                self._camera_shapes[key] = (int(height), int(width), int(channels))

        if self._tasks:
            matching = [row for row in self._tasks if row.get("task") == self.instruction]
            if not matching or int(matching[0].get("task_index", -1)) != 0:
                raise ValueError("Existing dataset uses a different task instruction.")
        else:
            self._tasks = [{"task_index": 0, "task": self.instruction}]
            _jsonl_dump_atomic(self.meta_dir / "tasks.jsonl", self._tasks)

        expected_indices = list(range(len(self._episodes)))
        actual_indices = [int(row["episode_index"]) for row in self._episodes]
        if actual_indices != expected_indices:
            raise ValueError("Existing dataset episode indices are not contiguous from zero.")

    def start_episode(self) -> None:
        if self._active:
            raise RuntimeError("An episode is already being recorded.")
        episode_index = len(self._episodes)
        self._staging_dir = Path(
            tempfile.mkdtemp(prefix=f"episode_{episode_index:06d}_", dir=self.staging_root)
        )
        self._encoders = {}
        self._states = []
        self._actions = []
        self._active = True

    def _start_encoders(self, images: Mapping[str, np.ndarray]) -> None:
        assert self._staging_dir is not None
        for key in self.camera_keys:
            image = np.asarray(images[key])
            if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
                raise ValueError(f"{key} must be a uint8 HxWx3 BGR image, got {image.shape}/{image.dtype}.")
            shape = tuple(int(value) for value in image.shape)
            previous_shape = self._camera_shapes.get(key)
            if previous_shape is not None and previous_shape != shape:
                raise ValueError(f"{key} shape changed from {previous_shape} to {shape}.")
            self._camera_shapes[key] = shape
            height, width, _ = shape
            self._encoders[key] = _VideoEncoder(
                self._staging_dir / f"{key}.mp4", width=width, height=height, fps=self.fps
            )

    def add_frame(
        self,
        state: Sequence[float] | np.ndarray,
        action: Sequence[float] | np.ndarray,
        images: Mapping[str, np.ndarray],
    ) -> None:
        if not self._active:
            return
        missing = [key for key in self.camera_keys if key not in images or images[key] is None]
        if missing:
            raise ValueError(f"Missing synchronized camera frames: {', '.join(missing)}")
        state_array = np.asarray(state, dtype=np.float32).reshape(-1)
        action_array = np.asarray(action, dtype=np.float32).reshape(-1)
        if state_array.size != len(self.state_names):
            raise ValueError(f"Expected {len(self.state_names)} state values, got {state_array.size}.")
        if action_array.size != len(self.action_names):
            raise ValueError(f"Expected {len(self.action_names)} action values, got {action_array.size}.")
        if not np.isfinite(state_array).all() or not np.isfinite(action_array).all():
            raise ValueError("State/action contains NaN or infinity.")

        if not self._encoders:
            self._start_encoders(images)
        for key, encoder in self._encoders.items():
            image = np.asarray(images[key])
            if tuple(image.shape) != self._camera_shapes[key] or image.dtype != np.uint8:
                raise ValueError(f"Camera frame format changed for {key}.")
            encoder.write(image)
        self._states.append(state_array.copy())
        self._actions.append(action_array.copy())

    def _close_encoders(self) -> None:
        errors = []
        for encoder in self._encoders.values():
            try:
                encoder.finish()
            except Exception as error:
                errors.append(str(error))
        if errors:
            raise RuntimeError("; ".join(errors))

    def _write_parquet(self, episode_index: int, destination: Path) -> None:
        frame_count = len(self._states)
        global_start = self.total_frames
        states = np.stack(self._states).astype(np.float32, copy=False)
        actions = np.stack(self._actions).astype(np.float32, copy=False)
        table = pa.table(
            {
                "observation.state": pa.array(
                    states.tolist(), type=pa.list_(pa.float32(), len(self.state_names))
                ),
                "action": pa.array(
                    actions.tolist(), type=pa.list_(pa.float32(), len(self.action_names))
                ),
                "timestamp": pa.array(np.arange(frame_count, dtype=np.float32) / self.fps),
                "frame_index": pa.array(np.arange(frame_count, dtype=np.int64)),
                "episode_index": pa.array(np.full(frame_count, episode_index, dtype=np.int64)),
                "index": pa.array(np.arange(global_start, global_start + frame_count, dtype=np.int64)),
                "task_index": pa.array(np.zeros(frame_count, dtype=np.int64)),
            }
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(".parquet.tmp")
        pq.write_table(table, temporary, compression="zstd")
        os.replace(temporary, destination)

    def _build_info(self) -> dict:
        total_episodes = len(self._episodes)
        total_frames = self.total_frames
        features = {
            "observation.state": {
                "dtype": "float32",
                "shape": [len(self.state_names)],
                "names": self.state_names,
            },
            "action": {
                "dtype": "float32",
                "shape": [len(self.action_names)],
                "names": self.action_names,
            },
        }
        for key, dtype in (
            ("timestamp", "float32"),
            ("frame_index", "int64"),
            ("episode_index", "int64"),
            ("index", "int64"),
            ("task_index", "int64"),
        ):
            features[key] = {"dtype": dtype, "shape": [1], "names": None}
        for key in self.camera_keys:
            height, width, channels = self._camera_shapes[key]
            features[key] = {
                "dtype": "video",
                "shape": [channels, height, width],
                "names": ["channels", "height", "width"],
                "info": {
                    "video.fps": self.fps,
                    "video.codec": "h264",
                    "video.pix_fmt": "yuv420p",
                    "video.is_depth_map": False,
                    "video.has_audio": False,
                },
            }
        return {
            "codebase_version": CODEBASE_VERSION,
            "robot_type": self.robot_type,
            "total_episodes": total_episodes,
            "total_frames": total_frames,
            "total_tasks": 1,
            "total_videos": total_episodes * len(self.camera_keys),
            "total_chunks": math.ceil(total_episodes / self.chunk_size) if total_episodes else 0,
            "chunks_size": self.chunk_size,
            "fps": self.fps,
            "splits": {"train": f"0:{total_episodes}"},
            "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
            "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
            "features": features,
        }

    def _aggregate_stats(self) -> dict:
        aggregated = {}
        if not self._episode_stats:
            return aggregated
        feature_keys = set.intersection(
            *(set(row["stats"].keys()) for row in self._episode_stats)
        )
        for key in sorted(feature_keys):
            entries = [row["stats"][key] for row in self._episode_stats]
            counts = np.asarray([entry["count"][0] for entry in entries], dtype=np.float64)
            total_count = float(counts.sum())
            means = np.asarray([entry["mean"] for entry in entries], dtype=np.float64)
            stds = np.asarray([entry["std"] for entry in entries], dtype=np.float64)
            mean = (means * counts[:, None]).sum(axis=0) / total_count
            second_moment = ((stds ** 2 + means ** 2) * counts[:, None]).sum(axis=0) / total_count
            variance = np.maximum(second_moment - mean ** 2, 0.0)
            aggregated[key] = {
                "min": np.min([entry["min"] for entry in entries], axis=0).astype(np.float32).tolist(),
                "max": np.max([entry["max"] for entry in entries], axis=0).astype(np.float32).tolist(),
                "mean": mean.astype(np.float32).tolist(),
                "std": np.sqrt(variance).astype(np.float32).tolist(),
                "count": [int(total_count)],
            }
        return aggregated

    def save_episode(self) -> tuple[int, int]:
        if not self._active:
            raise RuntimeError("No episode is being recorded.")
        if not self._states:
            self.discard_episode()
            raise RuntimeError("The episode contained no synchronized frames and was discarded.")
        assert self._staging_dir is not None
        episode_index = len(self._episodes)
        episode_chunk = episode_index // self.chunk_size
        frame_count = len(self._states)

        try:
            self._close_encoders()
            data_path = self.root / f"data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
            self._write_parquet(episode_index, data_path)
            for key in self.camera_keys:
                video_path = (
                    self.root
                    / f"videos/chunk-{episode_chunk:03d}/{key}/episode_{episode_index:06d}.mp4"
                )
                video_path.parent.mkdir(parents=True, exist_ok=True)
                os.replace(self._staging_dir / f"{key}.mp4", video_path)

            states = np.stack(self._states)
            actions = np.stack(self._actions)
            stats = {
                "observation.state": _feature_stats(states),
                "action": _feature_stats(actions),
                "timestamp": _feature_stats(np.arange(frame_count, dtype=np.float32) / self.fps),
            }
            self._episodes.append(
                {"episode_index": episode_index, "tasks": [self.instruction], "length": frame_count}
            )
            self._episode_stats.append({"episode_index": episode_index, "stats": stats})
            _jsonl_dump_atomic(self.meta_dir / "episodes.jsonl", self._episodes)
            _jsonl_dump_atomic(self.meta_dir / "episodes_stats.jsonl", self._episode_stats)
            _jsonl_dump_atomic(self.meta_dir / "tasks.jsonl", self._tasks)
            _json_dump_atomic(self.meta_dir / "info.json", self._build_info())
            _json_dump_atomic(self.meta_dir / "stats.json", self._aggregate_stats())
        except Exception:
            # Keep accepted metadata untouched when finalization fails.  Any already moved
            # files can be overwritten safely by retrying the same episode index.
            self._active = False
            self._cleanup_staging()
            self._states = []
            self._actions = []
            self._encoders = {}
            raise

        self._active = False
        self._cleanup_staging()
        self._states = []
        self._actions = []
        self._encoders = {}
        return episode_index, frame_count

    def _cleanup_staging(self) -> None:
        if self._staging_dir is not None:
            shutil.rmtree(self._staging_dir, ignore_errors=True)
        self._staging_dir = None

    def discard_episode(self) -> int:
        if not self._active:
            return 0
        frame_count = len(self._states)
        for encoder in self._encoders.values():
            encoder.abort()
        self._active = False
        self._cleanup_staging()
        self._states = []
        self._actions = []
        self._encoders = {}
        return frame_count

    def close(self) -> None:
        # An unfinished trajectory is never considered valid merely because the
        # process is exiting.
        self.discard_episode()
