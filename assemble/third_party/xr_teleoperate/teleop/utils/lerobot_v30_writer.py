"""Dependency-light, crash-recoverable LeRobot Dataset v3.0 writer.

The Unitree ``tv`` environment deliberately does not install the full LeRobot
training stack.  This module therefore writes the v3.0 on-disk contract using
only numpy, pyarrow and ffmpeg, all of which are already available there.

Each accepted episode is finalized as its own v3 data/video shard.  Payloads
are installed before metadata and a small transaction journal lets the next
process finish or roll back a commit interrupted by a hard shutdown.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Mapping, Sequence

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


CODEBASE_VERSION = "v3.0"
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_DATA_FILE_SIZE_IN_MB = 100
DEFAULT_VIDEO_FILE_SIZE_IN_MB = 200
DATA_PATH = "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
VIDEO_PATH = "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"
EPISODES_PATH = "meta/episodes/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
REPLAY_METADATA_PATH = (
    "meta/replay/chunk-{chunk_index:03d}/file-{file_index:03d}.json"
)


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


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
        _fsync_directory(path.parent)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _load_json(path: Path, default=None):
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _normalize_json_mapping(value: Mapping | None) -> dict | None:
    """Return an immutable JSON-safe copy and reject NaN/Infinity."""
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise TypeError("Replay metadata must be a mapping.")
    encoded = json.dumps(value, ensure_ascii=False, allow_nan=False)
    normalized = json.loads(encoded)
    if not isinstance(normalized, dict):
        raise TypeError("Replay metadata must encode to a JSON object.")
    return normalized


def _feature_stats(values: np.ndarray) -> dict:
    values = np.asarray(values)
    if values.ndim == 1:
        values = values[:, None]
    return {
        "min": values.min(axis=0).astype(np.float64).tolist(),
        "max": values.max(axis=0).astype(np.float64).tolist(),
        "mean": values.mean(axis=0).astype(np.float64).tolist(),
        "std": values.std(axis=0).astype(np.float64).tolist(),
        "count": [int(values.shape[0])],
    }


def _aggregate_stats(existing: dict | None, new: dict) -> dict:
    if not existing:
        return new
    result = {}
    for key in new:
        if key not in existing:
            result[key] = new[key]
            continue
        old_entry = existing[key]
        new_entry = new[key]
        old_count = float(old_entry["count"][0])
        new_count = float(new_entry["count"][0])
        total_count = old_count + new_count
        old_mean = np.asarray(old_entry["mean"], dtype=np.float64)
        new_mean = np.asarray(new_entry["mean"], dtype=np.float64)
        old_std = np.asarray(old_entry["std"], dtype=np.float64)
        new_std = np.asarray(new_entry["std"], dtype=np.float64)
        mean = (old_mean * old_count + new_mean * new_count) / total_count
        second_moment = (
            (old_std**2 + old_mean**2) * old_count
            + (new_std**2 + new_mean**2) * new_count
        ) / total_count
        result[key] = {
            "min": np.minimum(old_entry["min"], new_entry["min"]).tolist(),
            "max": np.maximum(old_entry["max"], new_entry["max"]).tolist(),
            "mean": mean.tolist(),
            "std": np.sqrt(np.maximum(second_moment - mean**2, 0.0)).tolist(),
            "count": [int(total_count)],
        }
    for key in existing:
        if key not in result:
            result[key] = existing[key]
    return result


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
        return_code = self._process.wait(timeout=60)
        if not self._log.closed:
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


class LeRobotV30Writer:
    """Write accepted trajectories directly as a LeRobot Dataset v3.0 dataset."""

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
        resume: bool = True,
    ):
        self.root = Path(root).expanduser().resolve()
        self.meta_dir = self.root / "meta"
        self.staging_root = self.root / ".staging"
        self.transaction_path = self.staging_root / "transaction.json"
        self.fps = float(fps)
        self.instruction = instruction.strip()
        self.state_names = list(state_names)
        self.action_names = list(action_names)
        self.camera_keys = list(camera_keys)
        self.robot_type = robot_type
        self.chunk_size = int(chunk_size)
        self.resume = bool(resume)
        if self.fps <= 0:
            raise ValueError("FPS must be positive.")
        if not self.instruction:
            raise ValueError("The task instruction must not be empty.")
        if not self.camera_keys or len(set(self.camera_keys)) != len(self.camera_keys):
            raise ValueError("Camera keys must be non-empty and unique.")

        # An existing dataset is never overwritten.  Resume mode validates it
        # and appends the next contiguous episode; disabling resume is a guard
        # for callers that intentionally require a brand-new output path.
        existing_info = self.meta_dir / "info.json"
        if existing_info.exists() and not self.resume:
            raise FileExistsError(
                f"Dataset already exists at {self.root}. "
                "Enable resume or choose an empty G1_LEROBOT_DATASET_DIR."
            )

        self.meta_dir.mkdir(parents=True, exist_ok=True)
        self.staging_root.mkdir(parents=True, exist_ok=True)
        self._recover_transaction()
        for stale in self.staging_root.iterdir():
            if stale == self.transaction_path:
                continue
            if stale.is_dir():
                shutil.rmtree(stale)
            else:
                stale.unlink()

        self._episodes: list[dict] = []
        self._info: dict | None = None
        self._global_stats: dict = {}
        self._episode_schema: pa.Schema | None = None
        self._camera_shapes: dict[str, tuple[int, int, int]] = {}
        self._replay_metadata_from_episode: int | None = None
        self._load_and_validate_existing_dataset()
        self._ensure_tasks_file()

        self._active = False
        self._staging_dir: Path | None = None
        self._encoders: dict[str, _VideoEncoder] = {}
        self._states: list[np.ndarray] = []
        self._actions: list[np.ndarray] = []
        self._replay_metadata: dict | None = None

    @property
    def is_recording(self) -> bool:
        return self._active

    @property
    def total_episodes(self) -> int:
        return len(self._episodes)

    @property
    def total_frames(self) -> int:
        return sum(int(row["length"]) for row in self._episodes)

    def _episode_files(self) -> list[Path]:
        return sorted((self.meta_dir / "episodes").glob("chunk-*/*.parquet"))

    def _load_and_validate_existing_dataset(self) -> None:
        self._info = _load_json(self.meta_dir / "info.json")
        self._global_stats = _load_json(self.meta_dir / "stats.json", {}) or {}
        episode_files = self._episode_files()
        self._episodes = []
        for path in episode_files:
            table = pq.read_table(path)
            if self._episode_schema is None:
                self._episode_schema = table.schema
            self._episodes.extend(table.to_pylist())
        self._episodes.sort(key=lambda row: int(row["episode_index"]))

        if self._info is None:
            if self._episodes:
                raise ValueError("Episode metadata exists but meta/info.json is missing.")
            return
        info = self._info
        if info.get("codebase_version") != CODEBASE_VERSION:
            raise ValueError(
                f"Existing dataset is {info.get('codebase_version')}, expected {CODEBASE_VERSION}."
            )
        if abs(float(info.get("fps", self.fps)) - self.fps) > 1e-6:
            raise ValueError("Existing dataset FPS does not match --frequency.")
        if int(info.get("chunks_size", self.chunk_size)) != self.chunk_size:
            raise ValueError("Existing dataset chunks_size does not match this writer.")
        features = info.get("features", {})
        state_feature = features.get("observation.state", {})
        action_feature = features.get("action", {})
        if state_feature.get("names") != self.state_names or state_feature.get("shape") != [len(self.state_names)]:
            raise ValueError("Existing dataset state schema does not match this task.")
        if action_feature.get("names") != self.action_names or action_feature.get("shape") != [len(self.action_names)]:
            raise ValueError("Existing dataset action schema does not match this task.")
        video_features = {key for key, value in features.items() if value.get("dtype") == "video"}
        if video_features != set(self.camera_keys):
            raise ValueError("Existing dataset camera keys do not match this task.")
        for key in self.camera_keys:
            feature = features[key]
            if feature.get("names") != ["channels", "height", "width"]:
                raise ValueError(f"Existing camera feature {key} is not channel-first RGB.")
            channels, height, width = feature["shape"]
            if int(channels) != 3:
                raise ValueError(f"Existing camera feature {key} is not RGB.")
            video_info = feature.get("info", {})
            if video_info.get("video.codec") != "h264" or video_info.get("video.pix_fmt") != "yuv420p":
                raise ValueError(f"Existing camera feature {key} is not H.264/yuv420p.")
            self._camera_shapes[key] = (int(height), int(width), int(channels))

        expected_indices = list(range(len(self._episodes)))
        actual_indices = [int(row["episode_index"]) for row in self._episodes]
        if actual_indices != expected_indices:
            raise ValueError("Existing dataset episode indices are not contiguous from zero.")
        expected_from = 0
        for row in self._episodes:
            length = int(row["length"])
            if int(row["dataset_from_index"]) != expected_from:
                raise ValueError("Existing dataset frame ranges are not contiguous.")
            if int(row["dataset_to_index"]) != expected_from + length:
                raise ValueError("Existing dataset frame range does not match episode length.")
            if row.get("tasks") != [self.instruction]:
                raise ValueError("Existing dataset uses a different task instruction.")
            self._validate_episode_payload(row)
            expected_from += length
        if int(info.get("total_episodes", -1)) != len(self._episodes):
            raise ValueError("meta/info.json total_episodes does not match episode metadata.")
        if int(info.get("total_frames", -1)) != expected_from:
            raise ValueError("meta/info.json total_frames does not match episode metadata.")
        # Replay state is deliberately kept outside the standard LeRobot
        # ``meta/info.json`` schema.  Discover the optional extension from the
        # episode sidecars so the official loader sees no unknown fields.  Old
        # datasets may start replay coverage after one or more legacy episodes,
        # but once coverage starts it must remain continuous.
        replay_presence = [
            self._replay_metadata_file(row).exists() for row in self._episodes
        ]
        if any(replay_presence):
            replay_from = replay_presence.index(True)
            if not all(replay_presence[replay_from:]):
                raise ValueError(
                    "Replay metadata coverage has a gap between episodes."
                )
            self._replay_metadata_from_episode = replay_from
            for row in self._episodes[replay_from:]:
                self._validate_replay_metadata_payload(row)
        else:
            self._replay_metadata_from_episode = None

    def _validate_episode_payload(self, row: dict) -> None:
        data_path = self.root / DATA_PATH.format(
            chunk_index=int(row["data/chunk_index"]), file_index=int(row["data/file_index"])
        )
        if not data_path.exists():
            raise ValueError(f"Missing episode data file: {data_path}")
        # A shared converted shard can contain several episodes, so it must have
        # at least the current episode's number of rows.
        if pq.read_metadata(data_path).num_rows < int(row["length"]):
            raise ValueError(f"Data file has too few rows: {data_path}")
        for key in self.camera_keys:
            video_path = self.root / VIDEO_PATH.format(
                video_key=key,
                chunk_index=int(row[f"videos/{key}/chunk_index"]),
                file_index=int(row[f"videos/{key}/file_index"]),
            )
            if not video_path.exists() or video_path.stat().st_size == 0:
                raise ValueError(f"Missing episode video file: {video_path}")

    def _replay_metadata_file(self, row: dict) -> Path:
        return self.root / REPLAY_METADATA_PATH.format(
            chunk_index=int(row["data/chunk_index"]),
            file_index=int(row["data/file_index"]),
        )

    def _validate_replay_metadata_payload(self, row: dict) -> dict:
        path = self._replay_metadata_file(row)
        metadata = _load_json(path)
        if not isinstance(metadata, dict):
            raise ValueError(f"Missing or invalid replay metadata: {path}")
        if int(metadata.get("episode_index", -1)) != int(row["episode_index"]):
            raise ValueError(f"Replay metadata episode index mismatch: {path}")
        if not isinstance(metadata.get("sim_state"), dict):
            raise ValueError(f"Replay metadata sim_state is missing: {path}")
        return metadata

    def load_replay_metadata(self, episode_index: int) -> dict:
        """Load the immutable reset snapshot for a previously saved episode."""
        episode_index = int(episode_index)
        if episode_index < 0 or episode_index >= len(self._episodes):
            raise IndexError(f"Episode index out of range: {episode_index}")
        if (
            self._replay_metadata_from_episode is None
            or episode_index < self._replay_metadata_from_episode
        ):
            raise ValueError(
                f"Episode {episode_index} predates replay-snapshot coverage."
            )
        return self._validate_replay_metadata_payload(
            self._episodes[episode_index]
        )

    def _ensure_tasks_file(self) -> None:
        path = self.meta_dir / "tasks.parquet"
        if path.exists():
            rows = pq.read_table(path).to_pylist()
            if rows != [{"task_index": 0, "task": self.instruction}]:
                raise ValueError("Existing meta/tasks.parquet uses a different instruction.")
            return
        table = pa.table(
            {
                "task_index": pa.array([0], type=pa.int64()),
                "task": pa.array([self.instruction], type=pa.string()),
            }
        )
        pandas_metadata = {
            "index_columns": ["task"],
            "column_indexes": [{
                "name": None,
                "field_name": None,
                "pandas_type": "unicode",
                "numpy_type": "object",
                "metadata": {"encoding": "UTF-8"},
            }],
            "columns": [
                {"name": "task_index", "field_name": "task_index", "pandas_type": "int64", "numpy_type": "int64", "metadata": None},
                {"name": "task", "field_name": "task", "pandas_type": "unicode", "numpy_type": "object", "metadata": None},
            ],
            "attributes": {},
            "creator": {"library": "pyarrow", "version": pa.__version__},
            "pandas_version": "2.0.0",
        }
        table = table.replace_schema_metadata({b"pandas": json.dumps(pandas_metadata).encode("utf-8")})
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".parquet.tmp")
        pq.write_table(table, temporary, compression="snappy")
        os.replace(temporary, path)
        _fsync_directory(path.parent)

    def _next_file_location(self) -> tuple[int, int]:
        if not self._episodes:
            return 0, 0
        largest = max(
            int(row["data/chunk_index"]) * self.chunk_size + int(row["data/file_index"])
            for row in self._episodes
        )
        linear_index = largest + 1
        return linear_index // self.chunk_size, linear_index % self.chunk_size

    def start_episode(self, replay_metadata: Mapping | None = None) -> None:
        if self._active:
            raise RuntimeError("An episode is already being recorded.")
        normalized_replay_metadata = _normalize_json_mapping(replay_metadata)
        if (
            self._replay_metadata_from_episode is not None
            and normalized_replay_metadata is None
        ):
            raise ValueError(
                "This dataset requires replay metadata for every appended episode."
            )
        if normalized_replay_metadata is not None:
            if not isinstance(normalized_replay_metadata.get("sim_state"), dict):
                raise ValueError("Replay metadata must contain a sim_state object.")
            if not normalized_replay_metadata.get("schema"):
                raise ValueError("Replay metadata must declare a schema.")
        episode_index = len(self._episodes)
        self._staging_dir = Path(
            tempfile.mkdtemp(prefix=f"episode_{episode_index:06d}_", dir=self.staging_root)
        )
        self._encoders = {}
        self._states = []
        self._actions = []
        self._replay_metadata = normalized_replay_metadata
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

    def _probe_video(self, path: Path, expected_frames: int, expected_shape: tuple[int, int, int]) -> None:
        command = [
            "ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
            "-show_entries", "stream=codec_name,pix_fmt,width,height,nb_read_frames",
            "-of", "json", str(path),
        ]
        result = subprocess.run(command, check=True, capture_output=True, text=True, timeout=60)
        streams = json.loads(result.stdout).get("streams", [])
        if len(streams) != 1:
            raise RuntimeError(f"Expected one video stream in {path}.")
        stream = streams[0]
        height, width, _ = expected_shape
        if stream.get("codec_name") != "h264" or stream.get("pix_fmt") != "yuv420p":
            raise RuntimeError(f"Unexpected video codec/pixel format in {path}.")
        if int(stream.get("width", -1)) != width or int(stream.get("height", -1)) != height:
            raise RuntimeError(f"Unexpected video dimensions in {path}.")
        if int(stream.get("nb_read_frames", -1)) != expected_frames:
            raise RuntimeError(f"Video frame count does not match trajectory in {path}.")

    def _write_data_parquet(self, destination: Path, episode_index: int) -> None:
        frame_count = len(self._states)
        states = np.stack(self._states).astype(np.float32, copy=False)
        actions = np.stack(self._actions).astype(np.float32, copy=False)
        table = pa.table(
            {
                "observation.state": pa.array(states.tolist(), type=pa.list_(pa.float32())),
                "action": pa.array(actions.tolist(), type=pa.list_(pa.float32())),
                "timestamp": pa.array(np.arange(frame_count, dtype=np.float32) / self.fps),
                "frame_index": pa.array(np.arange(frame_count, dtype=np.int64)),
                "episode_index": pa.array(np.full(frame_count, episode_index, dtype=np.int64)),
                "index": pa.array(np.arange(self.total_frames, self.total_frames + frame_count, dtype=np.int64)),
                "task_index": pa.array(np.zeros(frame_count, dtype=np.int64)),
            }
        )
        pq.write_table(table, destination, compression="zstd")
        if pq.read_metadata(destination).num_rows != frame_count:
            raise RuntimeError("Staged data Parquet has an unexpected row count.")

    def _build_info(self, total_episodes: int, total_frames: int) -> dict:
        if self._info is not None:
            info = json.loads(json.dumps(self._info))
            # Clean up the short-lived pre-release representation of replay
            # discovery fields.  The sidecar scan above is authoritative.
            info.pop("replay_metadata_path", None)
            info.pop("replay_metadata_from_episode", None)
            info["total_episodes"] = total_episodes
            info["total_frames"] = total_frames
            info["total_tasks"] = 1
            info["splits"] = {"train": f"0:{total_episodes}"}
            return info
        features = {
            "observation.state": {
                "dtype": "float32", "shape": [len(self.state_names)], "names": self.state_names, "fps": self.fps,
            },
            "action": {
                "dtype": "float32", "shape": [len(self.action_names)], "names": self.action_names, "fps": self.fps,
            },
        }
        for key, dtype in (
            ("timestamp", "float32"), ("frame_index", "int64"), ("episode_index", "int64"),
            ("index", "int64"), ("task_index", "int64"),
        ):
            features[key] = {"dtype": dtype, "shape": [1], "names": None, "fps": self.fps}
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
        info = {
            "codebase_version": CODEBASE_VERSION,
            "fps": self.fps,
            "features": features,
            "total_episodes": total_episodes,
            "total_frames": total_frames,
            "total_tasks": 1,
            "chunks_size": self.chunk_size,
            "data_files_size_in_mb": DEFAULT_DATA_FILE_SIZE_IN_MB,
            "video_files_size_in_mb": DEFAULT_VIDEO_FILE_SIZE_IN_MB,
            "data_path": DATA_PATH,
            "video_path": VIDEO_PATH,
            "robot_type": self.robot_type,
            "splits": {"train": f"0:{total_episodes}"},
        }
        return info

    def _episode_row(
        self,
        episode_index: int,
        frame_count: int,
        chunk_index: int,
        file_index: int,
        stats: dict,
    ) -> dict:
        row = {
            "episode_index": episode_index,
            "data/chunk_index": chunk_index,
            "data/file_index": file_index,
            "dataset_from_index": self.total_frames,
            "dataset_to_index": self.total_frames + frame_count,
        }
        duration = frame_count / self.fps
        for key in self.camera_keys:
            row[f"videos/{key}/chunk_index"] = chunk_index
            row[f"videos/{key}/file_index"] = file_index
            row[f"videos/{key}/from_timestamp"] = 0.0
            row[f"videos/{key}/to_timestamp"] = duration
        row["tasks"] = [self.instruction]
        row["length"] = frame_count
        for feature_key, feature_stats in stats.items():
            for statistic, value in feature_stats.items():
                row[f"stats/{feature_key}/{statistic}"] = value
        row["meta/episodes/chunk_index"] = chunk_index
        row["meta/episodes/file_index"] = file_index
        return row

    def _write_episode_parquet(self, destination: Path, row: dict) -> pa.Schema:
        if self._episode_schema is not None:
            table = pa.Table.from_pylist([row], schema=self._episode_schema)
        else:
            table = pa.Table.from_pylist([row])
        pq.write_table(table, destination, compression="snappy", use_dictionary=True)
        if pq.read_metadata(destination).num_rows != 1:
            raise RuntimeError("Staged episode metadata must contain exactly one row.")
        return table.schema

    def _recover_transaction(self) -> bool:
        journal = _load_json(self.transaction_path)
        if journal is None:
            return False
        targets = [self.root / relative for relative in journal["payloads"]]
        complete = all(path.exists() and path.stat().st_size > 0 for path in targets)
        if complete:
            _json_dump_atomic(self.meta_dir / "stats.json", journal["new_stats"])
            _json_dump_atomic(self.meta_dir / "info.json", journal["new_info"])
        else:
            for path in targets:
                path.unlink(missing_ok=True)
            if journal.get("previous_stats_exists"):
                _json_dump_atomic(self.meta_dir / "stats.json", journal["previous_stats"])
            else:
                (self.meta_dir / "stats.json").unlink(missing_ok=True)
            if journal.get("previous_info_exists"):
                _json_dump_atomic(self.meta_dir / "info.json", journal["previous_info"])
            else:
                (self.meta_dir / "info.json").unlink(missing_ok=True)
        self.transaction_path.unlink(missing_ok=True)
        _fsync_directory(self.staging_root)
        return complete

    def save_episode(self) -> tuple[int, int]:
        if not self._active:
            raise RuntimeError("No episode is being recorded.")
        if not self._states:
            self.discard_episode()
            raise RuntimeError("The episode contained no synchronized frames and was discarded.")
        assert self._staging_dir is not None
        episode_index = len(self._episodes)
        frame_count = len(self._states)
        chunk_index, file_index = self._next_file_location()
        data_relative = DATA_PATH.format(chunk_index=chunk_index, file_index=file_index)
        episode_relative = EPISODES_PATH.format(chunk_index=chunk_index, file_index=file_index)
        video_relatives = {
            key: VIDEO_PATH.format(video_key=key, chunk_index=chunk_index, file_index=file_index)
            for key in self.camera_keys
        }
        replay_relative = None
        if self._replay_metadata is not None:
            replay_relative = REPLAY_METADATA_PATH.format(
                chunk_index=chunk_index, file_index=file_index
            )
        payload_relatives = [
            data_relative,
            *video_relatives.values(),
            episode_relative,
            *([replay_relative] if replay_relative is not None else []),
        ]
        for relative in payload_relatives:
            if (self.root / relative).exists():
                self._clear_active(abort_encoders=True)
                raise RuntimeError(f"Refusing to overwrite existing v3 shard: {self.root / relative}")

        previous_info = self._info
        previous_stats = self._global_stats
        staged_data = self._staging_dir / "data.parquet"
        staged_episode = self._staging_dir / "episode.parquet"
        staged_replay = self._staging_dir / "replay.json"
        try:
            self._close_encoders()
            for key in self.camera_keys:
                self._probe_video(
                    self._staging_dir / f"{key}.mp4", frame_count, self._camera_shapes[key]
                )
            self._write_data_parquet(staged_data, episode_index)
            states = np.stack(self._states)
            actions = np.stack(self._actions)
            timestamps = np.arange(frame_count, dtype=np.float32) / self.fps
            episode_stats = {
                "observation.state": _feature_stats(states),
                "action": _feature_stats(actions),
                "timestamp": _feature_stats(timestamps),
            }
            row = self._episode_row(
                episode_index, frame_count, chunk_index, file_index, episode_stats
            )
            staged_schema = self._write_episode_parquet(staged_episode, row)
            if self._replay_metadata is not None:
                replay_payload = json.loads(json.dumps(self._replay_metadata))
                replay_payload["episode_index"] = episode_index
                replay_payload["frame_count"] = frame_count
                replay_payload["dataset_from_index"] = self.total_frames
                replay_payload["dataset_to_index"] = self.total_frames + frame_count
                _json_dump_atomic(staged_replay, replay_payload)
                if _load_json(staged_replay) != replay_payload:
                    raise RuntimeError("Staged replay metadata verification failed.")
            new_stats = _aggregate_stats(previous_stats, episode_stats)
            new_info = self._build_info(episode_index + 1, self.total_frames + frame_count)
            journal = {
                "episode_index": episode_index,
                "payloads": payload_relatives,
                "previous_info_exists": previous_info is not None,
                "previous_info": previous_info,
                "previous_stats_exists": bool(previous_stats),
                "previous_stats": previous_stats,
                "new_info": new_info,
                "new_stats": new_stats,
            }
            _json_dump_atomic(self.transaction_path, journal)

            staged_sources = {
                data_relative: staged_data,
                episode_relative: staged_episode,
                **{relative: self._staging_dir / f"{key}.mp4" for key, relative in video_relatives.items()},
            }
            if replay_relative is not None:
                staged_sources[replay_relative] = staged_replay
            for relative in payload_relatives:
                target = self.root / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                os.replace(staged_sources[relative], target)
                _fsync_directory(target.parent)
            _json_dump_atomic(self.meta_dir / "stats.json", new_stats)
            _json_dump_atomic(self.meta_dir / "info.json", new_info)
            self.transaction_path.unlink(missing_ok=True)
            _fsync_directory(self.staging_root)

            self._episodes.append(row)
            self._episode_schema = staged_schema
            self._global_stats = new_stats
            self._info = new_info
            if (
                replay_relative is not None
                and self._replay_metadata_from_episode is None
            ):
                self._replay_metadata_from_episode = episode_index
        except Exception as error:
            committed = self._recover_transaction()
            if committed:
                self._episode_schema = None
                self._load_and_validate_existing_dataset()
            self._clear_active(abort_encoders=True)
            if committed:
                return episode_index, frame_count
            raise error

        self._clear_active(abort_encoders=False)
        return episode_index, frame_count

    def _clear_active(self, abort_encoders: bool) -> None:
        if abort_encoders:
            for encoder in self._encoders.values():
                encoder.abort()
        self._active = False
        if self._staging_dir is not None:
            shutil.rmtree(self._staging_dir, ignore_errors=True)
        self._staging_dir = None
        self._encoders = {}
        self._states = []
        self._actions = []
        self._replay_metadata = None

    def discard_episode(self) -> int:
        if not self._active:
            return 0
        frame_count = len(self._states)
        self._clear_active(abort_encoders=True)
        return frame_count

    def close(self) -> None:
        # Exiting never implicitly accepts an unfinished demonstration.
        self.discard_episode()
