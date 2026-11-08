#!/usr/bin/env python3
"""Build a balanced 5x LeRobot v3 dataset with video-only augmentation.

For N source episodes (N must be even), the output contains N originals plus
eight N/2 augmentation blocks.  Every lighting and dust variant is generated
from the successful source video; robot/object physics are never replayed.
State and action are copied exactly, the source dataset is never modified, and
completed v3 episodes are transactional so a rerun resumes at the next output
episode.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime
import json
import math
import os
from pathlib import Path
import sys
import tempfile

import cv2
import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq


TASK_ROOT = Path(__file__).resolve().parent
ASSEMBLE_ROOT = TASK_ROOT.parents[1]
XR_ROOT = ASSEMBLE_ROOT / "third_party/xr_teleoperate"
if str(XR_ROOT) not in sys.path:
    sys.path.insert(0, str(XR_ROOT))
if str(TASK_ROOT) not in sys.path:
    sys.path.insert(0, str(TASK_ROOT))

from teleop.utils.lerobot_v30_writer import LeRobotV30Writer


PROFILE_BLOCKS = (
    ("low_illumination", False, "first"),
    ("low_illumination", True, "second"),
    ("warm_worklamp", False, "second"),
    ("warm_worklamp", True, "first"),
    ("cool_led", False, "first"),
    ("cool_led", True, "second"),
    ("headlamp_spot", False, "second"),
    ("headlamp_spot", True, "first"),
)

VIDEO_TRANSFORM_PROFILES = frozenset(
    {"low_illumination", "warm_worklamp", "cool_led", "headlamp_spot"}
)
PIPELINE_VERSION = "video_only_v3_dense_gray_dust"


@dataclass(frozen=True)
class PlanEntry:
    output_episode: int
    source_episode: int
    kind: str
    lighting: str
    dust: bool
    block_index: int | None


@dataclass(frozen=True)
class DatasetSpec:
    root: Path
    fps: float
    instruction: str
    state_names: tuple[str, ...]
    action_names: tuple[str, ...]
    camera_keys: tuple[str, ...]
    robot_type: str
    rows: dict[int, dict]


def write_json_atomic(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as file:
            json.dump(payload, file, ensure_ascii=False, indent=2)
            file.write("\n")
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary_name, path)
    except Exception:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def load_dataset_spec(root: Path) -> DatasetSpec:
    root = root.expanduser().resolve()
    info_path = root / "meta/info.json"
    if not info_path.is_file():
        raise FileNotFoundError(f"LeRobot v3 info file not found: {info_path}")
    info = json.loads(info_path.read_text(encoding="utf-8"))
    if info.get("codebase_version") != "v3.0":
        raise ValueError(f"Expected LeRobot v3.0, got {info.get('codebase_version')}")
    task_rows = pq.read_table(root / "meta/tasks.parquet").to_pylist()
    if len(task_rows) != 1:
        raise ValueError("The generic augmenter currently requires exactly one task instruction")
    features = info["features"]
    camera_keys = tuple(
        key for key, value in features.items() if value.get("dtype") == "video"
    )
    if not camera_keys:
        raise ValueError("Input dataset has no video features")
    rows: dict[int, dict] = {}
    for path in sorted((root / "meta/episodes").glob("chunk-*/*.parquet")):
        for row in pq.read_table(path).to_pylist():
            index = int(row["episode_index"])
            if index in rows:
                raise ValueError(f"Duplicate source episode metadata: {index}")
            rows[index] = row
    return DatasetSpec(
        root=root,
        fps=float(info["fps"]),
        instruction=str(task_rows[0]["task"]),
        state_names=tuple(features["observation.state"]["names"]),
        action_names=tuple(features["action"]["names"]),
        camera_keys=camera_keys,
        robot_type=str(info.get("robot_type", "unknown")),
        rows=rows,
    )


def build_plan(source_start: int, source_count: int) -> list[PlanEntry]:
    if source_start < 0 or source_count <= 0 or source_count % 2:
        raise ValueError("--source-start must be non-negative and --source-count must be positive/even")
    sources = list(range(source_start, source_start + source_count))
    half = source_count // 2
    first, second = sources[:half], sources[half:]
    entries = [
        PlanEntry(i, source, "original", "original", False, None)
        for i, source in enumerate(sources)
    ]
    output_index = source_count
    for block_index, (lighting, dust, selection) in enumerate(PROFILE_BLOCKS):
        selected = first if selection == "first" else second
        kind = "video_transform"
        for source in selected:
            entries.append(
                PlanEntry(
                    output_episode=output_index,
                    source_episode=source,
                    kind=kind,
                    lighting=lighting,
                    dust=dust,
                    block_index=block_index,
                )
            )
            output_index += 1
    return entries


def create_writer(spec: DatasetSpec, output: Path) -> LeRobotV30Writer:
    return LeRobotV30Writer(
        root=output,
        fps=spec.fps,
        instruction=spec.instruction,
        state_names=spec.state_names,
        action_names=spec.action_names,
        camera_keys=spec.camera_keys,
        robot_type=spec.robot_type,
        resume=True,
    )


def source_data(spec: DatasetSpec, episode_index: int):
    row = spec.rows.get(episode_index)
    if row is None:
        raise IndexError(f"Source episode {episode_index} does not exist")
    path = spec.root / (
        f"data/chunk-{int(row['data/chunk_index']):03d}/"
        f"file-{int(row['data/file_index']):03d}.parquet"
    )
    table = pq.read_table(path)
    if "episode_index" in table.column_names:
        table = table.filter(pc.equal(table["episode_index"], episode_index))
    expected = int(row["length"])
    if table.num_rows != expected:
        raise ValueError(
            f"Episode {episode_index}: expected {expected} data rows, got {table.num_rows}"
        )
    states = np.asarray(table["observation.state"].to_pylist(), dtype=np.float32)
    actions = np.asarray(table["action"].to_pylist(), dtype=np.float32)
    return row, states, actions


class SourceVideoReader:
    def __init__(self, spec: DatasetSpec, row: dict):
        self.captures: dict[str, cv2.VideoCapture] = {}
        for key in spec.camera_keys:
            path = spec.root / (
                f"videos/{key}/chunk-{int(row[f'videos/{key}/chunk_index']):03d}/"
                f"file-{int(row[f'videos/{key}/file_index']):03d}.mp4"
            )
            capture = cv2.VideoCapture(str(path))
            if not capture.isOpened():
                self.close()
                raise RuntimeError(f"Unable to open source video: {path}")
            start = int(round(float(row[f"videos/{key}/from_timestamp"]) * spec.fps))
            capture.set(cv2.CAP_PROP_POS_FRAMES, start)
            self.captures[key] = capture

    def read(self) -> dict[str, np.ndarray]:
        result = {}
        for key, capture in self.captures.items():
            ok, frame = capture.read()
            if not ok or frame is None:
                raise RuntimeError(f"Source video ended early for {key}")
            result[key] = frame
        return result

    def close(self) -> None:
        for capture in self.captures.values():
            capture.release()
        self.captures.clear()


def manifest_payload(
    spec: DatasetSpec,
    output: Path,
    plan: list[PlanEntry],
    completed: int,
    args: argparse.Namespace,
) -> dict:
    return {
        "schema_version": 2,
        "pipeline_version": PIPELINE_VERSION,
        "input_dataset": str(spec.root),
        "output_dataset": str(output),
        "source_start": args.source_start,
        "source_count": args.source_count,
        "target_episodes": len(plan),
        "completed_episodes": completed,
        "particle_count": args.particle_count,
        "seed": args.seed,
        "updated_at": datetime.now().astimezone().isoformat(),
        "entries": [
            {**asdict(entry), "completed": entry.output_episode < completed}
            for entry in plan
        ],
    }


def validate_existing_manifest(
    spec: DatasetSpec,
    output: Path,
    plan: list[PlanEntry],
    args: argparse.Namespace,
) -> None:
    path = output / "meta/augmentation_manifest.json"
    if not path.is_file():
        return
    manifest = json.loads(path.read_text(encoding="utf-8"))
    completed = int(manifest.get("completed_episodes", 0))
    info_path = output / "meta/info.json"
    if info_path.is_file():
        completed = max(
            completed,
            int(json.loads(info_path.read_text(encoding="utf-8"))["total_episodes"]),
        )
    stored_entries = manifest.get("entries", [])
    if completed > args.source_count:
        for expected_entry in plan[:completed]:
            if expected_entry.output_episode >= len(stored_entries):
                raise ValueError(
                    "Existing augmentation manifest has incomplete episode provenance"
                )
            stored = stored_entries[expected_entry.output_episode]
            expected = asdict(expected_entry)
            mismatched_fields = {
                key: (stored.get(key), value)
                for key, value in expected.items()
                if stored.get(key) != value
            }
            if mismatched_fields:
                raise ValueError(
                    "Existing output contains episodes made by a different "
                    "augmentation pipeline; choose a new --output directory: "
                    + json.dumps(mismatched_fields, ensure_ascii=False)
                )
    expected = {
        "input_dataset": str(spec.root),
        "output_dataset": str(output),
        "source_start": args.source_start,
        "source_count": args.source_count,
        "target_episodes": len(plan),
        "particle_count": args.particle_count,
    }
    if completed > args.source_count:
        expected["pipeline_version"] = PIPELINE_VERSION
        expected["seed"] = args.seed
    mismatches = {
        key: (manifest.get(key), value)
        for key, value in expected.items()
        if manifest.get(key) != value
    }
    if mismatches:
        raise ValueError(
            "The existing output belongs to a different augmentation plan: "
            + json.dumps(mismatches, ensure_ascii=False)
        )


def update_manifest(
    spec: DatasetSpec,
    output: Path,
    plan: list[PlanEntry],
    completed: int,
    args: argparse.Namespace,
) -> None:
    write_json_atomic(
        output / "meta/augmentation_manifest.json",
        manifest_payload(spec, output, plan, completed, args),
    )


def copy_originals(
    spec: DatasetSpec,
    output: Path,
    plan: list[PlanEntry],
    args: argparse.Namespace,
) -> None:
    writer = create_writer(spec, output)
    if writer.total_episodes > args.source_count:
        writer.close()
        return
    try:
        while writer.total_episodes < args.source_count:
            entry = plan[writer.total_episodes]
            row, states, actions = source_data(spec, entry.source_episode)
            reader = SourceVideoReader(spec, row)
            writer.start_episode()
            try:
                for frame_index in range(len(actions)):
                    writer.add_frame(states[frame_index], actions[frame_index], reader.read())
                output_index, frames = writer.save_episode()
            except Exception:
                writer.discard_episode()
                raise
            finally:
                reader.close()
            if output_index != entry.output_episode or frames != len(actions):
                raise RuntimeError("Original episode commit returned inconsistent indices")
            update_manifest(spec, output, plan, writer.total_episodes, args)
            print(
                f"[augmentation] copied original {entry.source_episode} -> "
                f"{output_index} ({frames} frames)",
                flush=True,
            )
    finally:
        writer.close()


def _lighting_luts(profile: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return deterministic B/G/R lookup tables for clean video augmentation."""
    x = np.arange(256, dtype=np.float32) / 255.0
    if profile == "low_illumination":
        # Stable under-exposure: darker mid-tones without power-flicker noise.
        curves = (0.52 * x**1.12,) * 3
    elif profile == "warm_worklamp":
        base = x**0.97
        curves = (0.78 * base, 0.98 * base, 1.18 * base)
    elif profile == "cool_led":
        base = x**0.98
        curves = (1.16 * base, 1.03 * base, 0.84 * base)
    elif profile == "headlamp_spot":
        curves = (x, x, x)
    else:
        raise ValueError(f"No video transform is defined for {profile}")
    return tuple(
        np.clip(np.rint(curve * 255.0), 0, 255).astype(np.uint8)
        for curve in curves
    )


def transform_lighting_bgr(
    image_bgr: np.ndarray,
    luts: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> np.ndarray:
    if image_bgr.dtype != np.uint8 or image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise ValueError("Lighting transform expects an HxWx3 uint8 BGR image")
    channels = cv2.split(image_bgr)
    transformed = [cv2.LUT(channel, lut) for channel, lut in zip(channels, luts)]
    return np.ascontiguousarray(cv2.merge(transformed))


def _radial_map(
    width: int,
    height: int,
    center_x: float,
    center_y: float,
    sigma_x: float,
    sigma_y: float,
) -> np.ndarray:
    y, x = np.mgrid[0:height, 0:width]
    dx = (x - center_x * width) / (sigma_x * width)
    dy = (y - center_y * height) / (sigma_y * height)
    return np.exp(-0.5 * (dx * dx + dy * dy)).astype(np.float32)


class VideoDustField:
    """Temporally coherent gray mine dust without bubble-shaped sprites."""

    def __init__(
        self,
        width: int,
        height: int,
        seed: int,
        particle_count: int,
        illumination: np.ndarray | None,
    ) -> None:
        self.width = width
        self.height = height
        self.illumination = illumination
        rng = np.random.default_rng(seed)

        # Low-frequency suspended haze is generated at low resolution and
        # slowly advected. This avoids per-frame noise and preserves temporal
        # continuity through the full episode.
        cloud_h = max(24, height // 8)
        cloud_w = max(32, width // 8)
        cloud = rng.normal(0.0, 1.0, (cloud_h, cloud_w)).astype(np.float32)
        cloud = cv2.GaussianBlur(cloud, (0, 0), 4.5)
        lower, upper = np.percentile(cloud, (2.0, 98.0))
        self.cloud = np.clip(
            (cloud - lower) / max(float(upper - lower), 1.0e-6), 0.0, 1.0
        )

        # The approved simulator used 5000 particles. Here they are projected
        # into image space as sub-pixel points/short streaks with depth-based
        # parallax. Vectorized rasterization keeps 5000 particles affordable.
        self.count = int(particle_count)
        self.x = rng.uniform(0.0, width, self.count).astype(np.float32)
        self.y = rng.uniform(0.0, height, self.count).astype(np.float32)
        self.depth = rng.uniform(0.0, 1.0, self.count).astype(np.float32)
        self.vx = (
            0.08 + 0.72 * self.depth + rng.normal(0.0, 0.06, self.count)
        ).astype(np.float32)
        self.vy = (
            -0.06 - 0.42 * self.depth + rng.normal(0.0, 0.05, self.count)
        ).astype(np.float32)
        self.alpha = rng.uniform(0.24, 0.60, self.count).astype(np.float32)
        self.brightness = rng.uniform(118.0, 210.0, self.count).astype(np.float32)
        self.size_class = rng.random(self.count).astype(np.float32)
        self.phase = rng.uniform(0.0, 2.0 * math.pi, self.count).astype(np.float32)

    def apply(self, frame: np.ndarray, frame_index: int) -> np.ndarray:
        shift_y = int(round(-0.018 * frame_index))
        shift_x = int(round(0.032 * frame_index))
        cloud = np.roll(self.cloud, shift=(shift_y, shift_x), axis=(0, 1))
        cloud = cv2.resize(
            cloud,
            (self.width, self.height),
            interpolation=cv2.INTER_CUBIC,
        )
        breathing = 0.94 + 0.06 * math.sin(0.009 * frame_index)
        haze = np.clip((0.045 + 0.135 * cloud) * breathing, 0.0, 0.19)

        output = frame.astype(np.float32)
        neutral_gray = np.asarray([108.0, 108.0, 108.0], dtype=np.float32)
        output = output * (1.0 - haze[..., None]) + neutral_gray * haze[..., None]
        local_mean = cv2.GaussianBlur(output, (0, 0), 7.0)
        output = local_mean + (output - local_mean) * (1.0 - 0.42 * haze[..., None])

        self.x = (self.x + self.vx) % self.width
        self.y = (self.y + self.vy) % self.height
        xi = np.clip(self.x.astype(np.int32), 0, self.width - 1)
        yi = np.clip(self.y.astype(np.int32), 0, self.height - 1)
        shimmer = 0.88 + 0.12 * np.sin(self.phase + 0.021 * frame_index)
        weights = self.alpha * shimmer

        particle_mask = np.zeros((self.height, self.width), dtype=np.float32)
        particle_layer = np.zeros((self.height, self.width), dtype=np.float32)
        np.add.at(particle_mask, (yi, xi), weights)
        np.add.at(particle_layer, (yi, xi), weights * self.brightness)
        # A second displaced sample turns nearby motes into short motion
        # streaks instead of circles/bubbles.
        xi2 = np.clip(xi - np.rint(1.8 * self.vx).astype(np.int32), 0, self.width - 1)
        yi2 = np.clip(yi - np.rint(1.8 * self.vy).astype(np.int32), 0, self.height - 1)
        np.add.at(particle_mask, (yi2, xi2), 0.56 * weights)
        np.add.at(particle_layer, (yi2, xi2), 0.56 * weights * self.brightness)

        # The nearest 20% of particles cover two or three pixels, matching the
        # projected width distribution of the approved 3D USD points.
        near = self.size_class >= 0.80
        very_near = self.size_class >= 0.97
        for dx, dy, scale in ((1, 0, 0.48), (0, 1, 0.42)):
            xn = np.clip(xi[near] + dx, 0, self.width - 1)
            yn = np.clip(yi[near] + dy, 0, self.height - 1)
            wn = scale * weights[near]
            np.add.at(particle_mask, (yn, xn), wn)
            np.add.at(particle_layer, (yn, xn), wn * self.brightness[near])
        for dx, dy, scale in ((-1, 0, 0.34), (0, -1, 0.30)):
            xn = np.clip(xi[very_near] + dx, 0, self.width - 1)
            yn = np.clip(yi[very_near] + dy, 0, self.height - 1)
            wn = scale * weights[very_near]
            np.add.at(particle_mask, (yn, xn), wn)
            np.add.at(particle_layer, (yn, xn), wn * self.brightness[very_near])

        particle_mask = cv2.GaussianBlur(particle_mask, (3, 3), 0.22)
        particle_layer = cv2.GaussianBlur(particle_layer, (3, 3), 0.22)

        if self.illumination is not None:
            particle_mask *= 0.72 + 0.95 * self.illumination
            particle_layer *= 0.72 + 0.95 * self.illumination
        particle_mask = np.clip(particle_mask, 0.0, 0.72)
        particle_gray = np.divide(
            particle_layer,
            np.maximum(particle_mask, 1.0e-6),
            out=np.full_like(particle_layer, 150.0),
        )
        particle_gray = np.clip(particle_gray, 105.0, 225.0)
        output = (
            output * (1.0 - particle_mask[..., None])
            + particle_gray[..., None] * particle_mask[..., None]
        )

        # A small amount of neutral backscatter inside the headlamp beam makes
        # particles react to light without turning them into white snow.
        if self.illumination is not None:
            backscatter = particle_mask * self.illumination * 0.18
            output = output * (1.0 - backscatter[..., None]) + 152.0 * backscatter[..., None]
        return np.clip(output, 0.0, 255.0).astype(np.uint8)


class VideoEffectProcessor:
    def __init__(
        self,
        profile: str,
        dust: bool,
        width: int,
        height: int,
        seed: int,
        particle_count: int,
    ) -> None:
        self.profile = profile
        self.luts = _lighting_luts(profile)
        self.headlamp = None
        if profile == "headlamp_spot":
            self.headlamp = _radial_map(width, height, 0.55, 0.47, 0.28, 0.34)
        self.dust = (
            VideoDustField(
                width,
                height,
                seed,
                particle_count,
                self.headlamp,
            )
            if dust
            else None
        )

    def apply(self, frame: np.ndarray, frame_index: int) -> np.ndarray:
        output = transform_lighting_bgr(frame, self.luts)
        if self.headlamp is not None:
            gain = 0.24 + 1.02 * self.headlamp
            output = output.astype(np.float32) * gain[..., None]
            output *= np.asarray([0.92, 1.00, 1.06], dtype=np.float32)[None, None, :]
            output = np.clip(output, 0.0, 255.0).astype(np.uint8)
        if self.dust is not None:
            output = self.dust.apply(output, frame_index)
        return output


def transform_video_block(
    spec: DatasetSpec,
    output: Path,
    plan: list[PlanEntry],
    args: argparse.Namespace,
    block_index: int,
) -> None:
    half = args.source_count // 2
    start = args.source_count + block_index * half
    stop = start + half
    writer = create_writer(spec, output)
    if writer.total_episodes >= stop:
        writer.close()
        return
    if writer.total_episodes < start:
        writer.close()
        raise RuntimeError(
            f"Block {block_index} starts at {start}, but output has only "
            f"{writer.total_episodes} episodes"
        )
    entry_at_start = plan[start]
    if entry_at_start.kind != "video_transform":
        writer.close()
        raise ValueError(f"Block {block_index} is not a video-transform block")
    try:
        while writer.total_episodes < stop:
            entry = plan[writer.total_episodes]
            if entry.kind != "video_transform":
                raise RuntimeError("Video-transform block contains an incompatible plan entry")
            row, states, actions = source_data(spec, entry.source_episode)
            reader = SourceVideoReader(spec, row)
            processors: dict[str, VideoEffectProcessor] = {}
            writer.start_episode()
            try:
                for frame_index in range(len(actions)):
                    source_images = reader.read()
                    for camera_index, (key, image) in enumerate(source_images.items()):
                        if key not in processors:
                            processors[key] = VideoEffectProcessor(
                                entry.lighting,
                                entry.dust,
                                image.shape[1],
                                image.shape[0],
                                args.seed
                                + entry.output_episode * 1009
                                + camera_index * 100_003,
                                args.particle_count,
                            )
                    images = {
                        key: processors[key].apply(image, frame_index)
                        for key, image in source_images.items()
                    }
                    writer.add_frame(states[frame_index], actions[frame_index], images)
                output_index, frames = writer.save_episode()
            except Exception:
                writer.discard_episode()
                raise
            finally:
                reader.close()
            if output_index != entry.output_episode or frames != len(actions):
                raise RuntimeError("Video-transform commit returned inconsistent indices")
            update_manifest(spec, output, plan, writer.total_episodes, args)
            print(
                f"[augmentation] transformed source={entry.source_episode} -> "
                f"output={output_index}, lighting={entry.lighting}, "
                f"dust={entry.dust}, frames={frames}",
                flush=True,
            )
    finally:
        writer.close()


def orchestrate(
    spec: DatasetSpec,
    output: Path,
    plan: list[PlanEntry],
    args: argparse.Namespace,
) -> None:
    copy_originals(spec, output, plan, args)
    half = args.source_count // 2
    for block_index, (lighting, dust, _selection) in enumerate(PROFILE_BLOCKS):
        current_info = json.loads(
            (output / "meta/info.json").read_text(encoding="utf-8")
        )
        completed = int(current_info["total_episodes"])
        block_end = args.source_count + (block_index + 1) * half
        if completed >= block_end:
            continue
        print(
            f"[augmentation] transforming block {block_index}: "
            f"lighting={lighting}, dust={dust} (video only)",
            flush=True,
        )
        transform_video_block(spec, output, plan, args, block_index)

    final_info = json.loads((output / "meta/info.json").read_text(encoding="utf-8"))
    total = int(final_info["total_episodes"])
    update_manifest(spec, output, plan, total, args)
    if total != len(plan):
        raise RuntimeError(f"Expected {len(plan)} episodes, output contains {total}")
    print(f"[augmentation] COMPLETE: {total} episodes at {output}")


def print_plan(plan: list[PlanEntry]) -> None:
    groups: list[tuple[int, int, str, str, bool]] = []
    start = 0
    while start < len(plan):
        first = plan[start]
        stop = start + 1
        while (
            stop < len(plan)
            and plan[stop].kind == first.kind
            and plan[stop].lighting == first.lighting
            and plan[stop].dust == first.dust
        ):
            stop += 1
        groups.append((start, stop - 1, first.kind, first.lighting, first.dust))
        start = stop
    for begin, end, kind, lighting, dust in groups:
        print(
            f"episode {begin:03d}-{end:03d}: {lighting}, dust={dust}, "
            f"generation={kind}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a balanced 5x LeRobot v3 dataset by video-only augmentation"
    )
    parser.add_argument("--input", type=Path, required=True, help="source LeRobot v3 dataset")
    parser.add_argument("--output", type=Path, required=True, help="new/resumable output dataset")
    parser.add_argument("--source-start", type=int, default=0)
    parser.add_argument("--source-count", type=int, default=50)
    parser.add_argument("--particle-count", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=260823)
    parser.add_argument(
        "--mode",
        choices=("run", "dry-run", "copy-originals", "video-block"),
        default="run",
    )
    parser.add_argument("--block-index", type=int)
    args = parser.parse_args()
    args.input = args.input.expanduser().resolve()
    args.output = args.output.expanduser().resolve()
    if args.input == args.output:
        parser.error("--output must differ from --input; source data is never modified")
    if not 1 <= args.particle_count <= 10000:
        parser.error("--particle-count must be in [1, 10000]")
    if args.mode == "video-block":
        if args.block_index is None or not 0 <= args.block_index < len(PROFILE_BLOCKS):
            parser.error(f"{args.mode} requires --block-index in [0, 7]")
    return args


def main() -> None:
    args = parse_args()
    spec = load_dataset_spec(args.input)
    print(
        "[augmentation] video-only pipeline: no Isaac Sim or DDS",
        flush=True,
    )
    plan = build_plan(args.source_start, args.source_count)
    missing = sorted({entry.source_episode for entry in plan} - set(spec.rows))
    if missing:
        raise IndexError(f"Input dataset is missing source episodes: {missing}")
    if args.mode == "dry-run":
        print_plan(plan)
        print(f"input episodes available: {len(spec.rows)}")
        print(f"output target episodes: {len(plan)}")
        return
    args.output.mkdir(parents=True, exist_ok=True)
    validate_existing_manifest(spec, args.output, plan, args)
    info_path = args.output / "meta/info.json"
    if info_path.is_file():
        existing_total = int(json.loads(info_path.read_text(encoding="utf-8"))["total_episodes"])
        if existing_total > len(plan):
            raise ValueError(
                f"Output contains {existing_total} episodes, beyond this plan's {len(plan)}"
            )
    if args.mode == "copy-originals":
        copy_originals(spec, args.output, plan, args)
    elif args.mode == "video-block":
        transform_video_block(spec, args.output, plan, args, args.block_index)
    else:
        orchestrate(spec, args.output, plan, args)


if __name__ == "__main__":
    main()
