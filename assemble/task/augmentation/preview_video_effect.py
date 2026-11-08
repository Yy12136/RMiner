#!/usr/bin/env python3
"""Render one first-person LeRobot episode with video-only augmentation."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from augment_lerobot_v3_video import (  # noqa: E402
    VideoEffectProcessor,
    load_dataset_spec,
    source_data,
)


class RawEncoder:
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
                str(path),
            ],
            stdin=subprocess.PIPE,
        )

    def write(self, frame: np.ndarray) -> None:
        if self.process.stdin is None:
            raise RuntimeError("ffmpeg stdin is unavailable")
        self.process.stdin.write(np.ascontiguousarray(frame).tobytes())

    def finish(self) -> None:
        if self.process.stdin is not None:
            self.process.stdin.close()
        code = self.process.wait(timeout=60)
        if code != 0:
            raise RuntimeError(f"ffmpeg failed with exit status {code}")

    def abort(self) -> None:
        if self.process.stdin is not None and not self.process.stdin.closed:
            self.process.stdin.close()
        if self.process.poll() is None:
            self.process.terminate()
            self.process.wait(timeout=5)


def add_label(frame: np.ndarray, label: str) -> np.ndarray:
    result = frame.copy()
    overlay = result.copy()
    cv2.rectangle(overlay, (8, 8), (430, 46), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.62, result, 0.38, 0.0, result)
    cv2.putText(
        result,
        label,
        (18, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.68,
        (245, 245, 245),
        2,
        cv2.LINE_AA,
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--camera-key", default="observation.images.head")
    parser.add_argument(
        "--lighting",
        choices=("low_illumination", "warm_worklamp", "cool_led", "headlamp_spot"),
        default="headlamp_spot",
    )
    parser.add_argument("--dust", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--particle-count", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=260823)
    args = parser.parse_args()

    dataset = load_dataset_spec(args.input.expanduser().resolve())
    row, _states, actions = source_data(dataset, args.episode)
    if args.camera_key not in dataset.camera_keys:
        parser.error(
            f"camera key {args.camera_key!r} is unavailable; choose from {dataset.camera_keys}"
        )
    video_path = dataset.root / (
        f"videos/{args.camera_key}/"
        f"chunk-{int(row[f'videos/{args.camera_key}/chunk_index']):03d}/"
        f"file-{int(row[f'videos/{args.camera_key}/file_index']):03d}.mp4"
    )
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open {video_path}")
    start = int(
        round(float(row[f"videos/{args.camera_key}/from_timestamp"]) * dataset.fps)
    )
    capture.set(cv2.CAP_PROP_POS_FRAMES, start)
    ok, first = capture.read()
    if not ok or first is None:
        raise RuntimeError("Source video has no readable first frame")
    height, width = first.shape[:2]
    capture.set(cv2.CAP_PROP_POS_FRAMES, start)

    effect = VideoEffectProcessor(
        args.lighting,
        args.dust,
        width,
        height,
        args.seed + args.episode * 1009,
        args.particle_count,
    )
    output = args.output.expanduser().resolve()
    comparison = output.with_name(f"{output.stem}_comparison{output.suffix}")
    processed_encoder = RawEncoder(output, width, height, dataset.fps)
    comparison_encoder = RawEncoder(comparison, 2 * width, height, dataset.fps)
    try:
        for frame_index in range(len(actions)):
            ok, frame = capture.read()
            if not ok or frame is None:
                raise RuntimeError(f"Source video ended at frame {frame_index}")
            processed = effect.apply(frame, frame_index)
            processed_encoder.write(processed)
            comparison_encoder.write(
                np.concatenate(
                    (
                        add_label(frame, "ORIGINAL SUCCESS"),
                        add_label(processed, "HEADLAMP + GRAY DUST"),
                    ),
                    axis=1,
                )
            )
        processed_encoder.finish()
        comparison_encoder.finish()
    except BaseException:
        processed_encoder.abort()
        comparison_encoder.abort()
        raise
    finally:
        capture.release()
    print(output)
    print(comparison)


if __name__ == "__main__":
    main()
