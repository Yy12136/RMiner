#!/usr/bin/env python3
"""Record the simulator-rendered head camera without modifying LeRobot data."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import time

import numpy as np

from benchmark_camera import CameraReader


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=15.0)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--camera-timeout", type=float, default=120.0)
    args = parser.parse_args()
    if args.seconds <= 0.0 or args.fps <= 0.0:
        raise ValueError("--seconds and --fps must be positive")

    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    target_frames = int(round(args.seconds * args.fps))
    process = None
    reader = CameraReader()
    try:
        deadline = time.monotonic() + args.camera_timeout
        first = None
        while time.monotonic() < deadline:
            first = reader.read("head")
            if first is not None and first.age_seconds < 2.0:
                break
            time.sleep(0.05)
        if first is None or first.age_seconds >= 2.0:
            raise TimeoutError("Head camera did not become ready")

        height, width = first.bgr.shape[:2]
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
            f"{args.fps:.8f}",
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
            str(output),
        ]
        process = subprocess.Popen(command, stdin=subprocess.PIPE)
        assert process.stdin is not None
        period = 1.0 / args.fps
        next_frame_time = time.monotonic()
        latest = first.bgr
        for frame_index in range(target_frames):
            while True:
                now = time.monotonic()
                if now >= next_frame_time:
                    break
                time.sleep(min(0.002, next_frame_time - now))
            camera_frame = reader.read("head")
            if camera_frame is not None and camera_frame.age_seconds < 2.0:
                latest = camera_frame.bgr
            process.stdin.write(np.ascontiguousarray(latest).tobytes())
            next_frame_time += period
            if (frame_index + 1) % 90 == 0:
                print(
                    f"[dust-preview] Recorded {frame_index + 1}/{target_frames} frames",
                    flush=True,
                )
        process.stdin.close()
        return_code = process.wait(timeout=120)
        if return_code != 0:
            raise RuntimeError(f"ffmpeg exited with status {return_code}")
        print(f"[dust-preview] Saved {output}")
    finally:
        reader.close()
        if process is not None and process.poll() is None:
            if process.stdin is not None and not process.stdin.closed:
                process.stdin.close()
            process.terminate()
            process.wait(timeout=5)


if __name__ == "__main__":
    main()

