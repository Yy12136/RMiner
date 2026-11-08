#!/usr/bin/env python3
"""Apply the deterministic Isaac light curve to rendered RGB as a demo fallback.

The simulator still animates the USD DomeLight.  This compositor is used only
because the current IsaacLab off-screen RGB sensor ignores live light/exposure
changes.  It never writes into the LeRobot dataset.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import subprocess

import cv2
import numpy as np


def multiplier(time_seconds: float, events: list[dict[str, float]]) -> float:
    value = 1.0 + 0.12 * math.sin(2.0 * math.pi * 0.90 * time_seconds)
    value += 0.05 * math.sin(2.0 * math.pi * 2.25 * time_seconds + 0.7)
    dip = 1.0
    for event in events:
        phase = (time_seconds - float(event["start"])) / float(event["duration"])
        if 0.0 < phase < 1.0:
            dip = min(dip, 1.0 - float(event["depth"]) * math.sin(math.pi * phase))
    return max(0.40, min(1.25, value * dip))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--curve", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()

    curve = json.loads(args.curve.read_text(encoding="utf-8"))
    events = curve["events"]
    capture = cv2.VideoCapture(str(args.input))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open {args.input}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-f", "rawvideo", "-pix_fmt", "bgr24", "-video_size", f"{width}x{height}",
        "-framerate", f"{fps:.8f}", "-i", "pipe:0", "-an",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "19",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(args.output),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    assert process.stdin is not None
    values = []
    frames = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            time_seconds = (frames + 0.5) / fps
            value = multiplier(time_seconds, events)
            # Apply in displayed RGB space so the intended industrial dips are
            # visible despite the sensor's ignored USD/camera exposure edits.
            adjusted = np.clip(frame.astype(np.float32) * value, 0.0, 255.0).astype(np.uint8)
            process.stdin.write(np.ascontiguousarray(adjusted).tobytes())
            values.append(value)
            frames += 1
        process.stdin.close()
        if process.wait(timeout=120) != 0:
            raise RuntimeError("ffmpeg failed")
    finally:
        capture.release()
        if process.poll() is None:
            process.terminate()

    payload = {
        "method": "deterministic_rgb_brightness_fallback",
        "reason": "IsaacLab off-screen RGB sensor ignored live DomeLight and Camera exposure edits",
        "source_video": str(args.input),
        "seed": curve["seed"],
        "fps": fps,
        "frames": frames,
        "multiplier_minimum": float(min(values)),
        "multiplier_maximum": float(max(values)),
        "multiplier_mean": float(sum(values) / len(values)),
        "dataset_modified": False,
    }
    args.manifest.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
