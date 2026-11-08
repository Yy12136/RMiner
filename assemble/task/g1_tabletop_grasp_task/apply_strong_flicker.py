#!/usr/bin/env python3
"""Create a stronger dark-and-color industrial flicker preview."""
from __future__ import annotations
import argparse, json, math, subprocess
from pathlib import Path
import cv2
import numpy as np

def base_multiplier(t: float, events: list[dict[str, float]]) -> float:
    value = 1.0 + 0.12 * math.sin(2 * math.pi * 0.90 * t)
    value += 0.05 * math.sin(2 * math.pi * 2.25 * t + 0.7)
    dip = 1.0
    for event in events:
        phase = (t - float(event["start"])) / float(event["duration"])
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
    cap = cv2.VideoCapture(str(args.input))
    if not cap.isOpened(): raise RuntimeError(f"Cannot open {args.input}")
    fps = float(cap.get(cv2.CAP_PROP_FPS)); width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    process = subprocess.Popen(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-video_size", f"{width}x{height}", "-framerate", f"{fps:.8f}", "-i", "pipe:0", "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "19", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(args.output)], stdin=subprocess.PIPE)
    assert process.stdin is not None
    multipliers, channel_scales = [], []; frames = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok: break
            t = (frames + 0.5) / fps; raw = base_multiplier(t, curve["events"])
            mapped = float(np.clip(0.08 + 0.92 * ((raw - 0.40) / 0.60), 0.08, 1.15))
            phase = math.sin(2 * math.pi * 0.45 * t + 0.5)
            warm = 1.0 if mapped > 0.35 else 1.0 + 0.06 * (1.0 - mapped / 0.35)
            b = (1.0 - 0.16 * phase) * (1.0 - 0.10 * (1.0 - mapped))
            g = 1.0 + 0.04 * math.sin(2 * math.pi * 0.70 * t)
            r = (1.0 + 0.14 * phase) * warm
            adjusted = frame.astype(np.float32); adjusted[:, :, 0] *= mapped * b; adjusted[:, :, 1] *= mapped * g; adjusted[:, :, 2] *= mapped * r
            process.stdin.write(np.ascontiguousarray(np.clip(adjusted, 0.0, 255.0).astype(np.uint8)).tobytes())
            multipliers.append(mapped); channel_scales.append((b, g, r)); frames += 1
        process.stdin.close()
        if process.wait(timeout=120) != 0: raise RuntimeError("ffmpeg failed")
    finally:
        cap.release()
        if process.poll() is None: process.terminate()
    scales = np.asarray(channel_scales)
    args.manifest.write_text(json.dumps({"method":"strong_dark_color_rgb_brightness_fallback", "reason":"off-screen RGB sensor ignored live DomeLight and camera exposure", "source_video":str(args.input), "seed":curve["seed"], "fps":fps, "frames":frames, "brightness_multiplier_minimum":min(multipliers), "brightness_multiplier_maximum":max(multipliers), "color_scale_minimum_bgr":scales.min(axis=0).tolist(), "color_scale_maximum_bgr":scales.max(axis=0).tolist(), "dataset_modified":False}, indent=2) + "\n", encoding="utf-8")

if __name__ == "__main__": main()
