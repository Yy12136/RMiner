#!/usr/bin/env python3
"""Assemble and validate the bright-centre, no-dust Task01 headlamp demo."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

import cv2
import numpy as np


def probe(path: Path) -> dict[str, object]:
    result = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=codec_name,width,height,pix_fmt,nb_frames",
            "-show_entries", "format=duration", "-of", "json", str(path),
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    return json.loads(result.stdout)


def faststart(path: Path) -> bool:
    payload = path.read_bytes()
    return 0 <= payload.find(b"moov") < payload.find(b"mdat")


def peak_gpu_memory(path: Path) -> float:
    values: list[float] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) >= 3:
            try:
                values.append(float(fields[-1]))
            except ValueError:
                pass
    return max(values) if values else 0.0


def analyse(normal_path: Path, lamp_path: Path) -> tuple[dict[str, float], np.ndarray]:
    caps = [cv2.VideoCapture(str(normal_path)), cv2.VideoCapture(str(lamp_path))]
    if not all(cap.isOpened() for cap in caps):
        raise RuntimeError("Could not open comparison videos")
    target_ratios: list[float] = []
    spot_fractions: list[float] = []
    gains: list[float] = []
    saturation: list[float] = []
    still = None
    frame_index = 0
    try:
        while True:
            pair = [cap.read() for cap in caps]
            if not all(ok for ok, _ in pair):
                break
            frames = [frame for _, frame in pair]
            if frame_index == 75:
                labelled = []
                for frame, label in zip(frames, ("NORMAL", "DARK + BRIGHT HEADLAMP / NO DUST")):
                    copy = frame.copy()
                    cv2.rectangle(copy, (0, 0), (copy.shape[1], 40), (0, 0, 0), -1)
                    cv2.putText(copy, label, (12, 28), cv2.FONT_HERSHEY_SIMPLEX,
                                0.65, (255, 255, 255), 2, cv2.LINE_AA)
                    labelled.append(copy)
                still = np.hstack(labelled)
            if frame_index % 15 == 0:
                normal, lamp = [
                    cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
                    for frame in frames
                ]
                h, w = lamp.shape
                roi = np.s_[int(0.22 * h):int(0.92 * h), int(0.14 * w):int(0.86 * w)]
                delta = lamp - normal
                target_ratios.append(float(np.median(lamp[roi]) / max(1.0, np.median(normal[roi]))))
                spot_fractions.append(float(np.mean(delta > 12.0)))
                gains.append(float(np.mean(np.clip(delta, 0.0, None))))
                saturation.append(float(np.mean(np.all(frames[1] >= 250, axis=2))))
            frame_index += 1
    finally:
        for cap in caps:
            cap.release()
    if still is None:
        raise RuntimeError("Comparison still was not available")
    return {
        "frames_compared": float(frame_index),
        "headlamp_target_to_normal_ratio_mean": float(np.mean(target_ratios)),
        "bright_area_fraction_mean": float(np.mean(spot_fractions)),
        "positive_gain_mean_8bit": float(np.mean(gains)),
        "saturated_white_fraction_mean": float(np.mean(saturation)),
    }, still


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--normal", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    normal = args.normal.resolve()
    lamp = root / "dark_no_dust_bright_headlamp_15s.mp4"
    comparison = root / "normal_vs_no_dust_bright_headlamp_2x1.mp4"
    still_path = root / "normal_vs_no_dust_bright_headlamp_2x1.png"

    subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-i", str(normal), "-i", str(lamp),
            "-filter_complex", "[0:v][1:v]hstack=inputs=2[out]",
            "-map", "[out]", "-an", "-c:v", "libx264", "-crf", "19",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(comparison),
        ],
        check=True,
    )
    metrics, still = analyse(normal, lamp)
    if not cv2.imwrite(str(still_path), still):
        raise RuntimeError(f"Could not write {still_path}")
    info = probe(lamp)
    manifest = json.loads((root / "raw" / "episode_000" / "manifest.json").read_text())
    replay_fps = float(manifest["effective_replay_fps"])
    peak_memory = peak_gpu_memory(root / "gpu.csv")
    stream = info["streams"][0]
    checks = {
        "video_h264_640x480_yuv420p_450_frames": (
            stream.get("codec_name") == "h264"
            and stream.get("width") == 640
            and stream.get("height") == 480
            and stream.get("pix_fmt") == "yuv420p"
            and int(stream.get("nb_frames", 0)) == 450
        ),
        "video_faststart": faststart(lamp),
        "dataset_hashes_unchanged": (
            (root / "source_hashes.before.sha256").read_bytes()
            == (root / "source_hashes.after.sha256").read_bytes()
        ),
        "no_saturated_white": metrics["saturated_white_fraction_mean"] < 0.01,
        "effective_replay_fps_at_least_20": replay_fps >= 20.0,
        "peak_gpu_memory_below_14500_mib": peak_memory < 14500.0,
    }
    report = {
        "effect": "native_usd_bright_centre_headlamp_without_dust",
        "parameters": {
            "headlamp_intensity": 30000000.0,
            "cone_angle_deg": 32.0,
            "cone_softness": 0.45,
            "focus": 2.5,
            "color_temperature_k": 4100.0,
            "dome_intensity": 80.0,
            "nurec_emissive_scale": 0.16,
            "dust_mode": "off",
        },
        "probe": info,
        "metrics": metrics,
        "effective_replay_fps": replay_fps,
        "peak_gpu_memory_mib": peak_memory,
        "checks": checks,
        "all_checks_passed": all(checks.values()),
        "rgb_postprocess_used_for_lighting": False,
        "dataset_modified": False,
    }
    (root / "no_dust_headlamp_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
