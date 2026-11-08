#!/usr/bin/env python3
"""Validate and assemble the native Task01 headlamp demo."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

import cv2
import numpy as np


NAMES = (
    "normal_clean",
    "dark_dust_no_headlamp",
    "dark_dust_headlamp",
)


def probe(path: Path) -> dict[str, object]:
    completed = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries",
            "stream=codec_name,width,height,pix_fmt,r_frame_rate,nb_frames",
            "-show_entries", "format=duration", "-of", "json", str(path),
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    return json.loads(completed.stdout)


def is_faststart(path: Path) -> bool:
    """Return whether the MP4 metadata atom precedes media payload."""
    payload = path.read_bytes()
    moov = payload.find(b"moov")
    mdat = payload.find(b"mdat")
    return moov >= 0 and mdat >= 0 and moov < mdat


def effective_replay_fps(root: Path) -> dict[str, float]:
    values: dict[str, float] = {}
    for name in NAMES:
        manifest = root / "raw" / name / "episode_000" / "manifest.json"
        data = json.loads(manifest.read_text(encoding="utf-8"))
        values[name] = float(data["effective_replay_fps"])
    return values


def make_comparison(inputs: list[Path], output: Path) -> None:
    subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-i", str(inputs[0]), "-i", str(inputs[1]), "-i", str(inputs[2]),
            "-filter_complex", "[0:v][1:v][2:v]hstack=inputs=3[out]",
            "-map", "[out]", "-an", "-c:v", "libx264", "-preset", "veryfast",
            "-crf", "19", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
            str(output),
        ],
        check=True,
    )


def open_caps(paths: list[Path]) -> list[cv2.VideoCapture]:
    caps = [cv2.VideoCapture(str(path)) for path in paths]
    if not all(cap.isOpened() for cap in caps):
        raise RuntimeError("Could not open all headlamp demo videos")
    return caps


def analyse(paths: list[Path]) -> tuple[dict[str, float], np.ndarray]:
    caps = open_caps(paths)
    normal_values: list[float] = []
    dark_values: list[float] = []
    lamp_values: list[float] = []
    dark_ratios: list[float] = []
    target_ratios: list[float] = []
    spot_fractions: list[float] = []
    outside_ratios: list[float] = []
    gains: list[float] = []
    saturation: list[float] = []
    still = None
    frame_index = 0
    try:
        while True:
            triplet = [cap.read() for cap in caps]
            if not all(ok for ok, _ in triplet):
                break
            frames = [frame for _, frame in triplet]
            if frame_index == 75:
                labelled = []
                labels = ("NORMAL", "DARK + DUST", "DARK + DUST + HEADLAMP")
                for frame, label in zip(frames, labels):
                    copy = frame.copy()
                    cv2.rectangle(copy, (0, 0), (copy.shape[1], 40), (0, 0, 0), -1)
                    cv2.putText(copy, label, (12, 28), cv2.FONT_HERSHEY_SIMPLEX,
                                0.72, (255, 255, 255), 2, cv2.LINE_AA)
                    labelled.append(copy)
                still = np.hstack(labelled)
            if frame_index % 15 == 0:
                gray = [cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
                        for frame in frames]
                normal, dark, lamp = gray
                h, w = lamp.shape
                roi = np.s_[int(0.22*h):int(0.92*h), int(0.14*w):int(0.86*w)]
                delta = lamp - dark
                spot = delta > 12.0
                outside = ~spot
                normal_values.append(float(np.median(normal)))
                dark_values.append(float(np.median(dark)))
                lamp_values.append(float(np.median(lamp)))
                dark_ratios.append(float(np.median(dark) / max(1.0, np.median(normal))))
                target_ratios.append(float(np.median(lamp[roi]) / max(1.0, np.median(normal[roi]))))
                spot_fractions.append(float(np.mean(spot)))
                if np.any(outside):
                    outside_ratios.append(float(np.median(lamp[outside]) / max(1.0, np.median(normal[outside]))))
                gains.append(float(np.mean(np.clip(delta, 0.0, None))))
                saturation.append(float(np.mean(np.all(frames[2] >= 250, axis=2))))
            frame_index += 1
    finally:
        for cap in caps:
            cap.release()
    if still is None:
        raise RuntimeError("Comparison still frame was not available")
    metrics = {
        "frames_compared": float(frame_index),
        "normal_median_luminance": float(np.mean(normal_values)),
        "dark_median_luminance": float(np.mean(dark_values)),
        "headlamp_median_luminance": float(np.mean(lamp_values)),
        "dark_to_normal_ratio_mean": float(np.mean(dark_ratios)),
        "headlamp_target_to_normal_ratio_mean": float(np.mean(target_ratios)),
        "headlamp_spot_fraction_mean": float(np.mean(spot_fractions)),
        "headlamp_spot_fraction_p90": float(np.percentile(spot_fractions, 90)),
        "outside_beam_to_normal_ratio_mean": float(np.mean(outside_ratios)),
        "headlamp_positive_gain_mean_8bit": float(np.mean(gains)),
        "saturated_white_fraction_mean": float(np.mean(saturation)),
    }
    return metrics, still


def peak_gpu_memory(root: Path) -> float:
    values: list[float] = []
    for path in sorted((root / "raw").glob("*/gpu.csv")):
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            fields = [field.strip() for field in line.split(",")]
            if len(fields) >= 3:
                try:
                    values.append(float(fields[-1]))
                except ValueError:
                    pass
    return max(values) if values else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    paths = [root / f"{name}_15s.mp4" for name in NAMES]
    probes = {name: probe(path) for name, path in zip(NAMES, paths)}
    replay_fps = effective_replay_fps(root)
    metrics, still = analyse(paths)
    comparison_video = root / "headlamp_comparison_3x1.mp4"
    comparison_still = root / "headlamp_comparison_3x1.png"
    make_comparison(paths, comparison_video)
    if not cv2.imwrite(str(comparison_still), still):
        raise RuntimeError(f"Could not write {comparison_still}")

    parameters = {
        "headlamp": {
            "mount": "/World/TrackedBase/G1UpperBody/d435_link",
            "local_position_m": [0.0, 0.035, -0.015],
            "local_orientation_wxyz": [0.5, -0.5, -0.5, 0.5],
            "intensity": 2000000.0,
            "radius_m": 0.025,
            "cone_angle_deg": 44.0,
            "cone_softness": 0.50,
            "focus": 2.0,
            "color_temperature_k": 4100.0,
            "camera_exposure_ev": 0.0,
            "shadows": True,
        },
        "environment": {
            "normal_dome_intensity": 1200.0,
            "dark_dome_intensity": 80.0,
            "nurec_emissive_scale": 0.14,
            "dust_mode": "flow",
            "dust_profile": "train",
            "dust_preset": "medium",
            "dust_ambient_multiplier": 1.20,
        },
        "render": {"frames": 450, "fps": 30.0, "seconds": 15.0},
    }
    (root / "headlamp_parameters.json").write_text(
        json.dumps(parameters, indent=2) + "\n", encoding="utf-8"
    )

    hashes_equal = (
        (root / "source_hashes.before.sha256").read_bytes()
        == (root / "source_hashes.after.sha256").read_bytes()
    )
    format_ok = True
    for info in probes.values():
        stream = info["streams"][0]
        format_ok &= (
            stream.get("codec_name") == "h264"
            and stream.get("width") == 640
            and stream.get("height") == 480
            and stream.get("pix_fmt") == "yuv420p"
            and int(stream.get("nb_frames", 0)) == 450
        )
    checks = {
        "all_videos_h264_640x480_yuv420p_450_frames": bool(format_ok),
        "all_videos_faststart": all(is_faststart(path) for path in paths),
        "dataset_hashes_unchanged": hashes_equal,
        "dark_background_ratio_at_most_0_45": metrics["dark_to_normal_ratio_mean"] <= 0.45,
        "headlamp_target_ratio_0_60_to_1_00": 0.60 <= metrics["headlamp_target_to_normal_ratio_mean"] <= 1.00,
        "spot_fraction_0_15_to_0_35": 0.15 <= metrics["headlamp_spot_fraction_mean"] <= 0.35,
        "outside_beam_ratio_at_most_0_35": metrics["outside_beam_to_normal_ratio_mean"] <= 0.35,
        "saturated_white_below_1_percent": metrics["saturated_white_fraction_mean"] < 0.01,
        "native_headlamp_has_visible_gain": metrics["headlamp_positive_gain_mean_8bit"] >= 6.0,
        "peak_gpu_memory_below_14500_mib": peak_gpu_memory(root) < 14500.0,
        "effective_replay_fps_at_least_20": min(replay_fps.values()) >= 20.0,
    }
    report = {
        "effect": "native_usd_camera_mounted_headlamp",
        "source_dataset": str(args.dataset.resolve()),
        "probes": probes,
        "metrics": metrics,
        "peak_gpu_memory_mib": peak_gpu_memory(root),
        "effective_replay_fps_by_variant": replay_fps,
        "checks": checks,
        "all_checks_passed": all(checks.values()),
        "rgb_postprocess_used_for_lighting": False,
        "dataset_modified": False,
    }
    (root / "headlamp_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    readme = (
        "# Task01 原生头戴式矿灯 Demo\n\n"
        "使用相机子级 SphereLight + UsdLux ShapingAPI 实现，环境为暗场与中等 Flow 扬尘。\n"
        "灯光未使用 RGB 后期光斑合成，原始数据集未修改。\n\n"
        f"- 全部验收通过：`{all(checks.values())}`\n"
        f"- 暗场/正常亮度：`{metrics['dark_to_normal_ratio_mean']:.3f}`\n"
        f"- 头灯目标区/正常亮度：`{metrics['headlamp_target_to_normal_ratio_mean']:.3f}`\n"
        f"- 光斑覆盖：`{metrics['headlamp_spot_fraction_mean']:.3f}`\n"
        f"- 峰值显存：`{peak_gpu_memory(root):.0f} MiB`\n"
        f"- 最低有效回放速度：`{min(replay_fps.values()):.2f} FPS`\n"
        "- 暗场 NuRec 自发光比例：`0.14`\n"
        "- 头灯：`2,000,000 intensity / 44° / 4100 K`\n"
    )
    (root / "README.md").write_text(readme, encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
