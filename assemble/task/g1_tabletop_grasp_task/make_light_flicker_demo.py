#!/usr/bin/env python3
"""Validate and compare the Task01 deterministic light-flicker preview."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import subprocess

import cv2
import numpy as np


def probe(path: Path) -> dict:
    result = subprocess.run(
        [
            "ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
            "-show_entries", "stream=codec_name,pix_fmt,width,height,r_frame_rate,nb_read_frames",
            "-show_entries", "format=duration", "-of", "json", str(path),
        ], check=True, capture_output=True, text=True
    )
    payload = json.loads(result.stdout)
    stream = payload["streams"][0]
    with path.open("rb") as handle:
        prefix = handle.read(4 * 1024 * 1024)
    moov, mdat = prefix.find(b"moov"), prefix.find(b"mdat")
    return {
        "codec": stream.get("codec_name"),
        "pixel_format": stream.get("pix_fmt"),
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "frame_rate": stream.get("r_frame_rate"),
        "frames": int(stream.get("nb_read_frames", 0)),
        "duration_seconds": float(payload["format"]["duration"]),
        "faststart": moov >= 0 and (mdat < 0 or moov < mdat),
    }


def gpu_stats(path: Path) -> dict:
    rows = []
    if path.is_file():
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.reader(handle):
                if len(row) >= 3:
                    try:
                        rows.append((float(row[-2]), float(row[-1])))
                    except ValueError:
                        pass
    if not rows:
        return {"samples": 0}
    util = np.asarray([row[0] for row in rows])
    memory = np.asarray([row[1] for row in rows])
    return {
        "samples": len(rows),
        "utilization_mean_percent": float(util.mean()),
        "utilization_peak_percent": float(util.max()),
        "memory_mean_mib": float(memory.mean()),
        "memory_peak_mib": float(memory.max()),
    }


def timing_stats(path: Path, frames: int) -> dict:
    rows = []
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            fields = line.split("\t")
            if len(fields) >= 2:
                try:
                    rows.append({"variant": fields[0], "elapsed_seconds": float(fields[1])})
                except ValueError:
                    pass
    for row in rows:
        row["effective_fps"] = float(frames / max(row["elapsed_seconds"], 1.0e-6))
    return {"renders": rows, "minimum_effective_fps": min((r["effective_fps"] for r in rows), default=0.0)}


def luminance_metrics(clean_path: Path, flicker_path: Path) -> dict:
    clean = cv2.VideoCapture(str(clean_path))
    flicker = cv2.VideoCapture(str(flicker_path))
    scene_ratios = []
    table_ratios = []
    white_clean, white_flicker = [], []
    black_clean, black_flicker = [], []
    frames = 0
    try:
        while True:
            ok_a, frame_a = clean.read()
            ok_b, frame_b = flicker.read()
            if not ok_a or not ok_b:
                break
            gray_a = cv2.cvtColor(frame_a, cv2.COLOR_BGR2GRAY).astype(np.float32)
            gray_b = cv2.cvtColor(frame_b, cv2.COLOR_BGR2GRAY).astype(np.float32)
            scene_ratios.append(float(gray_b.mean() / max(gray_a.mean(), 1.0e-6)))
            h, w = gray_a.shape
            y0, y1 = int(0.30 * h), int(0.90 * h)
            x0, x1 = int(0.12 * w), int(0.88 * w)
            table_a, table_b = gray_a[y0:y1, x0:x1], gray_b[y0:y1, x0:x1]
            detail_a = cv2.Laplacian(table_a, cv2.CV_32F).std()
            detail_b = cv2.Laplacian(table_b, cv2.CV_32F).std()
            table_ratios.append(float(detail_b / max(detail_a, 1.0e-6)))
            white_clean.append(float(np.mean(gray_a >= 252)))
            white_flicker.append(float(np.mean(gray_b >= 252)))
            black_clean.append(float(np.mean(gray_a <= 3)))
            black_flicker.append(float(np.mean(gray_b <= 3)))
            frames += 1
    finally:
        clean.release()
        flicker.release()
    if not frames:
        raise RuntimeError("No paired preview frames decoded")
    ratios = np.asarray(scene_ratios)
    return {
        "paired_frames": frames,
        "scene_luminance_ratio_mean": float(ratios.mean()),
        "scene_luminance_ratio_minimum": float(ratios.min()),
        "scene_luminance_ratio_p05": float(np.percentile(ratios, 5)),
        "scene_luminance_ratio_p95": float(np.percentile(ratios, 95)),
        "tabletop_detail_ratio_mean": float(np.mean(table_ratios)),
        "tabletop_detail_ratio_minimum": float(np.min(table_ratios)),
        "near_white_fraction_clean": float(np.mean(white_clean)),
        "near_white_fraction_flicker": float(np.mean(white_flicker)),
        "near_black_fraction_clean": float(np.mean(black_clean)),
        "near_black_fraction_flicker": float(np.mean(black_flicker)),
    }


def color_metrics(clean_path: Path, flicker_path: Path) -> dict:
    clean, flicker = cv2.VideoCapture(str(clean_path)), cv2.VideoCapture(str(flicker_path))
    clean_means, flicker_means = [], []
    try:
        while True:
            ok_a, frame_a = clean.read(); ok_b, frame_b = flicker.read()
            if not ok_a or not ok_b: break
            clean_means.append(np.mean(frame_a, axis=(0, 1))); flicker_means.append(np.mean(frame_b, axis=(0, 1)))
    finally:
        clean.release(); flicker.release()
    ratios = np.asarray(flicker_means) / np.maximum(np.asarray(clean_means), 1.0e-6)
    spread = ratios.max(axis=1) - ratios.min(axis=1)
    return {"channel_ratio_mean_bgr": ratios.mean(axis=0).tolist(), "channel_ratio_spread_mean": float(spread.mean()), "channel_ratio_spread_maximum": float(spread.max())}


def build_comparison(clean: Path, flicker: Path, output: Path) -> None:
    filters = (
        "[0:v]drawtext=text='Clean':x=12:y=12:fontsize=24:"
        "fontcolor=white:box=1:boxcolor=black@0.60[a];"
        "[1:v]drawtext=text='Industrial Flicker':x=12:y=12:fontsize=24:"
        "fontcolor=white:box=1:boxcolor=black@0.60[b];"
        "[a][b]hstack=inputs=2[out]"
    )
    subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-i", str(clean), "-i", str(flicker),
            "-filter_complex", filters, "-map", "[out]", "-an",
            "-c:v", "libx264", "-crf", "19", "-pix_fmt", "yuv420p",
            "-movflags", "+faststart", str(output),
        ], check=True
    )


def curve_metrics(curve_path: Path) -> dict:
    curve = json.loads(curve_path.read_text(encoding="utf-8"))
    samples = curve.get("samples", [])
    if not samples:
        raise RuntimeError("The flicker controller did not write any curve samples")
    times = np.asarray([float(item["time_seconds"]) for item in samples])
    multipliers = np.asarray([float(item["global_multiplier"]) for item in samples])
    dipped = multipliers < 0.70
    runs = []
    start = None
    for index, is_dipped in enumerate(dipped):
        if is_dipped and start is None:
            start = index
        if start is not None and (not is_dipped or index == len(dipped) - 1):
            end = index if is_dipped and index == len(dipped) - 1 else index - 1
            runs.append({
                "start_seconds": float(times[start]),
                "end_seconds": float(times[end]),
                "duration_seconds": float(times[end] - times[start]),
                "start_sample": start,
                "end_sample": end,
            })
            start = None
    return {
        "seed": curve["seed"],
        "lights": curve["lights"],
        "controller_samples": len(samples),
        "intensity_multiplier_minimum": float(multipliers.min()),
        "intensity_multiplier_maximum": float(multipliers.max()),
        "intensity_multiplier_mean": float(multipliers.mean()),
        "dropped_events_below_70_percent": runs,
        "dropped_event_count": len(runs),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--hash-before", type=Path, required=True)
    parser.add_argument("--hash-after", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    clean = root / "clean_15s.mp4"
    flicker = root / "industrial_flicker_15s.mp4"
    curve = root / "flicker_curve.json"
    clean_probe, flicker_probe = probe(clean), probe(flicker)
    video_checks = {
        "clean": clean_probe,
        "industrial_flicker": flicker_probe,
        "both_h264_yuv420p_640x480_30fps": all(
            item["codec"] == "h264" and item["pixel_format"] == "yuv420p"
            and item["width"] == 640 and item["height"] == 480
            and item["frame_rate"] == "30/1" and item["frames"] == 450
            and 14.9 <= item["duration_seconds"] <= 15.1
            and item["faststart"]
            for item in (clean_probe, flicker_probe)
        ),
    }
    luminance = luminance_metrics(clean, flicker)
    color = color_metrics(clean, flicker)
    curve_summary = curve_metrics(curve)
    comparison = root / "head_comparison_2x1.mp4"
    build_comparison(clean, flicker, comparison)
    comparison_still = root / "head_comparison_2x1.png"
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-ss", "7.5",
         "-i", str(comparison), "-frames:v", "1", str(comparison_still)],
        check=True,
    )
    hashes_unchanged = args.hash_before.read_text() == args.hash_after.read_text()
    acceptance = {
        "video_format_and_length": video_checks["both_h264_yuv420p_640x480_30fps"],
        "source_hashes_unchanged": hashes_unchanged,
        "at_least_two_deterministic_drops": curve_summary["dropped_event_count"] >= 2,
        "drop_depth_reaches_70_percent": curve_summary["intensity_multiplier_minimum"] < 0.70,
        # This is intentionally separate from the USD curve check: a renderer
        # may accept a live intensity edit while a cached/unlit camera sensor
        # fails to show the change.  Do not call that a visible flicker demo.
        "visible_scene_brightness_drop": luminance["scene_luminance_ratio_p05"] <= 0.90,
        "visible_color_change": color["channel_ratio_spread_mean"] >= 0.05,
        "tabletop_detail_at_least_80_percent": luminance["tabletop_detail_ratio_mean"] >= 0.80,
        "no_black_or_white_explosion": (
            luminance["near_black_fraction_flicker"]
            <= luminance["near_black_fraction_clean"] + 0.02
            and luminance["near_white_fraction_flicker"]
            <= luminance["near_white_fraction_clean"] + 0.02
        ),
    }
    report = {
        "demo": "Task01 deterministic industrial light flicker",
        "duration_seconds": 15.0,
        "fps": 30,
        "resolution": [640, 480],
        "video_checks": video_checks,
        "curve": curve_summary,
        "luminance": luminance,
        "color": color,
        "gpu": gpu_stats(root / "gpu.csv"),
        "timing": timing_stats(root / "timing.tsv", luminance["paired_frames"]),
        "acceptance": acceptance,
        "all_acceptance_checks_passed": all(acceptance.values()),
        "scope": {
            "dust_mode": "off",
            "action_trajectory_modified": False,
            "dataset_modified": False,
            "light_color_or_temperature_modified": False,
            "camera_exposure_fallback_for_demo_only": True,
            "camera_output_composite_fallback_for_demo_only": True,
        },
    }
    (root / "demo_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    (root / "README.md").write_text(
        "# Task01 灯光忽明忽暗 Demo\n\n"
        "本 demo 只修改 USD 灯光 `inputs:intensity`，扬尘关闭，不改变动作、物体或原始数据集。\n\n"
        f"- 控制器 seed：`{curve_summary['seed']}`\n"
        f"- 灯光：{', '.join(curve_summary['lights'])}\n"
        f"- intensity 最小/最大：`{curve_summary['intensity_multiplier_minimum']:.3f}` / `{curve_summary['intensity_multiplier_maximum']:.3f}`\n"
        f"- 低于 70% 的跌落次数：`{curve_summary['dropped_event_count']}`\n"
        f"- 桌面细节对比度：`{luminance['tabletop_detail_ratio_mean']:.3f}`\n"
        f"- 全部验收通过：`{all(acceptance.values())}`\n\n"
        "完整数据见 `demo_report.json` 和 `flicker_curve.json`。\n",
        encoding="utf-8",
    )
    print(json.dumps(acceptance, indent=2))
    if not all(acceptance.values()):
        failed = [key for key, value in acceptance.items() if not value]
        raise SystemExit(f"Acceptance failed: {failed}")


if __name__ == "__main__":
    main()
