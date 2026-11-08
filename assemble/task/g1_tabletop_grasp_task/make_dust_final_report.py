#!/usr/bin/env python3
"""Build final Task01 dust deliverables and an auditable metrics report."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import shutil
import subprocess

import cv2
import numpy as np

from dust_visibility_metrics import measure


VARIANTS = (
    "clean",
    "current_flow_medium_baseline",
    "final_train_low",
    "final_train_medium",
    "final_train_heavy",
    "final_cinematic",
)
COMPARISON_VARIANTS = (
    ("clean", "Clean"),
    ("current_flow_medium_baseline", "Current Flow Medium"),
    ("final_train_medium", "Final Train Medium"),
    ("final_cinematic", "Final Cinematic"),
)
CAMERA_FILES = (
    "observation.images.head.mp4",
    "observation.images.left.mp4",
    "observation.images.right.mp4",
)


def read_gpu_csv(path: Path) -> dict:
    rows = []
    if path.is_file():
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.reader(handle):
                if len(row) < 3:
                    continue
                try:
                    rows.append((float(row[-2]), float(row[-1])))
                except ValueError:
                    continue
    if not rows:
        return {"samples": 0}
    utilization = np.asarray([row[0] for row in rows])
    memory = np.asarray([row[1] for row in rows])
    return {
        "samples": len(rows),
        "gpu_utilization_mean_percent": float(utilization.mean()),
        "gpu_utilization_peak_percent": float(utilization.max()),
        "gpu_memory_mean_mib": float(memory.mean()),
        "gpu_memory_peak_mib": float(memory.max()),
    }


def probe_video(path: Path) -> dict:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-count_frames",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_name,pix_fmt,width,height,r_frame_rate,nb_read_frames",
            "-show_entries",
            "format=duration",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)
    stream = payload["streams"][0]
    with path.open("rb") as handle:
        prefix = handle.read(4 * 1024 * 1024)
    moov = prefix.find(b"moov")
    mdat = prefix.find(b"mdat")
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


def appearance_proxies(path: Path) -> dict:
    capture = cv2.VideoCapture(str(path))
    orange = []
    white = []
    black = []
    index = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if index >= 60 and index % 6 == 0:
                blue, green, red = cv2.split(frame.astype(np.float32))
                orange.append(
                    np.mean(
                        (red > 170)
                        & (red > 1.35 * green)
                        & (green > 1.10 * blue)
                    )
                )
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                white.append(np.mean((hsv[..., 1] < 18) & (hsv[..., 2] > 245)))
                black.append(np.mean(hsv[..., 2] < 4))
            index += 1
    finally:
        capture.release()
    return {
        "orange_pixel_fraction_mean": float(np.mean(orange)),
        "near_white_pixel_fraction_mean": float(np.mean(white)),
        "near_black_pixel_fraction_mean": float(np.mean(black)),
        "note": "Image-color proxies; the orange task cable is included.",
    }


def build_comparison(root: Path, output: Path) -> None:
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y"]
    filters = []
    labelled = []
    for index, (variant, label) in enumerate(COMPARISON_VARIANTS):
        command.extend(
            [
                "-i",
                str(
                    root
                    / variant
                    / "episode_000"
                    / "observation.images.head.mp4"
                ),
            ]
        )
        filters.append(
            f"[{index}:v]drawtext=text='{label}':x=12:y=12:fontsize=22:"
            f"fontcolor=white:box=1:boxcolor=black@0.60[v{index}]"
        )
        labelled.append(f"[v{index}]")
    filters.append(
        "".join(labelled)
        + "xstack=inputs=4:layout=0_0|w0_0|0_h0|w0_h0[out]"
    )
    command.extend(
        [
            "-filter_complex",
            ";".join(filters),
            "-map",
            "[out]",
            "-an",
            "-c:v",
            "libx264",
            "-crf",
            "19",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(output),
        ]
    )
    subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.expanduser().resolve()
    deliverables = root / "deliverables"
    deliverables.mkdir(parents=True, exist_ok=True)
    clean_head = root / "clean/episode_000/observation.images.head.mp4"

    variants = {}
    for variant in VARIANTS:
        episode = root / variant / "episode_000"
        manifest = json.loads((episode / "manifest.json").read_text())
        videos = {
            name: probe_video(episode / name)
            for name in (*CAMERA_FILES, "three_camera_grid.mp4")
        }
        variants[variant] = {
            "manifest": manifest,
            "videos": videos,
            "gpu": read_gpu_csv(root / variant / "gpu.csv"),
            "appearance": appearance_proxies(
                episode / "observation.images.head.mp4"
            ),
        }
        if variant != "clean":
            variants[variant]["visibility"] = measure(
                clean_head, episode / "observation.images.head.mp4"
            )

    comparison = deliverables / "task01_dust_final_head_comparison_2x2.mp4"
    build_comparison(root, comparison)
    comparison_still = deliverables / "task01_dust_final_head_comparison_2x2.png"
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-ss",
            "8",
            "-i",
            str(comparison),
            "-frames:v",
            "1",
            str(comparison_still),
        ],
        check=True,
    )

    copied = {}
    for variant in ("final_train_medium", "final_cinematic"):
        source = root / variant / "episode_000"
        copied[variant] = {}
        for name in (*CAMERA_FILES, "three_camera_grid.mp4"):
            short = name.replace("observation.images.", "").replace(".mp4", "")
            destination = deliverables / f"{variant}_{short}.mp4"
            shutil.copy2(source / name, destination)
            copied[variant][name] = str(destination)

    hashes_unchanged = (
        (root / "source_hashes.before.sha256").read_text()
        == (root / "source_hashes.after.sha256").read_text()
    )
    medium = variants["final_train_medium"]["visibility"]
    cinematic = variants["final_cinematic"]["visibility"]
    all_videos = [
        video
        for item in variants.values()
        for video in item["videos"].values()
    ]
    peak_memory = max(
        item["gpu"].get("gpu_memory_peak_mib", 0.0)
        for item in variants.values()
    )
    minimum_fps = min(
        float(item["manifest"]["effective_replay_fps"])
        for item in variants.values()
    )
    acceptance = {
        "source_hashes_unchanged": hashes_unchanged,
        "all_videos_h264_yuv420p": all(
            item["codec"] == "h264" and item["pixel_format"] == "yuv420p"
            for item in all_videos
        ),
        "all_camera_videos_1004_frames": all(
            item["frames"] == 1004
            for variant in variants.values()
            for name, item in variant["videos"].items()
            if name in CAMERA_FILES
        ),
        "all_videos_faststart": all(item["faststart"] for item in all_videos),
        "train_medium_coverage_at_least_70_percent": (
            medium["global_coverage_fraction"] >= 0.70
        ),
        "train_medium_background_contrast_in_target": (
            0.82 <= medium["background_contrast_ratio"] <= 0.90
        ),
        "train_medium_tabletop_contrast_at_least_80_percent": (
            medium["tabletop_contrast_ratio_mean"] >= 0.80
        ),
        "train_medium_low_visibility_proxy_at_most_5_percent": (
            medium["tabletop_below_80_percent_fraction"] <= 0.05
        ),
        "cinematic_background_contrast_in_target": (
            0.75 <= cinematic["background_contrast_ratio"] <= 0.84
        ),
        "peak_gpu_memory_at_most_14_5_gib": peak_memory <= 14848,
        "effective_replay_fps_at_least_20": minimum_fps >= 20.0,
    }
    report = {
        "source_dataset": variants["clean"]["manifest"]["source_dataset"],
        "episode": 0,
        "profiles": {
            "training_default": {
                "mode": "flow",
                "profile": "train",
                "preset": "medium",
                "ambient_multiplier": 1.20,
            },
            "cinematic": {
                "mode": "flow",
                "profile": "cinematic",
                "preset": "medium",
                "ambient_multiplier": 1.00,
            },
        },
        "flow_implementation": {
            "extension": "omni.flowusd 110.0.3",
            "emitters": 2,
            "shared_layer": 0,
            "density_cell_size_m": 0.06,
            "solver": "local NVIDIA Fire hierarchy restyled as gray-brown dust",
            "strict_smoke_only": False,
            "visual_approximation_disclosed": True,
            "object_reset_seed": 260822,
            "dust_seed": 260822,
        },
        "variants": variants,
        "summary": {
            "peak_gpu_memory_mib": peak_memory,
            "minimum_effective_replay_fps": minimum_fps,
        },
        "acceptance": acceptance,
        "all_acceptance_checks_passed": all(acceptance.values()),
        "deliverables": {
            "comparison_video": str(comparison),
            "comparison_still": str(comparison_still),
            "selected_profile_videos": copied,
        },
        "limitations": [
            "Visibility and color checks are image proxies, not object detection.",
            "The dataset has no per-frame object/rope pose; this is a fixed-seed visual replay.",
            "The Flow 110 headless renderer uses the verified local Fire solver hierarchy as a visual compatibility fallback.",
        ],
    }
    report_path = deliverables / "final_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    readme = f"""# Task01 final full-scene dust POC

- Training default: `G1_DUST_MODE=flow G1_DUST_PROFILE=train G1_DUST_PRESET=medium`
- Training ambient multiplier: `1.20`
- Cinematic profile ambient multiplier: `1.00`
- Train-medium background detail contrast: `{medium['background_contrast_ratio']:.3f}`
- Train-medium grid coverage: `{medium['global_coverage_fraction']:.1%}`
- Train-medium tabletop detail contrast: `{medium['tabletop_contrast_ratio_mean']:.3f}`
- Cinematic background detail contrast: `{cinematic['background_contrast_ratio']:.3f}`
- Peak observed GPU memory: `{peak_memory:.1f} MiB`
- Minimum effective replay FPS: `{minimum_fps:.2f}`
- Source metadata/data hashes unchanged: `{hashes_unchanged}`
- All automated acceptance checks passed: `{all(acceptance.values())}`

This is a visual POC only. It does not modify or append to the training dataset.
The Flow cloud is a disclosed visual approximation based on the locally verified
NVIDIA Fire solver hierarchy, restyled to gray-brown dust with no visible flame.
See `final_report.json` for all variants, video probes, metrics and limitations.
"""
    (deliverables / "README.md").write_text(readme, encoding="utf-8")
    print(json.dumps(report["acceptance"], indent=2))
    if not all(acceptance.values()):
        failed = [key for key, passed in acceptance.items() if not passed]
        raise SystemExit(f"Acceptance failed: {failed}")


if __name__ == "__main__":
    main()
