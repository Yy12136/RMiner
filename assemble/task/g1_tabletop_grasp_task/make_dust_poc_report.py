#!/usr/bin/env python3
"""Build Task01 dust POC comparison videos and a compact metrics report."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import shutil
import subprocess

import cv2
import numpy as np


VARIANTS = ("clean", "legacy", "flow_low", "flow_medium", "flow_heavy")


def contrast_visibility(clean_path: Path, candidate_path: Path) -> dict:
    clean = cv2.VideoCapture(str(clean_path))
    candidate = cv2.VideoCapture(str(candidate_path))
    compared = 0
    obscured = 0
    ratios: list[float] = []
    try:
        while True:
            ok_clean, clean_frame = clean.read()
            ok_candidate, candidate_frame = candidate.read()
            if not ok_clean or not ok_candidate:
                break
            if compared % 3 == 0:
                height, width = clean_frame.shape[:2]
                y0, y1 = int(0.23 * height), int(0.88 * height)
                x0, x1 = int(0.12 * width), int(0.88 * width)
                clean_gray = cv2.cvtColor(
                    clean_frame[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY
                )
                candidate_gray = cv2.cvtColor(
                    candidate_frame[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY
                )
                clean_contrast = float(clean_gray.std())
                candidate_contrast = float(candidate_gray.std())
                ratio = candidate_contrast / max(clean_contrast, 1.0e-6)
                ratios.append(ratio)
                if ratio < 0.55:
                    obscured += 1
            compared += 1
    finally:
        clean.release()
        candidate.release()
    if not ratios:
        raise RuntimeError(f"No comparable frames: {clean_path} vs {candidate_path}")
    return {
        "sampled_frames": len(ratios),
        "obscured_frames": obscured,
        "obscured_fraction": obscured / len(ratios),
        "mean_tabletop_contrast_ratio": float(np.mean(ratios)),
        "minimum_tabletop_contrast_ratio": float(np.min(ratios)),
    }


def read_gpu_csv(path: Path) -> dict:
    rows = []
    if path.is_file():
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.reader(handle):
                if len(row) < 3:
                    continue
                try:
                    rows.append((float(row[-2].strip()), float(row[-1].strip())))
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


def build_comparison(root: Path, output: Path) -> None:
    inputs = [
        root / "clean/episode_000/observation.images.head.mp4",
        root / "legacy/episode_000/observation.images.head.mp4",
        root / "flow_low/episode_000/observation.images.head.mp4",
        root / "flow_medium/episode_000/observation.images.head.mp4",
    ]
    labels = ("Clean", "Legacy RTX + USD Points", "Flow Low", "Flow Medium")
    filters = []
    for index, label in enumerate(labels):
        filters.append(
            f"[{index}:v]drawtext=text='{label}':x=12:y=12:fontsize=22:"
            "fontcolor=white:box=1:boxcolor=black@0.60[v%d]" % index
        )
    filters.append(
        "[v0][v1][v2][v3]xstack=inputs=4:"
        "layout=0_0|w0_0|0_h0|w0_h0[out]"
    )
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y"]
    for item in inputs:
        command.extend(["-i", str(item)])
    command.extend(
        [
            "-filter_complex",
            ";".join(filters),
            "-map",
            "[out]",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
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


def build_comparison_still(video: Path, output: Path) -> None:
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-ss",
            "3",
            "-i",
            str(video),
            "-frames:v",
            "1",
            str(output),
        ],
        check=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.expanduser().resolve()
    deliverables = root / "deliverables"
    deliverables.mkdir(parents=True, exist_ok=True)

    clean_head = root / "clean/episode_000/observation.images.head.mp4"
    visibility = {
        preset: contrast_visibility(
            clean_head,
            root / f"flow_{preset}/episode_000/observation.images.head.mp4",
        )
        for preset in ("low", "medium", "heavy")
    }
    best = "flow_medium"
    if visibility["medium"]["obscured_fraction"] > 0.20:
        best = "flow_low"

    comparison = deliverables / "task01_dust_head_comparison_2x2.mp4"
    build_comparison(root, comparison)
    comparison_still = deliverables / "task01_dust_head_comparison_2x2.png"
    build_comparison_still(comparison, comparison_still)
    best_source = root / best / "episode_000"
    copied = {}
    for name in (
        "observation.images.head.mp4",
        "observation.images.left.mp4",
        "observation.images.right.mp4",
        "three_camera_grid.mp4",
    ):
        destination = deliverables / f"best_{best}_{name}"
        shutil.copy2(best_source / name, destination)
        copied[name] = str(destination)

    variants = {}
    for variant in VARIANTS:
        variant_dir = root / variant / "episode_000"
        manifest = json.loads((variant_dir / "manifest.json").read_text())
        variants[variant] = {
            "effective_replay_fps": manifest["effective_replay_fps"],
            "replay_wall_seconds": manifest["replay_wall_seconds"],
            "gpu": read_gpu_csv(root / variant / "gpu.csv"),
        }

    hashes_before = (root / "source_hashes.before.sha256").read_text()
    hashes_after = (root / "source_hashes.after.sha256").read_text()
    report = {
        "episode": 0,
        "source_hashes_unchanged": hashes_before == hashes_after,
        "best_flow_variant": best,
        "selection_rule": (
            "flow_medium unless central-tabletop contrast is below 55% of "
            "clean in more than 20% of sampled frames"
        ),
        "visibility": visibility,
        "variants": variants,
        "flow_configuration": {
            "extension": "omni.flowusd 110.0.3",
            "density_cell_size_m": 0.06,
            "solver": "local NVIDIA Fire preset with neutral-gray dust shading",
            "smoke_only_requested_but_not_renderable": True,
            "emission_channels": {
                "fuel_temperature_combustion_fallback": True,
                "rendered_as_fire": False,
            },
            "shared_layer": 0,
            "object_reset_seed": 260822,
            "dust_seed": 260822,
        },
        "deliverables": {
            "head_comparison_2x2": str(comparison),
            "head_comparison_2x2_still": str(comparison_still),
            **copied,
        },
    }
    report_path = deliverables / "poc_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    peak_memory = max(
        item["gpu"].get("gpu_memory_peak_mib", 0.0)
        for item in variants.values()
    )
    minimum_fps = min(item["effective_replay_fps"] for item in variants.values())
    readme = f"""# Task01 NVIDIA Flow dust POC

- Episode: 0
- Selected Flow preset: `{best}`
- Peak observed GPU memory: `{peak_memory:.1f} MiB`
- Minimum effective replay FPS: `{minimum_fps:.2f}`
- Source dataset hashes unchanged: `{hashes_before == hashes_after}`
- Flow compatibility fallback: local Fire solver, restyled as neutral-gray dust
- Source dataset was replayed read-only; no source MP4 was decoded or edited.
- The dataset lacks per-frame object/rope/base transforms, so this is a fixed-seed visual replay rather than a pixel-identical reconstruction.

See `poc_report.json` for per-variant metrics and the automated visibility proxy.
"""
    (deliverables / "README.md").write_text(readme, encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
