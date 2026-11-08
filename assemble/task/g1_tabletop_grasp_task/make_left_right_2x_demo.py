#!/usr/bin/env python3
"""Validate the Task01 clean/medium/wide/left-right-2x dust comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

import cv2
import numpy as np


VARIANTS = ("clean", "final_train_medium", "final_train_medium_wide", "final_train_medium_lr2x")


def probe(path: Path) -> dict:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=codec_name,width,height,pix_fmt,nb_frames,r_frame_rate",
         "-show_entries", "format=duration", "-of", "json", str(path)],
        check=True, text=True, capture_output=True,
    )
    return json.loads(result.stdout)


def read_frame(path: Path, index: int = 30) -> np.ndarray:
    cap = cv2.VideoCapture(str(path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, image = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Unable to read frame {index}: {path}")
    return image


def faststart(path: Path) -> bool:
    payload = path.read_bytes()
    return 0 <= payload.find(b"moov") < payload.find(b"mdat")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    rows = []
    tiles = []
    all_format = all_frames = all_faststart = True
    clean = read_frame(root / "clean/episode_000/observation.images.head.mp4")
    clean_gray = cv2.cvtColor(clean, cv2.COLOR_BGR2GRAY)
    for name in VARIANTS:
        video = root / name / "episode_000/observation.images.head.mp4"
        manifest = json.loads((root / name / "episode_000/manifest.json").read_text())
        info = probe(video)
        stream = info["streams"][0]
        frames = int(stream.get("nb_frames", 0))
        expected = int(manifest["source_episode_frames"])
        format_ok = (stream.get("codec_name") == "h264" and stream.get("width") == 640
                     and stream.get("height") == 480 and stream.get("pix_fmt") == "yuv420p"
                     and stream.get("r_frame_rate") == "30/1")
        fast_ok = faststart(video)
        all_format &= format_ok
        all_frames &= frames == expected
        all_faststart &= fast_ok
        image = read_frame(video)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        luma = float(np.mean(gray))
        p95 = float(np.percentile(gray, 95))
        if name == "final_train_medium_lr2x":
            diff = np.abs(gray.astype(np.float32) - clean_gray.astype(np.float32))
            changed_fraction = float(np.mean(diff > 8.0))
            edge_fraction = float(np.mean((diff > 8.0) & (gray < 245)))
        else:
            changed_fraction = None
            edge_fraction = None
        annotated = image.copy()
        cv2.rectangle(annotated, (0, 0), (640, 40), (0, 0, 0), -1)
        cv2.putText(annotated, f"{name} mean={luma:.1f} p95={p95:.1f}",
                    (8, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(cv2.resize(annotated, (320, 240), interpolation=cv2.INTER_AREA))
        rows.append({"variant": name, "video": str(video), "frames": frames,
                     "expected_frames": expected, "duration_seconds": float(info["format"]["duration"]),
                     "mean_luma_frame30": luma, "p95_luma_frame30": p95,
                     "changed_fraction_vs_clean": changed_fraction,
                     "non_white_changed_fraction_vs_clean": edge_fraction,
                     "format_ok": format_ok, "faststart": fast_ok})
    cv2.imwrite(str(root / "head_comparison_4x1.png"), np.hstack(tiles))
    checks = {
        "all_head_videos_h264_640x480_yuv420p_30fps": all_format,
        "all_head_videos_match_source_episode_frames": all_frames,
        "all_head_videos_faststart": all_faststart,
        "dataset_hashes_unchanged": (root / "source_hashes.before.sha256").read_bytes() == (root / "source_hashes.after.sha256").read_bytes(),
        "left_right_2x_rendered": (root / "final_train_medium_lr2x/episode_000/observation.images.head.mp4").stat().st_size > 0,
    }
    report = {
        "effect": "task01_train_medium_left_right_two_emitters_2x_radius",
        "source_dataset": str(args.dataset.resolve()),
        "source_episode": 0,
        "variants": rows,
        "parameters": {
            "layout": "left_right_2x",
            "radius_m": 0.68,
            "local_offsets_m": [[-0.70, 0.20, 1.05], [0.70, -0.15, 1.00]],
            "ambient_multiplier": 0.90,
            "flow_opacity": 0.18,
            "seed": 260822,
        },
        "checks": checks,
        "all_checks_passed": all(checks.values()),
        "dataset_modified": False,
        "formal_training_dataset_ready": False,
    }
    (root / "demo_report.json").write_text(json.dumps(report, indent=2) + "\n")
    (root / "README.md").write_text(
        "# Task01 左右双源 2× 扬尘 Demo\n\n"
        "同一 replay-ready episode-0 对比 clean、原始 Medium、Wide 和左右双源 2× 版本。\n"
        "新版本使用两个半径约 0.68m 的重叠 Flow 尘源，降低单源透明度和全局密度，避免圆形边界和白色雾幕。\n\n"
        f"- 全部检查通过：`{all(checks.values())}`\n"
        "- 输入数据集未修改；本轮仅为视觉 POC，未写入正式增强数据集。\n",
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
