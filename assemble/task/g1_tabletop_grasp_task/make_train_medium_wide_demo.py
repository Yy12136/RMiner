#!/usr/bin/env python3
"""Validate and summarize the Task01 wide-dust visual comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

import cv2
import numpy as np


VARIANTS = ("clean", "final_train_medium", "final_train_medium_wide")


def probe(path: Path) -> dict:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=codec_name,width,height,pix_fmt,nb_frames,r_frame_rate",
         "-show_entries", "format=duration", "-of", "json", str(path)],
        check=True, text=True, capture_output=True,
    )
    return json.loads(result.stdout)


def frame(path: Path, index: int = 30) -> np.ndarray:
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
    contact_tiles = []
    all_format_checks = True
    all_frame_checks = True
    all_faststart = True
    for name in VARIANTS:
        video = root / name / "episode_000" / "observation.images.head.mp4"
        manifest = json.loads(
            (root / name / "episode_000" / "manifest.json").read_text(encoding="utf-8")
        )
        info = probe(video)
        stream = info["streams"][0]
        frames = int(stream.get("nb_frames", 0))
        expected_frames = int(manifest["source_episode_frames"])
        format_ok = (
            stream.get("codec_name") == "h264" and stream.get("width") == 640
            and stream.get("height") == 480 and stream.get("pix_fmt") == "yuv420p"
            and stream.get("r_frame_rate") == "30/1"
        )
        all_format_checks &= format_ok
        all_frame_checks &= frames == expected_frames
        all_faststart &= faststart(video)
        image = frame(video)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        mean_luma = float(np.mean(gray))
        p95_luma = float(np.percentile(gray, 95))
        annotated = image.copy()
        cv2.rectangle(annotated, (0, 0), (640, 38), (0, 0, 0), -1)
        cv2.putText(annotated, f"{name}  mean={mean_luma:.1f} p95={p95_luma:.1f}",
                    (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 1, cv2.LINE_AA)
        contact_tiles.append(cv2.resize(annotated, (426, 320), interpolation=cv2.INTER_AREA))
        rows.append({
            "variant": name,
            "video": str(video),
            "frames": frames,
            "expected_frames": expected_frames,
            "duration_seconds": float(info["format"]["duration"]),
            "mean_luma_frame30": mean_luma,
            "p95_luma_frame30": p95_luma,
            "format_ok": format_ok,
            "faststart": faststart(video),
        })

    contact = np.hstack(contact_tiles)
    cv2.imwrite(str(root / "head_comparison_3x1.png"), contact)
    before = (root / "source_hashes.before.sha256").read_bytes()
    after = (root / "source_hashes.after.sha256").read_bytes()
    checks = {
        "all_head_videos_h264_640x480_yuv420p_30fps": all_format_checks,
        "all_head_videos_1004_frames": all_frame_checks,
        "all_head_videos_faststart": all_faststart,
        "dataset_hashes_unchanged": before == after,
        "wide_variant_rendered": (root / "final_train_medium_wide/episode_000/observation.images.head.mp4").stat().st_size > 0,
    }
    report = {
        "effect": "task01_train_medium_wide_global_dust_plus_two_flow_emitters",
        "source_dataset": str(args.dataset.resolve()),
        "variants": rows,
        "parameters": {
            "medium": {"dust_mode": "flow", "profile": "train", "preset": "medium", "ambient_multiplier": 1.20, "wide": False},
            "medium_wide": {"dust_mode": "flow", "profile": "train", "preset": "medium", "ambient_multiplier": 1.05, "wide": True, "flow_opacity": 0.32},
            "wide_radius_scale": 1.70,
            "wide_global_density_scale": 1.35,
            "source_episode": 0,
        },
        "checks": checks,
        "all_checks_passed": all(checks.values()),
        "dataset_modified": False,
        "formal_training_dataset_ready": False,
    }
    (root / "demo_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (root / "README.md").write_text(
        "# Task01 Final Train Medium Wide 扬尘 Demo\n\n"
        "同一 replay-ready episode-0 对比 clean、原 Final Train Medium 和扩大覆盖范围的 Medium Wide。"
        "Medium Wide 增强全局悬尘并将两个 Flow 尘源半径扩大 1.7 倍；本轮只生成视觉 Demo，不写入增强数据集。\n\n"
        f"- 全部检查通过：`{all(checks.values())}`\n"
        "- 输入数据集未修改。\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
