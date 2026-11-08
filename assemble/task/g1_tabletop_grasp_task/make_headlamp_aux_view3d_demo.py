#!/usr/bin/env python3
"""Assemble and validate the first-person 3D-lighting demo."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

import cv2
import numpy as np


VARIANTS = ("headlamp_only", "auxiliary_only", "headlamp_plus_auxiliary")


def probe(path: Path) -> dict:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=codec_name,width,height,pix_fmt,nb_frames,r_frame_rate",
         "-show_entries", "format=duration", "-of", "json", str(path)],
        check=True, text=True, capture_output=True,
    )
    return json.loads(result.stdout)


def faststart(path: Path) -> bool:
    data = path.read_bytes()
    return 0 <= data.find(b"moov") < data.find(b"mdat")


def frame_at(path: Path, index: int = 30) -> np.ndarray:
    cap = cv2.VideoCapture(str(path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Could not read frame {index} from {path}")
    return frame


def hotspot_centres(frame: np.ndarray) -> list[tuple[int, int]]:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
    warmth = np.clip(frame[:, :, 2].astype(np.float32) - frame[:, :, 0], 0.0, None)
    score = cv2.GaussianBlur(gray + 0.5 * warmth, (61, 61), 0)
    points: list[tuple[int, int]] = []
    for _ in range(3):
        y, x = np.unravel_index(np.argmax(score), score.shape)
        if score[y, x] <= 0:
            break
        points.append((int(x), int(y)))
        cv2.circle(score, (int(x), int(y)), 55, -1.0, -1)
    return points


def make_contact_sheet(root: Path) -> None:
    tiles: list[np.ndarray] = []
    for episode in range(3):
        for variant in VARIANTS:
            video = root / "replays" / variant / f"episode_{episode:03d}" / "observation.images.head.mp4"
            frame = frame_at(video)
            label = f"EP{episode:02d} {variant}"
            cv2.rectangle(frame, (0, 0), (640, 32), (0, 0, 0), -1)
            cv2.putText(frame, label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX,
                        0.50, (255, 255, 255), 1, cv2.LINE_AA)
            tiles.append(cv2.resize(frame, (320, 240), interpolation=cv2.INTER_AREA))
    rows = [np.hstack(tiles[row * 3:(row + 1) * 3]) for row in range(3)]
    cv2.imwrite(str(root / "comparison_contact_sheet_3x3.png"), np.vstack(rows))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    rows = []
    all_format_ok = True
    all_counts_ok = True
    all_faststart = True
    assignments: dict[str, object] = {}

    for episode in range(3):
        aux_assignment = None
        for variant in VARIANTS:
            episode_root = root / "replays" / variant / f"episode_{episode:03d}"
            video = episode_root / "observation.images.head.mp4"
            manifest = json.loads((episode_root / "manifest.json").read_text())
            info = probe(video)
            stream = info["streams"][0]
            frames = int(stream.get("nb_frames", 0))
            expected = int(manifest["frames"])
            all_format_ok &= (
                stream.get("codec_name") == "h264"
                and stream.get("width") == 640
                and stream.get("height") == 480
                and stream.get("pix_fmt") == "yuv420p"
                and stream.get("r_frame_rate") == "30/1"
            )
            all_counts_ok &= frames == expected == int(manifest["source_episode_frames"])
            all_faststart &= faststart(video)
            row = {
                "episode_index": episode,
                "variant": variant,
                "frames": frames,
                "duration_seconds": float(info["format"]["duration"]),
                "effective_replay_fps": float(manifest["effective_replay_fps"]),
                "video": str(video),
                "hotspots_frame_30_xy": [list(point) for point in hotspot_centres(frame_at(video))],
            }
            rows.append(row)
            if variant == "auxiliary_only":
                assignment_path = episode_root / "aux_light_assignments.json"
                aux_assignment = json.loads(assignment_path.read_text())
                assignments[f"episode_{episode:03d}"] = aux_assignment
        if aux_assignment is None:
            raise RuntimeError(f"Missing auxiliary assignment for episode {episode}")

    target_sets = [
        json.dumps(value.get("lights", []), sort_keys=True)
        for value in assignments.values()  # type: ignore[union-attr]
    ]
    target_sets_unique = len(set(target_sets)) == 3
    all_fps_ok = min(row["effective_replay_fps"] for row in rows) >= 20.0
    hashes_equal = (root / "source_hashes.before.sha256").read_bytes() == (root / "source_hashes.after.sha256").read_bytes()
    report = {
        "effect": "task01_native_usd_headlamp_plus_view3d_auxiliary_spotlights",
        "source_dataset": str(args.dataset.resolve()),
        "variants": list(VARIANTS),
        "episodes": rows,
        "aux_light_assignments": assignments,
        "checks": {
            "three_episodes": len({row["episode_index"] for row in rows}) == 3,
            "three_variants_per_episode": len(rows) == 9,
            "targets_differ_between_episodes": target_sets_unique,
            "all_head_videos_h264_640x480_yuv420p_30fps": all_format_ok,
            "all_frame_counts_match_source": all_counts_ok,
            "all_videos_faststart": all_faststart,
            "minimum_effective_replay_fps_at_least_20": all_fps_ok,
            "dataset_hashes_unchanged": hashes_equal,
        },
        "rgb_postprocess_used_for_lighting": False,
        "dataset_modified": False,
        "formal_training_dataset_ready": False,
    }
    (root / "aux_light_assignments.json").write_text(json.dumps(assignments, indent=2) + "\n")
    make_contact_sheet(root)
    (root / "demo_report.json").write_text(json.dumps(report, indent=2) + "\n")
    (root / "README.md").write_text(
        "# Task01 第一人称视野内随机辅助光源 Demo\n\n"
        "使用 replay_ready 轨迹，固定辅助灯位置，在头部相机视野内为每个 episode 生成不同的 3D 目标点。\n\n"
        f"- 全部验收通过：`{all(report['checks'].values())}`\n"
        "- 仅为视觉 Replay Demo，未写入正式训练数据集。\n",
        encoding="utf-8",
    )
    if not all(report["checks"].values()):
        raise SystemExit(json.dumps(report["checks"], indent=2))


if __name__ == "__main__":
    main()
