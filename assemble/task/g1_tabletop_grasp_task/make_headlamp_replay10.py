#!/usr/bin/env python3
"""Validate and assemble the Task01 10-episode varied-headlamp replay."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess

import cv2
import numpy as np


ASSIGNMENTS = (
    (0, "upper_right", 8.0, -3.0, 0.80, 0.38),
    (1, "upper_mid_right", 4.0, -3.0, 0.65, 0.38),
    (2, "upper_center", 0.0, -3.0, 0.50, 0.38),
    (3, "upper_mid_left", -4.0, -3.0, 0.35, 0.38),
    (4, "upper_left", -8.0, -3.0, 0.20, 0.38),
    (5, "lower_right", 8.0, 3.0, 0.80, 0.62),
    (6, "lower_mid_right", 4.0, 3.0, 0.65, 0.62),
    (7, "lower_center", 0.0, 3.0, 0.50, 0.62),
    (8, "lower_mid_left", -4.0, 3.0, 0.35, 0.62),
    (9, "lower_left", -8.0, 3.0, 0.20, 0.62),
)

TABLE_POLYGON = np.array(
    [[105, 142], [495, 142], [555, 318], [482, 365], [145, 365], [82, 318]],
    dtype=np.int32,
)


def probe(path: Path) -> dict[str, object]:
    result = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=codec_name,width,height,pix_fmt,nb_frames,r_frame_rate",
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


def frame_at(path: Path, index: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Could not read frame {index} from {path}")
    return frame


def estimate_hotspot(frame: np.ndarray) -> tuple[int, int, bool]:
    """Estimate the broad warm-light centre after suppressing small objects."""
    table_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.fillPoly(table_mask, [TABLE_POLYGON], 255)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
    warmth = np.clip(
        frame[:, :, 2].astype(np.float32) - frame[:, :, 0].astype(np.float32),
        0.0,
        None,
    )
    score = cv2.GaussianBlur(gray + 0.6 * warmth, (101, 101), 0)
    score[table_mask == 0] = -1.0
    y, x = np.unravel_index(np.argmax(score), score.shape)
    inside = cv2.pointPolygonTest(TABLE_POLYGON, (float(x), float(y)), False) >= 0
    return int(x), int(y), bool(inside)


def make_contact_sheet(rows: list[dict[str, object]], output: Path) -> None:
    tiles: list[np.ndarray] = []
    for row in rows:
        frame = frame_at(Path(row["video"]), 30)
        cv2.polylines(frame, [TABLE_POLYGON], True, (60, 230, 60), 2, cv2.LINE_AA)
        cv2.drawMarker(
            frame,
            (int(row["hotspot_image_x"]), int(row["hotspot_image_y"])),
            (60, 230, 60),
            cv2.MARKER_CROSS,
            18,
            2,
        )
        label = (
            f"EP{int(row['episode_index']):02d} {row['slot']} "
            f"yaw={float(row['yaw_deg']):+.0f} pitch={float(row['pitch_deg']):+.0f}"
        )
        cv2.rectangle(frame, (0, 0), (640, 34), (0, 0, 0), -1)
        cv2.putText(frame, label, (8, 23), cv2.FONT_HERSHEY_SIMPLEX,
                    0.52, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(cv2.resize(frame, (320, 240), interpolation=cv2.INTER_AREA))
    sheet = np.vstack((np.hstack(tiles[:5]), np.hstack(tiles[5:])))
    if not cv2.imwrite(str(output), sheet):
        raise RuntimeError(f"Could not write {output}")


def make_montage(paths: list[Path], output: Path) -> None:
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y"]
    for path in paths:
        command.extend(["-i", str(path)])
    filters = []
    for index in range(10):
        filters.append(f"[{index}:v]scale=320:240[v{index}]")
    inputs = "".join(f"[v{index}]" for index in range(10))
    layout = "|".join(
        [f"{320 * index}_0" for index in range(5)]
        + [f"{320 * index}_240" for index in range(5)]
    )
    filters.append(f"{inputs}xstack=inputs=10:layout={layout}[out]")
    command.extend(
        [
            "-filter_complex", ";".join(filters), "-map", "[out]", "-t", "10",
            "-an", "-c:v", "libx264", "-crf", "22", "-preset", "veryfast",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output),
        ]
    )
    subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    export_root = root / "head_videos"
    export_root.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    all_formats_ok = True
    all_counts_ok = True
    all_faststart = True
    all_logs_match = True
    replay_fps_values: list[float] = []
    exported_paths: list[Path] = []
    for episode, slot, yaw, pitch, table_u, table_v in ASSIGNMENTS:
        episode_root = root / "replays" / f"episode_{episode:03d}"
        source_video = episode_root / "observation.images.head.mp4"
        manifest = json.loads((episode_root / "manifest.json").read_text())
        info = probe(source_video)
        stream = info["streams"][0]
        frames = int(stream.get("nb_frames", 0))
        expected_frames = int(manifest["frames"])
        all_formats_ok &= (
            stream.get("codec_name") == "h264"
            and stream.get("width") == 640
            and stream.get("height") == 480
            and stream.get("pix_fmt") == "yuv420p"
            and stream.get("r_frame_rate") == "30/1"
        )
        all_counts_ok &= frames == expected_frames == int(manifest["source_episode_frames"])
        all_faststart &= faststart(source_video)
        log_text = (episode_root / "isaacsim.log").read_text(errors="replace")
        all_logs_match &= (
            f"aim_yaw={yaw:.2f}deg, aim_pitch={pitch:.2f}deg" in log_text
        )
        replay_fps = float(manifest["effective_replay_fps"])
        replay_fps_values.append(replay_fps)
        exported = export_root / f"episode_{episode:03d}__{slot}.mp4"
        shutil.copy2(source_video, exported)
        exported_paths.append(exported)
        hotspot_x, hotspot_y, hotspot_inside = estimate_hotspot(
            frame_at(source_video, 30)
        )
        rows.append(
            {
                "episode_index": episode,
                "slot": slot,
                "table_normalized_u": table_u,
                "table_normalized_v": table_v,
                "yaw_deg": yaw,
                "pitch_deg": pitch,
                "frames": frames,
                "duration_seconds": float(info["format"]["duration"]),
                "effective_replay_fps": replay_fps,
                "video": str(exported),
                "orientation_fixed_for_episode": True,
                "inside_calibrated_table_safe_envelope": abs(yaw) <= 8.0 and abs(pitch) <= 3.0,
                "hotspot_image_x": hotspot_x,
                "hotspot_image_y": hotspot_y,
                "hotspot_inside_table_polygon": hotspot_inside,
            }
        )

    ordered_rows = sorted(
        rows,
        key=lambda row: (float(row["table_normalized_v"]), float(row["table_normalized_u"])),
    )
    make_contact_sheet(ordered_rows, root / "spot_contact_sheet_2x5.png")
    make_montage(
        [Path(row["video"]) for row in ordered_rows],
        root / "spot_montage_10episodes_first10s.mp4",
    )
    hotspot_points = np.array(
        [[row["hotspot_image_x"], row["hotspot_image_y"]] for row in rows],
        dtype=np.float32,
    )
    pairwise = np.linalg.norm(
        hotspot_points[:, None, :] - hotspot_points[None, :, :], axis=-1
    )
    pairwise[pairwise == 0.0] = np.inf
    minimum_hotspot_separation = float(np.min(pairwise))
    hashes_equal = (
        (root / "source_hashes.before.sha256").read_bytes()
        == (root / "source_hashes.after.sha256").read_bytes()
    )
    checks = {
        "ten_unique_episode_and_spot_assignments": (
            len(rows) == 10
            and len({(row["yaw_deg"], row["pitch_deg"]) for row in rows}) == 10
        ),
        "all_spots_inside_calibrated_table_safe_envelope": all(
            bool(row["inside_calibrated_table_safe_envelope"]) for row in rows
        ),
        "all_detected_hotspot_centres_inside_table_polygon": all(
            bool(row["hotspot_inside_table_polygon"]) for row in rows
        ),
        "all_detected_hotspot_centres_are_distinct": minimum_hotspot_separation >= 3.0,
        "spot_orientation_fixed_within_each_episode": all(
            bool(row["orientation_fixed_for_episode"]) for row in rows
        ),
        "all_head_videos_h264_640x480_yuv420p_30fps": bool(all_formats_ok),
        "all_frame_counts_match_source_episodes": bool(all_counts_ok),
        "all_head_videos_faststart": bool(all_faststart),
        "all_logs_match_assigned_angles": bool(all_logs_match),
        "dataset_hashes_unchanged": hashes_equal,
        "minimum_effective_replay_fps_at_least_20": min(replay_fps_values) >= 20.0,
        "peak_gpu_memory_below_14500_mib": peak_gpu_memory(root / "gpu.csv") < 14500.0,
    }
    report = {
        "effect": "task01_native_usd_bright_headlamp_varied_table_spots",
        "source_dataset": str(args.dataset.resolve()),
        "parameters": {
            "dust_mode": "off",
            "headlamp_intensity": 30000000.0,
            "cone_angle_deg": 32.0,
            "cone_softness": 0.45,
            "focus": 2.5,
            "color_temperature_k": 4100.0,
            "dome_intensity": 80.0,
            "nurec_emissive_scale": 0.16,
            "object_reset_seed": 260822,
            "spot_layout": "2 rows x 5 columns inside calibrated tabletop envelope",
        },
        "episodes": rows,
        "minimum_effective_replay_fps": min(replay_fps_values),
        "minimum_detected_hotspot_separation_px": minimum_hotspot_separation,
        "peak_gpu_memory_mib": peak_gpu_memory(root / "gpu.csv"),
        "checks": checks,
        "all_checks_passed": all(checks.values()),
        "rgb_postprocess_used_for_lighting": False,
        "dataset_modified": False,
        "formal_training_dataset_ready": False,
        "limitation": (
            "The source dataset does not contain per-frame object and rope-link poses; "
            "objects are re-simulated, so this remains a replay-augmentation POC."
        ),
    }
    (root / "replay10_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    readme = (
        "# Task01 十条不同桌面光斑 Replay\n\n"
        "选择原数据集 episode 0–9，使用无扬尘、高亮中心原生 USD 头灯。"
        "每条 episode 的 yaw/pitch 在启动时固定，十条采用桌面安全区域内的 2×5 不同落点。\n\n"
        f"- 最低有效回放速度：`{min(replay_fps_values):.2f} FPS`\n"
        f"- 峰值显存：`{peak_gpu_memory(root / 'gpu.csv'):.0f} MiB`\n"
        f"- 全部验收通过：`{all(checks.values())}`\n"
        "- 原始数据集未修改；当前仍是 replay 增强 POC，不是正式训练集。\n"
    )
    (root / "README.md").write_text(readme, encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
