#!/usr/bin/env python3
"""Validate and assemble the Task01 three-auxiliary-light replay set."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess

import cv2
import numpy as np


TABLE_POLYGON = np.array(
    [[105, 142], [495, 142], [555, 318], [482, 365], [145, 365], [82, 318]],
    dtype=np.int32,
)


def probe(path: Path) -> dict:
    result = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries",
            "stream=codec_name,width,height,pix_fmt,nb_frames,r_frame_rate",
            "-show_entries", "format=duration", "-of", "json", str(path),
        ], check=True, text=True, capture_output=True,
    )
    return json.loads(result.stdout)


def faststart(path: Path) -> bool:
    payload = path.read_bytes()
    return 0 <= payload.find(b"moov") < payload.find(b"mdat")


def peak_gpu_memory(path: Path) -> float:
    values = []
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


def estimate_hotspots(frame: np.ndarray, count: int = 3) -> list[tuple[int, int]]:
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [TABLE_POLYGON], 255)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
    warmth = np.clip(frame[:, :, 2].astype(np.float32) - frame[:, :, 0], 0.0, None)
    score = cv2.GaussianBlur(gray + 0.5 * warmth, (81, 81), 0)
    score[mask == 0] = -1.0
    points = []
    for _ in range(count):
        y, x = np.unravel_index(np.argmax(score), score.shape)
        if score[y, x] < 0:
            break
        points.append((int(x), int(y)))
        cv2.circle(score, (int(x), int(y)), 65, -1.0, -1)
    return points


def make_contact_sheet(rows: list[dict], output: Path) -> None:
    tiles = []
    for row in rows:
        frame = frame_at(Path(row["video"]), 30)
        cv2.polylines(frame, [TABLE_POLYGON], True, (60, 230, 60), 2, cv2.LINE_AA)
        for index, (x, y) in enumerate(row["hotspots_frame_30_xy"]):
            cv2.drawMarker(frame, (x, y), (60, 230, 60), cv2.MARKER_CROSS, 18, 2)
            cv2.putText(frame, str(index + 1), (x + 8, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (60, 230, 60), 1, cv2.LINE_AA)
        label = f"EP{int(row['episode_index']):02d}  3 fixed lights  seed+{int(row['episode_index'])}"
        cv2.rectangle(frame, (0, 0), (640, 34), (0, 0, 0), -1)
        cv2.putText(frame, label, (8, 23), cv2.FONT_HERSHEY_SIMPLEX,
                    0.48, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(cv2.resize(frame, (320, 240), interpolation=cv2.INTER_AREA))
    sheet = np.vstack((np.hstack(tiles[:5]), np.hstack(tiles[5:])))
    if not cv2.imwrite(str(output), sheet):
        raise RuntimeError(f"Could not write {output}")


def make_montage(paths: list[Path], output: Path) -> None:
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y"]
    for path in paths:
        command.extend(["-i", str(path)])
    filters = [f"[{i}:v]scale=320:240[v{i}]" for i in range(10)]
    inputs = "".join(f"[v{i}]" for i in range(10))
    layout = "|".join([f"{320 * i}_0" for i in range(5)] + [f"{320 * i}_240" for i in range(5)])
    filters.append(f"{inputs}xstack=inputs=10:layout={layout}[out]")
    command.extend([
        "-filter_complex", ";".join(filters), "-map", "[out]", "-t", "10", "-an",
        "-c:v", "libx264", "-crf", "22", "-preset", "veryfast",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output),
    ])
    subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    export_root = root / "head_videos"
    export_root.mkdir(parents=True, exist_ok=True)
    rows = []
    format_ok = counts_ok = faststart_ok = logs_ok = True
    fps_values = []
    target_sets = []
    exported_paths = []
    first_parameters = {}

    for episode in range(10):
        episode_root = root / "replays" / f"episode_{episode:03d}"
        video = episode_root / "observation.images.head.mp4"
        manifest = json.loads((episode_root / "manifest.json").read_text(encoding="utf-8"))
        assignment = json.loads((episode_root / "aux_light_assignments.json").read_text(encoding="utf-8"))
        if not first_parameters:
            first_parameters = dict(assignment.get("parameters", {}))
        targets = [tuple(light["target_local_m"][:2]) for light in assignment["lights"]]
        target_sets.append(targets)
        target_distances = [
            float(np.linalg.norm(np.array(a) - np.array(b)))
            for i, a in enumerate(targets) for b in targets[i + 1:]
        ]
        info = probe(video)
        stream = info["streams"][0]
        frames = int(stream.get("nb_frames", 0))
        expected = int(manifest["frames"])
        format_ok &= (
            stream.get("codec_name") == "h264" and stream.get("width") == 640
            and stream.get("height") == 480 and stream.get("pix_fmt") == "yuv420p"
            and stream.get("r_frame_rate") == "30/1"
        )
        counts_ok &= frames == expected == int(manifest["source_episode_frames"])
        faststart_ok &= faststart(video)
        log_text = (episode_root / "isaacsim.log").read_text(errors="replace")
        logs_ok &= (
            "[aux-lights] Pre-authored 3 fixed spotlights" in log_text
            and log_text.find("[aux-lights] Pre-authored") < log_text.find("[headlamp] Head camera auto exposure")
        )
        fps = float(manifest["effective_replay_fps"])
        fps_values.append(fps)
        exported = export_root / f"episode_{episode:03d}__aux3.mp4"
        shutil.copy2(video, exported)
        exported_paths.append(exported)
        hotspots = estimate_hotspots(frame_at(video, 30))
        rows.append({
            "episode_index": episode,
            "seed": int(assignment["seed"]),
            "target_local_points_m": [list(p) for p in targets],
            "minimum_target_separation_m": min(target_distances),
            "frames": frames,
            "duration_seconds": float(info["format"]["duration"]),
            "effective_replay_fps": fps,
            "video": str(exported),
            "hotspots_frame_30_xy": [list(p) for p in hotspots],
            "three_hotspots_detected": len(hotspots) == 3,
        })

    all_target_sets_unique = len({json.dumps(x, sort_keys=True) for x in target_sets}) == 10
    min_separation = min(float(row["minimum_target_separation_m"]) for row in rows)
    hashes_equal = (root / "source_hashes.before.sha256").read_bytes() == (root / "source_hashes.after.sha256").read_bytes()
    checks = {
        "ten_episode_assignments": len(rows) == 10,
        "three_targets_per_episode": all(len(row["target_local_points_m"]) == 3 for row in rows),
        "targets_differ_between_episodes": all_target_sets_unique,
        "targets_inside_calibrated_table_interior": all(
            all(-0.55 <= p[0] <= 0.55 and -0.34 <= p[1] <= 0.30 for p in row["target_local_points_m"])
            for row in rows
        ),
        "minimum_target_separation_at_least_0_35m": min_separation >= 0.35,
        "three_hotspots_detected_in_all_frame30": all(row["three_hotspots_detected"] for row in rows),
        "all_head_videos_h264_640x480_yuv420p_30fps": format_ok,
        "all_frame_counts_match_source_episodes": counts_ok,
        "all_head_videos_faststart": faststart_ok,
        "aux_lights_pre_authored_before_camera": logs_ok,
        "dataset_hashes_unchanged": hashes_equal,
        "minimum_effective_replay_fps_at_least_20": min(fps_values) >= 20.0,
        "peak_gpu_memory_below_14500_mib": peak_gpu_memory(root / "gpu.csv") < 14500.0,
    }
    ordered = sorted(rows, key=lambda row: row["episode_index"])
    make_contact_sheet(ordered, root / "spot_contact_sheet_2x5.png")
    make_montage(exported_paths, root / "spot_montage_10episodes_first10s.mp4")
    report = {
        "effect": "task01_native_usd_headlamp_plus_three_fixed_auxiliary_spotlights",
        "source_dataset": str(args.dataset.resolve()),
        "parameters": {
            "dust_mode": "off",
            "headlamp_intensity": 30000000.0,
            "auxiliary_light_count": 3,
            "auxiliary_light_profile": "distinct",
            "auxiliary_light_intensity": float(first_parameters.get("intensity", 12000000.0)),
            "auxiliary_light_seed": int(rows[0]["seed"]),
            "auxiliary_cone_angle_deg": float(first_parameters.get("cone_angle_deg", 20.0)),
            "auxiliary_cone_softness": float(first_parameters.get("cone_softness", 0.25)),
            "auxiliary_focus": float(first_parameters.get("focus", 4.0)),
            "color_temperature_k": float(first_parameters.get("color_temperature_k", 4100.0)),
            "dome_intensity": 80.0,
            "nurec_emissive_scale": 0.16,
            "object_reset_seed": 260822,
        },
        "episodes": rows,
        "minimum_target_separation_m": min_separation,
        "minimum_effective_replay_fps": min(fps_values),
        "peak_gpu_memory_mib": peak_gpu_memory(root / "gpu.csv"),
        "checks": checks,
        "all_checks_passed": all(checks.values()),
        "rgb_postprocess_used_for_lighting": False,
        "dataset_modified": False,
        "formal_training_dataset_ready": False,
    }
    (root / "replay10_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (root / "README.md").write_text(
        "# Task01 三辅助光源十条 Replay\n\n"
        "保留原生 USD 头灯，并在桌面周围增加三个固定场景聚光灯。"
        "每条 episode 使用 seed+episode 生成三个固定、分离的桌面照射目标。\n\n"
        f"- 最低有效回放速度：`{min(fps_values):.2f} FPS`\n"
        f"- 峰值显存：`{peak_gpu_memory(root / 'gpu.csv'):.0f} MiB`\n"
        f"- 全部验收通过：`{all(checks.values())}`\n"
        "- 扬尘关闭；原始数据集未修改；这是视觉 replay POC。\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
