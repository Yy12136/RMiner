#!/usr/bin/env python3
"""Run resumable native-light replay augmentation for all six G1 tasks."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

import cv2
import numpy as np

TASK_ROOT = Path(os.environ.get("BENCH_ASSEMBLE_ROOT", str(Path(__file__).resolve().parents[2])) + "/task")
AUG_ROOT = TASK_ROOT / "augmentation"
REPLAY_SCRIPT = TASK_ROOT / "g1_tabletop_grasp_task/replay_lerobot_episode_with_3d_dust.py"
TV_PYTHON = Path(os.environ.get("TV_PYTHON") or (Path(os.environ.get("CONDA_PREFIX", "")) / "bin" / "python"))

TASKS = {
    "task01": ("g1_tabletop_grasp_task", "detonator_wire_insertion_replay_ready", "Isaac-TabletopGrasp-G129-Dex1-Joint"),
    "task02": ("g1_roof_bolt_sleeve_task", "roof_bolt_sleeve_installation_replay_ready", "Isaac-RoofBoltSleeve-G129-Dex1-Joint"),
    "task03": ("g1_bin_pick_place_task", "sam3d_metal_component_to_randomized_bin_replay_ready", "Isaac-BinPickPlace-G129-Dex1-Joint"),
    "task04": ("g1_hand_cart_push_task", "hand_cart_push_replay_ready", "Isaac-HandCartPush-G129-Dex1-Joint"),
    "task05": ("g1_valve_handwheel_task", "valve_handwheel_rotation_replay_ready", "Isaac-ValveHandwheel-G129-Dex1-Joint"),
    "task06": ("g1_wall_hole_detonator_task", "wall_hole_loaded_detonator_insertion_replay_ready", "Isaac-WallHoleDetonator-G129-Dex1-Joint"),
}
CASES = ("headlamp_only", "headlamp_aux0", "headlamp_aux1", "headlamp_aux2", "headlamp_aux123")
MAX_READY_ATTEMPTS = 3
TASK_AUX_INTENSITY = {
    "task01": 30_000_000,
    "task02": 100_000_000,
    "task03": 30_000_000,
    "task04": 100_000_000,
    "task05": 120_000_000,
    "task06": 120_000_000,
}
TASK_HEADLAMP_INTENSITY = {
    "task01": 10_000_000,
    "task02": 8_000_000,
    "task03": 10_000_000,
    "task04": 6_000_000,
    "task05": 7_000_000,
    "task06": 6_000_000,
}

sys.path.insert(0, str(AUG_ROOT))
import augment_lerobot_v3_video as v3  # noqa: E402


def sha256_tree(root: Path) -> dict[str, str]:
    result = {}
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        h = hashlib.sha256()
        with path.open("rb") as f:
            for block in iter(lambda: f.read(1024 * 1024), b""):
                h.update(block)
        result[str(path.relative_to(root))] = h.hexdigest()
    return result


def load_json(path: Path, default):
    return json.loads(path.read_text()) if path.exists() else default


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    os.replace(tmp, path)


def ensure_originals(spec, output: Path, progress: dict) -> None:
    writer = v3.create_writer(spec, output)
    try:
        if writer.total_episodes >= 50:
            return
        while writer.total_episodes < 50:
            source = writer.total_episodes
            row, states, actions = v3.source_data(spec, source)
            reader = v3.SourceVideoReader(spec, row)
            writer.start_episode()
            try:
                for i in range(len(actions)):
                    writer.add_frame(states[i], actions[i], reader.read())
                index, frames = writer.save_episode()
                if index != source or frames != len(actions):
                    raise RuntimeError(f"original commit mismatch: {index=} {frames=}")
            except Exception:
                writer.discard_episode()
                raise
            finally:
                reader.close()
            progress["originals_copied"] = writer.total_episodes
            write_json(output.parent / "progress.json", progress)
            print(f"[batch] copied original {source}/50 frames={len(actions)}", flush=True)
    finally:
        writer.close()


def wait_ready(path: Path, proc: subprocess.Popen, timeout: float = 180.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.exists():
            return
        # Some Isaac/AppLauncher builds detach the final Kit Python process and
        # let the shell-tracked bootstrap exit with status 0.  The final
        # process owns the ready marker and writes its exact PID into it.
        if proc.poll() is not None and proc.returncode != 0:
            raise RuntimeError(f"simulator exited before ready: rc={proc.returncode}")
        time.sleep(0.5)
    raise TimeoutError(f"timed out waiting for simulator ready marker {path}")


def replay_metadata_path(dataset: Path, episode_index: int) -> Path:
    matches = []
    for path in sorted((dataset / "meta/replay").glob("chunk-*/*.json")):
        payload = load_json(path, {})
        if int(payload.get("episode_index", -1)) == episode_index:
            matches.append(path)
    if len(matches) != 1:
        raise RuntimeError(
            f"expected exactly one replay metadata file for episode "
            f"{episode_index}, found {len(matches)}"
        )
    return matches[0]


def stop_process(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.wait(timeout=10)


def stop_exact_pid(pid: int | None) -> None:
    if pid is None or pid <= 1 or pid == os.getpid():
        return
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return
    children = subprocess.run(
        ["pgrep", "-P", str(pid)], capture_output=True, text=True
    ).stdout.split()
    for child in children:
        try:
            os.kill(int(child), signal.SIGTERM)
        except ProcessLookupError:
            pass
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.monotonic() + 15.0
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return
        time.sleep(0.2)
    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def simulator_pids_for_ready_file(path: Path) -> list[int]:
    """Find only simulator processes carrying this run's unique ready path."""

    needle = f"UNITREE_SIM_READY_FILE={path}".encode()
    result = []
    proc_root = Path("/proc")
    if not proc_root.is_dir():
        return result
    for entry in proc_root.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            environment = (entry / "environ").read_bytes().split(b"\0")
        except (FileNotFoundError, PermissionError, ProcessLookupError, OSError):
            continue
        if needle in environment:
            result.append(int(entry.name))
    return result


def replay_one(task_key: str, task_dir: Path, task_id: str, dataset: Path, source_episode: int, case: str, out_dir: Path, log_path: Path) -> dict:
    ready = Path(f"/tmp/lighting_{task_key}_{os.getpid()}_{source_episode}_{case}.ready")
    ready.unlink(missing_ok=True)
    initial_state_file = replay_metadata_path(dataset, source_episode)
    metadata = load_json(initial_state_file, {})
    metadata_task = str((metadata.get("sim_state") or {}).get("task_name", ""))
    if metadata_task != task_id:
        raise RuntimeError(
            f"replay metadata task mismatch: {metadata_task!r} != {task_id!r}"
        )
    aux_indices = {
        "headlamp_only": "",
        "headlamp_aux0": "0",
        "headlamp_aux1": "1",
        "headlamp_aux2": "2",
        "headlamp_aux123": "0,1,2",
    }[case]
    aux_action_path = ready.with_suffix(".aux-action.json")
    aux_action_path.unlink(missing_ok=True)
    env = os.environ.copy()
    env.update({
        "G1_SHOW_CAMERA_VIEWPORTS": "0",
        # subprocess.Popen(start_new_session=True) below is the sole process-
        # group owner.  A second setsid wrapper can exit before Isaac is ready
        # and orphan the real simulator process.
        "G1_PROCESS_GROUP_MANAGED": "1",
        "G1_SIM_READY_FILE": str(ready),
        "UNITREE_SIM_READY_FILE": str(ready),
        "G1_TASK_SUCCESS_FILE": str(ready.with_suffix(".success")),
        # Keep task-local demo controllers disabled.  The common replay
        # controller below is the only owner of lights across all six tasks.
        "G1_HEADLAMP": "off",
        "G1_AUX_LIGHTS": "off",
        "ISAAC_REPLAY_HEADLAMP": "on",
        "ISAAC_REPLAY_AUX_LIGHTS": "off" if not aux_indices else "on",
        "ISAAC_REPLAY_AUX_LIGHT_INDICES": aux_indices,
        "G1_DUST_MODE": "off",
        "G1_ENABLE_3D_DUST": "0",
        "G1_LIGHT_FLICKER": "off",
        "G1_AUX_LIGHT_SEED": "260904",
        "G1_AUX_LIGHT_EPISODE": str(source_episode),
        "G1_HEADLAMP_INTENSITY": str(TASK_HEADLAMP_INTENSITY[task_key]),
        "G1_HEADLAMP_AMBIENT_INTENSITY": "80",
        "G1_AUX_LIGHT_INTENSITY": str(TASK_AUX_INTENSITY[task_key]),
        "G1_AUX_LIGHT_CONE_ANGLE_DEG": "20",
        "G1_AUX_LIGHT_CONE_SOFTNESS": "0.28",
        "G1_AUX_LIGHT_FOCUS": "4.0",
        "G1_AUX_LIGHT_COLOR_TEMPERATURES": "3000,4100,6000",
        "G1_HEADLAMP_NUREC_EMISSIVE_SCALE": "0.14",
        "G1_REPLAY_AMBIENT_SCALE": "0.08",
        "G1_REPLAY_INITIAL_STATE_FILE": str(initial_state_file),
        "G1_REPLAY_AUX_ACTION_FILE": str(aux_action_path),
        "ISAAC_REPLAY_DISABLE_SUCCESS_FREEZE": "1",
        "ISAAC_REPLAY_LIGHTING_PROFILE": case,
        "ISAAC_REPLAY_EFFECTS_MODULE_FILE": str(AUG_ROOT / "replay_augmentation_effects.py"),
        "G1_AUX_LIGHT_ASSIGNMENT_FILE": str(out_dir / "aux_light_assignments.json"),
    })
    task_launcher = task_dir / "run_task.sh"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = log_path.open("w")
    sim = subprocess.Popen([str(task_launcher), "--headless"], env=env, stdout=log_file, stderr=subprocess.STDOUT, start_new_session=True)
    simulator_pid = None
    try:
        wait_ready(ready, sim)
        try:
            simulator_pid = int(ready.read_text(encoding="utf-8").strip())
        except (OSError, ValueError):
            simulator_pid = None
        replay_env = env.copy()
        replay_env["G1_DUST_RECORDING_MARKER"] = str(ready.with_suffix(".recording"))
        replay_command = [
            str(TV_PYTHON), str(REPLAY_SCRIPT), "--dataset", str(dataset),
            "--episode", str(source_episode), "--output-dir", str(out_dir),
            "--fps", "30",
        ]
        max_frames = os.environ.get("G1_REPLAY_MAX_FRAMES", "").strip()
        if max_frames:
            replay_command.extend(["--max-frames", max_frames])
        replay = subprocess.run(
            replay_command, env=replay_env, stdout=log_file,
            stderr=subprocess.STDOUT, timeout=3600, check=True
        )
        manifest = load_json(out_dir / "manifest.json", {})
        if not manifest or int(manifest.get("frames", 0)) <= 0:
            raise RuntimeError("replay manifest missing or empty")
        if not manifest.get("initial_state_replay"):
            raise RuntimeError("replay did not confirm initial-state restoration")
        assignment = load_json(out_dir / "aux_light_assignments.json", {})
        if assignment.get("schema") != "lighting_replay_r2":
            raise RuntimeError("lighting assignment is missing r2 schema")
        if bool(assignment.get("headlamp_enabled")) is not True:
            raise RuntimeError("headlamp was not enabled in rendered assignment")
        expected_aux_count = len(aux_indices.split(",")) if aux_indices else 0
        if len(assignment.get("lights", [])) != expected_aux_count:
            raise RuntimeError("rendered auxiliary-light count does not match case")
        return manifest
    finally:
        stop_process(sim)
        stop_exact_pid(simulator_pid)
        for detached_pid in simulator_pids_for_ready_file(ready):
            stop_exact_pid(detached_pid)
        log_file.close()
        ready.unlink(missing_ok=True)
        aux_action_path.unlink(missing_ok=True)


def package_augmented(spec, output: Path, source_episode: int, video_dir: Path) -> tuple[int, int]:
    row, states, actions = v3.source_data(spec, source_episode)
    actual_state_path = video_dir / "actual_replay_state.npz"
    if not actual_state_path.exists():
        raise RuntimeError(f"missing replay feedback state: {actual_state_path}")
    actual_states = np.load(actual_state_path)["observation_state"]
    if actual_states.shape != states.shape:
        raise RuntimeError(
            f"replay state shape mismatch: {actual_states.shape} != {states.shape}"
        )
    captures = {}
    name_map = {
        "observation.images.head": video_dir / "observation.images.head.mp4",
        "observation.images.left_wrist": video_dir / "observation.images.left.mp4",
        "observation.images.right_wrist": video_dir / "observation.images.right.mp4",
    }
    for key, path in name_map.items():
        if key in spec.camera_keys:
            cap = cv2.VideoCapture(str(path))
            if not cap.isOpened():
                raise RuntimeError(f"unable to open replay video {path}")
            captures[key] = cap
    if set(captures) != set(spec.camera_keys):
        raise RuntimeError(f"camera mismatch: expected={spec.camera_keys} got={tuple(captures)}")
    writer = v3.create_writer(spec, output)
    try:
        writer.start_episode()
        for i in range(len(actions)):
            images = {}
            for key, cap in captures.items():
                ok, frame = cap.read()
                if not ok or frame is None:
                    raise RuntimeError(f"replay video ended early at frame {i}: {key}")
                images[key] = frame
            writer.add_frame(actual_states[i], actions[i], images)
        index, frames = writer.save_episode()
        return index, frames
    except Exception:
        writer.discard_episode()
        raise
    finally:
        for cap in captures.values():
            cap.release()
        writer.close()


def task_run(task_key: str, pilot: bool, root: Path, pilot_all_cases: bool = False, selected_cases=CASES) -> None:
    task_name, dataset_name, task_id = TASKS[task_key]
    task_dir = TASK_ROOT / task_name
    dataset = task_dir / "data/lerobot_v30" / dataset_name
    task_out_root = root / "pilot" / task_key if pilot else root / task_key
    output = task_out_root / "dataset"
    work = task_out_root / "replays"
    task_out_root.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    work.mkdir(parents=True, exist_ok=True)
    spec = v3.load_dataset_spec(dataset)
    progress_path = task_out_root / "progress.json"
    progress = load_json(progress_path, {"task": task_key, "source_dataset": str(dataset), "originals_copied": 0, "items": {}})
    if not progress.get("source_hashes_before"):
        progress["source_hashes_before"] = sha256_tree(dataset)
        write_json(progress_path, progress)
    if not pilot:
        ensure_originals(spec, output, progress)
    items = []
    for case in selected_cases:
        for source_episode in range(10):
            key = f"{case}/source_{source_episode:03d}"
            items.append((case, source_episode, key))
    if pilot:
        items = [(case, 0, f"{case}/source_000") for case in selected_cases] if pilot_all_cases else [(selected_cases[0], 0, f"{selected_cases[0]}/source_000")]
    for case, source_episode, item_key in items:
        if progress["items"].get(item_key, {}).get("status") == "completed":
            continue
        progress["items"][item_key] = {"status": "running", "case": case, "source_episode": source_episode}
        write_json(progress_path, progress)
        replay_dir = work / case / f"source_{source_episode:03d}"
        log_path = root / "logs" / ("pilot" if pilot else "full") / task_key / f"{case}_source_{source_episode:03d}.log"
        try:
            manifest = None
            for ready_attempt in range(1, MAX_READY_ATTEMPTS + 1):
                try:
                    manifest = replay_one(
                        task_key, task_dir, task_id, dataset, source_episode,
                        case, replay_dir, log_path,
                    )
                    break
                except TimeoutError as exc:
                    if ready_attempt >= MAX_READY_ATTEMPTS:
                        raise
                    if log_path.exists():
                        retry_log = log_path.with_name(
                            f"{log_path.stem}.ready-timeout-attempt-{ready_attempt}{log_path.suffix}"
                        )
                        shutil.copy2(log_path, retry_log)
                    # This directory belongs only to the uncommitted failed
                    # item.  Remove partial videos before the next attempt;
                    # the packaged LeRobot dataset is not touched.
                    if replay_dir.exists():
                        shutil.rmtree(replay_dir)
                    progress["items"][item_key] = {
                        "status": "retrying",
                        "case": case,
                        "source_episode": source_episode,
                        "ready_attempt": ready_attempt,
                        "error": repr(exc),
                    }
                    write_json(progress_path, progress)
                    print(
                        f"[batch] transient ready timeout: {task_key} {case} "
                        f"source={source_episode} attempt={ready_attempt}/"
                        f"{MAX_READY_ATTEMPTS}; retrying",
                        flush=True,
                    )
                    time.sleep(15.0)
            if manifest is None:
                raise RuntimeError("replay retry loop exited without a manifest")
            if pilot:
                provenance = {"source_episode": source_episode, "lighting_case": case, "seed": 260904 + source_episode, "frames": int(manifest.get("frames", 0)), "replay_manifest": manifest}
            else:
                index, frames = package_augmented(spec, output, source_episode, replay_dir)
                provenance = {"output_episode": index, "source_episode": source_episode, "lighting_case": case, "seed": 260904 + source_episode, "frames": frames, "replay_manifest": manifest}
                with (task_out_root / "provenance.jsonl").open("a") as f:
                    f.write(json.dumps(provenance, ensure_ascii=False) + "\n")
            progress["items"][item_key] = {"status": "completed", **provenance}
            progress["completed_augmented"] = sum(1 for x in progress["items"].values() if x.get("status") == "completed")
            write_json(progress_path, progress)
            print(f"[batch] {task_key} {case} source={source_episode} frames={provenance['frames']}", flush=True)
        except Exception as exc:
            progress["items"][item_key] = {"status": "failed", "case": case, "source_episode": source_episode, "error": repr(exc)}
            write_json(progress_path, progress)
            raise
    progress["source_hashes_after"] = sha256_tree(dataset)
    progress["source_unchanged"] = progress["source_hashes_before"] == progress["source_hashes_after"]
    progress["dataset_total_episodes"] = int(json.loads((output / "meta/info.json").read_text())["total_episodes"]) if (output / "meta/info.json").exists() else None
    progress["complete"] = (not pilot and progress["completed_augmented"] == 50 and progress["dataset_total_episodes"] == 100 and progress["source_unchanged"])
    write_json(progress_path, progress)
    if progress["complete"]:
        (task_out_root / "COMPLETE").write_text("lighting replay augmentation complete\n")
    elif pilot and progress.get("completed_augmented") == len(items) and progress.get("source_unchanged"):
        (task_out_root / "PILOT_COMPLETE").write_text("lighting replay pilot complete\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=TASK_ROOT / "replay_augmented/lighting_5case_20260908_r2")
    parser.add_argument("--mode", choices=("pilot", "full"), default="pilot")
    parser.add_argument("--task", action="append", choices=tuple(TASKS))
    parser.add_argument("--pilot-all-cases", action="store_true")
    parser.add_argument("--case", action="append", choices=CASES)
    parser.add_argument(
        "--visual-approval-marker",
        type=Path,
        help="Required marker file before a full 300-episode run is allowed.",
    )
    args = parser.parse_args()
    if args.mode == "full":
        marker = args.visual_approval_marker
        if marker is None or not marker.expanduser().is_file():
            raise RuntimeError(
                "Full replay is blocked until the six-task pilot is visually "
                "approved; pass --visual-approval-marker with an existing file."
            )
    root = args.root.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    selected = args.task or list(TASKS)
    selected_cases = tuple(args.case or CASES)
    write_json(root / "batch_manifest.json", {"tasks": selected, "cases": list(selected_cases), "mode": args.mode, "source_episodes": list(range(10)), "dataset_policy": "50 originals + 50 augmented", "dust": "off", "flicker": "off", "task_headlamp_intensity": TASK_HEADLAMP_INTENSITY, "task_aux_intensity": TASK_AUX_INTENSITY})
    for task_key in selected:
        task_run(task_key, args.mode == "pilot", root, args.pilot_all_cases, selected_cases)
    statuses = {task: load_json((root / "pilot" / task if args.mode == "pilot" else root / task) / "progress.json", {}) for task in selected}
    write_json(root / "batch_progress.json", {"mode": args.mode, "tasks": statuses, "all_complete": args.mode == "full" and all((root / task / "COMPLETE").exists() for task in selected)})
    if args.mode == "full" and all((root / task / "COMPLETE").exists() for task in selected):
        (root / "ALL_COMPLETE").write_text("all six lighting replay datasets complete\n")
    print(json.dumps({"root": str(root), "mode": args.mode, "tasks": selected}, indent=2), flush=True)


if __name__ == "__main__":
    main()
