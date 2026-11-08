#!/usr/bin/env python3
"""Start policy first, then simulator and bridge; own and clean every process group."""

from __future__ import annotations

import argparse
import fcntl
import os
import secrets
import signal
import subprocess
import sys
import time
from pathlib import Path

from task_specs import get_spec

_BENCH_ASSEMBLE = Path(__file__).resolve().parents[2]
CONDA_SH = Path(os.environ.get("CONDA_SH") or (Path.home() / "miniconda3" / "etc" / "profile.d" / "conda.sh"))
TELEOP_DIR = Path(os.environ.get("XR_TELEOP_DIR") or (_BENCH_ASSEMBLE / "third_party" / "xr_teleoperate" / "teleop"))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True); parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=20); parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout", type=float, default=60.0); parser.add_argument("--frequency", type=float, default=30.0)
    parser.add_argument("--inference", choices=("sync",), default="sync"); parser.add_argument("--headless", action="store_true")
    parser.add_argument("--network-interface", default=None)
    parser.add_argument("--perturb", default="none")
    parser.add_argument("--perturb-seed", type=int, default=0)
    parser.add_argument("--perturb-particle-count", type=int, default=5000)
    parser.add_argument("--timeout-steps", type=int, default=0)
    return parser.parse_args()


def wait_path(path: Path, process: subprocess.Popen, timeout: float, kind: str) -> None:
    deadline = time.monotonic() + timeout
    while not path.exists():
        if process.poll() is not None:
            raise RuntimeError(f"{kind} exited early with code {process.returncode}")
        if time.monotonic() >= deadline:
            raise TimeoutError(f"Timed out waiting for {kind}: {path}")
        time.sleep(0.2)


def terminate_group(process: subprocess.Popen | None, grace: float = 15.0) -> None:
    if process is None:
        return
    # A launcher may exit while an Isaac child is still alive in its process
    # group. Always address the PGID even when Popen.poll() has a return code.
    try: os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        if process.poll() is None:
            try: process.terminate()
            except ProcessLookupError: pass
        return
    deadline = time.monotonic() + grace
    while time.monotonic() < deadline:
        try: os.killpg(process.pid, 0)
        except ProcessLookupError: break
        time.sleep(0.1)
    try:
        os.killpg(process.pid, 0)
    except ProcessLookupError:
        pass
    else:
        try: os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError: pass
    if process.poll() is None:
        try: process.wait(timeout=3)
        except subprocess.TimeoutExpired: pass


def main() -> None:
    args, spec = parse_args(), None
    spec = get_spec(args.task)
    evaluation_root = Path(__file__).resolve().parent
    task_collection = evaluation_root.parent
    task_root = task_collection / spec.task_dir
    uid = os.getuid(); base = f"/tmp/g1_{spec.slug}_{uid}"
    paths = {
        "sim_ready": Path(base + ".ready"), "success": Path(base + ".success"),
        "failure": Path(base + ".failure"),
        "request": Path(base + ".eval_reset.json"), "status": Path(base + ".eval_status.json"),
        "active": Path(base + ".eval_active"), "socket": Path(base + ".eval.sock"),
        "policy_ready": Path(base + ".policy_ready"),
    }
    for path in paths.values(): path.unlink(missing_ok=True)
    lock_path = Path(f"/tmp/g1_benchmark_eval_{uid}.lock")
    lock = lock_path.open("w");
    try: fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError: raise RuntimeError("Another G1 formal evaluation already owns the global lock")
    paths["active"].touch(); authkey = secrets.token_hex(24)
    env = os.environ.copy()
    env.update({
        "G1_SIM_READY_FILE": str(paths["sim_ready"]), "G1_TASK_SUCCESS_FILE": str(paths["success"]),
        "G1_TASK_FAILURE_FILE": str(paths["failure"]),
        "G1_BENCHMARK_RESET_REQUEST_FILE": str(paths["request"]),
        "G1_BENCHMARK_RESET_STATUS_FILE": str(paths["status"]),
        "G1_BENCHMARK_ACTIVE_FILE": str(paths["active"]),
        "G1_SHOW_CAMERA_VIEWPORTS": "0" if args.headless else "1",
        # Keep the policy on CUDA while Isaac physics runs on CPU. Cameras are
        # still rendered by Vulkan on the GPU, avoiding a 16 GB VRAM collision
        # when PI0.5 and the simulator coexist on the evaluation workstation.
        "G1_EVAL_SIM_DEVICE": "cpu",
        # supervisor.py already starts run_task.sh in a fresh session. Avoid a
        # nested setsid so every Isaac descendant remains in the owned PGID.
        "G1_PROCESS_GROUP_MANAGED": "1",
    })
    evaluator = simulator = bridge = None
    stop_requested = False
    def request_stop(_signal, _frame):
        nonlocal stop_requested; stop_requested = True
        if evaluator and evaluator.poll() is None:
            try: os.killpg(evaluator.pid, signal.SIGINT)
            except ProcessLookupError: pass
    signal.signal(signal.SIGINT, request_stop); signal.signal(signal.SIGTERM, request_stop)
    try:
        eval_command = [sys.executable, str(evaluation_root / "evaluator.py"),
            "--task", spec.key, "--checkpoint", str(args.checkpoint), "--socket", str(paths["socket"]),
            "--authkey", authkey, "--output-root", str(task_root / "experiments/results"),
            "--success-marker-file", str(paths["success"]), "--failure-marker-file", str(paths["failure"]),
            "--ready-file", str(paths["policy_ready"]),
            "--episodes", str(args.episodes), "--seed", str(args.seed), "--timeout", str(args.timeout),
            "--frequency", str(args.frequency), "--inference", args.inference,
            "--perturb", args.perturb, "--perturb-seed", str(args.perturb_seed),
            "--perturb-particle-count", str(args.perturb_particle_count),
            "--timeout-steps", str(args.timeout_steps)]
        evaluator = subprocess.Popen(eval_command, env=env, start_new_session=True)
        wait_path(paths["policy_ready"], evaluator, 600, "policy loader")

        sim_command = [str(task_root / "run_task.sh")]
        if args.headless: sim_command.append("--headless")
        simulator = subprocess.Popen(sim_command, cwd=task_root, env=env, start_new_session=True)
        wait_path(paths["sim_ready"], simulator, 240, "Isaac simulator")

        bridge_command = ["bash", "-lc",
            f"source {CONDA_SH} && conda activate tv && cd {TELEOP_DIR} && exec python {evaluation_root / 'dds_bridge.py'} "
            f"--task {spec.key} --socket {paths['socket']} --authkey {authkey} "
            f"--reset-request-file {paths['request']} --reset-status-file {paths['status']}" +
            (f" --network-interface {args.network_interface}" if args.network_interface else "")]
        bridge = subprocess.Popen(bridge_command, env=env, start_new_session=True)
        wait_path(paths["socket"], bridge, 60, "DDS bridge")
        while evaluator.poll() is None and not stop_requested:
            if simulator.poll() is not None:
                raise RuntimeError(f"Simulator exited during evaluation ({simulator.returncode})")
            if bridge.poll() is not None:
                raise RuntimeError(f"DDS bridge exited during evaluation ({bridge.returncode})")
            time.sleep(0.5)
        if stop_requested and evaluator.poll() is None:
            terminate_group(evaluator, 10)
        code = evaluator.wait() if evaluator.poll() is None else evaluator.returncode
        if code:
            raise RuntimeError(f"Evaluator exited with code {code}")
    finally:
        # The evaluator requests a bridge safe shutdown. Let it finish first;
        # then stop the simulator and finally hard-kill only our process groups.
        terminate_group(evaluator, 5); terminate_group(bridge, 8); terminate_group(simulator, 20)
        for path in paths.values(): path.unlink(missing_ok=True)
        fcntl.flock(lock, fcntl.LOCK_UN); lock.close()


if __name__ == "__main__":
    main()
