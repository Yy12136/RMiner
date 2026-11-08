#!/usr/bin/env python3
"""JSON-line worker around LW-BenchHub's untouched X7S CasADi/IPOPT IK."""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import sys
import traceback
from pathlib import Path

import numpy as np


PREFIX = "X7S_JSON "


def emit(payload: dict) -> None:
    print(PREFIX + json.dumps(payload, separators=(",", ":")), flush=True)


def reset_ik(ik) -> None:
    # The upstream per-arm class exposes reset(); its bimanual wrapper does not.
    ik.left.reset()
    ik.right.reset()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--urdf", required=True)
    parser.add_argument("--package-dir", required=True)
    parser.add_argument("--vendor-dir", required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(Path(args.vendor_dir).resolve()))

    from x7s_ik_original import X7SBimanualIK

    ik = X7SBimanualIK(
        str(Path(args.urdf).resolve()),
        [str(Path(args.package_dir).resolve())],
    )
    # Trigger IPOPT's one-time banner and JIT setup before announcing ready.
    left_home = np.asarray([[0.3725, 0.1508, 0.263, 1.0, 0.0, 0.0, 0.0]])
    right_home = np.asarray([[0.3725, -0.1512, 0.263, 1.0, 0.0, 0.0, 0.0]])
    # IPOPT writes its one-time banner directly to file descriptor 1. Suppress
    # it so stdout remains a strict one-message-per-line protocol stream.
    sys.stdout.flush()
    saved_stdout = os.dup(sys.stdout.fileno())
    null_stdout = os.open(os.devnull, os.O_WRONLY)
    os.dup2(null_stdout, sys.stdout.fileno())
    try:
        ik.solve_pose_to_joints(left_home, right_home)
        ctypes.CDLL(None).fflush(None)
    finally:
        os.dup2(saved_stdout, sys.stdout.fileno())
        os.close(saved_stdout)
        os.close(null_stdout)
    reset_ik(ik)
    emit({"ready": True, "backend": "lw_benchhub_casadi_ipopt"})

    for line in sys.stdin:
        try:
            request = json.loads(line)
            command = request.get("cmd")
            if command == "solve":
                left = np.asarray(request["left"], dtype=np.float64)
                right = np.asarray(request["right"], dtype=np.float64)
                left_q, left_ok, right_q, right_ok = ik.solve_pose_to_joints(
                    left,
                    right,
                )
                emit(
                    {
                        "left_q": left_q.tolist(),
                        "left_ok": left_ok.tolist(),
                        "right_q": right_q.tolist(),
                        "right_ok": right_ok.tolist(),
                    }
                )
            elif command == "reset":
                reset_ik(ik)
                emit({"reset": True})
            elif command == "close":
                emit({"closed": True})
                return
            else:
                emit({"error": f"unknown command: {command}"})
        except Exception as error:
            traceback.print_exc(file=sys.stderr)
            emit({"error": f"{type(error).__name__}: {error}"})


if __name__ == "__main__":
    main()
