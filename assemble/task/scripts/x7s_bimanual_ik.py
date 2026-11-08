#!/usr/bin/env python3
"""LW-BenchHub X7S bimanual IK, ported to the local Isaac Sim runtime.

This module preserves the upstream X7S URDF model, reduced seven-joint arm
chains, joint limits, joint-11/joint-20 end-effector frames, warm start, error
gate, and output delta limiter.  LW-BenchHub's original implementation uses
``pinocchio.casadi`` and IPOPT.  The Pinocchio wheel bundled with Isaac Sim 6
does not ship its CasADi bindings, so a bounded SciPy least-squares solver is
used with separate position/orientation residuals.  Position is prioritized so
an unreachable wrist orientation cannot freeze Cartesian hand tracking.

The untouched upstream implementation is stored beside the robot asset at
``assets/robot/X7S/vendor/x7s_ik_original.py``.
"""

from __future__ import annotations

from pathlib import Path
import json
import os
import select
import subprocess
from types import SimpleNamespace
from typing import Optional, Tuple

import numpy as np
import pinocchio as pin
from pinocchio.robot_wrapper import RobotWrapper
from scipy.optimize import least_squares


class X7SPinocchioIK:
    """One reduced seven-DOF X7S arm using the LW-BenchHub IK contract."""

    def __init__(
        self,
        urdf_path: str | Path,
        package_dirs: list[str],
        dls_lambda: float = 1.0e-2,
        max_iters: int = 50,
        tol: float = 1.0e-4,
        step_gain: float = 1.0,
        joints_to_keep: Optional[list[str]] = None,
        ee_joint_name: Optional[str] = None,
    ) -> None:
        self.urdf_path = str(Path(urdf_path).resolve())
        self.package_dirs = [str(Path(path).resolve()) for path in package_dirs]
        self._full_robot = RobotWrapper.BuildFromURDF(
            self.urdf_path,
            self.package_dirs,
        )
        full_model = self._full_robot.model

        if joints_to_keep is None:
            joints_to_keep = [f"joint{number}" for number in range(5, 12)]
        keep_joint_names = set(joints_to_keep)
        joints_to_lock = [
            full_model.names[joint_id]
            for joint_id in range(1, full_model.njoints)
            if full_model.names[joint_id] not in keep_joint_names
        ]
        self._robot = self._full_robot.buildReducedRobot(
            list_of_joints_to_lock=joints_to_lock,
            reference_configuration=np.zeros(full_model.nq),
        )
        self._model = self._robot.model

        ee_joint_name = ee_joint_name or "joint11"
        ee_joint_id = self._model.getJointId(ee_joint_name)
        if ee_joint_id == 0:
            raise ValueError(f"X7S IK end-effector joint is missing: {ee_joint_name}")
        self._model.addFrame(
            pin.Frame(
                "ee",
                ee_joint_id,
                pin.SE3.Identity(),
                pin.FrameType.OP_FRAME,
            )
        )
        self._frame_id = self._model.getFrameId("ee")
        self._data = self._model.createData()
        if self._model.nq != 7 or self._model.nv != 7:
            raise RuntimeError(
                f"Unexpected reduced X7S arm size: nq={self._model.nq}, "
                f"nv={self._model.nv}"
            )

        self._lambda = float(dls_lambda)
        self._max_iters = int(max_iters)
        self._tol = float(tol)
        self._gain = float(step_gain)
        self._q_lower = np.asarray(self._model.lowerPositionLimit, dtype=np.float64)
        self._q_upper = np.asarray(self._model.upperPositionLimit, dtype=np.float64)
        self._last_q: Optional[np.ndarray] = None

        # Teleoperation validation is split by units.  A single norm that mixes
        # metres and radians can accept a visibly wrong Cartesian solution.
        self._position_gate_thresh = 0.025
        # IK is advanced only on fresh WebXR poses (~30 Hz), so this caps the
        # commanded arm speed at roughly 1.2 rad/s.  The
        # previous 0.15 rad/frame limit allowed a visibly violent 9 rad/s
        # target jump when the controller stream briefly glitched.
        self._max_delta_per_step = 0.04
        self.success_count = 0
        self.failure_count = 0
        self.last_error = 0.0
        self.last_position_error = 0.0
        self.last_orientation_error = 0.0

    @staticmethod
    def _homogeneous_from_pose(pos_wxyz: np.ndarray) -> np.ndarray:
        vector = np.asarray(pos_wxyz, dtype=np.float64)
        if vector.shape != (7,) or not np.all(np.isfinite(vector)):
            raise ValueError(f"Invalid X7S target pose: {vector}")
        quaternion = pin.Quaternion(
            float(vector[3]),
            float(vector[4]),
            float(vector[5]),
            float(vector[6]),
        ).normalized()
        target = np.eye(4, dtype=np.float64)
        target[:3, :3] = quaternion.toRotationMatrix()
        target[:3, 3] = vector[:3]
        return target

    def _pose_error(self, q: np.ndarray, target: pin.SE3) -> np.ndarray:
        pin.framesForwardKinematics(self._model, self._data, q)
        current = self._data.oMf[self._frame_id]
        return np.asarray(
            pin.log6(current.inverse() * target).vector,
            dtype=np.float64,
        )

    def forward_pose(self, q: np.ndarray) -> np.ndarray:
        """Return the reduced-model end-effector pose as a 4x4 matrix."""
        configuration = np.asarray(q, dtype=np.float64)
        if configuration.shape != (self._model.nq,):
            raise ValueError(
                f"Expected X7S configuration {(self._model.nq,)}, "
                f"got {configuration.shape}"
            )
        pin.framesForwardKinematics(self._model, self._data, configuration)
        pose = self._data.oMf[self._frame_id]
        result = np.eye(4, dtype=np.float64)
        result[:3, :3] = np.asarray(pose.rotation, dtype=np.float64)
        result[:3, 3] = np.asarray(pose.translation, dtype=np.float64)
        return result

    def _solve_single(
        self,
        target_matrix: np.ndarray,
        q_reference: np.ndarray,
    ) -> tuple[np.ndarray, bool, float, float]:
        target = pin.SE3(target_matrix[:3, :3], target_matrix[:3, 3])
        q_reference = np.clip(
            np.asarray(q_reference, dtype=np.float64),
            self._q_lower + 1.0e-8,
            self._q_upper - 1.0e-8,
        )
        position_weight = np.sqrt(120.0)
        # Keep wrist orientation useful without allowing a large/unreachable
        # rotation to dominate Cartesian hand translation.
        orientation_weight = np.sqrt(0.2)
        regularization_weight = np.sqrt(0.01)

        def residual(q: np.ndarray) -> np.ndarray:
            pin.framesForwardKinematics(self._model, self._data, q)
            current = self._data.oMf[self._frame_id]
            position_error = np.asarray(
                current.translation - target.translation,
                dtype=np.float64,
            )
            orientation_error = np.asarray(
                pin.log3(current.rotation.T @ target.rotation),
                dtype=np.float64,
            )
            return np.concatenate(
                (
                    position_weight * position_error,
                    orientation_weight * orientation_error,
                    regularization_weight * (q - q_reference),
                )
            )

        result = least_squares(
            residual,
            q_reference,
            bounds=(self._q_lower, self._q_upper),
            method="trf",
            ftol=self._tol,
            xtol=self._tol,
            gtol=self._tol,
            max_nfev=self._max_iters,
            x_scale="jac",
        )
        solution = np.asarray(result.x, dtype=np.float64)
        pin.framesForwardKinematics(self._model, self._data, solution)
        achieved = self._data.oMf[self._frame_id]
        position_error = float(
            np.linalg.norm(achieved.translation - target.translation)
        )
        orientation_error = float(
            np.linalg.norm(pin.log3(achieved.rotation.T @ target.rotation))
        )
        # Position is the hard teleoperation constraint.  A human wrist can
        # request an orientation outside the current 7-DOF workspace; rejecting
        # that otherwise good positional solution makes the whole arm appear
        # frozen.  Orientation remains in the objective and is reported for
        # diagnostics, but does not gate Cartesian translation.
        converged = bool(
            np.all(np.isfinite(solution))
            and np.isfinite(position_error)
            and np.isfinite(orientation_error)
            and position_error <= self._position_gate_thresh
        )
        return solution, converged, position_error, orientation_error

    def solve_pose_to_joints(
        self,
        targets_pos_wxyz: np.ndarray,
        warm_start: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        targets = np.asarray(targets_pos_wxyz, dtype=np.float64)
        if targets.ndim == 1:
            targets = targets[np.newaxis, :]
        if targets.ndim != 2 or targets.shape[1] != 7:
            raise ValueError(f"Expected X7S IK targets shaped (B, 7), got {targets.shape}")
        batch = targets.shape[0]
        warm = None if warm_start is None else np.asarray(warm_start, dtype=np.float64)
        if warm is not None and warm.ndim == 1:
            warm = warm[np.newaxis, :]

        q_output = np.zeros((batch, self._model.nq), dtype=np.float64)
        success = np.zeros(batch, dtype=bool)
        for index in range(batch):
            previous = None
            if self._last_q is not None and self._last_q.shape == q_output.shape:
                previous = self._last_q[index]
            if warm is not None:
                reference = warm[index]
            elif previous is not None:
                reference = previous
            else:
                reference = np.zeros(self._model.nq, dtype=np.float64)

            target_matrix = self._homogeneous_from_pose(targets[index])
            try:
                solution, converged, position_error, orientation_error = self._solve_single(
                    target_matrix,
                    reference,
                )
            except (ArithmeticError, RuntimeError, ValueError):
                solution = reference.copy()
                converged = False
                position_error = float("inf")
                orientation_error = float("inf")

            if converged:
                chosen = solution
            elif previous is not None:
                chosen = previous.copy()
            else:
                # Never apply a failed first solution.  Use the commanded
                # configuration until a finite target becomes reachable.
                chosen = reference.copy()
            if previous is not None:
                delta = np.clip(
                    chosen - previous,
                    -self._max_delta_per_step,
                    self._max_delta_per_step,
                )
                chosen = previous + delta
            chosen = np.clip(chosen, self._q_lower, self._q_upper)
            if not np.all(np.isfinite(chosen)):
                chosen = reference.copy()
                converged = False

            q_output[index] = chosen
            success[index] = converged
            self.last_position_error = position_error
            self.last_orientation_error = orientation_error
            self.last_error = position_error + orientation_error
            if converged:
                self.success_count += 1
            else:
                self.failure_count += 1

        self._last_q = q_output.copy()
        return q_output, success

    def reset(self) -> None:
        self._last_q = None
        self.last_error = 0.0
        self.last_position_error = 0.0
        self.last_orientation_error = 0.0


class X7SBimanualIK:
    """LW-BenchHub's left joint5..11 and right joint14..20 IK pair."""

    def __init__(
        self,
        urdf_path: str | Path,
        package_dirs: list[str],
        dls_lambda: float = 1.0e-2,
        max_iters: int = 50,
        tol: float = 1.0e-4,
    ) -> None:
        common = dict(
            urdf_path=urdf_path,
            package_dirs=package_dirs,
            dls_lambda=dls_lambda,
            max_iters=max_iters,
            tol=tol,
        )
        self.left = X7SPinocchioIK(
            **common,
            joints_to_keep=[f"joint{number}" for number in range(5, 12)],
            ee_joint_name="joint11",
        )
        self.right = X7SPinocchioIK(
            **common,
            joints_to_keep=[f"joint{number}" for number in range(14, 21)],
            ee_joint_name="joint20",
        )

    def solve_pose_to_joints(
        self,
        left_targets_pos_wxyz: np.ndarray,
        right_targets_pos_wxyz: np.ndarray,
        left_warm_start: Optional[np.ndarray] = None,
        right_warm_start: Optional[np.ndarray] = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        left_q, left_success = self.left.solve_pose_to_joints(
            left_targets_pos_wxyz,
            warm_start=left_warm_start,
        )
        right_q, right_success = self.right.solve_pose_to_joints(
            right_targets_pos_wxyz,
            warm_start=right_warm_start,
        )
        return left_q, left_success, right_q, right_success

    def reset(self) -> None:
        self.left.reset()
        self.right.reset()


class X7SOriginalIKClient:
    """Persistent bridge to LW-BenchHub's original Python 3.11 solver."""

    PREFIX = b"X7S_JSON "

    def __init__(
        self,
        urdf_path: str | Path,
        package_dirs: list[str],
        python_executable: str | Path,
        server_script: str | Path,
        vendor_dir: str | Path,
        timeout_s: float = 15.0,
        **_ignored,
    ) -> None:
        self.timeout_s = float(timeout_s)
        status = dict(
            success_count=0,
            failure_count=0,
            last_error=0.0,
            last_position_error=0.0,
            last_orientation_error=0.0,
        )
        self.left = SimpleNamespace(**status)
        self.right = SimpleNamespace(**status)
        worker_env = os.environ.copy()
        for name in (
            "PYTHONHOME",
            "PYTHONPATH",
            "LD_LIBRARY_PATH",
            "LD_PRELOAD",
            "CONDA_PREFIX",
            "CONDA_DEFAULT_ENV",
            "VIRTUAL_ENV",
        ):
            worker_env.pop(name, None)
        self.process = subprocess.Popen(
            [
                str(python_executable),
                str(server_script),
                "--urdf",
                str(urdf_path),
                "--package-dir",
                str(package_dirs[0]),
                "--vendor-dir",
                str(vendor_dir),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,
            text=False,
            bufsize=0,
            env=worker_env,
        )
        ready = self._read_response(timeout_s=30.0)
        if not ready.get("ready"):
            self.close()
            raise RuntimeError(f"LW-BenchHub X7S IK worker did not become ready: {ready}")

    def _read_response(self, timeout_s: float | None = None) -> dict:
        if self.process.stdout is None:
            raise RuntimeError("X7S IK worker stdout is unavailable")
        timeout = self.timeout_s if timeout_s is None else float(timeout_s)
        while True:
            if self.process.poll() is not None:
                raise RuntimeError(
                    f"LW-BenchHub X7S IK worker exited with {self.process.returncode}"
                )
            readable, _, _ = select.select([self.process.stdout], [], [], timeout)
            if not readable:
                raise TimeoutError("LW-BenchHub X7S IK worker response timed out")
            line = self.process.stdout.readline()
            if not line:
                raise RuntimeError("LW-BenchHub X7S IK worker closed stdout")
            if line.startswith(self.PREFIX):
                payload = json.loads(line[len(self.PREFIX) :].decode("utf-8"))
                if "error" in payload:
                    raise RuntimeError(payload["error"])
                return payload

    def _request(self, payload: dict) -> dict:
        if self.process.stdin is None or self.process.poll() is not None:
            raise RuntimeError("LW-BenchHub X7S IK worker is not running")
        self.process.stdin.write(
            (json.dumps(payload, separators=(",", ":")) + "\n").encode("utf-8")
        )
        self.process.stdin.flush()
        return self._read_response()

    def solve_pose_to_joints(
        self,
        left_targets_pos_wxyz: np.ndarray,
        right_targets_pos_wxyz: np.ndarray,
        left_warm_start: Optional[np.ndarray] = None,
        right_warm_start: Optional[np.ndarray] = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        del left_warm_start, right_warm_start
        response = self._request(
            {
                "cmd": "solve",
                "left": np.asarray(left_targets_pos_wxyz, dtype=float).tolist(),
                "right": np.asarray(right_targets_pos_wxyz, dtype=float).tolist(),
            }
        )
        left_q = np.asarray(response["left_q"], dtype=np.float64)
        right_q = np.asarray(response["right_q"], dtype=np.float64)
        left_ok = np.asarray(response["left_ok"], dtype=bool)
        right_ok = np.asarray(response["right_ok"], dtype=bool)
        for state, values in ((self.left, left_ok), (self.right, right_ok)):
            success = int(np.count_nonzero(values))
            failure = int(values.size - success)
            state.success_count += success
            state.failure_count += failure
            state.last_error = 0.0 if failure == 0 else float("inf")
        return left_q, left_ok, right_q, right_ok

    def reset(self) -> None:
        self._request({"cmd": "reset"})
        for state in (self.left, self.right):
            state.last_error = 0.0

    def close(self) -> None:
        process = getattr(self, "process", None)
        if process is None:
            return
        if process.poll() is None:
            try:
                if process.stdin is not None:
                    process.stdin.write(b'{"cmd":"close"}\n')
                    process.stdin.flush()
                process.wait(timeout=2.0)
            except Exception:
                process.terminate()
                try:
                    process.wait(timeout=2.0)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=2.0)
        self.process = None


def create_x7s_ik(
    urdf_path: str | Path,
    package_dirs: list[str],
    scripts_dir: str | Path,
    vendor_dir: str | Path,
    **kwargs,
) -> tuple[X7SBimanualIK | X7SOriginalIKClient, str]:
    """Create the X7S solver.

    The upstream CasADi/IPOPT implementation is retained for reproducibility,
    but its SE(3) log objective can emit NaNs for ordinary live Quest wrist
    rotations.  Use the numerically robust Pinocchio/SciPy implementation for
    interactive teleoperation by default.  Set ``X7S_IK_BACKEND=original`` to
    select the untouched upstream worker explicitly.
    """
    scripts_dir = Path(scripts_dir).resolve()
    python_executable = scripts_dir / ".x7s_ik_env/bin/python"
    server_script = scripts_dir / "x7s_original_ik_server.py"
    requested_backend = os.environ.get("X7S_IK_BACKEND", "robust").strip().lower()
    if requested_backend == "original" and python_executable.is_file() and server_script.is_file():
        try:
            return (
                X7SOriginalIKClient(
                    urdf_path=urdf_path,
                    package_dirs=package_dirs,
                    python_executable=python_executable,
                    server_script=server_script,
                    vendor_dir=vendor_dir,
                    **kwargs,
                ),
                "lw_benchhub_original_casadi_ipopt",
            )
        except Exception as error:
            print(
                f"[IK] Original LW-BenchHub worker unavailable ({error}); "
                "using the local compatibility solver.",
                flush=True,
            )
    return (
        X7SBimanualIK(
            urdf_path=urdf_path,
            package_dirs=package_dirs,
            **kwargs,
        ),
        "lw_benchhub_x7s_pinocchio_scipy_robust",
    )
