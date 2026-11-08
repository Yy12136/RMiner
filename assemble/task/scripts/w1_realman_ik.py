#!/usr/bin/env python3
"""RealMan RM75 inverse-kinematics backend for W1 Quest teleoperation.

The W1 USD uses the RM75 seven-joint limits, but its arm frames are mirrored
and the whole robot is authored at scale 1.1.  This adapter re-anchors every
small Quest motion at the current simulated pose, converts it into the RM75
model frame, and asks RealMan's official ``Robotic_Arm`` library for the next
joint configuration.

The adapter deliberately returns ``None`` for a failed, stagnant, or
discontinuous solution.  The caller can then use its legacy differential IK
for that frame; this is important at the saved all-zero arm pose, which is an
RM75 singular configuration.
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
QUEST_DEPENDENCIES = SCRIPT_DIR / ".quest_deps"
if str(QUEST_DEPENDENCIES) not in sys.path:
    sys.path.insert(0, str(QUEST_DEPENDENCIES))

_SDK_IMPORT_ERROR: Exception | None = None
try:
    from Robotic_Arm.rm_robot_interface import (
        Algo,
        rm_force_type_e,
        rm_inverse_kinematics_params_t,
        rm_robot_arm_model_e,
    )
except (ImportError, OSError) as error:
    _SDK_IMPORT_ERROR = error


def _quaternion_wxyz_to_matrix(quaternion: np.ndarray) -> np.ndarray:
    values = np.asarray(quaternion, dtype=np.float64)
    norm = float(np.linalg.norm(values))
    if values.shape != (4,) or norm < 1.0e-12:
        raise ValueError(f"Invalid quaternion: {values}")
    w, x, y, z = values / norm
    return np.asarray(
        [
            [
                1.0 - 2.0 * (y * y + z * z),
                2.0 * (x * y - z * w),
                2.0 * (x * z + y * w),
            ],
            [
                2.0 * (x * y + z * w),
                1.0 - 2.0 * (x * x + z * z),
                2.0 * (y * z - x * w),
            ],
            [
                2.0 * (x * z - y * w),
                2.0 * (y * z + x * w),
                1.0 - 2.0 * (x * x + y * y),
            ],
        ],
        dtype=np.float64,
    )


def _matrix_to_quaternion_wxyz(rotation: np.ndarray) -> np.ndarray:
    matrix = np.asarray(rotation, dtype=np.float64)
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        values = np.asarray(
            [
                0.25 * scale,
                (matrix[2, 1] - matrix[1, 2]) / scale,
                (matrix[0, 2] - matrix[2, 0]) / scale,
                (matrix[1, 0] - matrix[0, 1]) / scale,
            ]
        )
    else:
        diagonal = np.diag(matrix)
        axis = int(np.argmax(diagonal))
        if axis == 0:
            scale = math.sqrt(
                max(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2], 0.0)
            ) * 2.0
            values = np.asarray(
                [
                    (matrix[2, 1] - matrix[1, 2]) / scale,
                    0.25 * scale,
                    (matrix[0, 1] + matrix[1, 0]) / scale,
                    (matrix[0, 2] + matrix[2, 0]) / scale,
                ]
            )
        elif axis == 1:
            scale = math.sqrt(
                max(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2], 0.0)
            ) * 2.0
            values = np.asarray(
                [
                    (matrix[0, 2] - matrix[2, 0]) / scale,
                    (matrix[0, 1] + matrix[1, 0]) / scale,
                    0.25 * scale,
                    (matrix[1, 2] + matrix[2, 1]) / scale,
                ]
            )
        else:
            scale = math.sqrt(
                max(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1], 0.0)
            ) * 2.0
            values = np.asarray(
                [
                    (matrix[1, 0] - matrix[0, 1]) / scale,
                    (matrix[0, 2] + matrix[2, 0]) / scale,
                    (matrix[1, 2] + matrix[2, 1]) / scale,
                    0.25 * scale,
                ]
            )
    norm = float(np.linalg.norm(values))
    if norm < 1.0e-12:
        raise ValueError("Could not convert rotation matrix to quaternion")
    values /= norm
    if values[0] < 0.0:
        values *= -1.0
    return values


def _rotation_vector(rotation: np.ndarray) -> np.ndarray:
    matrix = np.asarray(rotation, dtype=np.float64)
    cosine = float(np.clip((np.trace(matrix) - 1.0) * 0.5, -1.0, 1.0))
    angle = math.acos(cosine)
    skew = np.asarray(
        [
            matrix[2, 1] - matrix[1, 2],
            matrix[0, 2] - matrix[2, 0],
            matrix[1, 0] - matrix[0, 1],
        ],
        dtype=np.float64,
    )
    if angle < 1.0e-7:
        return 0.5 * skew
    sine = math.sin(angle)
    if abs(sine) < 1.0e-7:
        return np.zeros(3, dtype=np.float64)
    return (angle / (2.0 * sine)) * skew


def _rotation_from_vector(vector: np.ndarray) -> np.ndarray:
    values = np.asarray(vector, dtype=np.float64)
    angle = float(np.linalg.norm(values))
    if angle < 1.0e-10:
        skew = np.asarray(
            [
                [0.0, -values[2], values[1]],
                [values[2], 0.0, -values[0]],
                [-values[1], values[0], 0.0],
            ]
        )
        return np.eye(3, dtype=np.float64) + skew
    axis = values / angle
    x, y, z = axis
    skew = np.asarray([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    return (
        np.eye(3, dtype=np.float64)
        + math.sin(angle) * skew
        + (1.0 - math.cos(angle)) * (skew @ skew)
    )


class RealManRM75IK:
    """Safe local-pose wrapper around RealMan's official RM75 IK."""

    # The mirrored W1 USD joint axes match RM75 after reversing joint 1.
    JOINT_SIGNS = np.asarray([-1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

    def __init__(self, model_scale: float = 1.1) -> None:
        if _SDK_IMPORT_ERROR is not None:
            raise RuntimeError(
                "RealMan Robotic_Arm SDK is unavailable; run "
                f"{SCRIPT_DIR / 'setup_w1_quest3s.sh'}"
            ) from _SDK_IMPORT_ERROR
        if not math.isfinite(model_scale) or model_scale <= 0.0:
            raise ValueError(f"Invalid RM75 model scale: {model_scale}")
        self.model_scale = float(model_scale)
        self.max_solution_jump = float(
            os.environ.get("W1_REALMAN_IK_MAX_SOLUTION_JUMP", "0.35")
        )
        self.finite_difference_degrees = float(
            os.environ.get("W1_REALMAN_IK_FD_DEGREES", "0.05")
        )
        self.algo = Algo(
            rm_robot_arm_model_e.RM_MODEL_RM_75_E,
            rm_force_type_e.RM_MODEL_RM_B_E,
        )
        # The official documentation recommends single-step redundancy mode
        # for small, continuous Cartesian targets such as teleoperation.
        self.algo.rm_algo_set_redundant_parameter_traversal_mode(False)
        self.version = str(self.algo.rm_algo_version())
        self.success_count = 0
        self.fallback_count = 0
        self.last_failure = ""

    def _fail(self, reason: str) -> None:
        self.fallback_count += 1
        self.last_failure = reason

    def _forward(self, joints_degrees: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        pose = self.algo.rm_algo_forward_kinematics(
            np.asarray(joints_degrees, dtype=np.float64).tolist(),
            0,
        )
        values = np.asarray(pose, dtype=np.float64)
        if values.shape != (7,) or not np.all(np.isfinite(values)):
            raise RuntimeError(f"Invalid RealMan forward-kinematics pose: {pose}")
        return values[:3], _quaternion_wxyz_to_matrix(values[3:])

    def _orientation_jacobian(
        self,
        joints_degrees: np.ndarray,
    ) -> np.ndarray:
        epsilon_degrees = self.finite_difference_degrees
        epsilon_radians = math.radians(epsilon_degrees)
        jacobian = np.zeros((3, 7), dtype=np.float64)
        for index in range(7):
            plus = joints_degrees.copy()
            minus = joints_degrees.copy()
            plus[index] += epsilon_degrees
            minus[index] -= epsilon_degrees
            _, plus_rotation = self._forward(plus)
            _, minus_rotation = self._forward(minus)
            jacobian[:, index] = _rotation_vector(
                plus_rotation @ minus_rotation.T
            ) / (2.0 * epsilon_radians)
        return jacobian

    @staticmethod
    def _best_frame_rotation(
        world_jacobian: np.ndarray,
        model_jacobian: np.ndarray,
    ) -> np.ndarray:
        covariance = world_jacobian @ model_jacobian.T
        left, singular_values, right_transpose = np.linalg.svd(covariance)
        if singular_values.shape != (3,) or singular_values[1] < 1.0e-5:
            raise RuntimeError(
                f"Insufficient rotational Jacobian rank: {singular_values}"
            )
        correction = np.eye(3, dtype=np.float64)
        correction[2, 2] = np.linalg.det(left @ right_transpose)
        return left @ correction @ right_transpose

    def solve(
        self,
        current_joints_radians: np.ndarray,
        position_error_world: np.ndarray,
        rotation_error_world: np.ndarray,
        rotation_jacobian_world: np.ndarray,
    ) -> np.ndarray | None:
        """Return an Isaac joint target, or ``None`` for legacy-IK fallback."""
        try:
            current = np.asarray(current_joints_radians, dtype=np.float64)
            position_error = np.asarray(position_error_world, dtype=np.float64)
            rotation_error = np.asarray(rotation_error_world, dtype=np.float64)
            world_jacobian = np.asarray(
                rotation_jacobian_world,
                dtype=np.float64,
            )
            if (
                current.shape != (7,)
                or position_error.shape != (3,)
                or rotation_error.shape != (3,)
                or world_jacobian.shape != (3, 7)
                or not np.all(np.isfinite(current))
                or not np.all(np.isfinite(position_error))
                or not np.all(np.isfinite(rotation_error))
                or not np.all(np.isfinite(world_jacobian))
            ):
                self._fail("invalid input shape or non-finite value")
                return None

            current_model_degrees = np.degrees(current * self.JOINT_SIGNS)
            current_position, current_rotation = self._forward(
                current_model_degrees
            )
            model_jacobian = self._orientation_jacobian(
                current_model_degrees
            ) * self.JOINT_SIGNS[np.newaxis, :]
            world_from_model = self._best_frame_rotation(
                world_jacobian,
                model_jacobian,
            )

            target_position = current_position + (
                world_from_model.T @ position_error
            ) / self.model_scale
            model_rotation_error = world_from_model.T @ rotation_error
            target_rotation = (
                _rotation_from_vector(model_rotation_error) @ current_rotation
            )
            target_quaternion = _matrix_to_quaternion_wxyz(target_rotation)
            target_pose = np.concatenate(
                (target_position, target_quaternion)
            ).tolist()
            params = rm_inverse_kinematics_params_t(
                current_model_degrees.tolist(),
                target_pose,
                0,
            )
            result, output = self.algo.rm_algo_inverse_kinematics(params)
            output_degrees = np.asarray(output, dtype=np.float64)
            if (
                result != 0
                or output_degrees.shape != (7,)
                or not np.all(np.isfinite(output_degrees))
            ):
                self._fail(f"official IK returned status {result}")
                return None

            output_position, output_rotation = self._forward(output_degrees)
            position_before = float(np.linalg.norm(target_position - current_position))
            position_after = float(np.linalg.norm(target_position - output_position))
            rotation_before = float(np.linalg.norm(model_rotation_error))
            rotation_after = float(
                np.linalg.norm(_rotation_vector(target_rotation @ output_rotation.T))
            )
            if position_before > 1.0e-5 and position_after > max(
                1.5e-3,
                0.85 * position_before,
            ):
                self._fail(
                    "official IK did not reduce position error "
                    f"({position_before:.4g} -> {position_after:.4g} m)"
                )
                return None
            if rotation_before > 1.0e-5 and rotation_after > max(
                1.0e-2,
                0.85 * rotation_before,
            ):
                self._fail(
                    "official IK did not reduce rotation error "
                    f"({rotation_before:.4g} -> {rotation_after:.4g} rad)"
                )
                return None

            delta_model_degrees = (
                output_degrees - current_model_degrees + 180.0
            ) % 360.0 - 180.0
            delta_isaac = (
                np.radians(delta_model_degrees) * self.JOINT_SIGNS
            )
            if float(np.max(np.abs(delta_isaac))) > self.max_solution_jump:
                self._fail(
                    "official IK selected a discontinuous branch "
                    f"({np.max(np.abs(delta_isaac)):.3f} rad)"
                )
                return None

            self.success_count += 1
            self.last_failure = ""
            return current + delta_isaac
        except (ArithmeticError, RuntimeError, ValueError, np.linalg.LinAlgError) as error:
            self._fail(str(error))
            return None
