#!/usr/bin/env python3
"""Offline Cartesian regression checks for the local X7S IK backend."""

from pathlib import Path
import sys

import numpy as np
import pinocchio as pin

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from x7s_bimanual_ik import X7SPinocchioIK


X7S_ROOT = (
    SCRIPT_DIR.parent
    / "w1_mobile_drilling_task"
    / "assets"
    / "robot"
    / "X7S"
)
LEFT_READY = np.asarray(
    [0.0513, -0.2732, -0.2523, 0.4505, -0.3086, -0.4492, 0.1101]
)
RIGHT_READY = np.asarray(
    [-0.0513, 0.2721, 0.2513, -0.4514, 0.3075, 0.4497, -0.1094]
)


def _pose_vector(matrix: np.ndarray) -> np.ndarray:
    quaternion = pin.Quaternion(matrix[:3, :3]).normalized()
    return np.r_[
        matrix[:3, 3],
        [quaternion.w, quaternion.x, quaternion.y, quaternion.z],
    ]


def _new_solver(side: str) -> X7SPinocchioIK:
    if side == "left":
        joints = [f"joint{number}" for number in range(5, 12)]
        ee_joint = "joint11"
    else:
        joints = [f"joint{number}" for number in range(14, 21)]
        ee_joint = "joint20"
    return X7SPinocchioIK(
        X7S_ROOT / "urdf" / "X7S.urdf",
        [str(X7S_ROOT.parent)],
        joints_to_keep=joints,
        ee_joint_name=ee_joint,
    )


def _run_case(
    label: str,
    translation: np.ndarray | None = None,
    rotation_vector: np.ndarray | None = None,
) -> tuple[bool, list[str]]:
    passed = True
    lines = []
    for side, ready in (("left", LEFT_READY), ("right", RIGHT_READY)):
        solver = _new_solver(side)
        initial = solver.forward_pose(ready)
        target = initial.copy()
        if translation is not None:
            target[:3, 3] += translation
        if rotation_vector is not None:
            target[:3, :3] = pin.exp3(rotation_vector) @ initial[:3, :3]
            # A controller rotation is about its own origin, so it must not
            # rotate the Cartesian position around the robot origin.
            target[:3, 3] = initial[:3, 3]

        joints, success = solver.solve_pose_to_joints(
            _pose_vector(target),
            warm_start=ready,
        )
        achieved = solver.forward_pose(joints[0])
        actual_translation = achieved[:3, 3] - initial[:3, 3]
        actual_rotation = np.asarray(
            pin.log3(achieved[:3, :3] @ initial[:3, :3].T)
        )
        position_error = float(
            np.linalg.norm(achieved[:3, 3] - target[:3, 3])
        )
        orientation_error = float(
            np.linalg.norm(
                pin.log3(achieved[:3, :3].T @ target[:3, :3])
            )
        )
        case_passed = bool(success[0]) and position_error <= 0.015
        if translation is not None:
            intended_axis = int(np.argmax(np.abs(translation)))
            cross_axis = np.delete(actual_translation, intended_axis)
            case_passed = case_passed and bool(
                np.max(np.abs(cross_axis)) <= 0.008
            )
        else:
            case_passed = case_passed and bool(
                np.linalg.norm(actual_translation) <= 0.008
                and orientation_error <= 0.30
            )
        passed = passed and case_passed
        lines.append(
            f"{side:5s} {'PASS' if case_passed else 'FAIL'} "
            f"dp={np.round(actual_translation, 4)} "
            f"dr={np.round(actual_rotation, 4)} "
            f"pos_err={position_error:.5f} rot_err={orientation_error:.5f}"
        )
    return passed, [label, *lines]


def _run_trajectory(side: str, ready: np.ndarray) -> tuple[bool, list[str]]:
    """Track a smooth mixed translation/rotation path like a live controller."""
    solver = _new_solver(side)
    initial = solver.forward_pose(ready)
    current_q = ready.copy()
    # Establish the same first warm-started anchor used by live teleoperation.
    solver.solve_pose_to_joints(_pose_vector(initial), warm_start=current_q)
    max_position_error = 0.0
    max_orientation_error = 0.0
    max_joint_step = 0.0
    failures = 0
    trajectory_frames = 240
    for frame in range(1, trajectory_frames + 1):
        phase = 2.0 * np.pi * frame / trajectory_frames
        target = initial.copy()
        target[:3, 3] += np.asarray(
            [
                0.08 * np.sin(phase),
                0.05 * np.sin(2.0 * phase),
                0.04 * (1.0 - np.cos(phase)),
            ]
        )
        rotation_vector = np.asarray(
            [
                np.deg2rad(25.0) * np.sin(phase),
                np.deg2rad(20.0) * np.sin(2.0 * phase),
                np.deg2rad(30.0) * (1.0 - np.cos(phase)) * 0.5,
            ]
        )
        target[:3, :3] = pin.exp3(rotation_vector) @ initial[:3, :3]
        solved, success = solver.solve_pose_to_joints(
            _pose_vector(target),
            warm_start=current_q,
        )
        next_q = solved[0]
        achieved = solver.forward_pose(next_q)
        max_position_error = max(
            max_position_error,
            float(np.linalg.norm(achieved[:3, 3] - target[:3, 3])),
        )
        max_orientation_error = max(
            max_orientation_error,
            float(
                np.linalg.norm(
                    pin.log3(achieved[:3, :3].T @ target[:3, :3])
                )
            ),
        )
        max_joint_step = max(
            max_joint_step,
            float(np.max(np.abs(next_q - current_q))),
        )
        failures += int(not success[0])
        current_q = next_q
    passed = bool(
        failures == 0
        and max_position_error <= 0.025
        and max_orientation_error <= 0.30
        and max_joint_step <= 0.040001
    )
    return passed, [
        f"mixed {trajectory_frames}-frame trajectory ({side})",
        f"{side:5s} {'PASS' if passed else 'FAIL'} failures={failures} "
        f"max_pos_err={max_position_error:.5f} "
        f"max_rot_err={max_orientation_error:.5f} "
        f"max_joint_step={max_joint_step:.5f}",
    ]


def _run_extreme_wrist_case(
    side: str,
    ready: np.ndarray,
) -> tuple[bool, list[str]]:
    """An unreachable wrist rotation must not freeze reachable translation."""
    solver = _new_solver(side)
    initial = solver.forward_pose(ready)
    target = initial.copy()
    target[:3, 3] += np.asarray([0.03, 0.0, 0.02])
    target[:3, :3] = (
        pin.exp3(np.asarray([0.0, 0.0, np.deg2rad(170.0)]))
        @ initial[:3, :3]
    )
    solved, success = solver.solve_pose_to_joints(
        _pose_vector(target),
        warm_start=ready,
    )
    achieved = solver.forward_pose(solved[0])
    position_error = float(
        np.linalg.norm(achieved[:3, 3] - target[:3, 3])
    )
    orientation_error = float(
        np.linalg.norm(pin.log3(achieved[:3, :3].T @ target[:3, :3]))
    )
    translation_motion = float(
        np.linalg.norm(achieved[:3, 3] - initial[:3, 3])
    )
    passed = bool(
        success[0]
        and position_error <= 0.015
        and translation_motion >= 0.025
    )
    return passed, [
        f"extreme wrist rotation with translation ({side})",
        f"{side:5s} {'PASS' if passed else 'FAIL'} "
        f"pos_err={position_error:.5f} rot_err={orientation_error:.5f} "
        f"translation_motion={translation_motion:.5f}",
    ]


def main() -> int:
    overall = True
    cases = []
    for axis, name in enumerate("xyz"):
        for sign in (1.0, -1.0):
            translation = np.zeros(3)
            translation[axis] = sign * 0.05
            cases.append(
                _run_case(
                    f"translate {'+' if sign > 0 else '-'}{name}",
                    translation=translation,
                )
            )
    cases.append(_run_trajectory("left", LEFT_READY))
    cases.append(_run_trajectory("right", RIGHT_READY))
    cases.append(_run_extreme_wrist_case("left", LEFT_READY))
    cases.append(_run_extreme_wrist_case("right", RIGHT_READY))
    for axis, name in enumerate("xyz"):
        for sign in (1.0, -1.0):
            rotation = np.zeros(3)
            rotation[axis] = sign * np.deg2rad(15.0)
            cases.append(
                _run_case(
                    f"rotate {'+' if sign > 0 else '-'}{name}",
                    rotation_vector=rotation,
                )
            )
    for passed, lines in cases:
        overall = overall and passed
        print("\n".join(lines), flush=True)
    print(f"X7S IK regression: {'PASS' if overall else 'FAIL'}", flush=True)
    return 0 if overall else 1


if __name__ == "__main__":
    raise SystemExit(main())
