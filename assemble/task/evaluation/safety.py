"""Pure safety functions shared by the bridge and unit tests."""

from __future__ import annotations

import numpy as np

MOBILE_LEGAL_ACTIONS = np.asarray(((0.0, 0.0, 0.0), (1.2, 1.2, 0.0)), dtype=np.float64)


def project_mobile_action(raw: np.ndarray) -> tuple[np.ndarray, bool]:
    raw = np.asarray(raw, dtype=np.float64).reshape(3)
    distances = np.linalg.norm(MOBILE_LEGAL_ACTIONS - raw[None, :], axis=1)
    projected = MOBILE_LEGAL_ACTIONS[int(np.argmin(distances))].copy()
    return projected, not np.allclose(raw, projected, atol=1e-8)


def apply_right_only_lock(action: np.ndarray, locked_left: np.ndarray) -> np.ndarray:
    action = np.asarray(action, dtype=np.float64).copy().reshape(16)
    locked_left = np.asarray(locked_left, dtype=np.float64).reshape(8)
    action[:7], action[14] = locked_left[:7], locked_left[7]
    return action


def sanitize_joint_action(requested, previous, lower, upper, velocity_limit, dt, gripper_delta):
    requested = np.asarray(requested, dtype=np.float64).reshape(-1)
    previous = np.asarray(previous, dtype=np.float64).reshape(-1)
    if requested.shape != (16,) or previous.shape != (16,) or not np.isfinite(requested).all():
        raise ValueError("Joint policy action must be a finite 16-vector")
    dt = float(np.clip(dt, 1 / 250, 0.1))
    arm = np.clip(requested[:14], lower, upper)
    arm = np.clip(arm, previous[:14] - velocity_limit * dt, previous[:14] + velocity_limit * dt)
    grip = np.clip(requested[14:], 0.0, 5.4)
    grip = np.clip(grip, previous[14:] - gripper_delta, previous[14:] + gripper_delta)
    return np.concatenate((arm, grip))


def synchronized_displacement_error(previous: np.ndarray, current: np.ndarray) -> float:
    """Compare Task04 base-plane and cart-XY displacement magnitudes."""
    previous, current = np.asarray(previous), np.asarray(current)
    base_delta = np.linalg.norm(current[16:18] - previous[16:18])
    cart_delta = np.linalg.norm(current[23:25] - previous[23:25])
    return float(abs(base_delta - cart_delta))
