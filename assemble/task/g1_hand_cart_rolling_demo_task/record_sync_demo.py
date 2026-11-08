"""Record a third-person W-key synchronization demo for the rolling hand cart."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser()
parser.add_argument("--output", required=True)
parser.add_argument("--report", required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
simulation_app = AppLauncher(args).app

import cv2
import gymnasium as gym
import numpy as np
import torch

import carb.input
import tasks.g1_tasks
from tasks.g1_tasks.hand_cart_rolling_demo_g1_29dof_dex1 import (
    hand_cart_rolling_demo_g1_29dof_dex1_env_cfg as demo_cfg,
)


output_path = Path(args.output)
report_path = Path(args.report)
output_path.parent.mkdir(parents=True, exist_ok=True)
report_path.parent.mkdir(parents=True, exist_ok=True)
intermediate_path = output_path.with_name(output_path.stem + "_mp4v.mp4")

cfg = demo_cfg.HandCartRollingDemoG129Dex1EnvCfg()
cfg.sim.device = args.device
cfg.scene.world_camera.spawn.focal_length = 12.0
env = gym.make(demo_cfg.TASK_ID, cfg=cfg).unwrapped
env.sim.reset()
env.reset()
controller = cfg.create_runtime_controller(env)
camera = env.scene["world_camera"]
action = torch.zeros((1, env.action_manager.total_action_dim), device=env.device)

initial_base_xy = np.asarray([controller.x, controller.y], dtype=np.float64)
initial_cart_xy = controller._cart_position[:2].copy()
subject_xy = 0.5 * (initial_base_xy + initial_cart_xy)
# Use the scene's known-clear observation point and turn it toward the cart.
# Positions computed behind the robot can fall inside this scanned tunnel's
# curved wall even when they are valid in an open scene.
camera_eye = np.asarray([[-0.7, 3.8, 2.05]], dtype=np.float32)
camera_target = np.asarray([[subject_xy[0], subject_xy[1], 0.9]], dtype=np.float32)
camera.set_world_poses_from_view(
    torch.as_tensor(camera_eye, device=env.device),
    torch.as_tensor(camera_target, device=env.device),
)
max_sync_error = 0.0
max_body_error = 0.0


def read_rgb() -> np.ndarray:
    rgb = camera.data.output["rgb"][0]
    image = rgb.detach().cpu().numpy() if torch.is_tensor(rgb) else np.asarray(rgb)
    if image.dtype != np.uint8:
        image = np.clip(image * (255.0 if image.max() <= 1.0 else 1.0), 0, 255).astype(np.uint8)
    if image.shape[-1] == 4:
        image = image[..., :3]
    return np.ascontiguousarray(image)


def add_overlay(image: np.ndarray, phase: str, frame_index: int) -> tuple[np.ndarray, dict]:
    global max_sync_error, max_body_error
    base_delta = np.asarray([controller.x, controller.y], dtype=np.float64) - initial_base_xy
    cart_delta = controller._cart_position[:2] - initial_cart_xy
    sync_error = float(np.linalg.norm(base_delta - cart_delta))
    body_position, _ = controller.cart_body.get_world_poses()
    body_xy = body_position[0, :2].detach().cpu().numpy()
    body_error = float(np.linalg.norm(body_xy - controller._cart_position[:2]))
    max_sync_error = max(max_sync_error, sync_error)
    max_body_error = max(max_body_error, body_error)

    forward = float(
        (controller.x - controller.initial[0]) * demo_cfg.TASK_FORWARD_X
        + (controller.y - controller.initial[1]) * demo_cfg.TASK_FORWARD_Y
    )
    cart_forward = float(
        cart_delta[0] * demo_cfg.TASK_FORWARD_X
        + cart_delta[1] * demo_cfg.TASK_FORWARD_Y
    )
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.rectangle(bgr, (12, 12), (625, 125), (15, 15, 15), -1)
    color = (40, 220, 255) if phase == "W PRESSED" else (225, 225, 225)
    cv2.putText(bgr, phase, (28, 43), cv2.FONT_HERSHEY_SIMPLEX, 0.78, color, 2, cv2.LINE_AA)
    cv2.putText(
        bgr,
        f"robot forward: {forward:+.3f} m   cart forward: {cart_forward:+.3f} m",
        (28, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.56,
        (240, 240, 240),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        bgr,
        f"sync error: {sync_error:.6f} m   rigid-body error: {body_error:.6f} m",
        (28, 104),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (90, 240, 120),
        1,
        cv2.LINE_AA,
    )
    return bgr, {
        "frame": frame_index,
        "phase": phase,
        "robot_forward_m": forward,
        "cart_forward_m": cart_forward,
        "sync_error_m": sync_error,
        "body_error_m": body_error,
    }


# Warm up sensors and PhysX before opening the video writer.
for _ in range(5):
    controller.update(0.01)
    env.step(action)

first = read_rgb()
height, width = first.shape[:2]
writer = cv2.VideoWriter(
    str(intermediate_path), cv2.VideoWriter_fourcc(*"mp4v"), 25.0, (width, height)
)
if not writer.isOpened():
    raise RuntimeError(f"Could not open video writer for {intermediate_path}")

samples = []
frame_index = 0


def record_phase(label: str, frames: int, press_w: bool, controller_dt: float) -> None:
    global frame_index
    if press_w:
        controller.held.add(carb.input.KeyboardInput.W)
    else:
        controller.held.discard(carb.input.KeyboardInput.W)
    for _ in range(frames):
        controller.update(controller_dt)
        env.step(action)
        bgr, sample = add_overlay(read_rgb(), label, frame_index)
        writer.write(bgr)
        samples.append(sample)
        frame_index += 1


record_phase("READY", 20, False, 0.02)
record_phase("W PRESSED", 120, True, 0.02)
record_phase("W RELEASED", 30, False, 0.02)
controller.held.discard(carb.input.KeyboardInput.W)
writer.release()

final_base_delta = np.asarray([controller.x, controller.y], dtype=np.float64) - initial_base_xy
final_cart_delta = controller._cart_position[:2] - initial_cart_xy
assert np.linalg.norm(final_base_delta - final_cart_delta) < 1.0e-8
assert max_body_error < 1.0e-4
assert samples[-1]["robot_forward_m"] > 0.1

subprocess.run(
    [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-i",
        str(intermediate_path),
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "20",
        "-pix_fmt",
        "yuv420p",
        str(output_path),
    ],
    check=True,
)
intermediate_path.unlink(missing_ok=True)

report = {
    "task_id": demo_cfg.TASK_ID,
    "frames": frame_index,
    "fps": 25,
    "controller_dt_s": 0.02,
    "final_robot_delta_xy_m": final_base_delta.tolist(),
    "final_cart_delta_xy_m": final_cart_delta.tolist(),
    "final_forward_m": samples[-1]["robot_forward_m"],
    "max_sync_error_m": max_sync_error,
    "max_rigid_body_error_m": max_body_error,
    "wheel_left_angle_rad": float(controller._wheel_angles_rad["Wheel_Left"]),
    "wheel_right_angle_rad": float(controller._wheel_angles_rad["Wheel_Right"]),
}
report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
print("HAND_CART_SYNC_VIDEO_PASS", json.dumps(report, separators=(",", ":")), flush=True)

env.close()
simulation_app.close()
