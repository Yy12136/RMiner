"""Launch the standalone rolling-demo task and verify its visual kinematics."""

import argparse

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
simulation_app = AppLauncher(
    parser.parse_args(["--headless", "--enable_cameras", "--device", "cpu"])
).app

import gymnasium as gym
import torch
from pxr import Usd, UsdPhysics

import carb.input

import tasks.g1_tasks  # Registers the standalone task after Kit has initialized.
from tasks.g1_tasks.hand_cart_rolling_demo_g1_29dof_dex1 import (
    hand_cart_rolling_demo_g1_29dof_dex1_env_cfg as demo_cfg,
)


cfg = demo_cfg.HandCartRollingDemoG129Dex1EnvCfg()
cfg.sim.device = "cpu"
env = gym.make(demo_cfg.TASK_ID, cfg=cfg).unwrapped
env.sim.reset()
env.reset()
controller = cfg.create_runtime_controller(env)

cart_root = controller.stage.GetPrimAtPath(controller.CART_PATH)
assert cart_root
assert UsdPhysics.RigidBodyAPI(cart_root).GetRigidBodyEnabledAttr().Get()
assert UsdPhysics.RigidBodyAPI(cart_root).GetKinematicEnabledAttr().Get()
for body_name in ("Chassis", "Wheel_Left", "Wheel_Right"):
    prim = controller.stage.GetPrimAtPath(f"{controller.CART_PATH}/{body_name}")
    assert prim and not prim.HasAPI(UsdPhysics.RigidBodyAPI)
for collision_prim in Usd.PrimRange(cart_root):
    if collision_prim.HasAPI(UsdPhysics.CollisionAPI):
        expected_collision = collision_prim.GetName().startswith(("Handle_Left_", "Handle_Right_"))
        assert bool(UsdPhysics.CollisionAPI(collision_prim).GetCollisionEnabledAttr().Get()) == expected_collision

initial_cart = controller._cart_position.copy()
controller.held.add(carb.input.KeyboardInput.W)
for _ in range(160):
    controller.update(0.01)
    env.step(torch.zeros((1, env.action_manager.total_action_dim), device=env.device))
controller.held.clear()

cart_delta = controller._cart_position - initial_cart
final_chassis_world, _ = controller.cart_body.get_world_poses()
base_delta = torch.tensor(
    [controller.x - controller.initial[0], controller.y - controller.initial[1]],
    dtype=torch.float64,
).numpy()
assert abs(cart_delta[0] - base_delta[0]) < 1.0e-9
assert abs(cart_delta[1] - base_delta[1]) < 1.0e-9
assert torch.allclose(
    final_chassis_world[0, :2],
    torch.as_tensor(controller._cart_position[:2], dtype=torch.float32, device=env.device),
    atol=1.0e-5,
)
left = controller._wheel_angles_rad["Wheel_Left"]
right = controller._wheel_angles_rad["Wheel_Right"]
assert left > 0.1 and right > 0.1
assert abs(left - right) < 1.0e-6
assert controller.track_drive.max_track_speed <= demo_cfg.DEMO_MAX_TRACK_SPEED_MPS
assert controller.track_drive.max_track_speed == 0.18
wheel_op = controller.wheel_orient_ops["Wheel_Left"].Get()
assert abs(wheel_op.GetImaginary()[0]) > 0.01

print(
    "ROLLING_DEMO_SMOKE_PASS",
    {
        "cart_delta_xy_m": cart_delta[:2].tolist(),
        "wheel_angles_rad": [left, right],
        "max_track_speed_mps": controller.track_drive.max_track_speed,
        "wheel_radius_m": demo_cfg.WHEEL_RADIUS_M,
    },
)
env.close()
simulation_app.close()
