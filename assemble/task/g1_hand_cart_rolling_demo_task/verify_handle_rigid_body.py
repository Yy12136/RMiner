"""Verify the six hand-cart handle colliders belong to a valid runtime rigid body."""

import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
simulation_app = AppLauncher(args).app

import json
import gymnasium as gym
from pxr import Usd, UsdGeom, UsdPhysics

import tasks.g1_tasks
from tasks.g1_tasks.hand_cart_rolling_demo_g1_29dof_dex1 import (
    hand_cart_rolling_demo_g1_29dof_dex1_env_cfg as demo_cfg,
)

cfg = demo_cfg.HandCartRollingDemoG129Dex1EnvCfg()
cfg.sim.device = args.device
env = gym.make(demo_cfg.TASK_ID, cfg=cfg).unwrapped
env.sim.reset()
env.reset()
controller = cfg.create_runtime_controller(env)
stage = controller.stage
root_path = controller.CART_PATH
root = stage.GetPrimAtPath(root_path)
root_body = UsdPhysics.RigidBodyAPI(root)

assert root and root.HasAPI(UsdPhysics.RigidBodyAPI)
assert bool(root_body.GetRigidBodyEnabledAttr().Get())
assert bool(root_body.GetKinematicEnabledAttr().Get())
assert controller.cart_body.is_physics_handle_valid()

rigid_paths = []
handle_colliders = []
all_colliders = []
for prim in Usd.PrimRange(root):
    if prim.HasAPI(UsdPhysics.RigidBodyAPI):
        rigid_paths.append(str(prim.GetPath()))
    if prim.HasAPI(UsdPhysics.CollisionAPI):
        enabled = bool(UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get())
        item = {
            "path": str(prim.GetPath()),
            "type": prim.GetTypeName(),
            "enabled": enabled,
        }
        all_colliders.append(item)
        if prim.GetName().startswith(("Handle_Left_", "Handle_Right_")):
            capsule = UsdGeom.Capsule(prim)
            item.update(
                radius_m=float(capsule.GetRadiusAttr().Get()),
                height_m=float(capsule.GetHeightAttr().Get()),
                axis=str(capsule.GetAxisAttr().Get()),
            )
            handle_colliders.append(item)

assert rigid_paths == [root_path], rigid_paths
assert len(all_colliders) == 6, all_colliders
assert len(handle_colliders) == 6, handle_colliders
assert all(item["enabled"] for item in handle_colliders)
assert all(item["type"] == "Capsule" for item in handle_colliders)
assert all(abs(item["radius_m"] - 0.013) < 1.0e-9 for item in handle_colliders)

report = {
    "task_id": demo_cfg.TASK_ID,
    "rigid_body_path": root_path,
    "rigid_body_enabled": bool(root_body.GetRigidBodyEnabledAttr().Get()),
    "kinematic_enabled": bool(root_body.GetKinematicEnabledAttr().Get()),
    "physx_handle_valid": bool(controller.cart_body.is_physics_handle_valid()),
    "rigid_body_paths_under_cart": rigid_paths,
    "handle_colliders": handle_colliders,
    "interpretation": "The six handle capsules are collision shapes owned by the HandCart root rigid body.",
}
print("HANDLE_RIGID_BODY_PASS", json.dumps(report, ensure_ascii=False), flush=True)

env.close()
simulation_app.close()
