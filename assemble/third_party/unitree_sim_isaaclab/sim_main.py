
# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0  
#!/usr/bin/env python3
# main.py
import os

project_root = os.path.dirname(os.path.abspath(__file__))
os.environ["PROJECT_ROOT"] = project_root

import argparse
import contextlib
import json
import time
import sys
import signal
import torch
import gymnasium as gym
from pathlib import Path

# Isaac Lab AppLauncher
from isaaclab.app import AppLauncher

from teleimager.image_server import run_isaacsim_server
from dds.dds_create import create_dds_objects,create_dds_objects_replay
# add command line arguments
parser = argparse.ArgumentParser(description="Unitree Simulation")
parser.add_argument("--task", type=str, default="Isaac-PickPlace-G129-Head-Waist-Fix", help="task name")
parser.add_argument("--action_source", type=str, default="dds", 
                   choices=["dds", "file", "trajectory", "policy", "replay","dds_wholebody"], 
                   help="Action source")


parser.add_argument("--robot_type", type=str, default="g129", help="robot type")
parser.add_argument("--enable_dex1_dds", action="store_true", help="enable gripper DDS")
parser.add_argument("--enable_dex3_dds", action="store_true", help="enable dexterous hand DDS")
parser.add_argument("--enable_inspire_dds", action="store_true", help="enable inspire hand DDS")
parser.add_argument("--stats_interval", type=float, default=10.0, help="statistics print interval (seconds)")

parser.add_argument("--file_path", type=str, default="/home/unitree/Code/xr_teleoperate/teleop/utils/data", help="file path (when action_source=file)")
parser.add_argument("--generate_data_dir", type=str, default="./data", help="save data dir")
parser.add_argument("--generate_data", action="store_true", default=False, help="generate data")
parser.add_argument("--rerun_log", action="store_true", default=False, help="rerun log")
parser.add_argument("--replay_data",  action="store_true", default=False, help="replay data")

parser.add_argument("--modify_light",  action="store_true", default=False, help="modify light")
parser.add_argument("--modify_camera",  action="store_true", default=False,    help="modify camera")

# performance analysis parameters
parser.add_argument("--step_hz", type=int, default=100, help="control frequency")
parser.add_argument("--enable_profiling", action="store_true", default=True, help="enable performance analysis")
parser.add_argument("--profile_interval", type=int, default=500, help="performance analysis report interval (steps)")

parser.add_argument("--model_path", type=str, default="assets/model/policy.onnx", help="model path")
parser.add_argument("--reward_interval", type=int, default=10, help="step interval for reward calculation")
parser.add_argument("--enable_wholebody_dds", action="store_true", default=False, help="enable wh dds")

parser.add_argument("--physics_dt", type=float, default=None, help="physics time step, e.g., 0.005")
parser.add_argument("--render_interval", type=int, default=None, help="render interval steps (>=1)")
parser.add_argument("--camera_write_interval", type=int, default=None, help="camera write interval steps (>=1)")


parser.add_argument("--no_render",action="store_true",default=False,help="disable rendering updates entirely (overrides render interval)",)
parser.add_argument("--public_ip",type=str,default="127.0.0.1",help="public ip")
parser.add_argument("--livestream_type", type=int, default=2, help="livestream type (0: no livestream, 1: WebRTC public network, 2:  WebRTC private network)")

parser.add_argument("--solver_iterations", type=int, default=None, help="physx solver iteration count (e.g., 4)")
parser.add_argument("--gravity_z", type=float, default=None, help="override gravity z (e.g., -9.8)")
parser.add_argument("--skip_cvtcolor", action="store_true", default=False, help="skip cv2.cvtColor if upstream already BGR")

parser.add_argument("--camera_jpeg", action="store_true", default=True, help="enable JPEG compression for camera frames")
parser.add_argument("--camera_jpeg_quality", type=int, default=85, help="JPEG quality (1-100)")

parser.add_argument("--physx_substeps", type=int, default=None, help="physx substeps per step")
parser.add_argument("--camera_include", type=str, default="front_camera,left_wrist_camera,right_wrist_camera", help="comma-separated camera names to enable")
parser.add_argument("--camera_exclude", type=str, default="world_camera", help="comma-separated camera names to disable")

parser.add_argument("--env_reward_interval", type=int, default=5, help="environment reward compute interval (steps)")
parser.add_argument("--seed", type=int, default=42, help="environment seed")
# add AppLauncher parameters
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if args_cli.no_render:
    os.environ["LIVESTREAM"] = str(args_cli.livestream_type)
    os.environ["PUBLIC_IP"] = args_cli.public_ip
else:
    os.environ["LIVESTREAM"] = "0"

if args_cli.enable_dex3_dds and args_cli.enable_dex1_dds and args_cli.enable_inspire_dds:
    print("Error: enable_dex3_dds and enable_dex1_dds and enable_inspire_dds cannot be enabled at the same time")
    print("Please select one of the options")
    sys.exit(1)


import pinocchio                 
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from layeredcontrol.robot_control_system import (
    RobotController, 
    ControlConfig,
)

from dds.reset_pose_dds import *
import tasks
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

from tools.augmentation_utils import (
    update_light,
    batch_augment_cameras_by_name,
)

from tools.data_json_load import sim_state_to_json
from dds.sim_state_dds import *
from action_provider.create_action_provider import create_action_provider
from tools.get_stiffness import get_robot_stiffness_from_env
from tools.get_reward import get_step_reward_value,get_current_rewards


def _load_replay_initial_state(path_value, device):
    """Load one replay-ready metadata snapshot as Isaac Lab tensors."""

    if not path_value:
        return None, None
    path = Path(path_value)
    payload = json.loads(path.read_text(encoding="utf-8"))
    sim_state = payload.get("sim_state") or {}
    initial = sim_state.get("init_state")
    if not isinstance(initial, dict):
        raise ValueError(f"Replay metadata has no sim_state.init_state: {path}")

    def convert(value):
        if isinstance(value, dict):
            return {key: convert(item) for key, item in value.items()}
        if isinstance(value, list):
            return torch.as_tensor(value, dtype=torch.float32, device=device)
        return value

    return convert(initial), payload


def _synchronize_runtime_controller(runtime_controller, replay_payload):
    """Align pose-driven task controllers with the restored scene snapshot."""

    if runtime_controller is None or replay_payload is None:
        return
    runtime = (replay_payload.get("sim_state") or {}).get("runtime_state") or {}
    replay_initial = runtime.get("replay_initial_state") or {}
    robot = replay_initial.get("robot") or {}
    pose = robot.get("tracked_base_pose_xy_yaw") or runtime.get("tracked_base_pose")
    if pose and len(pose) == 3:
        for name, value in zip(("x", "y", "yaw"), pose):
            if hasattr(runtime_controller, name):
                setattr(runtime_controller, name, float(value))
    lift = robot.get("lift_height_m", runtime.get("lift_height_m"))
    if lift is not None and hasattr(runtime_controller, "lift_height"):
        runtime_controller.lift_height = float(lift)
    apply_method = getattr(runtime_controller, "apply", None)
    if callable(apply_method):
        apply_method()
    print(
        "[replay-initial-state] runtime controller synchronized from metadata",
        flush=True,
    )


def _capture_static_snapshot_initial_state(env, runtime_controller):
    """Remember the initial PhysX and visual-link frames for snapshot calibration.

    PhysX body poses use a center-of-mass/body frame while the USD link prims
    may have a fixed authoring offset.  A later snapshot therefore needs both
    the initial PhysX frame (P0) and the initial USD visual frame (V0) in order
    to author ``Vt = Pt * inverse(P0) * V0``.  This is only collected for the
    explicit static-snapshot workflow; normal simulation behavior is unchanged.
    """
    if not os.environ.get("G1_STATIC_SNAPSHOT_REQUEST_FILE"):
        return
    try:
        from pxr import Usd, UsdGeom

        stage = __import__("omni.usd", fromlist=["get_context"]).get_context().get_stage()
        visual_world = {}
        if stage is not None:
            for root_path in (
                "/World/TrackedBase",
                "/World/envs/env_0/Leiguan",
                "/World/envs/env_0/ChaganShengziHolder",
                "/World/envs/env_0/ChaganShengzi",
            ):
                root = stage.GetPrimAtPath(root_path)
                if not root or not root.IsValid():
                    continue
                for prim in Usd.PrimRange(root):
                    if prim.IsA(UsdGeom.Xformable):
                        visual_world[str(prim.GetPath())] = UsdGeom.Xformable(
                            prim
                        ).ComputeLocalToWorldTransform(Usd.TimeCode.Default())

        robot = env.scene["robot"]
        robot_data = robot.data
        body_state = robot_data.body_state_w[0].detach().cpu().clone()
        body_names = list(robot.body_names)
        task_initial = {}

        def capture_view(view, paths):
            positions, orientations = view.get_world_poses(clone=True)
            return {
                str(path): (
                    positions[index].detach().cpu().clone(),
                    orientations[index].detach().cpu().clone(),
                )
                for index, path in enumerate(paths)
            }

        def capture_asset(asset, path):
            state = asset.data.root_state_w[0].detach().cpu().reshape(-1)
            return {
                str(path): (state[0:3].clone(), state[3:7].clone())
            }

        if runtime_controller is not None:
            task_initial.update(
                capture_view(
                    runtime_controller.vehicle_body,
                    ["/World/TrackedBase/Vehicle"],
                )
            )
            task_initial.update(
                capture_view(
                    runtime_controller.carriage_body,
                    ["/World/TrackedBase/LiftCarriage"],
                )
            )
            task_initial.update(
                capture_view(runtime_controller.rod_body, [runtime_controller.ROD_PATH])
            )
            rope_paths = sorted(
                str(prim.GetPath())
                for prim in Usd.PrimRange(
                    stage.GetPrimAtPath(
                        "/World/envs/env_0/ChaganShengzi/rope_40cm/Links"
                    )
                )
                if prim.IsA(UsdGeom.Xformable)
                and prim.GetName().startswith("Link_")
            )
            task_initial.update(
                capture_view(runtime_controller.rope_bodies, rope_paths)
            )
            task_initial.update(
                capture_asset(runtime_controller.leiguan, "/World/envs/env_0/Leiguan")
            )
            task_initial.update(
                capture_asset(
                    runtime_controller.holder,
                    "/World/envs/env_0/ChaganShengziHolder",
                )
            )

        env._static_snapshot_initial_robot_body_state_w = body_state
        env._static_snapshot_initial_robot_body_names = body_names
        env._static_snapshot_initial_task_poses = task_initial
        env._static_snapshot_initial_visual_world = visual_world
        print(
            "[static-snapshot] captured initial calibration frames: "
            f"robot={len(body_names)} task={len(task_initial)} visuals={len(visual_world)}",
            flush=True,
        )
    except Exception as exc:
        print(f"[static-snapshot] initial calibration capture failed: {exc!r}", flush=True)
        raise


def _restore_replay_scene(scene, replay_initial_state):
    """Restore a scene without sending wake-up velocity writes to kinematics."""

    rigid_objects = getattr(scene, "_rigid_objects", {})
    kinematic = {}
    for name, asset in list(rigid_objects.items()):
        rigid_props = getattr(getattr(getattr(asset, "cfg", None), "spawn", None), "rigid_props", None)
        if bool(getattr(rigid_props, "kinematic_enabled", False)):
            kinematic[name] = rigid_objects.pop(name)
    try:
        scene.reset_to(replay_initial_state, is_relative=False)
    finally:
        rigid_objects.update(kinematic)
    for name, asset in kinematic.items():
        asset_state = replay_initial_state["rigid_object"][name]
        asset.write_root_pose_to_sim(asset_state["root_pose"])
    if kinematic:
        print(
            "[replay-initial-state] restored kinematic rigid objects without "
            f"velocity wake-up: {sorted(kinematic)}",
            flush=True,
        )


def _capture_requested_highres_image(stage):
    """Render one native high-resolution image from the live replay stage.

    This is deliberately opt-in.  The normal camera shared-memory path stays
    at its dataset resolution; a snapshot request can additionally create a
    one-shot Replicator render product on the existing head camera while the
    Flow volume is still live in the running Isaac process.
    """
    output_value = os.environ.get("G1_STATIC_SNAPSHOT_IMAGE_OUTPUT", "").strip()
    if not output_value:
        return None
    camera_path = os.environ.get(
        "G1_STATIC_SNAPSHOT_IMAGE_CAMERA",
        "/World/TrackedBase/G1UpperBody/d435_link/front_cam",
    ).strip()
    width = int(os.environ.get("G1_STATIC_SNAPSHOT_IMAGE_WIDTH", "3840"))
    height = int(os.environ.get("G1_STATIC_SNAPSHOT_IMAGE_HEIGHT", "2880"))
    subframes = int(os.environ.get("G1_STATIC_SNAPSHOT_IMAGE_SUBFRAMES", "8"))
    if width <= 0 or height <= 0 or subframes <= 0:
        raise ValueError(
            "G1_STATIC_SNAPSHOT_IMAGE_WIDTH/HEIGHT/SUBFRAMES must be positive"
        )
    camera = stage.GetPrimAtPath(camera_path)
    if not camera or not camera.IsValid():
        raise RuntimeError(f"snapshot camera not found: {camera_path}")

    import numpy as np
    import omni.replicator.core as rep
    from PIL import Image

    # Let the just-updated PhysX/Flow state and materials reach RTX before
    # sampling.  This product is independent of the 640x480 SHM products.
    for _ in range(8):
        simulation_app.update()
    render_product = rep.create.render_product(camera, (width, height))
    annotator = rep.AnnotatorRegistry.get_annotator("rgb")
    annotator.attach([render_product])
    rep.orchestrator.step(rt_subframes=subframes)
    for _ in range(8):
        simulation_app.update()
    data = annotator.get_data()
    if data is None or getattr(data, "size", 0) == 0:
        raise RuntimeError("high-resolution RTX snapshot returned no pixels")
    array = np.asarray(data)
    if array.ndim != 3 or array.shape[0] != height or array.shape[1] != width:
        raise RuntimeError(
            f"unexpected high-resolution snapshot shape: {tuple(array.shape)}"
        )
    output_path = Path(output_value).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array[..., :3].astype(np.uint8), mode="RGB").save(
        output_path, format="PNG"
    )
    print(
        f"[static-snapshot] high-resolution image written to {output_path} "
        f"({width}x{height}, camera={camera_path})",
        flush=True,
    )
    return {
        "output": str(output_path),
        "camera": camera_path,
        "width": width,
        "height": height,
        "rt_subframes": subframes,
    }


def _export_requested_static_snapshot(request_path, status_path, env, runtime_controller=None):
    """Export the currently composed native USD stage on an explicit request.

    This hook is intentionally opt-in through environment variables.  It is
    used by the Task01 grasp-pose capture workflow to freeze one real replay
    frame without changing the normal controller/replay behavior.  The stage
    is exported before the main loop is frozen, so PhysX/Fabric transforms that
    were just rendered are included in the native USD snapshot.
    """
    if request_path is None or not request_path.is_file():
        return None
    try:
        request = json.loads(request_path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"[static-snapshot] invalid request: {exc}", flush=True)
        return None
    output_value = str(request.get("output", "")).strip()
    token = str(request.get("token", output_value)).strip()
    if not output_value:
        print("[static-snapshot] request has no output path", flush=True)
        return None

    import omni.usd

    stage = omni.usd.get_context().get_stage()
    if stage is None:
        print("[static-snapshot] no active USD stage", flush=True)
        return None
    output_path = Path(output_value).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Isaac's physics bridge normally updates the composed stage for rendering,
    # but explicitly enabling this setting makes the export deterministic for
    # articulation/rigid-body transforms on versions where Fabric is deferred.
    try:
        import carb

        settings = carb.settings.get_settings()
        settings.set("/physics/updateToUsd", True)
        settings.set("/physics/updateVelocitiesToUsd", True)
    except Exception as exc:
        print(f"[static-snapshot] updateToUsd setting unavailable: {exc}", flush=True)
    try:
        env.sim.forward()
    except Exception:
        pass
    if os.environ.get("G1_STATIC_SNAPSHOT_DEBUG", "0") == "1":
        try:
            robot = env.scene["robot"]
            data = getattr(robot, "data", None)
            print(
                "[static-snapshot-debug] robot="
                f"{type(robot).__name__} prim_path={getattr(robot.cfg, 'prim_path', None)} "
                f"num_bodies={getattr(robot, 'num_bodies', None)}",
                flush=True,
            )
            print(
                "[static-snapshot-debug] body_names="
                f"{list(getattr(robot, 'body_names', []))}",
                flush=True,
            )
            for name in (
                "root_state_w",
                "body_state_w",
                "body_pos_w",
                "body_quat_w",
                "joint_pos",
                "joint_vel",
            ):
                value = getattr(data, name, None) if data is not None else None
                if value is None:
                    continue
                print(
                    f"[static-snapshot-debug] data.{name} shape={tuple(value.shape)} "
                    f"sample={value.detach().cpu().reshape(-1)[:12].tolist()}",
                    flush=True,
                )
        except Exception as exc:
            print(f"[static-snapshot-debug] failed: {exc!r}", flush=True)

    # Isaac Lab keeps articulation and RigidPrim poses in PhysX tensors.  In
    # this task those poses are not necessarily mirrored into USD xform ops by
    # stage.Export(), especially for an articulation driven through DDS.  Bake
    # the actual world pose into a reset-stack transform on each dynamic body
    # before exporting.  This branch is only reached for the explicit snapshot
    # request; the normal simulator path is untouched.
    try:
        from pxr import Gf, Usd, UsdGeom

        def pose_matrix(position, orientation):
            position = position.detach().cpu().reshape(-1)
            orientation = orientation.detach().cpu().reshape(-1)
            matrix = Gf.Matrix4d(1.0)
            matrix.SetRotate(
                Gf.Quatd(
                    float(orientation[0]),
                    Gf.Vec3d(
                        float(orientation[1]),
                        float(orientation[2]),
                        float(orientation[3]),
                    ),
                )
            )
            matrix.SetTranslateOnly(
                Gf.Vec3d(
                    float(position[0]),
                    float(position[1]),
                    float(position[2]),
                )
            )
            return matrix

        initial_visual_world = getattr(
            env, "_static_snapshot_initial_visual_world", {}
        )
        initial_robot_state = getattr(
            env, "_static_snapshot_initial_robot_body_state_w", None
        )
        initial_robot_names = getattr(
            env, "_static_snapshot_initial_robot_body_names", []
        )
        initial_task_poses = getattr(
            env, "_static_snapshot_initial_task_poses", {}
        )

        def author_world_pose(path, position, orientation, initial_pose=None):
            prim = stage.GetPrimAtPath(path)
            if not prim or not prim.IsValid() or not prim.IsA(UsdGeom.Xformable):
                return False
            xform = UsdGeom.Xformable(prim)
            xform.ClearXformOpOrder()
            op = xform.AddTransformOp(
                UsdGeom.XformOp.PrecisionDouble, "staticWorld"
            )
            matrix = pose_matrix(position, orientation)
            if initial_pose is not None:
                initial_position, initial_orientation = initial_pose
                initial_physx = pose_matrix(initial_position, initial_orientation)
                initial_visual = initial_visual_world.get(path)
                if initial_visual is not None:
                    # Keep the fixed body-frame -> USD visual-link offset.
                    matrix = matrix * initial_physx.GetInverse() * initial_visual
            op.Set(matrix)
            xform.SetResetXformStack(True)
            return True

        baked = 0
        robot = env.scene["robot"]
        robot_data = robot.data
        body_state = robot_data.body_state_w[0]
        body_names = list(robot.body_names)
        for index, body_name in enumerate(body_names):
            initial_pose = None
            if initial_robot_state is not None and index < len(initial_robot_state):
                initial_pose = (
                    initial_robot_state[index, 0:3],
                    initial_robot_state[index, 3:7],
                )
            if author_world_pose(
                f"/World/TrackedBase/G1UpperBody/{body_name}",
                body_state[index, 0:3],
                body_state[index, 3:7],
                initial_pose,
            ):
                baked += 1

        def bake_view(view, paths):
            nonlocal baked
            positions, orientations = view.get_world_poses(clone=True)
            for path, position, orientation in zip(
                paths, positions, orientations
            ):
                initial_pose = initial_task_poses.get(str(path))
                if author_world_pose(path, position, orientation, initial_pose):
                    baked += 1

        def bake_asset(asset, path):
            nonlocal baked
            state = asset.data.root_state_w[0].detach().cpu().reshape(-1)
            if author_world_pose(
                path,
                state[0:3],
                state[3:7],
                initial_task_poses.get(str(path)),
            ):
                baked += 1

        if runtime_controller is not None:
            bake_view(
                runtime_controller.vehicle_body,
                ["/World/TrackedBase/Vehicle"],
            )
            bake_view(
                runtime_controller.carriage_body,
                ["/World/TrackedBase/LiftCarriage"],
            )
            bake_view(
                runtime_controller.rod_body,
                [runtime_controller.ROD_PATH],
            )
            rope_paths = sorted(
                str(prim.GetPath())
                for prim in Usd.PrimRange(
                    stage.GetPrimAtPath(
                        "/World/envs/env_0/ChaganShengzi/rope_40cm/Links"
                    )
                )
                if prim.IsA(UsdGeom.Xformable)
                and prim.GetName().startswith("Link_")
            )
            bake_view(runtime_controller.rope_bodies, rope_paths)
            bake_asset(
                runtime_controller.leiguan,
                "/World/envs/env_0/Leiguan",
            )
            bake_asset(
                runtime_controller.holder,
                "/World/envs/env_0/ChaganShengziHolder",
            )
        print(f"[static-snapshot] baked PhysX world poses: {baked}", flush=True)
    except Exception as exc:
        print(f"[static-snapshot] PhysX pose bake failed: {exc!r}", flush=True)
        raise
    stage.Export(str(output_path))
    highres_image = _capture_requested_highres_image(stage)
    status = {
        "ok": True,
        "token": token,
        "output": str(output_path),
        "frame": request.get("frame"),
        "root_layer": str(stage.GetRootLayer().identifier),
        "time": time.time(),
    }
    if highres_image is not None:
        status["highres_image"] = highres_image
    if status_path is not None:
        status_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = status_path.with_name(
            f".{status_path.name}.{os.getpid()}.tmp"
        )
        temporary.write_text(json.dumps(status, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary, status_path)
    print(
        f"[static-snapshot] exported native stage to {output_path} "
        f"(frame={request.get('frame')})",
        flush=True,
    )
    return token

def setup_signal_handlers(controller,dds_manager=None,image_server=None):
    """set signal handlers"""
    def signal_handler(signum, frame):
        print(f"\nreceived signal {signum}, stopping controller...")
        try:
            controller.stop()
        except Exception as e:
            print(f"Failed to stop controller: {e}")
        try:
            if dds_manager is not None:
                dds_manager.stop_all_communication()
        except Exception as e:
            print(f"Failed to stop DDS: {e}")
        try:
            if image_server is not None:
                image_server.stop()
        except Exception as e:
            print(f"Failed to stop image server: {e}")
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)



def main():
    """main function"""
    runtime_controller = None
    replay_augmentation_effects = None
    # import cProfile
    # import pstats
    # import io
    # profiler = cProfile.Profile()
    # profiler.enable()
    ready_file = os.environ.get("UNITREE_SIM_READY_FILE")
    inference_pause_request = os.environ.get("G1_INFERENCE_PAUSE_REQUEST_FILE")
    inference_pause_status = os.environ.get("G1_INFERENCE_PAUSE_STATUS_FILE")
    inference_pause_request_path = (
        Path(inference_pause_request) if inference_pause_request else None
    )
    inference_pause_status_path = (
        Path(inference_pause_status) if inference_pause_status else None
    )
    active_inference_pause_token = None
    static_snapshot_request = os.environ.get("G1_STATIC_SNAPSHOT_REQUEST_FILE")
    static_snapshot_status = os.environ.get("G1_STATIC_SNAPSHOT_STATUS_FILE")
    static_snapshot_request_path = (
        Path(static_snapshot_request) if static_snapshot_request else None
    )
    static_snapshot_status_path = (
        Path(static_snapshot_status) if static_snapshot_status else None
    )
    static_snapshot_token = None
    static_snapshot_frozen = False
    try:
        # Give this simulator its own process group so terminal signals do not
        # affect another task.  Resource owners below perform targeted cleanup;
        # never signal the entire group from an atexit handler because that
        # also terminates the current process before normal shutdown finishes.
        os.setpgrp()
        current_pgid = os.getpgrp()
        print(f"Setting process group: {current_pgid}")
    except PermissionError:
        # PTY/session launchers may already make this process the session leader.
        # In that case setpgrp() is unnecessary and Linux returns EPERM.
        print("[INFO] Process group already managed by the current terminal session.")
    except Exception as e:
        print(f"Failed to set process group: {e}")
    print("=" * 60)
    print("robot control system started")
    print(f"Task: {args_cli.task}")
    print(f"Action source: {args_cli.action_source}")
    print("=" * 60)

    # parse environment configuration
    try:
        env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
        env_cfg.env_name = args_cli.task
    except Exception as e:
        print(f"Failed to parse environment configuration: {e}")
        return
    
    # create environment
    print("\ncreate environment...")
    try:
        env_cfg.seed = args_cli.seed
        env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
        env.seed(args_cli.seed)
        try:
            sensors_dict = getattr(env.scene, "sensors", {})
            if sensors_dict:
                print("Sensors in the environment:")
                for name, sensor in sensors_dict.items():
                    print(name, sensor)
                print("="*60)
        except Exception as e:
            print(f"[sim] failed to list sensors: {e}")
        print(f"\ncreate environment success ...")
        try:
            env._reward_interval = max(1, int(args_cli.env_reward_interval))
            env._reward_counter = 0
            env._reward_last = None
            print(f"[env] reward compute interval set to {env._reward_interval} steps")
        except Exception as e:
            print(f"[env] failed to set reward interval: {e}")
        if args_cli.physics_dt is not None:
            try:
                env.sim.set_substep_time(args_cli.physics_dt)
                print(f"[sim] physics dt set to {args_cli.physics_dt}")
            except Exception:
                try:
                    env.sim.dt = args_cli.physics_dt
                    print(f"[sim] physics dt assigned to env.sim.dt={args_cli.physics_dt}")
                except Exception as e:
                    print(f"[sim] failed to set physics dt: {e}")
        headless_mode = bool(getattr(args_cli, "headless", False))
        render_interval = None
        if args_cli.render_interval is not None:
            try:
                render_interval = max(1, int(args_cli.render_interval))
            except Exception as e:
                print(f"[sim] invalid render_interval value {args_cli.render_interval}: {e}")
        try:
            if args_cli.no_render:
                env.sim.render_interval = 1_000_000
                env.sim.render_mode = "offscreen"
                print("[sim] rendering disabled via --no_render")
            elif headless_mode:
                env.sim.render_mode = "offscreen"
                env.sim.render_interval = render_interval or 1
                print(f"[sim] headless offscreen rendering every {env.sim.render_interval} steps")
            elif render_interval is not None:
                env.sim.render_interval = render_interval
                print(f"[sim] render_interval set to {env.sim.render_interval}")
        except Exception as e:
            print(f"[sim] failed to configure rendering: {e}")
        if args_cli.camera_write_interval is not None:
            try:
                import tasks.common_observations.camera_state as cam_state
                cam_state._camera_cache['write_interval_steps'] = max(1, int(args_cli.camera_write_interval))
                print(f"[camera] write interval steps set to {cam_state._camera_cache['write_interval_steps']}")
            except Exception as e:
                print(f"[camera] failed to set write interval: {e}")

        try:
            if args_cli.solver_iterations is not None:
                env.sim.physx.solver_iteration_count = int(args_cli.solver_iterations)
                print(f"[sim] solver_iteration_count={env.sim.physx.solver_iteration_count}")
            if args_cli.physx_substeps is not None:
                try:
                    env.sim.physx.substeps = int(args_cli.physx_substeps)
                except Exception:
                    try:
                        env.sim.set_substeps(int(args_cli.physx_substeps))
                    except Exception:
                        pass
                print(f"[sim] physx_substeps set to {args_cli.physx_substeps}")
            if args_cli.gravity_z is not None:
                g = float(args_cli.gravity_z)
                env.sim.physx.gravity = (0.0, 0.0, g)
                print(f"[sim] gravity set to {env.sim.physx.gravity}")
        except Exception as e:
            print(f"[sim] failed to set physx params: {e}")
        if args_cli.skip_cvtcolor:
            os.environ["CAMERA_SKIP_CVTCOLOR"] = "1"
        try:
            import tasks.common_observations.camera_state as cam_state
            enable_jpeg = bool(args_cli.camera_jpeg) or (os.getenv("CAMERA_JPEG") == "1")
            jpeg_quality = int(args_cli.camera_jpeg_quality if args_cli.camera_jpeg else os.getenv("CAMERA_JPEG_QUALITY", args_cli.camera_jpeg_quality))
            cam_state.set_writer_options(enable_jpeg=enable_jpeg, jpeg_quality=jpeg_quality, skip_cvtcolor=args_cli.skip_cvtcolor)
            include = [n.strip() for n in (args_cli.camera_include or "").split(',') if n.strip()]
            exclude = [n.strip() for n in (args_cli.camera_exclude or "").split(',') if n.strip()]
            try:
                cam_state.set_camera_allowlist(include)
            except Exception:
                pass
            try:
                sensors_dict = getattr(env.scene, "sensors", {})
                for name, sensor in sensors_dict.items():
                    lname = name.lower()
                    if "camera" not in lname:
                        continue
                    if exclude and name in exclude:
                        for attr_name, value in [("enabled", False), ("is_enabled", False)]:
                            if hasattr(sensor, attr_name):
                                try:
                                    setattr(sensor, attr_name, value)
                                except Exception:
                                    pass
                        for meth in ("set_active", "disable", "pause"):
                            if hasattr(sensor, meth):
                                try:
                                    getattr(sensor, meth)(False)
                                except Exception:
                                    pass
                        for attr_name in ("update_period", "_update_period"):
                            if hasattr(sensor, attr_name):
                                try:
                                    setattr(sensor, attr_name, 1e6)
                                except Exception:
                                    pass
                    elif include and name not in include:
                        for attr_name in ("update_period", "_update_period"):
                            if hasattr(sensor, attr_name):
                                try:
                                    setattr(sensor, attr_name, 1e6)
                                except Exception:
                                    pass
            except Exception as e:
                print(f"[camera] failed to tune sensors: {e}")
        except Exception as e:
            print(f"[camera] failed to apply writer options: {e}")
    except Exception as e:
        print(f"\nFailed to create environment: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # get robot stiffness and damping parameters from runtime environment
    print("\n" + "="*60)
    print("🔍 Getting robot stiffness and damping parameters from runtime environment")
    print("="*60)
    
    try:
        stiffness_data = get_robot_stiffness_from_env(env)
        if stiffness_data:
            print("✅ Successfully got robot parameters!")
        else:
            print("⚠️ Failed to get robot parameters, will try again after environment reset")
    except Exception as e:
        print(f"⚠️ Error getting robot parameters: {e}")
    
    print("="*60)
    
    if not getattr(args_cli, "headless", False) and not args_cli.no_render:
        print("\n")
        print("***  Please left-click on the Sim window to activate rendering. ***")
        print("\n")
    else:
        print("\n")
        print("***  Running without GUI; rendering handled offscreen. ***")
        print("\n")
    # reset environment
    if args_cli.modify_light:
        update_light(
            prim_path="/World/light",
            color=(0.75, 0.75, 0.75),
            intensity=500.0,
            # position=(1.0, 2.0, 3.0),
            radius=0.1,
            enabled=True,
            cast_shadows=True
        )
    if args_cli.modify_camera:
        batch_augment_cameras_by_name(
            names=["front_cam"],
            focal_length=3.0,
            horizontal_aperture=22.0,
            vertical_aperture=16.0,
            exposure=0.8,                
            focus_distance=1.2
        )

    # Author replay lights before the first simulation reset.  Camera render
    # products and RTX state are initialized by that reset; creating the
    # lights afterwards caused the previous five lighting cases to render as
    # effectively the same dark image.
    try:
        effects_module_file = os.environ.get("ISAAC_REPLAY_EFFECTS_MODULE_FILE")
        if effects_module_file:
            import importlib.util
            effects_spec = importlib.util.spec_from_file_location(
                "external_replay_augmentation_effects", effects_module_file
            )
            if effects_spec is None or effects_spec.loader is None:
                raise ImportError(
                    f"Unable to load replay effects module: {effects_module_file}"
                )
            effects_module = importlib.util.module_from_spec(effects_spec)
            effects_spec.loader.exec_module(effects_module)
            ReplayAugmentationEffects = effects_module.ReplayAugmentationEffects
            effects_requested = effects_module.effects_requested
        else:
            from tools.replay_augmentation_effects import (
                ReplayAugmentationEffects,
                effects_requested,
            )
        if effects_requested():
            replay_augmentation_effects = ReplayAugmentationEffects()
    except Exception as e:
        print(f"Failed to initialize replay augmentation effects: {e}")
        env.close()
        return

    env.sim.reset()
    env.reset()

    # Tasks may provide a small scene-specific runtime controller (for
    # example, a mobile base carrying the standard Unitree articulation).
    # The robot action path remains the normal DDS action provider.
    runtime_controller_factory = getattr(
        env_cfg, "create_runtime_controller", None
    )
    if callable(runtime_controller_factory):
        try:
            runtime_controller = runtime_controller_factory(env)
            print("[sim] task runtime controller initialized")
        except Exception as e:
            print(f"Failed to create task runtime controller: {e}")
            env.close()
            return

    # Replay-ready datasets carry the exact scene snapshot captured at the
    # beginning of each source episode.  Restore it after task runtime
    # controllers have finished their constructor resets, then synchronize
    # pose-driven controller state so it cannot overwrite the snapshot.
    try:
        replay_initial_state, replay_initial_payload = _load_replay_initial_state(
            os.environ.get("G1_REPLAY_INITIAL_STATE_FILE"), env.device
        )
        if replay_initial_state is not None:
            metadata_task = str(
                (replay_initial_payload.get("sim_state") or {}).get("task_name", "")
            )
            if metadata_task and metadata_task != args_cli.task:
                raise ValueError(
                    f"Replay metadata task mismatch: {metadata_task!r} != {args_cli.task!r}"
                )
            _restore_replay_scene(env.scene, replay_initial_state)
            _synchronize_runtime_controller(
                runtime_controller, replay_initial_payload
            )
            env.sim.forward()
            print(
                "[replay-initial-state] restored "
                f"{os.environ.get('G1_REPLAY_INITIAL_STATE_FILE')}",
                flush=True,
            )
    except Exception as e:
        print(f"Failed to restore replay initial state: {e}")
        env.close()
        return
    
    # create simplified control configuration
    try:    
        control_config = ControlConfig(
            step_hz=args_cli.step_hz,
            replay_mode=args_cli.replay_data
        )
    except Exception as e:
        print(f"Failed to create control configuration: {e}")
        return
    
    # create controller

    if not args_cli.replay_data:
        print("========= create image server =========")
        try:
            image_server = run_isaacsim_server()
        except Exception as e:
            print(f"Failed to create image server: {e}")
            return
        print("========= create image server success =========")
        print("========= create dds =========")
        try:
            reset_pose_dds,sim_state_dds,dds_manager = create_dds_objects(args_cli,env)
        except Exception as e:
            print(f"Failed to create dds: {e}")
            return
        print("========= create dds success =========")
    else:
        print("========= create dds =========")
        try:
            create_dds_objects_replay(args_cli,env)
        except Exception as e:
            print(f"Failed to create dds: {e}")
            return
        print("========= create dds success =========")
        from tools.data_json_load import get_data_json_list
        print("========= get data json list =========")
        data_idx=0
        data_json_list = get_data_json_list(args_cli.file_path)
        if args_cli.action_source != "replay":
            args_cli.action_source = "replay"
        print("========= get data json list success =========")
    # create action provider
    
    print(f"\ncreate action provider: {args_cli.action_source}...")
    try:
        print(f"args_cli.task: {args_cli.task}")
        if not args_cli.replay_data and ("Wholebody" in args_cli.task or args_cli.enable_wholebody_dds):
            args_cli.action_source = "dds_wholebody"
            args_cli.enable_wholebody_dds = True
            control_config.use_rl_action_mode = True
        action_provider = create_action_provider(env,args_cli)
        if action_provider is None:
            print("action provider creation failed, exiting")
            return
    except Exception as e:
        print(f"Failed to create action provider: {e}")
        return
    
    # set action provider
    print("========= create controller =========")
    controller = RobotController(env, control_config)
    controller.set_action_provider(action_provider)
    runtime_action_override = getattr(runtime_controller, "override_action", None)
    if callable(runtime_action_override):
        controller.set_action_override(runtime_action_override)
        print("========= runtime action override installed =========")
    print("========= create controller success =========")

    # Capture replay-start calibration only for the explicit static USD
    # export path. Normal Task01 runs do not allocate or mutate this state.
    if static_snapshot_request_path is not None:
        _capture_static_snapshot_initial_state(env, runtime_controller)
    
    # configure performance analysis
    if args_cli.enable_profiling:
        controller.set_profiling(True, args_cli.profile_interval)
        print(f"performance analysis enabled, report every {args_cli.profile_interval} steps")
    else:
        controller.set_profiling(False)
        print("performance analysis disabled")


    # set signal handlers
    if not args_cli.replay_data:
        setup_signal_handlers(controller,dds_manager,image_server)
    else:
        setup_signal_handlers(controller)
        
    print("Note: The DDS in Sim transmits messages on channel 1. Please ensure that other DDS instances use the same channel for message exchange by setting: ChannelFactoryInitialize(1).")
    try:
        # start controller - start asynchronous components
        print("========= start controller =========")
        controller.start()
        print("========= start controller success =========")
        if ready_file:
            ready_path = Path(ready_file)
            ready_path.parent.mkdir(parents=True, exist_ok=True)
            ready_path.write_text(f"{os.getpid()}\n", encoding="utf-8")
            print(f"[sim] ready marker created: {ready_path}")
        
        # main loop - execute in main thread to support rendering
        last_stats_time = time.time()
        loop_start_time = time.time()
        loop_count = 0
        last_loop_time = time.time()
        recent_loop_times = []  # for calculating moving average frequency
        
        
        reward_interval = max(1, args_cli.reward_interval)

        # use torch.inference_mode() and exception suppression
        with contextlib.suppress(KeyboardInterrupt), torch.inference_mode():
            while simulation_app.is_running() and controller.is_running:
                current_time = time.time()
                if not static_snapshot_frozen and static_snapshot_request_path is not None:
                    try:
                        requested_token = _export_requested_static_snapshot(
                            static_snapshot_request_path,
                            static_snapshot_status_path,
                            env,
                            runtime_controller,
                        )
                    except Exception as exc:
                        requested_token = None
                        print(f"[static-snapshot] export failed: {exc}", flush=True)
                        if static_snapshot_status_path is not None:
                            static_snapshot_status_path.parent.mkdir(parents=True, exist_ok=True)
                            static_snapshot_status_path.write_text(
                                json.dumps(
                                    {"ok": False, "error": str(exc)},
                                    sort_keys=True,
                                )
                                + "\n",
                                encoding="utf-8",
                            )
                    if requested_token:
                        static_snapshot_token = requested_token
                        static_snapshot_frozen = True
                        print(
                            f"[static-snapshot] physics/controller frozen after token {static_snapshot_token}",
                            flush=True,
                        )
                if static_snapshot_frozen:
                    # Keep Kit/render products alive for a caller that wants
                    # to inspect the captured frame, but never advance PhysX,
                    # the runtime controller, or DDS actions after export.
                    simulation_app.update()
                    time.sleep(0.01)
                    continue
                # Remote LingBot inference is substantially slower than one
                # action chunk. A file-based handshake lets its evaluator
                # freeze physics while a fresh chunk is generated. No other
                # policy creates this marker, so DP/PI0.5 are unchanged.
                if (
                    inference_pause_request_path is not None
                    and inference_pause_status_path is not None
                    and inference_pause_request_path.is_file()
                ):
                    try:
                        pause_token = inference_pause_request_path.read_text(
                            encoding="utf-8"
                        ).strip()
                    except OSError:
                        pause_token = ""
                    if pause_token:
                        if pause_token != active_inference_pause_token:
                            inference_pause_status_path.parent.mkdir(
                                parents=True, exist_ok=True
                            )
                            temporary = inference_pause_status_path.with_name(
                                f".{inference_pause_status_path.name}.{os.getpid()}.tmp"
                            )
                            temporary.write_text(pause_token + "\n", encoding="utf-8")
                            os.replace(temporary, inference_pause_status_path)
                            active_inference_pause_token = pause_token
                            print(
                                f"[sim] inference physics pause acknowledged: {pause_token}",
                                flush=True,
                            )
                        # Keep Kit responsive but do not run any task update or
                        # controller.step(), so physics and sensors stay frozen.
                        simulation_app.update()
                        last_loop_time = time.time()
                        time.sleep(0.005)
                        continue
                if active_inference_pause_token is not None:
                    inference_pause_status_path.unlink(missing_ok=True)
                    print(
                        f"[sim] inference physics pause released: "
                        f"{active_inference_pause_token}",
                        flush=True,
                    )
                    active_inference_pause_token = None
                    # Do not feed the inference wait into task wall-delta hooks.
                    last_loop_time = time.time()
                    current_time = last_loop_time
                loop_count += 1
                if not args_cli.replay_data:
                    try:
                        env_state = env.scene.get_state()
                        env_state_json =  sim_state_to_json(env_state)
                        sim_state = {"init_state":env_state_json,"task_name":args_cli.task}
                        recording_state_getter = getattr(
                            runtime_controller, "get_recording_state", None
                        )
                        if callable(recording_state_getter):
                            # Optional task-specific telemetry. Existing tasks do
                            # not implement this hook and retain their exact DDS
                            # payload and LeRobot schema.
                            sim_state["runtime_state"] = recording_state_getter()
                    except Exception as e:
                        print(f"Failed to get env state: {e}")
                        raise e
                    try:
                    # sim_state = json.dumps(sim_state)
                        sim_state_dds.write_sim_state_data(sim_state)
                    except Exception as e:
                        print(f"Failed to write sim state: {e}")
                        raise e
                    try:
                        reset_pose_cmd = reset_pose_dds.get_reset_pose_command()
                    except Exception as e:
                        print(f"Failed to get reset pose command: {e}")
                        raise e
                    # Compute current reward values manually if needed for debugging
                    try:
                        if (loop_count % reward_interval) == 0:
                            pass
                            # current_reward = get_step_reward_value(env)
                    except Exception as e:
                        print(f"奖励计算失败: {e}")
                        pass
                    
                    if reset_pose_cmd is not None:
                        try:
                            reset_category = reset_pose_cmd.get("reset_category")
                            if (args_cli.enable_wholebody_dds and (reset_category == '1' or reset_category == '2')) or (not args_cli.enable_wholebody_dds and reset_category == '1'):
                                print("reset object")
                                env_cfg.event_manager.trigger("reset_object_self", env)
                                reset_pose_dds.write_reset_pose_command(-1)
                            elif reset_category == '2' and not args_cli.enable_wholebody_dds:
                                print("reset all")
                                env_cfg.event_manager.trigger("reset_all_self", env)
                                if runtime_controller is not None:
                                    runtime_controller.reset()
                                reset_pose_dds.write_reset_pose_command(-1)
                        except Exception as e:
                            print(f"Failed to write reset pose command: {e}")
                            raise e
                else:
                    if action_provider.get_start_loop() and data_idx<len(data_json_list):
                        print(f"data_idx: {data_idx}")
                        try:
                            sim_state,task_name = action_provider.load_data(data_json_list[data_idx])
                            if task_name!=args_cli.task:
                                raise ValueError(f" The {task_name} in the dataset is different from the {args_cli.task} being executed .")
                        except Exception as e:
                            print(f"Failed to load data: {e}")
                            raise e
                        try:
                            env.reset_to(sim_state, torch.tensor([0], device=env.device), is_relative=True)
                            env.sim.reset()
                            time.sleep(1)
                            action_provider.start_replay()
                            data_idx+=1
                        except Exception as e:
                            print(f"Failed to start replay: {e}")
                            raise e
                # print(f"env_state: {env_state}")
                # calculate instantaneous loop time
                loop_dt = current_time - last_loop_time
                last_loop_time = current_time
                recent_loop_times.append(loop_dt)
                
                # keep recent 100 loop times
                if len(recent_loop_times) > 100:
                    recent_loop_times.pop(0)
                
                # execute control step (in main thread, support rendering)
                if replay_augmentation_effects is not None:
                    replay_augmentation_effects.update(loop_dt)
                if runtime_controller is not None:
                    runtime_controller.update(loop_dt)
                if (
                    os.environ.get("ISAAC_REPLAY_DISABLE_SUCCESS_FREEZE", "0") != "1"
                    and bool(
                        getattr(
                            runtime_controller, "is_terminal_frozen",
                            getattr(runtime_controller, "is_success_frozen", False),
                        )
                    )
                ):
                    # Keep Kit and DDS responsive so the user can still accept
                    # or discard/reset the episode, but do not advance physics,
                    # robot actions, or camera sensor frames past SUCCESS.
                    simulation_app.update()
                    time.sleep(0.01)
                    continue
                controller.step()

                # print statistics and loop frequency periodically
                if current_time - last_stats_time >= args_cli.stats_interval:
                    # calculate while loop execution frequency
                    elapsed_time = current_time - loop_start_time
                    loop_frequency = loop_count / elapsed_time if elapsed_time > 0 else 0
                    
                    # calculate moving average frequency (based on recent loop times)
                    if recent_loop_times:
                        avg_loop_time = sum(recent_loop_times) / len(recent_loop_times)
                        moving_avg_frequency = 1.0 / avg_loop_time if avg_loop_time > 0 else 0
                        min_loop_time = min(recent_loop_times)
                        max_loop_time = max(recent_loop_times)
                        max_freq = 1.0 / min_loop_time if min_loop_time > 0 else 0
                        min_freq = 1.0 / max_loop_time if max_loop_time > 0 else 0
                    else:
                        moving_avg_frequency = 0
                        min_freq = max_freq = 0
                    
                    print(f"\n=== While loop execution frequency statistics ===")
                    print(f"loop execution count: {loop_count}")
                    print(f"running time: {elapsed_time:.2f} seconds")
                    print(f"overall average frequency: {loop_frequency:.2f} Hz")
                    print(f"moving average frequency: {moving_avg_frequency:.2f} Hz (last {len(recent_loop_times)} times)")
                    print(f"frequency range: {min_freq:.2f} - {max_freq:.2f} Hz")
                    print(f"average loop time: {(elapsed_time/loop_count*1000):.2f} ms")
                    if recent_loop_times:
                        print(f"recent loop time: {(avg_loop_time*1000):.2f} ms")
                    print(f"=============================")
                    
                    # print_stats(controller)
                    last_stats_time = current_time
       
                # check environment state
                if env.sim.is_stopped():
                    print("\nenvironment stopped")
                    break
                # rate_limiter.sleep(env)
    except KeyboardInterrupt:
        print("\nuser interrupted program")
    
    except Exception as e:
        print(f"\nprogram exception: {e}")
    
    finally:
        # clean up resources
        print("\nclean up resources...")
        if ready_file:
            Path(ready_file).unlink(missing_ok=True)
        if inference_pause_status_path is not None:
            inference_pause_status_path.unlink(missing_ok=True)
        if runtime_controller is not None:
            try:
                runtime_controller.close()
            except Exception as e:
                print(f"Failed to close task runtime controller: {e}")
        if replay_augmentation_effects is not None:
            try:
                replay_augmentation_effects.close()
            except Exception as e:
                print(f"Failed to close replay augmentation effects: {e}")
        controller.cleanup()
        image_server.stop()
        env.close()
        print("cleanup completed")
    # profiler.disable()
    # s = io.StringIO()
    # ps = pstats.Stats(profiler, stream=s).strip_dirs().sort_stats("time")
    # ps.print_stats(30)  

    # print(s.getvalue())

if __name__ == "__main__":
    try:
        main()
    finally:
        print("Performing final cleanup...")
        # Never select processes by the generic string "sim_main.py": another
        # task may legitimately be running.  main() already stops this
        # instance's controller, DDS manager, image server, and environment.
        try:
            simulation_app.close()
        except Exception as e:
            print(f"Failed to close simulation application: {e}")
        print("Program exit completed")

# python sim_main.py --device cpu  --enable_cameras  --task  Isaac-PickPlace-Cylinder-G129-Dex1-Joint   --enable_dex1_dds --robot_type g129
# python sim_main.py --device cpu  --enable_cameras  --task Isaac-PickPlace-Cylinder-G129-Dex3-Joint    --enable_dex3_dds --robot_type g129
# python sim_main.py --device cpu  --enable_cameras  --task Isaac-PickPlace-Cylinder-G129-Inspire-Joint    --enable_inspire_dds --robot_type g129

# python sim_main.py --device cpu  --enable_cameras  --task Isaac-PickPlace-RedBlock-G129-Dex1-Joint     --enable_dex1_dds --robot_type g129
# python sim_main.py --device cpu  --enable_cameras  --task Isaac-PickPlace-RedBlock-G129-Dex3-Joint    --enable_dex3_dds --robot_type g129
# python sim_main.py --device cpu  --enable_cameras  --task  Isaac-PickPlace-RedBlock-G129-Inspire-Joint    --enable_inspire_dds --robot_type g129


# python sim_main.py --device cpu  --enable_cameras  --task Isaac-Stack-RgyBlock-G129-Dex1-Joint     --enable_dex1_dds --robot_type g129
# python sim_main.py --device cpu  --enable_cameras  --task Isaac-Stack-RgyBlock-G129-Dex3-Joint     --enable_dex3_dds --robot_type g129
# python sim_main.py --device cpu  --enable_cameras  --task Isaac-Stack-RgyBlock-G129-Inspire-Joint     --enable_inspire_dds --robot_type g129




# python sim_main.py --device cpu  --enable_cameras  --task Isaac-Move-Cylinder-G129-Dex1-Wholebody  --robot_type g129 --enable_dex1_dds 
# python sim_main.py --device cpu  --enable_cameras  --task Isaac-Move-Cylinder-G129-Dex3-Wholebody  --robot_type g129 --enable_dex3_dds 
# python sim_main.py --device cpu  --enable_cameras  --task Isaac-Move-Cylinder-G129-Inspire-Wholebody  --robot_type g129 --enable_inspire_dds 


# python sim_main.py --device cpu  --enable_cameras  --task Isaac-PickPlace-Cylinder-H12-27dof-Inspire-Joint  --enable_inspire_dds --robot_type h1_2
# python sim_main.py --device cpu  --enable_cameras  --task Isaac-PickPlace-RedBlock-H12-27dof-Inspire-Joint  --enable_inspire_dds --robot_type h1_2
# python sim_main.py --device cpu  --enable_cameras  --task Isaac-Stack-RgyBlock-H12-27dof-Inspire-Joint --enable_inspire_dds --robot_type h1_2
