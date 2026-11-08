"""Tracked-base controller plus randomized multi-body object reset."""

from __future__ import annotations

import json
import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from pxr import Gf, Usd, UsdGeom

import carb.input
import omni.appwindow
import omni.usd
import isaaclab.envs.mdp as base_mdp
from isaacsim.core.prims import RigidPrim

from .tabletop_grasp_g1_29dof_inspire_env_cfg import (
    ASSETS_ROOT,
    G1_MOUNT_LOCAL_ORIENTATION,
    TRACKED_BASE_X,
    TRACKED_BASE_Y,
    TRACKED_BASE_YAW,
    TRACKED_LIFT_HEIGHT,
    TABLE_YAW,
    TABLE_POSITION,
    W1_UPPER_MOUNT_LOCAL,
)


W1_TOOLS = ASSETS_ROOT / "robot/Forklift_W1_Final/tools"
INSERTED_POSE_PATH = (
    ASSETS_ROOT.parent / "task" / "g1_tabletop_grasp_task" / "saved_inserted_object_pose.json"
)
SUCCESS_MARKER_PATH = Path(
    os.environ.get(
        "G1_TASK_SUCCESS_FILE",
        f"/tmp/g1_tabletop_grasp_{os.getuid()}.success",
    )
)
BENCHMARK_RESET_REQUEST_PATH = Path(
    os.environ.get(
        "G1_BENCHMARK_RESET_REQUEST_FILE",
        f"/tmp/g1_tabletop_grasp_{os.getuid()}.eval_reset.json",
    )
)
BENCHMARK_RESET_STATUS_PATH = Path(
    os.environ.get(
        "G1_BENCHMARK_RESET_STATUS_FILE",
        f"/tmp/g1_tabletop_grasp_{os.getuid()}.eval_status.json",
    )
)
BENCHMARK_ACTIVE_PATH = Path(
    os.environ.get(
        "G1_BENCHMARK_ACTIVE_FILE",
        f"/tmp/g1_tabletop_grasp_{os.getuid()}.eval_active",
    )
)

# Dimensions measured from the actual composed USD assets.  Leiguan's mesh is
# authored with a 35 mm local inner radius and a 0.45 XY scale.
LEIGUAN_INNER_RADIUS = 0.01575
LEIGUAN_HALF_LENGTH = 0.175
ROD_MAX_RADIUS = 0.0105
ROD_LENGTH = 0.252
MIN_INSERTION_DEPTH = 0.12
RADIAL_TOLERANCE = 0.001
# The robot approaches the worktable from local -Y.  Allowing Leiguan to move
# farther than 2 cm in that direction overlaps the reset pose of the right
# gripper for part of the original +/-5 cm world-XY sampling square.  Keep the
# lateral and away-from-robot variation unchanged, but reject those physically
# invalid initial states before an episode starts.
LEIGUAN_MIN_TABLE_LOCAL_DEPTH_OFFSET = -0.02
MAX_OBJECT_SAMPLE_ATTEMPTS = 100
OBJECT_GRIPPER_CLEARANCE = float(
    os.environ.get("G1_OBJECT_GRIPPER_CLEARANCE_M", "0.06")
)
if not 0.0 < OBJECT_GRIPPER_CLEARANCE <= 0.20:
    raise ValueError(
        "G1_OBJECT_GRIPPER_CLEARANCE_M must be in the interval (0, 0.20] m"
    )
if str(W1_TOOLS) not in sys.path:
    sys.path.insert(0, str(W1_TOOLS))
from differential_drive import DifferentialTrackKinematics  # noqa: E402


class TabletopGraspController:
    """Preserve W1 controls and randomize both tabletop task objects."""

    ROOT_PATH = "/World/TrackedBase"
    VEHICLE_PATH = f"{ROOT_PATH}/Vehicle"
    CARRIAGE_PATH = f"{ROOT_PATH}/LiftCarriage"
    ROD_PATH = "/World/envs/env_0/ChaganShengzi/SmoothConnector"
    ROPE_LINKS_PATH = (
        "/World/envs/env_0/ChaganShengzi/rope_40cm/Links/Link_.*"
    )

    def __init__(self, env) -> None:
        self.env = env
        self.robot = env.scene["robot"]
        self.device = env.device
        self.stage = omni.usd.get_context().get_stage()
        self.x = float(TRACKED_BASE_X)
        self.y = float(TRACKED_BASE_Y)
        self.yaw = float(TRACKED_BASE_YAW)
        self.lift_height = float(TRACKED_LIFT_HEIGHT)
        self.initial = (self.x, self.y, self.yaw, self.lift_height)
        self.object_random_xy_limit = 0.05
        object_reset_seed = os.environ.get("G1_OBJECT_RESET_SEED")
        self._object_rng = np.random.default_rng(
            None if object_reset_seed is None else int(object_reset_seed)
        )
        self._reset_settle_steps = 10
        self._reset_settle_steps_remaining = 0
        self._benchmark_reset_status: dict | None = None
        self.success_latched = False
        self.success_insertion_depth = 0.0
        self.success_max_radial_error = float("inf")
        self._success_check_disabled = (
            os.environ.get("G1_DISABLE_SUCCESS_CHECK", "0") == "1"
        )
        self.held: set[carb.input.KeyboardInput] = set()
        self._input = None
        self._keyboard = None
        self._subscription = None
        self._mine_dust = None
        self._light_flicker = None
        self._headlamp = None
        self._auxiliary_lights = None
        self._falling_debris = None

        root = self.stage.GetPrimAtPath(self.ROOT_PATH)
        vehicle = self.stage.GetPrimAtPath(self.VEHICLE_PATH)
        carriage = self.stage.GetPrimAtPath(self.CARRIAGE_PATH)
        if not root or not vehicle or not carriage:
            raise RuntimeError(
                "Tracked base hierarchy is incomplete: "
                f"root={bool(root)}, vehicle={bool(vehicle)}, carriage={bool(carriage)}"
            )

        root_xform = UsdGeom.Xformable(root)
        self.root_translate_op = self._get_or_add_op(root_xform, "translate")
        self.root_orient_op = self._get_or_add_op(root_xform, "orient")

        carriage_xform = UsdGeom.Xformable(carriage)
        self.carriage_translate_op = self._get_or_add_op(
            carriage_xform, "translate"
        )

        self.track_attributes = {
            name: vehicle.GetAttribute(f"trackDrive:{name}")
            for name in (
                "leftVelocity",
                "rightVelocity",
                "actualLeftVelocity",
                "actualRightVelocity",
                "leftDistance",
                "rightDistance",
                "trackSeparation",
                "maxTrackSpeed",
                "maxTrackAcceleration",
                "headingOffsetRadians",
            )
        }
        missing = [name for name, attr in self.track_attributes.items() if not attr]
        if missing:
            raise RuntimeError(f"Tracked-base attributes are missing: {missing}")

        self.lift_attributes = {
            name: carriage.GetAttribute(f"liftDrive:{name}")
            for name in (
                "height",
                "targetHeight",
                "lowerLimit",
                "upperLimit",
                "maxSpeed",
            )
        }
        missing = [name for name, attr in self.lift_attributes.items() if not attr]
        if missing:
            raise RuntimeError(f"Tracked-lift attributes are missing: {missing}")

        self.lift_lower = float(self.lift_attributes["lowerLimit"].Get())
        self.lift_upper = float(self.lift_attributes["upperLimit"].Get())
        self.lift_speed = float(self.lift_attributes["maxSpeed"].Get())
        self.track_drive = DifferentialTrackKinematics(
            track_separation=float(self.track_attributes["trackSeparation"].Get()),
            max_track_speed=float(self.track_attributes["maxTrackSpeed"].Get()),
            max_track_acceleration=float(
                self.track_attributes["maxTrackAcceleration"].Get()
            ),
            heading_offset=float(
                self.track_attributes["headingOffsetRadians"].Get()
            ),
        )

        # These actors are authored kinematic in the original W1 asset. Direct
        # tensor poses preserve their collision behavior while the USD root
        # transform keeps the non-rigid lift visuals aligned.
        self.vehicle_body = RigidPrim(
            prim_paths_expr=self.VEHICLE_PATH,
            name="tracked_g1_vehicle",
            reset_xform_properties=False,
        )
        self.carriage_body = RigidPrim(
            prim_paths_expr=self.CARRIAGE_PATH,
            name="tracked_g1_lift_carriage",
            reset_xform_properties=False,
        )
        self.vehicle_body.initialize()
        self.carriage_body.initialize()
        if not self.vehicle_body.is_physics_handle_valid():
            raise RuntimeError("Tracked vehicle physics handle did not initialize")
        if not self.carriage_body.is_physics_handle_valid():
            raise RuntimeError("Tracked lift physics handle did not initialize")

        self.rod_body = RigidPrim(
            prim_paths_expr=self.ROD_PATH,
            name="tabletop_chagan_body",
            reset_xform_properties=False,
        )
        self.rope_bodies = RigidPrim(
            prim_paths_expr=self.ROPE_LINKS_PATH,
            name="tabletop_rope_bodies",
            reset_xform_properties=False,
        )
        self._task_object_views = (self.rod_body, self.rope_bodies)
        for label, view in zip(("rod", "rope links"), self._task_object_views):
            view.initialize()
            if not view.is_physics_handle_valid():
                raise RuntimeError(
                    f"Task object physics handle did not initialize: {label}"
                )
        if self.rod_body.count != 1 or self.rope_bodies.count != 26:
            raise RuntimeError(
                "Unexpected rod-and-rope body count: "
                f"rod={self.rod_body.count}, rope={self.rope_bodies.count}"
            )

        self.leiguan = self.env.scene["leiguan"]
        self.holder = self.env.scene["chaganshengzi_holder"]

        self._task_object_initial_poses = []
        for view in self._task_object_views:
            positions, orientations = view.get_world_poses(clone=True)
            self._task_object_initial_poses.append((positions, orientations))

        self._gripper_body_indices, self._gripper_body_names = (
            self.robot.find_bodies(
                ["left_hand_.*", "right_hand_.*", "L_.*", "R_.*"],
                preserve_order=True,
            )
        )
        if not self._gripper_body_indices:
            raise RuntimeError("No G1 hand bodies were found for reset clearance")

        self._subscribe_keyboard()
        dust_mode = os.environ.get("G1_DUST_MODE", "").strip().lower()
        if not dust_mode:
            dust_mode = (
                "legacy"
                if os.environ.get("G1_ENABLE_3D_DUST", "0") == "1"
                else "off"
            )
        if dust_mode not in {
            "off", "legacy", "flow", "volume", "suspended", "blender_vdb"
        }:
            raise ValueError(
                "G1_DUST_MODE must be one of: off, legacy, flow, volume, "
                "suspended, blender_vdb; "
                f"got {dust_mode!r}"
            )
        if dust_mode == "legacy":
            from .mine_dust_3d import MineDust3D

            self._mine_dust = MineDust3D(TABLE_POSITION)
        elif dust_mode == "flow":
            from .flow_dust_3d import FlowDust3D

            self._mine_dust = FlowDust3D(TABLE_POSITION, TABLE_YAW)
        elif dust_mode == "volume":
            from .volume_dust_3d import VolumeDust3D

            self._mine_dust = VolumeDust3D(TABLE_POSITION, TABLE_YAW)
        elif dust_mode == "suspended":
            from .suspended_dust_3d import SuspendedDust3D

            self._mine_dust = SuspendedDust3D(TABLE_POSITION, TABLE_YAW)
        elif dust_mode == "blender_vdb":
            from .blender_vdb_dust_3d import BlenderVdbDust3D

            self._mine_dust = BlenderVdbDust3D(TABLE_POSITION, TABLE_YAW)
        from .light_flicker import LightFlicker

        self._light_flicker = LightFlicker()
        from .headlamp import AuxiliarySpotlightController, HeadlampController

        self._headlamp = HeadlampController()
        self._auxiliary_lights = AuxiliarySpotlightController()
        debris_enabled = os.environ.get("G1_FALLING_DEBRIS", "0").strip().lower() in {
            "1", "true", "yes", "on"
        }
        if debris_enabled:
            from .falling_debris_3d import FallingDebris3D

            self._falling_debris = FallingDebris3D(self.stage, seed=260822)
        print(f"[tabletop-grasp] Dust mode: {dust_mode}")
        print(f"[tabletop-grasp] Falling debris: {'on' if debris_enabled else 'off'}")
        if self._success_check_disabled:
            print("[tabletop-grasp] Success checker disabled for trajectory replay.")
        self.apply()
        # Always protect the normal reset/home pose.  At later resets we also
        # include the current gripper pose, preventing an object teleport from
        # intersecting a hand that has not reached home yet.
        self._home_gripper_guard_points = self._read_gripper_guard_points()
        # The first demonstration should receive the same randomization as all
        # later reset cycles, instead of always starting at the authored center.
        self.reset_task_objects()
        print(
            "[tracked-g1] W1 tracked base + lift initialized; "
            "G1 pelvis synchronized to the original W1 upper-body mount."
        )
        print(
            "[tracked-g1] Base controls: W/S forward/reverse, A/D turn, "
            "T/Y lift, R reset robot and both objects, "
            "P save the current inserted-object relative pose. "
            "G1 Quest/Inspire controls are unchanged."
        )
        print(
            "[tabletop-grasp] Captured 27 rod-and-rope rigid-body poses for "
            "object reset; rod/rope randomizes within +/-5 cm in XY and "
            "both objects use a two-hand clearance guard "
            f"({100.0 * OBJECT_GRIPPER_CLEARANCE:.1f} cm)."
        )
        self._export_editable_stage_if_requested()

    def _export_editable_stage_if_requested(self) -> None:
        """Export the fully composed Task01 authoring layer on explicit request.

        The normal simulator path is unchanged because the environment variable
        is unset by default.  The exported layer keeps references to the source
        Task01 and rock assets on this development machine, so it remains small
        and opens as an ordinary editable USD stage in Isaac Sim.
        """

        raw_path = os.environ.get("G1_EXPORT_EDITABLE_STAGE", "").strip()
        if not raw_path:
            return

        output_path = Path(raw_path).expanduser()
        if not output_path.is_absolute():
            raise ValueError("G1_EXPORT_EDITABLE_STAGE must be an absolute path")
        if output_path.suffix.lower() not in {".usd", ".usda", ".usdc"}:
            raise ValueError(
                "G1_EXPORT_EDITABLE_STAGE must end in .usd, .usda, or .usdc"
            )

        debris_root = self.stage.GetPrimAtPath("/World/G1FallingDebris")
        if not debris_root.IsValid():
            raise RuntimeError(
                "Editable-stage export requires G1_FALLING_DEBRIS=1"
            )
        rock_count = sum(
            1
            for prim in Usd.PrimRange(debris_root)
            if prim.GetName().startswith("Rock_")
        )
        if rock_count < 1:
            raise RuntimeError("No Rock_XX prims are composed for export")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = output_path.with_name(
            f".{output_path.stem}.tmp{output_path.suffix}"
        )
        root_layer = self.stage.GetRootLayer()
        if not root_layer.Export(str(temporary_path)):
            raise RuntimeError(f"USD export failed: {temporary_path}")

        # Reopen the result before replacing any previous editor stage.  This
        # catches broken composition paths immediately while Isaac is running.
        exported_stage = Usd.Stage.Open(str(temporary_path))
        # Keep one file useful for both authoring and the final paper capture.
        # Author the variants only in the offline copy: changing variants on
        # the live physics stage invalidates Isaac tensor views.  The hidden
        # variant touches only renderable Gprims, leaving cameras and lights
        # below the tracked assembly active.
        exported_tracked_base = exported_stage.GetPrimAtPath(self.ROOT_PATH)
        visibility_variants = exported_tracked_base.GetVariantSets().AddVariantSet(
            "editRobotVisibility"
        )
        for variant_name, token in (
            ("visible", UsdGeom.Tokens.inherited),
            ("hidden", UsdGeom.Tokens.invisible),
        ):
            visibility_variants.AddVariant(variant_name)
            visibility_variants.SetVariantSelection(variant_name)
            with visibility_variants.GetVariantEditContext():
                for prim in Usd.PrimRange(exported_tracked_base):
                    if prim.IsA(UsdGeom.Gprim):
                        UsdGeom.Imageable(prim).CreateVisibilityAttr().Set(token)
        visibility_variants.SetVariantSelection("visible")
        exported_stage.GetRootLayer().Save()

        required_prims = (
            "/World/envs/env_0/Environment",
            "/World/envs/env_0/Worktable",
            "/World/TrackedBase",
            "/World/PerspectiveCamera",
            "/World/G1FallingDebris",
        )
        missing = [
            path
            for path in required_prims
            if not exported_stage.GetPrimAtPath(path).IsValid()
        ]
        exported_debris = exported_stage.GetPrimAtPath(
            "/World/G1FallingDebris"
        )
        exported_rock_count = sum(
            1
            for prim in Usd.PrimRange(exported_debris)
            if prim.GetName().startswith("Rock_")
        )
        if missing or exported_rock_count != rock_count:
            temporary_path.unlink(missing_ok=True)
            raise RuntimeError(
                "Exported editor stage failed validation: "
                f"missing={missing}, rocks={exported_rock_count}/{rock_count}"
            )

        temporary_path.replace(output_path)
        print(
            "[falling-debris] Exported complete editable Task01 stage: "
            f"{output_path} ({exported_rock_count} rocks, "
            "editRobotVisibility=visible)",
            flush=True,
        )

    @staticmethod
    def _get_or_add_op(xform: UsdGeom.Xformable, kind: str):
        op_name = f"xformOp:{kind}"
        found = next(
            (op for op in xform.GetOrderedXformOps() if op.GetOpName() == op_name),
            None,
        )
        if found is not None:
            return found
        if kind == "translate":
            return xform.AddTranslateOp()
        if kind == "orient":
            return xform.AddOrientOp()
        raise ValueError(kind)

    def _subscribe_keyboard(self) -> None:
        try:
            window = omni.appwindow.get_default_app_window()
            if window is None:
                return
            self._input = carb.input.acquire_input_interface()
            self._keyboard = window.get_keyboard()

            def on_key(event, *_args):
                key = event.input
                if event.type in (
                    carb.input.KeyboardEventType.KEY_PRESS,
                    carb.input.KeyboardEventType.KEY_REPEAT,
                ):
                    self.held.add(key)
                    if (
                        key == carb.input.KeyboardInput.R
                        and event.type == carb.input.KeyboardEventType.KEY_PRESS
                    ):
                        self.reset_everything()
                    elif (
                        key == carb.input.KeyboardInput.P
                        and event.type == carb.input.KeyboardEventType.KEY_PRESS
                    ):
                        self.capture_inserted_object_pose()
                elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
                    self.held.discard(key)
                return True

            self._subscription = self._input.subscribe_to_keyboard_events(
                self._keyboard, on_key
            )
        except Exception as error:
            print(f"[tracked-g1] Keyboard input unavailable: {error}")

    def _write_track_state(self, left: float, right: float) -> None:
        values = {
            "leftVelocity": left,
            "rightVelocity": right,
            "actualLeftVelocity": self.track_drive.left_velocity,
            "actualRightVelocity": self.track_drive.right_velocity,
            "leftDistance": self.track_drive.left_distance,
            "rightDistance": self.track_drive.right_distance,
        }
        for name, value in values.items():
            self.track_attributes[name].Set(float(value))

    def _robot_pose(self) -> tuple[np.ndarray, np.ndarray]:
        mount_x, mount_y, mount_z = W1_UPPER_MOUNT_LOCAL
        cosine = math.cos(self.yaw)
        sine = math.sin(self.yaw)
        position = np.asarray(
            [
                self.x + cosine * mount_x - sine * mount_y,
                self.y + sine * mount_x + cosine * mount_y,
                mount_z + self.lift_height,
            ],
            dtype=np.float32,
        )
        base_orientation = (
            math.cos(0.5 * self.yaw),
            0.0,
            0.0,
            math.sin(0.5 * self.yaw),
        )
        bw, bx, by, bz = base_orientation
        mw, mx, my, mz = G1_MOUNT_LOCAL_ORIENTATION
        orientation = np.asarray(
            [
                bw * mw - bx * mx - by * my - bz * mz,
                bw * mx + bx * mw + by * mz - bz * my,
                bw * my - bx * mz + by * mw + bz * mx,
                bw * mz + bx * my - by * mx + bz * mw,
            ],
            dtype=np.float32,
        )
        return position, orientation

    def apply(self) -> None:
        half_yaw = 0.5 * self.yaw
        orientation = np.asarray(
            [math.cos(half_yaw), 0.0, 0.0, math.sin(half_yaw)],
            dtype=np.float32,
        )
        self.root_translate_op.Set(Gf.Vec3d(self.x, self.y, 0.0))
        self.root_orient_op.Set(
            Gf.Quatd(
                float(orientation[0]),
                Gf.Vec3d(0.0, 0.0, float(orientation[3])),
            )
        )
        self.carriage_translate_op.Set(Gf.Vec3d(0.0, 0.0, self.lift_height))

        self.vehicle_body.set_world_poses(
            positions=torch.tensor(
                [[self.x, self.y, 0.0]], dtype=torch.float32, device=self.device
            ),
            orientations=torch.as_tensor(
                orientation[np.newaxis, :], dtype=torch.float32, device=self.device
            ),
        )
        self.carriage_body.set_world_poses(
            positions=torch.tensor(
                [[self.x, self.y, self.lift_height]],
                dtype=torch.float32,
                device=self.device,
            ),
            orientations=torch.as_tensor(
                orientation[np.newaxis, :], dtype=torch.float32, device=self.device
            ),
        )

        robot_position, robot_orientation = self._robot_pose()
        pose = torch.as_tensor(
            np.concatenate((robot_position, robot_orientation))[np.newaxis, :],
            dtype=torch.float32,
            device=self.device,
        )
        self.robot.write_root_pose_to_sim(pose)
        self.robot.write_root_velocity_to_sim(
            torch.zeros((1, 6), dtype=torch.float32, device=self.device)
        )
        self.lift_attributes["height"].Set(float(self.lift_height))
        self.lift_attributes["targetHeight"].Set(float(self.lift_height))

    def update(self, dt: float) -> None:
        if self._light_flicker is not None:
            self._light_flicker.update(dt)
            # Live teleoperation uses the normal renderer cadence. Preview and
            # replay jobs opt into a forced refresh when they need a fresh
            # off-screen sensor sample immediately after the light edit.
            if (
                self._light_flicker.is_enabled
                and os.environ.get("G1_LIGHT_FLICKER_FORCE_CAMERA_REFRESH", "0") == "1"
            ):
                try:
                    self.env.sim.render()
                    for sensor in getattr(self.env.scene, "sensors", {}).values():
                        if "camera" in str(getattr(sensor, "name", "")).lower():
                            sensor.update(0.0, force_recompute=True)
                except Exception as error:
                    print(f"[light-flicker] render refresh unavailable: {error}")
        if self._mine_dust is not None:
            self._mine_dust.update(dt)
        if self._falling_debris is not None:
            self._falling_debris.update(dt)
        if self.success_latched:
            return

        dt = float(np.clip(dt, 1.0e-4, 0.1))
        evaluation_active = BENCHMARK_ACTIVE_PATH.is_file()
        throttle = 0.0 if evaluation_active else float(
            (carb.input.KeyboardInput.W in self.held)
            - (carb.input.KeyboardInput.S in self.held)
        )
        steering = 0.0 if evaluation_active else float(
            (carb.input.KeyboardInput.A in self.held)
            - (carb.input.KeyboardInput.D in self.held)
        )
        left_axis = throttle - steering
        right_axis = throttle + steering
        normalization = max(1.0, abs(left_axis), abs(right_axis))
        left_command = self.track_drive.max_track_speed * left_axis / normalization
        right_command = (
            self.track_drive.max_track_speed * right_axis / normalization
        )
        self.x, self.y, self.yaw = self.track_drive.step(
            self.x,
            self.y,
            self.yaw,
            left_command,
            right_command,
            dt,
        )
        self._write_track_state(left_command, right_command)

        lift_direction = 0.0 if evaluation_active else float(
            (carb.input.KeyboardInput.T in self.held)
            - (carb.input.KeyboardInput.Y in self.held)
        )
        self.lift_height = float(
            np.clip(
                self.lift_height + lift_direction * self.lift_speed * dt,
                self.lift_lower,
                self.lift_upper,
            )
        )
        self.apply()
        if not self._success_check_disabled:
            self._update_success_checker()
        if self._reset_settle_steps_remaining > 0:
            # Give PhysX a short contact-settling window after teleporting the
            # complete assembly. Positions are allowed to resolve naturally,
            # while residual teleport velocities are suppressed.
            self._zero_rod_and_rope_velocities()
            self._reset_settle_steps_remaining -= 1
            if self._reset_settle_steps_remaining == 0:
                self._finish_benchmark_reset()
                print(
                    "[tabletop-grasp] Rod/holder contact settling complete."
                )

    def reset(self) -> None:
        self.x, self.y, self.yaw, self.lift_height = self.initial
        self.track_drive.reset()
        self._write_track_state(0.0, 0.0)
        self.apply()
        print("[tracked-g1] Base pose and lift reset to the saved task placement.")

    def reset_everything(self) -> None:
        """Reset G1, tracked base/lift, and both tabletop objects."""

        env_ids = torch.arange(self.env.num_envs, device=self.device)
        # Restores the managed G1 articulation and Leiguan rigid object to
        # their configured defaults, including zero velocities.
        base_mdp.reset_scene_to_default(self.env, env_ids)
        self.reset()
        # The rod-and-rope assembly is unmanaged, so restore all 27 of its
        # rigid bodies explicitly.
        self.reset_task_objects()
        print(
            "[tabletop-grasp] R reset complete: robot, tracked base/lift, "
            "Leiguan, rod, and all rope links restored."
        )

    def reset_task_objects(self) -> None:
        """Reset and independently randomize both task objects in XY.

        The rod, all 26 rope links, and the kinematic holder share one offset so
        their authored relative poses remain intact. Leiguan receives its own
        independent offset. Both candidates are rejected when any part of their
        sampled centerline/body points comes too close to either gripper in its
        home or pre-reset pose. Z positions and all orientations remain
        unchanged.
        """

        self._clear_success()
        if self._falling_debris is not None and hasattr(self._falling_debris, "reset"):
            self._falling_debris.reset()
        benchmark_request = self._consume_benchmark_reset_request()
        reset_rng = self._object_rng
        if benchmark_request is not None:
            reset_rng = np.random.default_rng(int(benchmark_request["seed"]))
        gripper_guard_points = self._combined_gripper_guard_points()
        leiguan_offset, leiguan_sample_attempts = (
            self._sample_collision_safe_leiguan_offset(
                reset_rng,
                gripper_guard_points,
            )
        )
        rod_offset, rod_sample_attempts = (
            self._sample_collision_safe_rod_offset(
                reset_rng,
                gripper_guard_points,
            )
        )

        env_ids = torch.arange(self.env.num_envs, device=self.device)
        # Move the kinematic holder through PhysX first. Unlike changing a USD
        # xform on a static collider, this updates its collision actor in the
        # same reset pass as the dynamic rod and rope bodies.
        self._reset_managed_object(self.holder, rod_offset, env_ids)
        self._reset_managed_object(self.leiguan, leiguan_offset, env_ids)

        rod_offset_tensor = torch.tensor(
            [float(rod_offset[0]), float(rod_offset[1]), 0.0],
            dtype=torch.float32,
            device=self.device,
        )

        for view, (positions, orientations) in zip(
            self._task_object_views, self._task_object_initial_poses
        ):
            view.set_velocities(
                torch.zeros(
                    (view.count, 6), dtype=torch.float32, device=self.device
                )
            )
            view.set_world_poses(
                positions=positions + rod_offset_tensor.unsqueeze(0),
                orientations=orientations,
            )
        self._zero_rod_and_rope_velocities()
        self._reset_settle_steps_remaining = self._reset_settle_steps
        if benchmark_request is not None:
            self._benchmark_reset_status = {
                "reset_id": str(benchmark_request["reset_id"]),
                "episode_index": int(benchmark_request["episode_index"]),
                "seed": int(benchmark_request["seed"]),
                "leiguan_offset_xy_m": [float(value) for value in leiguan_offset],
                "leiguan_table_local_depth_offset_m": (
                    self._table_local_depth_offset(leiguan_offset)
                ),
                "leiguan_sample_attempts": leiguan_sample_attempts,
                "rod_offset_xy_m": [float(value) for value in rod_offset],
                "rod_sample_attempts": rod_sample_attempts,
                "object_gripper_clearance_m": OBJECT_GRIPPER_CLEARANCE,
                "settled": False,
                "timestamp": datetime.now().astimezone().isoformat(),
            }
            self._write_json_atomic(
                BENCHMARK_RESET_STATUS_PATH,
                self._benchmark_reset_status,
            )
        print(
            "[tabletop-grasp] Object reset randomization (cm): "
            f"Leiguan dx={100.0 * leiguan_offset[0]:+.1f}, "
            f"dy={100.0 * leiguan_offset[1]:+.1f}, "
            f"table-depth={100.0 * self._table_local_depth_offset(leiguan_offset):+.1f} "
            f"(sample attempts={leiguan_sample_attempts}); "
            f"rod/rope/holder dx={100.0 * rod_offset[0]:+.1f}, "
            f"dy={100.0 * rod_offset[1]:+.1f} "
            f"(sample attempts={rod_sample_attempts}); "
            f"gripper clearance >= {100.0 * OBJECT_GRIPPER_CLEARANCE:.1f} cm."
        )

    @staticmethod
    def _table_local_depth_offset(world_xy_offset) -> float:
        """Project a world-XY offset onto the table's local Y axis."""

        return float(
            -math.sin(TABLE_YAW) * float(world_xy_offset[0])
            + math.cos(TABLE_YAW) * float(world_xy_offset[1])
        )

    def _sample_collision_safe_leiguan_offset(
        self,
        reset_rng: np.random.Generator,
        gripper_guard_points: np.ndarray,
    ) -> tuple[np.ndarray, int]:
        """Sample Leiguan while excluding both grippers and the old depth trap."""

        for attempt in range(1, MAX_OBJECT_SAMPLE_ATTEMPTS + 1):
            candidate = reset_rng.uniform(
                -self.object_random_xy_limit,
                self.object_random_xy_limit,
                size=2,
            )
            if (
                self._table_local_depth_offset(candidate)
                >= LEIGUAN_MIN_TABLE_LOCAL_DEPTH_OFFSET
                and self._has_gripper_clearance(
                    self._leiguan_guard_points(candidate),
                    gripper_guard_points,
                )
            ):
                return candidate, attempt
        raise RuntimeError(
            "Unable to sample a collision-safe Leiguan reset position after "
            f"{MAX_OBJECT_SAMPLE_ATTEMPTS} attempts"
        )

    def _sample_collision_safe_rod_offset(
        self,
        reset_rng: np.random.Generator,
        gripper_guard_points: np.ndarray,
    ) -> tuple[np.ndarray, int]:
        """Sample the complete rod/rope/holder assembly away from both hands."""

        for attempt in range(1, MAX_OBJECT_SAMPLE_ATTEMPTS + 1):
            candidate = reset_rng.uniform(
                -self.object_random_xy_limit,
                self.object_random_xy_limit,
                size=2,
            )
            if self._has_gripper_clearance(
                self._rod_assembly_guard_points(candidate),
                gripper_guard_points,
            ):
                return candidate, attempt
        raise RuntimeError(
            "Unable to sample a gripper-safe rod/rope reset position after "
            f"{MAX_OBJECT_SAMPLE_ATTEMPTS} attempts"
        )

    def _read_gripper_guard_points(self) -> np.ndarray:
        points = (
            self.robot.data.body_pos_w[0, self._gripper_body_indices]
            .detach()
            .cpu()
            .numpy()
            .astype(np.float64)
        )
        return points[np.all(np.isfinite(points), axis=1)]

    def _combined_gripper_guard_points(self) -> np.ndarray:
        current_points = self._read_gripper_guard_points()
        if current_points.size == 0:
            return self._home_gripper_guard_points.copy()
        return np.concatenate(
            (self._home_gripper_guard_points, current_points),
            axis=0,
        )

    def _leiguan_guard_points(self, xy_offset: np.ndarray) -> np.ndarray:
        root_state = (
            self.leiguan.data.default_root_state[0]
            .detach()
            .cpu()
            .numpy()
            .astype(np.float64)
        )
        center = root_state[:3] + (
            self.env.scene.env_origins[0].detach().cpu().numpy()
        )
        center[:2] += xy_offset
        axis = self._rotate_vector_by_quaternion(
            root_state[3:7],
            np.asarray([0.0, 0.0, 1.0], dtype=np.float64),
        )
        distances = np.linspace(
            -LEIGUAN_HALF_LENGTH,
            LEIGUAN_HALF_LENGTH,
            11,
        )
        return center[None, :] + distances[:, None] * axis[None, :]

    def _rod_assembly_guard_points(self, xy_offset: np.ndarray) -> np.ndarray:
        rod_positions, rod_orientations = self._task_object_initial_poses[0]
        rod_origin = rod_positions[0].detach().cpu().numpy().astype(np.float64)
        rod_orientation = (
            rod_orientations[0].detach().cpu().numpy().astype(np.float64)
        )
        rod_axis = self._rotate_vector_by_quaternion(
            rod_orientation,
            np.asarray([0.0, 0.0, 1.0], dtype=np.float64),
        )
        rod_distances = np.linspace(0.0, ROD_LENGTH, 11)
        rod_points = (
            rod_origin[None, :]
            + rod_distances[:, None] * rod_axis[None, :]
        )
        rope_points = (
            self._task_object_initial_poses[1][0]
            .detach()
            .cpu()
            .numpy()
            .astype(np.float64)
        )
        holder_center = (
            self.holder.data.default_root_state[0, :3]
            .detach()
            .cpu()
            .numpy()
            .astype(np.float64)
            + self.env.scene.env_origins[0].detach().cpu().numpy()
        )
        points = np.concatenate(
            (rod_points, rope_points, holder_center[None, :]),
            axis=0,
        )
        points[:, :2] += xy_offset[None, :]
        return points

    @staticmethod
    def _has_gripper_clearance(
        object_points: np.ndarray,
        gripper_points: np.ndarray,
    ) -> bool:
        if gripper_points.size == 0:
            return True
        differences = (
            object_points[:, None, :] - gripper_points[None, :, :]
        )
        minimum_distance = float(
            np.min(np.linalg.norm(differences, axis=2))
        )
        return minimum_distance >= OBJECT_GRIPPER_CLEARANCE

    @staticmethod
    def _write_json_atomic(path: Path, payload: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)

    def _consume_benchmark_reset_request(self) -> dict | None:
        """Consume an evaluator-authored deterministic reset request once."""

        if not BENCHMARK_RESET_REQUEST_PATH.is_file():
            return None
        try:
            request = json.loads(
                BENCHMARK_RESET_REQUEST_PATH.read_text(encoding="utf-8")
            )
            required = ("reset_id", "episode_index", "seed")
            missing = [key for key in required if key not in request]
            if missing:
                raise ValueError(f"missing fields: {missing}")
            request["episode_index"] = int(request["episode_index"])
            request["seed"] = int(request["seed"])
            return request
        except Exception as error:
            print(
                "[tabletop-grasp] Ignoring invalid benchmark reset request "
                f"{BENCHMARK_RESET_REQUEST_PATH}: {error}"
            )
            return None
        finally:
            BENCHMARK_RESET_REQUEST_PATH.unlink(missing_ok=True)

    def _finish_benchmark_reset(self) -> None:
        if self._benchmark_reset_status is None:
            return
        self._benchmark_reset_status["settled"] = True
        self._benchmark_reset_status["timestamp"] = (
            datetime.now().astimezone().isoformat()
        )
        self._write_json_atomic(
            BENCHMARK_RESET_STATUS_PATH,
            self._benchmark_reset_status,
        )
        print(
            "[tabletop-grasp] Benchmark reset settled: "
            f"id={self._benchmark_reset_status['reset_id']}, "
            f"episode={self._benchmark_reset_status['episode_index']}, "
            f"seed={self._benchmark_reset_status['seed']}."
        )
        self._benchmark_reset_status = None

    def _reset_managed_object(self, asset, xy_offset, env_ids) -> None:
        root_state = asset.data.default_root_state[env_ids].clone()
        root_state[:, :3] += self.env.scene.env_origins[env_ids]
        root_state[:, 0] += float(xy_offset[0])
        root_state[:, 1] += float(xy_offset[1])
        asset.write_root_pose_to_sim(root_state[:, :7], env_ids=env_ids)
        asset.write_root_velocity_to_sim(
            torch.zeros(
                (len(env_ids), 6), dtype=torch.float32, device=self.device
            ),
            env_ids=env_ids,
        )

    def _zero_rod_and_rope_velocities(self) -> None:
        for view in self._task_object_views:
            view.set_velocities(
                torch.zeros(
                    (view.count, 6), dtype=torch.float32, device=self.device
                )
            )

    @staticmethod
    def _rotate_vector_by_quaternion(
        quaternion: np.ndarray,
        vector: np.ndarray,
    ) -> np.ndarray:
        conjugate = quaternion.copy()
        conjugate[1:] *= -1.0
        rotated = TabletopGraspController._quaternion_multiply(
            TabletopGraspController._quaternion_multiply(
                quaternion,
                np.concatenate(([0.0], vector)),
            ),
            conjugate,
        )
        return rotated[1:]

    def _success_geometry(self) -> tuple[bool, float, float]:
        """Check cavity containment and insertion depth in Leiguan's frame."""

        leiguan_pose = (
            self.leiguan.data.root_pose_w[0].detach().cpu().numpy().astype(np.float64)
        )
        rod_positions, rod_orientations = self.rod_body.get_world_poses(clone=True)
        rod_pose = np.concatenate(
            (
                rod_positions[0].detach().cpu().numpy(),
                rod_orientations[0].detach().cpu().numpy(),
            )
        ).astype(np.float64)
        rod_origin, rod_orientation = self._pose_relative_to(
            leiguan_pose,
            rod_pose,
        )
        rod_axis = self._rotate_vector_by_quaternion(
            rod_orientation,
            np.asarray([0.0, 0.0, 1.0], dtype=np.float64),
        )

        axis_z = float(rod_axis[2])
        if abs(axis_z) < 1.0e-8:
            return False, 0.0, float("inf")

        first_crossing = (-LEIGUAN_HALF_LENGTH - rod_origin[2]) / axis_z
        second_crossing = (LEIGUAN_HALF_LENGTH - rod_origin[2]) / axis_z
        inserted_start = max(0.0, min(first_crossing, second_crossing))
        inserted_end = min(ROD_LENGTH, max(first_crossing, second_crossing))
        insertion_depth = max(0.0, inserted_end - inserted_start)
        if insertion_depth <= 0.0:
            return False, 0.0, float("inf")

        # The radial norm along a straight centerline is convex, so its maximum
        # over the inserted interval is attained at one of the two endpoints.
        start_point = rod_origin + inserted_start * rod_axis
        end_point = rod_origin + inserted_end * rod_axis
        max_radial_error = max(
            float(np.linalg.norm(start_point[:2])),
            float(np.linalg.norm(end_point[:2])),
        )
        inside_cavity = (
            max_radial_error + ROD_MAX_RADIUS
            <= LEIGUAN_INNER_RADIUS + RADIAL_TOLERANCE
        )
        success = inside_cavity and insertion_depth > MIN_INSERTION_DEPTH
        return success, insertion_depth, max_radial_error

    def _update_success_checker(self) -> None:
        success, insertion_depth, max_radial_error = self._success_geometry()
        self.success_insertion_depth = insertion_depth
        self.success_max_radial_error = max_radial_error
        if not success or self.success_latched:
            return

        self.success_latched = True
        SUCCESS_MARKER_PATH.write_text(
            json.dumps(
                {
                    "success": True,
                    "insertion_depth_m": insertion_depth,
                    "max_radial_error_m": max_radial_error,
                    "timestamp": datetime.now().astimezone().isoformat(),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print("\n" + "=" * 72)
        print("SUCCESS")
        print(
            "[tabletop-grasp] Rod is inside Leiguan: "
            f"insertion_depth={insertion_depth:.4f} m, "
            f"max_radial_error={1000.0 * max_radial_error:.2f} mm."
        )
        print(
            "[tabletop-grasp] Simulation, robot commands, camera recording, "
            "and displayed frames are frozen. Press right [A] to save or "
            "terminal [s] to discard/reset."
        )
        print("=" * 72 + "\n")

    @property
    def is_success_frozen(self) -> bool:
        return self.success_latched

    def _clear_success(self) -> None:
        was_latched = self.success_latched
        self.success_latched = False
        self.success_insertion_depth = 0.0
        self.success_max_radial_error = float("inf")
        SUCCESS_MARKER_PATH.unlink(missing_ok=True)
        if was_latched:
            print("[tabletop-grasp] Success freeze cleared by reset.")

    @staticmethod
    def _quaternion_multiply(left: np.ndarray, right: np.ndarray) -> np.ndarray:
        """Multiply two scalar-first (w, x, y, z) quaternions."""

        lw, lx, ly, lz = left
        rw, rx, ry, rz = right
        return np.asarray(
            [
                lw * rw - lx * rx - ly * ry - lz * rz,
                lw * rx + lx * rw + ly * rz - lz * ry,
                lw * ry - lx * rz + ly * rw + lz * rx,
                lw * rz + lx * ry - ly * rx + lz * rw,
            ],
            dtype=np.float64,
        )

    @classmethod
    def _pose_relative_to(
        cls,
        reference_pose: np.ndarray,
        target_pose: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return the target position and orientation in the reference frame."""

        reference_position = reference_pose[:3].astype(np.float64)
        reference_orientation = reference_pose[3:7].astype(np.float64)
        target_position = target_pose[:3].astype(np.float64)
        target_orientation = target_pose[3:7].astype(np.float64)
        reference_inverse = reference_orientation.copy()
        reference_inverse[1:] *= -1.0
        reference_inverse /= np.dot(reference_orientation, reference_orientation)

        delta = target_position - reference_position
        delta_quaternion = np.concatenate(([0.0], delta))
        relative_position = cls._quaternion_multiply(
            cls._quaternion_multiply(reference_inverse, delta_quaternion),
            reference_orientation,
        )[1:]
        relative_orientation = cls._quaternion_multiply(
            reference_inverse,
            target_orientation,
        )
        relative_orientation /= np.linalg.norm(relative_orientation)
        return relative_position, relative_orientation

    def capture_inserted_object_pose(self) -> None:
        """Persist the current rod pose relative to Leiguan plus full body poses."""

        leiguan_pose = (
            self.leiguan.data.root_pose_w[0].detach().cpu().numpy().astype(np.float64)
        )
        rod_positions, rod_orientations = self.rod_body.get_world_poses(clone=True)
        rope_positions, rope_orientations = self.rope_bodies.get_world_poses(
            clone=True
        )
        holder_pose = (
            self.holder.data.root_pose_w[0].detach().cpu().numpy().astype(np.float64)
        )
        rod_pose = np.concatenate(
            (
                rod_positions[0].detach().cpu().numpy(),
                rod_orientations[0].detach().cpu().numpy(),
            )
        ).astype(np.float64)
        relative_position, relative_orientation = self._pose_relative_to(
            leiguan_pose,
            rod_pose,
        )

        payload = {
            "format": "g1_tabletop_grasp_inserted_pose_v1",
            "captured_at": datetime.now().astimezone().isoformat(),
            "frames": {
                "reference": "leiguan",
                "target": "chaganshengzi_rod",
                "quaternion_order": "wxyz",
                "units": "meters",
            },
            "rod_relative_to_leiguan": {
                "position": relative_position.tolist(),
                "orientation": relative_orientation.tolist(),
            },
            "world_poses": {
                "leiguan": leiguan_pose.tolist(),
                "rod": rod_pose.tolist(),
                "holder": holder_pose.tolist(),
                "rope_positions": rope_positions.detach().cpu().tolist(),
                "rope_orientations": rope_orientations.detach().cpu().tolist(),
            },
        }
        temporary_path = INSERTED_POSE_PATH.with_suffix(".json.tmp")
        temporary_path.write_text(
            json.dumps(payload, indent=2),
            encoding="utf-8",
        )
        temporary_path.replace(INSERTED_POSE_PATH)
        print(
            "[tabletop-grasp] Saved inserted rod pose relative to Leiguan: "
            f"position={relative_position.tolist()}, "
            f"orientation(wxyz)={relative_orientation.tolist()} -> "
            f"{INSERTED_POSE_PATH}"
        )

    def get_recording_state(self) -> dict:
        """Publish live tabletop poses; the collector stores the rod pose per frame."""
        def pose(view, index=0):
            if hasattr(view, "get_world_poses"):
                positions, orientations = view.get_world_poses(clone=True)
            else:
                root_pose = view.data.root_pose_w
                positions, orientations = root_pose[:, :3], root_pose[:, 3:7]
            return np.concatenate((
                positions[index].detach().cpu().numpy(),
                orientations[index].detach().cpu().numpy(),
            )).astype(np.float64).tolist()

        rod_pose = pose(self.rod_body)
        return {
            "schema": "tabletop_grasp_recording_state_v2",
            "task_id": "Isaac-TabletopGrasp-G129-Inspire-Joint",
            "timestamp_ns": time.time_ns(),
            "active_object": "chaganshengzi_rod",
            "object_root_pose_wxyz": rod_pose,
            "replay_initial_state": {
                "objects": {
                    "leiguan": {"root_pose_wxyz": pose(self.leiguan)},
                    "rod": {"root_pose_wxyz": rod_pose},
                    "holder": {"root_pose_wxyz": pose(self.holder)},
                }
            },
            "objects": {
                "leiguan": {"root_pose_wxyz": pose(self.leiguan)},
                "rod": {"root_pose_wxyz": rod_pose},
                "holder": {"root_pose_wxyz": pose(self.holder)},
            },
        }

    def close(self) -> None:
        self._clear_success()
        if self._headlamp is not None:
            self._headlamp.close()
            self._headlamp = None
        if self._auxiliary_lights is not None:
            self._auxiliary_lights.close()
            self._auxiliary_lights = None
        if self._light_flicker is not None:
            self._light_flicker.close()
            self._light_flicker = None
        if self._mine_dust is not None:
            self._mine_dust.close()
            self._mine_dust = None
        if self._falling_debris is not None:
            self._falling_debris.close()
            self._falling_debris = None
        if self._input is not None and self._subscription is not None:
            try:
                self._input.unsubscribe_to_keyboard_events(
                    self._keyboard, self._subscription
                )
            except Exception:
                pass
        self._subscription = None
        self.held.clear()
