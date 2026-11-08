#!/usr/bin/env python3
"""Pinocchio/Pink constrained IK for the W1 USD arm chains.

The task asset has no URDF.  This module constructs one exact seven-joint
Pinocchio model per arm directly from the composed USD revolute-joint frames,
axes, limits, scale, and functional-gripper frame.  Pink then solves bounded
differential IK as a quadratic program.
"""

from __future__ import annotations

import math
import os

import numpy as np


_IMPORT_ERROR: Exception | None = None
try:
    import pinocchio as pin
    import pink
    import qpsolvers
    from pink.limits import AccelerationLimit, ConfigurationLimit, VelocityLimit
    from pink.tasks import DampingTask, FrameTask, LowAccelerationTask, PostureTask
    from pxr import Gf, UsdGeom, UsdPhysics
except (ImportError, OSError) as error:
    _IMPORT_ERROR = error


class _PinocchioArmIK:
    """One exact USD-derived W1 arm model and its stateful Pink QP."""

    def __init__(
        self,
        stage,
        side: str,
        robot_path: str,
        end_effector_path: str,
        model_scale: float,
    ) -> None:
        self.side = side
        self.model_scale = float(model_scale)
        self.frame_name = f"{side}_functional_gripper"
        self.solver = os.environ.get("W1_PINOCCHIO_QP_SOLVER", "quadprog")
        self.max_joint_velocity = float(
            os.environ.get("W1_PINOCCHIO_MAX_JOINT_VELOCITY", "1.2")
        )
        self.max_joint_acceleration = float(
            os.environ.get("W1_PINOCCHIO_MAX_JOINT_ACCELERATION", "6.0")
        )
        self.position_cost = float(
            os.environ.get("W1_PINOCCHIO_POSITION_COST", "1.0")
        )
        self.orientation_cost = float(
            os.environ.get("W1_PINOCCHIO_ORIENTATION_COST", "0.35")
        )
        self.posture_cost = float(
            os.environ.get("W1_PINOCCHIO_POSTURE_COST", "0.002")
        )
        self.damping_cost = float(
            os.environ.get("W1_PINOCCHIO_DAMPING_COST", "0.01")
        )
        self.low_acceleration_cost = float(
            os.environ.get("W1_PINOCCHIO_LOW_ACCELERATION_COST", "0.02")
        )
        self.singularity_escape = math.radians(
            float(os.environ.get("W1_PINOCCHIO_SINGULARITY_ESCAPE_DEG", "3.0"))
        )
        if self.solver not in qpsolvers.available_solvers:
            raise RuntimeError(
                f"Pink QP solver {self.solver!r} is unavailable; "
                f"available={qpsolvers.available_solvers}"
            )
        if (
            not math.isfinite(self.model_scale)
            or self.model_scale <= 0.0
            or self.max_joint_velocity <= 0.0
            or self.max_joint_acceleration <= 0.0
        ):
            raise RuntimeError("Invalid Pinocchio model scale or motion limits")

        self.model, self.frame_id = self._build_model(
            stage,
            robot_path,
            end_effector_path,
        )
        self.data = self.model.createData()
        self.validation_data = self.model.createData()
        self.configuration = pink.Configuration(
            self.model,
            self.data,
            pin.neutral(self.model),
        )
        self.frame_task = FrameTask(
            self.frame_name,
            position_cost=self.position_cost,
            orientation_cost=self.orientation_cost,
            gain=0.7,
            lm_damping=1.0e-3,
        )
        self.posture_task = PostureTask(cost=self.posture_cost)
        self.damping_task = DampingTask(cost=self.damping_cost)
        self.low_acceleration_task = LowAccelerationTask(
            cost=self.low_acceleration_cost
        )
        self.acceleration_limit = AccelerationLimit(
            self.model,
            np.full(self.model.nv, self.max_joint_acceleration),
        )
        self.limits = (
            ConfigurationLimit(self.model, config_limit_gain=0.5),
            VelocityLimit(self.model),
            self.acceleration_limit,
        )
        self.tasks = (
            self.frame_task,
            self.posture_task,
            self.damping_task,
            self.low_acceleration_task,
        )
        self.previous_velocity = np.zeros(self.model.nv, dtype=np.float64)
        self.success_count = 0
        self.fallback_count = 0
        self.last_failure = ""

        # The constructed zero pose must reproduce the actual USD geometry.
        self.configuration.update(pin.neutral(self.model))
        zero_pose = self.configuration.get_transform_frame_to_world(
            self.frame_name
        )
        if not (
            np.all(np.isfinite(zero_pose.translation))
            and np.all(np.isfinite(zero_pose.rotation))
        ):
            raise RuntimeError(f"{side} Pinocchio zero-pose FK is non-finite")

    @staticmethod
    def _rigid_world(cache, prim):
        matrix = cache.GetLocalToWorldTransform(prim)
        position = np.asarray(matrix.ExtractTranslation(), dtype=np.float64)
        quaternion = Gf.Transform(matrix).GetRotation().GetQuat()
        # Gf matrices operate on row vectors; Pinocchio uses column vectors.
        rotation = np.asarray(Gf.Matrix3d(quaternion), dtype=np.float64).T
        return pin.SE3(rotation, position)

    def _joint_world(self, stage, cache, joint):
        body_targets = joint.GetBody0Rel().GetTargets()
        if len(body_targets) != 1:
            raise RuntimeError(
                f"{joint.GetPath()} must have exactly one parent body"
            )
        parent_prim = stage.GetPrimAtPath(body_targets[0])
        if not parent_prim.IsValid():
            raise RuntimeError(f"Missing parent body: {body_targets[0]}")
        parent_world = self._rigid_world(cache, parent_prim)
        local_position = (
            np.asarray(joint.GetLocalPos0Attr().Get(), dtype=np.float64)
            * self.model_scale
        )
        local_quaternion = joint.GetLocalRot0Attr().Get()
        local_rotation = np.asarray(
            Gf.Matrix3d(local_quaternion),
            dtype=np.float64,
        ).T
        return parent_world * pin.SE3(local_rotation, local_position)

    def _build_model(self, stage, robot_path: str, end_effector_path: str):
        model = pin.Model()
        cache = UsdGeom.XformCache()
        parent_joint_id = 0
        previous_joint_world = None

        for number in range(1, 8):
            joint_path = f"{robot_path}/Physics/{self.side}_joint{number}"
            joint_prim = stage.GetPrimAtPath(joint_path)
            if not joint_prim.IsValid():
                raise RuntimeError(f"Missing W1 arm joint: {joint_path}")
            usd_joint = UsdPhysics.RevoluteJoint(joint_prim)
            current_joint_world = self._joint_world(stage, cache, usd_joint)
            placement = (
                pin.SE3.Identity()
                if previous_joint_world is None
                else previous_joint_world.inverse() * current_joint_world
            )
            axis = str(usd_joint.GetAxisAttr().Get())
            joint_model = {
                "X": pin.JointModelRX(),
                "Y": pin.JointModelRY(),
                "Z": pin.JointModelRZ(),
            }.get(axis)
            if joint_model is None:
                raise RuntimeError(f"Unsupported USD joint axis {axis}: {joint_path}")
            joint_id = model.addJoint(
                parent_joint_id,
                joint_model,
                placement,
                f"{self.side}_joint{number}",
            )
            model.appendBodyToJoint(
                joint_id,
                pin.Inertia.Zero(),
                pin.SE3.Identity(),
            )
            index_q = model.joints[joint_id].idx_q
            index_v = model.joints[joint_id].idx_v
            model.lowerPositionLimit[index_q] = math.radians(
                float(usd_joint.GetLowerLimitAttr().Get())
            )
            model.upperPositionLimit[index_q] = math.radians(
                float(usd_joint.GetUpperLimitAttr().Get())
            )
            model.velocityLimit[index_v] = self.max_joint_velocity
            parent_joint_id = joint_id
            previous_joint_world = current_joint_world

        end_effector_prim = stage.GetPrimAtPath(end_effector_path)
        if not end_effector_prim.IsValid():
            raise RuntimeError(f"Missing W1 end effector: {end_effector_path}")
        end_effector_world = self._rigid_world(cache, end_effector_prim)
        frame_placement = previous_joint_world.inverse() * end_effector_world
        frame_id = model.addFrame(
            pin.Frame(
                self.frame_name,
                parent_joint_id,
                parent_joint_id,
                frame_placement,
                pin.FrameType.OP_FRAME,
            )
        )
        if model.nq != 7 or model.nv != 7:
            raise RuntimeError(
                f"Unexpected {self.side} Pinocchio model size: "
                f"nq={model.nq}, nv={model.nv}"
            )
        return model, frame_id

    def reset(self) -> None:
        self.previous_velocity.fill(0.0)
        self.low_acceleration_task.set_last_integration(
            self.previous_velocity,
            1.0 / 60.0,
        )
        self.acceleration_limit.set_last_integration(
            self.previous_velocity,
            1.0 / 60.0,
        )

    def _fail(self, reason: str) -> None:
        self.fallback_count += 1
        self.last_failure = reason
        self.previous_velocity.fill(0.0)

    def solve(
        self,
        current_joints: np.ndarray,
        current_world_rotation: np.ndarray,
        position_error_world: np.ndarray,
        rotation_error_world: np.ndarray,
        dt: float,
    ) -> np.ndarray | None:
        try:
            current = np.asarray(current_joints, dtype=np.float64)
            world_rotation = np.asarray(
                current_world_rotation,
                dtype=np.float64,
            )
            position_error = np.asarray(position_error_world, dtype=np.float64)
            rotation_error = np.asarray(rotation_error_world, dtype=np.float64)
            if (
                current.shape != (7,)
                or world_rotation.shape != (3, 3)
                or position_error.shape != (3,)
                or rotation_error.shape != (3,)
                or not np.all(np.isfinite(current))
                or not np.all(np.isfinite(world_rotation))
                or not np.all(np.isfinite(position_error))
                or not np.all(np.isfinite(rotation_error))
            ):
                self._fail("invalid input shape or non-finite value")
                return None

            step_dt = float(np.clip(dt, 1.0 / 240.0, 1.0 / 30.0))
            current = np.clip(
                current,
                self.model.lowerPositionLimit + 1.0e-7,
                self.model.upperPositionLimit - 1.0e-7,
            )
            self.configuration.update(current)
            current_model_pose = self.configuration.get_transform_frame_to_world(
                self.frame_name
            )
            world_from_model_rotation = (
                world_rotation @ current_model_pose.rotation.T
            )
            target_position = current_model_pose.translation + (
                world_from_model_rotation.T @ position_error
            )
            model_rotation_error = world_from_model_rotation.T @ rotation_error
            target_rotation = (
                pin.exp3(model_rotation_error) @ current_model_pose.rotation
            )
            target_pose = pin.SE3(target_rotation, target_position)
            self.frame_task.set_target(target_pose)

            posture_target = current.copy()
            task_motion_requested = (
                np.linalg.norm(position_error) > 1.0e-4
                or np.linalg.norm(rotation_error) > 1.0e-4
            )
            frame_jacobian = self.configuration.get_frame_jacobian(
                self.frame_name
            )
            if (
                task_motion_requested
                and self.singularity_escape > 0.0
                and np.linalg.matrix_rank(frame_jacobian, tol=1.0e-5) < 6
                and abs(current[3]) < self.singularity_escape
            ):
                # Only while the user commands motion: introduce a tiny elbow
                # bend so the all-zero rank-5 pose can leave its singularity.
                posture_target[3] = self.singularity_escape
            self.posture_task.set_target(posture_target)
            self.low_acceleration_task.set_last_integration(
                self.previous_velocity,
                step_dt,
            )
            self.acceleration_limit.set_last_integration(
                self.previous_velocity,
                step_dt,
            )
            velocity = pink.solve_ik(
                self.configuration,
                self.tasks,
                step_dt,
                solver=self.solver,
                damping=1.0e-6,
                limits=self.limits,
                safety_break=True,
            )
            velocity = np.asarray(velocity, dtype=np.float64)
            if velocity.shape != (7,) or not np.all(np.isfinite(velocity)):
                self._fail("Pink returned a non-finite joint velocity")
                return None
            if np.max(np.abs(velocity)) > self.max_joint_velocity + 1.0e-5:
                self._fail("Pink violated the configured joint-velocity limit")
                return None

            candidate = pin.integrate(self.model, current, velocity * step_dt)
            if not np.all(np.isfinite(candidate)):
                self._fail("Pinocchio integration returned a non-finite target")
                return None
            if np.any(candidate < self.model.lowerPositionLimit - 1.0e-6) or np.any(
                candidate > self.model.upperPositionLimit + 1.0e-6
            ):
                self._fail("Pink target is outside USD joint limits")
                return None

            pin.framesForwardKinematics(
                self.model,
                self.validation_data,
                candidate,
            )
            candidate_pose = self.validation_data.oMf[self.frame_id]
            error_before = float(
                np.linalg.norm(
                    pin.log6(current_model_pose.actInv(target_pose)).vector
                )
            )
            error_after = float(
                np.linalg.norm(pin.log6(candidate_pose.actInv(target_pose)).vector)
            )
            if error_after > error_before + max(0.01, 0.5 * error_before):
                self._fail(
                    "Pink step increased task error excessively "
                    f"({error_before:.4g} -> {error_after:.4g})"
                )
                return None

            self.previous_velocity[:] = velocity
            self.success_count += 1
            self.last_failure = ""
            return candidate
        except (AssertionError, ArithmeticError, RuntimeError, ValueError) as error:
            self._fail(str(error))
            return None


class W1PinocchioIK:
    """Two-arm Pinocchio/Pink IK facade used by the Quest controller."""

    def __init__(
        self,
        stage,
        robot_path: str,
        end_effector_paths: dict[str, str],
        model_scale: float = 1.1,
    ) -> None:
        if _IMPORT_ERROR is not None:
            raise RuntimeError(
                "Pinocchio/Pink is unavailable in the Isaac Sim Python environment"
            ) from _IMPORT_ERROR
        self.arms = {
            side: _PinocchioArmIK(
                stage,
                side,
                robot_path,
                end_effector_paths[side],
                model_scale,
            )
            for side in ("left", "right")
        }
        self.version = f"Pinocchio {pin.__version__} / Pink {pink.__version__}"

    def solve(
        self,
        side: str,
        current_joints: np.ndarray,
        current_world_rotation: np.ndarray,
        position_error_world: np.ndarray,
        rotation_error_world: np.ndarray,
        dt: float,
    ) -> np.ndarray | None:
        return self.arms[side].solve(
            current_joints,
            current_world_rotation,
            position_error_world,
            rotation_error_world,
            dt,
        )

    def reset(self, side: str | None = None) -> None:
        """Clear QP integration history for one arm or both arms."""
        if side is None:
            for arm in self.arms.values():
                arm.reset()
            return
        self.arms[side].reset()
