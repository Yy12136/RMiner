"""Declarative specifications for the Task02--06 evaluation chain."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


ARM_NAMES = (
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint", "left_elbow_joint", "left_wrist_roll_joint",
    "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint", "right_elbow_joint", "right_wrist_roll_joint",
    "right_wrist_pitch_joint", "right_wrist_yaw_joint",
)
JOINT_STATE_NAMES = ARM_NAMES + ("left_gripper_joint", "right_gripper_joint")
CAMERA_SHAPES = {
    "observation.images.head": (3, 480, 640),
    "observation.images.left_wrist": (3, 480, 640),
    "observation.images.right_wrist": (3, 480, 640),
}
MOBILE_STATE_NAMES = (
    "base_forward_position_m", "base_lateral_position_m", "base_yaw_delta_rad",
    "lift_height_m", "left_track_velocity_mps", "right_track_velocity_mps",
    "lift_velocity_mps", "cart_world_x_m", "cart_world_y_m",
    "cart_world_z_m", "cart_world_qw", "cart_world_qx", "cart_world_qy",
    "cart_world_qz",
)
MOBILE_ACTION_NAMES = (
    "left_track_velocity_target_mps", "right_track_velocity_target_mps",
    "lift_velocity_target_mps",
)
OBJECT_POSE_NAMES = (
    "object_world_x_m", "object_world_y_m", "object_world_z_m",
    "object_world_qw", "object_world_qx", "object_world_qy", "object_world_qz",
)


@dataclass(frozen=True)
class TaskSpec:
    key: str
    slug: str
    task_dir: str
    sim_task: str
    instruction: str
    dataset: str
    state_names: tuple[str, ...]
    action_names: tuple[str, ...]
    initial_grippers: tuple[float, float]
    right_arm_only: bool
    left_arm_reset_target: tuple[float, ...] = (0.0,) * 7
    telemetry_schema: str | None = None

    @property
    def state_dim(self) -> int:
        return len(self.state_names)

    @property
    def action_dim(self) -> int:
        return len(self.action_names)

    @property
    def expected_inputs(self) -> dict[str, tuple[int, ...]]:
        return {"observation.state": (self.state_dim,), **CAMERA_SHAPES}

    @property
    def expected_outputs(self) -> dict[str, tuple[int, ...]]:
        return {"action": (self.action_dim,)}

    def task_root(self, assemble_root: Path) -> Path:
        return assemble_root / "task" / self.task_dir


SPECS = {
    "task02": TaskSpec(
        key="task02", slug="roof_bolt_sleeve", task_dir="g1_roof_bolt_sleeve_task",
        sim_task="Isaac-RoofBoltSleeve-G129-Dex1-Joint",
        instruction=("Pick up the coupling sleeve with the right gripper, place it "
                     "over the upright roof bolt, and rotate it while lowering it onto the bolt."),
        dataset="data/lerobot_v30/roof_bolt_sleeve_installation_final",
        state_names=JOINT_STATE_NAMES, action_names=JOINT_STATE_NAMES,
        initial_grippers=(2.945, 3.314), right_arm_only=True,
    ),
    "task03": TaskSpec(
        key="task03", slug="bin_pick_place", task_dir="g1_bin_pick_place_task",
        sim_task="Isaac-BinPickPlace-G129-Dex1-Joint",
        instruction=("Pick up the metal component from the right side of the table "
                     "and place it into the sorting bin on the left."),
        dataset="data/lerobot_v30/sam3d_metal_component_to_randomized_bin_final",
        state_names=JOINT_STATE_NAMES, action_names=JOINT_STATE_NAMES,
        initial_grippers=(2.945, 3.436), right_arm_only=True,
        left_arm_reset_target=(0.15, 0.675, 1.50, 0.0, 0.0, 0.0, 0.0),
    ),
    "task04": TaskSpec(
        key="task04", slug="hand_cart_push", task_dir="g1_hand_cart_push_task",
        sim_task="Isaac-HandCartPush-G129-Dex1-Joint",
        instruction="Grasp both handles of the hand cart and push it straight forward with the mobile base.",
        dataset="data/lerobot_v30/hand_cart_push_final",
        state_names=JOINT_STATE_NAMES + MOBILE_STATE_NAMES,
        action_names=JOINT_STATE_NAMES + MOBILE_ACTION_NAMES,
        initial_grippers=(1.473, 1.473), right_arm_only=False,
        telemetry_schema="tracked_base_v1",
    ),
    "task05": TaskSpec(
        key="task05", slug="valve_handwheel", task_dir="g1_valve_handwheel_task",
        sim_task="Isaac-ValveHandwheel-G129-Dex1-Joint",
        instruction="Rotate the blue handwheel on the three-way ball valve by one full turn.",
        dataset="data/lerobot_v30/valve_handwheel_rotation_final",
        state_names=JOINT_STATE_NAMES, action_names=JOINT_STATE_NAMES,
        initial_grippers=(3.068, 0.921), right_arm_only=False,
    ),
    "task06": TaskSpec(
        key="task06", slug="wall_hole_detonator", task_dir="g1_wall_hole_detonator_task",
        sim_task="Isaac-WallHoleDetonator-G129-Dex1-Joint",
        instruction="Pick up the loaded detonator assembly with the right gripper and insert it into the wall hole.",
        dataset="data/lerobot_v30/wall_hole_loaded_detonator_insertion_final",
        state_names=JOINT_STATE_NAMES + OBJECT_POSE_NAMES,
        action_names=JOINT_STATE_NAMES,
        initial_grippers=(2.945, 2.945), right_arm_only=True,
        telemetry_schema="wall_hole_detonator_replay_v1",
    ),
}


def get_spec(key: str) -> TaskSpec:
    try:
        return SPECS[key]
    except KeyError as error:
        raise ValueError(f"Unknown task {key!r}; choose one of {sorted(SPECS)}") from error
