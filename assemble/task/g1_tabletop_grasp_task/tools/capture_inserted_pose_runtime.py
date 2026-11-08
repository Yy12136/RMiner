"""Capture the live inserted pose from Isaac Sim's Script Editor."""

from tasks.g1_tasks.tabletop_grasp_g1_29dof_dex1.tabletop_grasp_g1_29dof_dex1_env_cfg import (
    _CONTROLLERS,
)
controller = next(iter(_CONTROLLERS.values()))
controller.capture_inserted_object_pose()
