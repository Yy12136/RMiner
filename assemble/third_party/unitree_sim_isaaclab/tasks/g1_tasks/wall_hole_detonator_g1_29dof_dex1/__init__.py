# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache-2.0

import gymnasium as gym

from . import wall_hole_detonator_g1_29dof_dex1_env_cfg


gym.register(
    id="Isaac-WallHoleDetonator-G129-Dex1-Joint",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": (
            wall_hole_detonator_g1_29dof_dex1_env_cfg.WallHoleDetonatorG129Dex1EnvCfg
        ),
    },
    disable_env_checker=True,
)

