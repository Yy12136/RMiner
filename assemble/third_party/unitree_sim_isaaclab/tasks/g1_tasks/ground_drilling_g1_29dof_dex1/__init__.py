# Copyright (c) 2025 Unitree Robotics Co., Ltd.
# License: Apache-2.0

import gymnasium as gym

from . import ground_drilling_g1_29dof_dex1_env_cfg


gym.register(
    id="Isaac-GroundDrilling-G129-Dex1-Joint",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": (
            ground_drilling_g1_29dof_dex1_env_cfg.GroundDrillingG129Dex1EnvCfg
        ),
    },
    disable_env_checker=True,
)
