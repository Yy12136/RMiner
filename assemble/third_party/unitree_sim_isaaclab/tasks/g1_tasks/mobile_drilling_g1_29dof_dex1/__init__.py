# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0

import gymnasium as gym

from . import mobile_drilling_g1_29dof_dex1_env_cfg


gym.register(
    id="Isaac-MobileDrilling-G129-Dex1-Joint",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": (
            mobile_drilling_g1_29dof_dex1_env_cfg.MobileDrillingG129Dex1EnvCfg
        ),
    },
    disable_env_checker=True,
)
