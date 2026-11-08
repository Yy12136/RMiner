# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache-2.0

import gymnasium as gym

from . import hand_cart_push_g1_29dof_dex1_env_cfg


gym.register(
    id="Isaac-HandCartPush-G129-Dex1-Joint",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": (
            hand_cart_push_g1_29dof_dex1_env_cfg.HandCartPushG129Dex1EnvCfg
        ),
    },
    disable_env_checker=True,
)
