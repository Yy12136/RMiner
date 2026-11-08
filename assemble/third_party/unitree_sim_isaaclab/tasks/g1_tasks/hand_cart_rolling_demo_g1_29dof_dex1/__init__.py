"""Standalone teleoperation demo for the rolling hand-cart visual asset."""

import gymnasium as gym

from . import hand_cart_rolling_demo_g1_29dof_dex1_env_cfg


gym.register(
    id="Isaac-HandCartRollingDemo-G129-Dex1-Joint",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": (
            hand_cart_rolling_demo_g1_29dof_dex1_env_cfg.HandCartRollingDemoG129Dex1EnvCfg
        ),
    },
    disable_env_checker=True,
)
