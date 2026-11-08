# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0

import gymnasium as gym

from . import tabletop_grasp_g1_29dof_inspire_env_cfg


gym.register(
    id="Isaac-TabletopGrasp-G129-Inspire-Joint",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": (
            tabletop_grasp_g1_29dof_inspire_env_cfg.TabletopGraspG129InspireEnvCfg
        ),
    },
    disable_env_checker=True,
)
