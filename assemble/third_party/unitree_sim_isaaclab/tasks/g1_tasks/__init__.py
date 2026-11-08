
# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0  
"""Unitree G1 robot task module
contains various task implementations for the G1 robot, such as pick and place, motion control, etc.
"""

# use relative import
from . import pick_place_cylinder_g1_29dof_dex3
from . import pick_place_cylinder_g1_29dof_dex1
from . import pick_place_cylinder_g1_29dof_inspire

from . import pick_place_redblock_g1_29dof_dex1
from . import pick_place_redblock_g1_29dof_dex3
from . import stack_rgyblock_g1_29dof_dex1
from . import stack_rgyblock_g1_29dof_dex3
from . import stack_rgyblock_g1_29dof_inspire
from . import pick_redblock_into_drawer_g1_29dof_dex1
from . import pick_redblock_into_drawer_g1_29dof_dex3
from . import pick_place_redblock_g1_29dof_inspire
from . import move_cylinder_g1_29dof_dex1_wholebody
from . import move_cylinder_g1_29dof_dex3_wholebody
from . import move_cylinder_g1_29dof_inspire_wholebody
from . import mobile_drilling_g1_29dof_dex1
from . import tabletop_grasp_g1_29dof_dex1
from . import tabletop_grasp_g1_29dof_inspire
from . import roof_bolt_sleeve_g1_29dof_dex1
from . import bin_pick_place_g1_29dof_dex1
from . import roof_bolt_upright_g1_29dof_dex1
from . import hand_cart_push_g1_29dof_dex1
from . import hand_cart_rolling_demo_g1_29dof_dex1
from . import valve_handwheel_g1_29dof_dex1
from . import ground_drilling_g1_29dof_dex1
from . import wall_hole_detonator_g1_29dof_dex1

# export all modules
__all__ = [
        "pick_place_cylinder_g1_29dof_dex3", "pick_place_cylinder_g1_29dof_dex1", 
        "pick_place_redblock_g1_29dof_dex1", "pick_place_redblock_g1_29dof_dex3", 
        "stack_rgyblock_g1_29dof_dex1", "stack_rgyblock_g1_29dof_dex3", 
        "stack_rgyblock_g1_29dof_inspire",
        "pick_redblock_into_drawer_g1_29dof_dex1","pick_redblock_into_drawer_g1_29dof_dex3",
        "pick_place_redblock_g1_29dof_inspire",
        "pick_place_cylinder_g1_29dof_inspire",
        "move_cylinder_g1_29dof_dex1_wholebody",
        "move_cylinder_g1_29dof_dex3_wholebody",
        "move_cylinder_g1_29dof_inspire_wholebody",
        "mobile_drilling_g1_29dof_dex1",
        "tabletop_grasp_g1_29dof_dex1",
        "tabletop_grasp_g1_29dof_inspire",
        "roof_bolt_sleeve_g1_29dof_dex1",
        "bin_pick_place_g1_29dof_dex1",
        "roof_bolt_upright_g1_29dof_dex1",
        "hand_cart_push_g1_29dof_dex1",
        "hand_cart_rolling_demo_g1_29dof_dex1",
        "valve_handwheel_g1_29dof_dex1",
        "ground_drilling_g1_29dof_dex1",
        "wall_hole_detonator_g1_29dof_dex1",
]
