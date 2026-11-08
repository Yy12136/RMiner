#!/usr/bin/env python3
"""Compatibility wrapper for the shared W1 mobile-manipulator teleop."""

import os
import runpy
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
TASK_ROOT = (SCRIPT_DIR.parent / "w1_mobile_drilling_task").resolve()
SHARED_TELEOP = SCRIPT_DIR / "w1_mobile_manipulator_teleop.py"
os.environ.setdefault("W1_TASK_ROOT", str(TASK_ROOT))
runpy.run_path(str(SHARED_TELEOP), run_name="__main__")
