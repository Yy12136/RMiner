#!/usr/bin/env python3
"""Launch the shared three-camera viewer with hand-cart task labels."""

from __future__ import annotations

import os
import runpy
from pathlib import Path


TASK_ROOT = Path(__file__).resolve().parent
SHARED_VIEWER = TASK_ROOT.parent / "g1_roof_bolt_sleeve_task" / "show_g1_cameras.py"
os.environ.setdefault(
    "G1_TASK_SUCCESS_FILE", f"/tmp/g1_hand_cart_push_{os.getuid()}.success"
)
os.environ.setdefault(
    "G1_CAMERA_WINDOW_NAME",
    "G1 Hand Cart Push Cameras | Head - Left Wrist - Right Wrist",
)
runpy.run_path(str(SHARED_VIEWER), run_name="__main__")
