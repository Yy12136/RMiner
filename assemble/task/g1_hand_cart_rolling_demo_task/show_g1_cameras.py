#!/usr/bin/env python3
"""Launch the shared three-camera viewer with rolling-demo labels."""

import os
import runpy
from pathlib import Path


TASK_ROOT = Path(__file__).resolve().parent
SHARED_VIEWER = TASK_ROOT.parent / "g1_roof_bolt_sleeve_task" / "show_g1_cameras.py"
os.environ.setdefault("G1_CAMERA_WINDOW_NAME", "G1 Rolling Hand-Cart Demo | Head - Left Wrist - Right Wrist")
runpy.run_path(str(SHARED_VIEWER), run_name="__main__")
