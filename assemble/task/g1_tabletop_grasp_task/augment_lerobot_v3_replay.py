#!/usr/bin/env python3
"""Backward-compatible entry point for the shared augmentation package."""

from pathlib import Path
import runpy


runpy.run_path(
    str(
        Path(__file__).resolve().parents[1]
        / "augmentation/augment_lerobot_v3_video.py"
    ),
    run_name="__main__",
)
