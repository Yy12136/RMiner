#!/usr/bin/env python3
"""Compatibility entry point for the renamed video-only augmenter."""

from pathlib import Path
import runpy


TARGET = Path(__file__).resolve().with_name("augment_lerobot_v3_video.py")
runpy.run_path(str(TARGET), run_name="__main__")
