"""Compatibility loader for the shared falling-debris augmentation."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


_SOURCE = Path(__file__).resolve().parents[5] / "task/augmentation/falling_debris_3d.py"
if not _SOURCE.is_file():
    raise FileNotFoundError(f"Falling debris implementation is missing: {_SOURCE}")
_MODULE_NAME = "task_replay_falling_debris_3d"
_MODULE = sys.modules.get(_MODULE_NAME)
if _MODULE is None:
    _SPEC = importlib.util.spec_from_file_location(_MODULE_NAME, _SOURCE)
    if _SPEC is None or _SPEC.loader is None:
        raise ImportError(f"Unable to load falling debris implementation: {_SOURCE}")
    _MODULE = importlib.util.module_from_spec(_SPEC)
    sys.modules[_MODULE_NAME] = _MODULE
    _SPEC.loader.exec_module(_MODULE)

FallingDebris3D = _MODULE.FallingDebris3D

__all__ = ["FallingDebris3D"]
