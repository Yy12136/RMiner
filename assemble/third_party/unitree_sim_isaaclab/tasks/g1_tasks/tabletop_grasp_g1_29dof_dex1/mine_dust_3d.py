"""Compatibility loader for the shared task-level dust implementation."""

from __future__ import annotations

import importlib.util
from pathlib import Path


_SOURCE = (
    Path(__file__).resolve().parents[5]
    / "task/augmentation/mine_dust_3d.py"
)
if not _SOURCE.is_file():
    raise FileNotFoundError(f"Replay dust implementation is missing: {_SOURCE}")
_SPEC = importlib.util.spec_from_file_location("task_replay_mine_dust_3d", _SOURCE)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"Unable to load replay dust implementation: {_SOURCE}")
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)

MineDust3D = _MODULE.MineDust3D

__all__ = ["MineDust3D"]
