"""Load the shared Task01 volume dust implementation from task/augmentation."""

from __future__ import annotations

import importlib.util
from pathlib import Path


_SOURCE = Path(__file__).resolve().parents[5] / "task/augmentation/volume_dust_3d.py"
_SPEC = importlib.util.spec_from_file_location("task_replay_volume_dust_3d", _SOURCE)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"Unable to load volume dust implementation: {_SOURCE}")
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)

VolumeDust3D = _MODULE.VolumeDust3D

__all__ = ["VolumeDust3D"]
