"""Load the shared Blender/OpenVDB dust implementation from task/augmentation."""

from __future__ import annotations

import importlib.util
from pathlib import Path


_SOURCE = Path(__file__).resolve().parents[5] / "task/augmentation/blender_vdb_dust_3d.py"
if not _SOURCE.is_file():
    raise FileNotFoundError(f"Blender VDB dust implementation is missing: {_SOURCE}")
_SPEC = importlib.util.spec_from_file_location("task_replay_blender_vdb_dust_3d", _SOURCE)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"Unable to load Blender VDB dust implementation: {_SOURCE}")
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)

BlenderVdbDust3D = _MODULE.BlenderVdbDust3D

__all__ = ["BlenderVdbDust3D"]
