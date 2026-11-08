"""Compatibility loader for the task-level replay augmentation package."""

from __future__ import annotations

import importlib.util
from pathlib import Path


_SOURCE = (
    # tools/ is inside assemble/third_party/unitree_sim_isaaclab; walk back
    # to assemble/ before resolving the shared task-level package.
    Path(__file__).resolve().parents[3]
    / "task/augmentation/replay_augmentation_effects.py"
)
if not _SOURCE.is_file():
    raise FileNotFoundError(f"Replay augmentation package is missing: {_SOURCE}")
_SPEC = importlib.util.spec_from_file_location(
    "task_replay_augmentation_effects", _SOURCE
)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"Unable to load replay augmentation effects: {_SOURCE}")
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)

ReplayAugmentationEffects = _MODULE.ReplayAugmentationEffects
effects_requested = _MODULE.effects_requested

__all__ = ["ReplayAugmentationEffects", "effects_requested"]
