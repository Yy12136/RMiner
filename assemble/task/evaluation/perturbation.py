"""Training-matched visual perturbations for evaluation episodes.

Opt-in perturbations that replicate the video-space augmentation used to
build the 250-episode training sets (pipeline video_only_v3_dense_gray_dust):
four photometric lighting profiles, each optionally combined with the
temporally coherent gray dust field. Lighting uses the shared augmentation
package directly; dust uses the fast renderer in dust_fast.py, which is
bit-compatible in construction and trajectory with the training
VideoDustField but roughly five times cheaper per frame so the control
frequency is preserved.

Nothing here runs unless the evaluator passes --perturb; importing this
module has no side effects and constructing an EpisodePerturber is the only
operation that loads the augmentation implementation.
"""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

# Base seed of the training augmentation runs (augmentation_manifest.json).
TRAINING_AUGMENTATION_SEED = 260823
TRAINING_PARTICLE_COUNT = 5000

LIGHTING_PROFILES = (
    "low_illumination",
    "warm_worklamp",
    "cool_led",
    "headlamp_spot",
)
PERTURBED_VARIANTS = tuple(
    f"{profile}{suffix}" for profile in LIGHTING_PROFILES for suffix in ("", "_dust")
)
PERTURBATION_VARIANTS = ("none",) + PERTURBED_VARIANTS

_AUGMENTATION_MODULE_PATH = (
    Path(__file__).resolve().parent.parent
    / "augmentation"
    / "augment_lerobot_v3_video.py"
)
_effects_module: Any = None


def _augmentation_effects() -> Any:
    """Lazily load the shared video-effect implementation (heavy imports)."""
    global _effects_module
    if _effects_module is None:
        spec = importlib.util.spec_from_file_location(
            "augment_lerobot_v3_video", _AUGMENTATION_MODULE_PATH
        )
        if spec is None or spec.loader is None:
            raise RuntimeError(
                f"Cannot load augmentation module: {_AUGMENTATION_MODULE_PATH}"
            )
        module = importlib.util.module_from_spec(spec)
        # Register before exec so dataclass definitions inside the module
        # can resolve their own module.
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        _effects_module = module
    return _effects_module


def choose_variant(episode_seed: int, salt: int = 0) -> str:
    """Deterministically pick a variant for one evaluation episode.

    The same episode seed always yields the same variant, so an
    infrastructure retry of one slot reproduces its perturbation.
    """
    generator = np.random.default_rng(
        [TRAINING_AUGMENTATION_SEED, int(salt), int(episode_seed)]
    )
    return PERTURBATION_VARIANTS[int(generator.integers(0, len(PERTURBATION_VARIANTS)))]


class EpisodePerturber:
    """Apply one training-matched perturbation to every camera frame.

    Cameras are processed with per-camera effect processors seeded exactly
    like the training augmentation: seed_base + episode_seed * 1009 +
    camera_index * 100_003, where camera_index follows first-seen order of
    the camera dict (readers construct it in the fixed CAMERA_KEYS order).
    Input frames are never mutated; processed copies are returned, so each
    freshly read frame is perturbed exactly once.
    """

    def __init__(
        self,
        variant: str,
        episode_seed: int,
        particle_count: int = TRAINING_PARTICLE_COUNT,
        seed_base: int = TRAINING_AUGMENTATION_SEED,
    ) -> None:
        if variant not in PERTURBED_VARIANTS:
            raise ValueError(f"Variant {variant!r} must be one of {PERTURBED_VARIANTS}")
        self.variant = variant
        self.episode_seed = int(episode_seed)
        self.profile = variant[: -len("_dust")] if variant.endswith("_dust") else variant
        self.dust = variant.endswith("_dust")
        self.particle_count = int(particle_count)
        self.seed_base = int(seed_base)
        self._processors: dict[str, Any] = {}
        self._frame_index = 0

    def _processor_for(self, key: str, shape: tuple) -> Any:
        module = _augmentation_effects()
        camera_index = len(self._processors)
        seed = self.seed_base + self.episode_seed * 1009 + camera_index * 100_003
        # LUT + headlamp gain always come from the shared implementation.
        processor = module.VideoEffectProcessor(
            self.profile, False, shape[1], shape[0], seed, self.particle_count
        )
        if self.dust:
            # Fast dust field seeded exactly like the training augmentation;
            # illumination map matches (headlamp beam for headlamp_spot only).
            from dust_fast import make_fast_dust_field

            processor.fast_dust = make_fast_dust_field(
                shape[1], shape[0], seed, self.particle_count, processor.headlamp
            )
        self._processors[key] = processor
        return processor

    def apply(self, frames: dict) -> dict:
        """Return a new frame dict with the perturbation applied per camera."""
        output = {}
        for key, frame in frames.items():
            processor = self._processors.get(key)
            if processor is None:
                processor = self._processor_for(key, frame.bgr.shape)
            bgr = processor.apply(frame.bgr, self._frame_index)
            if self.dust:
                bgr = processor.fast_dust.apply(bgr, self._frame_index)
            output[key] = replace(frame, bgr=bgr)
        self._frame_index += 1
        return output
