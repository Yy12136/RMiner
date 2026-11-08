"""Training-matched evaluation perturbation tests without Isaac/Omni startup."""
from __future__ import annotations

import sys
import time
import unittest
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from perturbation import (  # noqa: E402
    LIGHTING_PROFILES,
    PERTURBED_VARIANTS,
    PERTURBATION_VARIANTS,
    EpisodePerturber,
    choose_variant,
)
from dust_fast import augmentation_module, make_fast_dust_field  # noqa: E402

POLICY_EVAL_SOURCE = (
    Path(__file__).resolve().parent.parent
    / "g1_tabletop_grasp_task"
    / "policy_eval.py"
)
EVALUATOR_SOURCE = Path(__file__).resolve().parent / "evaluator.py"


def synthetic_frame(seed: int = 7) -> np.ndarray:
    generator = np.random.default_rng(seed)
    return generator.integers(0, 256, size=(480, 640, 3), dtype=np.uint8)


@dataclass(frozen=True)
class Frame:
    bgr: np.ndarray
    timestamp_ms: int


def _frame_like(bgr: np.ndarray) -> Frame:
    return Frame(bgr=bgr, timestamp_ms=1234)


class PerturbationVariantTests(unittest.TestCase):
    def test_variant_catalog(self):
        self.assertEqual(len(PERTURBATION_VARIANTS), 9)
        self.assertEqual(PERTURBATION_VARIANTS[0], "none")
        self.assertEqual(len(PERTURBED_VARIANTS), 8)
        for profile in LIGHTING_PROFILES:
            self.assertIn(profile, PERTURBED_VARIANTS)
            self.assertIn(f"{profile}_dust", PERTURBED_VARIANTS)

    def test_choose_variant_deterministic_per_seed(self):
        for seed in range(50):
            self.assertEqual(choose_variant(seed), choose_variant(seed))
        draws = {choose_variant(seed) for seed in range(200)}
        self.assertTrue(draws.issubset(set(PERTURBATION_VARIANTS)))
        self.assertGreater(len(draws), 1)
        self.assertIn("none", draws)

    def test_choose_variant_salt_changes_draw(self):
        differs = sum(
            choose_variant(seed, salt=0) != choose_variant(seed, salt=1)
            for seed in range(50)
        )
        self.assertGreater(differs, 0)


class EpisodePerturberTests(unittest.TestCase):
    def test_none_variant_rejected(self):
        with self.assertRaises(ValueError):
            EpisodePerturber("none", episode_seed=42)

    def test_lighting_reduces_brightness_without_mutating_input(self):
        frame = synthetic_frame()
        original = frame.copy()
        perturber = EpisodePerturber("low_illumination", episode_seed=42)
        output = perturber.apply({"observation.images.head": _frame_like(frame)})
        processed = output["observation.images.head"].bgr
        self.assertEqual(processed.shape, frame.shape)
        self.assertEqual(processed.dtype, frame.dtype)
        self.assertLess(processed.mean(), frame.mean())
        np.testing.assert_array_equal(frame, original)

    def test_headlamp_spot_center_brighter_than_corner(self):
        perturber = EpisodePerturber("headlamp_spot", episode_seed=42)
        frame = np.full((480, 640, 3), 120, dtype=np.uint8)
        output = perturber.apply({"observation.images.head": _frame_like(frame)})
        processed = output["observation.images.head"].bgr.astype(np.int32)
        center = processed[200:280, 280:360].mean()
        corner = processed[0:60, 0:60].mean()
        self.assertGreater(center, corner)

    def test_dust_variant_is_deterministic_and_differs_from_clean(self):
        frame = synthetic_frame()
        first = EpisodePerturber("low_illumination_dust", episode_seed=43)
        second = EpisodePerturber("low_illumination_dust", episode_seed=43)
        clean = EpisodePerturber("low_illumination", episode_seed=43)
        left = first.apply({"observation.images.head": _frame_like(frame)})
        right = second.apply({"observation.images.head": _frame_like(frame)})
        plain = clean.apply({"observation.images.head": _frame_like(frame)})
        np.testing.assert_array_equal(
            left["observation.images.head"].bgr, right["observation.images.head"].bgr
        )
        self.assertFalse(
            np.array_equal(
                left["observation.images.head"].bgr,
                plain["observation.images.head"].bgr,
            )
        )

    def test_per_camera_processing_with_distinct_seeds(self):
        perturber = EpisodePerturber("headlamp_spot_dust", episode_seed=44)
        frames = {
            "observation.images.head": _frame_like(synthetic_frame(1)),
            "observation.images.left_wrist": _frame_like(synthetic_frame(2)),
            "observation.images.right_wrist": _frame_like(synthetic_frame(3)),
        }
        output = perturber.apply(frames)
        self.assertEqual(set(output), set(frames))
        for key, frame in frames.items():
            self.assertFalse(np.array_equal(output[key].bgr, frame.bgr))
            self.assertEqual(output[key].timestamp_ms, frame.timestamp_ms)

    def test_lighting_is_frame_independent(self):
        # Pure LUT profiles depend only on pixel values, not frame index.
        perturber = EpisodePerturber("warm_worklamp", episode_seed=45)
        first = perturber.apply({"observation.images.head": _frame_like(synthetic_frame())})
        second = perturber.apply({"observation.images.head": _frame_like(synthetic_frame())})
        np.testing.assert_array_equal(
            first["observation.images.head"].bgr,
            second["observation.images.head"].bgr,
        )
        third = perturber.apply({"observation.images.head": _frame_like(synthetic_frame(9))})
        self.assertFalse(
            np.array_equal(
                first["observation.images.head"].bgr,
                third["observation.images.head"].bgr,
            )
        )


class FastDustFieldTests(unittest.TestCase):
    """The fast renderer must match the training renderer and stay cheap."""

    def test_matches_training_renderer(self):
        module = augmentation_module()
        illumination = module._radial_map(640, 480, 0.55, 0.47, 0.28, 0.34)
        for illum in (None, illumination):
            original = module.VideoDustField(640, 480, 123, 5000, illum)
            fast = make_fast_dust_field(640, 480, 123, 5000, illum)
            worst_mean, worst_p99, worst_max = 0.0, 0.0, 0
            for index in range(30):
                frame = synthetic_frame(100 + index)
                a = original.apply(frame, index)
                b = fast.apply(frame, index)
                diff = np.abs(a.astype(np.int16) - b.astype(np.int16))
                worst_mean = max(worst_mean, float(diff.mean()))
                worst_p99 = max(worst_p99, float(np.percentile(diff, 99)))
                worst_max = max(worst_max, int(diff.max()))
            self.assertLessEqual(worst_mean, 0.1, f"illum={illum is not None}")
            self.assertLessEqual(worst_p99, 2.0, f"illum={illum is not None}")
            self.assertLessEqual(worst_max, 2, f"illum={illum is not None}")

    def test_fast_dust_frame_budget(self):
        fast = make_fast_dust_field(640, 480, 123, 5000, None)
        frames = [synthetic_frame(200 + i) for i in range(23)]
        for index in range(3):
            fast.apply(frames[index], index)
        start = time.perf_counter()
        for index in range(3, 23):
            fast.apply(frames[index], index)
        per_frame_ms = (time.perf_counter() - start) / 20 * 1000
        self.assertLessEqual(per_frame_ms, 5.0, f"measured {per_frame_ms:.2f} ms/frame")

    def test_three_camera_step_budget(self):
        perturber = EpisodePerturber("low_illumination_dust", episode_seed=46)
        frames = {
            "head": _frame_like(synthetic_frame(1)),
            "lw": _frame_like(synthetic_frame(2)),
            "rw": _frame_like(synthetic_frame(3)),
        }
        for index in range(3):
            perturber.apply(frames)
        start = time.perf_counter()
        for index in range(3, 23):
            perturber.apply(frames)
        per_step_ms = (time.perf_counter() - start) / 20 * 1000
        self.assertLessEqual(per_step_ms, 15.0, f"measured {per_step_ms:.2f} ms/step")


class EvalContractTests(unittest.TestCase):
    def test_policy_eval_wiring(self):
        text = POLICY_EVAL_SOURCE.read_text()
        for token in (
            '"--perturb"',
            '"--perturb-seed"',
            '"--perturb-particle-count"',
            '"--timeout-steps"',
            "choose_variant(episode_seed",
            "perturber.apply(frames)",
            '"perturbation": perturbation_name',
            "perturbation_breakdown",
            "matched to clean baseline measured cap",
        ):
            self.assertIn(token, text)

    def test_common_evaluator_wiring(self):
        text = EVALUATOR_SOURCE.read_text()
        for token in (
            '"--perturb"',
            '"--timeout-steps"',
            "perturber.apply(frames)",
            '"perturbation": perturbation_name',
        ):
            self.assertIn(token, text)


if __name__ == "__main__":
    unittest.main()
