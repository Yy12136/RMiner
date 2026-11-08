"""Fast drop-in renderer for the training dust field.

The training augmentation renders dust with VideoDustField at roughly
32 ms/frame/camera on the evaluation machine, which starves the policy's
control frequency when three cameras are processed per control tick. This
module subclasses VideoDustField so construction (and therefore the RNG draw
sequence: particle positions, velocities, haze cloud) is bit-identical to the
training implementation, and rewrites apply() with the same mathematics but
a cheaper execution plan:

- cv2.blendLinear replaces per-pixel convex-composite expressions
  (measured 0.38 ms vs 4.05 ms for the numpy equivalent at 640x480);
- the sigma-7 local-mean blur runs at quarter resolution (sigma 1.75,
  measured deviation mean 0.39 / p99 1.7 of 255 levels);
- the full-resolution haze base is cached: its integer roll shifts change
  only every ~31 frames, so per-frame work on the haze map is a scalar
  multiply (the clip at 0.19 never binds because the base tops out at 0.18);
- the twelve np.add.at scatter groups collapse into two bincount calls.

Particle state evolution stays stateful (p += v each frame) exactly like the
original, so trajectories match bit-for-bit; only the local-mean
approximation and float summation order differ, which the equivalence gate
in test_perturbation.py bounds (mean <= 2/255, p99 <= 8/255).
"""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import cv2
import numpy as np

_AUGMENTATION_MODULE = None


def augmentation_module():
    """Lazily load the shared training augmentation implementation."""
    global _AUGMENTATION_MODULE
    if _AUGMENTATION_MODULE is None:
        path = (
            Path(__file__).resolve().parent.parent
            / "augmentation"
            / "augment_lerobot_v3_video.py"
        )
        spec = importlib.util.spec_from_file_location(
            "augment_lerobot_v3_video_dust_fast", path
        )
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Cannot load augmentation module: {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        _AUGMENTATION_MODULE = module
    return _AUGMENTATION_MODULE


def make_fast_dust_field(
    width: int, height: int, seed: int, particle_count: int, illumination
):
    """Construct a fast dust field bit-compatible with the training one."""
    return _fast_class()(width, height, seed, particle_count, illumination)


_fast_class_cache = None


def _fast_class():
    global _fast_class_cache
    if _fast_class_cache is None:
        base = augmentation_module().VideoDustField

        class FastVideoDustField(base):
            """Same state and semantics as VideoDustField, cheaper apply()."""

            def _setup_buffers(self) -> None:
                height, width = self.height, self.width
                if float(self.cloud.max()) * 0.135 + 0.045 > 0.19:
                    raise RuntimeError(
                        "Haze base exceeds the original clip bound; the "
                        "cached-haze shortcut would change semantics"
                    )
                self._gray3 = np.full((height, width, 3), 108.0, dtype=np.float32)
                self._white3 = np.full((height, width, 3), 152.0, dtype=np.float32)
                self._h0 = np.empty((height, width), dtype=np.float32)
                self._h = np.empty((height, width), dtype=np.float32)
                self._inv_h = np.empty((height, width), dtype=np.float32)
                self._w1 = np.empty((height, width), dtype=np.float32)
                self._w2 = np.empty((height, width), dtype=np.float32)
                self._buf_a = np.empty((height, width, 3), dtype=np.float32)
                self._buf_b = np.empty((height, width, 3), dtype=np.float32)
                self._small = np.empty(
                    (max(1, height // 4), max(1, width // 4), 3), dtype=np.float32
                )
                self._lm = np.empty((height, width, 3), dtype=np.float32)
                self._pg = np.empty((height, width), dtype=np.float32)
                self._pg3 = np.empty((height, width, 3), dtype=np.float32)
                self._haze_key = None

            def _rebuild_haze_base(self, shift_y: int, shift_x: int) -> None:
                cloud = np.roll(self.cloud, shift=(shift_y, shift_x), axis=(0, 1))
                resized = cv2.resize(
                    cloud, (self.width, self.height), interpolation=cv2.INTER_CUBIC
                )
                np.multiply(resized, 0.135, out=self._h0)
                self._h0 += 0.045
                self._haze_key = (shift_y, shift_x)

            def apply(self, frame: np.ndarray, frame_index: int) -> np.ndarray:
                if not hasattr(self, "_h0"):
                    self._setup_buffers()
                height, width = self.height, self.width

                shift_y = int(round(-0.018 * frame_index))
                shift_x = int(round(0.032 * frame_index))
                if (shift_y, shift_x) != self._haze_key:
                    self._rebuild_haze_base(shift_y, shift_x)
                breathing = 0.94 + 0.06 * math.sin(0.009 * frame_index)
                np.multiply(self._h0, breathing, out=self._h)
                np.subtract(1.0, self._h, out=self._inv_h)
                np.multiply(self._h, 0.42, out=self._w2)
                np.subtract(1.0, self._w2, out=self._w1)

                output = self._buf_a
                output[:] = frame
                # out1 = frame*(1-h) + gray*h, matching the original blend.
                cv2.blendLinear(
                    output, self._gray3, self._inv_h, self._h, dst=self._buf_b
                )
                # local mean at quarter resolution (sigma scales exactly).
                small = cv2.resize(
                    self._buf_b,
                    (self._small.shape[1], self._small.shape[0]),
                    interpolation=cv2.INTER_AREA,
                )
                small = cv2.GaussianBlur(small, (0, 0), 1.75)
                cv2.resize(
                    small,
                    (width, height),
                    interpolation=cv2.INTER_LINEAR,
                    dst=self._lm,
                )
                # out2 = out1*(1-0.42h) + lm*0.42h (the original detail restore).
                cv2.blendLinear(
                    self._buf_b, self._lm, self._w1, self._w2, dst=output
                )

                # --- particles: identical state update, fused scatter ---
                self.x = (self.x + self.vx) % width
                self.y = (self.y + self.vy) % height
                xi = np.clip(self.x.astype(np.int32), 0, width - 1).astype(np.intp)
                yi = np.clip(self.y.astype(np.int32), 0, height - 1).astype(np.intp)
                shimmer = 0.88 + 0.12 * np.sin(self.phase + 0.021 * frame_index)
                weights = self.alpha * shimmer

                near = self.size_class >= 0.80
                very_near = self.size_class >= 0.97
                idx_parts = [yi * width + xi]
                weight_parts = [weights]
                layer_parts = [weights * self.brightness]
                streak_w = 0.56 * weights
                xi2 = np.clip(xi - np.rint(1.8 * self.vx).astype(np.intp), 0, width - 1)
                yi2 = np.clip(yi - np.rint(1.8 * self.vy).astype(np.intp), 0, height - 1)
                idx_parts.append(yi2 * width + xi2)
                weight_parts.append(streak_w)
                layer_parts.append(streak_w * self.brightness)
                for dx, dy, scale in ((1, 0, 0.48), (0, 1, 0.42)):
                    xn = np.clip(xi[near] + dx, 0, width - 1)
                    yn = np.clip(yi[near] + dy, 0, height - 1)
                    wn = scale * weights[near]
                    idx_parts.append(yn * width + xn)
                    weight_parts.append(wn)
                    layer_parts.append(wn * self.brightness[near])
                for dx, dy, scale in ((-1, 0, 0.34), (0, -1, 0.30)):
                    xn = np.clip(xi[very_near] + dx, 0, width - 1)
                    yn = np.clip(yi[very_near] + dy, 0, height - 1)
                    wn = scale * weights[very_near]
                    idx_parts.append(yn * width + xn)
                    weight_parts.append(wn)
                    layer_parts.append(wn * self.brightness[very_near])
                flat = np.concatenate(idx_parts)
                pm = np.bincount(
                    flat, weights=np.concatenate(weight_parts), minlength=height * width
                ).reshape(height, width)
                pl = np.bincount(
                    flat,
                    weights=np.concatenate(layer_parts),
                    minlength=height * width,
                ).reshape(height, width)
                pm = pm.astype(np.float32)
                pl = pl.astype(np.float32)

                pm = cv2.GaussianBlur(pm, (3, 3), 0.22)
                pl = cv2.GaussianBlur(pl, (3, 3), 0.22)
                if self.illumination is not None:
                    gain = 0.72 + 0.95 * self.illumination
                    pm *= gain
                    pl *= gain
                np.clip(pm, 0.0, 0.72, out=pm)
                np.maximum(pm, 1.0e-6, out=self._pg)
                np.divide(pl, self._pg, out=self._pg)
                np.clip(self._pg, 105.0, 225.0, out=self._pg)
                cv2.merge((self._pg, self._pg, self._pg), dst=self._pg3)
                np.subtract(1.0, pm, out=self._w1)
                cv2.blendLinear(output, self._pg3, self._w1, pm, dst=self._buf_b)

                if self.illumination is not None:
                    np.multiply(pm, self.illumination, out=self._w2)
                    self._w2 *= 0.18
                    np.subtract(1.0, self._w2, out=self._w1)
                    cv2.blendLinear(
                        self._buf_b, self._white3, self._w1, self._w2, dst=output
                    )
                    return np.clip(output, 0.0, 255.0).astype(np.uint8)
                return np.clip(self._buf_b, 0.0, 255.0).astype(np.uint8)

        _fast_class_cache = FastVideoDustField
    return _fast_class_cache
