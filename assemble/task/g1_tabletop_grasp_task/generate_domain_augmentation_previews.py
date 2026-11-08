#!/usr/bin/env python3
"""Generate first-person lighting and mine-dust augmentation previews.

This utility only writes preview MP4 files.  It never modifies the source
LeRobot Dataset v3.0 directory or its metadata.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import subprocess

import cv2
import numpy as np


PREVIEW_NAMES = (
    "00_original_clean",
    "01_lighting_low_illumination",
    "02_lighting_warm_worklamp",
    "03_lighting_cool_led",
    "04_lighting_headlamp_spot",
    "05_air_realistic_mine_dust",
    "06_air_mine_dust_gaussian",
)


class RawVideoEncoder:
    def __init__(self, output: Path, width: int, height: int, fps: float):
        output.parent.mkdir(parents=True, exist_ok=True)
        self.output = output
        self.log = output.with_suffix(".ffmpeg.log")
        self._log_file = self.log.open("wb")
        self._process = subprocess.Popen(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "bgr24",
                "-video_size",
                f"{width}x{height}",
                "-framerate",
                f"{fps:.8f}",
                "-i",
                "pipe:0",
                "-an",
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-crf",
                "20",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(output),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=self._log_file,
        )

    def write(self, frame: np.ndarray) -> None:
        if self._process.stdin is None or self._process.poll() is not None:
            raise RuntimeError(f"Encoder stopped early: {self.output}")
        self._process.stdin.write(np.ascontiguousarray(frame).tobytes())

    def finish(self) -> None:
        if self._process.stdin is not None:
            self._process.stdin.close()
        return_code = self._process.wait(timeout=120)
        self._log_file.close()
        if return_code != 0:
            details = self.log.read_text(encoding="utf-8", errors="replace")
            raise RuntimeError(f"ffmpeg failed for {self.output}: {details}")
        self.log.unlink(missing_ok=True)

    def abort(self) -> None:
        if self._process.stdin is not None and not self._process.stdin.closed:
            self._process.stdin.close()
        if self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait(timeout=3)
        if not self._log_file.closed:
            self._log_file.close()


class MineDustField:
    """Advected dust clouds plus fine, non-circular airborne particles."""

    def __init__(
        self,
        width: int,
        height: int,
        seed: int,
    ):
        self.width = width
        self.height = height
        rng = np.random.default_rng(seed)

        # Two spatial scales make the attenuation look like suspended dust
        # clouds instead of a uniform gray overlay.  Textures are advected
        # slowly and wrapped, giving temporal continuity for the whole video.
        coarse = rng.normal(0.0, 1.0, (height, width)).astype(np.float32)
        medium = rng.normal(0.0, 1.0, (height, width)).astype(np.float32)
        coarse = cv2.GaussianBlur(coarse, (0, 0), 42.0)
        medium = cv2.GaussianBlur(medium, (0, 0), 13.0)
        cloud = 0.68 * self._normalize(coarse) + 0.32 * self._normalize(medium)
        self.cloud = self._normalize(cloud)

        # Most airborne mine dust is visually sub-pixel or a short streak.
        # Use thin line segments with three parallax layers; deliberately avoid
        # filled circles, which read as bubbles after blur.
        self.count = 520
        self.x = rng.uniform(0.0, width, self.count)
        self.y = rng.uniform(0.0, height, self.count)
        self.depth = rng.uniform(0.0, 1.0, self.count)
        self.vx = 0.12 + 0.85 * self.depth + rng.normal(0.0, 0.08, self.count)
        self.vy = -0.10 - 0.50 * self.depth + rng.normal(0.0, 0.07, self.count)
        self.length = 0.8 + 7.2 * self.depth**2
        self.alpha = 0.075 + 0.235 * self.depth**1.5
        self.brightness = rng.uniform(62.0, 138.0, self.count)
        self.phase = rng.uniform(0.0, 2.0 * math.pi, self.count)

    @staticmethod
    def _normalize(values: np.ndarray) -> np.ndarray:
        lower, upper = np.percentile(values, (2.0, 98.0))
        return np.clip((values - lower) / max(upper - lower, 1.0e-6), 0.0, 1.0)

    def apply(self, frame: np.ndarray, frame_index: int) -> np.ndarray:
        # Advect the low-frequency cloud diagonally.  A small slow oscillation
        # changes density without causing frame-to-frame flicker.
        cloud = np.roll(
            self.cloud,
            shift=(int(-0.10 * frame_index), int(0.18 * frame_index)),
            axis=(0, 1),
        )
        breathing = 0.92 + 0.08 * math.sin(0.012 * frame_index)
        density = np.clip((0.065 + 0.165 * cloud) * breathing, 0.0, 0.24)
        output = frame.astype(np.float32)
        dust_color = np.asarray([116.0, 134.0, 154.0], dtype=np.float32)
        output = output * (1.0 - density[..., None]) + dust_color * density[..., None]

        # Dust scatters more light in bright regions and attenuates contrast.
        luminance = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        airlight = density * np.clip((luminance - 0.42) * 0.30, 0.0, 0.12)
        output = output * (1.0 - airlight[..., None]) + 210.0 * airlight[..., None]
        local_mean = cv2.GaussianBlur(output, (0, 0), 9.0)
        output = local_mean + (output - local_mean) * (1.0 - 0.55 * density[..., None])

        self.x = (self.x + self.vx) % self.width
        self.y = (self.y + self.vy) % self.height
        layer = np.zeros_like(output)
        mask = np.zeros((self.height, self.width), dtype=np.float32)
        shimmer = 0.86 + 0.14 * np.sin(self.phase + 0.025 * frame_index)
        for index in range(self.count):
            value = float(self.brightness[index] * shimmer[index])
            start = (int(self.x[index]), int(self.y[index]))
            end = (
                int(self.x[index] - self.length[index] * self.vx[index]),
                int(self.y[index] - self.length[index] * self.vy[index]),
            )
            thickness = 1 if self.depth[index] < 0.86 else 2
            # Mine dust is predominantly dark gray-brown.  A small warm bias
            # avoids the bright white/snow-like look of the earlier preview.
            color = (0.66 * value, 0.80 * value, 0.94 * value)
            particle_alpha = float(self.alpha[index])
            premultiplied_color = tuple(
                channel * particle_alpha for channel in color
            )
            cv2.line(
                layer,
                start,
                end,
                premultiplied_color,
                thickness,
                cv2.LINE_AA,
            )
            cv2.line(
                mask,
                start,
                end,
                particle_alpha,
                thickness,
                cv2.LINE_AA,
            )

        layer = cv2.GaussianBlur(layer, (3, 3), 0.45)
        mask = cv2.GaussianBlur(mask, (3, 3), 0.45)
        # Bright surfaces/headlamps produce stronger particle backscatter.
        scattering_gain = 0.82 + 0.42 * luminance
        mask *= scattering_gain
        layer *= scattering_gain[..., None]
        mask = np.clip(mask, 0.0, 0.30)[..., None]
        output = output * (1.0 - mask) + layer
        return np.clip(output, 0.0, 255.0).astype(np.uint8)


def radial_map(
    width: int,
    height: int,
    center_x: float,
    center_y: float,
    sigma_x: float,
    sigma_y: float,
) -> np.ndarray:
    y, x = np.mgrid[0:height, 0:width]
    dx = (x - center_x * width) / (sigma_x * width)
    dy = (y - center_y * height) / (sigma_y * height)
    return np.exp(-0.5 * (dx * dx + dy * dy)).astype(np.float32)


def apply_channel_scale(frame: np.ndarray, scales: tuple[float, float, float]) -> np.ndarray:
    output = frame.astype(np.float32)
    output *= np.asarray(scales, dtype=np.float32)[None, None, :]
    return np.clip(output, 0.0, 255.0).astype(np.uint8)


def add_gaussian_sensor_noise(frame: np.ndarray, frame_index: int) -> np.ndarray:
    """Add low-light camera noise without turning it into airborne particles."""
    source = frame.astype(np.float32)
    luminance = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    # Read noise is strongest in dark regions.  A shared luminance component
    # avoids the over-saturated RGB confetti produced by independent noise.
    sigma = 3.2 + 5.0 * (1.0 - luminance)
    rng = np.random.default_rng(481_516 + frame_index)
    luminance_noise = rng.normal(0.0, 1.0, luminance.shape).astype(np.float32)
    chroma_noise = rng.normal(0.0, 1.15, source.shape).astype(np.float32)
    noisy = source + luminance_noise[..., None] * sigma[..., None] + chroma_noise
    return np.clip(noisy, 0.0, 255.0).astype(np.uint8)


def add_label(frame: np.ndarray, label: str) -> np.ndarray:
    output = frame.copy()
    overlay = output.copy()
    cv2.rectangle(overlay, (8, 8), (455, 46), (0, 0, 0), thickness=-1)
    cv2.addWeighted(overlay, 0.58, output, 0.42, 0.0, dst=output)
    cv2.putText(
        output,
        label,
        (18, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.68,
        (245, 245, 245),
        2,
        cv2.LINE_AA,
    )
    return output


def make_variants(
    frame: np.ndarray,
    frame_index: int,
    fps: float,
    vignette: np.ndarray,
    headlamp: np.ndarray,
    dust_field: MineDustField,
) -> dict[str, np.ndarray]:
    normalized = frame.astype(np.float32) / 255.0

    low = np.power(normalized, 1.28) * 255.0 * 0.62
    noise_rng = np.random.default_rng(91_337 + frame_index)
    low += noise_rng.normal(0.0, 3.2, low.shape)
    low = np.clip(low, 0.0, 255.0).astype(np.uint8)

    warm = apply_channel_scale(frame, (0.72, 0.96, 1.17)).astype(np.float32)
    warm *= (0.83 + 0.17 * vignette[..., None])
    warm = np.clip(warm, 0.0, 255.0).astype(np.uint8)

    cool = apply_channel_scale(frame, (1.16, 1.04, 0.84)).astype(np.float32)
    cool = (cool - 127.5) * 1.08 + 127.5
    cool = np.clip(cool, 0.0, 255.0).astype(np.uint8)

    spot_gain = 0.30 + 1.22 * headlamp
    spot = frame.astype(np.float32) * spot_gain[..., None]
    spot *= np.asarray([0.92, 1.00, 1.06], dtype=np.float32)[None, None, :]
    spot = np.clip(spot, 0.0, 255.0).astype(np.uint8)

    dust = dust_field.apply(frame, frame_index)
    return {
        "00_original_clean": add_label(frame, "ORIGINAL / CLEAN AIR"),
        "01_lighting_low_illumination": add_label(low, "LIGHTING: LOW ILLUMINATION"),
        "02_lighting_warm_worklamp": add_label(warm, "LIGHTING: WARM WORK LAMP"),
        "03_lighting_cool_led": add_label(cool, "LIGHTING: COOL LED"),
        "04_lighting_headlamp_spot": add_label(spot, "LIGHTING: DIRECTIONAL HEADLAMP"),
        "05_air_realistic_mine_dust": add_label(
            dust,
            "AIR: REALISTIC MINE DUST",
        ),
        "06_air_mine_dust_gaussian": add_label(
            add_gaussian_sensor_noise(dust, frame_index),
            "AIR: MINE DUST + GAUSSIAN SENSOR NOISE",
        ),
    }


def create_comparison_grid(outputs: list[Path], destination: Path) -> None:
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y"]
    for output in outputs:
        command.extend(["-i", str(output)])
    scaled = "".join(
        f"[{index}:v]scale=320:240[v{index}];"
        for index in range(len(outputs))
    )
    layout = "|".join(
        f"{320 * (index % 3)}_{240 * (index // 3)}"
        for index in range(len(outputs))
    )
    filter_graph = (
        scaled
        + "".join(f"[v{index}]" for index in range(len(outputs)))
        + f"xstack=inputs={len(outputs)}:layout={layout}[grid]"
    )
    command.extend(
        [
            "-filter_complex",
            filter_graph,
            "-map",
            "[grid]",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "22",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(destination),
        ]
    )
    subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    source = args.source.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open source video: {source}")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    expected_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if width <= 0 or height <= 0 or fps <= 0.0:
        raise RuntimeError("Source video metadata is invalid")

    vignette = radial_map(width, height, 0.50, 0.50, 0.72, 0.72)
    headlamp = radial_map(width, height, 0.55, 0.47, 0.28, 0.34)
    dust_field = MineDustField(width, height, seed=7001)
    output_paths = [output_dir / f"{name}.mp4" for name in PREVIEW_NAMES]
    encoders = {
        name: RawVideoEncoder(path, width, height, fps)
        for name, path in zip(PREVIEW_NAMES, output_paths)
    }

    frame_index = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            variants = make_variants(
                frame,
                frame_index,
                fps,
                vignette,
                headlamp,
                dust_field,
            )
            for name in PREVIEW_NAMES:
                encoders[name].write(variants[name])
            frame_index += 1
            if frame_index % 100 == 0:
                print(f"Processed {frame_index}/{expected_frames} frames", flush=True)
        for encoder in encoders.values():
            encoder.finish()
    except Exception:
        for encoder in encoders.values():
            encoder.abort()
        raise
    finally:
        capture.release()

    comparison_path = output_dir / "07_all_variants_comparison_grid.mp4"
    create_comparison_grid(output_paths, comparison_path)
    manifest = {
        "source": str(source),
        "source_frames": frame_index,
        "fps": fps,
        "resolution": [width, height],
        "outputs": [str(path) for path in output_paths],
        "comparison_grid": str(comparison_path),
        "note": "Preview-only files; the source LeRobot dataset was not modified.",
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Generated {len(output_paths)} previews and comparison grid in {output_dir}")


if __name__ == "__main__":
    main()
