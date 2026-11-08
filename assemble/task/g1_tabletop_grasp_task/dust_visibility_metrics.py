#!/usr/bin/env python3
"""Measure full-scene dust coverage and operation-area visibility.

The measurements are intentionally conservative image proxies.  They are used
to calibrate a visual augmentation preset; they are not object-detection or
task-success metrics.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


GRID_COLUMNS = 8
GRID_ROWS = 6


def _background_tile(row: int, column: int) -> bool:
    """Exclude the central robot/table near-field from the 8x6 grid."""
    return row <= 1 or (row == 2 and (column <= 1 or column >= 6))


def measure(clean_path: Path, candidate_path: Path) -> dict:
    clean_capture = cv2.VideoCapture(str(clean_path))
    candidate_capture = cv2.VideoCapture(str(candidate_path))
    tile_ratios: dict[tuple[int, int], list[float]] = {}
    tile_differences: dict[tuple[int, int], list[float]] = {}
    tabletop_ratios: list[float] = []
    extreme_fractions: list[float] = []
    sampled = 0
    frame_index = 0
    try:
        while True:
            clean_ok, clean_frame = clean_capture.read()
            candidate_ok, candidate_frame = candidate_capture.read()
            if not clean_ok or not candidate_ok:
                break
            # Let temporal accumulation and Flow reach a visible steady state.
            if frame_index >= 60 and frame_index % 3 == 0:
                clean_gray = cv2.cvtColor(clean_frame, cv2.COLOR_BGR2GRAY)
                candidate_gray = cv2.cvtColor(
                    candidate_frame, cv2.COLOR_BGR2GRAY
                )
                height, width = clean_gray.shape
                for row in range(GRID_ROWS):
                    for column in range(GRID_COLUMNS):
                        if not _background_tile(row, column):
                            continue
                        y0 = row * height // GRID_ROWS
                        y1 = (row + 1) * height // GRID_ROWS
                        x0 = column * width // GRID_COLUMNS
                        x1 = (column + 1) * width // GRID_COLUMNS
                        clean_tile = clean_gray[y0:y1, x0:x1]
                        candidate_tile = candidate_gray[y0:y1, x0:x1]
                        key = (row, column)
                        clean_detail = cv2.Laplacian(
                            clean_tile, cv2.CV_32F
                        )
                        candidate_detail = cv2.Laplacian(
                            candidate_tile, cv2.CV_32F
                        )
                        tile_ratios.setdefault(key, []).append(
                            float(candidate_detail.std())
                            / max(float(clean_detail.std()), 1.0e-6)
                        )
                        tile_differences.setdefault(key, []).append(
                            float(
                                np.mean(
                                    np.abs(
                                        candidate_tile.astype(np.float32)
                                        - clean_tile.astype(np.float32)
                                    )
                                )
                            )
                        )
                y0, y1 = int(0.30 * height), int(0.90 * height)
                x0, x1 = int(0.12 * width), int(0.88 * width)
                clean_table = clean_gray[y0:y1, x0:x1]
                candidate_table = candidate_gray[y0:y1, x0:x1]
                clean_table_detail = cv2.Laplacian(
                    clean_table, cv2.CV_32F
                )
                candidate_table_detail = cv2.Laplacian(
                    candidate_table, cv2.CV_32F
                )
                tabletop_ratios.append(
                    float(candidate_table_detail.std())
                    / max(float(clean_table_detail.std()), 1.0e-6)
                )
                extreme_fractions.append(
                    float(
                        np.mean(
                            (candidate_gray <= 3) | (candidate_gray >= 252)
                        )
                    )
                )
                sampled += 1
            frame_index += 1
    finally:
        clean_capture.release()
        candidate_capture.release()
    if not sampled:
        raise RuntimeError("No comparable steady-state frames were decoded")

    tile_summary = []
    affected = 0
    for row, column in sorted(tile_ratios):
        contrast_ratio = float(np.median(tile_ratios[(row, column)]))
        mean_abs_difference = float(
            np.median(tile_differences[(row, column)])
        )
        is_affected = contrast_ratio < 0.97 or mean_abs_difference >= 3.0
        affected += int(is_affected)
        tile_summary.append(
            {
                "row": row,
                "column": column,
                "contrast_ratio": contrast_ratio,
                "mean_abs_difference": mean_abs_difference,
                "affected": is_affected,
            }
        )
    all_background_ratios = [
        value for values in tile_ratios.values() for value in values
    ]
    # Dust may introduce a slow cloud boundary into a formerly flat wall tile,
    # producing a ratio above 1 even though fine scene detail was attenuated.
    # Cap each tile at 1 so added dust edges cannot cancel losses elsewhere.
    capped_tile_ratios = [
        min(float(item["contrast_ratio"]), 1.0) for item in tile_summary
    ]
    return {
        "sampled_frames": sampled,
        "decoded_frames": frame_index,
        "background_tiles": len(tile_summary),
        "affected_background_tiles": affected,
        "global_coverage_fraction": affected / len(tile_summary),
        "background_contrast_ratio": float(np.mean(capped_tile_ratios)),
        "background_raw_median_contrast_ratio": float(
            np.median(all_background_ratios)
        ),
        "tabletop_contrast_ratio_mean": float(np.mean(tabletop_ratios)),
        "tabletop_contrast_ratio_minimum": float(np.min(tabletop_ratios)),
        "tabletop_below_80_percent_fraction": float(
            np.mean(np.asarray(tabletop_ratios) < 0.80)
        ),
        "black_or_white_pixel_fraction_mean": float(
            np.mean(extreme_fractions)
        ),
        "background_tile_details": tile_summary,
        "notes": (
            "Visibility values are image-contrast proxies. Background tiles "
            "exclude the central robot/table near-field."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = measure(args.clean, args.candidate)
    payload = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")


if __name__ == "__main__":
    main()
