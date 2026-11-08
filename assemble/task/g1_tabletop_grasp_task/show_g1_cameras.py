#!/usr/bin/env python3
"""Display the G1 head and wrist camera shared-memory streams side by side."""

from __future__ import annotations

import ctypes
import os
import signal
import time
from multiprocessing import resource_tracker, shared_memory
from pathlib import Path

import cv2
import numpy as np


CAMERAS = (
    ("head", "Head Camera"),
    ("left", "Left Wrist Camera"),
    ("right", "Right Wrist Camera"),
)
FRAME_SIZE = (480, 360)
LABEL_HEIGHT = 36
MAX_FRAME_AGE_SECONDS = 2.0
WINDOW_NAME = "G1 Cameras | Head - Left Wrist - Right Wrist"
SUCCESS_MARKER_PATH = Path(
    os.environ.get(
        "G1_TASK_SUCCESS_FILE",
        f"/tmp/g1_tabletop_grasp_{os.getuid()}.success",
    )
)


class ImageHeader(ctypes.LittleEndianStructure):
    _fields_ = [
        ("timestamp", ctypes.c_uint64),
        ("height", ctypes.c_uint32),
        ("width", ctypes.c_uint32),
        ("channels", ctypes.c_uint32),
        ("image_name", ctypes.c_char * 16),
        ("data_size", ctypes.c_uint32),
        ("encoding", ctypes.c_uint32),
        ("quality", ctypes.c_uint32),
    ]


class CameraReader:
    def __init__(self) -> None:
        self._segments: dict[str, shared_memory.SharedMemory] = {}

    def _open(self, name: str) -> shared_memory.SharedMemory | None:
        if name in self._segments:
            return self._segments[name]
        try:
            segment = shared_memory.SharedMemory(
                name=f"isaac_{name}_image_shm", create=False
            )
        except FileNotFoundError:
            return None

        # This process is only a reader. Prevent Python's resource tracker from
        # unlinking the simulator-owned segment when this window is closed.
        try:
            resource_tracker.unregister(segment._name, "shared_memory")
        except Exception:
            pass
        self._segments[name] = segment
        return segment

    def read(self, name: str) -> np.ndarray | None:
        segment = self._open(name)
        if segment is None:
            return None

        header_size = ctypes.sizeof(ImageHeader)
        try:
            header_before = ImageHeader.from_buffer_copy(
                bytes(segment.buf[:header_size])
            )
            if (
                header_before.timestamp == 0
                or header_before.height == 0
                or header_before.width == 0
                or header_before.channels != 3
                or header_before.data_size == 0
                or header_size + header_before.data_size > segment.size
            ):
                return None
            if time.time() - header_before.timestamp / 1000.0 > MAX_FRAME_AGE_SECONDS:
                return None

            payload = bytes(
                segment.buf[
                    header_size : header_size + header_before.data_size
                ]
            )
            header_after = ImageHeader.from_buffer_copy(
                bytes(segment.buf[:header_size])
            )
            if header_before.timestamp != header_after.timestamp:
                return None

            if header_before.encoding == 1:
                return cv2.imdecode(
                    np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR
                )

            expected = (
                header_before.height
                * header_before.width
                * header_before.channels
            )
            raw = np.frombuffer(payload, dtype=np.uint8)
            if raw.size != expected:
                return None
            return raw.reshape(
                header_before.height,
                header_before.width,
                header_before.channels,
            ).copy()
        except (BufferError, ValueError):
            return None

    def close(self) -> None:
        for segment in self._segments.values():
            try:
                segment.close()
            except Exception:
                pass
        self._segments.clear()


def make_tile(frame: np.ndarray | None, title: str) -> np.ndarray:
    width, height = FRAME_SIZE
    if frame is None:
        image = np.full((height, width, 3), 28, dtype=np.uint8)
        cv2.putText(
            image,
            "Waiting for camera...",
            (105, height // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (180, 180, 180),
            2,
            cv2.LINE_AA,
        )
    else:
        image = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)

    label = np.full((LABEL_HEIGHT, width, 3), 18, dtype=np.uint8)
    cv2.putText(
        label,
        title,
        (12, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (245, 245, 245),
        2,
        cv2.LINE_AA,
    )
    return cv2.vconcat((label, image))


def main() -> None:
    running = True
    last_frames: dict[str, np.ndarray] = {}

    def request_stop(_signum, _frame) -> None:
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    reader = CameraReader()
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(
        WINDOW_NAME,
        FRAME_SIZE[0] * len(CAMERAS),
        FRAME_SIZE[1] + LABEL_HEIGHT,
    )
    cv2.moveWindow(WINDOW_NAME, 20, 40)
    print("[camera-viewer] Three-camera window opened; waiting for simulator frames.")

    try:
        while running:
            tiles = []
            for name, title in CAMERAS:
                frame = reader.read(name)
                if frame is not None:
                    last_frames[name] = frame
                tiles.append(make_tile(last_frames.get(name), title))
            canvas = cv2.hconcat(tiles)
            if SUCCESS_MARKER_PATH.is_file():
                overlay = canvas.copy()
                cv2.rectangle(
                    overlay,
                    (0, 0),
                    (canvas.shape[1], canvas.shape[0]),
                    (0, 0, 0),
                    -1,
                )
                cv2.addWeighted(overlay, 0.55, canvas, 0.45, 0.0, canvas)
                text = "SUCCESS"
                font = cv2.FONT_HERSHEY_DUPLEX
                scale = 3.0
                thickness = 6
                (text_width, text_height), _ = cv2.getTextSize(
                    text, font, scale, thickness
                )
                origin = (
                    (canvas.shape[1] - text_width) // 2,
                    (canvas.shape[0] + text_height) // 2,
                )
                cv2.putText(
                    canvas,
                    text,
                    origin,
                    font,
                    scale,
                    (60, 255, 80),
                    thickness,
                    cv2.LINE_AA,
                )
            cv2.imshow(WINDOW_NAME, canvas)
            key = cv2.waitKey(30) & 0xFF
            if key in (27, ord("q")):
                break
            if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        reader.close()
        cv2.destroyAllWindows()
        print("[camera-viewer] Three-camera window closed.")


if __name__ == "__main__":
    main()
