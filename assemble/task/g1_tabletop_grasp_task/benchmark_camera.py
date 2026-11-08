#!/usr/bin/env python3
"""Read the three simulator-owned camera shared-memory streams safely."""

from __future__ import annotations

import ctypes
import time
from dataclasses import dataclass
from multiprocessing import resource_tracker, shared_memory

import cv2
import numpy as np


CAMERA_KEYS = {
    "observation.images.head": "head",
    "observation.images.left_wrist": "left",
    "observation.images.right_wrist": "right",
}
THIRD_PERSON_KEY = {"observation.images.third_person": "third"}


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


@dataclass(frozen=True)
class CameraFrame:
    bgr: np.ndarray
    timestamp_ms: int

    @property
    def age_seconds(self) -> float:
        return max(0.0, time.time() - self.timestamp_ms / 1000.0)


class CameraReader:
    """Non-owning reader; closing it never unlinks simulator SHM."""

    def __init__(self, include_third: bool = False) -> None:
        self._segments: dict[str, shared_memory.SharedMemory] = {}
        self._camera_keys = dict(CAMERA_KEYS)
        if include_third:
            self._camera_keys.update(THIRD_PERSON_KEY)

    def _open(self, short_name: str) -> shared_memory.SharedMemory | None:
        if short_name in self._segments:
            return self._segments[short_name]
        try:
            segment = shared_memory.SharedMemory(
                name=f"isaac_{short_name}_image_shm", create=False
            )
        except FileNotFoundError:
            return None
        try:
            resource_tracker.unregister(segment._name, "shared_memory")
        except Exception:
            pass
        self._segments[short_name] = segment
        return segment

    def read(self, short_name: str) -> CameraFrame | None:
        segment = self._open(short_name)
        if segment is None:
            return None
        header_size = ctypes.sizeof(ImageHeader)
        try:
            before = ImageHeader.from_buffer_copy(bytes(segment.buf[:header_size]))
            if (
                before.timestamp == 0
                or before.height == 0
                or before.width == 0
                or before.channels != 3
                or before.data_size == 0
                or header_size + before.data_size > segment.size
            ):
                return None
            payload = bytes(segment.buf[header_size : header_size + before.data_size])
            after = ImageHeader.from_buffer_copy(bytes(segment.buf[:header_size]))
            if before.timestamp != after.timestamp:
                return None
            if before.encoding == 1:
                bgr = cv2.imdecode(
                    np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR
                )
                if bgr is None:
                    return None
            else:
                expected = before.height * before.width * before.channels
                raw = np.frombuffer(payload, dtype=np.uint8)
                if raw.size != expected:
                    return None
                bgr = raw.reshape(before.height, before.width, before.channels).copy()
            return CameraFrame(bgr=bgr, timestamp_ms=int(before.timestamp))
        except (BufferError, ValueError):
            return None

    def read_all(
        self, attempts: int = 8, retry_interval: float = 0.002
    ) -> dict[str, CameraFrame] | None:
        """Read one coherent frame per camera, tolerating brief SHM write races."""

        for attempt in range(attempts):
            frames: dict[str, CameraFrame] = {}
            for feature_key, short_name in self._camera_keys.items():
                frame = self.read(short_name)
                if frame is None:
                    break
                frames[feature_key] = frame
            if len(frames) == len(self._camera_keys):
                return frames
            if attempt + 1 < attempts:
                time.sleep(retry_interval)
        return None

    def close(self) -> None:
        for segment in self._segments.values():
            try:
                segment.close()
            except Exception:
                pass
        self._segments.clear()

    def __enter__(self) -> "CameraReader":
        return self

    def __exit__(self, *_args) -> None:
        self.close()
