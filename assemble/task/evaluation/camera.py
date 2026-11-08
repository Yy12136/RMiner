"""Non-owning reader for the three Isaac camera shared-memory streams."""

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


class ImageHeader(ctypes.LittleEndianStructure):
    _fields_ = [
        ("timestamp", ctypes.c_uint64), ("height", ctypes.c_uint32),
        ("width", ctypes.c_uint32), ("channels", ctypes.c_uint32),
        ("image_name", ctypes.c_char * 16), ("data_size", ctypes.c_uint32),
        ("encoding", ctypes.c_uint32), ("quality", ctypes.c_uint32),
    ]


@dataclass(frozen=True)
class CameraFrame:
    bgr: np.ndarray
    timestamp_ms: int

    @property
    def age_seconds(self) -> float:
        return max(0.0, time.time() - self.timestamp_ms / 1000.0)


class CameraReader:
    def __init__(self) -> None:
        self._segments: dict[str, shared_memory.SharedMemory] = {}

    def _open(self, name: str):
        if name in self._segments:
            return self._segments[name]
        try:
            segment = shared_memory.SharedMemory(name=f"isaac_{name}_image_shm", create=False)
        except FileNotFoundError:
            return None
        try:
            resource_tracker.unregister(segment._name, "shared_memory")
        except Exception:
            pass
        self._segments[name] = segment
        return segment

    def read(self, name: str) -> CameraFrame | None:
        segment = self._open(name)
        if segment is None:
            return None
        size = ctypes.sizeof(ImageHeader)
        try:
            before = ImageHeader.from_buffer_copy(bytes(segment.buf[:size]))
            if (before.timestamp == 0 or before.height == 0 or before.width == 0
                    or before.channels != 3 or before.data_size == 0
                    or size + before.data_size > segment.size):
                return None
            payload = bytes(segment.buf[size:size + before.data_size])
            after = ImageHeader.from_buffer_copy(bytes(segment.buf[:size]))
            if before.timestamp != after.timestamp:
                return None
            if before.encoding == 1:
                image = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_COLOR)
            else:
                raw = np.frombuffer(payload, np.uint8)
                expected = before.height * before.width * before.channels
                image = raw.reshape(before.height, before.width, before.channels).copy() if raw.size == expected else None
            if image is None:
                return None
            return CameraFrame(image, int(before.timestamp))
        except (BufferError, ValueError):
            return None

    def read_all(self, attempts: int = 8) -> dict[str, CameraFrame] | None:
        for attempt in range(attempts):
            frames = {key: self.read(short) for key, short in CAMERA_KEYS.items()}
            if all(value is not None for value in frames.values()):
                return frames
            if attempt + 1 < attempts:
                time.sleep(0.002)
        return None

    def close(self) -> None:
        for segment in self._segments.values():
            try:
                segment.close()
            except Exception:
                pass
        self._segments.clear()
