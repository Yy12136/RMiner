#!/usr/bin/env python3
"""Reusable Quest 3S WebXR bridge for W1 task controllers.

The Vuer server runs in a separate Python process.  Image and controller data
are exchanged through shared memory so starting WebXR never re-imports the
Isaac Sim application in a multiprocessing child.
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
from multiprocessing import shared_memory
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
HELPER = SCRIPT_DIR / "w1_quest3s_server.py"
QUEST_DEPS = SCRIPT_DIR / ".quest_deps"
QUEST_RUNTIME = SCRIPT_DIR / ".quest_runtime"


def local_ip() -> str:
    """Return the IPv4 address used by the default LAN interface."""
    try:
        route = subprocess.run(
            ["ip", "route", "show", "default"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.split()
        interface = route[route.index("dev") + 1]
        addresses = subprocess.run(
            ["ip", "-4", "-o", "addr", "show", "dev", interface, "scope", "global"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.split()
        return addresses[addresses.index("inet") + 1].split("/", maxsplit=1)[0]
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            sock.connect(("8.8.8.8", 80))
            return sock.getsockname()[0]
        except OSError:
            return "127.0.0.1"
        finally:
            sock.close()


def ensure_certificate(ip_address: str) -> tuple[Path, Path]:
    """Create a reusable 30-day self-signed certificate for WebXR."""
    QUEST_RUNTIME.mkdir(parents=True, exist_ok=True)
    safe_ip = ip_address.replace(".", "-")
    cert_file = QUEST_RUNTIME / f"quest3s-{safe_ip}-cert.pem"
    key_file = QUEST_RUNTIME / f"quest3s-{safe_ip}-key.pem"
    if cert_file.exists() and key_file.exists():
        return cert_file, key_file
    subprocess.run(
        [
            "openssl",
            "req",
            "-x509",
            "-newkey",
            "rsa:2048",
            "-sha256",
            "-nodes",
            "-days",
            "30",
            "-keyout",
            str(key_file),
            "-out",
            str(cert_file),
            "-subj",
            "/CN=W1 Quest 3S Teleoperation",
            "-addext",
            f"subjectAltName=IP:{ip_address},IP:127.0.0.1,DNS:localhost",
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return cert_file, key_file


class QuestBridge:
    """Own the WebXR helper process and its shared image/input buffers."""

    _POSE_SLICE = slice(0, 16)
    _STATE_SLICE = slice(16, 23)
    _UPDATE_INDEX = 23

    def __init__(
        self,
        image_height: int = 360,
        eye_width: int = 640,
        port: int = 8012,
        stream_fps: int = 20,
    ) -> None:
        if not (QUEST_DEPS / "vuer").is_dir():
            raise RuntimeError(
                f"Quest dependencies are missing. Run {SCRIPT_DIR / 'setup_w1_quest3s.sh'} first."
            )
        if not HELPER.is_file():
            raise FileNotFoundError(f"Quest server helper is missing: {HELPER}")

        self.image_height = int(image_height)
        self.eye_width = int(eye_width)
        self.stereo_width = 2 * self.eye_width
        self.image_shm = shared_memory.SharedMemory(
            create=True,
            size=self.image_height * self.stereo_width * 3,
        )
        self.control_shm = shared_memory.SharedMemory(
            create=True,
            size=2 * 24 * np.dtype(np.float64).itemsize,
        )
        self.image = np.ndarray(
            (self.image_height, self.stereo_width, 3),
            dtype=np.uint8,
            buffer=self.image_shm.buf,
        )
        self.control = np.ndarray(
            (2, 24),
            dtype=np.float64,
            buffer=self.control_shm.buf,
        )
        self.image.fill(28)
        self.control.fill(0.0)

        ip_address = local_ip()
        cert_file, key_file = ensure_certificate(ip_address)
        self.process = subprocess.Popen(
            [
                sys.executable,
                str(HELPER),
                "--image-shm",
                self.image_shm.name,
                "--control-shm",
                self.control_shm.name,
                "--height",
                str(self.image_height),
                "--stereo-width",
                str(self.stereo_width),
                "--cert",
                str(cert_file),
                "--key",
                str(key_file),
                "--lan-ip",
                ip_address,
                "--port",
                str(int(port)),
                "--stream-fps",
                str(int(stream_fps)),
                "--parent-pid",
                str(os.getpid()),
            ],
        )
        self.certificate_url = f"https://{ip_address}:{int(port)}"
        self.quest_url = f"https://vuer.ai?ws=wss://{ip_address}:{int(port)}"

    @staticmethod
    def _state(values: np.ndarray) -> dict[str, float | bool]:
        trigger, squeeze, stick_x, stick_y, stick, a_button, b_button = values
        return {
            "trigger": float(trigger),
            "squeeze": float(squeeze),
            "thumbstick_x": float(stick_x),
            "thumbstick_y": float(stick_y),
            "thumbstick": bool(stick),
            "a_button": bool(a_button),
            "b_button": bool(b_button),
        }

    def pose(self, side: str) -> np.ndarray:
        row = 0 if side == "left" else 1
        return self.control[row, self._POSE_SLICE].copy().reshape(4, 4, order="F")

    def state(self, side: str) -> dict[str, float | bool]:
        row = 0 if side == "left" else 1
        return self._state(self.control[row, self._STATE_SLICE].copy())

    @property
    def update_counts(self) -> tuple[int, int]:
        return (
            int(self.control[0, self._UPDATE_INDEX]),
            int(self.control[1, self._UPDATE_INDEX]),
        )

    def write_mono_image(self, rgb: np.ndarray) -> None:
        image = np.asarray(rgb)
        if image.shape != (self.image_height, self.eye_width, 3):
            raise ValueError(
                f"Unexpected Quest image shape {image.shape}; expected "
                f"{(self.image_height, self.eye_width, 3)}"
            )
        if image.dtype != np.uint8:
            if np.issubdtype(image.dtype, np.floating) and image.size:
                if float(np.nanmax(image)) <= 1.0:
                    image = image * 255.0
            image = np.clip(image, 0, 255).astype(np.uint8)
        self.image[:, : self.eye_width] = image
        self.image[:, self.eye_width :] = image

    def close(self) -> None:
        if getattr(self, "process", None) is not None and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=2)
        for memory in (self.image_shm, self.control_shm):
            memory.close()
            try:
                memory.unlink()
            except FileNotFoundError:
                pass
