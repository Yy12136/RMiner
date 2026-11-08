#!/usr/bin/env python3
"""Vuer/WebXR helper process for the shared W1 Quest bridge."""

from __future__ import annotations

import argparse
import asyncio
import ctypes
import os
import signal
import sys
import traceback
from multiprocessing import resource_tracker, shared_memory
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.append(str(SCRIPT_DIR / ".quest_deps"))

import numpy as np
from vuer import Vuer
from vuer.schemas import DefaultScene, ImageBackground, MotionControllers


class W1Vuer(Vuer):
    """Vuer server that advertises the LAN address selected by the task."""

    advertised_lan_ip = "127.0.0.1"

    @property
    def local_ip(self) -> str:
        return self.advertised_lan_ip


class QuestWebXRServer:
    def __init__(self, args: argparse.Namespace) -> None:
        self.height = args.height
        self.stereo_width = args.stereo_width
        self.eye_width = self.stereo_width // 2
        self.stream_fps = args.stream_fps
        self.image_shm = shared_memory.SharedMemory(name=args.image_shm)
        self.control_shm = shared_memory.SharedMemory(name=args.control_shm)
        # The Isaac process owns and unlinks both segments.  This helper is an
        # attaching subprocess, so unregister them from its independent Python
        # resource tracker to avoid premature unlinking and shutdown warnings.
        resource_tracker.unregister(self.image_shm._name, "shared_memory")
        resource_tracker.unregister(self.control_shm._name, "shared_memory")
        self.image = np.ndarray(
            (self.height, self.stereo_width, 3),
            dtype=np.uint8,
            buffer=self.image_shm.buf,
        )
        self.control = np.ndarray(
            (2, 24),
            dtype=np.float64,
            buffer=self.control_shm.buf,
        )
        self.app = W1Vuer(
            host="0.0.0.0",
            port=args.port,
            cert=str(args.cert),
            key=str(args.key),
            queries=dict(grid=False),
            queue_len=3,
        )
        self.app.advertised_lan_ip = args.lan_ip
        self.app.add_handler("CONTROLLER_MOVE")(self._on_controller_move)

    async def _on_controller_move(self, event, session, fps=60) -> None:
        del session, fps
        try:
            for row, side in enumerate(("left", "right")):
                if side not in event.value:
                    continue
                pose = event.value.get(side)
                if isinstance(pose, list) and len(pose) >= 16:
                    self.control[row, :16] = np.asarray(pose[:16], dtype=np.float64)
                state = event.value.get(f"{side}State", {})
                if state:
                    thumbstick = state.get("thumbstickValue", [0.0, 0.0])
                    self.control[row, 16:23] = np.asarray(
                        [
                            float(state.get("triggerValue", 1.0 if state.get("trigger") else 0.0)),
                            float(state.get("squeezeValue", 1.0 if state.get("squeeze") else 0.0)),
                            -float(thumbstick[1]),
                            -float(thumbstick[0]),
                            float(state.get("thumbstick", False)),
                            float(state.get("aButton", state.get("aButtonValue", False))),
                            float(state.get("bButton", state.get("bButtonValue", False))),
                        ],
                        dtype=np.float64,
                    )
                self.control[row, 23] += 1.0
        except Exception:
            traceback.print_exc()

    async def _main_image(self, session, fps=60) -> None:
        del fps
        session.set @ DefaultScene(frameloop="always", grid=False)
        session.upsert @ MotionControllers(
            stream=True,
            key="motion-controllers-both",
            left=True,
            right=True,
        )
        while True:
            image = self.image.copy()
            session.upsert(
                [
                    ImageBackground(
                        image[:, : self.eye_width],
                        aspect=16.0 / 9.0,
                        height=1,
                        distanceToCamera=1,
                        layers=1,
                        format="jpeg",
                        quality=70,
                        key="background-left",
                        interpolate=True,
                    ),
                    ImageBackground(
                        image[:, self.eye_width :],
                        aspect=16.0 / 9.0,
                        height=1,
                        distanceToCamera=1,
                        layers=2,
                        format="jpeg",
                        quality=70,
                        key="background-right",
                        interpolate=True,
                    ),
                ],
                to="bgChildren",
            )
            await asyncio.sleep(1.0 / self.stream_fps)

    def run(self) -> None:
        self.app.spawn(start=True)(self._main_image)


def terminate_with_parent(expected_parent_pid: int) -> None:
    """Ask Linux to terminate this helper if its Isaac parent disappears."""
    if not sys.platform.startswith("linux"):
        return
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(1, signal.SIGTERM) != 0:  # PR_SET_PDEATHSIG
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))
    # Close the race in which the parent exits immediately before prctl.
    if os.getppid() != expected_parent_pid:
        raise SystemExit("Isaac parent exited before Quest helper startup")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-shm", required=True)
    parser.add_argument("--control-shm", required=True)
    parser.add_argument("--height", type=int, required=True)
    parser.add_argument("--stereo-width", type=int, required=True)
    parser.add_argument("--cert", type=Path, required=True)
    parser.add_argument("--key", type=Path, required=True)
    parser.add_argument("--lan-ip", required=True)
    parser.add_argument("--port", type=int, default=8012)
    parser.add_argument("--stream-fps", type=int, default=20)
    parser.add_argument("--parent-pid", type=int, required=True)
    args = parser.parse_args()
    terminate_with_parent(args.parent_pid)
    server = QuestWebXRServer(args)
    try:
        server.run()
    finally:
        server.image_shm.close()
        server.control_shm.close()


if __name__ == "__main__":
    main()
