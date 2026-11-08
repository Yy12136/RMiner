"""Animated Blender/OpenVDB dust rendered natively by NVIDIA Flow.

The effect is opt-in through ``G1_DUST_MODE=blender_vdb``.  Blender authors a
loopable OpenVDB sequence; Isaac Sim reads each density grid through
``FlowEmitterNanoVdb`` and uses its own RTX/Flow ray marcher.  Camera RGB is
never composited or modified after rendering.
"""

from __future__ import annotations

import importlib.util
import json
import math
import os
from pathlib import Path
import time

from pxr import Gf, Sdf


try:
    from .flow_dust_3d import FlowDust3D
except ImportError:
    _FLOW_SOURCE = Path(__file__).with_name("flow_dust_3d.py")
    _FLOW_SPEC = importlib.util.spec_from_file_location(
        "task_replay_flow_dust_for_vdb", _FLOW_SOURCE
    )
    if _FLOW_SPEC is None or _FLOW_SPEC.loader is None:
        raise ImportError(f"Unable to load Flow dust implementation: {_FLOW_SOURCE}")
    _FLOW_MODULE = importlib.util.module_from_spec(_FLOW_SPEC)
    _FLOW_SPEC.loader.exec_module(_FLOW_MODULE)
    FlowDust3D = _FLOW_MODULE.FlowDust3D


def _env_float(name: str, default: float) -> float:
    value = float(os.environ.get(name, str(default)))
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


class BlenderVdbDust3D(FlowDust3D):
    """Play a pre-authored OpenVDB density sequence through native Flow."""

    def __init__(
        self,
        table_position: tuple[float, float, float],
        table_yaw: float = 0.0,
    ) -> None:
        raw_directory = os.environ.get("G1_DUST_VDB_DIR", "").strip()
        if not raw_directory:
            raise ValueError("G1_DUST_VDB_DIR is required for blender_vdb dust")
        self._vdb_directory = Path(raw_directory).expanduser().resolve()
        if not self._vdb_directory.is_dir():
            raise FileNotFoundError(
                f"OpenVDB sequence directory not found: {self._vdb_directory}"
            )
        manifest_path = self._vdb_directory / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("grid") != "density":
            raise ValueError("OpenVDB manifest must declare the 'density' grid")
        file_names = manifest.get("files")
        if not isinstance(file_names, list) or len(file_names) < 2:
            raise ValueError("OpenVDB manifest must contain at least two frames")
        self._vdb_frames = [self._vdb_directory / str(name) for name in file_names]
        missing = [str(path) for path in self._vdb_frames if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"OpenVDB sequence has missing frames: {missing[:3]}")
        self._vdb_fps = _env_float("G1_DUST_VDB_FPS", 6.0)
        if not 0.5 <= self._vdb_fps <= 30.0:
            raise ValueError("G1_DUST_VDB_FPS must be in [0.5, 30]")
        self._vdb_frame = -1
        self._vdb_started = time.monotonic()

        # Reuse the verified Flow renderer, neutral dust colormap and settings
        # restoration.  Its procedural emitters are disabled before the first
        # simulation/render step and replaced by the VDB emitter below.
        super().__init__(table_position, table_yaw)
        for enabled in self._emitter_enabled:
            enabled.Set(False)
        for prim in self.stage.Traverse():
            if not str(prim.GetPath()).startswith(self.ROOT_PATH):
                continue
            if prim.GetTypeName() not in {
                "FlowEmitterSphere", "FlowEmitterBox", "FlowEmitterTexture",
                "FlowEmitterMesh", "FlowEmitterPoint",
            }:
                continue
            enabled = prim.GetAttribute("enabled")
            if enabled:
                enabled.Set(False)
        self._emitter_enabled = []
        self._emitter_velocity = []
        self._base_velocities = []

        simulate = self.stage.GetPrimAtPath(f"{self.ROOT_PATH}/flowSimulate")
        for name, value in (
            ("densityCellSize", _env_float("G1_DUST_VDB_CELL_SIZE", 0.10)),
            ("forceDisableCoreSimulation", True),
            ("forceDisableEmitters", False),
            ("forceSimulate", True),
            ("interpolateTimeSteps", False),
        ):
            attribute = simulate.GetAttribute(name)
            if not attribute:
                raise RuntimeError(f"Flow simulation attribute is missing: {name}")
            attribute.Set(value)

        emitter = self._define(
            f"{self.ROOT_PATH}/flowEmitterNanoVdb", "FlowEmitterNanoVdb"
        )
        smoke_scale = _env_float("G1_DUST_VDB_SMOKE_SCALE", 1.0)
        temperature_scale = _env_float("G1_DUST_VDB_TEMPERATURE_SCALE", 1.35)
        if not 0.05 <= smoke_scale <= 8.0:
            raise ValueError("G1_DUST_VDB_SMOKE_SCALE must be in [0.05, 8]")
        if not 0.05 <= temperature_scale <= 8.0:
            raise ValueError("G1_DUST_VDB_TEMPERATURE_SCALE must be in [0.05, 8]")
        values = {
            "allocateActiveLeaves": (Sdf.ValueTypeNames.Bool, True),
            "allocationScale": (Sdf.ValueTypeNames.Float, 1.0),
            "applyPostPressure": (Sdf.ValueTypeNames.Bool, False),
            "coupleRateBurn": (Sdf.ValueTypeNames.Float, 0.0),
            "coupleRateDivergence": (Sdf.ValueTypeNames.Float, 0.0),
            "coupleRateFuel": (Sdf.ValueTypeNames.Float, 0.0),
            "coupleRateSmoke": (Sdf.ValueTypeNames.Float, 100000000.0),
            "coupleRateTemperature": (Sdf.ValueTypeNames.Float, 100000000.0),
            "coupleRateVelocity": (Sdf.ValueTypeNames.Float, 0.0),
            "enabled": (Sdf.ValueTypeNames.Bool, True),
            "enableStreaming": (Sdf.ValueTypeNames.Bool, False),
            "layer": (Sdf.ValueTypeNames.Int, self.layer),
            "level": (Sdf.ValueTypeNames.Int, 0),
            "levelCount": (Sdf.ValueTypeNames.Int, 1),
            "minSmoke": (Sdf.ValueTypeNames.Float, 0.001),
            "smoke": (Sdf.ValueTypeNames.Float, 0.0),
            "smokeScale": (Sdf.ValueTypeNames.Float, smoke_scale),
            "streamClearAtStart": (Sdf.ValueTypeNames.Bool, False),
            "streamOnce": (Sdf.ValueTypeNames.Bool, False),
            "temperature": (Sdf.ValueTypeNames.Float, 0.0),
            "temperatureScale": (
                Sdf.ValueTypeNames.Float, temperature_scale
            ),
            "updateCoarseDensity": (Sdf.ValueTypeNames.Bool, True),
        }
        for name, (type_name, value) in values.items():
            self._set_attr(emitter, name, type_name, value)
        self._smoke_asset = emitter.CreateAttribute(
            "nanoVdbSmokes:assetPath", Sdf.ValueTypeNames.Asset, custom=False
        )
        self._set_attr(
            emitter, "nanoVdbSmokes:gridName", Sdf.ValueTypeNames.Token, "density"
        )
        self._temperature_asset = emitter.CreateAttribute(
            "nanoVdbTemperatures:assetPath", Sdf.ValueTypeNames.Asset, custom=False
        )
        self._set_attr(
            emitter,
            "nanoVdbTemperatures:gridName",
            Sdf.ValueTypeNames.Token,
            "density",
        )

        # Keep only a faint participating-air layer.  The visible structure is
        # the VDB density itself, not a uniform screen-space fog overlay.
        self.settings.set_float(
            "/rtx/raytracing/inscattering/densityMult",
            _env_float("G1_DUST_VDB_AMBIENT_DENSITY", 0.055),
        )
        self.settings.set_float(
            "/rtx/fog/fogDistanceDensity",
            _env_float("G1_DUST_VDB_FOG_DENSITY", 0.012),
        )
        self.settings.set_float("/rtx/fog/fogHeightDensity", 0.004)
        self._set_vdb_frame(0)
        print(
            "[blender-vdb-dust] Native animated OpenVDB enabled: "
            f"frames={len(self._vdb_frames)}, fps={self._vdb_fps:.2f}, "
            f"directory={self._vdb_directory}",
            flush=True,
        )

    def _set_vdb_frame(self, index: int) -> None:
        index %= len(self._vdb_frames)
        if index == self._vdb_frame:
            return
        asset = Sdf.AssetPath(str(self._vdb_frames[index]))
        self._smoke_asset.Set(asset)
        self._temperature_asset.Set(asset)
        self._vdb_frame = index

    def update(self, dt: float) -> None:
        del dt
        elapsed = max(0.0, time.monotonic() - self._vdb_started)
        self._set_vdb_frame(int(elapsed * self._vdb_fps))


__all__ = ["BlenderVdbDust3D"]


# Task01 one-shot VDB burst opt-in (20260919)
if os.environ.get("G1_VDB_BURST", "0") == "1":
    _burst_path = Path(__file__).with_name("task01_vdb_burst_trigger.py")
    _burst_spec = importlib.util.spec_from_file_location("task01_vdb_burst_trigger", _burst_path)
    _burst_module = importlib.util.module_from_spec(_burst_spec)
    _burst_spec.loader.exec_module(_burst_module)
    BlenderVdbDust3D = _burst_module.install_triggered_playback(BlenderVdbDust3D)
