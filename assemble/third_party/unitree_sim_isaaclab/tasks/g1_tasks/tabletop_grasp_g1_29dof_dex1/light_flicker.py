"""Deterministic USD light-intensity flicker for visual replay only.

The controller changes only ``inputs:intensity`` on discovered UsdLux lights.
It does not touch light color, temperature, exposure, scene geometry, physics,
robot actions, or dust settings.  All timing is simulation-time based so a
fixed seed produces the same intensity curve on every replay.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import random

import omni.usd
from pxr import UsdGeom


_LIGHT_TYPES = {
    "RectLight",
    "SphereLight",
    "DiskLight",
    "CylinderLight",
    "DistantLight",
    "GeometryLight",
    "MeshLight",
}


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, str(default)))


def _smoothstep(value: float) -> float:
    value = max(0.0, min(1.0, value))
    return value * value * (3.0 - 2.0 * value)


_COORDINATED_ROLES = ("headlamp", "aux0", "aux1", "aux2")


def _coordinated_role(path: str) -> str | None:
    if path.endswith("/Task01Headlamp"):
        return "headlamp"
    for index in range(3):
        if path.endswith(f"/AuxLight{index}"):
            return f"aux{index}"
    return None


def _disabled_coordinated_roles() -> set[str]:
    """Read optional scene-light exclusions for a coordinated profile."""

    raw = os.environ.get("G1_LIGHT_FLICKER_DISABLED_ROLES", "")
    disabled = {item.strip().lower() for item in raw.split(",") if item.strip()}
    unknown = disabled - set(_COORDINATED_ROLES)
    if unknown:
        raise ValueError(
            "G1_LIGHT_FLICKER_DISABLED_ROLES contains unknown roles: "
            + ",".join(sorted(unknown))
        )
    return disabled


def _make_coordinated_events(seed: int, horizon: float = 3600.0) -> list[dict[str, float]]:
    """Build one deterministic hour of staggered power-fault events."""

    rng = random.Random(seed)
    events: list[dict[str, float]] = []
    start = 2.15
    while start < horizon:
        # A short but still readable power-fault cadence for teleoperation.
        # The full 18-second demo now contains six or seven separate events.
        next_interval = rng.uniform(2.20, 3.00)
        duration = rng.uniform(0.75, 1.05)
        aux1_delay = rng.uniform(0.18, 0.30)
        events.append(
            {
                "start": start,
                "duration": duration,
                "aux1_delay": aux1_delay,
                "aux2_start": start + rng.uniform(0.12, 0.38),
                "aux2_duration": rng.uniform(0.28, 0.50),
                "recovery_start": start + duration + aux1_delay,
                "recovery_duration": rng.uniform(0.30, 0.45),
                "recovery_peak": rng.uniform(1.08, 1.18),
            }
        )
        start += next_interval
    return events


def _fault_envelope(time_value: float, start: float, duration: float) -> float:
    """Smooth attack, hold and release envelope in [0, 1]."""

    local = time_value - start
    if local <= 0.0 or local >= duration:
        return 0.0
    attack = duration * 0.28
    hold_end = duration * 0.53
    if local < attack:
        return _smoothstep(local / attack)
    if local <= hold_end:
        return 1.0
    return 1.0 - _smoothstep((local - hold_end) / (duration - hold_end))


def _recovery_envelope(time_value: float, start: float, duration: float) -> float:
    local = time_value - start
    if local <= 0.0 or local >= duration:
        return 0.0
    return math.sin(math.pi * local / duration) ** 2


def _coordinated_multipliers(
    time_value: float, events: list[dict[str, float]]
) -> tuple[dict[str, float], float]:
    """Return per-light multipliers and a telemetry-only weighted mean."""

    if time_value < 2.0:
        values = {role: 1.0 for role in _COORDINATED_ROLES}
        return values, 1.0

    baselines = {
        "headlamp": max(
            0.92,
            min(
                1.05,
                1.0
                + 0.035 * math.sin(2.0 * math.pi * 0.22 * time_value)
                + 0.015 * math.sin(2.0 * math.pi * 0.61 * time_value + 0.4),
            ),
        ),
        "aux0": max(
            0.88,
            min(
                1.16,
                1.0
                + 0.125 * math.sin(2.0 * math.pi * 0.18 * time_value + 0.2)
                + 0.035 * math.sin(2.0 * math.pi * 0.47 * time_value),
            ),
        ),
        "aux1": max(
            0.86,
            min(
                1.10,
                1.0
                + 0.075 * math.sin(2.0 * math.pi * 0.20 * time_value + 1.4)
                + 0.025 * math.sin(2.0 * math.pi * 0.51 * time_value + 0.6),
            ),
        ),
        "aux2": max(
            0.85,
            min(
                1.16,
                1.0
                - 0.125 * math.sin(2.0 * math.pi * 0.18 * time_value + 0.2)
                + 0.035 * math.sin(2.0 * math.pi * 0.43 * time_value + 1.1),
            ),
        ),
    }
    values = dict(baselines)
    # Keep the fault visibly dim while preserving a usable view of the task.
    minimums = {"headlamp": 0.55, "aux0": 0.30, "aux1": 0.40, "aux2": 0.25}
    recovery = 0.0
    recovery_peak = 1.0
    for event in events:
        if event["start"] > time_value + 1.0:
            break
        if time_value > event["recovery_start"] + event["recovery_duration"]:
            continue
        delays = {"headlamp": 0.0, "aux0": 0.0, "aux1": event["aux1_delay"], "aux2": 0.10}
        for role in _COORDINATED_ROLES:
            envelope = _fault_envelope(
                time_value, event["start"] + delays[role], event["duration"]
            )
            if envelope > 0.0:
                values[role] = min(
                    values[role],
                    (1.0 - envelope) * baselines[role] + envelope * minimums[role],
                )
        outage = _fault_envelope(
            time_value, event["aux2_start"], event["aux2_duration"]
        )
        if outage > 0.0:
            values["aux2"] = min(
                values["aux2"], (1.0 - outage) * values["aux2"] + outage * 0.02
            )
        event_recovery = _recovery_envelope(
            time_value, event["recovery_start"], event["recovery_duration"]
        )
        if event_recovery > recovery:
            recovery = event_recovery
            recovery_peak = event["recovery_peak"]
    if recovery > 0.0:
        for role in _COORDINATED_ROLES:
            values[role] = max(
                values[role],
                (1.0 - recovery) * values[role] + recovery * recovery_peak,
            )
    bounds = {
        "headlamp": (0.55, 1.18),
        "aux0": (0.30, 1.18),
        "aux1": (0.40, 1.18),
        "aux2": (0.02, 1.18),
    }
    for role, (lower, upper) in bounds.items():
        values[role] = max(lower, min(upper, values[role]))
    weights = {"headlamp": 0.25, "aux0": 0.35, "aux1": 0.25, "aux2": 0.15}
    weighted = sum(values[role] * weights[role] for role in _COORDINATED_ROLES)
    return values, weighted


class LightFlicker:
    """Animate selected USD light intensities with smooth deterministic dips."""

    def __init__(self) -> None:
        self.mode = os.environ.get("G1_LIGHT_FLICKER", "off").strip().lower()
        if self.mode not in {"off", "subtle", "industrial", "coordinated"}:
            raise ValueError(
                "G1_LIGHT_FLICKER must be one of: off, subtle, industrial, "
                "coordinated; "
                f"got {self.mode!r}"
            )
        self.seed = _env_int("G1_LIGHT_FLICKER_SEED", 260830)
        self.elapsed = 0.0
        self._rng = random.Random(self.seed)
        self._previous: dict[str, float] = {}
        self._lights: list[tuple[str, object, float, float]] = []
        self._cameras: list[tuple[str, object, float]] = []
        self._camera_exposure_fallback = os.environ.get(
            "G1_LIGHT_FLICKER_CAMERA_EXPOSURE_FALLBACK", "0"
        ) == "1"
        self._disabled_coordinated_roles = _disabled_coordinated_roles()
        self._coordinated_active_roles: set[str] = set()
        self._events = self._make_events()
        self._curve: list[dict[str, object]] = []
        curve_path = os.environ.get("G1_LIGHT_FLICKER_CURVE_PATH")
        self._curve_path = Path(curve_path) if curve_path else None

        if self.mode == "off":
            print("[light-flicker] Disabled (G1_LIGHT_FLICKER=off).")
            return
        self._discover_lights()
        if self.mode == "coordinated":
            roles = {_coordinated_role(path) for path, *_ in self._lights}
            expected_roles = set(_COORDINATED_ROLES) - self._disabled_coordinated_roles
            if roles != expected_roles:
                raise RuntimeError(
                    "[light-flicker] coordinated mode light roles do not match; "
                    f"expected={sorted(expected_roles)}, "
                    f"discovered={sorted(str(role) for role in roles)}"
                )
            self._coordinated_active_roles = set(expected_roles)
        if self._camera_exposure_fallback:
            self._discover_cameras()
        if not self._lights:
            raise RuntimeError(
                "[light-flicker] No UsdLux lights with inputs:intensity were found"
            )
        print(
            "[light-flicker] Enabled "
            f"mode={self.mode}, seed={self.seed}, lights="
            f"{','.join(path for path, *_ in self._lights)}, "
            f"events={len(self._events)}"
        )

    def _make_events(self) -> list[dict[str, float]]:
        if self.mode == "coordinated":
            return _make_coordinated_events(self.seed)
        if self.mode != "industrial":
            return []
        # The first event is early enough to be visible in a 15-second demo;
        # subsequent events are seed-dependent but always leave recovery time.
        events: list[dict[str, float]] = []
        start = 2.2
        while start < 57.0:
            duration = self._rng.uniform(0.12, 0.18)
            depth = self._rng.uniform(0.45, 0.58)
            events.append({"start": start, "duration": duration, "depth": depth})
            start += self._rng.uniform(2.7, 3.5)
        return events

    @property
    def is_enabled(self) -> bool:
        return self.mode != "off" and bool(self._lights)

    @staticmethod
    def _has_intensity(prim) -> bool:
        return bool(prim and prim.GetAttribute("inputs:intensity"))

    def _discover_lights(self) -> None:
        stage = omni.usd.get_context().get_stage()
        explicit = os.environ.get("G1_LIGHT_FLICKER_LIGHT_PATHS", "").strip()
        if explicit:
            paths = [item.strip() for item in explicit.split(",") if item.strip()]
            prims = []
            for path in paths:
                prim = stage.GetPrimAtPath(path)
                if not self._has_intensity(prim):
                    raise RuntimeError(
                        f"[light-flicker] Invalid light path or intensity attr: {path}"
                    )
                prims.append(prim)
        else:
            prims = [
                prim
                for prim in stage.Traverse()
                if prim.GetTypeName() in _LIGHT_TYPES
                and self._has_intensity(prim)
            ]
            if not prims:
                prims = [
                    prim
                    for prim in stage.Traverse()
                    if prim.GetTypeName() == "DomeLight"
                    and self._has_intensity(prim)
                ]

        for index, prim in enumerate(prims):
            path = str(prim.GetPath())
            role = _coordinated_role(path)
            if self.mode == "coordinated" and role in self._disabled_coordinated_roles:
                continue
            attr = prim.GetAttribute("inputs:intensity")
            base = float(attr.Get() or 0.0)
            if base <= 0.0:
                continue
            # Stable path-derived phase avoids Python's process-randomized hash.
            digest = hashlib.sha256(path.encode("utf-8")).digest()
            phase = int.from_bytes(digest[:4], "big") / 2**32 * 2.0 * math.pi
            self._previous[path] = base
            self._lights.append((path, attr, base, phase + index * 0.17))

    def _discover_cameras(self) -> None:
        stage = omni.usd.get_context().get_stage()
        for prim in stage.Traverse():
            if prim.GetTypeName() != "Camera":
                continue
            camera = UsdGeom.Camera(prim)
            attr = camera.GetExposureAttr()
            if not attr or not attr.IsValid():
                continue
            value = attr.Get()
            base = float(value if value is not None else 0.0)
            self._cameras.append((str(prim.GetPath()), attr, base))
        if self._cameras:
            print(
                "[light-flicker] Camera exposure fallback enabled: "
                + ",".join(path for path, *_ in self._cameras)
            )

    def _global_multiplier(self, time_value: float) -> float:
        if self.mode == "subtle":
            slow = 1.0 + 0.06 * math.sin(2.0 * math.pi * 0.75 * time_value)
            slow += 0.03 * math.sin(2.0 * math.pi * 2.1 * time_value + 0.7)
            return max(0.75, min(1.20, slow))
        slow = 1.0 + 0.12 * math.sin(2.0 * math.pi * 0.90 * time_value)
        slow += 0.05 * math.sin(2.0 * math.pi * 2.25 * time_value + 0.7)
        dip = 1.0
        for event in self._events:
            normalized = (time_value - event["start"]) / event["duration"]
            if 0.0 < normalized < 1.0:
                envelope = math.sin(math.pi * normalized)
                dip = min(dip, 1.0 - event["depth"] * envelope)
        return max(0.40, min(1.25, slow * dip))

    def update(self, dt: float) -> None:
        if self.mode == "off" or not self._lights:
            return
        self.elapsed += max(0.0, min(float(dt), 0.1))
        coordinated = None
        if self.mode == "coordinated":
            coordinated, _global_multiplier = _coordinated_multipliers(
                self.elapsed, self._events
            )
            weights = {"headlamp": 0.25, "aux0": 0.35, "aux1": 0.25, "aux2": 0.15}
            active_weight = sum(weights[role] for role in self._coordinated_active_roles)
            global_multiplier = sum(
                coordinated[role] * weights[role]
                for role in self._coordinated_active_roles
            ) / active_weight
        else:
            global_multiplier = self._global_multiplier(self.elapsed)
        curve_lights = {}
        for index, (path, attr, base, phase) in enumerate(self._lights):
            if coordinated is not None:
                role = _coordinated_role(path)
                if role is None:
                    raise RuntimeError(f"Unexpected coordinated light path: {path}")
                multiplier = coordinated[role]
            elif self.mode == "subtle":
                multiplier = global_multiplier * (
                    1.0 + 0.025 * math.sin(2.0 * math.pi * 0.55 * self.elapsed + phase)
                )
            else:
                # Lights on the same notional circuit share the dip but have a
                # small slow phase offset, preventing a perfectly synthetic sync.
                multiplier = global_multiplier * (
                    1.0 + 0.035 * math.sin(2.0 * math.pi * 0.50 * self.elapsed + phase)
                )
            # Coordinated mode already applies role-specific safety bounds.
            # Keep the legacy 0.40 floor only for subtle/industrial so the
            # planned 0.30/0.08/0.15/0.02 fault levels reach the USD lights.
            if coordinated is None:
                multiplier = max(0.40, min(1.25, multiplier))
            intensity = base * multiplier
            attr.Set(float(intensity))
            curve_lights[path] = {
                "base_intensity": base,
                "multiplier": multiplier,
                "intensity": intensity,
            }
        if self._cameras and global_multiplier > 0.0:
            exposure_delta = math.log2(global_multiplier)
            for _path, attr, base in self._cameras:
                attr.Set(float(base + exposure_delta))
        if self._curve_path is not None:
            self._curve.append(
                {
                    "time_seconds": self.elapsed,
                    "global_multiplier": global_multiplier,
                    "lights": curve_lights,
                }
            )

    def close(self) -> None:
        for path, attr, base, _phase in self._lights:
            attr.Set(float(base))
        for _path, attr, base in self._cameras:
            attr.Set(float(base))
        if self._curve_path and self.mode != "off":
            self._curve_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "mode": self.mode,
                "seed": self.seed,
                "events": self._events,
                "lights": [path for path, *_ in self._lights],
                "samples": self._curve,
            }
            temporary = self._curve_path.with_suffix(self._curve_path.suffix + ".tmp")
            temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
            temporary.replace(self._curve_path)
        self._lights.clear()
        self._cameras.clear()


__all__ = ["LightFlicker"]
