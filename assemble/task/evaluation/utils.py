from __future__ import annotations

import json
import math
import os
import re
from datetime import datetime
from pathlib import Path

import numpy as np


def write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def checkpoint_step(checkpoint: Path) -> str:
    for part in reversed(checkpoint.parts):
        match = re.fullmatch(r"(?:checkpoint-)?(\d{3,})", part)
        if match:
            return match.group(1)
    return "unknown"


def model_label(checkpoint: Path) -> str:
    parts = checkpoint.resolve().parts
    step = checkpoint_step(checkpoint)
    for index, part in enumerate(parts):
        if part == step and index:
            return parts[index - 1]
    return checkpoint.parent.name if checkpoint.name == "pretrained_model" else checkpoint.name


def percentile(values: list[float], q: float) -> float | None:
    finite = np.asarray([value for value in values if np.isfinite(value)], dtype=np.float64)
    return None if finite.size == 0 else float(np.percentile(finite, q))


def wilson_interval(successes: int, total: int) -> list[float] | None:
    if total == 0:
        return None
    z, p = 1.959963984540054, successes / total
    denominator = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denominator
    radius = z * math.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total)) / denominator
    return [max(0.0, center - radius), min(1.0, center + radius)]


def summarize(records: list[dict], latencies: list[float]) -> dict:
    valid = [record for record in records if record.get("valid")]
    successes = [record for record in valid if record["status"] == "success"]
    failures = [record for record in valid if record["status"] == "failure"]
    times = [float(record["elapsed_seconds"]) for record in successes]
    p50, p95 = percentile(latencies, 50), percentile(latencies, 95)
    return {
        "success_rate": len(successes) / len(valid) if valid else None,
        "wilson_95_interval": wilson_interval(len(successes), len(valid)),
        "successful_episodes": len(successes),
        "failure_episodes": len(failures),
        "timeout_episodes": sum(record["status"] == "timeout" for record in valid),
        "valid_episodes": len(valid), "invalid_episodes": len(records) - len(valid),
        "mean_success_time_seconds": float(np.mean(times)) if times else None,
        "inference_latency_ms_p50": None if p50 is None else 1000 * p50,
        "inference_latency_ms_p95": None if p95 is None else 1000 * p95,
        "updated_at": datetime.now().astimezone().isoformat(),
    }
