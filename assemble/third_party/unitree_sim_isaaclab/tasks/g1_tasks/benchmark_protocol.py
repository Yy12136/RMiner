"""Small file-based reset handshake shared by the G1 benchmark controllers."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path


class BenchmarkProtocol:
    def __init__(self) -> None:
        request = os.environ.get("G1_BENCHMARK_RESET_REQUEST_FILE")
        status = os.environ.get("G1_BENCHMARK_RESET_STATUS_FILE")
        active = os.environ.get("G1_BENCHMARK_ACTIVE_FILE")
        self.request_path = Path(request) if request else None
        self.status_path = Path(status) if status else None
        self.active_path = Path(active) if active else None
        self._last_reset_id: str | None = None
        self._pending: dict | None = None
        self._settle_steps = 0

    @property
    def active(self) -> bool:
        return bool(self.active_path and self.active_path.is_file())

    def consume_request(self) -> dict | None:
        if self.request_path is None or not self.request_path.is_file():
            return None
        try:
            request = json.loads(self.request_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        reset_id = request.get("reset_id")
        if not reset_id or reset_id == self._last_reset_id:
            return None
        self._last_reset_id = str(reset_id)
        return request

    def begin_settle(self, request: dict | None, details: dict, steps: int = 10) -> None:
        if request is None or self.status_path is None:
            return
        self._pending = {
            "reset_id": request["reset_id"], "episode_index": int(request["episode_index"]),
            "seed": int(request["seed"]), "settled": False, **details,
        }
        self._settle_steps = max(1, int(steps))
        self._write(self._pending)

    def tick(self) -> None:
        if self._pending is None:
            return
        self._settle_steps -= 1
        if self._settle_steps > 0:
            return
        self._pending["settled"] = True
        self._pending["settled_at"] = time.time()
        self._write(self._pending)
        self._pending = None

    def _write(self, payload: dict) -> None:
        assert self.status_path is not None
        self.status_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.status_path.with_name(f".{self.status_path.name}.{os.getpid()}.tmp")
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, self.status_path)
