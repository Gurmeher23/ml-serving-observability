"""Rolling in-memory store of recent request features, consumed by the drift job."""

from __future__ import annotations

import threading
from collections import deque
from collections.abc import Iterable


class RollingCapture:
    def __init__(self, max_rows: int) -> None:
        self._rows: deque[dict[str, float]] = deque(maxlen=max_rows or None)
        self._enabled = max_rows > 0
        self._lock = threading.Lock()

    def add(self, rows: Iterable[dict[str, float]]) -> None:
        if not self._enabled:
            return
        with self._lock:
            self._rows.extend(rows)

    def recent(self, limit: int | None = None) -> list[dict[str, float]]:
        with self._lock:
            rows = list(self._rows)
        return rows if limit is None else rows[-limit:]

    def __len__(self) -> int:
        return len(self._rows)
