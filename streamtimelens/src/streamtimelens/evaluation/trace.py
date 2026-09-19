"""Monotonic, JSON-serializable ingest/query trace events."""

from __future__ import annotations

import json
import time
from collections.abc import Iterator
from typing import Any


class TraceLog:
    def __init__(self) -> None:
        self._records: list[dict[str, Any]] = []

    def record(self, *, kind: str, timestamp_s: float, reason: str, object_id: str | None = None, **details: Any) -> dict[str, Any]:
        row = {"sequence": len(self._records), "kind": kind, "t": timestamp_s, "reason": reason,
               "object_id": object_id, "wall_time_s": time.monotonic(), **details}
        self._records.append(row)
        return row

    def to_jsonl(self) -> str:
        return "".join(json.dumps(row, sort_keys=True, default=str) + "\n" for row in self._records)

    def __iter__(self) -> Iterator[dict[str, Any]]:
        return iter(self._records)

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, index):
        return self._records[index]
