"""Wall, CPU, and optional CUDA timing with exception-safe records."""

from __future__ import annotations

import time
from contextlib import AbstractContextManager
from dataclasses import asdict
from typing import Any

from .types import ResourceRecord


class ComponentTimer(AbstractContextManager):
    def __init__(self, component: str, *, torch_module: Any | None = None) -> None:
        self.component = component
        self.torch = torch_module
        self._wall_start = 0.0
        self._cpu_start = 0.0
        self._cuda_start = None
        self._cuda_end = None
        self.record: ResourceRecord | None = None

    def __enter__(self) -> "ComponentTimer":
        if self.torch is not None and self.torch.cuda.is_available():
            self.torch.cuda.synchronize()
            self._cuda_start = self.torch.cuda.Event(enable_timing=True)
            self._cuda_end = self.torch.cuda.Event(enable_timing=True)
            self._cuda_start.record()
        self._wall_start, self._cpu_start = time.perf_counter(), time.process_time()
        return self

    def __exit__(self, exc_type, exc, traceback) -> bool:
        wall_s, cpu_s = time.perf_counter() - self._wall_start, time.process_time() - self._cpu_start
        cuda_s = None
        if self._cuda_start is not None and self._cuda_end is not None:
            self._cuda_end.record()
            self.torch.cuda.synchronize()
            cuda_s = self._cuda_start.elapsed_time(self._cuda_end) / 1000.0
        self.record = ResourceRecord(self.component, wall_s, cpu_s, cuda_s, "error" if exc_type else "ok")
        return False

    def as_dict(self) -> dict[str, Any]:
        if self.record is None:
            raise RuntimeError("timer has not completed")
        return asdict(self.record)


def aggregate_resource_records(records: list[ResourceRecord]) -> dict[str, dict[str, float | int]]:
    """Aggregate completed calls while retaining failed-call counts."""
    totals: dict[str, dict[str, float | int]] = {}
    for record in records:
        item = totals.setdefault(record.component, {"calls": 0, "errors": 0, "wall_s": 0.0, "cpu_s": 0.0, "cuda_s": 0.0})
        item["calls"] += 1
        item["errors"] += int(record.status != "ok")
        item["wall_s"] += record.wall_s
        item["cpu_s"] += record.cpu_s
        item["cuda_s"] += record.cuda_s or 0.0
    return totals
