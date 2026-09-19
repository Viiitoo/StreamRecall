"""Ingest/query resource aggregation and multi-query amortization."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Literal


@dataclass(frozen=True)
class ResourceObservation:
    video_id: str
    phase: Literal["ingest", "query"]
    component: str
    wall_s: float
    cpu_s: float = 0.0
    cuda_s: float | None = None
    query_id: str | None = None
    cold_query: bool | None = None
    state_peak_bytes: int | None = None
    snapshot_bytes: int | None = None
    stream_duration_s: float | None = None
    max_backlog_s: float | None = None

    def __post_init__(self) -> None:
        if not self.video_id or not self.component or self.phase not in ("ingest", "query"):
            raise ValueError("invalid resource observation identity")
        values = [self.wall_s, self.cpu_s]
        values.extend(value for value in (
            self.cuda_s, self.stream_duration_s, self.max_backlog_s,
        ) if value is not None)
        if any(not math.isfinite(value) or value < 0 for value in values):
            raise ValueError("resource times must be finite and non-negative")
        if self.phase == "query" and not self.query_id:
            raise ValueError("query resource observations need a query ID")
        if any(value is not None and value < 0 for value in (self.state_peak_bytes, self.snapshot_bytes)):
            raise ValueError("resource byte counts cannot be negative")


def aggregate_resources(observations: Iterable[ResourceObservation]) -> dict:
    rows = list(observations)
    components: dict[str, dict[str, dict[str, float | int]]] = {"ingest": {}, "query": {}}
    for row in rows:
        bucket = components[row.phase].setdefault(row.component, {
            "calls": 0, "wall_s": 0.0, "cpu_s": 0.0, "cuda_s": 0.0,
        })
        bucket["calls"] += 1
        bucket["wall_s"] += row.wall_s
        bucket["cpu_s"] += row.cpu_s
        bucket["cuda_s"] += row.cuda_s or 0.0
    query_totals: dict[tuple[str, str], float] = {}
    query_cold: dict[tuple[str, str], bool] = {}
    ingest_totals: dict[str, float] = {}
    stream_durations: dict[str, float] = {}
    for row in rows:
        if row.phase == "query":
            key = (row.video_id, str(row.query_id))
            query_totals[key] = query_totals.get(key, 0.0) + row.wall_s
            query_cold[key] = query_cold.get(key, False) or bool(row.cold_query)
        else:
            ingest_totals[row.video_id] = ingest_totals.get(row.video_id, 0.0) + row.wall_s
            if row.stream_duration_s is not None:
                stream_durations[row.video_id] = max(
                    stream_durations.get(row.video_id, 0.0), row.stream_duration_s,
                )
    query_by_video: dict[str, list[tuple[str, float]]] = {}
    for (video_id, query_id), wall in query_totals.items():
        query_by_video.setdefault(video_id, []).append((query_id, wall))
    for values in query_by_video.values():
        values.sort(key=lambda item: item[0])

    def amortized(limit: int | None) -> dict[str, float | int]:
        per_video = []
        query_count = 0
        for video_id in sorted(set(ingest_totals) | set(query_by_video)):
            queries = query_by_video.get(video_id, [])
            selected = queries if limit is None else queries[:limit]
            if not selected:
                continue
            query_count += len(selected)
            per_video.append(
                (ingest_totals.get(video_id, 0.0) + sum(value for _, value in selected))
                / len(selected)
            )
        return {
            "video_count": len(per_video), "query_count": query_count,
            "wall_s_per_query": sum(per_video) / len(per_video) if per_video else 0.0,
        }

    cold_values = [wall for key, wall in query_totals.items() if query_cold.get(key, False)]
    warm_values = [wall for key, wall in query_totals.items() if not query_cold.get(key, False)]
    throughput = {
        video_id: duration / ingest_totals[video_id]
        for video_id, duration in stream_durations.items()
        if ingest_totals.get(video_id, 0) > 0
    }
    state_values = [row.state_peak_bytes for row in rows if row.state_peak_bytes is not None]
    snapshot_values = [row.snapshot_bytes for row in rows if row.snapshot_bytes is not None]
    backlog_values = [row.max_backlog_s for row in rows if row.max_backlog_s is not None]
    return {
        "components": components,
        "ingest_wall_s": sum(ingest_totals.values()),
        "query_wall_s": sum(query_totals.values()),
        "single_query_wall_s_mean": (
            sum(query_totals.values()) / len(query_totals) if query_totals else 0.0
        ),
        "cold_query_wall_s_mean": sum(cold_values) / len(cold_values) if cold_values else 0.0,
        "warm_query_wall_s_mean": sum(warm_values) / len(warm_values) if warm_values else 0.0,
        "amortized": {"1": amortized(1), "5": amortized(5), "all": amortized(None)},
        "state_peak_bytes": max(state_values, default=0),
        "snapshot_bytes_mean": sum(snapshot_values) / len(snapshot_values) if snapshot_values else 0.0,
        "snapshot_bytes_max": max(snapshot_values, default=0),
        "realtime_throughput_mean": sum(throughput.values()) / len(throughput) if throughput else 0.0,
        "max_backlog_s": max(backlog_values, default=0.0),
    }
