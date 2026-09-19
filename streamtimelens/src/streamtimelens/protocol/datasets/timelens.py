"""Normalize TimeLens-style annotations without rewriting their source JSON."""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass
from typing import Any, Iterable


def normalize_query(query: str) -> str:
    return re.sub(r"\s+", " ", query.strip()).lower()


def stable_query_id(video_id: str, query: str, span: tuple[float, float]) -> str:
    material = f"{video_id}\x1f{normalize_query(query)}\x1f{span[0]:.6f}\x1f{span[1]:.6f}"
    return hashlib.sha1(material.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class TimeLensQuery:
    query_id: str
    video_id: str
    query: str
    gt_span: tuple[float, float]
    duration: float
    dataset: str | None = None
    source: str | None = None
    split: str | None = None
    unit: str = "seconds"


def _first(row: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in row:
            return row[key]
    raise ValueError(f"annotation missing one of {keys}")


def adapt_timelens_annotations(
    rows: Iterable[dict[str, Any]], *, available_video_ids: set[str] | None = None
) -> list[TimeLensQuery]:
    """Accept common official field aliases and produce a stable, sorted view."""
    output: list[TimeLensQuery] = []
    seen: set[str] = set()
    for row in rows:
        video_id = str(_first(row, "video_id", "vid", "video"))
        if available_video_ids is not None and video_id not in available_video_ids:
            raise FileNotFoundError(f"annotation references unavailable video: {video_id}")
        query = str(_first(row, "query", "description", "sentence"))
        duration = float(_first(row, "duration", "video_duration"))
        raw_span = _first(row, "gt_span", "span", "timestamps")
        if isinstance(raw_span, list) and raw_span and isinstance(raw_span[0], (list, tuple)):
            spans = raw_span
        else:
            spans = [raw_span]
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError(f"invalid duration for {video_id}")
        for raw_start, raw_end in spans:
            start, end = float(raw_start), float(raw_end)
            if not all(math.isfinite(value) for value in (start, end)) or not 0 <= start < end <= duration:
                raise ValueError(f"invalid span for {video_id}: {(start, end)}")
            query_id = stable_query_id(video_id, query, (start, end))
            if query_id in seen:
                raise ValueError(f"duplicate query key: {query_id}")
            seen.add(query_id)
            output.append(TimeLensQuery(query_id, video_id, query, (start, end), duration,
                                        row.get("dataset"), row.get("source"), row.get("split"), row.get("unit", "seconds")))
    return sorted(output, key=lambda item: (item.video_id, item.query_id))


def group_by_video(queries: Iterable[TimeLensQuery]) -> dict[str, list[TimeLensQuery]]:
    grouped: dict[str, list[TimeLensQuery]] = {}
    for query in queries:
        grouped.setdefault(query.video_id, []).append(query)
    return {key: sorted(value, key=lambda item: item.query_id) for key, value in sorted(grouped.items())}
