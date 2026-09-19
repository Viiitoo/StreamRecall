"""Cohort and duration-bin metric tables with explicit empty groups."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

from streamtimelens.evaluation.metrics import VTGMetricExample, evaluate_vtg
from streamtimelens.protocol.arrival import LAG_BINS


@dataclass(frozen=True)
class GroupedVTGExample:
    metric: VTGMetricExample
    cohort: str
    eligible: bool
    lag_bin: str
    video_duration_s: float

    def __post_init__(self) -> None:
        if self.cohort not in ("natural", "fixed_0.25", "delta"):
            raise ValueError("unknown delayed-query cohort")
        if not math.isfinite(self.video_duration_s) or self.video_duration_s <= 0:
            raise ValueError("video duration must be positive")


def _duration_label(value: float, boundaries: tuple[float, float]) -> str:
    if value < boundaries[0]:
        return f"short:<{boundaries[0]:g}s"
    if value < boundaries[1]:
        return f"medium:[{boundaries[0]:g},{boundaries[1]:g})s"
    return f"long:>={boundaries[1]:g}s"


def evaluate_grouped_vtg(
    examples: Iterable[GroupedVTGExample],
    *,
    video_boundaries_s: tuple[float, float] = (60.0, 300.0),
    event_boundaries_s: tuple[float, float] = (5.0, 20.0),
) -> dict[str, dict[str, dict]]:
    if not 0 < video_boundaries_s[0] < video_boundaries_s[1] or (
        not 0 < event_boundaries_s[0] < event_boundaries_s[1]
    ):
        raise ValueError("group duration boundaries must be increasing and positive")
    rows = list(examples)
    eligible = [row for row in rows if row.eligible]
    lag_labels = [f"[{low:g},{high:g})" for low, high in LAG_BINS]
    video_labels = [
        _duration_label(0, video_boundaries_s),
        _duration_label(video_boundaries_s[0], video_boundaries_s),
        _duration_label(video_boundaries_s[1], video_boundaries_s),
    ]
    event_labels = [
        _duration_label(0, event_boundaries_s),
        _duration_label(event_boundaries_s[0], event_boundaries_s),
        _duration_label(event_boundaries_s[1], event_boundaries_s),
    ]

    def table(labels, selector):
        result = {}
        for label in labels:
            selected = [row.metric for row in eligible if selector(row) == label]
            # The same base query legitimately appears at several rho/cohort
            # arrivals. Give each joined evaluation row a local stable key.
            metrics = [
                VTGMetricExample(
                    f"{item.query_id}::{index}", item.video_id, item.gt_span,
                    item.predicted_span, item.status,
                )
                for index, item in enumerate(selected)
            ]
            result[label] = evaluate_vtg(metrics).to_dict()
        return result

    return {
        "cohort": table(
            ["natural", "fixed_0.25"], lambda row: row.cohort,
        ),
        "lag": table(lag_labels, lambda row: row.lag_bin),
        "video_length": table(
            video_labels, lambda row: _duration_label(row.video_duration_s, video_boundaries_s),
        ),
        "event_length": table(
            event_labels,
            lambda row: _duration_label(
                row.metric.gt_span[1] - row.metric.gt_span[0], event_boundaries_s,
            ),
        ),
    }
