"""Frozen long-video extension selection and memory-aging diagnostics."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Iterable


@dataclass(frozen=True)
class LongVideoConfig:
    config_id: str
    method: str
    frozen_sha256: str
    dev_score: float
    is_full: bool
    is_pareto: bool

    def __post_init__(self) -> None:
        if not self.config_id or not self.method or len(self.frozen_sha256) != 64:
            raise ValueError("long-video configs must come from a hashed freeze")
        if not math.isfinite(self.dev_score):
            raise ValueError("long-video dev score must be finite")


def select_long_video_configs(configs: Iterable[LongVideoConfig]) -> dict:
    rows = list(configs)
    if len({row.config_id for row in rows}) != len(rows):
        raise ValueError("long-video config IDs must be unique")
    full = sorted(
        (row for row in rows if row.is_full and row.is_pareto),
        key=lambda row: (-row.dev_score, row.config_id),
    )
    baselines = sorted(
        (row for row in rows if not row.is_full),
        key=lambda row: (-row.dev_score, row.config_id),
    )
    if len(full) < 2 or len(baselines) < 2:
        raise ValueError("long-video extension needs two Pareto full configs and two baselines")
    selected = (*full[:2], *baselines[:2])
    return {
        "selected": [asdict(row) for row in selected],
        "full_config_ids": [row.config_id for row in full[:2]],
        "baseline_config_ids": [row.config_id for row in baselines[:2]],
        "selection_source": "frozen_independent_dev_only",
    }


@dataclass(frozen=True)
class MemoryAgingObservation:
    config_id: str
    video_id: str
    video_duration_s: float
    writer_coverage: float
    candidate_recall_at_5: float
    earliest_coverage_anchor_retained: bool
    cohort: str = "natural"

    def __post_init__(self) -> None:
        rates = (self.writer_coverage, self.candidate_recall_at_5)
        if not self.config_id or not self.video_id or self.video_duration_s <= 0:
            raise ValueError("memory-aging observation identity/duration is invalid")
        if any(not math.isfinite(value) or not 0 <= value <= 1 for value in rates):
            raise ValueError("memory-aging rates must be finite values in [0,1]")
        if self.cohort not in ("natural", "fixed_0.25"):
            raise ValueError("memory-aging cohort is unsupported")


def _aggregate_aging(rows: list[MemoryAgingObservation]) -> dict:
    return {
        "count": len(rows),
        "video_count": len({row.video_id for row in rows}),
        "writer_coverage": (
            sum(row.writer_coverage for row in rows) / len(rows) if rows else 0.0
        ),
        "candidate_recall_at_5": (
            sum(row.candidate_recall_at_5 for row in rows) / len(rows) if rows else 0.0
        ),
        "earliest_anchor_retention_rate": (
            sum(row.earliest_coverage_anchor_retained for row in rows) / len(rows)
            if rows else 0.0
        ),
    }


def memory_aging_report(observations: Iterable[MemoryAgingObservation]) -> dict:
    rows = list(observations)
    bins = {
        "short:<60s": lambda value: value < 60,
        "medium:[60,300)s": lambda value: 60 <= value < 300,
        "long:>=300s": lambda value: value >= 300,
    }
    length_bins = {}
    for label, predicate in bins.items():
        selected = [row for row in rows if predicate(row.video_duration_s)]
        length_bins[label] = _aggregate_aging(selected)
    return {
        "length_bins": length_bins,
        "fixed_0.25": _aggregate_aging([row for row in rows if row.cohort == "fixed_0.25"]),
    }
