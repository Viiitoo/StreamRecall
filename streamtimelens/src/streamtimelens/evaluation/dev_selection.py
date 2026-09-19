"""Auditable diagnostic-first selection for the frozen development grid."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping


@dataclass(frozen=True)
class DevRun:
    config_id: str
    writer_coverage: float
    candidate_recall_at_5: float
    both_boundary_hit_rate: float
    refinement_delta_iou: float
    gpu_s: float
    realtime_throughput: float
    diagnostic_keep: bool = False

    def __post_init__(self) -> None:
        values = (
            self.writer_coverage, self.candidate_recall_at_5,
            self.both_boundary_hit_rate, self.refinement_delta_iou,
            self.gpu_s, self.realtime_throughput,
        )
        if not self.config_id or any(not math.isfinite(value) for value in values):
            raise ValueError("invalid dev-grid run")
        if min(values[:3]) < 0 or max(values[:3]) > 1 or self.gpu_s < 0 or self.realtime_throughput < 0:
            raise ValueError("dev-grid rates/resources are invalid")


def _dominates(left: DevRun, right: DevRun) -> bool:
    maximize_left = (
        left.writer_coverage, left.candidate_recall_at_5,
        left.both_boundary_hit_rate, left.refinement_delta_iou,
        left.realtime_throughput,
    )
    maximize_right = (
        right.writer_coverage, right.candidate_recall_at_5,
        right.both_boundary_hit_rate, right.refinement_delta_iou,
        right.realtime_throughput,
    )
    no_worse = all(a >= b for a, b in zip(maximize_left, maximize_right)) and left.gpu_s <= right.gpu_s
    strictly_better = any(a > b for a, b in zip(maximize_left, maximize_right)) or left.gpu_s < right.gpu_s
    return no_worse and strictly_better


def select_dev_configs(runs: Iterable[DevRun]) -> dict:
    rows = sorted(runs, key=lambda item: item.config_id)
    if not rows or len({row.config_id for row in rows}) != len(rows):
        raise ValueError("dev selection needs unique non-empty runs")
    pareto = [
        row for row in rows if not any(_dominates(other, row) for other in rows if other != row)
    ]
    selected = sorted({row.config_id for row in pareto} | {
        row.config_id for row in rows if row.diagnostic_keep
    })
    return {
        "selection_order": [
            "writer_coverage_and_candidate_recall",
            "boundary_hit_and_refinement_delta",
            "gpu_seconds_and_realtime",
            "pareto_or_diagnostic_keep",
        ],
        "stage_1_retrieval": [{
            "config_id": row.config_id, "writer_coverage": row.writer_coverage,
            "candidate_recall_at_5": row.candidate_recall_at_5,
        } for row in rows],
        "stage_2_localization": [{
            "config_id": row.config_id, "both_boundary_hit_rate": row.both_boundary_hit_rate,
            "refinement_delta_iou": row.refinement_delta_iou,
        } for row in rows],
        "stage_3_resources": [{
            "config_id": row.config_id, "gpu_s": row.gpu_s,
            "realtime_throughput": row.realtime_throughput,
        } for row in rows],
        "pareto_config_ids": [row.config_id for row in pareto],
        "selected_config_ids": selected,
        "runs": [asdict(row) for row in rows],
    }


@dataclass(frozen=True)
class VisualDevRun:
    config_id: str
    method: str
    budget_bytes: int
    frame_recall_at_1: float
    frame_recall_at_5: float
    frame_recall_at_8: float
    candidate_recall_at_1: float
    candidate_recall_at_5: float
    candidate_oracle_miou: float
    left_boundary_hit_rate: float
    right_boundary_hit_rate: float
    coarse_miou: float
    coarse_recall_at_1: float
    snapshot_bytes: float
    ingest_gpu_s: float
    query_gpu_s: float
    query_latency_s: float
    fixed_c025_by_rho: Mapping[str, float]
    paired_bootstrap_vs_uniform: Mapping[str, Any] | None = None
    diagnostic_keep: bool = False

    def __post_init__(self) -> None:
        rates = (
            self.frame_recall_at_1, self.frame_recall_at_5, self.frame_recall_at_8,
            self.candidate_recall_at_1, self.candidate_recall_at_5,
            self.candidate_oracle_miou, self.left_boundary_hit_rate,
            self.right_boundary_hit_rate, self.coarse_miou, self.coarse_recall_at_1,
        )
        resources = (
            self.snapshot_bytes, self.ingest_gpu_s, self.query_gpu_s,
            self.query_latency_s,
        )
        degradation = tuple(float(value) for value in self.fixed_c025_by_rho.values())
        if not self.config_id or self.method not in ("uniform_raw", "semantic_reservoir"):
            raise ValueError("invalid visual dev run identity")
        if self.budget_bytes <= 0 or any(not math.isfinite(value) for value in (*rates, *resources, *degradation)):
            raise ValueError("invalid visual dev metric")
        if min(rates, default=0) < 0 or max(rates, default=0) > 1 or min(resources) < 0:
            raise ValueError("visual dev rates/resources are invalid")


def _visual_dominates(left: VisualDevRun, right: VisualDevRun) -> bool:
    maximize_left = (
        left.frame_recall_at_5, left.frame_recall_at_8,
        left.candidate_recall_at_5, left.candidate_oracle_miou,
        left.coarse_miou, left.coarse_recall_at_1,
    )
    maximize_right = (
        right.frame_recall_at_5, right.frame_recall_at_8,
        right.candidate_recall_at_5, right.candidate_oracle_miou,
        right.coarse_miou, right.coarse_recall_at_1,
    )
    minimize_left = (
        left.snapshot_bytes, left.ingest_gpu_s + left.query_gpu_s,
        left.query_latency_s,
    )
    minimize_right = (
        right.snapshot_bytes, right.ingest_gpu_s + right.query_gpu_s,
        right.query_latency_s,
    )
    no_worse = (
        all(a >= b for a, b in zip(maximize_left, maximize_right))
        and all(a <= b for a, b in zip(minimize_left, minimize_right))
    )
    strictly_better = (
        any(a > b for a, b in zip(maximize_left, maximize_right))
        or any(a < b for a, b in zip(minimize_left, minimize_right))
    )
    return no_worse and strictly_better


def select_visual_dev_configs(runs: Iterable[VisualDevRun]) -> dict[str, Any]:
    """Apply the V2 retrieval-first gate and retain only Pareto configurations."""
    rows = sorted(runs, key=lambda item: item.config_id)
    if not rows or len({row.config_id for row in rows}) != len(rows):
        raise ValueError("visual dev selection needs unique non-empty runs")
    one_mib = [row for row in rows if row.budget_bytes == 1024 * 1024]
    candidate_gate = max((row.candidate_recall_at_5 for row in one_mib), default=0.0) >= 0.70
    semantic_wins = []
    for semantic in (row for row in rows if row.method == "semantic_reservoir"):
        evidence = semantic.paired_bootstrap_vs_uniform or {}
        comparisons = [
            value for value in evidence.values()
            if isinstance(value, Mapping) and "ci_low" in value
        ]
        if any(float(value["ci_low"]) > 0 for value in comparisons):
            semantic_wins.append(semantic.config_id)
    chosen_method = "semantic_reservoir" if semantic_wins else "uniform_raw"
    eligible = [row for row in rows if row.method == chosen_method]
    pareto = [
        row for row in eligible
        if not any(_visual_dominates(other, row) for other in eligible if other != row)
    ]
    selected = sorted(
        {row.config_id for row in pareto}
        | {row.config_id for row in rows if row.diagnostic_keep}
    )
    passed = bool(pareto)
    return {
        "schema_version": 1,
        "passed": passed,
        "refiner_feasible": candidate_gate,
        "candidate_recall_at_5_1m_gate": candidate_gate,
        "semantic_better_budget_points": sorted(semantic_wins),
        "selected_method": chosen_method,
        "selection_order": [
            "frame_recall_at_1_5_8", "candidate_recall_at_1_5",
            "candidate_oracle_miou_and_boundary_hits", "fixed_c025_rho_degradation",
            "snapshot_bytes_and_gpu_seconds_and_query_latency",
            "coarse_miou_and_recall_at_1", "pareto_or_diagnostic_keep",
        ],
        "pareto_config_ids": sorted(row.config_id for row in pareto),
        "selected_config_ids": selected,
        "runs": [asdict(row) for row in rows],
    }
