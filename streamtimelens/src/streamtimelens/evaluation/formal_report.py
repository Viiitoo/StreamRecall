"""Validation and four-table rendering for frozen formal evaluation runs."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Iterable, Sequence


@dataclass(frozen=True, init=False)
class FormalRunSummary:
    method: str
    budget_bytes: int
    rho_q: float
    cohort: str
    count: int
    miou: float
    recall_at_03: float
    recall_at_05: float
    recall_at_07: float
    writer_coverage: float | None
    candidate_recall_at_5: float
    both_boundary_hit_rate: float
    refinement_delta_iou: float
    gpu_s: float
    realtime_throughput: float
    snapshot_bytes: float
    snapshot_integrity_passed: bool
    budget_audit_passed: bool
    frame_recall_at_1: float | None = None
    frame_recall_at_5: float | None = None
    frame_recall_at_8: float | None = None
    candidate_recall_at_1: float | None = None

    def __init__(self, *args, **kwargs) -> None:
        legacy_names = (
            "method", "budget_bytes", "rho_q", "cohort", "count", "miou",
            "recall_at_03", "recall_at_05", "recall_at_07", "writer_coverage",
            "candidate_recall_at_5", "both_boundary_hit_rate",
            "refinement_delta_iou", "gpu_s", "realtime_throughput",
            "snapshot_bytes", "snapshot_integrity_passed", "budget_audit_passed",
        )
        optional_names = (
            "writer_coverage", "frame_recall_at_1", "frame_recall_at_5",
            "frame_recall_at_8", "candidate_recall_at_1",
        )
        if args:
            if kwargs or len(args) != len(legacy_names):
                raise TypeError("formal summary positional input must use the legacy 18 fields")
            values = dict(zip(legacy_names, args))
        else:
            unknown = set(kwargs) - set(self.__dataclass_fields__)
            required = set(legacy_names) - {"writer_coverage"}
            missing = required - set(kwargs)
            if unknown or missing:
                detail = sorted(unknown or missing)
                raise TypeError(f"invalid formal summary fields: {', '.join(detail)}")
            values = dict(kwargs)
        for name in optional_names:
            values.setdefault(name, None)
        for name in self.__dataclass_fields__:
            object.__setattr__(self, name, values[name])
        self.__post_init__()

    def __post_init__(self) -> None:
        numeric = (
            self.rho_q, self.miou, self.recall_at_03, self.recall_at_05,
            self.recall_at_07, self.candidate_recall_at_5,
            self.both_boundary_hit_rate, self.refinement_delta_iou, self.gpu_s,
            self.realtime_throughput, self.snapshot_bytes,
        )
        optional = (
            self.writer_coverage, self.frame_recall_at_1, self.frame_recall_at_5,
            self.frame_recall_at_8, self.candidate_recall_at_1,
        )
        if not self.method or self.budget_bytes <= 0 or self.count < 0:
            raise ValueError("invalid formal run identity/count")
        if any(not math.isfinite(value) for value in numeric) or not 0 < self.rho_q <= 1:
            raise ValueError("invalid formal run metric")
        if any(value is not None and not math.isfinite(value) for value in optional):
            raise ValueError("invalid optional formal run metric")
        if self.cohort not in ("natural", "fixed_0.25"):
            raise ValueError("formal run cohort is unsupported")


def _dominates(left: dict, right: dict) -> bool:
    return (
        left["miou"] >= right["miou"] and left["gpu_s"] <= right["gpu_s"]
        and left["snapshot_bytes"] <= right["snapshot_bytes"]
        and (
            left["miou"] > right["miou"] or left["gpu_s"] < right["gpu_s"]
            or left["snapshot_bytes"] < right["snapshot_bytes"]
        )
    )


def build_formal_report(
    runs: Iterable[FormalRunSummary],
    *,
    expected_methods: Sequence[str],
    expected_rhos: tuple[float, ...] = (.25, .5, .75, 1.0),
) -> dict:
    rows = list(runs)
    if not rows or not expected_methods:
        raise ValueError("formal report needs runs and expected methods")
    if any(not row.snapshot_integrity_passed or not row.budget_audit_passed for row in rows):
        raise ValueError("formal query metrics cannot precede snapshot integrity and budget audit")
    keys = [(row.method, row.budget_bytes, row.rho_q, row.cohort) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("formal report contains duplicate method/budget/rho/cohort rows")
    budgets = sorted({row.budget_bytes for row in rows})
    expected = {
        (method, budget, rho, cohort)
        for method in expected_methods for budget in budgets for rho in expected_rhos
        for cohort in ("natural", "fixed_0.25")
    }
    missing = expected - set(keys)
    if missing:
        raise ValueError(f"formal report matrix is incomplete; missing {len(missing)} rows")
    accuracy = [{
        key: value for key, value in asdict(row).items()
        if key in {
            "method", "budget_bytes", "rho_q", "cohort", "count", "miou",
            "recall_at_03", "recall_at_05", "recall_at_07",
        }
    } for row in rows]
    diagnostics = [{
        key: value for key, value in asdict(row).items()
        if key in {
            "method", "budget_bytes", "rho_q", "cohort", "writer_coverage",
            "frame_recall_at_1", "frame_recall_at_5", "frame_recall_at_8",
            "candidate_recall_at_1", "candidate_recall_at_5",
            "both_boundary_hit_rate", "refinement_delta_iou",
        } and (key != "writer_coverage" or value is not None)
    } for row in rows]
    resources = [{
        key: value for key, value in asdict(row).items()
        if key in {
            "method", "budget_bytes", "rho_q", "cohort", "gpu_s",
            "realtime_throughput", "snapshot_bytes",
        }
    } for row in rows]
    aggregate = []
    for method in expected_methods:
        for budget in budgets:
            selected = [
                row for row in rows if row.method == method and row.budget_bytes == budget
                and row.cohort == "natural"
            ]
            aggregate.append({
                "method": method, "budget_bytes": budget,
                "miou": sum(row.miou for row in selected) / len(selected),
                "gpu_s": sum(row.gpu_s for row in selected),
                "snapshot_bytes": sum(row.snapshot_bytes for row in selected) / len(selected),
            })
    pareto = [
        row for row in aggregate
        if not any(_dominates(other, row) for other in aggregate if other != row)
    ]
    return {
        "accuracy_table": accuracy,
        "diagnostic_table": diagnostics,
        "resource_table": resources,
        "pareto_table": pareto,
    }


def formal_report_markdown(report: dict) -> str:
    lines = ["# Frozen formal evaluation", "", "## Accuracy", ""]
    lines.extend(
        f"- {row['method']} B={row['budget_bytes']} rho={row['rho_q']:.2f} "
        f"{row['cohort']}: mIoU={row['miou']:.4f}, n={row['count']}"
        for row in report["accuracy_table"]
    )
    lines.extend(["", "## Pareto configurations", ""])
    lines.extend(
        f"- {row['method']} B={row['budget_bytes']}: mIoU={row['miou']:.4f}, "
        f"GPU={row['gpu_s']:.3f}s, snapshot={row['snapshot_bytes']:.1f} bytes"
        for row in report["pareto_table"]
    )
    return "\n".join(lines) + "\n"
