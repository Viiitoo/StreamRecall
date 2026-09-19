"""Paired development metrics and gates for HEM-01."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Mapping, Sequence

import numpy as np

from streamtimelens.evaluation.bootstrap import (
    BootstrapConfig,
    PairedMetricObservation,
    paired_video_bootstrap,
)


METHOD = "HEM-01"
BASELINE = "frozen_visual_v2"
MAX_READOUT_P95_MS = 50.0


def _video_delta(rows: Sequence[Mapping[str, Any]], left: str, right: str) -> float:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[str(row["video_id"])].append(float(row[left]) - float(row[right]))
    if not grouped:
        raise ValueError("empty HEM-01 metric slice")
    return float(np.mean([np.mean(values) for values in grouped.values()]))


def _bootstrap(
    rows: Sequence[Mapping[str, Any]], left: str, right: str, *, seed: int, resamples: int,
) -> dict[str, Any]:
    observations = []
    for row in rows:
        common = {
            "budget_bytes": int(row["budget_bytes"]),
            "video_id": str(row["video_id"]),
            "sample_id": str(row["observation_id"]),
        }
        observations.extend((
            PairedMetricObservation(METHOD, value=float(row[left]), **common),
            PairedMetricObservation(BASELINE, value=float(row[right]), **common),
        ))
    return paired_video_bootstrap(
        observations, method_a=METHOD, method_b=BASELINE,
        config=BootstrapConfig(seed=seed, resamples=resamples),
    )


def summarize_hem_v2_development(
    rows: Sequence[Mapping[str, Any]], *, support_complete: bool,
    zero_seek: bool, snapshots_unchanged: bool, byte_audit_complete: bool,
    coverage_complete: bool, provenance_complete: bool,
    seed: int = 20260911, resamples: int = 2000,
) -> dict[str, Any]:
    rows = [dict(row) for row in rows]
    if not rows or len({row["observation_id"] for row in rows}) != len(rows):
        raise ValueError("HEM-01 summary rows are empty or duplicated")
    for row in rows:
        row["baseline_r07"] = float(row["baseline_iou"] >= 0.7)
        row["hem_r07"] = float(row["hem_iou"] >= 0.7)
        row["baseline_r05"] = float(row["baseline_iou"] >= 0.5)
        row["hem_oracle_r05"] = float(row["hem_candidate_oracle_iou"] >= 0.5)
    standard = {
        "observation_count": len(rows),
        "baseline_miou": float(np.mean([row["baseline_iou"] for row in rows])),
        "hem_miou": float(np.mean([row["hem_iou"] for row in rows])),
        "delta_miou": float(np.mean([row["hem_iou"] - row["baseline_iou"] for row in rows])),
        "baseline_r07": float(np.mean([row["baseline_r07"] for row in rows])),
        "hem_r07": float(np.mean([row["hem_r07"] for row in rows])),
        "delta_r07": float(np.mean([row["hem_r07"] - row["baseline_r07"] for row in rows])),
        "baseline_r05": float(np.mean([row["baseline_r05"] for row in rows])),
        "hem_candidate_oracle_miou_at5": float(np.mean([
            row["hem_candidate_oracle_iou"] for row in rows
        ])),
        "hem_candidate_recall_at5_iou05": float(np.mean([row["hem_oracle_r05"] for row in rows])),
    }
    miou = _bootstrap(rows, "hem_iou", "baseline_iou", seed=seed, resamples=resamples)
    r07 = _bootstrap(rows, "hem_r07", "baseline_r07", seed=seed + 1, resamples=resamples)
    rho = {
        f"{value:.2f}": _video_delta(
            [row for row in rows if abs(float(row["rho"]) - value) < 1e-8],
            "hem_iou", "baseline_iou",
        )
        for value in sorted({float(row["rho"]) for row in rows})
    }
    good = [row for row in rows if bool(row["good_baseline"])]
    good_delta = _video_delta(good, "hem_iou", "baseline_iou") if good else None
    latency_p95 = float(np.percentile([row["readout_latency_ms"] for row in rows], 95))
    state_bytes = [int(row["snapshot_state_bytes"]) for row in rows]
    event_counts = [int(row["event_count"]) for row in rows]
    checks = {
        "video_equal_miou_ci_low_positive": miou["ci_low"] > 0,
        "standard_r07_nonnegative": standard["delta_r07"] >= 0,
        "rho_slices_above_floor": all(value >= -0.02 for value in rho.values()),
        "good_baseline_above_floor": good_delta is not None and good_delta >= -0.01,
        "candidate_oracle_miou_no_worse_than_frozen_final": (
            standard["hem_candidate_oracle_miou_at5"] >= standard["baseline_miou"]
        ),
        "candidate_recall_no_worse_than_frozen_final": (
            standard["hem_candidate_recall_at5_iou05"] >= standard["baseline_r05"]
        ),
        "readout_latency_p95_at_most_50ms": latency_p95 <= MAX_READOUT_P95_MS,
        "support_complete": bool(support_complete),
        "zero_seek": bool(zero_seek),
        "snapshots_unchanged": bool(snapshots_unchanged),
        "byte_audit_complete": bool(byte_audit_complete),
        "coverage_complete": bool(coverage_complete),
        "provenance_complete": bool(provenance_complete),
    }
    return {
        "schema_version": 1,
        "stage": "hem01_v2_original_development",
        "passed": all(checks.values()),
        "decision": "freeze_hem01_r1" if all(checks.values()) else "hem01_r1_not_frozen",
        "gate_checks": checks,
        "standard_metrics": standard,
        "miou_bootstrap": miou,
        "r07_bootstrap": r07,
        "rho_deltas": rho,
        "good_baseline_delta": good_delta,
        "readout_latency_ms": {"p95": latency_p95, "limit": MAX_READOUT_P95_MS},
        "snapshot_state_bytes": {
            "mean": float(np.mean(state_bytes)), "max": max(state_bytes),
        },
        "event_count": {"mean": float(np.mean(event_counts)), "max": max(event_counts)},
    }
