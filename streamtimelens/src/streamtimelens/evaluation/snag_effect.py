"""Accuracy, latency and effect gates for SnAG-adapt development runs."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Mapping, Sequence

from streamtimelens.evaluation.metrics import temporal_iou


def evaluate_ranked_predictions(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    if not rows:
        raise ValueError("SnAG evaluation requires predictions")
    aggregate: dict[str, list[float]] = defaultdict(list)
    start_errors = []
    end_errors = []
    by_rho: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        gt = tuple(map(float, row["gt_span"]))
        spans = [tuple(map(float, span[:2])) for span in row.get("spans", [])]
        top1 = temporal_iou(gt, spans[0] if spans else None)
        top5 = max((temporal_iou(gt, span) for span in spans[:5]), default=0.0)
        aggregate["iou_top1"].append(top1)
        aggregate["iou_top5"].append(top5)
        if spans:
            start_errors.append(spans[0][0] - gt[0])
            end_errors.append(spans[0][1] - gt[1])
        by_rho[f"{float(row['rho']):.2f}"].append(row)

    def summary(subset: Sequence[Mapping[str, object]]) -> dict[str, object]:
        top1_values = []
        top5_values = []
        for row in subset:
            gt = tuple(map(float, row["gt_span"]))
            spans = [tuple(map(float, span[:2])) for span in row.get("spans", [])]
            top1_values.append(temporal_iou(gt, spans[0] if spans else None))
            top5_values.append(max((temporal_iou(gt, span) for span in spans[:5]), default=0.0))
        scale = 100.0 / len(subset)
        result: dict[str, object] = {"count": len(subset), "miou": sum(top1_values) * scale}
        for rank, values in ((1, top1_values), (5, top5_values)):
            for threshold in (0.3, 0.5, 0.7):
                result[f"r{rank}_iou_{threshold:.1f}"] = sum(value >= threshold for value in values) * scale
        return result

    overall = summary(rows)
    overall.update({
        "start_abs_error_s": sum(map(abs, start_errors)) / len(start_errors) if start_errors else None,
        "end_abs_error_s": sum(map(abs, end_errors)) / len(end_errors) if end_errors else None,
        "start_signed_error_s": sum(start_errors) / len(start_errors) if start_errors else None,
        "end_signed_error_s": sum(end_errors) / len(end_errors) if end_errors else None,
    })
    return {"overall": overall, "by_rho": {key: summary(value) for key, value in sorted(by_rho.items())}}


def percentile(values: Sequence[float], q: float) -> float:
    if not values or not 0 <= q <= 1:
        raise ValueError("invalid percentile input")
    ordered = sorted(map(float, values))
    position = q * (len(ordered) - 1)
    low = int(math.floor(position))
    high = int(math.ceil(position))
    if low == high:
        return ordered[low]
    return ordered[low] * (high - position) + ordered[high] * (position - low)


def snag_effect_gate(
    metrics: Mapping[str, object],
    *,
    prediction_count: int,
    expected_count: int,
    protocol_audit_passed: bool,
    checkpoint_frozen: bool,
) -> dict[str, object]:
    overall = metrics.get("overall", {})
    finite_metrics = isinstance(overall, Mapping) and all(
        math.isfinite(float(overall[name]))
        for name in ("miou", "r1_iou_0.3", "r5_iou_0.3")
    )
    checks = {
        "complete_predictions": prediction_count == expected_count > 0,
        "finite_metrics": finite_metrics,
        "non_degenerate_reader": float(overall.get("r5_iou_0.3", 0.0)) > 0,
        "runtime_protocol_audit": protocol_audit_passed,
        "checkpoint_frozen": checkpoint_frozen,
    }
    return {
        "schema_version": 1,
        "gate": "SnAG-G3-independent-dev-effect",
        "passed": all(checks.values()),
        "checks": checks,
        "note": "The gate requires a working baseline, not superiority over Hybrid V3.",
    }
