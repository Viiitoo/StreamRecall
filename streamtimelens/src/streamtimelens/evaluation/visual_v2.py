"""Independent-dev evaluation for bounded candidate-window expansion."""

from __future__ import annotations

from typing import Any, Iterable, Mapping

from streamtimelens.evaluation.bootstrap import PairedMetricObservation, paired_video_bootstrap
from streamtimelens.evaluation.metrics import temporal_iou


def oracle_candidate_for_condition(
    report: Mapping[str, Any], condition_name: str,
) -> dict[str, Any]:
    """Extract a refiner-gate candidate without weakening the Oracle tolerance."""
    condition = report.get("conditions", {}).get(condition_name)
    if not isinstance(condition, Mapping):
        raise ValueError(f"oracle condition is absent: {condition_name}")
    metrics = condition.get("metrics")
    gate = condition.get("p0_gate")
    if not isinstance(metrics, Mapping) or not isinstance(gate, Mapping):
        raise ValueError(f"oracle condition is incomplete: {condition_name}")
    dense = metrics.get("dense_crop")
    sparse = metrics.get("sparse_adapter")
    if not isinstance(dense, Mapping) or not isinstance(sparse, Mapping):
        raise ValueError(f"oracle condition omits dense/sparse metrics: {condition_name}")
    sampling_interval_s = float(gate.get("bias_tolerance_s", 0))
    if sampling_interval_s <= 0:
        raise ValueError(f"oracle condition has no valid sampling tolerance: {condition_name}")
    return {
        "dense_miou": float(dense["miou"]),
        "sparse_miou": float(sparse["miou"]),
        "sampling_interval_s": sampling_interval_s,
        "start_signed_bias_s": float(sparse["signed_start_bias_s"]),
        "end_signed_bias_s": float(sparse["signed_end_bias_s"]),
        "source_condition": condition_name,
        "source_p0_gate_passed": bool(gate.get("passed")),
    }


def expanded_span(
    span: tuple[float, float], margin_s: float, upper_bound_s: float,
) -> tuple[float, float]:
    if margin_s < 0 or upper_bound_s <= 0 or not 0 <= span[0] <= span[1]:
        raise ValueError("candidate expansion bounds are invalid")
    return max(0.0, span[0] - margin_s), min(upper_bound_s, span[1] + margin_s)


def summarize_candidate_margin(
    rows: Iterable[Mapping[str, Any]], *, margin_s: float, coarse_margin_s: float,
) -> dict[str, Any]:
    selected = list(rows)
    if not selected or margin_s <= 0 or coarse_margin_s < 0:
        raise ValueError("candidate-margin summary needs rows and valid margins")
    candidate_hits = []
    baseline_hits = []
    coarse_ious = []
    paired_rows = []
    left_hits = []
    right_hits = []
    for row in selected:
        gt = tuple(map(float, row["gt_span"]))
        upper_bound_s = float(row["upper_bound_s"])
        candidates = [tuple(map(float, span)) for span in row["candidate_spans"]]
        expanded = [expanded_span(span, margin_s, upper_bound_s) for span in candidates]
        baseline_hit = float(max((temporal_iou(gt, span) for span in candidates[:5]), default=0) >= .5)
        candidate_hit = float(max((temporal_iou(gt, span) for span in expanded[:5]), default=0) >= .5)
        baseline_hits.append(baseline_hit)
        candidate_hits.append(candidate_hit)
        paired_rows.extend([
            PairedMetricObservation(
                "candidate_margin_v2", int(row["budget_bytes"]), str(row["video_id"]),
                str(row["sample_id"]), candidate_hit,
            ),
            PairedMetricObservation(
                "uniform_raw_v1", int(row["budget_bytes"]), str(row["video_id"]),
                str(row["sample_id"]), baseline_hit,
            ),
        ])
        if expanded:
            coarse = expanded_span(expanded[0], coarse_margin_s, upper_bound_s)
        else:
            coarse = None
        coarse_ious.append(temporal_iou(gt, coarse))
        left_hits.append(float(any(span[0] <= gt[0] <= span[1] for span in expanded)))
        right_hits.append(float(any(span[0] <= gt[1] <= span[1] for span in expanded)))
    return {
        "count": len(selected),
        "candidate_margin_s": margin_s,
        "coarse_margin_s": coarse_margin_s,
        "candidate_recall_at_5": sum(candidate_hits) / len(selected),
        "baseline_candidate_recall_at_5": sum(baseline_hits) / len(selected),
        "coarse_miou": sum(coarse_ious) / len(selected),
        "left_boundary_hit_rate": sum(left_hits) / len(selected),
        "right_boundary_hit_rate": sum(right_hits) / len(selected),
        "paired_bootstrap_vs_uniform_v1": paired_video_bootstrap(
            paired_rows, method_a="candidate_margin_v2", method_b="uniform_raw_v1",
        ),
    }


def select_candidate_margin_configs(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    summaries = list(rows)
    selected = []
    for budget in sorted({int(row["budget_bytes"]) for row in summaries}):
        eligible = [
            row for row in summaries
            if int(row["budget_bytes"]) == budget
            and float(row["candidate_recall_at_5"]) >= .70
            and float(row["paired_bootstrap_vs_uniform_v1"]["ci_low"]) > 0
        ]
        if eligible:
            selected.append(sorted(
                eligible,
                key=lambda row: (
                    -float(row["candidate_recall_at_5"]), -float(row["coarse_miou"]),
                    float(row["candidate_margin_s"]), str(row["config_id"]),
                ),
            )[0])
    one_mib = [row for row in selected if int(row["budget_bytes"]) == 1048576]
    return {
        "passed": bool(one_mib),
        "candidate_recall_at_5_1m_gate": bool(one_mib),
        "selected_config_ids": [str(row["config_id"]) for row in selected],
        "selected": selected,
        "selection_source": "independent_dev_only",
        "gate": {
            "candidate_recall_at_5_minimum": .70,
            "paired_bootstrap_ci_low_must_exceed": 0.0,
        },
    }
