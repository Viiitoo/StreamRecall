"""Join GT-bearing frozen arrival plans with GT-free prediction values."""

from __future__ import annotations

from typing import Any, Iterable

from streamtimelens.evaluation.groups import GroupedVTGExample, evaluate_grouped_vtg
from streamtimelens.evaluation.metrics import VTGMetricExample, evaluate_vtg
from streamtimelens.protocol.arrival import ArrivalRecord


def evaluate_predictions(
    plan: Iterable[ArrivalRecord], predictions: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    rows = list(predictions)
    by_key = {}
    for row in rows:
        key = (str(row.get("video_id", "")), str(row.get("query_id", "")), round(float(row["rho_q"]), 8))
        if not all(key[:2]) or key in by_key:
            raise ValueError("predictions need unique video/query/rho keys")
        forbidden = {"gt", "gt_span", "ground_truth", "ground_truth_span"} & {
            str(name).lower() for name in row
        }
        if forbidden:
            raise ValueError("prediction value contains ground truth")
        by_key[key] = row
    grouped = []
    missing = []
    for index, arrival in enumerate(plan):
        if not arrival.eligible:
            continue
        key = (arrival.video_id, arrival.query_id, round(arrival.rho_q, 8))
        prediction = by_key.get(key)
        if prediction is None:
            missing.append(key)
            continue
        raw_span = prediction.get("span")
        if raw_span is None and prediction.get("start_s") is not None:
            raw_span = [prediction["start_s"], prediction["end_s"]]
        span = tuple(map(float, raw_span)) if raw_span is not None else None
        status = str(prediction.get("status", "error")).lower()
        metric = VTGMetricExample(
            f"{arrival.query_id}::{arrival.rho_q:.8f}::{arrival.cohort}::{index}",
            arrival.video_id, arrival.gt_span, span, status,
        )
        duration = arrival.t_q / arrival.rho_q
        grouped.append(GroupedVTGExample(
            metric, arrival.cohort, True, arrival.lag_bin, duration,
        ))
    if missing:
        raise ValueError(f"missing {len(missing)} eligible prediction keys")
    natural = [row.metric for row in grouped if row.cohort == "natural"]
    return {
        "overall_natural": evaluate_vtg(natural).to_dict(),
        "groups": evaluate_grouped_vtg(grouped),
        "joined_prediction_values": len(by_key),
        "eligible_evaluation_rows": len(grouped),
    }
