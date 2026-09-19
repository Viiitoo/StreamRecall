"""Evaluation-only Hybrid V3 recall, selection, paired quality, and reliability."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable, Mapping

from streamtimelens.evaluation.bootstrap import (
    BootstrapConfig,
    PairedMetricObservation,
    paired_video_bootstrap,
)
from streamtimelens.evaluation.metrics import VTGMetricExample, evaluate_vtg, temporal_iou


def _span(value: Any) -> tuple[float, float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    result = (float(value[0]), float(value[1]))
    return result if 0 <= result[0] < result[1] else None


def _duration_s(row: Mapping[str, Any]) -> float:
    """Normalize the two duration spellings used by repository benchmarks."""
    duration_s = row.get("duration_s")
    duration = row.get("duration")
    if duration_s is not None and duration is not None:
        if abs(float(duration_s) - float(duration)) > 1e-6:
            raise ValueError("annotation duration and duration_s disagree")
    return float(duration_s if duration_s is not None else duration or 0)


def _annotation_map(rows: Iterable[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    result = {}
    for row in rows:
        query_id = str(row.get("query_id", ""))
        if not query_id or query_id in result:
            raise ValueError("Hybrid V3 annotations need unique query IDs")
        duration_s = _duration_s(row)
        if _span(row.get("gt_span")) is None or duration_s <= 0:
            raise ValueError(f"Hybrid V3 annotation is incomplete: {query_id}")
        result[query_id] = {**row, "duration_s": duration_s}
    return result


def _metric_rows(rows, key):
    examples = []
    for row in rows:
        prediction = row[key]
        span = _span(prediction.get("span"))
        examples.append(VTGMetricExample(
            str(row["sample_id"]), str(row["video_id"]), row["gt_span"], span,
            str(prediction.get("status", "invalid")),
        ))
    return evaluate_vtg(examples).to_dict()


def _candidate_span(row: Mapping[str, Any], candidate_id: str | None) -> tuple[float, float] | None:
    if not candidate_id:
        return None
    for candidate in row.get("candidates", []):
        if str(candidate.get("candidate_id")) == candidate_id:
            return _span((candidate.get("start_s"), candidate.get("end_s")))
    return None


def _stage_predictions(row: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Build evaluation-only readouts without exposing GT to the query worker."""
    coarse = dict(row["clip_coarse_prediction"])
    timelens = row.get("timelens_prediction", {})
    parse_status = str(timelens.get("parse_status", "not_called"))
    selected_span = _candidate_span(row, row.get("selected_candidate_id"))
    selected = (
        {"span": list(selected_span), "status": "ok"}
        if parse_status in ("ok", "keep") and selected_span is not None
        else coarse
    )
    refined_span = _span(timelens.get("span"))
    refined = (
        {"span": list(refined_span), "status": "ok"}
        if parse_status == "ok" and refined_span is not None
        else selected if parse_status == "keep" else coarse
    )
    gt_span = row["gt_span"]
    candidate_spans = [
        _span((item.get("start_s"), item.get("end_s")))
        for item in row.get("candidates", [])
    ]
    valid_candidates = [span for span in candidate_spans if span is not None]
    oracle_span = max(
        valid_candidates,
        key=lambda value: temporal_iou(gt_span, value),
        default=None,
    )
    oracle = {
        "span": list(oracle_span) if oracle_span is not None else None,
        "status": "ok" if oracle_span is not None else "NOT_FOUND",
    }
    return {
        "selected_candidate_or_fallback": selected,
        "timelens_refined_or_fallback": refined,
        "candidate_oracle": oracle,
    }


def _readout_diagnostics(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    parse_status_counts: dict[str, int] = defaultdict(int)
    candidate_label_counts: dict[str, int] = defaultdict(int)
    score_rank_counts: dict[str, int] = defaultdict(int)
    final_reason_counts: dict[str, int] = defaultdict(int)
    count = 0
    for row in rows:
        count += 1
        final_reason_counts[str(row.get("final_selection_reason", "missing"))] += 1
        prediction = row.get("timelens_prediction", {})
        status = str(prediction.get("parse_status", "not_called"))
        parse_status_counts[status] += 1
        if status not in ("ok", "keep"):
            continue
        label = prediction.get("candidate_label")
        if label:
            candidate_label_counts[str(label)] += 1
        selected_id = str(row.get("selected_candidate_id") or "")
        rank = next((
            index for index, candidate in enumerate(row.get("candidates", []), 1)
            if str(candidate.get("candidate_id")) == selected_id
        ), None)
        if rank is not None:
            score_rank_counts[str(rank)] += 1
    return {
        "count": count,
        "parse_status_counts": dict(sorted(parse_status_counts.items())),
        "candidate_label_counts": dict(sorted(candidate_label_counts.items())),
        "selected_candidate_score_rank_counts": dict(sorted(
            score_rank_counts.items(), key=lambda item: int(item[0]),
        )),
        "keep_rate": parse_status_counts.get("keep", 0) / count if count else None,
        "final_selection_reason_counts": dict(sorted(final_reason_counts.items())),
    }


def summarize_hybrid_v3(
    predictions: Iterable[Mapping[str, Any]],
    annotations: Iterable[Mapping[str, Any]],
    *,
    budget_bytes: int,
    bootstrap_resamples: int = 10_000,
) -> dict[str, Any]:
    """Join GT only here, filter past-only, and compare paired CLIP/Hybrid rows."""
    if budget_bytes <= 0 or bootstrap_resamples <= 0:
        raise ValueError("Hybrid V3 evaluation budget/bootstrap are invalid")
    annotation_by_id = _annotation_map(annotations)
    eligible = []
    excluded_future = 0
    seen = set()
    for raw in predictions:
        query_id = str(raw.get("query_id", ""))
        annotation = annotation_by_id.get(query_id)
        if annotation is None or str(annotation.get("video_id")) != str(raw.get("video_id")):
            raise ValueError(f"Hybrid V3 prediction/annotation mismatch: {query_id}")
        sample_id = str(raw.get("sample_id") or f"{query_id}@{float(raw['rho_q']):.2f}")
        if sample_id in seen:
            raise ValueError(f"duplicate Hybrid V3 prediction sample: {sample_id}")
        seen.add(sample_id)
        gt = _span(annotation["gt_span"])
        assert gt is not None
        t_q = float(raw["rho_q"]) * float(annotation["duration_s"])
        if gt[1] > t_q + 1e-6:
            excluded_future += 1
            continue
        row = {**raw, "sample_id": sample_id, "gt_span": gt, "t_q": t_q}
        eligible.append({**row, **_stage_predictions(row)})
    if not eligible:
        raise ValueError("Hybrid V3 evaluation has no past-only eligible predictions")

    final_metrics = _metric_rows(eligible, "final_prediction")
    clip_metrics = _metric_rows(eligible, "clip_coarse_prediction")
    stage_metrics = {
        "clip_coarse": clip_metrics,
        "selected_candidate_or_fallback": _metric_rows(
            eligible, "selected_candidate_or_fallback",
        ),
        "timelens_refined_or_fallback": _metric_rows(
            eligible, "timelens_refined_or_fallback",
        ),
        "candidate_oracle": _metric_rows(eligible, "candidate_oracle"),
    }
    observations = []
    candidate_hits = {1: [], 4: [], 6: []}
    evidence_hits = []
    selection_hits = []
    selection_eligible = 0
    parse_valid = 0
    parse_attempts = 0
    exception_count = 0
    fallback_count = 0
    by_rho = defaultdict(list)
    for row in eligible:
        final_span = _span(row["final_prediction"].get("span"))
        clip_span = _span(row["clip_coarse_prediction"].get("span"))
        final_iou = temporal_iou(row["gt_span"], final_span)
        clip_iou = temporal_iou(row["gt_span"], clip_span)
        for method, value in (("hybrid_v3", final_iou), ("clip_coarse", clip_iou)):
            observations.append(PairedMetricObservation(
                method, budget_bytes, str(row["video_id"]), str(row["sample_id"]), value,
            ))
        candidate_spans = [
            (float(item["start_s"]), float(item["end_s"]))
            for item in row.get("candidates", [])
        ]
        for limit in candidate_hits:
            candidate_hits[limit].append(float(max(
                (temporal_iou(row["gt_span"], span) for span in candidate_spans[:limit]),
                default=0.0,
            ) >= .5))
        allocation = next(
            (item for item in row.get("query_trace", []) if item.get("kind") == "frame_allocation"),
            {},
        )
        selected_times = [float(item["timestamp_s"]) for item in allocation.get("selected_frames", [])]
        evidence_hits.append(float(any(row["gt_span"][0] <= value <= row["gt_span"][1] for value in selected_times)))
        selected_id = row.get("selected_candidate_id")
        if selected_id:
            selection_eligible += 1
            selected_span = next((
                span for item, span in zip(row.get("candidates", []), candidate_spans)
                if item.get("candidate_id") == selected_id
            ), None)
            selection_hits.append(float(temporal_iou(row["gt_span"], selected_span) >= .5))
        status = str(row.get("timelens_prediction", {}).get("parse_status", "not_called"))
        if status != "not_called":
            parse_attempts += 1
            parse_valid += int(status in ("ok", "keep", "not_found"))
        exception_count += int(row.get("timelens_prediction", {}).get("status") == "exception")
        fallback_count += int(bool(row.get("fallback_used")))
        by_rho[f"{float(row['rho_q']):.2f}"].append(row)
    count = len(eligible)
    return {
        "schema_version": 1,
        "protocol": "evaluation_only_past_visible",
        "count": count,
        "excluded_future_gt": excluded_future,
        "budget_bytes": budget_bytes,
        "hybrid": final_metrics,
        "clip_coarse": clip_metrics,
        "stage_metrics": stage_metrics,
        "readout_diagnostics": _readout_diagnostics(eligible),
        "paired_miou": paired_video_bootstrap(
            observations, method_a="hybrid_v3", method_b="clip_coarse",
            config=BootstrapConfig(resamples=bootstrap_resamples),
        ),
        "candidate_recall_at_iou_05": {
            str(limit): sum(values) / count for limit, values in candidate_hits.items()
        },
        "bundle_midpoint_evidence_recall": sum(evidence_hits) / count,
        "selected_candidate_accuracy_at_iou_05": (
            sum(selection_hits) / selection_eligible if selection_eligible else None
        ),
        "selected_candidate_count": selection_eligible,
        "parse_valid_rate": parse_valid / parse_attempts if parse_attempts else None,
        "parse_attempts": parse_attempts,
        "fallback_rate": fallback_count / count,
        "exception_rate": exception_count / count,
        "by_rho": {
            rho: {
                "count": len(group),
                "hybrid": _metric_rows(group, "final_prediction"),
                "clip_coarse": _metric_rows(group, "clip_coarse_prediction"),
                "stage_metrics": {
                    "clip_coarse": _metric_rows(group, "clip_coarse_prediction"),
                    "selected_candidate_or_fallback": _metric_rows(
                        group, "selected_candidate_or_fallback",
                    ),
                    "timelens_refined_or_fallback": _metric_rows(
                        group, "timelens_refined_or_fallback",
                    ),
                    "candidate_oracle": _metric_rows(group, "candidate_oracle"),
                },
                "readout_diagnostics": _readout_diagnostics(group),
            }
            for rho, group in sorted(by_rho.items())
        },
    }
