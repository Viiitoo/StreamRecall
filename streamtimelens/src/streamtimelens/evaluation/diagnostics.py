"""Stage-by-stage retrieval, boundary, writer, and compaction diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from streamtimelens.evaluation.metrics import temporal_iou


@dataclass(frozen=True)
class PipelineDiagnosticExample:
    query_id: str
    gt_span: tuple[float, float]
    cards: tuple[dict[str, Any], ...]
    retrieved_card_ids: tuple[str, ...]
    candidate_spans: tuple[tuple[float, float], ...]
    coarse_span: tuple[float, float] | None
    refined_span: tuple[float, float] | None


@dataclass(frozen=True)
class VisualDiagnosticExample:
    query_id: str
    gt_span: tuple[float, float]
    retrieved_frames: tuple[dict[str, Any], ...]
    candidate_spans: tuple[tuple[float, float], ...]
    coarse_span: tuple[float, float] | None
    refined_span: tuple[float, float] | None = None


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def evaluate_pipeline_diagnostics(
    examples: Iterable[PipelineDiagnosticExample],
    trace_rows: Iterable[dict[str, Any]],
    *,
    recall_iou_threshold: float = 0.5,
) -> dict[str, Any]:
    if not 0 <= recall_iou_threshold <= 1:
        raise ValueError("diagnostic recall IoU threshold is invalid")
    rows = list(examples)
    if len({row.query_id for row in rows}) != len(rows):
        raise ValueError("diagnostic examples need unique query IDs")
    writer_coverage = []
    oracle_gaps = []
    deltas = []
    card_recall = {1: [], 5: [], 8: []}
    candidate_recall = {1: [], 5: [], 8: []}
    left_hits = []
    right_hits = []
    both_hits = []
    for row in rows:
        cards = {str(card["id"]): card for card in row.cards}
        card_ious = {
            card_id: temporal_iou(
                row.gt_span, (float(card["t_start"]), float(card["t_end"])),
            )
            for card_id, card in cards.items()
        }
        coverage = max(card_ious.values(), default=0.0)
        writer_coverage.append(coverage)
        retrieved_ious = [card_ious.get(card_id, 0.0) for card_id in row.retrieved_card_ids]
        oracle_gaps.append(coverage - max(retrieved_ious, default=0.0))
        for k in (1, 5, 8):
            card_recall[k].append(float(max(retrieved_ious[:k], default=0.0) >= recall_iou_threshold))
            candidate_recall[k].append(float(max(
                (temporal_iou(row.gt_span, span) for span in row.candidate_spans[:k]),
                default=0.0,
            ) >= recall_iou_threshold))
        relevant = [
            cards[card_id] for card_id, value in card_ious.items()
            if value >= recall_iou_threshold
        ]
        left = any(bool((card.get("boundary_cache") or {}).get("left_hit")) for card in relevant)
        right = any(bool((card.get("boundary_cache") or {}).get("right_hit")) for card in relevant)
        left_hits.append(float(left))
        right_hits.append(float(right))
        both_hits.append(float(left and right))
        deltas.append(
            temporal_iou(row.gt_span, row.refined_span)
            - temporal_iou(row.gt_span, row.coarse_span)
        )
    traces = list(trace_rows)
    writer_rows = [row for row in traces if row.get("kind") == "writer_called"]
    parse_counts = {
        status: sum(str(row.get("parse_status", row.get("reason", ""))) == status for row in writer_rows)
        for status in ("valid", "repaired", "fallback")
    }
    writer_total = len(writer_rows)
    merge_count = sum(row.get("kind") in ("nodes_merged", "card_merged") for row in traces)
    eviction_count = sum(row.get("kind") == "payload_evicted" for row in traces)
    return {
        "count": len(rows),
        "writer_coverage_mean_iou": _mean(writer_coverage),
        "card_recall": {str(key): _mean(value) for key, value in card_recall.items()},
        "candidate_recall": {str(key): _mean(value) for key, value in candidate_recall.items()},
        "left_boundary_hit_rate": _mean(left_hits),
        "right_boundary_hit_rate": _mean(right_hits),
        "both_boundary_hit_rate": _mean(both_hits),
        "oracle_retrieval_gap_mean_iou": _mean(oracle_gaps),
        "post_minus_pre_refinement_mean_iou": _mean(deltas),
        "writer_parse_counts": parse_counts,
        "writer_parse_rates": {
            key: value / writer_total if writer_total else 0.0
            for key, value in parse_counts.items()
        },
        "writer_calls": writer_total,
        "merge_count": merge_count,
        "merge_per_writer_call": merge_count / writer_total if writer_total else 0.0,
        "eviction_count": eviction_count,
        "eviction_per_writer_call": eviction_count / writer_total if writer_total else 0.0,
    }


def evaluate_visual_diagnostics(
    examples: Iterable[VisualDiagnosticExample],
    *,
    candidate_iou_threshold: float = 0.5,
) -> dict[str, Any]:
    """Compute V2 diagnostics directly from retrieved frames and envelopes."""
    if not 0 <= candidate_iou_threshold <= 1:
        raise ValueError("visual diagnostic IoU threshold is invalid")
    rows = list(examples)
    if len({row.query_id for row in rows}) != len(rows):
        raise ValueError("visual diagnostic examples need unique query IDs")
    frame_recall = {1: [], 5: [], 8: []}
    candidate_recall = {1: [], 5: []}
    candidate_oracle_ious = []
    left_hits = []
    right_hits = []
    deltas = []
    for row in rows:
        start, end = row.gt_span
        if start < 0 or start >= end:
            raise ValueError("visual diagnostic ground truth is invalid")
        ordered_frames = sorted(
            row.retrieved_frames,
            key=lambda item: (int(item.get("rank", 10**9)), str(item.get("frame_ref", ""))),
        )
        timestamps = [float(item["timestamp_s"]) for item in ordered_frames]
        for k in (1, 5, 8):
            frame_recall[k].append(float(any(start <= timestamp <= end for timestamp in timestamps[:k])))
        candidate_ious = [temporal_iou(row.gt_span, span) for span in row.candidate_spans]
        candidate_oracle_ious.append(max(candidate_ious, default=0.0))
        for k in (1, 5):
            candidate_recall[k].append(float(
                max(candidate_ious[:k], default=0.0) >= candidate_iou_threshold
            ))
        left_hits.append(float(any(left <= start <= right for left, right in row.candidate_spans)))
        right_hits.append(float(any(left <= end <= right for left, right in row.candidate_spans)))
        if row.refined_span is not None:
            deltas.append(
                temporal_iou(row.gt_span, row.refined_span)
                - temporal_iou(row.gt_span, row.coarse_span)
            )
    return {
        "count": len(rows),
        "frame_recall": {str(key): _mean(values) for key, values in frame_recall.items()},
        "candidate_recall": {
            str(key): _mean(values) for key, values in candidate_recall.items()
        },
        "candidate_oracle_miou": _mean(candidate_oracle_ious),
        "left_boundary_hit_rate": _mean(left_hits),
        "right_boundary_hit_rate": _mean(right_hits),
        "both_boundary_hit_rate": _mean([
            float(left and right) for left, right in zip(left_hits, right_hits)
        ]),
        "post_minus_pre_refinement_mean_iou": _mean(deltas),
        "refined_count": len(deltas),
    }
