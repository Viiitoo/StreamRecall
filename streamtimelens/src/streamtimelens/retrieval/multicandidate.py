"""Deterministic temporal diversity filtering for Hybrid V3 candidates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from streamtimelens.retrieval.frame_candidates import FrameCandidate


@dataclass(frozen=True)
class MultiCandidateSelection:
    candidates: tuple[FrameCandidate, ...]
    decisions: tuple[dict[str, Any], ...]


def temporal_iou(left: tuple[float, float], right: tuple[float, float]) -> float:
    intersection = max(0.0, min(left[1], right[1]) - max(left[0], right[0]))
    union = max(left[1], right[1]) - min(left[0], right[0])
    return intersection / union if union > 0 else 0.0


def select_diverse_candidates(
    candidates: Iterable[FrameCandidate],
    *,
    max_candidates: int,
    temporal_nms_iou: float,
    min_cluster_separation_s: float,
) -> MultiCandidateSelection:
    """Apply score-ordered temporal NMS and centre-distance diversity.

    Input order is deliberately ignored. The same candidate set therefore
    produces byte-identical selection decisions across runs and shards.
    """
    if (
        max_candidates <= 0 or not 0 <= temporal_nms_iou <= 1
        or min_cluster_separation_s < 0
    ):
        raise ValueError("multi-candidate selection parameters are invalid")
    ranked = sorted(
        candidates,
        key=lambda item: (-item.score, item.start_s, item.end_s, item.candidate_id),
    )
    selected: list[FrameCandidate] = []
    decisions: list[dict[str, Any]] = []
    for rank, candidate in enumerate(ranked, 1):
        suppressor = None
        reason = None
        overlap = 0.0
        centre_distance = None
        for kept in selected:
            current_iou = temporal_iou(candidate.span, kept.span)
            current_distance = abs(
                (candidate.start_s + candidate.end_s) / 2.0
                - (kept.start_s + kept.end_s) / 2.0
            )
            if current_iou >= temporal_nms_iou and current_iou > 0:
                suppressor, reason, overlap = kept, "temporal_nms", current_iou
                centre_distance = current_distance
                break
            if min_cluster_separation_s > 0 and current_distance < min_cluster_separation_s:
                suppressor, reason, overlap = kept, "minimum_cluster_separation", current_iou
                centre_distance = current_distance
                break
        accepted = suppressor is None and len(selected) < max_candidates
        if accepted:
            selected.append(candidate)
            reason = "accepted"
        elif suppressor is None:
            reason = "max_candidate_clusters"
        decisions.append({
            "candidate_id": candidate.candidate_id,
            "input_rank": rank,
            "span": list(candidate.span),
            "score": candidate.score,
            "accepted": accepted,
            "reason": reason,
            "suppressed_by": suppressor.candidate_id if suppressor else None,
            "temporal_iou": overlap,
            "centre_distance_s": centre_distance,
        })
    return MultiCandidateSelection(tuple(selected), tuple(decisions))
