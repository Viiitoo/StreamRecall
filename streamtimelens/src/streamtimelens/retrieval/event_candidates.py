"""Fixed query-time retrieval over immutable HEM-01 event records."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from typing import Any, Sequence

import numpy as np

from streamtimelens.memory.hierarchical_event import HierarchicalEvent, load_event_memory
from streamtimelens.observer.clip_encoder import l2_normalize
from streamtimelens.protocol.snapshot import SnapshotReader
from streamtimelens.protocol.types import Prediction
from streamtimelens.retrieval.frame_candidates import FrameCandidate, RetrievedFrame


EVENT_TOP_K = 8
EVENT_MERGE_GAP_S = 4.0
EVENT_EXPAND_NEIGHBORS = 1
EVENT_CANDIDATE_MARGIN_S = 0.0


def _cosine(left: Any, right: Any) -> float:
    first = np.asarray(l2_normalize(left), dtype=np.float32)
    second = np.asarray(l2_normalize(right), dtype=np.float32)
    if first.shape != second.shape:
        raise ValueError("query and HEM event embedding dimensions differ")
    return float(np.clip(np.dot(first, second), -1.0, 1.0))


def _candidate_id(event_ids: Sequence[str], start_s: float, end_s: float) -> str:
    payload = json.dumps(
        {"event_ids": list(event_ids), "start_s": start_s, "end_s": end_s},
        sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")
    return "hem-" + hashlib.sha1(payload).hexdigest()[:16]


def build_event_candidates(
    query_embedding: Sequence[float], events: Sequence[HierarchicalEvent], *,
    upper_bound_s: float,
) -> list[FrameCandidate]:
    """Rank all events, group Top-8 temporal hits, and include one adjacent event."""
    if upper_bound_s <= 0 or not events:
        return []
    timeline = sorted(events, key=lambda row: (row.start_s, row.first_frame_index, row.event_id))
    if any(event.end_s > upper_bound_s + 1e-6 for event in timeline):
        raise ValueError("HEM event exceeds query-visible time")
    scored = [
        (event, _cosine(query_embedding, event.embedding)) for event in timeline
    ]
    ranked = sorted(
        scored,
        key=lambda row: (-row[1], row[0].start_s, row[0].first_frame_index, row[0].event_id),
    )[:EVENT_TOP_K]
    ranks = {event.event_id: rank for rank, (event, _) in enumerate(ranked, start=1)}
    hits = sorted(ranked, key=lambda row: (row[0].start_s, row[0].first_frame_index))
    groups: list[list[tuple[HierarchicalEvent, float]]] = []
    for hit in hits:
        if not groups or hit[0].start_s - groups[-1][-1][0].end_s > EVENT_MERGE_GAP_S:
            groups.append([hit])
        else:
            groups[-1].append(hit)
    positions = {event.event_id: index for index, event in enumerate(timeline)}
    candidates = []
    for group in groups:
        left = max(0, min(positions[event.event_id] for event, _ in group) - EVENT_EXPAND_NEIGHBORS)
        right = min(
            len(timeline) - 1,
            max(positions[event.event_id] for event, _ in group) + EVENT_EXPAND_NEIGHBORS,
        )
        selected = timeline[left:right + 1]
        event_ids = tuple(event.event_id for event in selected)
        start_s = max(0.0, selected[0].start_s - EVENT_CANDIDATE_MARGIN_S)
        end_s = min(upper_bound_s, selected[-1].end_s + EVENT_CANDIDATE_MARGIN_S)
        if end_s <= start_s:
            end_s = min(upper_bound_s, start_s + 1e-3)
            if end_s <= start_s:
                continue
        hit_ids = tuple(event.event_id for event, _ in group)
        retrieved = tuple(
            RetrievedFrame(
                event.event_id, (event.start_s + event.end_s) / 2,
                event.first_frame_index, score, ranks[event.event_id],
            )
            for event, score in sorted(group, key=lambda row: ranks[row[0].event_id])
        )
        candidates.append(FrameCandidate(
            _candidate_id(event_ids, start_s, end_s), start_s, end_s,
            max(score for _, score in group), event_ids, hit_ids, retrieved,
        ))
    return sorted(candidates, key=lambda row: (-row.score, row.start_s, row.candidate_id))


def retrieve_event_candidates(
    query_embedding: Sequence[float], snapshot: SnapshotReader,
) -> list[FrameCandidate]:
    if not isinstance(snapshot, SnapshotReader):
        raise TypeError("HEM-01 query reader requires a verified SnapshotReader")
    return build_event_candidates(
        query_embedding, load_event_memory(snapshot),
        upper_bound_s=float(snapshot.manifest.t_q),
    )


def locate_event_memory(
    query_embedding: Sequence[float], snapshot: SnapshotReader,
) -> tuple[Prediction, dict[str, Any]]:
    """Return the fixed best event envelope and a complete external debug trace."""
    candidates = retrieve_event_candidates(query_embedding, snapshot)
    if not candidates:
        return Prediction(None, None, 0.0, "NOT_FOUND"), {
            "method": "HEM-01", "candidate_count": 0, "candidates": [],
        }
    best = candidates[0]
    confidence = max(0.0, min(1.0, (best.score + 1.0) / 2.0))
    prediction = Prediction(
        best.start_s, best.end_s, confidence, "fallback",
        evidence_ids=best.frame_refs, candidate_ids=(best.candidate_id,),
    )
    return prediction, {
        "method": "HEM-01",
        "candidate_count": len(candidates),
        "selected_candidate_id": best.candidate_id,
        "candidates": [asdict(candidate) for candidate in candidates],
    }
