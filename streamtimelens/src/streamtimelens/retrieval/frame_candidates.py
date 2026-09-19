"""Shared CLIP frame-to-query retrieval for Uniform and Semantic baselines."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

from streamtimelens.observer.clip_encoder import deserialize_embedding, l2_normalize
from streamtimelens.protocol.snapshot import SnapshotReader
from streamtimelens.protocol.types import Prediction


@dataclass(frozen=True)
class RetrievedFrame:
    frame_ref: str
    timestamp_s: float
    frame_index: int
    score: float
    rank: int


@dataclass(frozen=True)
class FrameCandidate:
    candidate_id: str
    start_s: float
    end_s: float
    score: float
    frame_refs: tuple[str, ...]
    hit_refs: tuple[str, ...]
    retrieved_frames: tuple[RetrievedFrame, ...] = ()

    @property
    def span(self) -> tuple[float, float]:
        return self.start_s, self.end_s


def _cosine(left: Any, right: Any) -> float:
    import numpy as np

    first, second = l2_normalize(left), l2_normalize(right)
    if first.shape != second.shape:
        raise ValueError("query and frame embeddings have different dimensions")
    return float(np.clip(np.dot(first, second), -1.0, 1.0))


def _candidate_id(refs: Iterable[str], start_s: float, end_s: float) -> str:
    payload = json.dumps(
        {"refs": list(refs), "start_s": start_s, "end_s": end_s},
        sort_keys=True, separators=(",", ":"),
    )
    return "frame-" + hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


def build_frame_candidates(
    query_embedding: Any,
    frame_metadata: dict[str, dict[str, Any]],
    *,
    top_k: int = 8,
    merge_gap_s: float = 4.0,
    expand_neighbors: int = 1,
    candidate_margin_s: float = 0.0,
    upper_bound_s: float | None = None,
    merge_gap_tolerance_s: float = 0.0,
) -> list[FrameCandidate]:
    """Rank frames, merge adjacent hits and expand to cached neighbors."""
    if top_k <= 0 or merge_gap_s < 0 or merge_gap_tolerance_s < 0 or expand_neighbors < 0 or candidate_margin_s < 0 or (
        upper_bound_s is not None and upper_bound_s <= 0
    ):
        raise ValueError("candidate retrieval parameters are invalid")
    available = []
    for frame_ref, row in frame_metadata.items():
        if "clip_embedding" not in row:
            continue
        timestamp = float(row["timestamp_s"])
        if timestamp < 0:
            raise ValueError("cached frame timestamp must be non-negative")
        score = _cosine(query_embedding, deserialize_embedding(row["clip_embedding"]))
        available.append((timestamp, int(row["frame_index"]), frame_ref, score))
    if not available:
        return []
    timeline = sorted(available, key=lambda item: (item[0], item[1], item[2]))
    ranked = sorted(timeline, key=lambda item: (-item[3], item[0], item[1], item[2]))[:top_k]
    ranks = {item[2]: rank for rank, item in enumerate(ranked, start=1)}
    hits = sorted(ranked, key=lambda item: (item[0], item[1], item[2]))
    groups: list[list[tuple[float, int, str, float]]] = []
    for hit in hits:
        if not groups or hit[0] - groups[-1][-1][0] > merge_gap_s + merge_gap_tolerance_s:
            groups.append([hit])
        else:
            groups[-1].append(hit)

    positions = {item[2]: index for index, item in enumerate(timeline)}
    candidates: list[FrameCandidate] = []
    for group in groups:
        left = max(0, min(positions[item[2]] for item in group) - expand_neighbors)
        right = min(len(timeline) - 1, max(positions[item[2]] for item in group) + expand_neighbors)
        selected = timeline[left:right + 1]
        refs = tuple(item[2] for item in selected)
        start_s = max(0.0, selected[0][0] - candidate_margin_s)
        end_s = selected[-1][0] + candidate_margin_s
        if upper_bound_s is not None:
            start_s, end_s = min(start_s, upper_bound_s), min(end_s, upper_bound_s)
        # A one-frame hit still needs an ordered local-refiner interval.
        if end_s <= start_s:
            if upper_bound_s is None:
                end_s = start_s + 1e-3
            else:
                start_s = max(0.0, end_s - 1e-3)
                if start_s >= end_s:
                    continue
        score = max(item[3] for item in group)
        hit_refs = tuple(item[2] for item in group)
        retrieved = tuple(
            RetrievedFrame(item[2], item[0], item[1], item[3], ranks[item[2]])
            for item in sorted(group, key=lambda value: ranks[value[2]])
        )
        candidates.append(FrameCandidate(
            _candidate_id(refs, start_s, end_s), start_s, end_s, score, refs, hit_refs,
            retrieved,
        ))
    return sorted(candidates, key=lambda item: (-item.score, item.start_s, item.candidate_id))


def retrieve_frame_candidates(
    query: str,
    snapshot: SnapshotReader,
    encoder: Any,
    *,
    top_k: int = 8,
    merge_gap_s: float = 4.0,
    expand_neighbors: int = 1,
    candidate_margin_s: float = 0.0,
    merge_gap_tolerance_s: float = 0.0,
) -> list[FrameCandidate]:
    query_embedding = encoder.encode_text(query)
    return build_frame_candidates(
        query_embedding, snapshot.read_frame_metadata(), top_k=top_k,
        merge_gap_s=merge_gap_s, expand_neighbors=expand_neighbors,
        candidate_margin_s=candidate_margin_s,
        upper_bound_s=snapshot.manifest.t_q,
        merge_gap_tolerance_s=merge_gap_tolerance_s,
    )


def coarse_prediction(
    candidates: Sequence[FrameCandidate], *, margin_s: float = 0.0,
    upper_bound_s: float | None = None,
) -> Prediction:
    """Return the best temporal-cluster envelope with a bounded frozen margin."""
    if margin_s < 0 or (upper_bound_s is not None and upper_bound_s <= 0):
        raise ValueError("coarse prediction bounds are invalid")
    if not candidates:
        return Prediction(None, None, 0.0, "NOT_FOUND")
    best = candidates[0]
    start_s = max(0.0, best.start_s - margin_s)
    end_s = best.end_s + margin_s
    if upper_bound_s is not None:
        end_s = min(end_s, upper_bound_s)
    if end_s <= start_s:
        # This is only reachable for a malformed upper bound/candidate pairing;
        # do not silently manufacture an interval outside query-visible time.
        return Prediction(None, None, 0.0, "NOT_FOUND")
    confidence = max(0.0, min(1.0, (best.score + 1.0) / 2.0))
    return Prediction(
        start_s, end_s, confidence, "fallback",
        evidence_ids=best.frame_refs, candidate_ids=(best.candidate_id,),
    )


def _uniform_subset(values: Sequence[str], limit: int) -> tuple[str, ...]:
    if limit <= 0:
        raise ValueError("refiner frame limit must be positive")
    if len(values) <= limit:
        return tuple(values)
    if limit == 1:
        return (values[len(values) // 2],)
    indices = [round(index * (len(values) - 1) / (limit - 1)) for index in range(limit)]
    return tuple(values[index] for index in indices)


def refine_frame_candidate(
    query: str,
    snapshot: SnapshotReader,
    candidate: FrameCandidate,
    service: Any,
    *,
    max_frames: int = 16,
    max_new_tokens: int = 128,
) -> tuple[Prediction, dict[str, Any]]:
    """Run the shared sparse local TimeLens path on one frame candidate."""
    import numpy as np
    from PIL import Image

    from streamtimelens.refiner.parse import parse_refiner_output
    from streamtimelens.refiner.prompts import SPARSE_LOCAL_VERSION, build_grounding_prompt, video_messages
    from streamtimelens.refiner.timelens_adapter import SparseFrame, prepare_sparse_video

    metadata = snapshot.read_frame_metadata()
    refs = _uniform_subset(candidate.frame_refs, max_frames)
    frames = []
    fps = float(snapshot.manifest.video_meta["original_fps"])
    total_frames = int(snapshot.manifest.video_meta["total_num_frames"])
    for frame_ref in refs:
        row = metadata[frame_ref]
        with Image.open(snapshot.frame_path(frame_ref)) as image:
            pixels = np.asarray(image.convert("RGB"))
        frame_index = int(row["frame_index"])
        frames.append(SparseFrame(frame_index, frame_index / fps, pixels))
    prepared = prepare_sparse_video(
        frames, original_fps=fps, total_num_frames=total_frames, k_frames=len(frames),
    )
    prompt = build_grounding_prompt(
        SPARSE_LOCAL_VERSION, query=query, candidate=candidate.span, card_summary=None,
    )
    answer = service.generate(video_messages(prompt), [prepared.processor_video], max_new_tokens)
    parsed = parse_refiner_output(answer, t_q=snapshot.manifest.t_q, candidate=candidate.span)
    confidence = max(0.0, min(1.0, (candidate.score + 1.0) / 2.0))
    trace = {
        "kind": "refiner_completed" if parsed.valid else "refiner_fallback",
        "candidate_id": candidate.candidate_id,
        "candidate_span": list(candidate.span),
        "frame_refs": list(refs),
        "timestamp_audit": list(prepared.timestamp_audit),
        "parse_status": parsed.status,
        "multiple_span": parsed.multiple_span,
        "raw_answer": answer,
        "model_stats": dict(getattr(service, "last_call_stats", {})),
    }
    if parsed.span is None:
        return Prediction(
            candidate.start_s, candidate.end_s, confidence, "fallback",
            evidence_ids=refs, candidate_ids=(candidate.candidate_id,),
        ), trace
    return Prediction(
        parsed.span[0], parsed.span[1], confidence, "ok",
        evidence_ids=refs, candidate_ids=(candidate.candidate_id,),
    ), trace
