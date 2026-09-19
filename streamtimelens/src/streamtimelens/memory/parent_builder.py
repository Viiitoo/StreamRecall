"""Deterministic, generation-free construction of compact parent cards."""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from typing import Any, Sequence

from streamtimelens.memory.cards import EvidenceCard
from streamtimelens.observer.clip_encoder import deserialize_embedding, serialize_embedding


def _unique_limited(values: Sequence[str], limit: int) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        normalized = " ".join(str(value).split())
        key = normalized.casefold()
        if normalized and key not in seen:
            seen.add(key)
            result.append(normalized)
            if len(result) == limit:
                break
    return result


def _coverage_sample(values: Sequence[Any], limit: int) -> list[Any]:
    ordered = list(dict.fromkeys(values))
    if len(ordered) <= limit:
        return ordered
    if limit == 1:
        return [ordered[0]]
    indices = [round(index * (len(ordered) - 1) / (limit - 1)) for index in range(limit)]
    return [ordered[index] for index in indices]


def _weights(children: Sequence[EvidenceCard]) -> list[float]:
    # A longer event with more direct visual observations contributes more;
    # zero-duration point events retain a finite vote.
    raw = [
        max(child.t_end - child.t_start, 1e-3) * max(1, len(child.support_timestamps))
        for child in children
    ]
    total = sum(raw)
    return [value / total for value in raw]


def _pool_embedding(
    children: Sequence[EvidenceCard], attribute: str, weights: Sequence[float],
) -> dict[str, Any] | None:
    present = [(index, getattr(child, attribute)) for index, child in enumerate(children)]
    if not all(record is not None for _, record in present):
        return None
    import numpy as np

    vectors = [deserialize_embedding(record) for _, record in present]
    if len({tuple(vector.shape) for vector in vectors}) != 1:
        raise ValueError("child embedding dimensions do not match")
    pooled = sum(weight * vector for weight, vector in zip(weights, vectors))
    if not np.isfinite(pooled).all() or float(np.linalg.norm(pooled)) <= 0:
        return None
    return serialize_embedding(pooled, "fp16")


def _parent_id(children: Sequence[EvidenceCard], hard: bool) -> str:
    payload = {"children": [child.id for child in children], "mode": "hard" if hard else "soft"}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return "parent-" + hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:24]


def build_parent_card(
    children: Sequence[EvidenceCard], *, hard: bool,
    max_actors: int = 8, max_actions: int = 8, max_objects: int = 8,
    max_summary_chars: int = 512, max_support_timestamps: int = 64,
    max_raw_refs: int = 16, max_source_chunks: int = 32,
    compacted_child_count: int | None = None,
) -> EvidenceCard:
    """Build a parent without a VLM call or any novel raw-frame reference."""
    if len(children) < 2:
        raise ValueError("a parent requires at least two children")
    limits = (
        max_actors, max_actions, max_objects, max_summary_chars,
        max_support_timestamps, max_raw_refs, max_source_chunks,
    )
    if any(not isinstance(value, int) or isinstance(value, bool) or value <= 0 for value in limits):
        raise ValueError("parent field limits must be positive integers")
    ordered = sorted(children, key=lambda child: (child.t_start, child.t_end, child.id))
    if len({child.id for child in ordered}) != len(ordered):
        raise ValueError("parent children must have unique IDs")
    weights = _weights(ordered)
    start, end = min(child.t_start for child in ordered), max(child.t_end for child in ordered)
    summaries = _unique_limited([child.summary for child in ordered], len(ordered))
    summary = " | ".join(summaries)
    if len(summary) > max_summary_chars:
        summary = summary[:max_summary_chars - 1].rstrip() + "…"
    scenes: dict[str, float] = defaultdict(float)
    for child, weight in zip(ordered, weights):
        scenes[child.scene] += weight
    scene = min(scenes, key=lambda value: (-scenes[value], value))
    raw_refs = _coverage_sample(
        sorted({frame_id for child in ordered for frame_id in child.raw_ref_ids}),
        max_raw_refs,
    )
    raw_status = {
        frame_id: (
            "available" if any(child.raw_ref_status.get(frame_id) == "available" for child in ordered)
            else next((child.raw_ref_status[frame_id] for child in ordered if frame_id in child.raw_ref_status), "evicted")
        )
        for frame_id in raw_refs
    }
    if compacted_child_count is None:
        compacted_child_count = sum(
            max(1, child.compacted_child_count + len(child.child_ids)) for child in ordered
        ) if hard else 0
    if compacted_child_count < 0:
        raise ValueError("compacted child count must be non-negative")
    source_chunks = _unique_limited(
        [source for child in ordered for source in child.source_chunk_ids],
        max_source_chunks,
    )
    support = _coverage_sample(
        sorted({timestamp for child in ordered for timestamp in child.support_timestamps}),
        max_support_timestamps,
    )
    left_child = min(ordered, key=lambda child: (child.t_start, child.t_end, child.id))
    right_child = max(ordered, key=lambda child: (child.t_end, child.t_start, child.id))
    left_ids = [
        frame_id for frame_id in left_child.boundary_cache.get("left_frame_ids", [])
        if frame_id in raw_refs and raw_status.get(frame_id) == "available"
    ]
    right_ids = [
        frame_id for frame_id in right_child.boundary_cache.get("right_frame_ids", [])
        if frame_id in raw_refs and raw_status.get(frame_id) == "available"
    ]
    internal_ids = [frame_id for frame_id in raw_refs if frame_id not in set((*left_ids, *right_ids))]
    card = EvidenceCard(
        id=_parent_id(ordered, hard), level=max(child.level for child in ordered) + 1,
        t_start=start, t_end=end, summary=summary,
        actors=_unique_limited([value for child in ordered for value in child.actors], max_actors),
        actions=_unique_limited([value for child in ordered for value in child.actions], max_actions),
        objects=_unique_limited([value for child in ordered for value in child.objects], max_objects),
        scene=scene, support_timestamps=support,
        left_uncertainty_s=left_child.left_uncertainty_s,
        right_uncertainty_s=right_child.right_uncertainty_s,
        raw_ref_ids=raw_refs,
        child_ids=[] if hard else [child.id for child in ordered],
        source_chunk_ids=source_chunks,
        phase="ongoing" if right_child.phase == "ongoing" else "complete",
        normalized_text=(
            f"summary: {summary} | actors: {', '.join(_unique_limited([value for child in ordered for value in child.actors], max_actors)) or 'none'} | "
            f"actions: {', '.join(_unique_limited([value for child in ordered for value in child.actions], max_actions)) or 'none'} | "
            f"objects: {', '.join(_unique_limited([value for child in ordered for value in child.objects], max_objects)) or 'none'} | scene: {scene}"
        ),
        text_embedding=_pool_embedding(ordered, "text_embedding", weights),
        visual_centroid=_pool_embedding(ordered, "visual_centroid", weights),
        writer_model_revision="pooled-no-generation",
        prompt_version="parent_pool_v1", prompt_hash="none",
        writer_provenance={
            "merge_builder": "duration_support_pool_v1",
            "child_writer_revisions": sorted({child.writer_model_revision for child in ordered}),
        },
        generated_tokens=0, raw_ref_status=raw_status,
        boundary_cache={
            "left_frame_ids": left_ids, "right_frame_ids": right_ids,
            "internal_frame_ids": internal_ids, "left_hit": bool(left_ids),
            "right_hit": bool(right_ids), "both_hit": bool(left_ids and right_ids),
        },
        merge_mode="hard" if hard else "soft",
        compacted_child_count=int(compacted_child_count),
    )
    if not math.isfinite(card.t_start) or not math.isfinite(card.t_end):
        raise ValueError("parent span must be finite")
    card.serializable()
    return card
