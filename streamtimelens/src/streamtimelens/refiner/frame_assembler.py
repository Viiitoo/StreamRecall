"""Assemble candidate boundary evidence into one sparse TimeLens input."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from streamtimelens.protocol.snapshot import SnapshotReader
from streamtimelens.refiner.timelens_adapter import PreparedSparseVideo, SparseFrame, prepare_sparse_video
from streamtimelens.retrieval.candidates import RetrievalCandidate


AssemblyStatus = Literal["ok", "NO_VISUAL_EVIDENCE"]


@dataclass(frozen=True)
class FrameAssembly:
    status: AssemblyStatus
    frame_refs: tuple[str, ...]
    prepared: PreparedSparseVideo | None
    diagnostics: dict[str, Any]


def _ordered_unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result = []
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result


def assemble_refinement_frames(
    snapshot: SnapshotReader,
    candidate: RetrievalCandidate,
    *,
    max_frames: int,
) -> FrameAssembly:
    """Keep both boundaries, then fill remaining slots by novelty and coverage."""
    if max_frames <= 0:
        raise ValueError("refinement frame cap must be positive")
    cards = {str(row["id"]): row for row in snapshot.read_cards()}
    metadata = snapshot.read_frame_metadata()
    selected_cards = [
        cards[card_id] for card_id in candidate.contributing_card_ids if card_id in cards
    ]
    left_refs: list[str] = []
    right_refs: list[str] = []
    internal_refs: list[str] = []
    representative_refs: list[str] = []
    for card in selected_cards:
        cache = card.get("boundary_cache") or {}
        left_refs.extend(map(str, cache.get("left_frame_ids") or []))
        right_refs.extend(map(str, cache.get("right_frame_ids") or []))
        internal_refs.extend(map(str, cache.get("internal_frame_ids") or []))
        representative_refs.extend(map(str, card.get("raw_ref_ids") or []))
    available = {
        frame_ref for frame_ref in _ordered_unique(
            left_refs + right_refs + internal_refs + representative_refs
        ) if frame_ref in metadata
    }
    if not available:
        return FrameAssembly(
            "NO_VISUAL_EVIDENCE", (), None,
            {"candidate_id": candidate.candidate_id, "reason": "no_available_raw_refs"},
        )
    left_order = sorted(
        available & set(left_refs),
        key=lambda ref: (abs(float(metadata[ref]["timestamp_s"]) - candidate.start_s), ref),
    )
    right_order = sorted(
        available & set(right_refs),
        key=lambda ref: (abs(float(metadata[ref]["timestamp_s"]) - candidate.end_s), ref),
    )
    retained: list[str] = []
    if left_order:
        retained.append(left_order[0])
    if max_frames >= 2 and right_order:
        retained.append(right_order[0])
    retained = _ordered_unique(retained)
    boundary_refs = set(left_refs) | set(right_refs)
    remainder = sorted(
        available - set(retained),
        key=lambda ref: (
            -float(metadata[ref].get("novelty", metadata[ref].get("novelty_score", 0.0))),
            -(ref in boundary_refs),
            float(metadata[ref]["timestamp_s"]),
            ref,
        ),
    )
    retained.extend(remainder[:max(0, max_frames - len(retained))])
    retained = sorted(
        _ordered_unique(retained),
        key=lambda ref: (
            float(metadata[ref]["timestamp_s"]), int(metadata[ref]["frame_index"]), ref,
        ),
    )
    import numpy as np
    from PIL import Image

    fps = float(snapshot.manifest.video_meta["original_fps"])
    total_frames = int(snapshot.manifest.video_meta["total_num_frames"])
    sparse_frames = []
    for frame_ref in retained:
        with Image.open(snapshot.frame_path(frame_ref)) as image:
            pixels = np.asarray(image.convert("RGB").copy())
        index = int(metadata[frame_ref]["frame_index"])
        sparse_frames.append(SparseFrame(index, index / fps, pixels))
    prepared = prepare_sparse_video(
        sparse_frames, original_fps=fps, total_num_frames=total_frames,
        k_frames=max_frames,
    )
    return FrameAssembly(
        "ok", tuple(retained), prepared,
        {
            "candidate_id": candidate.candidate_id,
            "available_count": len(available), "selected_count": len(retained),
            "left_boundary_retained": bool(set(retained) & set(left_order)),
            "right_boundary_retained": bool(set(retained) & set(right_order)),
            "timestamp_audit": list(prepared.timestamp_audit),
        },
    )
