"""Versioned, hash-stable Hybrid V3 multi-candidate prompt."""

from __future__ import annotations

import hashlib
from typing import Any, Mapping, Sequence

from streamtimelens.refiner.prompts import PromptSpec
from streamtimelens.retrieval.frame_candidates import FrameCandidate


SPARSE_MULTICANDIDATE_VERSION = "sparse_multicandidate_v1"
SPARSE_MULTICANDIDATE_KEEP_VERSION = "sparse_multicandidate_keep_v2"
SPARSE_MULTICANDIDATE_ADAPTIVE_VERSION = "sparse_multicandidate_adaptive_v3"
SCORE_LABEL_ORDER = "score_v1"
REVERSE_SCORE_LABEL_ORDER = "reverse_score_v1"

_TEMPLATE_V1 = """You are given sparse video frames that may be non-contiguous. Each frame retains its global timestamp from the original video.
Find the event described by: {query}
The visible history ends at {t_q:.3f} seconds. Never return a time after it.
The candidate windows below are coarse CLIP search priors. Compare all candidates globally, choose the best event, and tighten its boundaries within an overlapping candidate envelope.
{candidate_table}
If the event is visible, return exactly one line: Candidate Cxx; The event happens in x - y seconds
If the event is absent from visible history, return exactly: NOT_FOUND"""
_TEMPLATE_V2 = """You are given sparse video frames that may be non-contiguous. Each frame retains its global timestamp from the original video.
Find the event described by: {query}
The visible history ends at {t_q:.3f} seconds. Never return a time after it.
The candidate windows below are coarse CLIP search priors. Compare all candidates globally and choose the best event.
{candidate_table}
If both event boundaries are visually supported, refine them inside the selected candidate and return exactly: REFINE Cxx; The event happens in x - y seconds
If the event is visible but sparse evidence does not support tighter boundaries, preserve the selected candidate and return exactly: KEEP Cxx
If the event is absent from visible history, return exactly: NOT_FOUND"""
_TEMPLATE_V3 = """You are given sparse video frames that may be non-contiguous. Each frame retains its global timestamp from the original video.
Find the event described by: {query}
The visible history ends at {t_q:.3f} seconds. Never return a time after it.
The candidate windows below are coarse CLIP search priors. Compare all candidates globally and choose the best event.
There are {frame_count} unique sparse input frames. Apply this query-visible safety rule exactly:
{safety_instruction}
{candidate_table}
If the event is absent from visible history, return exactly: NOT_FOUND"""
_TEMPLATES = {
    SPARSE_MULTICANDIDATE_VERSION: _TEMPLATE_V1,
    SPARSE_MULTICANDIDATE_KEEP_VERSION: _TEMPLATE_V2,
    SPARSE_MULTICANDIDATE_ADAPTIVE_VERSION: _TEMPLATE_V3,
}
SPARSE_MULTICANDIDATE_TEMPLATE_SHA256 = hashlib.sha256(_TEMPLATE_V1.encode("utf-8")).hexdigest()


def multicandidate_template_sha256(version: str) -> str:
    try:
        template = _TEMPLATES[version]
    except KeyError as exc:
        raise ValueError(f"unknown multi-candidate prompt version: {version}") from exc
    return hashlib.sha256(template.encode("utf-8")).hexdigest()


def candidate_labels(
    candidates: Sequence[FrameCandidate], *, order: str = SCORE_LABEL_ORDER,
) -> dict[str, str]:
    ordered = list(candidates)
    if order == REVERSE_SCORE_LABEL_ORDER:
        ordered.reverse()
    elif order != SCORE_LABEL_ORDER:
        raise ValueError(f"unknown candidate label order: {order}")
    return {
        f"C{index:02d}": candidate.candidate_id
        for index, candidate in enumerate(ordered, 1)
    }


def build_multicandidate_prompt(
    *,
    version: str,
    query: str,
    candidates: Sequence[FrameCandidate],
    candidate_frame_refs: Mapping[str, Sequence[str]],
    frame_metadata: Mapping[str, Mapping[str, Any]],
    t_q: float,
    candidate_label_order: str = SCORE_LABEL_ORDER,
    selected_frame_count: int | None = None,
    sparse_keep_max_frames: int = 7,
) -> PromptSpec:
    """Render the only prompt accepted by the Hybrid V3 readout."""
    if version not in _TEMPLATES:
        raise ValueError(f"unknown multi-candidate prompt version: {version}")
    if not query.strip() or not candidates or t_q <= 0:
        raise ValueError("multi-candidate prompt needs query, candidates, and visible history")
    if sparse_keep_max_frames < 0:
        raise ValueError("sparse KEEP threshold must be non-negative")
    if version == SPARSE_MULTICANDIDATE_ADAPTIVE_VERSION and (
        selected_frame_count is None or selected_frame_count <= 0
    ):
        raise ValueError("adaptive prompt needs a positive selected frame count")
    labels = candidate_labels(candidates, order=candidate_label_order)
    rows = []
    for label, candidate_id in labels.items():
        candidate = next(item for item in candidates if item.candidate_id == candidate_id)
        refs = tuple(candidate_frame_refs.get(candidate_id, ()))
        timestamps = [float(frame_metadata[ref]["timestamp_s"]) for ref in refs]
        if any(timestamp > t_q + 1e-9 for timestamp in timestamps):
            raise ValueError("prompt cannot include a future frame")
        rendered_times = ", ".join(f"{timestamp:.3f}" for timestamp in timestamps) or "none"
        rows.append(
            f"{label} | coarse={candidate.start_s:.3f}-{candidate.end_s:.3f}s "
            f"| clip_score={candidate.score:.6f} | frame_timestamps={rendered_times}s"
        )
    sparse = selected_frame_count is not None and selected_frame_count <= sparse_keep_max_frames
    safety_instruction = (
        f"SPARSE mode ({selected_frame_count} <= {sparse_keep_max_frames}): if the event is visible, "
        "you MUST preserve its selected candidate and return exactly: KEEP Cxx. "
        "REFINE is forbidden in SPARSE mode."
        if sparse else
        f"DENSE mode ({selected_frame_count} > {sparse_keep_max_frames}): if the event is visible, "
        "KEEP is forbidden. You MUST refine both visually supported boundaries inside the selected "
        "candidate. Do not output a candidate label. Return exactly: REFINE; x - y seconds"
    )
    text = _TEMPLATES[version].format(
        query=" ".join(query.strip().split()), t_q=t_q,
        frame_count=selected_frame_count, safety_instruction=safety_instruction,
        candidate_table="\n".join(rows),
    )
    return PromptSpec(
        version,
        text,
        hashlib.sha256(text.encode("utf-8")).hexdigest(),
    )
