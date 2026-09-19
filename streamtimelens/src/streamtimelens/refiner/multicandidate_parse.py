"""Strict single-pass parsing for Hybrid V3 model output."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Literal, Mapping, Sequence

from streamtimelens.refiner.parse import official_extract_time


MultiCandidateParseStatus = Literal[
    "ok", "keep", "not_found", "invalid_format", "missing_candidate_id",
    "multiple_candidate_ids", "unknown_candidate_id", "no_timestamp",
    "multiple_spans", "invalid_order", "out_of_bounds",
    "no_candidate_overlap", "outside_candidate_envelope", "invalid_frame_reference",
]
_LABEL = re.compile(r"\bC\d{2}\b")
_NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
_STRICT_LINE = re.compile(
    rf"Candidate (?P<label>C\d{{2}}); The event happens in "
    rf"(?P<start>{_NUMBER}) - (?P<end>{_NUMBER}) seconds"
)
_REFINE_LINE = re.compile(
    rf"REFINE (?P<label>C\d{{2}}); The event happens in "
    rf"(?P<start>{_NUMBER}) - (?P<end>{_NUMBER}) seconds"
)
_ADAPTIVE_REFINE_LINE = re.compile(
    rf"REFINE;[ \t]*(?P<start>{_NUMBER})[ \t]*-[ \t]*"
    rf"(?P<end>{_NUMBER})[ \t]+seconds"
)
_KEEP_LINE = re.compile(r"KEEP (?P<label>C\d{2})")
_SCIENTIFIC_NUMBER = re.compile(r"(?<![\w.])[-+]?\d+(?:\.\d+)?[eE][-+]?\d+")
_CONTAINMENT_TOLERANCE_S = 5e-4 + 1e-9


@dataclass(frozen=True)
class MultiCandidateParseResult:
    status: MultiCandidateParseStatus
    span: tuple[float, float] | None
    selected_label: str | None
    selected_candidate_id: str | None
    raw_spans: tuple[tuple[float, float], ...]
    reason: str | None = None

    @property
    def valid(self) -> bool:
        return self.status in ("ok", "keep", "not_found")

    @property
    def not_found(self) -> bool:
        return self.status == "not_found"


def _failed(
    status: MultiCandidateParseStatus,
    reason: str,
    *,
    label: str | None = None,
    candidate_id: str | None = None,
    spans: tuple[tuple[float, float], ...] = (),
) -> MultiCandidateParseResult:
    return MultiCandidateParseResult(status, None, label, candidate_id, spans, reason)


def parse_multicandidate_output(
    answer: str,
    *,
    t_q: float,
    candidate_labels: Mapping[str, str],
    candidate_spans: Mapping[str, tuple[float, float]],
    candidate_frame_refs: Mapping[str, Sequence[str]],
    available_frame_refs: Sequence[str],
    prompt_version: str = "sparse_multicandidate_v1",
    extractor: Callable[[str], list[tuple[float, float]]] = official_extract_time,
) -> MultiCandidateParseResult:
    """Parse once and reject every contract violation without clamping."""
    stripped = answer.strip()
    if stripped == "NOT_FOUND":
        return MultiCandidateParseResult("not_found", None, None, None, ())
    keep_enabled = prompt_version in (
        "sparse_multicandidate_keep_v2", "sparse_multicandidate_adaptive_v3",
    )
    if prompt_version not in (
        "sparse_multicandidate_v1", "sparse_multicandidate_keep_v2",
        "sparse_multicandidate_adaptive_v3",
    ):
        raise ValueError(f"unknown multi-candidate prompt version: {prompt_version}")
    adaptive_match = (
        _ADAPTIVE_REFINE_LINE.fullmatch(stripped)
        if prompt_version == "sparse_multicandidate_adaptive_v3" else None
    )
    labels = _LABEL.findall(stripped)
    unique_labels = tuple(dict.fromkeys(labels))
    if not unique_labels and adaptive_match is None:
        return _failed("missing_candidate_id", "model output has no candidate ID")
    if len(unique_labels) > 1:
        return _failed(
            "multiple_candidate_ids", "model output names inconsistent candidate IDs",
            label=unique_labels[0],
        )
    label = unique_labels[0] if unique_labels else None
    candidate_id = candidate_labels.get(label) if label is not None else None
    if label is not None and (candidate_id is None or candidate_id not in candidate_spans):
        return _failed("unknown_candidate_id", "model output names an unknown candidate", label=label)
    if keep_enabled and _KEEP_LINE.fullmatch(stripped):
        refs = tuple(candidate_frame_refs.get(candidate_id, ()))
        allowed = set(available_frame_refs)
        if not refs or any(ref not in allowed for ref in refs):
            return _failed(
                "invalid_frame_reference", "selected candidate evidence is not snapshot-owned",
                label=label, candidate_id=candidate_id,
            )
        return MultiCandidateParseResult(
            "keep", candidate_spans[candidate_id], label, candidate_id, (),
        )
    strict_match = (
        adaptive_match
        if prompt_version == "sparse_multicandidate_adaptive_v3"
        else _REFINE_LINE.fullmatch(stripped)
        if keep_enabled else _STRICT_LINE.fullmatch(stripped)
    )
    normalized = _SCIENTIFIC_NUMBER.sub(lambda match: format(float(match.group(0)), "f"), stripped)
    spans = tuple((float(start), float(end)) for start, end in extractor(normalized))
    if not spans:
        return _failed(
            "no_timestamp", "official parser found no timestamp pair",
            label=label, candidate_id=candidate_id,
        )
    if len(spans) > 1:
        return _failed(
            "multiple_spans", "strict output contains more than one timestamp pair",
            label=label, candidate_id=candidate_id, spans=spans,
        )
    if strict_match is None:
        return _failed(
            "invalid_format", "output does not match the required single-line format",
            label=label, candidate_id=candidate_id, spans=spans,
        )
    raw_start = float(strict_match.group("start"))
    raw_end = float(strict_match.group("end"))
    if raw_start < 0 or raw_start >= raw_end:
        return _failed(
            "invalid_order", "timestamp pair is negative or reversed",
            label=label, candidate_id=candidate_id, spans=spans,
        )
    start, end = spans[0]
    if abs(start - raw_start) > 1e-9 or abs(end - raw_end) > 1e-9:
        return _failed(
            "invalid_format", "official parser disagrees with the strict timestamp pair",
            label=label, candidate_id=candidate_id, spans=spans,
        )
    if start < 0 or start >= end:
        return _failed(
            "invalid_order", "timestamp pair is negative or reversed",
            label=label, candidate_id=candidate_id, spans=spans,
        )
    if end > t_q:
        return _failed(
            "out_of_bounds", "timestamp pair exceeds query arrival",
            label=label, candidate_id=candidate_id, spans=spans,
        )
    if adaptive_match is not None:
        candidate_id = next((
            item_id for item_id, (left, right) in candidate_spans.items()
            if start >= left - _CONTAINMENT_TOLERANCE_S
            and end <= right + _CONTAINMENT_TOLERANCE_S
        ), None)
        if candidate_id is None:
            return _failed(
                "outside_candidate_envelope",
                "timestamp pair is outside every candidate envelope",
                spans=spans,
            )
        label = next(
            (item_label for item_label, item_id in candidate_labels.items() if item_id == candidate_id),
            None,
        )
        if label is None:
            return _failed(
                "unknown_candidate_id", "mapped candidate has no prompt label",
                candidate_id=candidate_id, spans=spans,
            )
    assert candidate_id is not None
    left, right = candidate_spans[candidate_id]
    if left >= right or max(start, left) >= min(end, right):
        return _failed(
            "no_candidate_overlap", "timestamp pair does not overlap selected candidate",
            label=label, candidate_id=candidate_id, spans=spans,
        )
    if keep_enabled and (
        start < left - _CONTAINMENT_TOLERANCE_S
        or end > right + _CONTAINMENT_TOLERANCE_S
    ):
        return _failed(
            "outside_candidate_envelope",
            "timestamp pair is outside the selected candidate envelope",
            label=label, candidate_id=candidate_id, spans=spans,
        )
    refs = tuple(candidate_frame_refs.get(candidate_id, ()))
    allowed = set(available_frame_refs)
    if not refs or any(ref not in allowed for ref in refs):
        return _failed(
            "invalid_frame_reference", "selected candidate evidence is not snapshot-owned",
            label=label, candidate_id=candidate_id, spans=spans,
        )
    return MultiCandidateParseResult(
        "ok", (start, end), label, candidate_id, spans,
    )
