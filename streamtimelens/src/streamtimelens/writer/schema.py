"""Strict, query-independent schema for one closed-segment writer call."""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Literal, Sequence, Tuple

from pydantic import BaseModel, StrictStr, root_validator, validator


class _StrictModel(BaseModel):
    class Config:
        extra = "forbid"
        allow_mutation = False


class WriterEvent(_StrictModel):
    summary: StrictStr
    actors: List[StrictStr]
    actions: List[StrictStr]
    objects: List[StrictStr]
    scene: StrictStr
    span: Tuple[float, float]
    phase: Literal["complete", "ongoing"]
    visual_support: List[float]

    @validator("summary", "scene", allow_reuse=True)
    def _non_empty_text(cls, value: str) -> str:
        normalized = " ".join(value.split())
        if not normalized:
            raise ValueError("event text fields must not be empty")
        return normalized

    @validator("actors", "actions", "objects", allow_reuse=True)
    def _normalized_terms(cls, value: List[str]) -> List[str]:
        result = [" ".join(str(item).split()) for item in value]
        if any(not item for item in result):
            raise ValueError("structured event terms must not be empty")
        return result

    @validator("span", pre=True, allow_reuse=True)
    def _numeric_span(cls, value: Any) -> Any:
        if (not isinstance(value, (list, tuple)) or len(value) != 2 or
                any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in value)):
            raise ValueError("event span must contain two numeric timestamps")
        return value

    @validator("span", allow_reuse=True)
    def _ordered_span(cls, value: Tuple[float, float]) -> Tuple[float, float]:
        start, end = map(float, value)
        if not math.isfinite(start) or not math.isfinite(end) or start > end:
            raise ValueError("event span must be finite and ordered")
        return start, end

    @validator("visual_support", pre=True, allow_reuse=True)
    def _numeric_support(cls, value: Any) -> Any:
        if (not isinstance(value, list) or
                any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in value)):
            raise ValueError("visual_support must be a numeric timestamp list")
        return value

    @validator("visual_support", allow_reuse=True)
    def _finite_support(cls, value: List[float]) -> List[float]:
        result = [float(item) for item in value]
        if not result or any(not math.isfinite(item) for item in result):
            raise ValueError("visual_support must contain finite timestamps")
        return result


class WriterDocument(_StrictModel):
    segment: Tuple[float, float]
    events: List[WriterEvent]

    @validator("segment", pre=True, allow_reuse=True)
    def _numeric_segment(cls, value: Any) -> Any:
        if (not isinstance(value, (list, tuple)) or len(value) != 2 or
                any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in value)):
            raise ValueError("segment must contain two numeric timestamps")
        return value

    @validator("segment", allow_reuse=True)
    def _ordered_segment(cls, value: Tuple[float, float]) -> Tuple[float, float]:
        start, end = map(float, value)
        if not math.isfinite(start) or not math.isfinite(end) or start >= end:
            raise ValueError("segment must be finite and have positive duration")
        return start, end

    @root_validator(allow_reuse=True)
    def _events_stay_inside_segment(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        segment = values.get("segment")
        if segment is None:
            return values
        for event in values.get("events") or []:
            if event.span[0] < segment[0] - 1e-6 or event.span[1] > segment[1] + 1e-6:
                raise ValueError("event span lies outside the declared segment")
        return values


def validate_document(
    value: Any,
    *,
    segment: tuple[float, float],
    sampled_timestamps: Sequence[float],
    epsilon_s: float = 1e-6,
) -> WriterDocument:
    """Validate syntax and facts observable in the actual writer input.

    The epsilon only tolerates floating-point serialization error. It is not a
    semantic clamp and cannot turn a materially out-of-range result into one.
    """
    if epsilon_s < 0:
        raise ValueError("epsilon_s must be non-negative")
    actual_start, actual_end = map(float, segment)
    timestamps = tuple(float(item) for item in sampled_timestamps)
    if (not math.isfinite(actual_start) or not math.isfinite(actual_end) or
            actual_start >= actual_end or not timestamps or
            any(not math.isfinite(item) for item in timestamps)):
        raise ValueError("actual closed segment and sampled timestamps are required")
    document = WriterDocument.parse_obj(value)
    if (abs(document.segment[0] - actual_start) > epsilon_s or
            abs(document.segment[1] - actual_end) > epsilon_s):
        raise ValueError("writer-declared segment disagrees with its input segment")
    for event in document.events:
        if event.span[0] < actual_start - epsilon_s or event.span[1] > actual_end + epsilon_s:
            raise ValueError("event span lies outside the actual segment")
        for support in event.visual_support:
            if support < actual_start - epsilon_s or support > actual_end + epsilon_s:
                raise ValueError("support timestamp lies outside the actual segment")
            if not any(abs(support - observed) <= epsilon_s for observed in timestamps):
                raise ValueError("support timestamp was not sampled by the reservoir")
    return document


def document_dict(document: WriterDocument) -> dict[str, Any]:
    """Pydantic v1/v2-neutral JSON-ready representation."""
    if hasattr(document, "model_dump"):
        return document.model_dump()  # type: ignore[attr-defined,no-any-return]
    return document.dict()


def fallback_document(
    segment: tuple[float, float], sampled_timestamps: Iterable[float]
) -> WriterDocument:
    timestamps = sorted(set(float(item) for item in sampled_timestamps))
    if not timestamps:
        raise ValueError("fallback requires actual reservoir timestamps")
    return WriterDocument.parse_obj({
        "segment": [float(segment[0]), float(segment[1])],
        "events": [{
            "summary": "unknown_event", "actors": [], "actions": [], "objects": [],
            "scene": "unknown", "span": [float(segment[0]), float(segment[1])],
            "phase": "ongoing", "visual_support": timestamps,
        }],
    })
