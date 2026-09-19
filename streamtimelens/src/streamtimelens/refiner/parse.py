"""Strict TimeLens output parsing; fallback belongs to the query pipeline."""

from __future__ import annotations

import sys
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal


WORK_ROOT = Path(__file__).resolve().parents[4]
TIMELENS_ROOT = WORK_ROOT / "third_party" / "TimeLens"
if str(TIMELENS_ROOT) not in sys.path:
    sys.path.insert(0, str(TIMELENS_ROOT))

from timelens.utils import extract_time as official_extract_time  # noqa: E402


ParseStatus = Literal["ok", "no_timestamp", "invalid_order", "out_of_bounds", "no_candidate_overlap"]
_SCIENTIFIC_NUMBER = re.compile(r"(?<![\w.])[-+]?\d+(?:\.\d+)?[eE][-+]?\d+")


@dataclass(frozen=True)
class RefinerParseResult:
    status: ParseStatus
    span: tuple[float, float] | None
    raw_spans: tuple[tuple[float, float], ...]
    multiple_span: bool
    reason: str | None = None

    @property
    def valid(self) -> bool:
        return self.status == "ok"


def parse_refiner_output(
    answer: str,
    *,
    t_q: float,
    candidate: tuple[float, float],
    extractor: Callable[[str], list[tuple[float, float]]] = official_extract_time,
) -> RefinerParseResult:
    """Validate only the first official timestamp pair without clamping/fallback."""
    normalized_answer = _SCIENTIFIC_NUMBER.sub(lambda match: format(float(match.group(0)), "f"), answer)
    spans = tuple((float(start), float(end)) for start, end in extractor(normalized_answer))
    multiple = len(spans) > 1
    if not spans:
        return RefinerParseResult("no_timestamp", None, spans, False, "official parser found no timestamp pair")
    start, end = spans[0]
    if start < 0 or start >= end:
        return RefinerParseResult("invalid_order", None, spans, multiple, "timestamp pair is negative or reversed")
    if end > t_q:
        return RefinerParseResult("out_of_bounds", None, spans, multiple, "timestamp pair exceeds query arrival")
    left, right = candidate
    if left >= right or max(start, left) >= min(end, right):
        return RefinerParseResult("no_candidate_overlap", None, spans, multiple, "timestamp pair does not overlap candidate")
    return RefinerParseResult("ok", (start, end), spans, multiple)
