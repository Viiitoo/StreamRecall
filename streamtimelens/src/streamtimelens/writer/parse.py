"""Local-only JSON recovery and semantic validation for writer output."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Literal, Sequence

from pydantic import ValidationError

from .schema import WriterDocument, fallback_document, validate_document


ParseStatus = Literal["valid", "repaired", "fallback"]


@dataclass(frozen=True)
class WriterParseResult:
    status: ParseStatus
    document: WriterDocument
    raw_output: str
    extracted_json: str | None = None
    error: str | None = None


def _first_json_object(text: str) -> str:
    start = text.find("{")
    if start < 0:
        raise ValueError("no JSON object found")
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        character = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
            continue
        if character == '"':
            in_string = True
        elif character == "{":
            depth += 1
        elif character == "}":
            depth -= 1
            if depth == 0:
                return text[start:index + 1]
    raise ValueError("incomplete JSON object")


def _repair_syntax(text: str) -> str:
    # Deliberately limited to a mechanical JSON grammar repair. No key or
    # semantic value is ever synthesized here.
    return re.sub(r",\s*([}\]])", r"\1", text)


def parse_writer_output(
    raw_output: str,
    *,
    segment: tuple[float, float],
    sampled_timestamps: Sequence[float],
    epsilon_s: float = 1e-6,
) -> WriterParseResult:
    """Parse without model calls; every invalid result becomes one fallback event."""
    raw = str(raw_output)
    try:
        extracted = _first_json_object(raw)
        repaired = _repair_syntax(extracted)
        payload = json.loads(repaired)
        document = validate_document(
            payload, segment=segment, sampled_timestamps=sampled_timestamps,
            epsilon_s=epsilon_s,
        )
        direct = raw.strip() == extracted and repaired == extracted
        return WriterParseResult("valid" if direct else "repaired", document, raw, repaired)
    except (ValueError, TypeError, json.JSONDecodeError, ValidationError) as exc:
        return WriterParseResult(
            "fallback", fallback_document(segment, sampled_timestamps), raw,
            error=f"{type(exc).__name__}: {exc}",
        )
