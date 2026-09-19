"""Versioned prompt for deterministic closed-segment evidence writing."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Sequence


WRITER_PROMPT_VERSION = "structured_evidence_v1"
WRITER_MAX_NEW_TOKENS = 256
WRITER_TEMPERATURE = 0

_TEMPLATE = """You are an evidence recorder for a closed video segment.
Describe only directly visible events in the supplied frames. Do not use or infer any user query, intent, goal, cause, or event outside the segment. All timestamps are global video seconds.

Actual closed segment: [{segment_start:.6f}, {segment_end:.6f}]
Allowed visual_support timestamps (copy values only from this list): {timestamps}

Return exactly one JSON object and no Markdown. It must have this shape:
{{"segment":[start,end],"events":[{{"summary":"visible event","actors":[],"actions":[],"objects":[],"scene":"unknown","span":[start,end],"phase":"complete|ongoing","visual_support":[sampled_timestamp]}}]}}

The events array may be empty and may contain multiple atomic events. Use phase "ongoing" when an event crosses a segment boundary. Every span must stay inside the actual segment; never invent a timestamp."""


@dataclass(frozen=True)
class WriterPromptSpec:
    version: str
    text: str
    sha256: str
    template_sha256: str
    max_new_tokens: int = WRITER_MAX_NEW_TOKENS
    temperature: int = WRITER_TEMPERATURE


def build_writer_prompt(
    *, segment: tuple[float, float], sampled_timestamps: Sequence[float]
) -> WriterPromptSpec:
    start, end = map(float, segment)
    timestamps = tuple(sorted(set(float(item) for item in sampled_timestamps)))
    if start < 0 or start >= end or not timestamps:
        raise ValueError("writer prompt requires a closed segment and sampled timestamps")
    if timestamps[0] < start - 1e-6 or timestamps[-1] > end + 1e-6:
        raise ValueError("writer prompt timestamps must lie inside the segment")
    rendered = _TEMPLATE.format(
        segment_start=start, segment_end=end,
        timestamps="[" + ", ".join(f"{item:.6f}" for item in timestamps) + "]",
    )
    return WriterPromptSpec(
        WRITER_PROMPT_VERSION, rendered,
        hashlib.sha256(rendered.encode("utf-8")).hexdigest(),
        hashlib.sha256(_TEMPLATE.encode("utf-8")).hexdigest(),
    )


def writer_video_messages(prompt: WriterPromptSpec) -> list[dict[str, Any]]:
    return [{"role": "user", "content": [
        {"type": "video", "video": "<in-memory-sparse-video>"},
        {"type": "text", "text": prompt.text},
    ]}]
