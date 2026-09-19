"""Versioned prompts used by dense and sparse TimeLens refinement."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Mapping


OFFICIAL_CROP_VERSION = "official_crop_v1"
SPARSE_LOCAL_VERSION = "sparse_local_v1"

_OFFICIAL = (
    "Please find the visual event described by the sentence '{query}', determining its starting and ending times. "
    "The format should be: 'The event happens in <start time> - <end time> seconds'."
)
_SPARSE = (
    "You are given sparse video frames that may be non-contiguous. The number before each frame is its global "
    "timestamp in seconds, not time relative to this clip. Locate the event described by: {query}\n"
    "Candidate window (a coarse search prior only): {candidate_start:.3f} - {candidate_end:.3f} seconds.\n"
    "Evidence-card summaries (may be incomplete): {card_summary}\n"
    "Return exactly one line in this format and nothing else: The event happens in x - y seconds"
)
OFFICIAL_CROP_TEMPLATE_SHA256 = hashlib.sha256(_OFFICIAL.encode("utf-8")).hexdigest()
SPARSE_LOCAL_TEMPLATE_SHA256 = hashlib.sha256(_SPARSE.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class PromptSpec:
    version: str
    text: str
    sha256: str


def _normalize_cards(cards: Any) -> str:
    if cards is None or cards == "":
        return "none"
    if isinstance(cards, str):
        return " ".join(cards.split())
    return json.dumps(cards, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def build_grounding_prompt(
    version: str,
    *,
    query: str,
    candidate: tuple[float, float] | None = None,
    card_summary: str | Mapping[str, Any] | None = None,
) -> PromptSpec:
    """Render a prompt. Ground truth is deliberately not accepted by this API."""
    if not query.strip():
        raise ValueError("query must not be empty")
    if version == OFFICIAL_CROP_VERSION:
        text = _OFFICIAL.format(query=query.strip())
    elif version == SPARSE_LOCAL_VERSION:
        if candidate is None or candidate[0] < 0 or candidate[0] >= candidate[1]:
            raise ValueError("sparse local prompt requires an ordered candidate window")
        text = _SPARSE.format(
            query=query.strip(), candidate_start=candidate[0], candidate_end=candidate[1],
            card_summary=_normalize_cards(card_summary),
        )
    else:
        raise ValueError(f"unknown prompt version: {version}")
    return PromptSpec(version, text, hashlib.sha256(text.encode("utf-8")).hexdigest())


def video_messages(prompt: PromptSpec) -> list[dict[str, Any]]:
    """Build the canonical single-video chat structure expected by TimeLens."""
    return [{"role": "user", "content": [
        {"type": "video", "video": "<in-memory-sparse-video>"},
        {"type": "text", "text": prompt.text},
    ]}]
