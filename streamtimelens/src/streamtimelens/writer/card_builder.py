"""Deterministic conversion from validated writer events to leaf cards."""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Sequence

from streamtimelens.memory.cards import EvidenceCard

from .parse import WriterParseResult
from .prompts import WriterPromptSpec
from .schema import WriterEvent, document_dict


def normalized_event_text(event: WriterEvent) -> str:
    """Frozen template shared with the card text embedder."""
    parts = [
        f"summary: {event.summary}",
        f"actors: {', '.join(event.actors) if event.actors else 'none'}",
        f"actions: {', '.join(event.actions) if event.actions else 'none'}",
        f"objects: {', '.join(event.objects) if event.objects else 'none'}",
        f"scene: {event.scene}",
    ]
    return " | ".join(parts)


def _local_sampling_interval(support: Sequence[float], target: float, segment_duration: float) -> float:
    ordered = sorted(set(float(item) for item in support))
    if len(ordered) == 1:
        return max(0.0, segment_duration)
    closest_index = min(range(len(ordered)), key=lambda index: abs(ordered[index] - target))
    gaps = []
    if closest_index:
        gaps.append(ordered[closest_index] - ordered[closest_index - 1])
    if closest_index + 1 < len(ordered):
        gaps.append(ordered[closest_index + 1] - ordered[closest_index])
    return min(gaps) if gaps else 0.0


def endpoint_uncertainty(
    endpoint_s: float, support: Sequence[float], *, segment_duration_s: float
) -> float:
    if not support:
        return float(segment_duration_s)
    distance = min(abs(float(endpoint_s) - float(item)) for item in support)
    interval = _local_sampling_interval(support, endpoint_s, segment_duration_s)
    return float(distance + interval / 2.0)


def _event_dict(event: WriterEvent) -> dict[str, Any]:
    if hasattr(event, "model_dump"):
        return event.model_dump()  # type: ignore[attr-defined,no-any-return]
    return event.dict()


def _stable_card_id(chunk_id: str, event: WriterEvent, ordinal: int) -> str:
    payload = {"chunk_id": chunk_id, "event": _event_dict(event), "ordinal": ordinal}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return "card-" + hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:24]


def build_evidence_cards(
    parse_result: WriterParseResult,
    *,
    source_chunk_id: str,
    segment: tuple[float, float],
    prompt: WriterPromptSpec,
    writer_revision: str,
    writer_provenance: dict[str, Any] | None = None,
    generated_tokens: int = 0,
    epsilon_s: float = 1e-6,
) -> list[EvidenceCard]:
    """Build one stable leaf card per event, sorted by global span."""
    if not source_chunk_id or generated_tokens < 0 or epsilon_s < 0:
        raise ValueError("card construction metadata is invalid")
    start, end = map(float, segment)
    ordered = sorted(
        parse_result.document.events,
        key=lambda event: (event.span[0], event.span[1], normalized_event_text(event)),
    )
    cards: list[EvidenceCard] = []
    for ordinal, event in enumerate(ordered):
        event_start, event_end = map(float, event.span)
        if event_start < start - epsilon_s or event_end > end + epsilon_s:
            raise ValueError("materially out-of-segment event cannot become a card")
        event_start = min(end, max(start, event_start))
        event_end = min(end, max(start, event_end))
        if event_start > event_end or not all(math.isfinite(item) for item in (event_start, event_end)):
            raise ValueError("invalid event span")
        support = sorted(set(float(item) for item in event.visual_support))
        card = EvidenceCard(
            id=_stable_card_id(source_chunk_id, event, ordinal), level=0,
            t_start=event_start, t_end=event_end, summary=event.summary,
            actors=list(event.actors), actions=list(event.actions), objects=list(event.objects),
            scene=event.scene, support_timestamps=support,
            left_uncertainty_s=endpoint_uncertainty(
                event_start, support, segment_duration_s=end - start,
            ),
            right_uncertainty_s=endpoint_uncertainty(
                event_end, support, segment_duration_s=end - start,
            ),
            source_chunk_ids=[source_chunk_id], phase=event.phase,
            normalized_text=normalized_event_text(event),
            writer_model_revision=writer_revision, prompt_version=prompt.version,
            prompt_hash=prompt.template_sha256,
            writer_provenance={
                **(writer_provenance or {}), "rendered_prompt_sha256": prompt.sha256,
                "parse_status": parse_result.status,
                "document_sha256": hashlib.sha256(json.dumps(
                    document_dict(parse_result.document), sort_keys=True, separators=(",", ":"),
                ).encode("utf-8")).hexdigest(),
            },
            generated_tokens=int(generated_tokens),
        )
        card.serializable()
        cards.append(card)
    return cards
