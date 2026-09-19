"""Frozen hybrid retrieval over query-visible evidence cards."""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any, Iterable

from streamtimelens.observer.clip_encoder import deserialize_embedding, l2_normalize
from streamtimelens.protocol.snapshot import SnapshotReader
from streamtimelens.protocol.types import Prediction


def _tokens(value: str) -> Counter[str]:
    return Counter(re.findall(r"[\w]+", value.lower()))


def _cosine(left: Counter[str], right: Counter[str]) -> float:
    numerator = sum(left[key] * right[key] for key in left.keys() & right.keys())
    denominator = math.sqrt(sum(value * value for value in left.values()) * sum(value * value for value in right.values()))
    return numerator / denominator if denominator else 0.0


def card_text(card: dict) -> str:
    return " ".join([str(card.get("summary", "")), *map(str, card.get("actors", [])),
                     *map(str, card.get("actions", [])), *map(str, card.get("objects", [])), str(card.get("scene", ""))])


@dataclass(frozen=True)
class RankWeights:
    text: float = 1.0
    visual: float = 0.0
    boundary_bonus: float = 0.10
    parent_penalty: float = 0.05

    def __post_init__(self) -> None:
        values = (self.text, self.visual, self.boundary_bonus, self.parent_penalty)
        if any(not math.isfinite(value) or value < 0 for value in values):
            raise ValueError("rank weights must be finite and non-negative")
        if self.text + self.visual <= 0:
            raise ValueError("at least one semantic rank weight must be positive")


@dataclass(frozen=True)
class RankedCard:
    card_id: str
    score: float
    span: tuple[float, float]
    level: int
    diagnostics: dict[str, float]


def _vector_cosine(left: Any, right: Any) -> float:
    import numpy as np

    first = l2_normalize(left)
    second = l2_normalize(right)
    if first.shape != second.shape:
        raise ValueError("query and persisted embeddings have different dimensions")
    return float(np.clip(np.dot(first, second), -1.0, 1.0))


def _boundary_completeness(card: dict[str, Any]) -> float:
    boundary = card.get("boundary_cache") or {}
    if bool(boundary.get("both_hit")):
        return 1.0
    if bool(boundary.get("left_hit")) or bool(boundary.get("right_hit")):
        return 0.5
    statuses = card.get("raw_ref_status") or {}
    if card.get("raw_ref_ids") and any(value == "available" for value in statuses.values()):
        return 0.25
    return 0.0


def hybrid_rank_cards(
    query_embedding: Any,
    cards: Iterable[dict[str, Any]],
    *,
    weights: RankWeights,
    query_visual_embedding: Any | None = None,
    upper_bound_s: float | None = None,
) -> list[RankedCard]:
    """Rank once and retain each frozen score component for diagnostics."""
    ranked: list[RankedCard] = []
    for card in cards:
        card_id = str(card.get("id", ""))
        embedding = card.get("text_embedding")
        if not card_id or not isinstance(embedding, dict):
            raise ValueError("every ranked card needs an ID and text embedding")
        start, end = float(card["t_start"]), float(card["t_end"])
        if start < 0 or end <= start or (upper_bound_s is not None and end > upper_bound_s + 1e-6):
            raise ValueError(f"card span is outside query-visible history: {card_id}")
        text_score = _vector_cosine(query_embedding, deserialize_embedding(embedding))
        visual_score = 0.0
        if query_visual_embedding is not None and card.get("visual_centroid") is not None:
            visual_score = _vector_cosine(
                query_visual_embedding, deserialize_embedding(card["visual_centroid"]),
            )
        boundary = _boundary_completeness(card)
        level = int(card.get("level", 0))
        if level < 0:
            raise ValueError("card level cannot be negative")
        text_component = weights.text * text_score
        visual_component = weights.visual * visual_score
        boundary_component = weights.boundary_bonus * boundary
        level_component = weights.parent_penalty * level
        total = text_component + visual_component + boundary_component - level_component
        diagnostics = {
            "text_cosine": text_score,
            "visual_cosine": visual_score,
            "raw_boundary_completeness": boundary,
            "text_component": text_component,
            "visual_component": visual_component,
            "boundary_component": boundary_component,
            "parent_penalty": level_component,
            "total": total,
        }
        ranked.append(RankedCard(card_id, total, (start, end), level, diagnostics))
    return sorted(ranked, key=lambda item: (-item.score, item.span[0], item.span[1], item.card_id))


def locate_lexically(query: str, snapshot: SnapshotReader, *, minimum_score: float = 0.01) -> Prediction:
    """Return a coarse interval; P2 replaces this with embedding/MMR/refinement."""
    query_tokens = _tokens(query)
    ranked = sorted(((_cosine(query_tokens, _tokens(card_text(card))), card) for card in snapshot.read_cards()),
                    key=lambda item: (-item[0], str(item[1].get("id", ""))))
    if not ranked or ranked[0][0] < minimum_score:
        return Prediction(None, None, 0.0, "NOT_FOUND")
    score, card = ranked[0]
    start, end = float(card["t_start"]), float(card["t_end"])
    if not (0 <= start < end <= snapshot.manifest.t_q + 1e-6):
        return Prediction(start, end, score, "fallback", (str(card["id"]),))
    return Prediction(start, end, score, "ok", (str(card["id"]),))
