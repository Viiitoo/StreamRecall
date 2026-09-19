"""Hierarchy-aware deterministic maximal-marginal-relevance selection."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

from streamtimelens.observer.clip_encoder import deserialize_embedding, l2_normalize
from streamtimelens.retrieval.ranker import RankedCard


@dataclass(frozen=True)
class MMRChoice:
    card_id: str
    rank_score: float
    mmr_score: float
    semantic_redundancy: float
    temporal_redundancy: float


@dataclass(frozen=True)
class MMRResult:
    choices: tuple[MMRChoice, ...]
    prefixes: dict[int, tuple[str, ...]]


def temporal_iou(left: tuple[float, float], right: tuple[float, float]) -> float:
    intersection = max(0.0, min(left[1], right[1]) - max(left[0], right[0]))
    union = max(left[1], right[1]) - min(left[0], right[0])
    return intersection / union if union > 0 else 0.0


def _cosine(left: Any, right: Any) -> float:
    import numpy as np

    first, second = l2_normalize(left), l2_normalize(right)
    if first.shape != second.shape:
        raise ValueError("card embedding dimensions differ")
    return float(np.clip(np.dot(first, second), -1.0, 1.0))


def _relations(cards: dict[str, dict[str, Any]]) -> dict[str, set[str]]:
    related = {card_id: set() for card_id in cards}
    for parent_id, card in cards.items():
        pending = list(card.get("child_ids") or [])
        visited: set[str] = set()
        while pending:
            child_id = str(pending.pop())
            if child_id in visited:
                raise ValueError("snapshot card hierarchy contains a cycle")
            if child_id not in cards:
                raise ValueError("snapshot card hierarchy has a dangling child")
            visited.add(child_id)
            related[parent_id].add(child_id)
            related[child_id].add(parent_id)
            pending.extend(cards[child_id].get("child_ids") or [])
    return related


def hierarchy_mmr(
    ranked: Sequence[RankedCard],
    card_rows: Iterable[dict[str, Any]],
    *,
    semantic_penalty: float,
    temporal_penalty: float,
    prefix_sizes: tuple[int, ...] = (1, 5, 8),
) -> MMRResult:
    if any(not math.isfinite(value) or value < 0 for value in (semantic_penalty, temporal_penalty)):
        raise ValueError("MMR penalties must be finite and non-negative")
    if not prefix_sizes or any(value <= 0 for value in prefix_sizes):
        raise ValueError("MMR prefix sizes must be positive")
    row_list = list(card_rows)
    rows = {str(row.get("id", "")): row for row in row_list}
    if "" in rows or len(rows) != len(row_list):
        raise ValueError("card rows need unique non-empty IDs")
    if set(item.card_id for item in ranked) != set(rows):
        raise ValueError("ranked cards and snapshot rows differ")
    relations = _relations(rows)
    vectors = {
        card_id: deserialize_embedding(row["text_embedding"])
        for card_id, row in rows.items()
    }
    ranked_by_id = {item.card_id: item for item in ranked}
    remaining = set(ranked_by_id)
    choices: list[MMRChoice] = []
    max_count = min(max(prefix_sizes), len(remaining))
    while remaining and len(choices) < max_count:
        scored = []
        selected_ids = [choice.card_id for choice in choices]
        for card_id in remaining:
            item = ranked_by_id[card_id]
            semantic = max(
                (max(0.0, _cosine(vectors[card_id], vectors[chosen])) for chosen in selected_ids),
                default=0.0,
            )
            temporal = max(
                (temporal_iou(item.span, ranked_by_id[chosen].span) for chosen in selected_ids),
                default=0.0,
            )
            mmr_score = item.score - semantic_penalty * semantic - temporal_penalty * temporal
            scored.append((mmr_score, item.score, item.span, card_id, semantic, temporal))
        score, rank_score, _, selected, semantic, temporal = min(
            scored, key=lambda value: (-value[0], -value[1], value[2][0], value[2][1], value[3]),
        )
        choices.append(MMRChoice(selected, rank_score, score, semantic, temporal))
        remaining.remove(selected)
        remaining.difference_update(relations[selected])
    identifiers = tuple(choice.card_id for choice in choices)
    prefixes = {size: identifiers[:size] for size in prefix_sizes}
    return MMRResult(tuple(choices), prefixes)
