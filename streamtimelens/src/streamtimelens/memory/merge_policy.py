"""Versioned heap of deterministic, temporally adjacent merge candidates."""

from __future__ import annotations

import heapq
import math
import re
from dataclasses import dataclass
from typing import Iterable, Sequence

from streamtimelens.memory.card_store import CardStore
from streamtimelens.memory.cards import EvidenceCard
from streamtimelens.observer.clip_encoder import deserialize_embedding


@dataclass(frozen=True)
class MergeReason:
    semantic_similarity: float
    temporal_gap_s: float
    boundary_importance: float
    semantic_term: float
    gap_penalty: float
    boundary_penalty: float
    score: float


@dataclass(frozen=True)
class MergeCandidate:
    left_id: str
    right_id: str
    reason: MergeReason


def _lexical_similarity(left: str, right: str) -> float:
    left_tokens = set(re.findall(r"[\w]+", left.lower()))
    right_tokens = set(re.findall(r"[\w]+", right.lower()))
    union = left_tokens | right_tokens
    return len(left_tokens & right_tokens) / len(union) if union else 0.0


def semantic_similarity(left: EvidenceCard, right: EvidenceCard) -> float:
    if left.text_embedding is not None and right.text_embedding is not None:
        import numpy as np

        first = deserialize_embedding(left.text_embedding)
        second = deserialize_embedding(right.text_embedding)
        if first.shape != second.shape:
            raise ValueError("adjacent card embedding dimensions do not match")
        return float(np.clip(np.dot(first, second), -1.0, 1.0))
    return _lexical_similarity(
        left.normalized_text or left.summary,
        right.normalized_text or right.summary,
    )


def boundary_importance(left: EvidenceCard, right: EvidenceCard) -> float:
    uncertainty = max(0.0, left.right_uncertainty_s) + max(0.0, right.left_uncertainty_s)
    certainty = 1.0 / (1.0 + uncertainty)
    left_hit = bool(left.boundary_cache.get("right_hit"))
    right_hit = bool(right.boundary_cache.get("left_hit"))
    raw_support = (float(left_hit) + float(right_hit)) / 2.0
    phase_boundary = float(left.phase == "complete" and right.phase == "complete")
    return min(1.0, 0.50 * certainty + 0.35 * raw_support + 0.15 * phase_boundary)


class MergeCandidateQueue:
    """A lazy-invalidated heap; old entries never become valid again."""

    def __init__(
        self, *, semantic_weight: float = 1.0, gap_weight: float = 0.35,
        boundary_weight: float = 0.35,
    ) -> None:
        weights = (semantic_weight, gap_weight, boundary_weight)
        if any(not math.isfinite(value) or value < 0 for value in weights):
            raise ValueError("merge weights must be finite and non-negative")
        self.semantic_weight = float(semantic_weight)
        self.gap_weight = float(gap_weight)
        self.boundary_weight = float(boundary_weight)
        self._heap: list[tuple[float, str, str, int, int, int, MergeCandidate]] = []
        self._versions: dict[str, int] = {}
        self._epoch = 0

    def invalidate(self, card_id: str) -> None:
        self._versions[card_id] = self._versions.get(card_id, 0) + 1

    def _candidate(self, left: EvidenceCard, right: EvidenceCard) -> MergeCandidate:
        if CardStore.key(left) > CardStore.key(right):
            raise ValueError("merge candidates must be in temporal order")
        similarity = semantic_similarity(left, right)
        gap = max(0.0, right.t_start - left.t_end)
        importance = boundary_importance(left, right)
        semantic_term = self.semantic_weight * similarity
        gap_penalty = self.gap_weight * (gap / (1.0 + gap))
        boundary_penalty = self.boundary_weight * importance
        score = semantic_term - gap_penalty - boundary_penalty
        return MergeCandidate(left.id, right.id, MergeReason(
            similarity, gap, importance, semantic_term, gap_penalty,
            boundary_penalty, score,
        ))

    def rebuild(self, store: CardStore, ordered_ids: Sequence[str] | None = None) -> None:
        ids = tuple(ordered_ids if ordered_ids is not None else store.ids)
        if len(ids) != len(set(ids)) or any(card_id not in store for card_id in ids):
            raise ValueError("merge queue received invalid ordered card IDs")
        self._epoch += 1
        for left_id, right_id in zip(ids, ids[1:]):
            left, right = store.get(left_id), store.get(right_id)
            candidate = self._candidate(left, right)
            heapq.heappush(self._heap, (
                -candidate.reason.score, left_id, right_id, self._epoch,
                self._versions.get(left_id, 0), self._versions.get(right_id, 0),
                candidate,
            ))

    def pop_best(
        self, store: CardStore, ordered_ids: Iterable[str] | None = None,
    ) -> MergeCandidate | None:
        ids = tuple(ordered_ids if ordered_ids is not None else store.ids)
        positions = {card_id: index for index, card_id in enumerate(ids)}
        while self._heap:
            _, left_id, right_id, epoch, left_version, right_version, candidate = heapq.heappop(self._heap)
            if epoch != self._epoch:
                continue
            if self._versions.get(left_id, 0) != left_version or self._versions.get(right_id, 0) != right_version:
                continue
            if left_id not in store or right_id not in store:
                continue
            if positions.get(right_id) != positions.get(left_id, -2) + 1:
                continue
            return candidate
        return None

    @property
    def heap_entries(self) -> int:
        return len(self._heap)
