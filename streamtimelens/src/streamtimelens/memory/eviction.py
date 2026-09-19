"""Delay-query-safe card utility and coverage-preserving eviction."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Iterable

from streamtimelens.memory.card_store import CardStore
from streamtimelens.memory.cards import EvidenceCard
from streamtimelens.memory.merge_policy import semantic_similarity

if TYPE_CHECKING:  # pragma: no cover
    from streamtimelens.memory.forest import EventForest


@dataclass(frozen=True)
class UtilityWeights:
    novelty: float = 0.30
    boundary: float = 0.30
    inverse_density: float = 0.25
    has_raw: float = 0.15

    def __post_init__(self) -> None:
        values = (self.novelty, self.boundary, self.inverse_density, self.has_raw)
        if any(not math.isfinite(value) or value < 0 for value in values) or sum(values) <= 0:
            raise ValueError("utility weights must be finite, non-negative, and non-zero")


@dataclass(frozen=True)
class UtilityScore:
    novelty: float
    boundary: float
    inverse_density: float
    has_raw: float
    total: float


@dataclass(frozen=True)
class EvictionRecord:
    kind: str
    object_id: str
    card_id: str
    reason: str
    state_bytes_before: int
    state_bytes_after: int
    utility: dict[str, float]


def _clamp(value: float) -> float:
    return min(1.0, max(0.0, float(value)))


class UtilityEvictor:
    """No positive recency term; logarithmic buckets protect old coverage."""

    def __init__(self, weights: UtilityWeights | None = None, *, bucket_base_s: float = 1.0) -> None:
        if not math.isfinite(bucket_base_s) or bucket_base_s <= 0:
            raise ValueError("coverage bucket base must be finite and positive")
        self.weights = weights or UtilityWeights()
        self.bucket_base_s = float(bucket_base_s)

    def coverage_bucket(self, card: EvidenceCard) -> int:
        midpoint = max(0.0, (card.t_start + card.t_end) / 2.0)
        return int(math.floor(math.log2(1.0 + midpoint / self.bucket_base_s)))

    def score(self, card: EvidenceCard, store: CardStore) -> UtilityScore:
        neighbors = [item for item in store.temporal_neighbors(card.id) if item is not None]
        similarities = [semantic_similarity(card, neighbor) for neighbor in neighbors]
        novelty = _clamp(1.0 - max(similarities, default=0.0))
        uncertainty = max(0.0, card.left_uncertainty_s) + max(0.0, card.right_uncertainty_s)
        endpoint_certainty = 1.0 / (1.0 + uncertainty)
        hits = (
            float(bool(card.boundary_cache.get("left_hit"))) +
            float(bool(card.boundary_cache.get("right_hit")))
        ) / 2.0
        boundary = _clamp((endpoint_certainty + hits) / 2.0)
        overlap_count = sum(
            other.id != card.id and other.t_start <= card.t_end and other.t_end >= card.t_start
            for other in store
        )
        inverse_density = 1.0 / (1.0 + overlap_count)
        has_raw = float(any(card.raw_ref_status.get(ref) == "available" for ref in card.raw_ref_ids))
        total = (
            self.weights.novelty * novelty + self.weights.boundary * boundary +
            self.weights.inverse_density * inverse_density + self.weights.has_raw * has_raw
        )
        return UtilityScore(novelty, boundary, inverse_density, has_raw, total)

    def coverage_anchors(
        self, cards: Iterable[EvidenceCard], store: CardStore,
    ) -> dict[int, str]:
        anchors: dict[int, tuple[float, str]] = {}
        for card in cards:
            bucket = self.coverage_bucket(card)
            candidate = (self.score(card, store).total, card.id)
            current = anchors.get(bucket)
            if current is None or candidate[0] > current[0] or (
                candidate[0] == current[0] and candidate[1] < current[1]
            ):
                anchors[bucket] = candidate
        return {bucket: value[1] for bucket, value in anchors.items()}

    def choose_card(self, cards: Iterable[EvidenceCard], store: CardStore) -> EvidenceCard | None:
        values = list(cards)
        if not values:
            return None
        anchors = self.coverage_anchors(values, store)
        anchor_ids = set(anchors.values())
        candidates = [card for card in values if card.id not in anchor_ids]
        if not candidates:
            earliest_bucket = min(anchors)
            earliest_id = anchors[earliest_bucket]
            candidates = [card for card in values if card.id != earliest_id]
        if not candidates:
            return values[0]
        return min(
            candidates,
            key=lambda card: (self.score(card, store).total, card.t_start, card.t_end, card.id),
        )

    def evict_raw(self, forest: "EventForest") -> EvictionRecord | None:
        ranked: list[tuple[float, int, float, str, str, EvidenceCard]] = []
        for card in forest.store:
            utility = self.score(card, forest.store).total
            left = set(card.boundary_cache.get("left_frame_ids", []))
            right = set(card.boundary_cache.get("right_frame_ids", []))
            internal = set(card.boundary_cache.get("internal_frame_ids", []))
            for frame_id in card.raw_ref_ids:
                if card.raw_ref_status.get(frame_id) != "available" or frame_id not in forest.raw:
                    continue
                role = 0 if frame_id in internal else (2 if frame_id in left or frame_id in right else 1)
                timestamp = float(forest.raw.metadata({frame_id})[frame_id]["timestamp_s"])
                ranked.append((utility, role, timestamp, card.id, frame_id, card))
        if not ranked:
            return None
        _, _, _, card_id, frame_id, card = min(ranked)
        utility = self.score(card, forest.store)
        before = forest.state_bytes
        forest.raw.release(frame_id, f"card:{card.id}")
        card.raw_ref_ids = [value for value in card.raw_ref_ids if value != frame_id]
        card.raw_ref_status[frame_id] = "evicted"
        for key in ("left_frame_ids", "right_frame_ids", "internal_frame_ids"):
            card.boundary_cache[key] = [
                value for value in card.boundary_cache.get(key, []) if value != frame_id
            ]
        card.boundary_cache["left_hit"] = bool(card.boundary_cache.get("left_frame_ids"))
        card.boundary_cache["right_hit"] = bool(card.boundary_cache.get("right_frame_ids"))
        card.boundary_cache["both_hit"] = bool(
            card.boundary_cache["left_hit"] and card.boundary_cache["right_hit"]
        )
        card.serializable()
        forest.store.touch(card.id)
        forest.merge_queue.invalidate(card.id)
        return EvictionRecord(
            "raw", frame_id, card_id, "utility_raw_first", before,
            forest.state_bytes, asdict(utility),
        )

    def evict_card(self, forest: "EventForest") -> EvictionRecord | None:
        card = self.choose_card(forest.roots, forest.store)
        if card is None:
            return None
        utility = self.score(card, forest.store)
        before = forest.state_bytes
        forest.delete_root(card.id)
        return EvictionRecord(
            "card", card.id, card.id, "utility_coverage_bucket", before,
            forest.state_bytes, asdict(utility),
        )
