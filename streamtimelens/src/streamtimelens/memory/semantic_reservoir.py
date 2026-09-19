"""Query-blind semantic novelty reservoir with temporal coverage anchors."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from streamtimelens.observer.clip_encoder import l2_normalize, serialize_embedding
from streamtimelens.protocol.types import FramePacket

from .raw_cache import CachedFrame, RawFrameCache


@dataclass(frozen=True)
class SemanticFrame:
    ref: CachedFrame
    embedding: Any
    role: str
    novelty: float
    anchor_slot: int | None = None
    target_timestamp_s: float | None = None


def cosine_similarity(left: Any, right: Any) -> float:
    import numpy as np

    first, second = l2_normalize(left), l2_normalize(right)
    if first.shape != second.shape:
        raise ValueError("semantic embeddings must have the same dimension")
    return float(np.clip(np.dot(first, second), -1.0, 1.0))


class SemanticReservoir:
    """Retain temporal anchors and the most complementary CLIP observations."""

    owner = "semantic_reservoir"

    def __init__(
        self,
        store: RawFrameCache,
        capacity: int,
        *,
        duration_s: float,
        anchor_fraction: float = 0.25,
        embedding_precision: str = "fp16",
    ) -> None:
        if capacity <= 0 or duration_s <= 0:
            raise ValueError("semantic capacity and duration must be positive")
        if not 0 < anchor_fraction <= 1:
            raise ValueError("anchor_fraction must be in (0,1]")
        if embedding_precision not in ("fp16", "int8"):
            raise ValueError("embedding precision must be fp16 or int8")
        self.store = store
        self.capacity = capacity
        self.duration_s = duration_s
        self.anchor_fraction = anchor_fraction
        self.embedding_precision = embedding_precision
        self.anchor_capacity = min(capacity, max(1, math.ceil(capacity * anchor_fraction)))
        if capacity > 1 and self.anchor_capacity == capacity:
            self.anchor_capacity -= 1
        self._anchors: dict[int, SemanticFrame] = {}
        self._novel: dict[str, SemanticFrame] = {}

    def novelty(self, embedding: Any) -> float:
        vector = l2_normalize(embedding)
        current = [item.embedding for item in (*self._anchors.values(), *self._novel.values())]
        if not current:
            return 1.0
        return max(0.0, min(2.0, 1.0 - max(cosine_similarity(vector, item) for item in current)))

    def _anchor_slot(self, timestamp_s: float) -> tuple[int, float]:
        clipped = min(max(timestamp_s, 0.0), self.duration_s)
        slot = min(self.anchor_capacity - 1, int(clipped / self.duration_s * self.anchor_capacity))
        target = (slot + 0.5) * self.duration_s / self.anchor_capacity
        return slot, target

    def _add(self, packet: FramePacket, embedding: Any, role: str, novelty: float,
             slot: int | None = None, target: float | None = None) -> SemanticFrame:
        ref = self.store.add(packet, owner=self.owner)
        return SemanticFrame(ref, l2_normalize(embedding), role, novelty, slot, target)

    def observe(self, packet: FramePacket, embedding: Any) -> bool:
        vector = l2_normalize(embedding)
        serialize_embedding(vector, self.embedding_precision)
        novelty = self.novelty(vector)

        slot, target = self._anchor_slot(packet.timestamp_s)
        anchor = self._anchors.get(slot)
        if anchor is None or (
            abs(packet.timestamp_s - target), packet.frame_index
        ) < (abs(anchor.ref.timestamp_s - target), anchor.ref.frame_index):
            selected = self._add(packet, vector, "temporal_anchor", novelty, slot, target)
            self._anchors[slot] = selected
            if anchor is not None and anchor.ref.frame_id != selected.ref.frame_id:
                self.store.release(anchor.ref.frame_id, self.owner)
            # One timestamp has one role; anchors are already semantic evidence.
            self._novel.pop(selected.ref.frame_id, None)
            return True

        novelty_capacity = self.capacity - self.anchor_capacity
        if novelty_capacity <= 0 or anchor.ref.frame_id == f"{packet.frame_index:09d}.jpg":
            return False
        frame_id = f"{packet.frame_index:09d}.jpg"
        if frame_id in self._novel or frame_id in {item.ref.frame_id for item in self._anchors.values()}:
            return False
        candidate = self._add(packet, vector, "semantic_novelty", novelty)
        if len(self._novel) < novelty_capacity:
            self._novel[candidate.ref.frame_id] = candidate
            return True
        worst = min(
            self._novel.values(),
            key=lambda item: (item.novelty, -item.ref.timestamp_s, -item.ref.frame_index),
        )
        if (novelty, -packet.frame_index) <= (worst.novelty, -worst.ref.frame_index):
            self.store.release(candidate.ref.frame_id, self.owner)
            return False
        self._novel.pop(worst.ref.frame_id)
        self.store.release(worst.ref.frame_id, self.owner)
        self._novel[candidate.ref.frame_id] = candidate
        return True

    def evict_worst(self) -> SemanticFrame | None:
        if self._novel:
            removed = min(
                self._novel.values(),
                key=lambda item: (item.novelty, -item.ref.timestamp_s, -item.ref.frame_index),
            )
            self._novel.pop(removed.ref.frame_id)
        elif self._anchors:
            slot = max(self._anchors)
            removed = self._anchors.pop(slot)
        else:
            return None
        self.store.release(removed.ref.frame_id, self.owner)
        return removed

    def metadata(self) -> dict[str, dict[str, object]]:
        frames = self.frames
        result = self.store.metadata({item.ref.frame_id for item in frames})
        for item in frames:
            result[item.ref.frame_id].update({
                "selection": item.role,
                "novelty": item.novelty,
                "anchor_slot": item.anchor_slot,
                "target_timestamp_s": item.target_timestamp_s,
                "clip_embedding": serialize_embedding(item.embedding, self.embedding_precision),
            })
        return result

    @property
    def frames(self) -> tuple[SemanticFrame, ...]:
        return tuple(sorted(
            (*self._anchors.values(), *self._novel.values()),
            key=lambda item: (item.ref.timestamp_s, item.ref.frame_index),
        ))

    @property
    def logical_bytes(self) -> int:
        import base64

        embedding_bytes = sum(
            len(base64.b64decode(serialize_embedding(item.embedding, self.embedding_precision)["data_b64"]))
            for item in self.frames
        )
        return self.store.byte_size + embedding_bytes

    def __len__(self) -> int:
        return len(self._anchors) + len(self._novel)
