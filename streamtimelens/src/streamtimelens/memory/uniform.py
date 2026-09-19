"""Online uniform raw-frame baseline for known and unknown stream lengths."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

from streamtimelens.observer.clip_encoder import serialize_embedding
from streamtimelens.protocol.types import FramePacket

from .raw_cache import CachedFrame, RawFrameCache


@dataclass(frozen=True)
class UniformFrame:
    ref: CachedFrame
    embedding: Any | None
    slot: int
    target_timestamp_s: float | None


class UniformRawCache:
    """Fixed-capacity online sampling without ever rereading an evicted frame.

    With known duration, immutable temporal slot centers are used. With unknown
    duration, seeded Algorithm R reservoir sampling is used. Resizing only
    considers frames that remain in memory.
    """

    owner = "uniform"

    def __init__(
        self,
        store: RawFrameCache,
        capacity: int,
        *,
        duration_s: float | None = None,
        seed: int = 0,
        pixel_only: bool = False,
        embedding_precision: str = "fp16",
    ) -> None:
        if capacity <= 0:
            raise ValueError("uniform capacity must be positive")
        if duration_s is not None and duration_s <= 0:
            raise ValueError("known duration must be positive")
        if embedding_precision not in ("fp16", "int8"):
            raise ValueError("embedding precision must be fp16 or int8")
        self.store = store
        self.capacity = capacity
        self.duration_s = duration_s
        self.seed = seed
        self.pixel_only = pixel_only
        self.embedding_precision = embedding_precision
        self._rng = random.Random(seed)
        self._seen = 0
        self._slots: dict[int, UniformFrame] = {}

    def _slot(self, timestamp_s: float) -> tuple[int, float]:
        assert self.duration_s is not None
        clipped = min(max(timestamp_s, 0.0), self.duration_s)
        slot = min(self.capacity - 1, int(clipped / self.duration_s * self.capacity))
        target = (slot + 0.5) * self.duration_s / self.capacity
        return slot, target

    def _validate_embedding(self, embedding: Any | None) -> None:
        if embedding is None and not self.pixel_only:
            raise ValueError("uniform baseline requires an embedding unless pixel_only is configured")
        if embedding is not None:
            # Serialization validates shape, finiteness and normalization input.
            serialize_embedding(embedding, self.embedding_precision)

    def _select(
        self, slot: int, packet: FramePacket, embedding: Any | None, target: float | None
    ) -> bool:
        existing = self._slots.get(slot)
        ref = self.store.add(packet, owner=self.owner)
        self._slots[slot] = UniformFrame(ref, embedding, slot, target)
        if existing is not None and existing.ref.frame_id != ref.frame_id:
            self.store.release(existing.ref.frame_id, self.owner)
        return True

    def observe(self, packet: FramePacket, embedding: Any | None = None) -> bool:
        self._validate_embedding(embedding)
        self._seen += 1
        if self.duration_s is not None:
            slot, target = self._slot(packet.timestamp_s)
            current = self._slots.get(slot)
            if current is not None:
                current_distance = abs(current.ref.timestamp_s - target)
                candidate_distance = abs(packet.timestamp_s - target)
                if (candidate_distance, packet.frame_index) >= (current_distance, current.ref.frame_index):
                    return False
            return self._select(slot, packet, embedding, target)

        if len(self._slots) < self.capacity:
            return self._select(len(self._slots), packet, embedding, None)
        slot = self._rng.randrange(self._seen)
        if slot >= self.capacity:
            return False
        return self._select(slot, packet, embedding, None)

    def resize(self, capacity: int) -> None:
        """Recompute slots from retained frames only; evicted history is unavailable."""
        if capacity <= 0:
            raise ValueError("uniform capacity must be positive")
        retained = sorted(self._slots.values(), key=lambda item: (item.ref.timestamp_s, item.ref.frame_index))
        self.capacity = capacity
        selected: dict[int, UniformFrame] = {}
        if self.duration_s is not None:
            for item in retained:
                slot, target = self._slot(item.ref.timestamp_s)
                current = selected.get(slot)
                candidate = UniformFrame(item.ref, item.embedding, slot, target)
                if current is None or (
                    abs(item.ref.timestamp_s - target), item.ref.frame_index
                ) < (abs(current.ref.timestamp_s - target), current.ref.frame_index):
                    selected[slot] = candidate
        else:
            if len(retained) <= capacity:
                chosen = retained
            elif capacity == 1:
                chosen = [retained[len(retained) // 2]]
            else:
                chosen = [retained[round(index * (len(retained) - 1) / (capacity - 1))]
                          for index in range(capacity)]
            selected = {
                index: UniformFrame(item.ref, item.embedding, index, None)
                for index, item in enumerate(chosen)
            }
        keep = {item.ref.frame_id for item in selected.values()}
        for item in retained:
            if item.ref.frame_id not in keep:
                self.store.release(item.ref.frame_id, self.owner)
        self._slots = selected

    def evict_worst(self) -> UniformFrame | None:
        if not self._slots:
            return None
        if self.duration_s is None:
            slot = max(self._slots)
        else:
            slot = max(
                self._slots,
                key=lambda index: (
                    abs(self._slots[index].ref.timestamp_s - float(self._slots[index].target_timestamp_s)),
                    self._slots[index].ref.timestamp_s,
                ),
            )
        removed = self._slots.pop(slot)
        self.store.release(removed.ref.frame_id, self.owner)
        return removed

    def metadata(self) -> dict[str, dict[str, object]]:
        result = self.store.metadata(self.frame_ids)
        by_id = {item.ref.frame_id: item for item in self._slots.values()}
        for frame_id, row in result.items():
            item = by_id[frame_id]
            row.update({
                "selection": "uniform",
                "slot": item.slot,
                "target_timestamp_s": item.target_timestamp_s,
                "pixel_only": self.pixel_only,
            })
            if item.embedding is not None:
                row["clip_embedding"] = serialize_embedding(item.embedding, self.embedding_precision)
        return result

    @property
    def frames(self) -> tuple[UniformFrame, ...]:
        return tuple(sorted(self._slots.values(), key=lambda item: (item.ref.timestamp_s, item.ref.frame_index)))

    @property
    def frame_ids(self) -> set[str]:
        return {item.ref.frame_id for item in self._slots.values()}

    @property
    def embedding_bytes(self) -> int:
        total = 0
        for item in self._slots.values():
            if item.embedding is not None:
                record = serialize_embedding(item.embedding, self.embedding_precision)
                import base64
                total += len(base64.b64decode(record["data_b64"]))
        return total

    @property
    def logical_bytes(self) -> int:
        return self.store.byte_size + self.embedding_bytes

    def __len__(self) -> int:
        return len(self._slots)
