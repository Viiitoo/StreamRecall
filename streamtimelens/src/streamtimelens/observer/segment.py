"""Bounded active-segment frame reservoir shared with the raw frame store."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable

from streamtimelens.memory.raw_cache import CachedFrame, RawFrameCache
from streamtimelens.observer.clip_encoder import l2_normalize
from streamtimelens.protocol.types import FramePacket


@dataclass(frozen=True)
class SegmentFrame:
    ref: CachedFrame
    embedding: Any | None = None
    novelty: float = 0.0


class ActiveSegmentReservoir:
    """Online uniform+novelty selection between successful writer calls.

    The segment owns references, not JPEG copies.  Its capacity is one of
    8/16/32 and is selected from the currently remaining logical budget.
    """

    owner = "active_segment"
    capacity_choices = (8, 16, 32)

    def __init__(
        self,
        store: RawFrameCache,
        *,
        remaining_budget_bytes: int,
        bytes_per_frame_estimate: int = 16 * 1024,
        overlap_s: float = 1.0,
        max_capacity: int = 32,
    ) -> None:
        if remaining_budget_bytes < 0 or bytes_per_frame_estimate <= 0:
            raise ValueError("segment budget and frame estimate must be valid")
        if overlap_s < 0 or max_capacity not in self.capacity_choices:
            raise ValueError("segment overlap/capacity configuration is invalid")
        self.store = store
        self.bytes_per_frame_estimate = int(bytes_per_frame_estimate)
        self.overlap_s = float(overlap_s)
        self.max_capacity = int(max_capacity)
        self.capacity = self.capacity_for_budget(remaining_budget_bytes)
        self._uniform: dict[str, SegmentFrame] = {}
        self._novelty: dict[str, SegmentFrame] = {}
        self._last_timestamp: float | None = None

    def capacity_for_budget(self, remaining_budget_bytes: int) -> int:
        remaining = int(remaining_budget_bytes)
        if remaining < 0:
            raise ValueError("remaining segment budget must be non-negative")
        allowed = [choice for choice in self.capacity_choices if choice <= self.max_capacity]
        fitting = [choice for choice in allowed if choice * self.bytes_per_frame_estimate <= remaining]
        return max(fitting) if fitting else min(allowed)

    @staticmethod
    def _deduplicate(frames: Iterable[SegmentFrame]) -> list[SegmentFrame]:
        result: dict[str, SegmentFrame] = {}
        for item in frames:
            result[item.ref.frame_id] = item
        return sorted(result.values(), key=lambda item: (item.ref.timestamp_s, item.ref.frame_index))

    @classmethod
    def _uniform_subset(
        cls, frames: Iterable[SegmentFrame], capacity: int, *, recent_window_s: float = 0.0,
    ) -> list[SegmentFrame]:
        candidates = cls._deduplicate(frames)
        if len(candidates) <= capacity:
            return candidates
        # Always retain true temporal endpoints and the overlap tail, then
        # greedily maximize timestamp coverage of the remaining retained pool.
        latest = candidates[-1].ref.timestamp_s
        recent = [item for item in candidates if item.ref.timestamp_s >= latest - recent_window_s - 1e-9]
        selected = cls._deduplicate((candidates[0], *recent))
        if len(selected) > capacity:
            tail_capacity = max(1, capacity - 1)
            selected = [candidates[0], *recent[-tail_capacity:]]
            selected = cls._deduplicate(selected)
        remaining = [item for item in candidates if item not in selected]
        while len(selected) < capacity and remaining:
            chosen = max(
                remaining,
                key=lambda item: (
                    min(abs(item.ref.timestamp_s - kept.ref.timestamp_s) for kept in selected),
                    -item.ref.frame_index,
                ),
            )
            selected.append(chosen)
            remaining.remove(chosen)
        return sorted(selected, key=lambda item: (item.ref.timestamp_s, item.ref.frame_index))

    def _novelty_score(self, embedding: Any | None) -> tuple[Any | None, float]:
        if embedding is None:
            return None, 0.0
        import numpy as np

        vector = l2_normalize(embedding)
        prior = [item.embedding for item in self.frames if item.embedding is not None]
        if not prior:
            return vector, 1.0
        similarities = [float(np.clip(np.dot(vector, other), -1.0, 1.0)) for other in prior]
        return vector, max(0.0, 1.0 - max(similarities))

    def _apply_selection(
        self, uniform: Iterable[SegmentFrame], novelty: Iterable[SegmentFrame]
    ) -> None:
        old_ids = self.frame_ids
        uniform_map = {item.ref.frame_id: item for item in uniform}
        novelty_map = {item.ref.frame_id: item for item in novelty}
        new_ids = set(uniform_map) | set(novelty_map)
        for frame_id in old_ids - new_ids:
            self.store.release(frame_id, self.owner)
        self._uniform, self._novelty = uniform_map, novelty_map

    def resize_for_budget(self, remaining_budget_bytes: int) -> int:
        capacity = self.capacity_for_budget(remaining_budget_bytes)
        self.capacity = capacity
        uniform_capacity = max(2, capacity // 2)
        novelty_capacity = capacity - uniform_capacity
        uniform = self._uniform_subset(
            self._uniform.values(), uniform_capacity, recent_window_s=self.overlap_s,
        )
        uniform_ids = {item.ref.frame_id for item in uniform}
        novelty = sorted(
            (item for item in self._novelty.values() if item.ref.frame_id not in uniform_ids),
            key=lambda item: (-item.novelty, item.ref.timestamp_s, item.ref.frame_index),
        )[:novelty_capacity]
        self._apply_selection(uniform, novelty)
        return capacity

    def observe(
        self,
        packet: FramePacket,
        embedding: Any | None = None,
        *,
        novelty_score: float | None = None,
        remaining_budget_bytes: int | None = None,
    ) -> bool:
        if self._last_timestamp is not None and packet.timestamp_s < self._last_timestamp:
            raise ValueError("active segment timestamps must be monotonic")
        self._last_timestamp = packet.timestamp_s
        if remaining_budget_bytes is not None:
            self.resize_for_budget(remaining_budget_bytes)
        vector, derived_novelty = self._novelty_score(embedding)
        novelty = derived_novelty if novelty_score is None else float(novelty_score)
        if not math.isfinite(novelty) or novelty < 0:
            raise ValueError("segment novelty must be finite and non-negative")
        ref = self.store.add(packet, owner=self.owner)
        candidate = SegmentFrame(ref, vector, novelty)
        uniform_capacity = max(2, self.capacity // 2)
        novelty_capacity = self.capacity - uniform_capacity
        uniform = self._uniform_subset(
            (*self._uniform.values(), candidate), uniform_capacity,
            recent_window_s=self.overlap_s,
        )
        uniform_ids = {item.ref.frame_id for item in uniform}
        novelty_pool = self._deduplicate((*self._novelty.values(), candidate))
        novelty_selected = sorted(
            (item for item in novelty_pool if item.ref.frame_id not in uniform_ids),
            key=lambda item: (-item.novelty, item.ref.timestamp_s, item.ref.frame_index),
        )[:novelty_capacity]
        self._apply_selection(uniform, novelty_selected)
        if ref.frame_id not in self.frame_ids:
            self.store.release(ref.frame_id, self.owner)
            return False
        return True

    def discard(self, frame_id: str) -> bool:
        present = frame_id in self.frame_ids
        self._uniform.pop(frame_id, None)
        self._novelty.pop(frame_id, None)
        if present:
            self.store.release(frame_id, self.owner)
        return present

    def mark_written(self, timestamp_s: float) -> tuple[str, ...]:
        """Reset the segment while retaining an exact one-second overlap."""
        timestamp = float(timestamp_s)
        if self._last_timestamp is not None and timestamp > self._last_timestamp + 1e-9:
            raise ValueError("segment cannot be written in the future")
        cutoff = timestamp - self.overlap_s
        kept = [
            item for item in self.frames
            if cutoff - 1e-9 <= item.ref.timestamp_s <= timestamp + 1e-9
        ]
        kept = self._uniform_subset(kept, min(max(2, self.capacity // 2), len(kept))) if kept else []
        self._apply_selection(kept, ())
        return tuple(item.ref.frame_id for item in kept)

    def writer_packets(self) -> tuple[FramePacket, ...]:
        return tuple(
            FramePacket(
                item.ref.timestamp_s, item.ref.frame_index, self.store.get(item.ref.frame_id),
                item.ref.width, item.ref.height, source="active_segment", video_id=item.ref.video_id,
            )
            for item in self.frames
        )

    @property
    def frames(self) -> tuple[SegmentFrame, ...]:
        return tuple(self._deduplicate((*self._uniform.values(), *self._novelty.values())))

    @property
    def frame_ids(self) -> set[str]:
        return set(self._uniform) | set(self._novelty)

    @property
    def span(self) -> tuple[float, float] | None:
        frames = self.frames
        if not frames:
            return None
        return frames[0].ref.timestamp_s, frames[-1].ref.timestamp_s

    @property
    def writable(self) -> bool:
        span = self.span
        return span is not None and span[1] > span[0]

    def __len__(self) -> int:
        return len(self.frame_ids)
