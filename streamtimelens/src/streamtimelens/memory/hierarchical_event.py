"""Query-independent, serializable hierarchical event memory for HEM-01."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from streamtimelens.observer.clip_encoder import (
    deserialize_embedding,
    l2_normalize,
    serialize_embedding,
)
from streamtimelens.protocol.snapshot import SnapshotManifest, SnapshotReader, SnapshotWriter
from streamtimelens.protocol.types import Budget, VideoMeta


HEM_SCHEMA_VERSION = "hem_event_memory_v1"
LEVEL_CAPACITIES = (64, 32, 16)
BOUNDARY_COSINE = 0.85
MAX_FINE_EVENT_DURATION_S = 8.0
EMBEDDING_PRECISION = "fp16"


def _finite(value: Any) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("HEM-01 values must be finite")
    return result


def _event_id(scale: int, first_frame_index: int, last_frame_index: int) -> str:
    return f"hem-s{scale}-{first_frame_index:09d}-{last_frame_index:09d}"


@dataclass(frozen=True)
class HierarchicalEvent:
    scale: int
    start_s: float
    end_s: float
    support_count: int
    first_frame_index: int
    last_frame_index: int
    embedding: tuple[float, ...]

    def __post_init__(self) -> None:
        values = np.asarray(self.embedding, dtype=np.float32)
        if (
            self.scale < 0 or self.scale >= len(LEVEL_CAPACITIES)
            or not math.isfinite(self.start_s) or not math.isfinite(self.end_s)
            or self.start_s < 0 or self.end_s < self.start_s
            or self.support_count <= 0
            or self.first_frame_index < 0 or self.last_frame_index < self.first_frame_index
            or values.ndim != 1 or values.size == 0 or not np.isfinite(values).all()
            or abs(float(np.linalg.norm(values)) - 1.0) > 1e-5
        ):
            raise ValueError("invalid HEM-01 event")

    @property
    def event_id(self) -> str:
        return _event_id(self.scale, self.first_frame_index, self.last_frame_index)

    def to_record(self) -> dict[str, Any]:
        return {
            "schema_version": HEM_SCHEMA_VERSION,
            "event_id": self.event_id,
            "scale": self.scale,
            "start_s": self.start_s,
            "end_s": self.end_s,
            "support_count": self.support_count,
            "first_frame_index": self.first_frame_index,
            "last_frame_index": self.last_frame_index,
            "embedding": serialize_embedding(self.embedding, EMBEDDING_PRECISION),
        }

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> "HierarchicalEvent":
        required = {
            "schema_version", "event_id", "scale", "start_s", "end_s",
            "support_count", "first_frame_index", "last_frame_index", "embedding",
        }
        if set(record) != required or record.get("schema_version") != HEM_SCHEMA_VERSION:
            raise ValueError("HEM-01 event record schema mismatch")
        event = cls(
            int(record["scale"]), _finite(record["start_s"]), _finite(record["end_s"]),
            int(record["support_count"]), int(record["first_frame_index"]),
            int(record["last_frame_index"]),
            tuple(map(float, deserialize_embedding(dict(record["embedding"])))),
        )
        if record["event_id"] != event.event_id:
            raise ValueError("HEM-01 event ID mismatch")
        return event


def _merge_events(left: HierarchicalEvent, right: HierarchicalEvent, scale: int) -> HierarchicalEvent:
    if (
        left.end_s > right.start_s + 1e-9
        or left.last_frame_index >= right.first_frame_index
        or len(left.embedding) != len(right.embedding)
    ):
        raise ValueError("HEM-01 only merges ordered adjacent event state")
    support = left.support_count + right.support_count
    vector = (
        np.asarray(left.embedding, dtype=np.float32) * left.support_count
        + np.asarray(right.embedding, dtype=np.float32) * right.support_count
    )
    vector = l2_normalize(vector)
    return HierarchicalEvent(
        scale, left.start_s, right.end_s, support,
        left.first_frame_index, right.last_frame_index, tuple(map(float, vector)),
    )


class HierarchicalEventMemory:
    """Single-pass recent-fine/old-coarse event state with fixed capacities."""

    def __init__(
        self, *, capacities: Sequence[int] = LEVEL_CAPACITIES,
        boundary_cosine: float = BOUNDARY_COSINE,
        max_fine_event_duration_s: float = MAX_FINE_EVENT_DURATION_S,
    ) -> None:
        if (
            tuple(capacities) != LEVEL_CAPACITIES
            or float(boundary_cosine) != BOUNDARY_COSINE
            or float(max_fine_event_duration_s) != MAX_FINE_EVENT_DURATION_S
        ):
            raise ValueError("HEM-01 v1 structural settings are frozen")
        self.capacities = tuple(map(int, capacities))
        self.boundary_cosine = float(boundary_cosine)
        self.max_fine_event_duration_s = float(max_fine_event_duration_s)
        self._levels: list[list[HierarchicalEvent]] = [[] for _ in self.capacities]
        self._open: HierarchicalEvent | None = None
        self._last_timestamp_s: float | None = None
        self._last_frame_index: int | None = None
        self._seen = 0

    def observe(self, *, timestamp_s: float, frame_index: int, embedding: Sequence[float]) -> str:
        """Update from one stream token; the API intentionally has no query or GT."""
        timestamp = _finite(timestamp_s)
        vector = np.asarray(l2_normalize(embedding), dtype=np.float32)
        if timestamp < 0 or frame_index < 0:
            raise ValueError("HEM-01 stream identity is invalid")
        if self._last_timestamp_s is not None and timestamp <= self._last_timestamp_s:
            raise ValueError("HEM-01 timestamps must be strictly increasing")
        if self._last_frame_index is not None and frame_index <= self._last_frame_index:
            raise ValueError("HEM-01 frame indices must be strictly increasing")
        leaf = HierarchicalEvent(
            0, timestamp, timestamp, 1, frame_index, frame_index,
            tuple(map(float, vector)),
        )
        reason = "open_first_event"
        if self._open is None:
            self._open = leaf
        else:
            cosine = float(np.dot(self._open.embedding, leaf.embedding))
            duration = timestamp - self._open.start_s
            if cosine >= self.boundary_cosine and duration <= self.max_fine_event_duration_s:
                self._open = _merge_events(self._open, leaf, 0)
                reason = "merge_similar"
            else:
                self._seal_open()
                self._open = leaf
                reason = "boundary_visual_change" if cosine < self.boundary_cosine else "boundary_duration"
        self._last_timestamp_s = timestamp
        self._last_frame_index = int(frame_index)
        self._seen += 1
        self._enforce_capacities(0)
        self._assert_invariants()
        return reason

    def _seal_open(self) -> None:
        if self._open is None:
            return
        self._levels[0].append(self._open)
        self._open = None
        self._enforce_capacities(0)

    def _enforce_capacities(self, start_scale: int) -> None:
        for scale in range(start_scale, len(self._levels)):
            level = self._levels[scale]
            limit = self.capacities[scale] - int(scale == 0 and self._open is not None)
            while len(level) > limit:
                left, right = level.pop(0), level.pop(0)
                target_scale = min(scale + 1, len(self._levels) - 1)
                merged = _merge_events(left, right, target_scale)
                if target_scale == scale:
                    level.insert(0, merged)
                else:
                    self._levels[target_scale].append(merged)
                    self._levels[target_scale].sort(key=lambda row: row.first_frame_index)

    def events(self) -> tuple[HierarchicalEvent, ...]:
        result = [event for level in self._levels for event in level]
        if self._open is not None:
            result.append(self._open)
        return tuple(sorted(result, key=lambda row: (row.first_frame_index, row.scale)))

    def records(self) -> list[dict[str, Any]]:
        self._assert_invariants()
        return [event.to_record() for event in self.events()]

    def _assert_invariants(self) -> None:
        for scale, level in enumerate(self._levels):
            if len(level) > self.capacities[scale] or any(row.scale != scale for row in level):
                raise RuntimeError("HEM-01 level capacity invariant failed")
        events = self.events()
        if sum(event.support_count for event in events) != self._seen:
            raise RuntimeError("HEM-01 support conservation failed")
        for left, right in zip(events, events[1:]):
            if left.end_s > right.start_s + 1e-9 or left.last_frame_index >= right.first_frame_index:
                raise RuntimeError("HEM-01 event coverage overlaps or reorders history")
        if len({event.event_id for event in events}) != len(events):
            raise RuntimeError("HEM-01 event IDs are not unique")

    @property
    def seen_count(self) -> int:
        return self._seen

    @property
    def level_counts(self) -> tuple[int, ...]:
        counts = [len(level) for level in self._levels]
        if self._open is not None:
            counts[0] += 1
        return tuple(counts)


def load_event_memory(snapshot: SnapshotReader) -> tuple[HierarchicalEvent, ...]:
    """Validate query-visible persisted events without access to ingest history."""
    if not isinstance(snapshot, SnapshotReader):
        raise TypeError("HEM-01 requires a verified SnapshotReader")
    events = tuple(HierarchicalEvent.from_record(row) for row in snapshot.read_event_memory())
    if not events:
        raise ValueError("HEM-01 snapshot contains no event memory")
    ordered = tuple(sorted(events, key=lambda row: (row.first_frame_index, row.scale)))
    if events != ordered or any(event.end_s > snapshot.manifest.t_q + 1e-6 for event in events):
        raise ValueError("HEM-01 snapshot exposes reordered or future event state")
    if sum(event.support_count for event in events) <= 0:
        raise ValueError("HEM-01 snapshot support is invalid")
    for left, right in zip(events, events[1:]):
        if left.end_s > right.start_s + 1e-9 or left.last_frame_index >= right.first_frame_index:
            raise ValueError("HEM-01 snapshot event coverage overlaps")
    return events


def write_event_snapshot(
    memory: HierarchicalEventMemory,
    writer: SnapshotWriter,
    *,
    name: str,
    t_q: float,
    meta: VideoMeta,
    budget: Budget,
    config: Mapping[str, Any],
    source_revision: str,
) -> SnapshotManifest:
    """Persist only immutable event records and account their filesystem bytes."""
    if writer.source_revision != source_revision:
        raise ValueError("HEM-01 writer/source revision mismatch")
    if memory.seen_count <= 0 or memory.events()[-1].end_s > t_q + 1e-6:
        raise ValueError("HEM-01 snapshot time excludes observed state")
    return writer.write(
        name=name, t_q=t_q, meta=meta, budget=budget, cards=[], raw_frames=[],
        raw_metadata={}, writer_calls=0, config=dict(config),
        method="hierarchical_event_memory", pixel_only=False,
        embedder={
            "kind": "frozen_clip_visual",
            "persistence": EMBEDDING_PRECISION,
            "query_independent": True,
        },
        event_memory=memory.records(),
    )


def event_memory_sha256(events: Sequence[HierarchicalEvent]) -> str:
    payload = json.dumps(
        [event.to_record() for event in events], sort_keys=True,
        separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
