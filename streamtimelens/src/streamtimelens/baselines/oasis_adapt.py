"""Online OASIS adaptation with strict all-node byte accounting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from streamtimelens.baselines.vst_summary import VSTIngestConfig, VSTSummaryIngestor
from streamtimelens.memory.cards import EvidenceCard
from streamtimelens.memory.forest import EventForest
from streamtimelens.memory.raw_cache import RawFrameCache
from streamtimelens.protocol.types import Budget, VideoMeta
from streamtimelens.retrieval.embedder import CardTextEmbedder


@dataclass(frozen=True)
class OASISIngestConfig:
    segment_interval_s: float = 32.0
    max_frames: int = 32
    max_roots: int = 8
    overlap_s: float = 1.0
    bytes_per_frame_estimate: int = 16 * 1024
    source_revision: str = "unknown"

    def __post_init__(self) -> None:
        if self.segment_interval_s <= 0 or self.max_frames not in (8, 16, 32):
            raise ValueError("invalid OASIS fixed segment configuration")
        if self.max_roots < 1 or self.overlap_s < 0 or self.bytes_per_frame_estimate <= 0:
            raise ValueError("invalid OASIS hierarchy/budget configuration")


class OASISMemory:
    """Summary hierarchy whose retained children and parents all count."""

    def __init__(self, limit_bytes: int, raw: RawFrameCache, *, max_roots: int) -> None:
        if limit_bytes <= 0 or max_roots <= 0:
            raise ValueError("invalid OASIS memory limit/root cap")
        self.limit_bytes = limit_bytes
        self.raw = raw
        self.max_roots = max_roots
        self.forest = EventForest(limit_bytes, raw_cache=raw, require_embeddings=True)

    @property
    def cards(self) -> list[EvidenceCard]:
        return list(self.forest.cards)

    @property
    def card_bytes(self) -> int:
        return sum(
            value for key, value in self.forest.accounting_components().items()
            if key != "raw_frames"
        )

    @property
    def state_bytes(self) -> int:
        return self.forest.state_bytes

    @property
    def eviction_trace(self) -> list[dict[str, Any]]:
        return self.forest.eviction_trace

    @property
    def merge_trace(self) -> list[dict[str, Any]]:
        return self.forest.merge_trace

    def accounting(self) -> dict[str, int]:
        return {
            f"oasis_{key}": value
            for key, value in self.forest.accounting_components().items()
        }

    def insert(self, card: EvidenceCard) -> None:
        self.forest.insert(card)
        while len(self.forest.roots) > self.max_roots:
            if self.forest.merge_best() is None:
                break
        self.enforce()

    def enforce(self) -> None:
        now = max((card.t_end for card in self.forest.cards), default=0.0)
        self.forest.enforce_budget(current_stream_time=now)

    def evict_one(self, reason: str = "oasis_memory_budget") -> object | None:
        record = self.forest.evictor.evict_raw(self.forest)
        if record is None:
            record = self.forest.evictor.evict_card(self.forest)
        if record is None:
            return None
        row = dict(record.__dict__)
        row["reason"] = reason
        self.forest.eviction_trace.append(row)
        return record


class OASISAdaptIngestor(VSTSummaryIngestor):
    """Fixed-segment online OASIS baseline; never preloads a full video."""

    def __init__(
        self, meta: VideoMeta, budget: Budget, config: OASISIngestConfig,
        *, text_embedder: CardTextEmbedder, writer: Any | None = None,
    ) -> None:
        super().__init__(
            meta, budget,
            VSTIngestConfig(
                policy="fifo", segment_interval_s=config.segment_interval_s,
                max_frames=config.max_frames, overlap_s=config.overlap_s,
                bytes_per_frame_estimate=config.bytes_per_frame_estimate,
                source_revision=config.source_revision,
            ),
            text_embedder=text_embedder, writer=writer,
        )
        self.oasis_config = config
        self.memory = OASISMemory(budget.memory_bytes, self.raw, max_roots=config.max_roots)
        self.snapshot_method = "oasis_adapt"
        self.memory_label = "oasis_adjacent_hierarchy"
        self._merge_cursor = 0

    def _write(self, timestamp_s: float) -> None:
        super()._write(timestamp_s)
        for row in self.memory.merge_trace[self._merge_cursor:]:
            self.trace.record(
                kind="nodes_merged", timestamp_s=timestamp_s, reason=row["mode"],
                object_id=row["parent_id"], child_ids=row["child_ids"],
                merge_reason=row["reason"], state_bytes_after=row["state_bytes_after"],
            )
        self._merge_cursor = len(self.memory.merge_trace)
