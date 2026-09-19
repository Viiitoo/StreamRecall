"""VST-style query-independent free-text summary baseline."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from io import BytesIO
from typing import Any, Iterable, Literal, Sequence

from streamtimelens.evaluation.cost import ComponentTimer
from streamtimelens.evaluation.trace import TraceLog
from streamtimelens.memory.budget import BudgetLedger
from streamtimelens.memory.cards import EvidenceCard
from streamtimelens.memory.raw_cache import RawFrameCache
from streamtimelens.observer.quota import WriterQuota
from streamtimelens.observer.segment import ActiveSegmentReservoir
from streamtimelens.protocol.snapshot import SnapshotManifest, SnapshotWriter
from streamtimelens.protocol.types import Budget, FramePacket, VideoMeta
from streamtimelens.refiner.timelens_adapter import SparseFrame, prepare_sparse_video
from streamtimelens.retrieval.embedder import CardTextEmbedder


VST_PROMPT_VERSION = "vst_observation_v1"
VST_MAX_NEW_TOKENS = 256
_VST_TEMPLATE = """You observe one closed segment of a streaming video.
Frames are sparse and their labels are global seconds. Describe only visible observations
and optionally a brief thought about temporal continuity. Do not infer user intent and do
not output JSON or structured fields. Segment: {start:.3f}-{end:.3f}s.
Sampled timestamps: {timestamps}. Return concise free text only."""
VST_PROMPT_SHA256 = hashlib.sha256(_VST_TEMPLATE.encode("utf-8")).hexdigest()


def build_vst_prompt(segment: tuple[float, float], timestamps: Sequence[float]) -> str:
    start, end = segment
    if start < 0 or start >= end or not timestamps:
        raise ValueError("VST prompt needs a closed observed segment")
    if min(timestamps) < start - 1e-6 or max(timestamps) > end + 1e-6:
        raise ValueError("VST prompt timestamps must stay inside the segment")
    return _VST_TEMPLATE.format(
        start=start, end=end,
        timestamps=", ".join(f"{value:.3f}s" for value in timestamps),
    )


@dataclass(frozen=True)
class SummaryCall:
    text: str
    raw_output: str
    generated_tokens: int
    resource: dict[str, Any]
    prompt_hash: str = VST_PROMPT_SHA256
    prompt_version: str = VST_PROMPT_VERSION


class TimeLensFreeTextWriter:
    """Use the same frozen service and 256-token budget as the structured writer."""

    def __init__(self, service: Any, *, max_frames: int = 32) -> None:
        if max_frames not in (8, 16, 32):
            raise ValueError("VST writer frame cap must be 8, 16, or 32")
        self.service = service
        self.max_frames = max_frames

    @staticmethod
    def _decode(packet: FramePacket) -> Any:
        from PIL import Image
        import numpy as np

        try:
            with Image.open(BytesIO(packet.image)) as image:
                return np.asarray(image.convert("RGB").copy())
        except OSError as exc:
            raise ValueError("VST writer received an undecodable frame") from exc

    def summarize(self, frames: Sequence[FramePacket], meta: VideoMeta) -> SummaryCall:
        ordered = tuple(sorted(frames, key=lambda item: (item.timestamp_s, item.frame_index)))
        if len(ordered) < 2 or len(ordered) > self.max_frames:
            raise ValueError("VST writer needs one bounded closed segment")
        prepared = prepare_sparse_video(
            [SparseFrame(item.frame_index, item.timestamp_s, self._decode(item)) for item in ordered],
            original_fps=meta.original_fps, total_num_frames=meta.total_num_frames,
            k_frames=self.max_frames,
        )
        prompt = build_vst_prompt(
            (prepared.timestamps_s[0], prepared.timestamps_s[-1]), prepared.timestamps_s,
        )
        messages = [{"role": "user", "content": [
            {"type": "video"}, {"type": "text", "text": prompt},
        ]}]
        with ComponentTimer(
            "timelens_vst_writer", torch_module=getattr(self.service, "torch", None),
        ) as timer:
            raw = str(self.service.generate(
                messages, [prepared.processor_video], VST_MAX_NEW_TOKENS,
            ))
        text = " ".join(raw.strip().split())
        if not text:
            text = "unknown observation"
        return SummaryCall(
            text, raw, int(getattr(self.service, "last_call_stats", {}).get("generated_tokens", 0)),
            timer.as_dict(),
        )


class VSTSummaryMemory:
    """Exact full-state accounting for FIFO or first-plus-recent retention."""

    def __init__(
        self, limit_bytes: int, raw: RawFrameCache,
        *, policy: Literal["first_recent", "fifo"] = "first_recent",
    ) -> None:
        if limit_bytes <= 0 or policy not in ("first_recent", "fifo"):
            raise ValueError("invalid VST memory configuration")
        self.limit_bytes = int(limit_bytes)
        self.raw = raw
        self.policy = policy
        self.cards: list[EvidenceCard] = []
        self.eviction_trace: list[dict[str, Any]] = []

    @property
    def card_bytes(self) -> int:
        return sum(len(json.dumps(
            card.serializable(), sort_keys=True, separators=(",", ":"),
            ensure_ascii=False, allow_nan=False,
        ).encode("utf-8")) + 1 for card in self.cards)

    @property
    def token_bytes(self) -> int:
        return sum(max(0, card.generated_tokens) * 4 for card in self.cards)

    @property
    def embedding_bytes(self) -> int:
        return sum(
            len(json.dumps(card.text_embedding, sort_keys=True).encode("utf-8"))
            for card in self.cards if card.text_embedding is not None
        )

    @property
    def index_bytes(self) -> int:
        return len(json.dumps({"card_ids": [card.id for card in self.cards]}).encode("utf-8"))

    @property
    def state_bytes(self) -> int:
        # card_bytes already contains the embedded vector; expose embedding and
        # token components diagnostically without counting either twice.
        return self.card_bytes + self.token_bytes + self.index_bytes + self.raw.byte_size

    def accounting(self) -> dict[str, int]:
        return {
            "summary_payload_and_embedding": self.card_bytes,
            "summary_token_ids": self.token_bytes,
            "summary_index": self.index_bytes,
            "raw_frames": self.raw.byte_size,
        }

    def insert(self, card: EvidenceCard) -> None:
        if any(item.id == card.id for item in self.cards):
            raise ValueError("duplicate VST summary ID")
        self.cards.append(card)
        self.cards.sort(key=lambda item: (item.t_start, item.t_end, item.id))
        self.enforce()

    def _eviction_index(self) -> int:
        if self.policy == "fifo" or len(self.cards) == 1:
            return 0
        return 1  # preserve the first observation and the most recent tail

    def evict_one(self, reason: str = "vst_memory_budget") -> EvidenceCard | None:
        if not self.cards:
            return None
        card = self.cards.pop(self._eviction_index())
        for frame_id in tuple(card.raw_ref_ids):
            self.raw.release(frame_id, f"card:{card.id}")
        self.eviction_trace.append({"object_id": card.id, "reason": reason})
        return card

    def enforce(self) -> None:
        while self.state_bytes > self.limit_bytes and len(self.cards) > 1:
            self.evict_one()
        if self.state_bytes > self.limit_bytes:
            self.evict_one()
        if self.state_bytes > self.limit_bytes:
            raise MemoryError("VST raw working set alone exceeds the memory budget")


@dataclass(frozen=True)
class VSTIngestConfig:
    policy: Literal["first_recent", "fifo"] = "first_recent"
    segment_interval_s: float = 32.0
    max_frames: int = 32
    overlap_s: float = 1.0
    bytes_per_frame_estimate: int = 16 * 1024
    source_revision: str = "unknown"

    def __post_init__(self) -> None:
        if self.policy not in ("first_recent", "fifo") or self.segment_interval_s <= 0:
            raise ValueError("invalid VST ingest policy/interval")
        if self.max_frames not in (8, 16, 32) or self.overlap_s < 0:
            raise ValueError("invalid VST segment frame/overlap configuration")


class VSTSummaryIngestor:
    """Single-pass, query-blind VST-summary ingestion under the shared budgets."""

    def __init__(
        self, meta: VideoMeta, budget: Budget, config: VSTIngestConfig,
        *, text_embedder: CardTextEmbedder, writer: Any | None = None,
    ) -> None:
        self.meta, self.budget, self.config = meta, budget, config
        self.text_embedder, self.writer = text_embedder, writer
        self.raw = RawFrameCache()
        self.segment = ActiveSegmentReservoir(
            self.raw, remaining_budget_bytes=budget.memory_bytes,
            bytes_per_frame_estimate=config.bytes_per_frame_estimate,
            overlap_s=config.overlap_s, max_capacity=config.max_frames,
        )
        self.memory = VSTSummaryMemory(budget.memory_bytes, self.raw, policy=config.policy)
        initial = 1.0 if budget.writer_calls_per_minute > 0 else 0.0
        self.quota = WriterQuota(budget.writer_calls_per_minute, initial_tokens=initial)
        self.ledger = BudgetLedger(budget.memory_bytes)
        self.trace = TraceLog()
        self._last_write_s: float | None = None
        self._counter = 0
        self.snapshot_method = "vst_summary"
        self.memory_label = config.policy

    def _sync_budget(self) -> None:
        self.memory.enforce()
        self.ledger.replace_components(self.memory.accounting())
        self.ledger.assert_within_budget()

    def _write(self, timestamp_s: float) -> None:
        frames = self.segment.writer_packets()
        if len(frames) < 2:
            return
        span = (frames[0].timestamp_s, frames[-1].timestamp_s)
        if self.writer is None:
            with ComponentTimer("vst_fallback_writer") as timer:
                text = f"Observation from {span[0]:.3f} to {span[1]:.3f} seconds."
            call = SummaryCall(text, text, 0, timer.as_dict())
        else:
            call = self.writer.summarize(frames, self.meta)
        self._counter += 1
        card_id = f"vst-{self._counter:06d}"
        raw_refs = [f"{frames[0].frame_index:09d}.jpg", f"{frames[-1].frame_index:09d}.jpg"]
        raw_refs = list(dict.fromkeys(raw_refs))
        retained = []
        for frame_id in raw_refs:
            if frame_id in self.raw:
                self.raw.retain(frame_id, f"card:{card_id}")
                retained.append(frame_id)
        card = EvidenceCard(
            card_id, 0, span[0], span[1], call.text,
            support_timestamps=[span[0], span[1]], raw_ref_ids=retained,
            source_chunk_ids=[card_id], normalized_text=call.text,
            writer_model_revision="shared-timelens" if self.writer is not None else "fallback",
            prompt_version=call.prompt_version, prompt_hash=call.prompt_hash,
            writer_provenance={"memory_policy": self.memory_label},
            generated_tokens=call.generated_tokens,
            raw_ref_status={frame_id: "available" for frame_id in retained},
            boundary_cache={
                "left_frame_ids": retained[:1], "right_frame_ids": retained[-1:],
                "internal_frame_ids": [], "left_hit": bool(retained),
                "right_hit": bool(retained), "both_hit": bool(retained),
            },
        )
        self.text_embedder.embed_cards([card])
        self.memory.insert(card)
        self.ledger.writer_calls += 1
        self.segment.mark_written(timestamp_s)
        self._last_write_s = timestamp_s
        self._sync_budget()
        self.trace.record(
            kind="writer_called", timestamp_s=timestamp_s, reason="periodic_vst",
            object_id=card_id, raw_output=call.raw_output,
            generated_tokens=call.generated_tokens, resource=call.resource,
            prompt_version=call.prompt_version, prompt_hash=call.prompt_hash,
        )
        for row in self.memory.eviction_trace:
            if not row.get("traced"):
                self.trace.record(
                    kind="payload_evicted", timestamp_s=timestamp_s,
                    reason=row["reason"], object_id=row["object_id"],
                )
                row["traced"] = True

    def observe(self, packet: FramePacket) -> None:
        self.trace.record(
            kind="frame_seen", timestamp_s=packet.timestamp_s,
            reason=packet.source, object_id=str(packet.frame_index),
        )
        self.segment.observe(
            packet, remaining_budget_bytes=max(0, self.budget.memory_bytes - self.memory.card_bytes),
        )
        due = self._last_write_s is None or packet.timestamp_s - self._last_write_s >= self.config.segment_interval_s
        if due and self.segment.writable:
            if self.quota.try_consume(packet.timestamp_s):
                self._write(packet.timestamp_s)
            else:
                self.trace.record(
                    kind="trigger_dropped", timestamp_s=packet.timestamp_s,
                    reason=f"writer_quota:{self.quota.last_reason}",
                )
        self._sync_budget()

    def snapshot(
        self, writer: SnapshotWriter, *, name: str, t_q: float, config: dict[str, object],
    ) -> SnapshotManifest:
        self._sync_budget()
        while True:
            try:
                manifest = writer.write(
                    name=name, t_q=t_q, meta=self.meta, budget=self.budget,
                    cards=[card.serializable() for card in self.memory.cards],
                    raw_frames=self.raw.items(), raw_metadata=self.raw.metadata(),
                    writer_calls=self.ledger.writer_calls, config=config,
                    method=self.snapshot_method, embedder=vars(self.text_embedder.metadata),
                )
                break
            except MemoryError:
                if self.memory.evict_one("snapshot_filesystem_budget") is None:
                    removed = self.raw.evict_oldest()
                    if removed is None:
                        raise
                    self.segment.discard(removed.frame_id)
                self._sync_budget()
        filesystem = self.ledger.reconcile(writer.output_root / name)
        self.trace.record(
            kind="snapshot_written", timestamp_s=t_q, reason="arrival", object_id=name,
            logical_state_bytes=self.ledger.logical_bytes,
            snapshot_filesystem_bytes=filesystem,
        )
        return manifest

    def run(
        self, packets: Iterable[FramePacket], arrival_times: Iterable[float], writer: SnapshotWriter,
        *, config: dict[str, object], snapshot_prefix: str = "snapshot",
    ) -> dict[float, SnapshotManifest]:
        requested = sorted(set(float(value) for value in arrival_times))
        if any(not math.isfinite(value) or value < 0 or value > self.meta.duration_s + 1e-6 for value in requested):
            raise ValueError("snapshot time outside video duration")
        output: dict[float, SnapshotManifest] = {}
        cursor = 0
        last: FramePacket | None = None
        for packet in packets:
            if last is not None and packet.timestamp_s < last.timestamp_s:
                raise ValueError("decoder yielded non-monotonic timestamps")
            while cursor < len(requested) and requested[cursor] < packet.timestamp_s:
                t_q = requested[cursor]
                output[t_q] = self.snapshot(
                    writer, name=f"{snapshot_prefix}/rho_{t_q / self.meta.duration_s:.2f}",
                    t_q=t_q, config=config,
                )
                cursor += 1
            self.observe(packet)
            last = packet
            while cursor < len(requested) and requested[cursor] == packet.timestamp_s:
                t_q = requested[cursor]
                output[t_q] = self.snapshot(
                    writer, name=f"{snapshot_prefix}/rho_{t_q / self.meta.duration_s:.2f}",
                    t_q=t_q, config=config,
                )
                cursor += 1
        while cursor < len(requested):
            if last is None:
                raise ValueError("cannot snapshot an empty stream")
            t_q = requested[cursor]
            output[t_q] = self.snapshot(
                writer, name=f"{snapshot_prefix}/rho_{t_q / self.meta.duration_s:.2f}",
                t_q=t_q, config=config,
            )
            cursor += 1
        return output
