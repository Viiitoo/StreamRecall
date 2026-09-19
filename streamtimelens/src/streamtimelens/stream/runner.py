"""One-pass ingestion runner that emits several snapshots in the same decode."""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Any, Iterable

from streamtimelens.memory.budget import BudgetLedger
from streamtimelens.memory.cards import EvidenceCard
from streamtimelens.memory.boundary_allocator import BoundaryFrameAllocator
from streamtimelens.memory.eviction import UtilityEvictor, UtilityWeights
from streamtimelens.memory.forest import EventForest
from streamtimelens.memory.merge_policy import MergeCandidateQueue
from streamtimelens.memory.raw_cache import RawFrameCache
from streamtimelens.evaluation.cost import ComponentTimer
from streamtimelens.evaluation.trace import TraceLog
from streamtimelens.observer.normalizer import OnlineNormalizer
from streamtimelens.observer.quota import WriterQuota
from streamtimelens.observer.segment import ActiveSegmentReservoir
from streamtimelens.observer.signals import LiteSignalObserver
from streamtimelens.observer.trigger import JointTrigger
from streamtimelens.protocol.snapshot import SnapshotManifest, SnapshotWriter
from streamtimelens.protocol.types import Budget, FramePacket, StreamEvent, VideoMeta
from streamtimelens.writer.card_builder import build_evidence_cards
from streamtimelens.writer.parse import parse_writer_output
from streamtimelens.writer.prompts import build_writer_prompt


@dataclass
class IngestConfig:
    method_name: str = "full"
    ring_window_s: float = 12.0
    max_active_frames: int = 32
    decision_interval_s: float = 4.0
    trigger_mode: str = "periodic"
    trigger_threshold: float = 0.65
    minimum_gap_s: float = 1.0
    max_gap_s: float = 60.0
    lite_weight: float = 0.45
    semantic_weight: float = 0.35
    age_weight: float = 0.20
    initial_writer_tokens: float = 1.0
    segment_overlap_s: float = 1.0
    segment_bytes_per_frame: int = 16 * 1024
    hard_cut_hsv_threshold: float = 0.72
    hard_cut_ssim_threshold: float = 0.65
    black_luma_threshold: float = 2.0
    normalizer_alpha: float = 0.05
    normalizer_warmup_samples: int = 8
    normalizer_epsilon: float = 1e-6
    normalizer_z_threshold: float = 2.0
    warmup_hsv_threshold: float = 0.45
    warmup_ssim_threshold: float = 0.35
    warmup_flow_threshold: float = 1.5
    source_revision: str = "unknown"
    boundary_window_s: float = 2.0
    boundary_frames_per_side: int = 4
    boundary_internal_frames: int = 2
    forest_max_roots: int = 8
    merge_semantic_weight: float = 1.0
    merge_gap_weight: float = 0.35
    merge_boundary_weight: float = 0.35
    utility_novelty_weight: float = 0.30
    utility_boundary_weight: float = 0.30
    utility_inverse_density_weight: float = 0.25
    utility_has_raw_weight: float = 0.15
    coverage_bucket_base_s: float = 1.0
    forest_debug: bool = False

    def __post_init__(self) -> None:
        if self.method_name not in ("full", "streamtimelens", "evidence_fixed"):
            raise ValueError("invalid structured evidence method name")


class StreamingIngestor:
    """Query-blind state machine.

    `run` accepts only frames and requested snapshot times.  It intentionally
    has no query argument, which is the first protocol boundary enforced by
    P0; all query-related code is in `retrieval` and runs over `SnapshotReader`.
    """

    def __init__(
        self,
        meta: VideoMeta,
        budget: Budget,
        config: IngestConfig | None = None,
        *,
        evidence_writer: Any | None = None,
        text_embedder: Any | None = None,
    ) -> None:
        self.meta, self.budget, self.config = meta, budget, config or IngestConfig()
        self.evidence_writer = evidence_writer
        self.text_embedder = text_embedder
        self.ring: deque[FramePacket] = deque()
        self.raw = RawFrameCache()
        self.segment = ActiveSegmentReservoir(
            self.raw, remaining_budget_bytes=budget.memory_bytes,
            bytes_per_frame_estimate=self.config.segment_bytes_per_frame,
            overlap_s=self.config.segment_overlap_s,
            max_capacity=self.config.max_active_frames,
        )
        self.boundary_allocator = BoundaryFrameAllocator(
            self.raw, boundary_window_s=self.config.boundary_window_s,
            max_per_boundary=self.config.boundary_frames_per_side,
            max_internal=self.config.boundary_internal_frames,
        )
        self.ledger = BudgetLedger(budget.memory_bytes)
        self.forest = EventForest(
            budget.memory_bytes, raw_cache=self.raw,
            merge_queue=MergeCandidateQueue(
                semantic_weight=self.config.merge_semantic_weight,
                gap_weight=self.config.merge_gap_weight,
                boundary_weight=self.config.merge_boundary_weight,
            ),
            evictor=UtilityEvictor(
                UtilityWeights(
                    novelty=self.config.utility_novelty_weight,
                    boundary=self.config.utility_boundary_weight,
                    inverse_density=self.config.utility_inverse_density_weight,
                    has_raw=self.config.utility_has_raw_weight,
                ),
                bucket_base_s=self.config.coverage_bucket_base_s,
            ),
            debug=self.config.forest_debug,
            require_embeddings=self.text_embedder is not None,
        )
        initial_tokens = self.config.initial_writer_tokens if budget.writer_calls_per_minute > 0 else 0.0
        self.quota = WriterQuota(budget.writer_calls_per_minute, initial_tokens=initial_tokens)
        self.trigger = JointTrigger(
            self.config.trigger_mode, threshold=self.config.trigger_threshold,
            periodic_interval_s=self.config.decision_interval_s,
            minimum_gap_s=self.config.minimum_gap_s, max_gap_s=self.config.max_gap_s,
            lite_weight=self.config.lite_weight, semantic_weight=self.config.semantic_weight,
            age_weight=self.config.age_weight,
        )
        self.signals = LiteSignalObserver(
            hard_cut_hsv_threshold=self.config.hard_cut_hsv_threshold,
            hard_cut_ssim_threshold=self.config.hard_cut_ssim_threshold,
            black_luma_threshold=self.config.black_luma_threshold,
        )
        self.normalizer = OnlineNormalizer(
            alpha=self.config.normalizer_alpha,
            warmup_samples=self.config.normalizer_warmup_samples,
            epsilon=self.config.normalizer_epsilon,
            z_threshold=self.config.normalizer_z_threshold,
            fixed_thresholds={
                "hsv": self.config.warmup_hsv_threshold,
                "ssim": self.config.warmup_ssim_threshold,
                "flow": self.config.warmup_flow_threshold,
            },
        )
        self.trace = TraceLog()
        self._card_index = 0
        self._merge_trace_cursor = 0
        self._eviction_trace_cursor = 0

    @property
    def cards(self) -> list[EvidenceCard]:
        """Stable compatibility view over every query-visible forest node."""
        return list(self.forest.cards)

    def _record(self, event: StreamEvent, **details: object) -> None:
        self.trace.record(kind=event.kind, timestamp_s=event.timestamp_s, reason=event.reason,
                          object_id=event.object_id, **details)

    def _estimated_components(self) -> dict[str, int]:
        return {
            **self.forest.accounting_components(),
            "working_set": 16 * (len(self.ring) + len(self.segment)),
        }

    def _flush_forest_trace(self, timestamp_s: float) -> None:
        for row in self.forest.merge_trace[self._merge_trace_cursor:]:
            details = dict(row)
            details["merge_reason"] = details.pop("reason")
            self._record(StreamEvent(
                "card_merged", timestamp_s, row["mode"], row["parent_id"],
            ), **details)
        self._merge_trace_cursor = len(self.forest.merge_trace)
        for row in self.forest.eviction_trace[self._eviction_trace_cursor:]:
            details = dict(row)
            details["eviction_reason"] = details.pop("reason")
            details.pop("object_id", None)
            self._record(StreamEvent(
                "payload_evicted", timestamp_s, row["reason"], row["object_id"],
            ), **details)
        self._eviction_trace_cursor = len(self.forest.eviction_trace)

    def _drop_frame_references(self, frame_index: int) -> None:
        frame_id = f"{frame_index:09d}.jpg"
        self.ring = deque((packet for packet in self.ring if packet.frame_index != frame_index))
        self.segment.discard(frame_id)
        for card in self.forest.cards:
            changed = frame_id in card.raw_ref_ids
            card.mark_raw_ref(frame_id, "evicted")
            card.raw_ref_ids = [value for value in card.raw_ref_ids if value != frame_id]
            for key in ("left_frame_ids", "right_frame_ids", "internal_frame_ids"):
                card.boundary_cache[key] = [
                    value for value in card.boundary_cache.get(key, []) if value != frame_id
                ]
            card.boundary_cache["left_hit"] = any(
                card.raw_ref_status.get(ref) == "available"
                for ref in card.boundary_cache.get("left_frame_ids", [])
            )
            card.boundary_cache["right_hit"] = any(
                card.raw_ref_status.get(ref) == "available"
                for ref in card.boundary_cache.get("right_frame_ids", [])
            )
            card.boundary_cache["both_hit"] = bool(
                card.boundary_cache.get("left_hit") and card.boundary_cache.get("right_hit")
            )
            if changed:
                self.forest.store.touch(card.id)
                self.forest.merge_queue.invalidate(card.id)

    def _enforce_budget(self, timestamp_s: float) -> None:
        components = self._estimated_components()
        while sum(components.values()) + self.ledger.reserved_bytes > self.budget.memory_bytes:
            evicted = self.raw.evict_oldest(exclude_owner_prefix="card:")
            if evicted is not None:
                self._drop_frame_references(evicted.frame_index)
                self._record(StreamEvent("payload_evicted", timestamp_s, "raw_budget", evicted.frame_id))
            else:
                before = self.forest.state_bytes
                if len(self.forest.roots) >= 2:
                    self.forest.merge_best(force_hard=True)
                if self.forest.state_bytes >= before:
                    record = self.forest.evictor.evict_raw(self.forest)
                    if record is not None:
                        self.forest.eviction_trace.append(record.__dict__)
                    else:
                        record = self.forest.evictor.evict_card(self.forest)
                        if record is None:
                            raise MemoryError("ingest state cannot satisfy memory budget")
                        self.forest.eviction_trace.append(record.__dict__)
                self._flush_forest_trace(timestamp_s)
            components = self._estimated_components()
        self.ledger.replace_components(components)
        self.ledger.assert_within_budget()
        self.forest.assert_invariants(current_stream_time=timestamp_s, enforce_budget=False)

    @staticmethod
    def _normalized_lite_score(values: dict[str, object]) -> float:
        scores: list[float] = []
        for item in values.values():
            ready = bool(getattr(item, "ready"))
            if ready:
                z_score = float(getattr(item, "z_score"))
                scores.append(min(1.0, max(0.0, z_score / 2.0)))
            else:
                scores.append(1.0 if bool(getattr(item, "active")) else 0.0)
        return max(scores, default=0.0)

    def observe(
        self, packet: FramePacket, *, semantic_score: float | None = None,
        semantic_embedding: object | None = None,
    ) -> list[StreamEvent]:
        before = len(self.trace)
        self._record(StreamEvent("frame_seen", packet.timestamp_s, packet.source, str(packet.frame_index)))
        self.ring.append(packet)
        self.raw.add(packet, owner="now_ring")
        while self.ring and packet.timestamp_s - self.ring[0].timestamp_s > self.config.ring_window_s:
            expired = self.ring.popleft()
            self.raw.release(f"{expired.frame_index:09d}.jpg", "now_ring")
        remaining = max(0, self.budget.memory_bytes - self.raw.byte_size)
        self.segment.observe(
            packet, semantic_embedding, novelty_score=semantic_score,
            remaining_budget_bytes=remaining,
        )

        lite = self.signals.observe(packet.image, packet.timestamp_s)
        normalized = self.normalizer.observe({
            "hsv": lite.hsv_distance, "ssim": lite.ssim_change, "flow": lite.flow_p90,
        })
        lite_score = self._normalized_lite_score(normalized)
        proposal = self.trigger.propose(
            packet.timestamp_s, lite_score=lite_score,
            semantic_score=semantic_score, hard_cut=lite.hard_cut,
        )
        if proposal:
            reason = ",".join(proposal.reasons)
            self._record(
                StreamEvent("trigger_proposed", packet.timestamp_s, reason),
                trigger_score=proposal.score, trigger_reasons=proposal.reasons,
                signal_statuses=lite.statuses,
            )
            if not self.segment.writable:
                self._record(StreamEvent("trigger_dropped", packet.timestamp_s, "segment_too_short"))
            elif self.quota.try_consume(packet.timestamp_s):
                self._card_index += 1
                writer_frames = self.segment.writer_packets()
                chunk_id = f"chunk-{self._card_index:06d}"
                writer_details: dict[str, Any]
                if self.evidence_writer is None:
                    with ComponentTimer("writer_fallback") as timer:
                        span = (writer_frames[0].timestamp_s, writer_frames[-1].timestamp_s)
                        prompt = build_writer_prompt(
                            segment=span,
                            sampled_timestamps=[frame.timestamp_s for frame in writer_frames],
                        )
                        parsed = parse_writer_output(
                            "", segment=span,
                            sampled_timestamps=[frame.timestamp_s for frame in writer_frames],
                        )
                        new_cards = build_evidence_cards(
                            parsed, source_chunk_id=chunk_id, segment=span, prompt=prompt,
                            writer_revision="deterministic-fallback", generated_tokens=0,
                        )
                    writer_details = {
                        "parse_status": "fallback", "raw_output": "", "parse_error": parsed.error,
                        "generated_tokens": 0, "resource": timer.as_dict(),
                        "prompt_version": prompt.version, "prompt_hash": prompt.template_sha256,
                    }
                else:
                    try:
                        result = self.evidence_writer.write(chunk_id, writer_frames, self.meta)
                        new_cards = list(result.cards)
                        writer_details = {
                            "parse_status": result.parse_status, "raw_output": result.raw_output,
                            "parse_error": result.error, "prompt_version": result.prompt.version,
                            "prompt_hash": result.prompt.template_sha256, **result.stats,
                        }
                    except Exception as exc:
                        span = (writer_frames[0].timestamp_s, writer_frames[-1].timestamp_s)
                        prompt = build_writer_prompt(
                            segment=span,
                            sampled_timestamps=[frame.timestamp_s for frame in writer_frames],
                        )
                        parsed = parse_writer_output(
                            "", segment=span,
                            sampled_timestamps=[frame.timestamp_s for frame in writer_frames],
                        )
                        new_cards = build_evidence_cards(
                            parsed, source_chunk_id=chunk_id, segment=span, prompt=prompt,
                            writer_revision="writer-error-fallback", generated_tokens=0,
                            writer_provenance={"writer_error": type(exc).__name__},
                        )
                        writer_details = {
                            "parse_status": "fallback", "raw_output": "",
                            "parse_error": f"{type(exc).__name__}: {exc}",
                            "generated_tokens": 0, "writer_failed": True,
                            "prompt_version": prompt.version, "prompt_hash": prompt.template_sha256,
                        }
                if self.text_embedder is not None and new_cards:
                    try:
                        self.text_embedder.embed_cards(new_cards)
                    except Exception as exc:
                        writer_details["embedding_error"] = f"{type(exc).__name__}: {exc}"
                candidates = (*self.ring, *self.segment.frames)
                for card in new_cards:
                    self.forest.insert(card)
                non_raw_bytes = sum(
                    value for key, value in self._estimated_components().items() if key != "raw_frames"
                )
                raw_budget = max(0, self.budget.memory_bytes - non_raw_bytes)
                allocations = []
                for card in new_cards:
                    persistent_bytes = self.raw.bytes_owned_by_prefix("card:")
                    allocation = self.boundary_allocator.allocate(
                        card, candidates, budget_bytes=max(0, raw_budget - persistent_bytes),
                    )
                    allocations.append({
                        "card_id": card.id, "left_hit": allocation.left_hit,
                        "right_hit": allocation.right_hit, "both_hit": allocation.both_hit,
                        "selected_frame_ids": allocation.selected_frame_ids,
                        "rejected_frame_ids": allocation.rejected_frame_ids,
                        "bytes_admitted": allocation.bytes_admitted,
                    })
                    self.forest.store.touch(card.id)
                    self.forest.merge_queue.invalidate(card.id)
                while len(self.forest.roots) > self.config.forest_max_roots:
                    if self.forest.merge_best() is None:
                        break
                self._flush_forest_trace(packet.timestamp_s)
                self.ledger.writer_calls += 1
                self.trigger.mark_written(packet.timestamp_s)
                overlap_ids = self.segment.mark_written(packet.timestamp_s)
                object_id = new_cards[0].id if new_cards else chunk_id
                self._record(StreamEvent("writer_called", packet.timestamp_s, reason, object_id), input_frames=len(writer_frames),
                             emitted_cards=len(new_cards), segment_span=(writer_frames[0].timestamp_s, writer_frames[-1].timestamp_s),
                             overlap_frame_ids=overlap_ids, boundary_allocations=allocations,
                             trigger_score=proposal.score, trigger_reasons=proposal.reasons,
                             **writer_details)
                for card in new_cards:
                    self._record(StreamEvent("card_inserted", packet.timestamp_s, writer_details["parse_status"], card.id))
            else:
                self._record(
                    StreamEvent("trigger_dropped", packet.timestamp_s, f"writer_quota:{self.quota.last_reason}"),
                    trigger_score=proposal.score, trigger_reasons=proposal.reasons,
                )
        self._enforce_budget(packet.timestamp_s)
        return [StreamEvent(row["kind"], row["t"], row["reason"], row.get("object_id")) for row in self.trace[before:]]

    def snapshot(
        self, writer: SnapshotWriter, *, name: str, t_q: float, config: dict[str, object]
    ) -> SnapshotManifest:
        # `raw` includes the current ring.  Calling this before later frames are
        # observed prevents snapshots at different rho values from re-decoding.
        # Promote the query-visible ring before serializing it.  These are
        # references to the same deduplicated frame store, not a second cache.
        for packet in self.ring:
            self.raw.add(packet, owner="now_ring")
        self._enforce_budget(t_q)
        self.ledger.assert_within_budget()
        while True:
            try:
                manifest = writer.write(
                    name=name, t_q=t_q, meta=self.meta, budget=self.budget,
                    cards=self.forest.store.rows(), raw_frames=self.raw.items(),
                    raw_metadata=self.raw.metadata(), writer_calls=self.ledger.writer_calls,
                    config=config,
                    method=self.config.method_name,
                    embedder=(
                        dict(vars(self.text_embedder.metadata))
                        if self.text_embedder is not None and hasattr(self.text_embedder, "metadata")
                        else None
                    ),
                )
                break
            except MemoryError:
                evicted = self.raw.evict_oldest(exclude_owner_prefix="card:")
                if evicted is None:
                    evicted = self.raw.evict_oldest()
                if evicted is not None:
                    self._drop_frame_references(evicted.frame_index)
                    self._record(StreamEvent(
                        "payload_evicted", t_q, "snapshot_filesystem_budget", evicted.frame_id,
                    ))
                else:
                    record = self.forest.evictor.evict_raw(self.forest)
                    if record is None:
                        record = self.forest.evictor.evict_card(self.forest)
                    if record is None:
                        raise
                    self.forest.eviction_trace.append(record.__dict__)
                    self._flush_forest_trace(t_q)
                self._enforce_budget(t_q)
        filesystem_bytes = self.ledger.reconcile(writer.output_root / name)
        if filesystem_bytes != manifest.state_bytes:
            raise RuntimeError("snapshot writer returned an inconsistent state byte count")
        self.forest.assert_invariants(
            current_stream_time=t_q, allowed_raw_refs=set(self.raw.metadata()),
        )
        self._record(StreamEvent("snapshot_written", t_q, "arrival", name), logical_state_bytes=self.ledger.logical_bytes,
                     snapshot_filesystem_bytes=filesystem_bytes)
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
        last_packet: FramePacket | None = None
        for packet in packets:
            if last_packet is not None and packet.timestamp_s < last_packet.timestamp_s:
                raise ValueError("decoder yielded non-monotonic timestamps")
            # Materialize arrivals strictly before this packet from the prior
            # state so no frame after t_q can enter M_tq.
            while cursor < len(requested) and requested[cursor] < packet.timestamp_s:
                t_q = requested[cursor]
                output[t_q] = self.snapshot(writer, name=f"{snapshot_prefix}/rho_{t_q / self.meta.duration_s:.2f}", t_q=t_q, config=config)
                cursor += 1
            self.observe(packet)
            last_packet = packet
            # A frame whose timestamp equals t_q is part of the legal history.
            while cursor < len(requested) and requested[cursor] == packet.timestamp_s:
                t_q = requested[cursor]
                output[t_q] = self.snapshot(writer, name=f"{snapshot_prefix}/rho_{t_q / self.meta.duration_s:.2f}", t_q=t_q, config=config)
                cursor += 1
        while cursor < len(requested):
            # A stream with a terminal timestamp slightly below duration still
            # gets a legal end snapshot from its final observed state.
            if last_packet is None:
                raise ValueError("cannot snapshot an empty stream")
            t_q = requested[cursor]
            output[t_q] = self.snapshot(writer, name=f"{snapshot_prefix}/rho_{t_q / self.meta.duration_s:.2f}", t_q=t_q, config=config)
            cursor += 1
        return output
