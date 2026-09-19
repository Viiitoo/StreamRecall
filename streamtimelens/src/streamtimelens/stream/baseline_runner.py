"""One-pass snapshot runner for the Uniform and Semantic raw-frame baselines."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from io import BytesIO
from typing import Any, Iterable, Literal

from streamtimelens.evaluation.trace import TraceLog
from streamtimelens.memory.budget import BudgetLedger
from streamtimelens.memory.raw_cache import RawFrameCache
from streamtimelens.memory.semantic_reservoir import SemanticReservoir
from streamtimelens.memory.uniform import UniformRawCache
from streamtimelens.protocol.snapshot import SnapshotManifest, SnapshotWriter
from streamtimelens.protocol.types import Budget, FramePacket, VideoMeta


BaselineMethod = Literal["uniform_raw", "semantic_reservoir"]


@dataclass(frozen=True)
class BaselineIngestConfig:
    method: BaselineMethod
    capacity: int
    pixel_only: bool = False
    embedding_precision: str = "fp16"
    seed: int = 0
    anchor_fraction: float = 0.25
    source_revision: str = "unknown"

    def __post_init__(self) -> None:
        if self.method not in ("uniform_raw", "semantic_reservoir") or self.capacity <= 0:
            raise ValueError("invalid raw baseline configuration")
        if self.method == "semantic_reservoir" and self.pixel_only:
            raise ValueError("semantic reservoir cannot be pixel-only")


def _decode_rgb(packet: FramePacket) -> Any:
    try:
        from PIL import Image

        with Image.open(BytesIO(packet.image)) as image:
            return image.convert("RGB").copy()
    except OSError as exc:
        raise ValueError(f"frame {packet.frame_index} is not a decodable image") from exc


class RawBaselineIngestor:
    """Query-blind baseline state; its public API has no query or GT argument."""

    def __init__(
        self,
        meta: VideoMeta,
        budget: Budget,
        config: BaselineIngestConfig,
        *,
        clip_encoder: Any | None,
    ) -> None:
        if clip_encoder is None and not config.pixel_only:
            raise ValueError("an ingest-side CLIP encoder is required unless pixel_only is explicit")
        self.meta, self.budget, self.config = meta, budget, config
        self.clip_encoder = clip_encoder
        self.store = RawFrameCache()
        if config.method == "uniform_raw":
            self.cache: Any = UniformRawCache(
                self.store, config.capacity, duration_s=meta.duration_s, seed=config.seed,
                pixel_only=config.pixel_only, embedding_precision=config.embedding_precision,
            )
        else:
            self.cache = SemanticReservoir(
                self.store, config.capacity, duration_s=meta.duration_s,
                anchor_fraction=config.anchor_fraction,
                embedding_precision=config.embedding_precision,
            )
        self.ledger = BudgetLedger(budget.memory_bytes)
        self.trace = TraceLog()
        self._pending: list[tuple[FramePacket, Any]] = []
        self._last_pixel_timestamp: float | None = None

    def _due(self, timestamp_s: float) -> bool:
        if self.clip_encoder is not None:
            return bool(self.clip_encoder.due(timestamp_s))
        interval = 2.0  # explicit 0.5 FPS pixel-only baseline sampling
        if self._last_pixel_timestamp is None or timestamp_s + 1e-9 >= self._last_pixel_timestamp + interval:
            self._last_pixel_timestamp = timestamp_s
            return True
        return False

    def observe(self, packet: FramePacket) -> None:
        self.trace.record(kind="frame_seen", timestamp_s=packet.timestamp_s, reason=packet.source,
                          object_id=str(packet.frame_index))
        if not self._due(packet.timestamp_s):
            return
        if self.config.pixel_only:
            before_ids = {item.ref.frame_id for item in self.cache.frames}
            before_bytes = sum(self._logical_components().values())
            selected = self.cache.observe(packet, None)
            after_ids = {item.ref.frame_id for item in self.cache.frames}
            after_bytes = sum(self._logical_components().values())
            removed = sorted(before_ids - after_ids)
            self.trace.record(
                kind=("frame_replaced" if selected and removed else
                      "frame_admitted" if selected else "frame_rejected"),
                timestamp_s=packet.timestamp_s, reason="uniform_pixel_only",
                object_id=f"{packet.frame_index:09d}.jpg",
                replaced_frame_ids=removed,
                state_bytes_before=before_bytes, state_bytes_after=after_bytes,
            )
            self._enforce_logical_budget(packet.timestamp_s)
            return
        try:
            image = _decode_rgb(packet)
        except ValueError as exc:
            self.trace.record(kind="clip_failed", timestamp_s=packet.timestamp_s,
                              reason=str(exc), object_id=str(packet.frame_index))
            return
        self._pending.append((packet, image))
        if len(self._pending) >= int(self.clip_encoder.batch_size):
            self.flush()

    def flush(self) -> None:
        if not self._pending:
            return
        pending, self._pending = self._pending, []
        try:
            embeddings = self.clip_encoder.encode_images([image for _, image in pending])
        except Exception as exc:
            for packet, _ in pending:
                self.trace.record(kind="clip_failed", timestamp_s=packet.timestamp_s,
                                  reason=type(exc).__name__, object_id=str(packet.frame_index))
            raise
        if len(embeddings) != len(pending):
            raise RuntimeError("CLIP encoder returned the wrong batch length")
        all_resources = self.clip_encoder.resource_dicts()
        resource = all_resources[-1]
        clip_call_index = len(all_resources) - 1
        for (packet, _), embedding in zip(pending, embeddings):
            before_ids = {item.ref.frame_id for item in self.cache.frames}
            before_bytes = sum(self._logical_components().values())
            self.trace.record(
                kind="clip_encoded", timestamp_s=packet.timestamp_s,
                reason=self.config.method, object_id=str(packet.frame_index),
                state_bytes_before=before_bytes, state_bytes_after=before_bytes,
                resource=resource, clip_call_index=clip_call_index,
            )
            selected = self.cache.observe(packet, embedding)
            after_ids = {item.ref.frame_id for item in self.cache.frames}
            after_bytes = sum(self._logical_components().values())
            removed = sorted(before_ids - after_ids)
            self.trace.record(
                kind=("frame_replaced" if selected and removed else
                      "frame_admitted" if selected else "frame_rejected"),
                timestamp_s=packet.timestamp_s, reason=self.config.method,
                object_id=f"{packet.frame_index:09d}.jpg", resource=resource,
                clip_call_index=clip_call_index,
                replaced_frame_ids=removed,
                state_bytes_before=before_bytes, state_bytes_after=after_bytes,
            )
            self._enforce_logical_budget(packet.timestamp_s)

    def _logical_components(self) -> dict[str, int]:
        metadata_bytes = len(json.dumps(self.cache.metadata(), sort_keys=True).encode("utf-8"))
        return {
            "raw_frames": self.store.byte_size,
            "clip_embeddings": max(0, self.cache.logical_bytes - self.store.byte_size),
            "frame_metadata": metadata_bytes,
            "index": 32 * len(self.cache),
        }

    def _enforce_logical_budget(self, timestamp_s: float) -> None:
        components = self._logical_components()
        while sum(components.values()) > self.budget.memory_bytes:
            before_bytes = sum(components.values())
            removed = self.cache.evict_worst()
            if removed is None:
                break
            components = self._logical_components()
            self.trace.record(
                kind="frame_evicted", timestamp_s=timestamp_s,
                reason="baseline_memory_budget", object_id=removed.ref.frame_id,
                state_bytes_before=before_bytes,
                state_bytes_after=sum(components.values()),
            )
        self.ledger.replace_components(components)
        self.ledger.assert_within_budget()

    def snapshot(
        self, writer: SnapshotWriter, *, name: str, t_q: float, config: dict[str, object]
    ) -> SnapshotManifest:
        self.flush()
        self._enforce_logical_budget(t_q)
        while True:
            try:
                manifest = writer.write(
                    name=name, t_q=t_q, meta=self.meta, budget=self.budget, cards=[],
                    raw_frames=self.store.items(), raw_metadata=self.cache.metadata(),
                    writer_calls=0, config=config, method=self.config.method,
                    pixel_only=self.config.pixel_only,
                )
                break
            except MemoryError:
                before_bytes = sum(self._logical_components().values())
                removed = self.cache.evict_worst()
                if removed is None:
                    raise
                after_bytes = sum(self._logical_components().values())
                self.trace.record(
                    kind="frame_evicted", timestamp_s=t_q,
                    reason="snapshot_filesystem_budget", object_id=removed.ref.frame_id,
                    state_bytes_before=before_bytes, state_bytes_after=after_bytes,
                )
                self._enforce_logical_budget(t_q)
        filesystem_bytes = self.ledger.reconcile(writer.output_root / name)
        self.trace.record(
            kind="snapshot_written", timestamp_s=t_q, reason="arrival", object_id=name,
            logical_state_bytes=self.ledger.logical_bytes,
            snapshot_filesystem_bytes=filesystem_bytes, retained_frames=len(self.cache),
            state_bytes_before=self.ledger.logical_bytes,
            state_bytes_after=filesystem_bytes,
        )
        return manifest

    def run(
        self,
        packets: Iterable[FramePacket],
        arrival_times: Iterable[float],
        writer: SnapshotWriter,
        *,
        config: dict[str, object],
        snapshot_prefix: str = "snapshot",
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
            while cursor < len(requested) and requested[cursor] < packet.timestamp_s:
                t_q = requested[cursor]
                output[t_q] = self.snapshot(
                    writer, name=f"{snapshot_prefix}/rho_{t_q / self.meta.duration_s:.2f}",
                    t_q=t_q, config=config,
                )
                cursor += 1
            self.observe(packet)
            last_packet = packet
            while cursor < len(requested) and requested[cursor] == packet.timestamp_s:
                t_q = requested[cursor]
                output[t_q] = self.snapshot(
                    writer, name=f"{snapshot_prefix}/rho_{t_q / self.meta.duration_s:.2f}",
                    t_q=t_q, config=config,
                )
                cursor += 1
        while cursor < len(requested):
            if last_packet is None:
                raise ValueError("cannot snapshot an empty stream")
            t_q = requested[cursor]
            output[t_q] = self.snapshot(
                writer, name=f"{snapshot_prefix}/rho_{t_q / self.meta.duration_s:.2f}",
                t_q=t_q, config=config,
            )
            cursor += 1
        return output
