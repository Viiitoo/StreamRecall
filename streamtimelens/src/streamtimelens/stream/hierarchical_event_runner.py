"""Single-pass CLIP ingestion into HEM-01 event snapshots."""

from __future__ import annotations

import math
from io import BytesIO
from typing import Any, Iterable, Mapping

from streamtimelens.memory.hierarchical_event import (
    HEM_SCHEMA_VERSION,
    HierarchicalEventMemory,
    write_event_snapshot,
)
from streamtimelens.protocol.snapshot import SnapshotManifest, SnapshotWriter
from streamtimelens.protocol.types import Budget, FramePacket, VideoMeta


def _decode_rgb(packet: FramePacket) -> Any:
    try:
        from PIL import Image

        with Image.open(BytesIO(packet.image)) as image:
            return image.convert("RGB").copy()
    except OSError as exc:
        raise ValueError(f"frame {packet.frame_index} is not decodable") from exc


class HierarchicalEventIngestor:
    """Query-blind online encoder whose persistent state is only event records."""

    def __init__(
        self, meta: VideoMeta, budget: Budget, *, clip_encoder: Any,
        source_revision: str,
    ) -> None:
        if clip_encoder is None or not source_revision:
            raise ValueError("HEM-01 requires frozen visual encoding and a source revision")
        self.meta = meta
        self.budget = budget
        self.clip_encoder = clip_encoder
        reset_clock = getattr(self.clip_encoder, "reset_visual_clock", None)
        if reset_clock is not None:
            reset_clock()
        self.source_revision = source_revision
        self.memory = HierarchicalEventMemory()
        self._pending: list[tuple[FramePacket, Any]] = []

    def observe(self, packet: FramePacket) -> None:
        if not self.clip_encoder.due(packet.timestamp_s):
            return
        self._pending.append((packet, _decode_rgb(packet)))
        if len(self._pending) >= int(self.clip_encoder.batch_size):
            self.flush()

    def flush(self) -> None:
        if not self._pending:
            return
        pending, self._pending = self._pending, []
        embeddings = self.clip_encoder.encode_images([image for _, image in pending])
        if len(embeddings) != len(pending):
            raise RuntimeError("HEM-01 encoder returned the wrong batch length")
        for (packet, _), embedding in zip(pending, embeddings):
            self.memory.observe(
                timestamp_s=packet.timestamp_s, frame_index=packet.frame_index,
                embedding=embedding,
            )

    def snapshot(
        self, writer: SnapshotWriter, *, name: str, t_q: float,
        config: Mapping[str, Any],
    ) -> SnapshotManifest:
        self.flush()
        return write_event_snapshot(
            self.memory, writer, name=name, t_q=t_q, meta=self.meta,
            budget=self.budget, config=dict(config), source_revision=self.source_revision,
        )

    def run(
        self, packets: Iterable[FramePacket], arrival_times: Iterable[float],
        writer: SnapshotWriter, *, config: Mapping[str, Any],
        snapshot_prefix: str = "snapshot",
    ) -> dict[float, SnapshotManifest]:
        requested = sorted(set(map(float, arrival_times)))
        if any(
            not math.isfinite(value) or value <= 0 or value > self.meta.duration_s + 1e-6
            for value in requested
        ):
            raise ValueError("HEM-01 snapshot time is outside video duration")
        output: dict[float, SnapshotManifest] = {}
        cursor = 0
        previous: FramePacket | None = None
        for packet in packets:
            if previous is not None and packet.timestamp_s < previous.timestamp_s:
                raise ValueError("HEM-01 decoder yielded non-monotonic timestamps")
            while cursor < len(requested) and requested[cursor] < packet.timestamp_s:
                t_q = requested[cursor]
                output[t_q] = self.snapshot(
                    writer, name=f"{snapshot_prefix}/rho_{t_q / self.meta.duration_s:.2f}",
                    t_q=t_q, config=config,
                )
                cursor += 1
            self.observe(packet)
            previous = packet
            while cursor < len(requested) and requested[cursor] == packet.timestamp_s:
                t_q = requested[cursor]
                output[t_q] = self.snapshot(
                    writer, name=f"{snapshot_prefix}/rho_{t_q / self.meta.duration_s:.2f}",
                    t_q=t_q, config=config,
                )
                cursor += 1
        while cursor < len(requested):
            t_q = requested[cursor]
            output[t_q] = self.snapshot(
                writer, name=f"{snapshot_prefix}/rho_{t_q / self.meta.duration_s:.2f}",
                t_q=t_q, config=config,
            )
            cursor += 1
        if len(output) != len(requested):
            raise RuntimeError("HEM-01 did not write every requested snapshot")
        return output


def hem_ingest_config() -> dict[str, Any]:
    return {
        "method": "HEM-01",
        "event_schema": HEM_SCHEMA_VERSION,
        "query_independent": True,
        "writer_calls": 0,
    }
