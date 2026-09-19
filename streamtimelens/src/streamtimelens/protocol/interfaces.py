"""Public interfaces that preserve the ingest/query separation."""

from __future__ import annotations

from typing import Protocol

from .snapshot import SnapshotManifest, SnapshotReader
from .types import Budget, FramePacket, Prediction, StreamEvent, VideoMeta


class StreamMethod(Protocol):
    def reset(self, video_meta: VideoMeta, budget: Budget) -> None:
        ...

    def observe(self, packet: FramePacket) -> list[StreamEvent]:
        ...

    def snapshot(self, t_q: float) -> SnapshotManifest:
        ...


class QueryMethod(Protocol):
    def locate(self, query: str, snapshot: SnapshotReader) -> Prediction:
        ...
