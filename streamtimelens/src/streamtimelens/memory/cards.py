"""Query-independent, snapshot-ready evidence card records."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable

from streamtimelens.protocol.types import FramePacket


@dataclass
class EvidenceCard:
    id: str
    level: int
    t_start: float
    t_end: float
    summary: str
    actors: list[str] = field(default_factory=list)
    actions: list[str] = field(default_factory=list)
    objects: list[str] = field(default_factory=list)
    scene: str = "unknown"
    support_timestamps: list[float] = field(default_factory=list)
    left_uncertainty_s: float = 0.0
    right_uncertainty_s: float = 0.0
    raw_ref_ids: list[str] = field(default_factory=list)
    child_ids: list[str] = field(default_factory=list)
    source_chunk_ids: list[str] = field(default_factory=list)
    phase: str = "complete"
    normalized_text: str = ""
    text_embedding: dict[str, Any] | None = None
    visual_centroid: dict[str, Any] | None = None
    writer_model_revision: str = "unknown"
    prompt_version: str = "unknown"
    prompt_hash: str = "unknown"
    writer_provenance: dict[str, Any] = field(default_factory=dict)
    generated_tokens: int = 0
    raw_ref_status: dict[str, str] = field(default_factory=dict)
    boundary_cache: dict[str, Any] = field(default_factory=dict)
    merge_mode: str = "leaf"
    compacted_child_count: int = 0
    byte_size: int = 0

    def serializable(self) -> dict:
        result = asdict(self)
        size = 0
        for _ in range(4):
            result["byte_size"] = size
            actual = len(json.dumps(
                result, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                allow_nan=False,
            ).encode("utf-8"))
            if actual == size:
                break
            size = actual
        result["byte_size"] = size
        self.byte_size = size
        return result

    def mark_raw_ref(self, frame_id: str, status: str) -> None:
        if frame_id in self.raw_ref_ids:
            self.raw_ref_status[frame_id] = status


def unknown_event_card(card_id: str, frames: Iterable[FramePacket]) -> EvidenceCard:
    selected = list(frames)
    if not selected:
        raise ValueError("writer needs at least one observed frame")
    timestamps = [item.timestamp_s for item in selected]
    refs = [f"{item.frame_index:09d}.jpg" for item in selected]
    return EvidenceCard(
        id=card_id, level=0, t_start=min(timestamps), t_end=max(timestamps),
        summary="unknown_event", support_timestamps=[timestamps[0], timestamps[-1]],
        left_uncertainty_s=0.0, right_uncertainty_s=0.0, raw_ref_ids=refs,
        source_chunk_ids=[card_id], phase="ongoing", normalized_text="unknown_event",
        raw_ref_status={frame_id: "available" for frame_id in refs},
    )
