"""Validated, versioned objects shared by physically separated processes."""

from __future__ import annotations

import base64
import json
import math
from dataclasses import asdict, dataclass
from typing import Any, ClassVar, Literal, Type, TypeVar


SCHEMA_VERSION = 1
T = TypeVar("T", bound="PersistentRecord")


def _finite(name: str, value: float, *, positive: bool = False, non_negative: bool = False) -> float:
    numeric = float(value)
    if not math.isfinite(numeric) or (positive and numeric <= 0) or (non_negative and numeric < 0):
        raise ValueError(f"{name} must be finite and valid")
    return numeric


class PersistentRecord:
    record_type: ClassVar[str]

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result.update({"schema_version": SCHEMA_VERSION, "record_type": self.record_type})
        return result

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, ensure_ascii=False)

    def to_msgpack(self) -> bytes:
        try:
            import msgpack
        except ImportError as exc:  # pragma: no cover - dependency declared by package
            raise RuntimeError("msgpack is required for MessagePack serialization") from exc
        return msgpack.packb(self.to_dict(), use_bin_type=True)

    @classmethod
    def from_json(cls: Type[T], payload: str) -> T:
        return cls.from_dict(json.loads(payload))

    @classmethod
    def from_msgpack(cls: Type[T], payload: bytes) -> T:
        try:
            import msgpack
        except ImportError as exc:  # pragma: no cover - dependency declared by package
            raise RuntimeError("msgpack is required for MessagePack serialization") from exc
        return cls.from_dict(msgpack.unpackb(payload, raw=False))

    @classmethod
    def _clean(cls, data: dict[str, Any]) -> dict[str, Any]:
        if data.get("schema_version") != SCHEMA_VERSION or data.get("record_type") != cls.record_type:
            raise ValueError(f"unsupported {cls.record_type} schema")
        return {key: value for key, value in data.items() if key not in {"schema_version", "record_type"}}


@dataclass(frozen=True)
class VideoMeta(PersistentRecord):
    record_type: ClassVar[str] = "video_meta"
    video_id: str
    duration_s: float
    original_fps: float
    total_num_frames: int
    width: int = 0
    height: int = 0

    def __post_init__(self) -> None:
        if not self.video_id or self.total_num_frames < 0 or self.width < 0 or self.height < 0:
            raise ValueError("invalid video metadata")
        _finite("duration_s", self.duration_s, positive=True)
        _finite("original_fps", self.original_fps, positive=True)

    @property
    def fps(self) -> float:
        return self.original_fps

    @property
    def total_frames(self) -> int:
        return self.total_num_frames

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "VideoMeta":
        return cls(**cls._clean(data))


@dataclass(frozen=True)
class Budget:
    memory_bytes: int
    writer_calls_per_minute: float
    refine_calls_per_query: int = 1
    max_frames_per_refine: int = 16

    def __post_init__(self) -> None:
        if self.memory_bytes <= 0 or self.writer_calls_per_minute < 0:
            raise ValueError("memory_bytes must be positive and writer budget non-negative")


@dataclass(frozen=True)
class FramePacket(PersistentRecord):
    record_type: ClassVar[str] = "frame_packet"
    timestamp_s: float
    frame_index: int
    image: bytes
    width: int
    height: int
    source: str = "decoder"
    video_id: str = ""

    def __post_init__(self) -> None:
        _finite("timestamp_s", self.timestamp_s, non_negative=True)
        if self.frame_index < 0 or self.width < 0 or self.height < 0 or not isinstance(self.image, bytes):
            raise ValueError("invalid frame packet")

    def to_dict(self) -> dict[str, Any]:
        result = super().to_dict()
        result["image_b64"] = base64.b64encode(self.image).decode("ascii")
        result.pop("image")
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FramePacket":
        clean = cls._clean(data)
        try:
            clean["image"] = base64.b64decode(clean.pop("image_b64"), validate=True)
        except (KeyError, ValueError) as exc:
            raise ValueError("invalid frame image") from exc
        return cls(**clean)


@dataclass(frozen=True)
class QueryRecord(PersistentRecord):
    record_type: ClassVar[str] = "query_record"
    query_id: str
    video_id: str
    query: str
    gt_span: tuple[float, float]
    t_q: float
    rho_q: float
    cohort: str

    def __post_init__(self) -> None:
        if not self.query_id or not self.video_id or not self.query or not self.cohort:
            raise ValueError("query record identifiers and text are required")
        start, end = map(float, self.gt_span)
        _finite("gt_start", start, non_negative=True)
        _finite("gt_end", end, non_negative=True)
        if start >= end:
            raise ValueError("ground-truth span must be ordered")
        _finite("t_q", self.t_q, non_negative=True)
        if not 0 < _finite("rho_q", self.rho_q, positive=True) <= 1:
            raise ValueError("rho_q must be in (0,1]")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "QueryRecord":
        clean = cls._clean(data)
        clean["gt_span"] = tuple(clean["gt_span"])
        return cls(**clean)


@dataclass(frozen=True)
class StreamEvent:
    kind: Literal[
        "frame_seen", "clip_encoded", "frame_admitted", "frame_replaced",
        "frame_evicted", "trigger_proposed", "trigger_dropped", "writer_called",
        "card_inserted", "raw_cached", "nodes_merged", "payload_evicted",
        "snapshot_written",
    ]
    timestamp_s: float
    reason: str
    object_id: str | None = None


@dataclass(frozen=True)
class Prediction(PersistentRecord):
    record_type: ClassVar[str] = "prediction"
    start_s: float | None
    end_s: float | None
    confidence: float
    status: Literal["ok", "NOT_FOUND", "fallback"]
    evidence_ids: tuple[str, ...] = ()
    query_id: str | None = None
    candidate_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not 0 <= _finite("confidence", self.confidence, non_negative=True) <= 1:
            raise ValueError("confidence must be in [0,1]")
        if (self.start_s is None) != (self.end_s is None):
            raise ValueError("prediction span must be fully present or absent")
        if self.start_s is not None:
            if _finite("start_s", self.start_s, non_negative=True) >= _finite("end_s", self.end_s, non_negative=True):
                raise ValueError("prediction span must be ordered")

    @property
    def span(self) -> tuple[float, float] | None:
        return None if self.start_s is None else (self.start_s, self.end_s)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Prediction":
        clean = cls._clean(data)
        clean["evidence_ids"] = tuple(clean.get("evidence_ids", ()))
        clean["candidate_ids"] = tuple(clean.get("candidate_ids", ()))
        return cls(**clean)
