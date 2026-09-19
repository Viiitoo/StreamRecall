"""Protocol-separated adapter for official query-known OnVTG results."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Literal


QUERY_KNOWN_PROTOCOL = "query_known_onvtg_v1"


@dataclass(frozen=True)
class OnVTGReference:
    repository: Path
    config: Path
    checkpoint: Path
    repository_revision: str
    checkpoint_sha256: str
    feature_revision: str

    def __post_init__(self) -> None:
        if not self.repository.is_dir() or not self.config.is_file():
            raise ValueError("official OnVTG repository/config is unavailable")
        if not self.repository_revision or not self.checkpoint_sha256 or not self.feature_revision:
            raise ValueError("OnVTG reference revisions/hashes must be frozen")

    def evaluation_command(self, gpu_ids: str = "0") -> tuple[str, ...]:
        if not gpu_ids:
            raise ValueError("OnVTG GPU IDs cannot be empty")
        return (
            "env", f"CUDA_VISIBLE_DEVICES={gpu_ids}", "bash",
            str(self.repository / "scripts" / "eval.sh"),
            str(self.config), str(self.checkpoint),
        )


@dataclass(frozen=True)
class QueryKnownPrediction:
    query_id: str
    video_id: str
    span: tuple[float, float] | None
    confidence: float
    status: Literal["ok", "not_found", "error"]
    protocol: str = QUERY_KNOWN_PROTOCOL
    method: str = "onvtg-official"

    def __post_init__(self) -> None:
        if not self.query_id or not self.video_id or self.protocol != QUERY_KNOWN_PROTOCOL:
            raise ValueError("invalid query-known prediction identity/protocol")
        if not math.isfinite(self.confidence) or not 0 <= self.confidence <= 1:
            raise ValueError("invalid query-known confidence")
        if self.span is not None and (
            not all(math.isfinite(value) for value in self.span)
            or self.span[0] < 0 or self.span[0] >= self.span[1]
        ):
            raise ValueError("invalid query-known prediction span")

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["span"] = list(self.span) if self.span is not None else None
        return value


def adapt_official_onvtg_rows(rows: Iterable[dict[str, Any]]) -> list[QueryKnownPrediction]:
    """Normalize frozen official output without touching delayed-query tables."""
    predictions = []
    seen: set[str] = set()
    for row in rows:
        query_id = str(row.get("query_id", ""))
        if not query_id or query_id in seen:
            raise ValueError("official OnVTG rows need unique query IDs")
        seen.add(query_id)
        raw_span = row.get("span", row.get("pred_span"))
        span = tuple(map(float, raw_span)) if raw_span is not None else None
        if span is not None and len(span) != 2:
            raise ValueError("official OnVTG span must have two endpoints")
        status = str(row.get("status", "ok" if span is not None else "not_found")).lower()
        if status not in ("ok", "not_found", "error"):
            raise ValueError("official OnVTG status is unsupported")
        predictions.append(QueryKnownPrediction(
            query_id, str(row.get("video_id", "")), span,  # type: ignore[arg-type]
            float(row.get("confidence", row.get("score", 0.0))), status,  # type: ignore[arg-type]
        ))
    return predictions


def assert_protocol_table(rows: Iterable[dict[str, Any]], expected_protocol: str) -> None:
    protocols = {str(row.get("protocol", "")) for row in rows}
    if protocols != {expected_protocol}:
        raise ValueError("result table mixes incompatible query protocols")
