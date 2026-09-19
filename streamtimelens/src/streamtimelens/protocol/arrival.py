"""Frozen delayed-query plans derived offline from annotations."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


LAG_BINS = ((0.0, 0.1), (0.1, 0.25), (0.25, 0.5), (0.5, 1.0000001))


@dataclass(frozen=True)
class ArrivalRecord:
    video_id: str
    query_id: str
    query: str
    gt_span: tuple[float, float]
    t_q: float
    rho_q: float
    cohort: str
    lag_bin: str
    eligible: bool
    arrival_kind: str = "rho"
    delay_s: float | None = None

    @property
    def prediction_key(self) -> str:
        start, end = self.gt_span
        suffix = f"rho={self.rho_q:.2f}" if self.arrival_kind == "rho" else f"delta={self.delay_s:g}s"
        return f"{self.video_id} >>> {self.query_id} >>> {suffix} >>> [{start:.3f},{end:.3f}]"


def _lag_bin(end_s: float, t_q: float, duration_s: float) -> str:
    lag = max(0.0, min(1.0, (t_q - end_s) / max(duration_s, 1e-9)))
    for low, high in LAG_BINS:
        if low <= lag < high:
            return f"[{low:g},{high:g})"
    raise AssertionError("unreachable")


def build_arrival_plan(
    examples: Iterable[dict], ratios: Iterable[float] = (0.25, 0.5, 0.75, 1.0),
    delays_s: Iterable[float] = (),
) -> list[ArrivalRecord]:
    """Create natural and fixed-0.25 cohorts without exposing them to ingest."""
    frozen_ratios = tuple(float(ratio) for ratio in ratios)
    frozen_delays = tuple(float(delay) for delay in delays_s)
    if not frozen_ratios or any(not math.isfinite(ratio) or not 0 < ratio <= 1 for ratio in frozen_ratios):
        raise ValueError("arrival ratios must be in (0, 1]")
    if len(set(frozen_ratios)) != len(frozen_ratios) or any(
        not math.isfinite(delay) or delay <= 0 for delay in frozen_delays
    ):
        raise ValueError("arrival ratios and delays must be unique positive values")
    output: list[ArrivalRecord] = []
    for item in examples:
        duration = float(item["duration"])
        start, end = map(float, item["gt_span"])
        if not all(math.isfinite(value) for value in (duration, start, end)) or not 0 <= start < end <= duration:
            raise ValueError(f"invalid ground-truth span for {item['video_id']}")
        for ratio in frozen_ratios:
            t_q = ratio * duration
            eligible = end <= t_q
            common = dict(
                video_id=str(item["video_id"]), query_id=str(item["query_id"]),
                query=str(item["query"]), gt_span=(start, end), t_q=t_q, rho_q=ratio,
                lag_bin=_lag_bin(end, t_q, duration), eligible=eligible,
            )
            output.append(ArrivalRecord(**common, cohort="natural"))
            if ratio > 0.25:
                fixed_common = {**common, "eligible": end <= .25 * duration}
                output.append(ArrivalRecord(**fixed_common, cohort="fixed_0.25"))
        for delay in frozen_delays:
            t_q = min(duration, end + delay)
            output.append(ArrivalRecord(
                video_id=str(item["video_id"]), query_id=str(item["query_id"]), query=str(item["query"]),
                gt_span=(start, end), t_q=t_q, rho_q=t_q / duration, cohort="delta",
                lag_bin=_lag_bin(end, t_q, duration), eligible=True, arrival_kind="delta", delay_s=delay,
            ))
    return sorted(output, key=lambda row: (row.video_id, row.query_id, row.t_q, row.arrival_kind, row.cohort, row.delay_s or 0.0))


def write_arrival_plan(records: Iterable[ArrivalRecord], path: Path, *, overwrite: bool = False) -> str:
    """Write canonical JSONL and a content SHA-256 sidecar for immutability."""
    path.parent.mkdir(parents=True, exist_ok=True)
    sidecar = path.with_suffix(path.suffix + ".sha256")
    if (path.exists() or sidecar.exists()) and not overwrite:
        raise FileExistsError(f"arrival plan is frozen; use explicit overwrite: {path}")
    ordered = sorted(records, key=lambda row: (row.video_id, row.query_id, row.t_q, row.arrival_kind, row.cohort, row.delay_s or 0.0))
    text = "".join(json.dumps(asdict(row), sort_keys=True) + "\n" for row in ordered)
    path.write_text(text, encoding="utf-8")
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    sidecar.write_text(digest + "\n", encoding="ascii")
    return digest


def verify_arrival_plan(path: Path) -> list[ArrivalRecord]:
    """Validate a frozen plan before using it to derive queries or metrics."""
    sidecar = path.with_suffix(path.suffix + ".sha256")
    if not path.is_file() or not sidecar.is_file():
        raise FileNotFoundError("arrival plan and .sha256 sidecar are both required")
    text = path.read_text(encoding="utf-8")
    actual = hashlib.sha256(text.encode("utf-8")).hexdigest()
    expected = sidecar.read_text(encoding="ascii").strip()
    if actual != expected:
        raise ValueError("arrival plan SHA-256 does not match sidecar")
    records: list[ArrivalRecord] = []
    for line in text.splitlines():
        data = json.loads(line)
        data["gt_span"] = tuple(data["gt_span"])
        records.append(ArrivalRecord(**data))
    if records != sorted(records, key=lambda row: (row.video_id, row.query_id, row.t_q, row.arrival_kind, row.cohort, row.delay_s or 0.0)):
        raise ValueError("arrival plan is not in canonical order")
    return records
