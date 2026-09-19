"""Official-compatible video temporal grounding metrics."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Iterable


@dataclass(frozen=True)
class VTGMetricExample:
    query_id: str
    video_id: str
    gt_span: tuple[float, float]
    predicted_span: tuple[float, float] | None
    status: str

    def __post_init__(self) -> None:
        if not self.query_id or not self.video_id:
            raise ValueError("metric example IDs are required")
        if not all(math.isfinite(value) for value in self.gt_span) or (
            self.gt_span[0] < 0 or self.gt_span[0] >= self.gt_span[1]
        ):
            raise ValueError("metric ground-truth span is invalid")


def temporal_iou(
    ground_truth: tuple[float, float], prediction: tuple[float, float] | None,
) -> float:
    """Match ``timelens.utils.iou`` for valid spans; invalid spans score zero."""
    if prediction is None or not all(math.isfinite(value) for value in prediction):
        return 0.0
    if prediction[0] < 0 or prediction[0] >= prediction[1]:
        return 0.0
    intersection = max(min(ground_truth[1], prediction[1]) - max(ground_truth[0], prediction[0]), 0.0)
    union = max(ground_truth[1], prediction[1]) - min(ground_truth[0], prediction[0])
    return intersection / union if union > 0 else 0.0


@dataclass(frozen=True)
class VTGMetrics:
    count: int
    video_count: int
    miou: float
    recall_at_03: float
    recall_at_05: float
    recall_at_07: float
    valid_span_count: int
    invalid_or_not_found_count: int
    start_abs_error_s: float | None
    end_abs_error_s: float | None
    start_signed_error_s: float | None
    end_signed_error_s: float | None

    def to_dict(self) -> dict:
        return asdict(self)


def evaluate_vtg(examples: Iterable[VTGMetricExample]) -> VTGMetrics:
    rows = list(examples)
    if not rows:
        return VTGMetrics(0, 0, 0.0, 0.0, 0.0, 0.0, 0, 0, None, None, None, None)
    identifiers = [row.query_id for row in rows]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("metric examples contain duplicate query IDs")
    ious = []
    errors = []
    for row in rows:
        valid = row.status.lower() not in ("not_found", "error", "invalid")
        prediction = row.predicted_span if valid else None
        ious.append(temporal_iou(row.gt_span, prediction))
        if prediction is not None and all(math.isfinite(value) for value in prediction) and (
            0 <= prediction[0] < prediction[1]
        ):
            errors.append((prediction[0] - row.gt_span[0], prediction[1] - row.gt_span[1]))
    scale = 100.0 / len(rows)
    if errors:
        start_signed = sum(value[0] for value in errors) / len(errors)
        end_signed = sum(value[1] for value in errors) / len(errors)
        start_abs = sum(abs(value[0]) for value in errors) / len(errors)
        end_abs = sum(abs(value[1]) for value in errors) / len(errors)
    else:
        start_signed = end_signed = start_abs = end_abs = None
    return VTGMetrics(
        len(rows), len({row.video_id for row in rows}), sum(ious) * scale,
        sum(value >= 0.3 for value in ious) * scale,
        sum(value >= 0.5 for value in ious) * scale,
        sum(value >= 0.7 for value in ious) * scale,
        len(errors), len(rows) - len(errors),
        start_abs, end_abs, start_signed, end_signed,
    )
