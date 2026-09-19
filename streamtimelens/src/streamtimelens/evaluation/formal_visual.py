"""Frozen visual-route preparation and formal aggregation helpers."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from streamtimelens.evaluation.metrics import VTGMetricExample, evaluate_vtg, temporal_iou
from streamtimelens.protocol.arrival import ArrivalRecord, build_arrival_plan


@dataclass(frozen=True)
class FormalVisualObservation:
    config_id: str
    dataset: str
    budget_bytes: int
    video_id: str
    query_id: str
    rho_q: float
    cohort: str
    lag_bin: str
    duration_s: float
    gt_span: tuple[float, float]
    predicted_span: tuple[float, float] | None
    status: str
    retrieved_timestamps: tuple[float, ...]
    candidate_spans: tuple[tuple[float, float], ...]
    snapshot_bytes: int
    ingest_gpu_s: float
    ingest_wall_s: float
    query_gpu_s: float
    query_latency_s: float
    earliest_anchor_retained: bool = False
    retained_frame_count: int = 0
    queries_per_snapshot: int = 1


def timelens_bench_records(
    dataset: str, annotations: dict[str, Any], video_root: Path,
) -> tuple[list[dict], list[dict], list[dict], list[ArrivalRecord]]:
    """Convert the official dictionary format into GT-free manifests and a frozen plan."""
    if not dataset or not annotations or not video_root.is_dir():
        raise ValueError("formal dataset name, annotations and video root are required")
    videos = []
    queries = []
    ground_truth = []
    for video_id, row in sorted(annotations.items()):
        duration = float(row["duration"])
        spans = row["spans"]
        texts = row["queries"]
        if not math.isfinite(duration) or duration <= 0 or len(spans) != len(texts):
            raise ValueError(f"invalid TimeLens-Bench row: {dataset}/{video_id}")
        video_path = video_root / f"{video_id}.mp4"
        if not video_path.is_file():
            raise FileNotFoundError(f"formal video is missing: {video_path}")
        videos.append({"video_id": str(video_id), "path": str(video_path.resolve())})
        for index, (span, query) in enumerate(zip(spans, texts)):
            query_id = f"{dataset}__{video_id}__{index:04d}"
            start, end = map(float, span)
            # TimeLens-Bench stores several Charades endpoints as integer
            # seconds while its probed video duration is fractional (for
            # example, 35 versus 34.93). Accept only that final-second
            # rounding case and evaluate against the real video boundary.
            if not str(query).strip() or not 0 <= start < end <= math.ceil(duration):
                raise ValueError(f"invalid TimeLens-Bench query: {query_id}")
            end = min(end, duration)
            queries.append({
                "query_id": query_id, "video_id": str(video_id), "query": str(query),
            })
            ground_truth.append({
                "dataset": dataset, "query_id": query_id, "video_id": str(video_id),
                "query": str(query), "gt_span": [start, end],
                "duration": duration, "duration_s": duration,
            })
    arrival = build_arrival_plan(ground_truth)
    return videos, queries, ground_truth, arrival


def frozen_visual_grids(frozen: dict[str, Any]) -> dict[int, list[dict[str, Any]]]:
    """Return each exact frozen query grid, partitioned by its byte budget."""
    if frozen.get("status") != "frozen" or not frozen.get("configs"):
        raise ValueError("visual configuration is not frozen")
    grids: dict[int, list[dict[str, Any]]] = {}
    for config_id, row in sorted(frozen["configs"].items()):
        resolved = row["resolved"]
        visual = row["visual"]
        if resolved["method"]["method"] != "uniform_raw" or visual.get("timelens_enabled"):
            raise ValueError("formal visual route must match the frozen Uniform coarse decision")
        budget = int(resolved["budget"]["memory_bytes"])
        grids.setdefault(budget, []).append({
            "config_id": config_id,
            "top_k": int(visual["top_k"]),
            "expand_neighbors": int(visual["expand_neighbors"]),
            "merge_gap_s": float(visual["merge_gap_s"]),
            "coarse_margin_s": float(visual["coarse_margin_s"]),
        })
    return grids


def _percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def video_bootstrap_ci(
    observations: Iterable[FormalVisualObservation], *, seed: int = 20260830,
    resamples: int = 10_000,
) -> dict[str, Any]:
    """Bootstrap absolute mIoU over videos while retaining within-video query arrivals."""
    rows = list(observations)
    if not rows or resamples <= 0:
        raise ValueError("formal bootstrap needs observations and positive resamples")
    by_video: dict[str, list[float]] = {}
    for row in rows:
        by_video.setdefault(row.video_id, []).append(temporal_iou(row.gt_span, row.predicted_span))
    per_video = {key: sum(values) / len(values) for key, values in by_video.items()}
    videos = sorted(per_video)
    values = np.asarray([per_video[video] for video in videos], dtype=np.float64)
    generator = np.random.default_rng(seed)
    draws = []
    # Batch resamples to keep peak memory bounded on the full formal datasets.
    for offset in range(0, resamples, 256):
        count = min(256, resamples - offset)
        indices = generator.integers(0, len(values), size=(count, len(values)))
        draws.extend(values[indices].mean(axis=1).tolist())
    return {
        "unit": "video", "seed": seed, "resamples": resamples,
        "video_count": len(videos), "sample_count": len(rows),
        "miou": sum(per_video.values()) / len(per_video),
        "ci_low": _percentile(draws, .025), "ci_high": _percentile(draws, .975),
    }


def paired_formal_bootstrap(
    left: Iterable[FormalVisualObservation], right: Iterable[FormalVisualObservation],
    *, seed: int = 20260830, resamples: int = 10_000,
) -> dict[str, Any]:
    """Paired video bootstrap for two frozen configurations on identical arrivals."""
    left_rows = list(left)
    right_rows = list(right)
    left_values = {
        (row.video_id, row.query_id, row.rho_q, row.cohort): temporal_iou(
            row.gt_span, row.predicted_span,
        )
        for row in left_rows
    }
    right_values = {
        (row.video_id, row.query_id, row.rho_q, row.cohort): temporal_iou(
            row.gt_span, row.predicted_span,
        )
        for row in right_rows
    }
    if not left_values or len(left_values) != len(left_rows) or set(left_values) != set(right_values):
        raise ValueError("formal bootstrap configurations are not exactly paired")
    by_video: dict[str, list[float]] = {}
    for key, value in left_values.items():
        by_video.setdefault(key[0], []).append(value - right_values[key])
    values = np.asarray([
        sum(by_video[video]) / len(by_video[video]) for video in sorted(by_video)
    ], dtype=np.float64)
    generator = np.random.default_rng(seed)
    draws = []
    for offset in range(0, resamples, 256):
        count = min(256, resamples - offset)
        indices = generator.integers(0, len(values), size=(count, len(values)))
        draws.extend(values[indices].mean(axis=1).tolist())
    return {
        "unit": "video", "seed": seed, "resamples": resamples,
        "video_count": len(values), "sample_count": len(left_values),
        "delta_miou": float(values.mean()),
        "ci_low": _percentile(draws, .025), "ci_high": _percentile(draws, .975),
    }


def summarize_formal_group(rows: Iterable[FormalVisualObservation]) -> dict[str, Any]:
    selected = list(rows)
    if not selected:
        raise ValueError("formal group is empty")
    metric_rows = [
        VTGMetricExample(
            f"{row.query_id}::{index}", row.video_id, row.gt_span,
            row.predicted_span, row.status,
        )
        for index, row in enumerate(selected)
    ]
    metrics = evaluate_vtg(metric_rows).to_dict()
    frame_recall = {
        str(k): sum(any(
            row.gt_span[0] <= value <= row.gt_span[1]
            for value in row.retrieved_timestamps[:k]
        ) for row in selected) / len(selected)
        for k in (1, 5, 8)
    }
    candidate_recall = {
        str(k): sum(max(
            (temporal_iou(row.gt_span, span) for span in row.candidate_spans[:k]),
            default=0.0,
        ) >= .5 for row in selected) / len(selected)
        for k in (1, 5)
    }
    left = sum(any(a <= row.gt_span[0] <= b for a, b in row.candidate_spans) for row in selected)
    right = sum(any(a <= row.gt_span[1] <= b for a, b in row.candidate_spans) for row in selected)
    unique_videos = {row.video_id: row.duration_s for row in selected}
    ingest_gpu = sum({row.video_id: row.ingest_gpu_s for row in selected}.values())
    ingest_wall = sum({row.video_id: row.ingest_wall_s for row in selected}.values())
    video_seconds = sum(unique_videos.values())
    first = selected[0]
    return {
        "config_id": first.config_id, "dataset": first.dataset,
        "budget_bytes": first.budget_bytes, "rho_q": first.rho_q,
        "cohort": first.cohort, **metrics,
        "frame_recall": frame_recall, "candidate_recall": candidate_recall,
        "left_boundary_hit_rate": left / len(selected),
        "right_boundary_hit_rate": right / len(selected),
        "both_boundary_hit_rate": sum(
            any(a <= row.gt_span[0] <= b for a, b in row.candidate_spans)
            and any(a <= row.gt_span[1] <= b for a, b in row.candidate_spans)
            for row in selected
        ) / len(selected),
        "snapshot_bytes": sum(row.snapshot_bytes for row in selected) / len(selected),
        "ingest_gpu_s": ingest_gpu, "query_gpu_s": sum(row.query_gpu_s for row in selected),
        "query_latency_s": sum(row.query_latency_s for row in selected) / len(selected),
        "realtime_throughput": video_seconds / ingest_wall if ingest_wall > 0 else 0.0,
        "bootstrap_miou": video_bootstrap_ci(selected),
    }


def formal_observation_dict(row: FormalVisualObservation) -> dict[str, Any]:
    return asdict(row)


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def canonical_formal_rho(measured: float, planned: float) -> float:
    """Return the frozen arrival ratio after checking measured-ratio agreement."""
    measured_value = float(measured)
    planned_value = float(planned)
    if round(measured_value, 8) != round(planned_value, 8):
        raise ValueError(
            f"measured snapshot rho {measured_value} does not match plan {planned_value}"
        )
    return planned_value


def select_activitynet_configs(
    frozen: dict[str, Any], formal_summaries: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    """Choose at most one frozen V5 winner per budget for the V6 extension."""
    frozen_ids = set(frozen.get("configs", {}))
    natural = [
        row for row in formal_summaries
        if row.get("cohort") == "natural" and row.get("config_id") in frozen_ids
    ]
    if not natural:
        raise ValueError("V6 selection needs completed frozen V5 natural summaries")
    grouped: dict[tuple[int, str], list[float]] = {}
    for row in natural:
        key = (int(row["budget_bytes"]), str(row["config_id"]))
        grouped.setdefault(key, []).append(float(row["miou"]))
    by_budget: dict[int, list[tuple[float, str]]] = {}
    for (budget, config_id), values in grouped.items():
        by_budget.setdefault(budget, []).append((sum(values) / len(values), config_id))
    selected = []
    for budget, rows in sorted(by_budget.items()):
        score, config_id = sorted(rows, key=lambda item: (-item[0], item[1]))[0]
        selected.append({
            "config_id": config_id, "budget_bytes": budget,
            "v5_natural_miou": score,
            "frozen_visual_sha256": frozen["configs"][config_id]["visual_sha256"],
        })
    return {
        "selected": selected[:2], "selection_source": "completed_v5_frozen_pareto_only",
        "selection_rule": "highest_mean_natural_miou_per_budget_then_config_id",
    }


def summarize_long_video_observations(rows: Iterable[FormalVisualObservation]) -> dict[str, Any]:
    """Aggregate V6 length, fixed-C0.25 aging and multi-query amortization diagnostics."""
    observations = list(rows)
    if not observations:
        raise ValueError("V6 summary needs ActivityNet observations")
    length_bins = {
        "short:<60s": lambda value: value < 60,
        "medium:[60,300)s": lambda value: 60 <= value < 300,
        "long:>=300s": lambda value: value >= 300,
    }

    def aggregate(selected: list[FormalVisualObservation]) -> dict[str, Any]:
        if not selected:
            return {
                "count": 0, "video_count": 0, "miou": 0.0,
                "candidate_recall_at_5": 0.0, "earliest_anchor_retention_rate": 0.0,
                "amortized_gpu_s_per_query": 0.0, "snapshot_bytes": 0.0,
            }
        ious = [temporal_iou(row.gt_span, row.predicted_span) for row in selected]
        candidate = [float(max(
            (temporal_iou(row.gt_span, span) for span in row.candidate_spans[:5]),
            default=0.0,
        ) >= .5) for row in selected]
        return {
            "count": len(selected), "video_count": len({row.video_id for row in selected}),
            "miou": sum(ious) / len(ious),
            "candidate_recall_at_5": sum(candidate) / len(candidate),
            "earliest_anchor_retention_rate": (
                sum(row.earliest_anchor_retained for row in selected) / len(selected)
            ),
            "amortized_gpu_s_per_query": sum(
                row.query_gpu_s + row.ingest_gpu_s / max(1, row.queries_per_snapshot)
                for row in selected
            ) / len(selected),
            "snapshot_bytes": sum(row.snapshot_bytes for row in selected) / len(selected),
        }

    configs = {}
    for config_id in sorted({row.config_id for row in observations}):
        selected = [row for row in observations if row.config_id == config_id]
        configs[config_id] = {
            "length_bins": {
                label: aggregate([row for row in selected if predicate(row.duration_s)])
                for label, predicate in length_bins.items()
            },
            "fixed_0.25_by_rho": {
                f"{rho:.2f}": aggregate([
                    row for row in selected if row.cohort == "fixed_0.25" and row.rho_q == rho
                ])
                for rho in (.5, .75, 1.0)
            },
            "natural": aggregate([row for row in selected if row.cohort == "natural"]),
        }
    return {"configs": configs}
