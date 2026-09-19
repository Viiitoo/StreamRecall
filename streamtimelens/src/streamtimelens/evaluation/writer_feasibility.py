"""Metrics for a fixed, model-paired closed-chunk writer feasibility run."""

from __future__ import annotations

import math
from collections import defaultdict
from statistics import mean
from typing import Any, Iterable

from streamtimelens.writer.parse import parse_writer_output


def _iou(first: tuple[float, float], second: tuple[float, float]) -> float:
    intersection = max(0.0, min(first[1], second[1]) - max(first[0], second[0]))
    union = max(first[1], second[1]) - min(first[0], second[0])
    return 0.0 if union <= 0 else intersection / union


def semantic_reservoir_oracle_coverage(records: Iterable[dict[str, Any]]) -> float:
    """Upper-bound GT coverage using only pairs of fixed sampled timestamps.

    This is query-aware only at evaluation time: it chooses the best possible
    temporal span whose two endpoints were retained by the query-blind visual
    sample set.  It never supplies GT or query text to either writer.
    """
    chunks: dict[str, tuple[tuple[float, ...], tuple[tuple[float, float], ...]]] = {}
    for row in records:
        chunk_id = str(row.get("chunk_id", "")).strip()
        sampled = tuple(sorted(set(map(float, row.get("sampled_timestamps", ())))))
        gt_spans = tuple(tuple(map(float, span)) for span in row.get("gt_spans", ()))
        if not chunk_id or len(sampled) < 2 or any(len(span) != 2 for span in gt_spans):
            raise ValueError("semantic reservoir oracle needs valid chunk samples and GT spans")
        value = (sampled, gt_spans)
        previous = chunks.setdefault(chunk_id, value)
        if previous != value:
            raise ValueError(f"paired records disagree on fixed chunk {chunk_id}")
    coverages = []
    for sampled, gt_spans in chunks.values():
        candidates = [
            (sampled[left], sampled[right])
            for left in range(len(sampled)) for right in range(left + 1, len(sampled))
        ]
        coverages.extend(max(_iou(gt, candidate) for candidate in candidates) for gt in gt_spans)
    if not coverages:
        raise ValueError("semantic reservoir oracle needs at least one GT span")
    return mean(coverages)


def evaluate_writer_records(
    records: Iterable[dict[str, Any]], *, expected_chunks: int | None = 100
) -> dict[str, Any]:
    """Aggregate already-generated outputs without invoking or repairing a model."""
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    parsed_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        model = str(row.get("model", "")).strip()
        chunk_id = str(row.get("chunk_id", "")).strip()
        segment = tuple(map(float, row.get("segment", ())))
        sampled = tuple(map(float, row.get("sampled_timestamps", ())))
        if not model or not chunk_id or len(segment) != 2 or not sampled:
            raise ValueError("each feasibility row needs model/chunk/segment/sampled timestamps")
        if not math.isfinite(float(row.get("gpu_s", 0))) or float(row.get("gpu_s", 0)) < 0:
            raise ValueError("gpu_s must be finite and non-negative")
        if any(existing["chunk_id"] == chunk_id for existing in grouped[model]):
            raise ValueError(f"duplicate model/chunk feasibility row: {model}/{chunk_id}")
        grouped[model].append({**row, "chunk_id": chunk_id})
        parsed = parse_writer_output(
            str(row.get("raw_output", "")), segment=segment,
            sampled_timestamps=sampled,
        )
        gt_spans = [tuple(map(float, value)) for value in row.get("gt_spans", [])]
        event_spans = ([] if parsed.status == "fallback" else
                       [event.span for event in parsed.document.events])
        coverages = [max((_iou(gt, event) for event in event_spans), default=0.0) for gt in gt_spans]
        endpoint_errors = []
        for gt in gt_spans:
            if event_spans:
                best = max(event_spans, key=lambda value: _iou(gt, value))
                endpoint_errors.extend((abs(gt[0] - best[0]), abs(gt[1] - best[1])))
        parsed_rows[model].append({
            "chunk_id": chunk_id, "parse_status": parsed.status,
            "event_count": len(event_spans), "coverages": coverages,
            "endpoint_errors": endpoint_errors, "gpu_s": float(row.get("gpu_s", 0)),
        })
    if len(grouped) < 2:
        raise ValueError("writer feasibility requires at least two paired checkpoints")
    chunk_sets = {model: {row["chunk_id"] for row in rows} for model, rows in grouped.items()}
    reference = next(iter(chunk_sets.values()))
    if any(chunks != reference for chunks in chunk_sets.values()):
        raise ValueError("writer checkpoints must be evaluated on identical fixed chunks")
    if expected_chunks is not None and len(reference) != expected_chunks:
        raise ValueError(f"expected {expected_chunks} fixed chunks, found {len(reference)}")

    metrics = {}
    for model, rows in sorted(parsed_rows.items()):
        coverage = [value for row in rows for value in row["coverages"]]
        endpoint = [value for row in rows for value in row["endpoint_errors"]]
        metrics[model] = {
            "chunks": len(rows),
            "json_valid_rate": mean(row["parse_status"] == "valid" for row in rows),
            "schema_usable_rate": mean(row["parse_status"] != "fallback" for row in rows),
            "fallback_rate": mean(row["parse_status"] == "fallback" for row in rows),
            "mean_events_per_chunk": mean(row["event_count"] for row in rows),
            "mean_gt_coverage_iou": mean(coverage) if coverage else None,
            "mean_endpoint_abs_error_s": mean(endpoint) if endpoint else None,
            "gpu_s": sum(row["gpu_s"] for row in rows),
        }
    return {"fixed_chunk_ids": sorted(reference), "models": metrics}


def feasibility_decision(
    metrics: dict[str, Any], *, semantic_oracle_coverage: float | None = None
) -> dict[str, Any]:
    models = metrics["models"]
    ranked = sorted(
        models,
        key=lambda name: (
            -(models[name]["mean_gt_coverage_iou"] or 0.0),
            -models[name]["schema_usable_rate"], models[name]["gpu_s"], name,
        ),
    )
    selected = ranked[0]
    stop = False
    if semantic_oracle_coverage is not None:
        if not 0 <= semantic_oracle_coverage <= 1:
            raise ValueError("semantic oracle coverage must be in [0,1]")
        stop = all((item["mean_gt_coverage_iou"] or 0.0) + 0.05 < semantic_oracle_coverage
                   for item in models.values())
    return {
        "selected_writer": selected,
        "decision": "stop_writer" if stop else "go",
        "semantic_oracle_coverage": semantic_oracle_coverage,
    }
