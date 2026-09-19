"""Deterministic Oracle-refiner experiment construction and metrics."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import Callable, Iterable, Literal, Mapping, Sequence


CacheStrategy = Literal["uniform", "boundary-heavy"]
OracleMode = Literal["dense_crop", "sparse_adapter", "offline_timelens"]
ORACLE_MODES: tuple[OracleMode, ...] = ("dense_crop", "sparse_adapter", "offline_timelens")


@dataclass(frozen=True)
class OracleExample:
    example_id: str
    query_id: str
    video_id: str
    query: str
    gt_span: tuple[float, float]
    video_duration_s: float
    original_fps: float
    total_num_frames: int
    crop_span: tuple[float, float]
    margin_s: float
    k_frames: int
    strategy: CacheStrategy
    frame_indices: tuple[int, ...]
    timestamps_s: tuple[float, ...]
    sampling_interval_s: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class OraclePrediction:
    example_id: str
    mode: OracleMode
    predicted_span: tuple[float, float] | None
    status: str
    raw_answer: str = ""


def _choose_positions(count: int, k: int, strategy: CacheStrategy) -> list[int]:
    if count < k:
        raise ValueError(f"crop has only {count} frames, fewer than requested {k} unique frames")
    if k == 1:
        return [count // 2]
    if strategy == "uniform":
        targets = [round(i * (count - 1) / (k - 1)) for i in range(k)]
    elif strategy == "boundary-heavy":
        left = (k + 1) // 2
        right = k - left
        left_targets = [round(i * 0.25 * (count - 1) / max(1, left - 1)) for i in range(left)]
        right_targets = [round((0.75 + 0.25 * i / max(1, right - 1)) * (count - 1)) for i in range(right)]
        targets = left_targets + right_targets
    else:
        raise ValueError(f"unknown cache strategy: {strategy}")
    selected = list(dict.fromkeys(targets))
    if len(selected) < k:
        chosen = set(selected)
        selected.extend(index for index in range(count) if index not in chosen and len(selected) < k)
    return sorted(selected[:k])


def build_oracle_examples(
    records: Iterable[Mapping[str, object]],
    *,
    margins_s: Sequence[float] = (2.0, 5.0, 10.0),
    frame_counts: Sequence[int] = (16, 32),
    strategies: Sequence[CacheStrategy] = ("uniform", "boundary-heavy"),
) -> list[OracleExample]:
    """Expand dev annotations into frozen GT-window sparse-cache conditions."""
    examples: list[OracleExample] = []
    for record in records:
        query_id, video_id, query = str(record["query_id"]), str(record["video_id"]), str(record["query"])
        start, end = map(float, record["gt_span"])  # type: ignore[arg-type]
        duration, fps = float(record["duration_s"]), float(record["fps"])
        total_frames = int(record.get("total_num_frames", math.floor(duration * fps) + 1))
        if not (0 <= start < end <= duration) or fps <= 0 or total_frames <= 0:
            raise ValueError(f"invalid Oracle annotation: {query_id}")
        for margin in margins_s:
            if margin < 0:
                raise ValueError("Oracle margins must be non-negative")
            crop = (max(0.0, start - float(margin)), min(duration, end + float(margin)))
            first = max(0, int(math.ceil(crop[0] * fps - 1e-9)))
            last = min(total_frames - 1, int(math.floor(crop[1] * fps + 1e-9)))
            candidates = list(range(first, last + 1))
            for k_frames in frame_counts:
                for strategy in strategies:
                    positions = _choose_positions(len(candidates), int(k_frames), strategy)
                    indices = tuple(candidates[position] for position in positions)
                    timestamps = tuple(index / fps for index in indices)
                    payload = f"{query_id}|{margin:g}|{k_frames}|{strategy}"
                    examples.append(OracleExample(
                        example_id=hashlib.sha1(payload.encode("utf-8")).hexdigest(),
                        query_id=query_id, video_id=video_id, query=query, gt_span=(start, end),
                        video_duration_s=duration, original_fps=fps, total_num_frames=total_frames,
                        crop_span=crop, margin_s=float(margin),
                        k_frames=int(k_frames), strategy=strategy, frame_indices=indices,
                        timestamps_s=timestamps,
                        sampling_interval_s=max((crop[1] - crop[0]) / max(1, int(k_frames) - 1), 1.0 / fps),
                    ))
    return sorted(examples, key=lambda item: (item.video_id, item.query_id, item.margin_s, item.k_frames, item.strategy))


def temporal_iou(prediction: tuple[float, float], target: tuple[float, float]) -> float:
    intersection = max(0.0, min(prediction[1], target[1]) - max(prediction[0], target[0]))
    union = max(prediction[1], target[1]) - min(prediction[0], target[0])
    return intersection / union if union > 0 else 0.0


def evaluate_oracle_predictions(
    examples: Sequence[OracleExample], predictions: Sequence[OraclePrediction],
) -> dict[str, object]:
    by_example = {example.example_id: example for example in examples}
    if len(by_example) != len(examples):
        raise ValueError("Oracle examples contain duplicate example IDs")
    expected = {(example.example_id, mode) for example in examples for mode in ORACLE_MODES}
    observed: set[tuple[str, str]] = set()
    groups: dict[str, list[tuple[OracleExample, OraclePrediction]]] = {}
    for prediction in predictions:
        if prediction.example_id not in by_example:
            raise ValueError(f"prediction references unknown example {prediction.example_id}")
        if prediction.mode not in ORACLE_MODES:
            raise ValueError(f"prediction uses unknown Oracle mode {prediction.mode}")
        key = (prediction.example_id, prediction.mode)
        if key in observed:
            raise ValueError(f"duplicate Oracle prediction for {prediction.example_id}/{prediction.mode}")
        observed.add(key)
        groups.setdefault(prediction.mode, []).append((by_example[prediction.example_id], prediction))
    missing = expected - observed
    if missing:
        preview = ", ".join(f"{example_id}/{mode}" for example_id, mode in sorted(missing)[:3])
        raise ValueError(f"Oracle prediction matrix is incomplete; missing {len(missing)} entries: {preview}")
    metrics: dict[str, dict[str, float | int]] = {}
    for mode, pairs in groups.items():
        valid = [(example, prediction) for example, prediction in pairs if prediction.predicted_span is not None]
        count = len(pairs)
        if not valid:
            metrics[mode] = {"count": count, "valid_count": 0, "miou": 0.0, "start_mae_s": float("nan"),
                             "end_mae_s": float("nan"), "signed_start_bias_s": float("nan"), "signed_end_bias_s": float("nan")}
            continue
        start_errors = [prediction.predicted_span[0] - example.gt_span[0] for example, prediction in valid]  # type: ignore[index]
        end_errors = [prediction.predicted_span[1] - example.gt_span[1] for example, prediction in valid]  # type: ignore[index]
        metrics[mode] = {
            "count": count, "valid_count": len(valid),
            "miou": sum(temporal_iou(prediction.predicted_span, example.gt_span) for example, prediction in valid) / count,  # type: ignore[arg-type]
            "start_mae_s": sum(abs(value) for value in start_errors) / len(valid),
            "end_mae_s": sum(abs(value) for value in end_errors) / len(valid),
            "signed_start_bias_s": sum(start_errors) / len(valid),
            "signed_end_bias_s": sum(end_errors) / len(valid),
        }
    dense = metrics.get("dense_crop")
    sparse = metrics.get("sparse_adapter")
    # The aggregate mixes several frame budgets and crop widths.  Using the
    # largest interval would let a coarse condition hide bias that exceeds a
    # denser condition's one-sample tolerance, so the global gate is strict.
    interval = min((example.sampling_interval_s for example in examples), default=0.0)
    gate = None
    if dense is not None and sparse is not None:
        gate = {
            "miou_drop": float(dense["miou"]) - float(sparse["miou"]),
            "allowed_miou_drop": 0.05,
            "bias_tolerance_s": interval,
        }
        gate["passed"] = bool(
            gate["miou_drop"] <= 0.05
            and abs(float(sparse["signed_start_bias_s"])) <= interval
            and abs(float(sparse["signed_end_bias_s"])) <= interval
        )
    return {"schema_version": 1, "metrics": metrics, "p0_gate": gate}


def evaluate_oracle_conditions(
    examples: Sequence[OracleExample], predictions: Sequence[OraclePrediction],
) -> dict[str, object]:
    """Break the complete Oracle matrix down by every frozen sampling condition."""
    by_id = {example.example_id: example for example in examples}
    groups: dict[tuple[float, int, str], list[OracleExample]] = {}
    for example in examples:
        groups.setdefault(
            (example.margin_s, example.k_frames, example.strategy), [],
        ).append(example)
    predictions_by_id: dict[str, list[OraclePrediction]] = {}
    for prediction in predictions:
        if prediction.example_id not in by_id:
            raise ValueError(f"prediction references unknown example {prediction.example_id}")
        predictions_by_id.setdefault(prediction.example_id, []).append(prediction)
    result = {}
    for (margin, frames, strategy), group in sorted(groups.items()):
        key = f"margin_{margin:g}s__frames_{frames}__{strategy}"
        group_predictions = [
            prediction for example in group
            for prediction in predictions_by_id.get(example.example_id, ())
        ]
        result[key] = evaluate_oracle_predictions(group, group_predictions)
    return result


class OracleRunner:
    """Run all three conditions through injected real inference functions."""

    def __init__(self, dense_crop: Callable[[OracleExample], OraclePrediction],
                 sparse_adapter: Callable[[OracleExample], OraclePrediction],
                 offline_timelens: Callable[[OracleExample], OraclePrediction]) -> None:
        self._runners = {"dense_crop": dense_crop, "sparse_adapter": sparse_adapter, "offline_timelens": offline_timelens}

    def run(self, examples: Sequence[OracleExample]) -> tuple[list[OraclePrediction], dict[str, object]]:
        predictions = [runner(example) for example in examples for runner in self._runners.values()]
        return predictions, evaluate_oracle_predictions(examples, predictions)


def oracle_jsonl(examples: Sequence[OracleExample]) -> str:
    return "".join(json.dumps(example.to_dict(), sort_keys=True, ensure_ascii=False) + "\n" for example in examples)


def oracle_gate_decision(metrics: Mapping[str, object]) -> dict[str, str]:
    gate = metrics.get("p0_gate")
    groups = metrics.get("metrics")
    if not isinstance(gate, Mapping) or not isinstance(groups, Mapping):
        raise ValueError("Oracle metrics do not contain a P0 gate")
    sparse = groups.get("sparse_adapter")
    dense = groups.get("dense_crop")
    if not isinstance(sparse, Mapping) or not isinstance(dense, Mapping):
        raise ValueError("Oracle metrics are missing dense/sparse conditions")
    if bool(gate.get("passed")):
        return {"decision": "go", "reason": "sparse adapter satisfies mIoU and timestamp-bias gates"}
    if int(sparse.get("valid_count", 0)) == 0 or int(dense.get("valid_count", 0)) == 0:
        return {"decision": "stop", "reason": "dense or sparse condition produced no valid spans"}
    return {"decision": "fix", "reason": "sparse adapter requires correction before large writer runs"}
