"""Query-independent temporal cell tokens with learned event extents.

This is the minimal X1 candidate generator frozen at revision 8f8c3f7.  JQ-01
uses only ``predict_extent_candidates`` and never uses X1's safety selector.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from streamtimelens.observer.clip_encoder import deserialize_embedding, l2_normalize
from streamtimelens.retrieval.frame_candidates import FrameCandidate


DEFAULT_OFFSETS = (-8, -4, -2, -1, 0, 1, 2, 4, 8)
_FORBIDDEN = {
    "query", "queries", "query_id", "query_text", "gt", "gt_span",
    "ground_truth", "ground_truth_span", "video_path", "video_root", "trace",
}


@dataclass(frozen=True)
class TemporalCellToken:
    cell_index: int
    cell_id: str
    start_s: float
    end_s: float
    center_s: float
    support_count: int
    frame_refs: tuple[str, ...]
    pooled_embedding: tuple[float, ...]
    change_left: float
    change_right: float


@dataclass(frozen=True)
class ScoredTemporalToken:
    token: TemporalCellToken
    score: float
    rank: int


@dataclass(frozen=True)
class RidgeHead:
    feature_mean: tuple[float, ...]
    feature_scale: tuple[float, ...]
    coefficients: tuple[float, ...]
    intercept: float

    def predict(self, features: Sequence[float]) -> float:
        values = np.asarray(features, dtype=np.float64)
        mean = np.asarray(self.feature_mean, dtype=np.float64)
        scale = np.asarray(self.feature_scale, dtype=np.float64)
        if values.shape != mean.shape:
            raise ValueError("extent ridge feature shape changed")
        normalized = (values - mean) / scale
        return float(normalized.dot(np.asarray(self.coefficients)) + self.intercept)


@dataclass(frozen=True)
class ExtentPipeline:
    left_head: RidgeHead
    right_head: RidgeHead
    selector_head: RidgeHead
    cell_width_s: float
    top_k: int
    offsets: tuple[int, ...]


@dataclass(frozen=True)
class ExtentObservation:
    observation_id: str
    group_id: str
    query_embedding: tuple[float, ...]
    tokens: tuple[TemporalCellToken, ...]
    upper_bound_s: float
    gt_span: tuple[float, float] | None = None


def _reject(value: Any) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if str(key).lower() in _FORBIDDEN:
                raise ValueError("temporal token input contains query, GT, video, or trace state")
            _reject(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            _reject(child)


def build_temporal_cell_tokens(
    frame_metadata: Mapping[str, Mapping[str, Any]], *, cell_width_s: float = 2.0,
    upper_bound_s: float,
) -> tuple[TemporalCellToken, ...]:
    """Pool cached embeddings into deterministic query-independent physical cells."""
    if cell_width_s <= 0 or upper_bound_s <= 0:
        raise ValueError("temporal token bounds are invalid")
    _reject(frame_metadata)
    groups: dict[int, list[tuple[float, int, str, np.ndarray]]] = {}
    for frame_ref, row in frame_metadata.items():
        if "clip_embedding" not in row:
            continue
        timestamp = float(row["timestamp_s"])
        if not math.isfinite(timestamp) or not 0 <= timestamp <= upper_bound_s + 1e-9:
            raise ValueError("temporal token frame is outside the snapshot")
        index = int(math.floor(timestamp / cell_width_s))
        groups.setdefault(index, []).append((
            timestamp, int(row["frame_index"]), frame_ref,
            deserialize_embedding(dict(row["clip_embedding"])),
        ))
    if not groups:
        return tuple()
    means = {
        index: l2_normalize(np.mean([item[3] for item in rows], axis=0))
        for index, rows in groups.items()
    }
    ordered_indices = sorted(groups)
    tokens = []
    for position, index in enumerate(ordered_indices):
        rows = sorted(groups[index], key=lambda item: (item[0], item[1], item[2]))
        previous = means[ordered_indices[position - 1]] if position else means[index]
        following = (
            means[ordered_indices[position + 1]]
            if position + 1 < len(ordered_indices) else means[index]
        )
        start_s = index * cell_width_s
        end_s = min(upper_bound_s, (index + 1) * cell_width_s)
        tokens.append(TemporalCellToken(
            cell_index=index, cell_id=f"extent-cell-{index:08d}",
            start_s=start_s, end_s=end_s, center_s=(start_s + end_s) / 2.0,
            support_count=len(rows), frame_refs=tuple(item[2] for item in rows),
            pooled_embedding=tuple(float(value) for value in means[index]),
            change_left=float(np.linalg.norm(means[index] - previous)),
            change_right=float(np.linalg.norm(following - means[index])),
        ))
    return tuple(tokens)


def score_temporal_tokens(
    query_embedding: Sequence[float], tokens: Sequence[TemporalCellToken], *, top_k: int = 8,
) -> tuple[ScoredTemporalToken, ...]:
    if top_k <= 0:
        raise ValueError("temporal token Top-K must be positive")
    query = l2_normalize(query_embedding)
    ranked = sorted(
        ((token, float(np.dot(query, np.asarray(token.pooled_embedding)))) for token in tokens),
        key=lambda item: (-item[1], item[0].center_s, item[0].cell_index),
    )
    return tuple(
        ScoredTemporalToken(token, score, rank)
        for rank, (token, score) in enumerate(ranked[:top_k], 1)
    )


def token_features(
    selected: ScoredTemporalToken, all_tokens: Sequence[TemporalCellToken],
    query_embedding: Sequence[float], *, upper_bound_s: float,
    offsets: Sequence[int] = DEFAULT_OFFSETS,
) -> tuple[float, ...]:
    query = l2_normalize(query_embedding)
    score_by_index = {
        token.cell_index: float(np.dot(query, np.asarray(token.pooled_embedding)))
        for token in all_tokens
    }
    token = selected.token
    base = (
        selected.score, 1.0 / selected.rank, token.center_s / upper_bound_s,
        math.log1p(token.support_count), token.change_left, token.change_right,
    )
    neighborhood = tuple(score_by_index.get(token.cell_index + int(offset), -1.0) for offset in offsets)
    return base + neighborhood


def fit_ridge_head(
    features: Sequence[Sequence[float]], targets: Sequence[float], *, alpha: float,
) -> RidgeHead:
    matrix = np.asarray(features, dtype=np.float64)
    values = np.asarray(targets, dtype=np.float64)
    if (
        matrix.ndim != 2 or values.ndim != 1 or matrix.shape[0] != values.shape[0]
        or matrix.shape[0] < 2 or matrix.shape[1] == 0 or alpha <= 0
        or not np.isfinite(matrix).all() or not np.isfinite(values).all()
    ):
        raise ValueError("extent ridge training data are invalid")
    mean = matrix.mean(axis=0)
    scale = matrix.std(axis=0)
    scale[scale < 1e-8] = 1.0
    normalized = (matrix - mean) / scale
    design = np.column_stack((np.ones(matrix.shape[0]), normalized))
    penalty = np.eye(design.shape[1]) * alpha
    penalty[0, 0] = 0.0
    coefficients = np.linalg.solve(design.T @ design + penalty, design.T @ values)
    return RidgeHead(
        tuple(map(float, mean)), tuple(map(float, scale)),
        tuple(map(float, coefficients[1:])), float(coefficients[0]),
    )


def _iou(left: Sequence[float], right: Sequence[float]) -> float:
    intersection = max(0.0, min(left[1], right[1]) - max(left[0], right[0]))
    union = max(left[1], right[1]) - min(left[0], right[0])
    return intersection / union if union > 0 else 0.0


def temporal_iou(left: Sequence[float], right: Sequence[float]) -> float:
    """Public temporal IoU helper used by JQ-01 evaluation only."""
    return _iou(left, right)


def _extent_candidates(
    observation: ExtentObservation, left_head: RidgeHead, right_head: RidgeHead, *,
    top_k: int, cell_width_s: float, offsets: Sequence[int],
) -> list[tuple[ScoredTemporalToken, tuple[float, ...], float, float]]:
    selected = score_temporal_tokens(observation.query_embedding, observation.tokens, top_k=top_k)
    rows = []
    for item in selected:
        features = token_features(
            item, observation.tokens, observation.query_embedding,
            upper_bound_s=observation.upper_bound_s, offsets=offsets,
        )
        left = max(cell_width_s / 2.0, math.expm1(max(0.0, left_head.predict(features))))
        right = max(cell_width_s / 2.0, math.expm1(max(0.0, right_head.predict(features))))
        start = max(0.0, item.token.center_s - left)
        end = min(observation.upper_bound_s, item.token.center_s + right)
        if end <= start:
            start, end = item.token.start_s, item.token.end_s
        rows.append((item, features, start, end))
    return rows


def fit_extent_pipeline(
    observations: Sequence[ExtentObservation], *, cell_width_s: float = 2.0,
    top_k: int = 8, offsets: Sequence[int] = DEFAULT_OFFSETS,
    extent_alpha: float = 1.0, selector_alpha: float = 1.0,
) -> ExtentPipeline:
    left_features, left_targets, right_targets = [], [], []
    for observation in observations:
        if observation.gt_span is None:
            raise ValueError("extent training requires GT only on training observations")
        gt = observation.gt_span
        for selected in score_temporal_tokens(
            observation.query_embedding, observation.tokens, top_k=top_k,
        ):
            token = selected.token
            if token.start_s < gt[1] and token.end_s > gt[0]:
                features = token_features(
                    selected, observation.tokens, observation.query_embedding,
                    upper_bound_s=observation.upper_bound_s, offsets=offsets,
                )
                left_features.append(features)
                left_targets.append(math.log1p(max(0.0, token.center_s - gt[0])))
                right_targets.append(math.log1p(max(0.0, gt[1] - token.center_s)))
    left_head = fit_ridge_head(left_features, left_targets, alpha=extent_alpha)
    right_head = fit_ridge_head(left_features, right_targets, alpha=extent_alpha)
    selector_features, selector_targets = [], []
    for observation in observations:
        assert observation.gt_span is not None
        for selected, features, start, end in _extent_candidates(
            observation, left_head, right_head, top_k=top_k,
            cell_width_s=cell_width_s, offsets=offsets,
        ):
            selector_features.append(features + (
                math.log1p(end - start), start / observation.upper_bound_s,
                end / observation.upper_bound_s,
            ))
            selector_targets.append(_iou((start, end), observation.gt_span))
    selector_head = fit_ridge_head(selector_features, selector_targets, alpha=selector_alpha)
    return ExtentPipeline(
        left_head, right_head, selector_head, cell_width_s, top_k, tuple(map(int, offsets)),
    )


def predict_extent_candidates(
    pipeline: ExtentPipeline, observation: ExtentObservation,
) -> tuple[tuple[FrameCandidate, ...], tuple[dict[str, Any], ...]]:
    rows = []
    for selected, features, start, end in _extent_candidates(
        observation, pipeline.left_head, pipeline.right_head, top_k=pipeline.top_k,
        cell_width_s=pipeline.cell_width_s, offsets=pipeline.offsets,
    ):
        selector_features = features + (
            math.log1p(end - start), start / observation.upper_bound_s,
            end / observation.upper_bound_s,
        )
        quality = pipeline.selector_head.predict(selector_features)
        payload = json.dumps({
            "cell_id": selected.token.cell_id, "start_s": start, "end_s": end,
        }, sort_keys=True, separators=(",", ":"))
        candidate_id = "extent-" + hashlib.sha1(payload.encode()).hexdigest()[:16]
        candidate = FrameCandidate(
            candidate_id, start, end, quality, selected.token.frame_refs,
            selected.token.frame_refs,
        )
        rows.append((candidate, {
            "candidate_id": candidate_id, "cell_id": selected.token.cell_id,
            "cell_rank": selected.rank, "token_score": selected.score,
            "predicted_quality": quality, "predicted_left_s": selected.token.center_s - start,
            "predicted_right_s": end - selected.token.center_s,
            "span": [start, end], "support_count": selected.token.support_count,
            "change_left": selected.token.change_left,
            "change_right": selected.token.change_right,
        }))
    rows.sort(key=lambda item: (-item[0].score, item[0].start_s, item[0].candidate_id))
    return tuple(item[0] for item in rows), tuple(item[1] for item in rows)
