"""Baseline-preserving JQ-01 candidate quality ranking.

The selector is deliberately snapshot-only.  Ground truth is accepted by the
training helpers in this module, never by feature extraction or inference.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from streamtimelens.observer.clip_encoder import deserialize_embedding, l2_normalize
from streamtimelens.protocol.snapshot import SnapshotReader
from streamtimelens.retrieval.extent_tokens import temporal_iou
from streamtimelens.retrieval.frame_candidates import FrameCandidate


SCHEMA_VERSION = "jq_feature_v1"
ARCHITECTURE = "layernorm-linear64-gelu-linear32-gelu-linear1"
BOUNDARY_WINDOW_S = 2.0
FEATURE_NAMES = (
    "pooled_cos", "inside_mean", "inside_max", "inside_std", "inside_p90",
    "left_mean", "left_max", "right_mean", "right_max",
    "inside_minus_left", "inside_minus_right",
    "start_norm", "end_norm", "center_norm", "width_norm", "log_width",
    "support_norm", "iou_with_baseline", "center_distance_norm", "log_width_ratio",
    "is_frozen_baseline", "raw_rank_norm", "pool_best_score", "pool_second_score",
    "score_to_best", "score_to_next", "pool_score_z", "max_pair_iou",
    "mean_pair_iou", "rho", "is_8mib", "inside_missing", "left_missing",
    "right_missing",
)
BINARY_FEATURES = (
    "is_frozen_baseline", "is_8mib", "inside_missing", "left_missing", "right_missing",
)
FORBIDDEN_INFERENCE_KEYS = {
    "gt", "gt_span", "ground_truth", "ground_truth_span", "annotation", "annotations",
    "good_baseline", "failure_layer", "oracle", "oracle_iou", "source", "dataset",
    "video_path", "video_root", "future_frame", "future_frames",
    "original_video", "original_video_path", "raw_video_path", "dataset_name",
    "data_source", "candidate_target", "candidate_targets",
}
CHECKPOINT_KEYS = {
    "format_version", "state_dict", "scaler", "feature_schema",
    "feature_schema_sha256", "architecture", "input_dimension", "loss",
    "optimizer", "outer_train_video_sha256", "outer_train_group_sha256",
    "code_revision", "training", "checkpoint_sha256",
}


def feature_schema() -> dict[str, Any]:
    """Return the canonical, ordered jq_feature_v1 contract."""
    return {
        "version": SCHEMA_VERSION,
        "feature_names": list(FEATURE_NAMES),
        "dimension": len(FEATURE_NAMES),
        "binary_features": list(BINARY_FEATURES),
        "boundary_window_s": BOUNDARY_WINDOW_S,
        "inside_interval": "[start_s,end_s]",
        "left_interval": "[start_s-2,start_s)",
        "right_interval": "(end_s,end_s+2]",
        "empty_window_value": 0.0,
        "pool_score": "pooled_cos",
        "raw_rank_normalizer": 8,
        "standardization": "outer-train continuous features only",
    }


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False,
    ).encode("utf-8")


def schema_sha256(schema: Mapping[str, Any] | None = None) -> str:
    return hashlib.sha256(canonical_json(dict(schema or feature_schema()))).hexdigest()


def write_feature_schema(path: Path | str) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(feature_schema(), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return destination


@dataclass(frozen=True)
class JointCandidate:
    candidate_id: str
    start_s: float
    end_s: float
    is_frozen_baseline: bool
    raw_rank: int
    extent_score: float = 0.0
    frame_refs: tuple[str, ...] = ()

    @property
    def span(self) -> tuple[float, float]:
        return self.start_s, self.end_s


@dataclass(frozen=True)
class CandidateFeatureRow:
    candidate: JointCandidate
    values: tuple[float, ...]

    def as_mapping(self) -> dict[str, float]:
        return dict(zip(FEATURE_NAMES, self.values))


@dataclass(frozen=True)
class FeatureScaler:
    mean: tuple[float, ...]
    scale: tuple[float, ...]
    fitted_video_sha256: str
    fitted_group_sha256: str

    def transform(self, values: Sequence[Sequence[float]]) -> np.ndarray:
        matrix = np.asarray(values, dtype=np.float32)
        if matrix.ndim != 2 or matrix.shape[1] != len(FEATURE_NAMES):
            raise ValueError("JQ feature dimension changed")
        if not np.isfinite(matrix).all():
            raise ValueError("JQ features contain NaN or Inf")
        return (matrix - np.asarray(self.mean, dtype=np.float32)) / np.asarray(
            self.scale, dtype=np.float32,
        )


@dataclass(frozen=True)
class QualityTrainingObservation:
    observation_id: str
    video_id: str
    group_id: str
    features: tuple[tuple[float, ...], ...]
    targets: tuple[float, ...]
    candidate_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        count = len(self.features)
        if (
            not self.observation_id or not self.video_id or not self.group_id
            or not 1 <= count <= 9 or len(self.targets) != count
            or len(self.candidate_ids) != count or len(set(self.candidate_ids)) != count
            or any(len(row) != len(FEATURE_NAMES) for row in self.features)
            or not np.isfinite(np.asarray(self.features, dtype=np.float64)).all()
            or not np.isfinite(np.asarray(self.targets, dtype=np.float64)).all()
            or any(not 0.0 <= value <= 1.0 for value in self.targets)
        ):
            raise ValueError("invalid JQ training observation")


def _validate_span(span: Sequence[float], duration_s: float, label: str) -> tuple[float, float]:
    if len(span) != 2:
        raise ValueError(f"{label} span must have two endpoints")
    start, end = float(span[0]), float(span[1])
    if (
        not math.isfinite(start) or not math.isfinite(end) or duration_s <= 0
        or start < 0 or end < start or end > duration_s + 1e-6
    ):
        raise ValueError(f"{label} span is invalid")
    return start, end


def _reject_forbidden(value: Any, path: str = "") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            normalized = str(key).lower().replace("-", "_")
            if normalized in FORBIDDEN_INFERENCE_KEYS:
                raise ValueError(f"forbidden JQ inference field: {path}{key}")
            _reject_forbidden(child, f"{path}{key}.")
    elif isinstance(value, (list, tuple)):
        for child in value:
            _reject_forbidden(child, path)


def validate_inference_payload(value: Any) -> None:
    """Reject annotation-, cohort-, future-, and original-video-bearing state."""
    _reject_forbidden(value)


def build_joint_candidates(
    baseline_span: Sequence[float], extent_candidates: Sequence[FrameCandidate], *, duration_s: float,
    upper_bound_s: float | None = None,
) -> tuple[JointCandidate, ...]:
    """Build the fixed 1..9 union while retaining duplicate baseline identity."""
    bound = duration_s if upper_bound_s is None else float(upper_bound_s)
    if not math.isfinite(bound) or bound <= 0 or bound > duration_s + 1e-6:
        raise ValueError("JQ query-visible upper bound is invalid")
    start, end = _validate_span(baseline_span, bound, "baseline")
    if len(extent_candidates) > 8:
        raise ValueError("JQ extent candidate cap is eight")
    seen_ids = {"frozen-v2-final"}
    normalized = []
    for candidate in extent_candidates:
        c_start, c_end = _validate_span(candidate.span, bound, "extent")
        if not candidate.candidate_id or candidate.candidate_id in seen_ids:
            raise ValueError("JQ candidate IDs must be unique")
        score = float(candidate.score)
        if not math.isfinite(score):
            raise ValueError("JQ extent score is not finite")
        seen_ids.add(candidate.candidate_id)
        normalized.append((candidate, c_start, c_end, score))
    # Recreate the frozen X1 order instead of trusting caller traversal order.
    normalized.sort(key=lambda row: (-row[3], row[1], row[0].candidate_id))
    baseline = JointCandidate("frozen-v2-final", start, end, True, 0)
    successors = tuple(
        JointCandidate(
            row[0].candidate_id, row[1], row[2], False, rank, row[3],
            tuple(row[0].frame_refs),
        )
        for rank, row in enumerate(normalized, 1)
    )
    return (baseline,) + successors


def _window_stats(scores: np.ndarray) -> tuple[float, float, float, float, float]:
    if scores.size == 0:
        return 0.0, 0.0, 0.0, 0.0, 1.0
    return (
        float(scores.mean()), float(scores.max()), float(scores.std()),
        float(np.percentile(scores, 90)), 0.0,
    )


def compute_jq_features(
    candidates: Sequence[JointCandidate], frame_metadata: Mapping[str, Mapping[str, Any]],
    query_embedding: Sequence[float], *, duration_s: float, rho: float, budget_bytes: int,
    upper_bound_s: float | None = None,
) -> tuple[CandidateFeatureRow, ...]:
    """Compute jq_feature_v1 without annotation, source, or original-video state."""
    _reject_forbidden(frame_metadata)
    if (
        not 1 <= len(candidates) <= 9 or not math.isfinite(float(duration_s))
        or duration_s <= 0 or not math.isfinite(float(rho)) or not 0 < rho <= 1
        or int(budget_bytes) <= 0
    ):
        raise ValueError("invalid JQ feature context")
    bound = duration_s if upper_bound_s is None else float(upper_bound_s)
    if not math.isfinite(bound) or bound <= 0 or bound > duration_s + 1e-6:
        raise ValueError("JQ query-visible upper bound is invalid")
    query = l2_normalize(query_embedding)
    if query.ndim != 1 or not np.isfinite(query).all():
        raise ValueError("invalid JQ query embedding")
    tokens = []
    dimension = None
    for frame_ref, row in frame_metadata.items():
        if "clip_embedding" not in row:
            continue
        timestamp = float(row["timestamp_s"])
        embedding = np.asarray(deserialize_embedding(dict(row["clip_embedding"])), dtype=np.float64)
        if (
            not math.isfinite(timestamp) or timestamp < 0 or timestamp > bound + 1e-6
            or embedding.ndim != 1 or not np.isfinite(embedding).all()
        ):
            raise ValueError("invalid JQ snapshot token")
        if dimension is None:
            dimension = embedding.shape[0]
        if embedding.shape[0] != dimension:
            raise ValueError("JQ snapshot token dimensions differ")
        if embedding.shape != query.shape:
            raise ValueError("JQ query and token dimensions differ")
        score = float(np.clip(np.dot(query, l2_normalize(embedding)), -1.0, 1.0))
        tokens.append((timestamp, str(frame_ref), l2_normalize(embedding), score))
    tokens.sort(key=lambda row: (row[0], row[1]))
    baseline = next((item for item in candidates if item.is_frozen_baseline), None)
    if baseline is None or sum(item.is_frozen_baseline for item in candidates) != 1:
        raise ValueError("JQ pool must contain exactly one frozen baseline")
    for item in candidates:
        _validate_span(item.span, bound, item.candidate_id)

    primitive = []
    for candidate in candidates:
        inside = [row for row in tokens if candidate.start_s <= row[0] <= candidate.end_s]
        left = np.asarray([
            row[3] for row in tokens
            if candidate.start_s - BOUNDARY_WINDOW_S <= row[0] < candidate.start_s
        ], dtype=np.float64)
        right = np.asarray([
            row[3] for row in tokens
            if candidate.end_s < row[0] <= candidate.end_s + BOUNDARY_WINDOW_S
        ], dtype=np.float64)
        inside_scores = np.asarray([row[3] for row in inside], dtype=np.float64)
        inside_mean, inside_max, inside_std, inside_p90, inside_missing = _window_stats(inside_scores)
        left_mean, left_max, _, _, left_missing = _window_stats(left)
        right_mean, right_max, _, _, right_missing = _window_stats(right)
        if inside:
            pooled = l2_normalize(np.mean([row[2] for row in inside], axis=0))
            pooled_cos = float(np.clip(np.dot(query, pooled), -1.0, 1.0))
        else:
            pooled_cos = 0.0
        primitive.append({
            "candidate": candidate, "pooled_cos": pooled_cos,
            "inside_mean": inside_mean, "inside_max": inside_max, "inside_std": inside_std,
            "inside_p90": inside_p90, "left_mean": left_mean, "left_max": left_max,
            "right_mean": right_mean, "right_max": right_max,
            "inside_missing": inside_missing, "left_missing": left_missing,
            "right_missing": right_missing, "support": len(inside),
        })

    scores = np.asarray([row["pooled_cos"] for row in primitive], dtype=np.float64)
    ranked_indices = sorted(
        range(len(candidates)), key=lambda index: (-scores[index], candidates[index].candidate_id),
    )
    rank_position = {index: position for position, index in enumerate(ranked_indices)}
    best = float(scores[ranked_indices[0]])
    second = float(scores[ranked_indices[1]]) if len(scores) > 1 else best
    score_std = float(scores.std())
    score_mean = float(scores.mean())
    rows = []
    for index, values in enumerate(primitive):
        candidate = values["candidate"]
        width = candidate.end_s - candidate.start_s
        baseline_width = baseline.end_s - baseline.start_s
        center = (candidate.start_s + candidate.end_s) / 2.0
        baseline_center = (baseline.start_s + baseline.end_s) / 2.0
        others = [
            temporal_iou(candidate.span, other.span)
            for other in candidates if other.candidate_id != candidate.candidate_id
        ]
        position = rank_position[index]
        next_score = float(scores[ranked_indices[position + 1]]) if position + 1 < len(scores) else None
        mapping = {
            "pooled_cos": values["pooled_cos"],
            "inside_mean": values["inside_mean"], "inside_max": values["inside_max"],
            "inside_std": values["inside_std"], "inside_p90": values["inside_p90"],
            "left_mean": values["left_mean"], "left_max": values["left_max"],
            "right_mean": values["right_mean"], "right_max": values["right_max"],
            "inside_minus_left": (
                values["inside_mean"] - values["left_mean"]
                if not values["inside_missing"] and not values["left_missing"] else 0.0
            ),
            "inside_minus_right": (
                values["inside_mean"] - values["right_mean"]
                if not values["inside_missing"] and not values["right_missing"] else 0.0
            ),
            "start_norm": candidate.start_s / duration_s,
            "end_norm": candidate.end_s / duration_s,
            "center_norm": center / duration_s, "width_norm": width / duration_s,
            "log_width": math.log1p(width), "support_norm": values["support"] / max(1, len(tokens)),
            "iou_with_baseline": temporal_iou(candidate.span, baseline.span),
            "center_distance_norm": abs(center - baseline_center) / duration_s,
            "log_width_ratio": math.log((width + 1e-6) / (baseline_width + 1e-6)),
            "is_frozen_baseline": float(candidate.is_frozen_baseline),
            "raw_rank_norm": 0.0 if candidate.is_frozen_baseline else candidate.raw_rank / 8.0,
            "pool_best_score": best, "pool_second_score": second,
            "score_to_best": float(scores[index]) - best,
            "score_to_next": 0.0 if next_score is None else float(scores[index]) - next_score,
            "pool_score_z": 0.0 if score_std == 0 else (float(scores[index]) - score_mean) / score_std,
            "max_pair_iou": max(others) if others else 0.0,
            "mean_pair_iou": float(np.mean(others)) if others else 0.0,
            "rho": float(rho), "is_8mib": float(int(budget_bytes) == 8 * 1024 * 1024),
            "inside_missing": values["inside_missing"], "left_missing": values["left_missing"],
            "right_missing": values["right_missing"],
        }
        vector = tuple(float(mapping[name]) for name in FEATURE_NAMES)
        if not np.isfinite(np.asarray(vector)).all():
            raise ValueError("JQ features contain NaN or Inf")
        rows.append(CandidateFeatureRow(candidate, vector))
    return tuple(rows)


def _identity_sha(values: Sequence[str]) -> str:
    return hashlib.sha256(canonical_json(sorted(set(map(str, values))))).hexdigest()


def _valid_sha(value: Any, length: int = 64) -> bool:
    text = str(value)
    return len(text) == length and all(character in "0123456789abcdef" for character in text)


def fit_feature_scaler(
    observations: Sequence[QualityTrainingObservation], *,
    heldout_video_ids: Sequence[str] = (), heldout_group_ids: Sequence[str] = (),
) -> FeatureScaler:
    if not observations:
        raise ValueError("cannot fit JQ scaler without outer-train observations")
    videos = {row.video_id for row in observations}
    groups = {row.group_id for row in observations}
    if videos & set(heldout_video_ids) or groups & set(heldout_group_ids):
        raise ValueError("held-out identities entered JQ scaler fit")
    matrix = np.asarray([feature for row in observations for feature in row.features], dtype=np.float64)
    mean = matrix.mean(axis=0)
    scale = matrix.std(axis=0)
    binary_indices = {FEATURE_NAMES.index(name) for name in BINARY_FEATURES}
    for index in range(len(FEATURE_NAMES)):
        if index in binary_indices:
            mean[index], scale[index] = 0.0, 1.0
        elif scale[index] < 1e-8:
            scale[index] = 1.0
    return FeatureScaler(
        tuple(map(float, mean)), tuple(map(float, scale)),
        _identity_sha(tuple(videos)), _identity_sha(tuple(groups)),
    )


def _torch():
    try:
        import torch
        from torch import nn
    except ImportError as exc:  # pragma: no cover - deployment dependency guard
        raise RuntimeError("JQ-01 requires PyTorch for its quality head") from exc
    return torch, nn


class JointQualityHead:
    """Thin wrapper around the frozen shared MLP architecture."""

    def __init__(self, input_dim: int = len(FEATURE_NAMES), *, seed: int = 20260911) -> None:
        if input_dim != len(FEATURE_NAMES):
            raise ValueError("JQ head input dimension must match jq_feature_v1")
        torch, nn = _torch()
        torch.manual_seed(seed)
        self.input_dim = input_dim
        self.module = nn.Sequential(
            nn.LayerNorm(input_dim), nn.Linear(input_dim, 64), nn.GELU(),
            nn.Linear(64, 32), nn.GELU(), nn.Linear(32, 1),
        )

    def __call__(self, values: Any) -> Any:
        return self.module(values).squeeze(-1)

    def state_dict(self) -> dict[str, Any]:
        return self.module.state_dict()

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self.module.load_state_dict(state, strict=True)


def joint_quality_loss(logit_groups: Sequence[Any], target_groups: Sequence[Any]) -> Any:
    """IoU-balanced SmoothL1 plus observation-equal ListNet."""
    torch, _ = _torch()
    if not logit_groups or len(logit_groups) != len(target_groups):
        raise ValueError("JQ loss needs paired observation groups")
    losses = []
    for logits, targets in zip(logit_groups, target_groups):
        targets = torch.as_tensor(targets, dtype=logits.dtype, device=logits.device)
        if logits.ndim != 1 or targets.shape != logits.shape or logits.numel() < 1:
            raise ValueError("invalid JQ loss group")
        quality = torch.sigmoid(logits)
        bin_masks = (
            targets <= 0.3,
            (targets > 0.3) & (targets <= 0.7),
            targets > 0.7,
        )
        bin_losses = [
            torch.nn.functional.smooth_l1_loss(quality[mask], targets[mask], reduction="mean")
            for mask in bin_masks if bool(mask.any())
        ]
        quality_loss = torch.stack(bin_losses).mean()
        target_distribution = torch.softmax(targets / 0.1, dim=0)
        rank_loss = -(target_distribution * torch.log_softmax(logits / 1.0, dim=0)).sum()
        losses.append(quality_loss + rank_loss)
    return torch.stack(losses).mean()


def stable_argmax(logits: Sequence[float], candidate_ids: Sequence[str]) -> int:
    values = np.asarray(logits, dtype=np.float64)
    if (
        values.ndim != 1 or len(values) != len(candidate_ids) or not len(values)
        or not np.isfinite(values).all() or len(set(candidate_ids)) != len(candidate_ids)
    ):
        raise ValueError("invalid JQ scores")
    return min(range(len(values)), key=lambda index: (-values[index], candidate_ids[index]))


def train_joint_quality_head(
    training: Sequence[QualityTrainingObservation],
    validation: Sequence[QualityTrainingObservation], *, seed: int = 20260911,
    device: str = "cpu", max_epochs: int = 100, patience: int = 10,
    learning_rate: float = 1e-3, weight_decay: float = 1e-4,
    heldout_video_ids: Sequence[str] = (), heldout_group_ids: Sequence[str] = (),
) -> tuple[JointQualityHead, FeatureScaler, dict[str, Any]]:
    if (
        not training or not validation or max_epochs != 100 or patience != 10
        or learning_rate != 1e-3 or weight_decay != 1e-4
    ):
        raise ValueError("JQ training requires frozen epoch and patience settings")
    train_videos, train_groups = {row.video_id for row in training}, {row.group_id for row in training}
    val_videos, val_groups = {row.video_id for row in validation}, {row.group_id for row in validation}
    if train_videos & val_videos or train_groups & val_groups:
        raise ValueError("JQ inner train and validation identities overlap")
    if (train_videos | val_videos) & set(heldout_video_ids) or (
        (train_groups | val_groups) & set(heldout_group_ids)
    ):
        raise ValueError("JQ outer-held-out identity entered training")
    scaler = fit_feature_scaler(
        training, heldout_video_ids=tuple(val_videos) + tuple(heldout_video_ids),
        heldout_group_ids=tuple(val_groups) + tuple(heldout_group_ids),
    )
    torch, _ = _torch()
    torch.manual_seed(seed)
    if hasattr(torch, "use_deterministic_algorithms"):
        torch.use_deterministic_algorithms(True)
    model = JointQualityHead(seed=seed)
    model.module.to(device)
    optimizer = torch.optim.AdamW(
        model.module.parameters(), lr=learning_rate, weight_decay=weight_decay,
    )

    def tensors(rows: Sequence[QualityTrainingObservation]):
        return [(
            torch.as_tensor(scaler.transform(row.features), dtype=torch.float32, device=device),
            torch.as_tensor(row.targets, dtype=torch.float32, device=device), row,
        ) for row in sorted(rows, key=lambda item: item.observation_id)]

    train_tensors, validation_tensors = tensors(training), tensors(validation)
    best_state, best_score, best_epoch, stale = None, -math.inf, 0, 0
    history = []
    for epoch in range(1, max_epochs + 1):
        model.module.train()
        optimizer.zero_grad(set_to_none=True)
        logits = [model(features) for features, _, _ in train_tensors]
        loss = joint_quality_loss(logits, [targets for _, targets, _ in train_tensors])
        loss.backward()
        optimizer.step()
        model.module.eval()
        with torch.no_grad():
            ious_by_video: dict[str, list[float]] = {}
            for features, targets, row in validation_tensors:
                predicted = model(features).detach().cpu().numpy()
                selected = stable_argmax(predicted, row.candidate_ids)
                ious_by_video.setdefault(row.video_id, []).append(float(targets[selected].item()))
            score = float(np.mean([
                np.mean(values) for values in ious_by_video.values()
            ]))
        history.append({"epoch": epoch, "train_loss": float(loss.item()), "validation_candidate_miou": score})
        if score > best_score + 1e-12:
            best_score, best_epoch, stale = score, epoch, 0
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        else:
            stale += 1
            if stale >= patience:
                break
    assert best_state is not None
    model.load_state_dict(best_state)
    model.module.to("cpu").eval()
    return model, scaler, {
        "best_epoch": best_epoch, "best_validation_candidate_miou": best_score,
        "epochs_ran": len(history), "history": history, "seed": seed, "device": device,
        "deterministic_algorithms": True, "optimizer": "AdamW", "learning_rate": learning_rate,
        "weight_decay": weight_decay, "max_epochs": max_epochs, "patience": patience,
    }


def fit_joint_quality_fixed_epochs(
    training: Sequence[QualityTrainingObservation], scaler: FeatureScaler, *,
    epochs: int, seed: int = 20260911, device: str = "cpu",
    learning_rate: float = 1e-3, weight_decay: float = 1e-4,
) -> JointQualityHead:
    """Fit the frozen head for an inner-selected number of epochs."""
    if (
        not training or not 1 <= int(epochs) <= 100
        or learning_rate != 1e-3 or weight_decay != 1e-4
    ):
        raise ValueError("invalid fixed-epoch JQ training request")
    if scaler.fitted_video_sha256 != _identity_sha([row.video_id for row in training]) or (
        scaler.fitted_group_sha256 != _identity_sha([row.group_id for row in training])
    ):
        raise ValueError("JQ fixed-epoch scaler does not match training identities")
    torch, _ = _torch()
    torch.manual_seed(seed)
    if hasattr(torch, "use_deterministic_algorithms"):
        torch.use_deterministic_algorithms(True)
    model = JointQualityHead(seed=seed)
    model.module.to(device)
    optimizer = torch.optim.AdamW(
        model.module.parameters(), lr=learning_rate, weight_decay=weight_decay,
    )
    tensors = [(
        torch.as_tensor(scaler.transform(row.features), dtype=torch.float32, device=device),
        torch.as_tensor(row.targets, dtype=torch.float32, device=device),
    ) for row in sorted(training, key=lambda item: item.observation_id)]
    for _ in range(int(epochs)):
        optimizer.zero_grad(set_to_none=True)
        loss = joint_quality_loss(
            [model(features) for features, _ in tensors],
            [targets for _, targets in tensors],
        )
        loss.backward()
        optimizer.step()
    model.module.to("cpu").eval()
    return model


def _serialize_state_dict(state: Mapping[str, Any]) -> dict[str, Any]:
    return {name: value.detach().cpu().tolist() for name, value in sorted(state.items())}


def _deserialize_state_dict(state: Mapping[str, Any]) -> dict[str, Any]:
    torch, _ = _torch()
    return {name: torch.as_tensor(value, dtype=torch.float32) for name, value in state.items()}


def make_checkpoint(
    model: JointQualityHead, scaler: FeatureScaler, *, outer_train_video_sha256: str,
    outer_train_group_sha256: str, code_revision: str, training_metadata: Mapping[str, Any],
) -> dict[str, Any]:
    _reject_forbidden(training_metadata)
    if (
        not _valid_sha(outer_train_video_sha256)
        or not _valid_sha(outer_train_group_sha256)
        or not _valid_sha(code_revision, 40)
        or scaler.fitted_video_sha256 != outer_train_video_sha256
        or scaler.fitted_group_sha256 != outer_train_group_sha256
    ):
        raise ValueError("JQ checkpoint training identity is invalid")
    schema = feature_schema()
    payload = {
        "format_version": 1, "state_dict": _serialize_state_dict(model.state_dict()),
        "scaler": asdict(scaler), "feature_schema": schema,
        "feature_schema_sha256": schema_sha256(schema), "architecture": ARCHITECTURE,
        "input_dimension": len(FEATURE_NAMES),
        "loss": {
            "quality": "three-bin-equal-SmoothL1(sigmoid(logit),iou)",
            "bins": [[0.0, 0.3], [0.3, 0.7], [0.7, 1.0]],
            "listwise": "ListNet", "target_temperature": 0.1, "logit_temperature": 1.0,
            "quality_weight": 1.0, "ranking_weight": 1.0,
        },
        "optimizer": {
            "name": "AdamW", "learning_rate": 0.001, "weight_decay": 0.0001,
            "max_epochs": 100, "patience": 10,
        },
        "outer_train_video_sha256": outer_train_video_sha256,
        "outer_train_group_sha256": outer_train_group_sha256,
        "code_revision": code_revision, "training": dict(training_metadata),
    }
    payload["checkpoint_sha256"] = hashlib.sha256(canonical_json(payload)).hexdigest()
    return payload


def save_checkpoint(checkpoint: Mapping[str, Any], path: Path | str) -> Path:
    _reject_forbidden(checkpoint)
    candidate = dict(checkpoint)
    claimed = candidate.pop("checkpoint_sha256", None)
    if claimed != hashlib.sha256(canonical_json(candidate)).hexdigest():
        raise ValueError("JQ checkpoint payload SHA mismatch")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(
        dict(checkpoint), ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ) + "\n"
    destination.write_text(serialized, encoding="utf-8")
    digest = hashlib.sha256(destination.read_bytes()).hexdigest()
    destination.with_suffix(destination.suffix + ".sha256").write_text(digest + "\n", encoding="ascii")
    return destination


def load_checkpoint(path: Path | str) -> tuple[JointQualityHead, FeatureScaler, dict[str, Any]]:
    source = Path(path)
    sidecar = source.with_suffix(source.suffix + ".sha256")
    if not source.is_file() or not sidecar.is_file():
        raise ValueError("JQ checkpoint or SHA sidecar is missing")
    if hashlib.sha256(source.read_bytes()).hexdigest() != sidecar.read_text(encoding="ascii").strip():
        raise ValueError("JQ checkpoint file SHA mismatch")
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or set(payload) != CHECKPOINT_KEYS:
        raise ValueError("JQ checkpoint fields changed")
    _reject_forbidden(payload)
    claimed = payload.pop("checkpoint_sha256", None)
    if claimed != hashlib.sha256(canonical_json(payload)).hexdigest():
        raise ValueError("JQ checkpoint payload SHA mismatch")
    payload["checkpoint_sha256"] = claimed
    schema = feature_schema()
    if (
        payload.get("format_version") != 1 or payload.get("feature_schema") != schema
        or payload.get("feature_schema_sha256") != schema_sha256(schema)
        or payload.get("architecture") != ARCHITECTURE
        or payload.get("input_dimension") != len(FEATURE_NAMES)
    ):
        raise ValueError("JQ checkpoint schema or architecture mismatch")
    scaler_row = dict(payload.get("scaler", {}))
    scaler_row["mean"] = tuple(scaler_row.get("mean", ()))
    scaler_row["scale"] = tuple(scaler_row.get("scale", ()))
    scaler = FeatureScaler(**scaler_row)
    if (
        len(scaler.mean) != len(FEATURE_NAMES) or len(scaler.scale) != len(FEATURE_NAMES)
        or not np.isfinite(np.asarray(scaler.mean)).all()
        or not np.isfinite(np.asarray(scaler.scale)).all()
        or any(value <= 0 for value in scaler.scale)
        or not _valid_sha(scaler.fitted_video_sha256)
        or not _valid_sha(scaler.fitted_group_sha256)
        or not _valid_sha(payload.get("code_revision"), 40)
        or payload.get("outer_train_video_sha256") != scaler.fitted_video_sha256
        or payload.get("outer_train_group_sha256") != scaler.fitted_group_sha256
        or payload.get("loss") != {
            "quality": "three-bin-equal-SmoothL1(sigmoid(logit),iou)",
            "bins": [[0.0, 0.3], [0.3, 0.7], [0.7, 1.0]],
            "listwise": "ListNet", "target_temperature": 0.1, "logit_temperature": 1.0,
            "quality_weight": 1.0, "ranking_weight": 1.0,
        }
        or payload.get("optimizer") != {
            "name": "AdamW", "learning_rate": 0.001, "weight_decay": 0.0001,
            "max_epochs": 100, "patience": 10,
        }
    ):
        raise ValueError("JQ checkpoint scaler dimension mismatch")
    model = JointQualityHead()
    model.load_state_dict(_deserialize_state_dict(payload["state_dict"]))
    model.module.eval()
    return model, scaler, payload


def snapshot_fingerprint(snapshot: SnapshotReader) -> tuple[tuple[str, int, str], ...]:
    rows = []
    for relative in sorted(snapshot.manifest.allowed_files):
        path = snapshot.path(relative)
        rows.append((relative, path.stat().st_size, hashlib.sha256(path.read_bytes()).hexdigest()))
    return tuple(rows)


def select_joint_quality(
    baseline_row: Mapping[str, Any], snapshot: SnapshotReader, query_embedding: Sequence[float],
    extent_candidates: Sequence[FrameCandidate], *, checkpoint_path: Path | str,
    rho: float, budget_bytes: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Select a candidate or return an untouched deep copy of Frozen V2 on any fault."""
    fallback = copy.deepcopy(dict(baseline_row))
    debug: dict[str, Any] = {
        "fallback": True, "fallback_reason": None, "pre_span": copy.deepcopy(fallback.get("final_span")),
        "final_span": copy.deepcopy(fallback.get("final_span")), "candidates": [],
    }
    try:
        if not isinstance(snapshot, SnapshotReader):
            raise ValueError("JQ inference requires a verified SnapshotReader")
        verified_snapshot = SnapshotReader(snapshot.root)
        _reject_forbidden({"selector_input": {key: value for key, value in baseline_row.items() if key != "final_span"}})
        before = snapshot_fingerprint(verified_snapshot)
        duration = float(verified_snapshot.manifest.video_meta["duration_s"])
        candidates = build_joint_candidates(
            baseline_row["final_span"], extent_candidates, duration_s=duration,
            upper_bound_s=float(verified_snapshot.manifest.t_q),
        )
        if len(candidates) == 1:
            raise ValueError("no valid X1 extent candidates")
        features = compute_jq_features(
            candidates, verified_snapshot.read_frame_metadata(), query_embedding,
            duration_s=duration, rho=rho, budget_bytes=budget_bytes,
            upper_bound_s=float(verified_snapshot.manifest.t_q),
        )
        model, scaler, checkpoint = load_checkpoint(checkpoint_path)
        torch, _ = _torch()
        matrix = scaler.transform([row.values for row in features])
        with torch.no_grad():
            logits = model(torch.as_tensor(matrix, dtype=torch.float32)).cpu().numpy()
        selected_index = stable_argmax(logits, [row.candidate.candidate_id for row in features])
        after_reader = SnapshotReader(verified_snapshot.root)
        after = snapshot_fingerprint(after_reader)
        if before != after:
            raise ValueError("snapshot changed during JQ inference")
        selected = features[selected_index].candidate
        debug["candidates"] = [
            {
                "candidate_id": row.candidate.candidate_id, "span": list(row.candidate.span),
                "features": row.as_mapping(), "logit": float(logits[index]),
                "rank": sorted(
                    range(len(logits)), key=lambda item: (-float(logits[item]), features[item].candidate.candidate_id),
                ).index(index) + 1,
                "selected": index == selected_index,
            }
            for index, row in enumerate(features)
        ]
        debug["checkpoint_sha256"] = checkpoint["checkpoint_sha256"]
        debug["selected_candidate_id"] = selected.candidate_id
        debug["fallback"] = bool(selected.is_frozen_baseline)
        debug["fallback_reason"] = "model_selected_frozen_baseline" if selected.is_frozen_baseline else None
        if selected.is_frozen_baseline:
            return fallback, debug
        result = copy.deepcopy(fallback)
        result["final_span"] = [selected.start_s, selected.end_s]
        result["jq01_selected_candidate_id"] = selected.candidate_id
        debug["final_span"] = list(selected.span)
        return result, debug
    except Exception as exc:
        debug["fallback_reason"] = f"{type(exc).__name__}: {exc}"
        return fallback, debug


# The v1 aggregate-feature implementation above is retained for the immutable r1
# record and the generic J2 fixtures.  The active final JQ-01 revision uses the
# temporal contract below.  Keeping a separate schema/checkpoint format makes it
# impossible to silently load an r1 checkpoint into the structural model.
TEMPORAL_SCHEMA_VERSION = "jq_temporal_v2"
TEMPORAL_ARCHITECTURE = "split-gru16-state48-pair64-context-abstain-v1"
TEMPORAL_TOKEN_FEATURES = (
    "query_cosine", "relative_time", "absolute_time", "cosine_delta", "time_gap",
)
CENTER_TOKEN_CAP = 16
BOUNDARY_TOKEN_CAP = 8
TEMPORAL_HIDDEN_DIM = 16
TEMPORAL_STATE_DIM = 48
TEMPORAL_LOSS_WEIGHTS = {
    "quality": 1.0,
    "gain": 1.0,
    "left_boundary": 0.5,
    "right_boundary": 0.5,
    "listwise": 1.0,
    "abstention": 0.5,
    "relative_regret": 2.0,
}
TEMPORAL_CHECKPOINT_KEYS = {
    "format_version", "state_dict", "scaler", "temporal_schema",
    "temporal_schema_sha256", "architecture", "loss", "optimizer",
    "outer_train_video_sha256", "outer_train_group_sha256", "code_revision",
    "training", "checkpoint_sha256",
}


def temporal_feature_schema() -> dict[str, Any]:
    """Return the frozen candidate-local temporal-token contract."""
    return {
        "version": TEMPORAL_SCHEMA_VERSION,
        "scalar_schema": feature_schema(),
        "token_feature_names": list(TEMPORAL_TOKEN_FEATURES),
        "token_dimension": len(TEMPORAL_TOKEN_FEATURES),
        "center_token_cap": CENTER_TOKEN_CAP,
        "left_token_cap": BOUNDARY_TOKEN_CAP,
        "right_token_cap": BOUNDARY_TOKEN_CAP,
        "boundary_window_s": BOUNDARY_WINDOW_S,
        "center_interval": "[start_s,end_s]",
        "left_interval": "[start_s-2,start_s)",
        "right_interval": "(end_s,end_s+2]",
        "sampling": "stable-endpoint-preserving-uniform-index",
        "ordering": "timestamp-then-frame-ref",
        "encoder": "independent-center-left-right-GRU",
        "pool_context": "candidate-state-mean-and-max",
        "selection": "candidate-rank-logit-with-baseline-abstention-logit-stable-argmax",
    }


def temporal_schema_sha256(schema: Mapping[str, Any] | None = None) -> str:
    return hashlib.sha256(canonical_json(dict(schema or temporal_feature_schema()))).hexdigest()


def write_temporal_feature_schema(path: Path | str) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(temporal_feature_schema(), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return destination


@dataclass(frozen=True)
class TemporalCandidateFeatureRow:
    """Inference-only structural representation for one fixed candidate."""

    candidate: JointCandidate
    scalar_values: tuple[float, ...]
    center_tokens: tuple[tuple[float, ...], ...]
    left_tokens: tuple[tuple[float, ...], ...]
    right_tokens: tuple[tuple[float, ...], ...]

    def __post_init__(self) -> None:
        sequences = (self.center_tokens, self.left_tokens, self.right_tokens)
        if (
            len(self.scalar_values) != len(FEATURE_NAMES)
            or len(self.center_tokens) > CENTER_TOKEN_CAP
            or len(self.left_tokens) > BOUNDARY_TOKEN_CAP
            or len(self.right_tokens) > BOUNDARY_TOKEN_CAP
            or any(
                len(token) != len(TEMPORAL_TOKEN_FEATURES)
                for sequence in sequences for token in sequence
            )
            or not np.isfinite(np.asarray(self.scalar_values, dtype=np.float64)).all()
            or any(
                not np.isfinite(np.asarray(sequence, dtype=np.float64)).all()
                for sequence in sequences if sequence
            )
        ):
            raise ValueError("invalid JQ temporal candidate features")


@dataclass(frozen=True)
class TemporalTrainingObservation:
    """Evaluation-owned targets paired with inference-only temporal rows."""

    observation_id: str
    video_id: str
    group_id: str
    rows: tuple[TemporalCandidateFeatureRow, ...]
    quality_targets: tuple[float, ...]
    gain_targets: tuple[float, ...]
    left_boundary_targets: tuple[float, ...]
    right_boundary_targets: tuple[float, ...]

    def __post_init__(self) -> None:
        count = len(self.rows)
        targets = (
            self.quality_targets, self.gain_targets,
            self.left_boundary_targets, self.right_boundary_targets,
        )
        if (
            not self.observation_id or not self.video_id or not self.group_id
            or not 1 <= count <= 9
            or any(len(values) != count for values in targets)
            or len({row.candidate.candidate_id for row in self.rows}) != count
            or sum(row.candidate.is_frozen_baseline for row in self.rows) != 1
            or any(not np.isfinite(np.asarray(values, dtype=np.float64)).all() for values in targets)
            or any(not 0.0 <= value <= 1.0 for value in self.quality_targets)
            or any(not -1.0 <= value <= 1.0 for value in self.gain_targets)
            or any(not 0.0 <= value <= 1.0 for value in self.left_boundary_targets)
            or any(not 0.0 <= value <= 1.0 for value in self.right_boundary_targets)
        ):
            raise ValueError("invalid JQ temporal training observation")

    @property
    def candidate_ids(self) -> tuple[str, ...]:
        return tuple(row.candidate.candidate_id for row in self.rows)

    @property
    def baseline_index(self) -> int:
        return next(index for index, row in enumerate(self.rows) if row.candidate.is_frozen_baseline)


def make_temporal_training_observation(
    observation_id: str, video_id: str, group_id: str,
    rows: Sequence[TemporalCandidateFeatureRow], gt_span: Sequence[float], *, duration_s: float,
) -> TemporalTrainingObservation:
    """Create supervision in the evaluation layer; no target enters a feature row."""
    gt_start, gt_end = _validate_span(gt_span, duration_s, "ground-truth")
    rows = tuple(rows)
    if not rows:
        raise ValueError("JQ temporal training observation has no candidates")
    quality = tuple(temporal_iou(row.candidate.span, (gt_start, gt_end)) for row in rows)
    baseline_index = next(
        index for index, row in enumerate(rows) if row.candidate.is_frozen_baseline
    )
    gain = tuple(value - quality[baseline_index] for value in quality)
    gt_width = max(gt_end - gt_start, 1e-6)
    left = tuple(
        float(math.exp(-abs(row.candidate.start_s - gt_start) / gt_width)) for row in rows
    )
    right = tuple(
        float(math.exp(-abs(row.candidate.end_s - gt_end) / gt_width)) for row in rows
    )
    return TemporalTrainingObservation(
        observation_id, video_id, group_id, rows, quality, gain, left, right,
    )


def _stable_subsample(rows: Sequence[Any], cap: int) -> list[Any]:
    """Uniformly cap an ordered sequence while preserving both endpoints."""
    rows = list(rows)
    if len(rows) <= cap:
        return rows
    indices = np.rint(np.linspace(0, len(rows) - 1, cap)).astype(np.int64)
    if len(set(map(int, indices))) != cap:
        raise RuntimeError("JQ temporal subsampling produced duplicate indices")
    return [rows[int(index)] for index in indices]


def _temporal_region_features(
    tokens: Sequence[tuple[float, str, float]], *, region_start: float, region_end: float,
    duration_s: float, cap: int,
) -> tuple[tuple[float, ...], ...]:
    selected = _stable_subsample(tokens, cap)
    width = max(region_end - region_start, 1e-6)
    result = []
    previous_score = None
    previous_time = None
    for timestamp, _, score in selected:
        relative = 2.0 * (timestamp - region_start) / width - 1.0
        result.append((
            float(score), float(np.clip(relative, -1.0, 1.0)),
            float(timestamp / duration_s),
            0.0 if previous_score is None else float(score - previous_score),
            0.0 if previous_time is None else float((timestamp - previous_time) / width),
        ))
        previous_score, previous_time = score, timestamp
    return tuple(result)


def compute_temporal_jq_features(
    candidates: Sequence[JointCandidate], frame_metadata: Mapping[str, Mapping[str, Any]],
    query_embedding: Sequence[float], *, duration_s: float, rho: float, budget_bytes: int,
    upper_bound_s: float | None = None,
) -> tuple[TemporalCandidateFeatureRow, ...]:
    """Build ordered candidate-local center/left/right token sequences."""
    scalar_rows = compute_jq_features(
        candidates, frame_metadata, query_embedding, duration_s=duration_s, rho=rho,
        budget_bytes=budget_bytes, upper_bound_s=upper_bound_s,
    )
    bound = duration_s if upper_bound_s is None else float(upper_bound_s)
    query = l2_normalize(query_embedding)
    tokens = []
    dimension = None
    for frame_ref, raw in frame_metadata.items():
        if "clip_embedding" not in raw:
            continue
        timestamp = float(raw["timestamp_s"])
        embedding = np.asarray(
            deserialize_embedding(dict(raw["clip_embedding"])), dtype=np.float64,
        )
        if dimension is None:
            dimension = embedding.shape[0]
        if (
            not math.isfinite(timestamp) or timestamp < 0 or timestamp > bound + 1e-6
            or embedding.ndim != 1 or embedding.shape[0] != dimension
            or embedding.shape != query.shape or not np.isfinite(embedding).all()
        ):
            raise ValueError("invalid JQ temporal snapshot token")
        score = float(np.clip(np.dot(query, l2_normalize(embedding)), -1.0, 1.0))
        tokens.append((timestamp, str(frame_ref), score))
    tokens.sort(key=lambda row: (row[0], row[1]))
    result = []
    for scalar_row in scalar_rows:
        candidate = scalar_row.candidate
        center = [row for row in tokens if candidate.start_s <= row[0] <= candidate.end_s]
        left_start = max(0.0, candidate.start_s - BOUNDARY_WINDOW_S)
        left = [row for row in tokens if left_start <= row[0] < candidate.start_s]
        right_end = min(bound, candidate.end_s + BOUNDARY_WINDOW_S)
        right = [row for row in tokens if candidate.end_s < row[0] <= right_end]
        result.append(TemporalCandidateFeatureRow(
            candidate=candidate,
            scalar_values=scalar_row.values,
            center_tokens=_temporal_region_features(
                center, region_start=candidate.start_s, region_end=candidate.end_s,
                duration_s=duration_s, cap=CENTER_TOKEN_CAP,
            ),
            left_tokens=_temporal_region_features(
                left, region_start=left_start, region_end=candidate.start_s,
                duration_s=duration_s, cap=BOUNDARY_TOKEN_CAP,
            ),
            right_tokens=_temporal_region_features(
                right, region_start=candidate.end_s, region_end=right_end,
                duration_s=duration_s, cap=BOUNDARY_TOKEN_CAP,
            ),
        ))
    return tuple(result)


class TemporalJointQualityHead:
    """Split event/boundary encoder with pairwise and listwise context."""

    def __init__(self, *, seed: int = 20260911) -> None:
        torch, nn = _torch()
        torch.manual_seed(seed)
        token_dim = len(TEMPORAL_TOKEN_FEATURES)
        self.module = nn.ModuleDict({
            "center_gru": nn.GRU(token_dim, TEMPORAL_HIDDEN_DIM, batch_first=True),
            "left_gru": nn.GRU(token_dim, TEMPORAL_HIDDEN_DIM, batch_first=True),
            "right_gru": nn.GRU(token_dim, TEMPORAL_HIDDEN_DIM, batch_first=True),
            "center_missing": nn.Embedding(1, TEMPORAL_HIDDEN_DIM),
            "left_missing": nn.Embedding(1, TEMPORAL_HIDDEN_DIM),
            "right_missing": nn.Embedding(1, TEMPORAL_HIDDEN_DIM),
            "scalar": nn.Sequential(
                nn.LayerNorm(len(FEATURE_NAMES)), nn.Linear(len(FEATURE_NAMES), 32), nn.GELU(),
            ),
            "state": nn.Sequential(
                nn.Linear(3 * TEMPORAL_HIDDEN_DIM + 32, TEMPORAL_STATE_DIM), nn.GELU(),
            ),
            "pair": nn.Sequential(
                nn.Linear(6 * TEMPORAL_STATE_DIM, 64), nn.GELU(),
                nn.Linear(64, 32), nn.GELU(),
            ),
            "quality": nn.Linear(32, 1),
            "gain": nn.Linear(32, 1),
            "left_quality": nn.Linear(32, 1),
            "right_quality": nn.Linear(32, 1),
            "rank": nn.Linear(32, 1),
            "abstain": nn.Sequential(
                nn.Linear(3 * TEMPORAL_STATE_DIM, 32), nn.GELU(), nn.Linear(32, 1),
            ),
        })

    def state_dict(self) -> dict[str, Any]:
        return self.module.state_dict()

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self.module.load_state_dict(state, strict=True)

    def _encode_branch(self, name: str, values: Any, lengths: Any) -> Any:
        torch, _ = _torch()
        encoded, _ = self.module[f"{name}_gru"](values)
        indices = torch.clamp(lengths - 1, min=0)
        gathered = encoded[torch.arange(encoded.shape[0], device=encoded.device), indices]
        missing = self.module[f"{name}_missing"](
            torch.zeros(encoded.shape[0], dtype=torch.long, device=encoded.device)
        )
        return torch.where((lengths > 0).unsqueeze(1), gathered, missing)

    def __call__(self, batch: Mapping[str, Any]) -> dict[str, Any]:
        torch, _ = _torch()
        branches = [
            self._encode_branch(name, batch[f"{name}_tokens"], batch[f"{name}_lengths"])
            for name in ("center", "left", "right")
        ]
        scalar = self.module["scalar"](batch["scalars"])
        state = self.module["state"](torch.cat((*branches, scalar), dim=1))
        means, maxima, baseline_states = [], [], []
        for start, end, baseline in batch["groups"]:
            group = state[start:end]
            means.append(group.mean(dim=0))
            maxima.append(group.max(dim=0).values)
            baseline_states.append(state[baseline])
        mean = torch.stack(means)
        maximum = torch.stack(maxima)
        baseline_state = torch.stack(baseline_states)
        observation_index = batch["observation_index"]
        repeated_baseline = baseline_state[observation_index]
        repeated_mean = mean[observation_index]
        repeated_maximum = maximum[observation_index]
        pair = self.module["pair"](torch.cat((
            state, repeated_baseline, state - repeated_baseline,
            state * repeated_baseline, repeated_mean, repeated_maximum,
        ), dim=1))
        rank = self.module["rank"](pair).squeeze(1)
        abstain = self.module["abstain"](
            torch.cat((baseline_state, mean, maximum), dim=1)
        ).squeeze(1)
        selection = rank.clone()
        for observation, (_, _, baseline) in enumerate(batch["groups"]):
            selection[baseline] = abstain[observation]
        return {
            "selection_logits": selection,
            "quality_logits": self.module["quality"](pair).squeeze(1),
            "gain_logits": self.module["gain"](pair).squeeze(1),
            "left_quality_logits": self.module["left_quality"](pair).squeeze(1),
            "right_quality_logits": self.module["right_quality"](pair).squeeze(1),
            "abstention_logits": abstain,
        }


def _pad_temporal_sequences(rows: Sequence[Sequence[Sequence[float]]], cap: int) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.zeros((len(rows), max(1, cap), len(TEMPORAL_TOKEN_FEATURES)), dtype=np.float32)
    lengths = np.zeros(len(rows), dtype=np.int64)
    for index, sequence in enumerate(rows):
        if len(sequence) > cap:
            raise ValueError("JQ temporal token cap changed")
        lengths[index] = len(sequence)
        if sequence:
            matrix[index, :len(sequence)] = np.asarray(sequence, dtype=np.float32)
    return matrix, lengths


def tensorize_temporal_observations(
    observations: Sequence[TemporalTrainingObservation] | Sequence[Sequence[TemporalCandidateFeatureRow]],
    scaler: FeatureScaler, *, device: str,
) -> dict[str, Any]:
    """Flatten variable candidate pools into one deterministic padded model batch."""
    if not observations:
        raise ValueError("cannot tensorize empty JQ temporal observations")
    if isinstance(observations[0], TemporalTrainingObservation):
        pools = [row.rows for row in observations]
    else:
        pools = [tuple(row) for row in observations]
    flat = [candidate for pool in pools for candidate in pool]
    if not flat:
        raise ValueError("JQ temporal candidate pools are empty")
    groups, observation_index, offset = [], [], 0
    for pool in pools:
        if not 1 <= len(pool) <= 9:
            raise ValueError("JQ temporal candidate pool size changed")
        baselines = [index for index, row in enumerate(pool) if row.candidate.is_frozen_baseline]
        if len(baselines) != 1:
            raise ValueError("JQ temporal pool needs exactly one baseline")
        groups.append((offset, offset + len(pool), offset + baselines[0]))
        observation_index.extend([len(groups) - 1] * len(pool))
        offset += len(pool)
    center, center_lengths = _pad_temporal_sequences([row.center_tokens for row in flat], CENTER_TOKEN_CAP)
    left, left_lengths = _pad_temporal_sequences([row.left_tokens for row in flat], BOUNDARY_TOKEN_CAP)
    right, right_lengths = _pad_temporal_sequences([row.right_tokens for row in flat], BOUNDARY_TOKEN_CAP)
    torch, _ = _torch()
    return {
        "scalars": torch.as_tensor(
            scaler.transform([row.scalar_values for row in flat]), dtype=torch.float32, device=device,
        ),
        "center_tokens": torch.as_tensor(center, dtype=torch.float32, device=device),
        "left_tokens": torch.as_tensor(left, dtype=torch.float32, device=device),
        "right_tokens": torch.as_tensor(right, dtype=torch.float32, device=device),
        "center_lengths": torch.as_tensor(center_lengths, dtype=torch.long, device=device),
        "left_lengths": torch.as_tensor(left_lengths, dtype=torch.long, device=device),
        "right_lengths": torch.as_tensor(right_lengths, dtype=torch.long, device=device),
        "observation_index": torch.as_tensor(observation_index, dtype=torch.long, device=device),
        "groups": tuple(groups),
    }


def temporal_joint_quality_loss(
    outputs: Mapping[str, Any], observations: Sequence[TemporalTrainingObservation],
) -> tuple[Any, dict[str, float]]:
    """Fixed multi-task, listwise, abstention and relative-regret objective."""
    torch, _ = _torch()
    if not observations:
        raise ValueError("JQ temporal loss needs observations")
    counts: dict[str, int] = {}
    for row in observations:
        counts[row.video_id] = counts.get(row.video_id, 0) + 1
    raw_weights = np.asarray([1.0 / counts[row.video_id] for row in observations], dtype=np.float64)
    raw_weights /= raw_weights.sum()
    component_rows: dict[str, list[Any]] = {name: [] for name in TEMPORAL_LOSS_WEIGHTS}
    for observation_index, row in enumerate(observations):
        start = sum(len(previous.rows) for previous in observations[:observation_index])
        end = start + len(row.rows)
        selection = outputs["selection_logits"][start:end]
        quality_logits = outputs["quality_logits"][start:end]
        gain_logits = outputs["gain_logits"][start:end]
        left_logits = outputs["left_quality_logits"][start:end]
        right_logits = outputs["right_quality_logits"][start:end]
        quality = torch.as_tensor(row.quality_targets, dtype=selection.dtype, device=selection.device)
        gain = torch.as_tensor(row.gain_targets, dtype=selection.dtype, device=selection.device)
        left = torch.as_tensor(row.left_boundary_targets, dtype=selection.dtype, device=selection.device)
        right = torch.as_tensor(row.right_boundary_targets, dtype=selection.dtype, device=selection.device)
        bin_masks = (quality <= 0.3, (quality > 0.3) & (quality <= 0.7), quality > 0.7)
        bin_losses = [
            torch.nn.functional.smooth_l1_loss(
                torch.sigmoid(quality_logits[mask]), quality[mask], reduction="mean",
            )
            for mask in bin_masks if bool(mask.any())
        ]
        component_rows["quality"].append(torch.stack(bin_losses).mean())
        component_rows["gain"].append(torch.nn.functional.smooth_l1_loss(
            torch.tanh(gain_logits), gain, reduction="mean",
        ))
        component_rows["left_boundary"].append(torch.nn.functional.smooth_l1_loss(
            torch.sigmoid(left_logits), left, reduction="mean",
        ))
        component_rows["right_boundary"].append(torch.nn.functional.smooth_l1_loss(
            torch.sigmoid(right_logits), right, reduction="mean",
        ))
        target_distribution = torch.softmax(quality / 0.1, dim=0)
        component_rows["listwise"].append(
            -(target_distribution * torch.log_softmax(selection, dim=0)).sum()
        )
        baseline = row.baseline_index
        successors = [index for index in range(len(row.rows)) if index != baseline]
        if successors:
            best_successor = torch.max(quality[successors])
            abstain_target = torch.sigmoid((quality[baseline] - best_successor) / 0.05)
            bad = gain < 0
            bad[baseline] = False
            if bool(bad.any()):
                regret = (-gain[bad]) * torch.nn.functional.softplus(
                    selection[bad] - selection[baseline]
                )
                regret_loss = regret.mean()
            else:
                regret_loss = selection.sum() * 0.0
        else:
            abstain_target = torch.ones((), dtype=selection.dtype, device=selection.device)
            regret_loss = selection.sum() * 0.0
        component_rows["abstention"].append(torch.nn.functional.binary_cross_entropy_with_logits(
            outputs["abstention_logits"][observation_index], abstain_target,
        ))
        component_rows["relative_regret"].append(regret_loss)
    weights = torch.as_tensor(raw_weights, dtype=outputs["selection_logits"].dtype,
                              device=outputs["selection_logits"].device)
    components = {
        name: torch.stack(values).dot(weights) for name, values in component_rows.items()
    }
    total = sum(TEMPORAL_LOSS_WEIGHTS[name] * value for name, value in components.items())
    return total, {name: float(value.detach().cpu().item()) for name, value in components.items()}


def _fit_temporal_scaler(
    observations: Sequence[TemporalTrainingObservation], *,
    heldout_video_ids: Sequence[str] = (), heldout_group_ids: Sequence[str] = (),
) -> FeatureScaler:
    proxy = [QualityTrainingObservation(
        row.observation_id, row.video_id, row.group_id,
        tuple(candidate.scalar_values for candidate in row.rows), row.quality_targets,
        row.candidate_ids,
    ) for row in observations]
    return fit_feature_scaler(
        proxy, heldout_video_ids=heldout_video_ids, heldout_group_ids=heldout_group_ids,
    )


def _temporal_selected_miou(
    model: TemporalJointQualityHead, observations: Sequence[TemporalTrainingObservation],
    scaler: FeatureScaler, *, device: str,
) -> float:
    outputs = model(tensorize_temporal_observations(observations, scaler, device=device))
    by_video: dict[str, list[float]] = {}
    offset = 0
    scores = outputs["selection_logits"].detach().cpu().numpy()
    for row in observations:
        local = scores[offset:offset + len(row.rows)]
        selected = stable_argmax(local, row.candidate_ids)
        by_video.setdefault(row.video_id, []).append(row.quality_targets[selected])
        offset += len(row.rows)
    return float(np.mean([np.mean(values) for values in by_video.values()]))


def train_temporal_joint_quality_head(
    training: Sequence[TemporalTrainingObservation],
    validation: Sequence[TemporalTrainingObservation], *, seed: int = 20260911,
    device: str = "cpu", max_epochs: int = 100, patience: int = 10,
    learning_rate: float = 1e-3, weight_decay: float = 1e-4,
    heldout_video_ids: Sequence[str] = (), heldout_group_ids: Sequence[str] = (),
) -> tuple[TemporalJointQualityHead, FeatureScaler, dict[str, Any]]:
    if (
        not training or not validation or max_epochs != 100 or patience != 10
        or learning_rate != 1e-3 or weight_decay != 1e-4
    ):
        raise ValueError("JQ temporal training settings changed")
    train_videos, train_groups = {row.video_id for row in training}, {row.group_id for row in training}
    val_videos, val_groups = {row.video_id for row in validation}, {row.group_id for row in validation}
    if train_videos & val_videos or train_groups & val_groups:
        raise ValueError("JQ temporal inner train and validation identities overlap")
    if (train_videos | val_videos) & set(heldout_video_ids) or (
        (train_groups | val_groups) & set(heldout_group_ids)
    ):
        raise ValueError("JQ temporal outer-held-out identity entered training")
    scaler = _fit_temporal_scaler(
        training, heldout_video_ids=tuple(val_videos) + tuple(heldout_video_ids),
        heldout_group_ids=tuple(val_groups) + tuple(heldout_group_ids),
    )
    torch, _ = _torch()
    torch.manual_seed(seed)
    if hasattr(torch, "use_deterministic_algorithms"):
        torch.use_deterministic_algorithms(True)
    model = TemporalJointQualityHead(seed=seed)
    model.module.to(device)
    optimizer = torch.optim.AdamW(model.module.parameters(), lr=learning_rate, weight_decay=weight_decay)
    train_batch = tensorize_temporal_observations(training, scaler, device=device)
    best_state, best_score, best_epoch, stale = None, -math.inf, 0, 0
    history = []
    for epoch in range(1, max_epochs + 1):
        model.module.train()
        optimizer.zero_grad(set_to_none=True)
        loss, components = temporal_joint_quality_loss(model(train_batch), training)
        loss.backward()
        optimizer.step()
        model.module.eval()
        with torch.no_grad():
            score = _temporal_selected_miou(model, validation, scaler, device=device)
        history.append({
            "epoch": epoch, "train_loss": float(loss.item()),
            "validation_candidate_miou": score, "loss_components": components,
        })
        if score > best_score + 1e-12:
            best_state = {
                name: value.detach().cpu().clone() for name, value in model.state_dict().items()
            }
            best_score, best_epoch, stale = score, epoch, 0
        else:
            stale += 1
            if stale >= patience:
                break
    assert best_state is not None
    model.load_state_dict(best_state)
    model.module.to("cpu").eval()
    return model, scaler, {
        "best_epoch": best_epoch, "best_validation_candidate_miou": best_score,
        "epochs_ran": len(history), "history": history, "seed": seed, "device": device,
        "deterministic_algorithms": True, "optimizer": "AdamW",
        "learning_rate": learning_rate, "weight_decay": weight_decay,
        "max_epochs": max_epochs, "patience": patience,
        "training_weighting": "video_equal",
    }


def fit_temporal_joint_quality_fixed_epochs(
    training: Sequence[TemporalTrainingObservation], scaler: FeatureScaler, *, epochs: int,
    seed: int = 20260911, device: str = "cpu", learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
) -> TemporalJointQualityHead:
    if (
        not training or not 1 <= int(epochs) <= 100
        or learning_rate != 1e-3 or weight_decay != 1e-4
        or scaler.fitted_video_sha256 != _identity_sha([row.video_id for row in training])
        or scaler.fitted_group_sha256 != _identity_sha([row.group_id for row in training])
    ):
        raise ValueError("invalid fixed-epoch JQ temporal training request")
    torch, _ = _torch()
    torch.manual_seed(seed)
    if hasattr(torch, "use_deterministic_algorithms"):
        torch.use_deterministic_algorithms(True)
    model = TemporalJointQualityHead(seed=seed)
    model.module.to(device)
    optimizer = torch.optim.AdamW(model.module.parameters(), lr=learning_rate, weight_decay=weight_decay)
    batch = tensorize_temporal_observations(training, scaler, device=device)
    for _ in range(int(epochs)):
        optimizer.zero_grad(set_to_none=True)
        loss, _ = temporal_joint_quality_loss(model(batch), training)
        loss.backward()
        optimizer.step()
    model.module.to("cpu").eval()
    return model


def make_temporal_checkpoint(
    model: TemporalJointQualityHead, scaler: FeatureScaler, *,
    outer_train_video_sha256: str, outer_train_group_sha256: str,
    code_revision: str, training_metadata: Mapping[str, Any],
) -> dict[str, Any]:
    _reject_forbidden(training_metadata)
    if (
        not _valid_sha(outer_train_video_sha256)
        or not _valid_sha(outer_train_group_sha256)
        or not _valid_sha(code_revision, 40)
        or scaler.fitted_video_sha256 != outer_train_video_sha256
        or scaler.fitted_group_sha256 != outer_train_group_sha256
    ):
        raise ValueError("JQ temporal checkpoint training identity is invalid")
    schema = temporal_feature_schema()
    payload = {
        "format_version": 2,
        "state_dict": _serialize_state_dict(model.state_dict()),
        "scaler": asdict(scaler),
        "temporal_schema": schema,
        "temporal_schema_sha256": temporal_schema_sha256(schema),
        "architecture": TEMPORAL_ARCHITECTURE,
        "loss": {
            "weights": dict(TEMPORAL_LOSS_WEIGHTS),
            "quality": "three-bin-equal-SmoothL1(sigmoid(logit),iou)",
            "gain": "SmoothL1(tanh(logit),candidate-minus-baseline-iou)",
            "boundary": "SmoothL1(sigmoid(logit),exp(-absolute-error/gt-width))",
            "listwise": "ListNet(target_temperature=0.1)",
            "abstention": "BCEWithLogits(sigmoid((baseline-best-successor-iou)/0.05))",
            "relative_regret": "mean((-negative-gain)*softplus(candidate-minus-abstain-logit))",
        },
        "optimizer": {
            "name": "AdamW", "learning_rate": 0.001, "weight_decay": 0.0001,
            "max_epochs": 100, "patience": 10, "epoch_selection": "video_equal_candidate_miou",
        },
        "outer_train_video_sha256": outer_train_video_sha256,
        "outer_train_group_sha256": outer_train_group_sha256,
        "code_revision": code_revision,
        "training": dict(training_metadata),
    }
    payload["checkpoint_sha256"] = hashlib.sha256(canonical_json(payload)).hexdigest()
    return payload


def load_temporal_checkpoint(
    path: Path | str,
) -> tuple[TemporalJointQualityHead, FeatureScaler, dict[str, Any]]:
    source = Path(path)
    sidecar = source.with_suffix(source.suffix + ".sha256")
    if not source.is_file() or not sidecar.is_file():
        raise ValueError("JQ temporal checkpoint or SHA sidecar is missing")
    if hashlib.sha256(source.read_bytes()).hexdigest() != sidecar.read_text(encoding="ascii").strip():
        raise ValueError("JQ temporal checkpoint file SHA mismatch")
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or set(payload) != TEMPORAL_CHECKPOINT_KEYS:
        raise ValueError("JQ temporal checkpoint fields changed")
    _reject_forbidden(payload)
    claimed = payload.pop("checkpoint_sha256", None)
    if claimed != hashlib.sha256(canonical_json(payload)).hexdigest():
        raise ValueError("JQ temporal checkpoint payload SHA mismatch")
    payload["checkpoint_sha256"] = claimed
    if (
        payload.get("format_version") != 2
        or payload.get("temporal_schema") != temporal_feature_schema()
        or payload.get("temporal_schema_sha256") != temporal_schema_sha256()
        or payload.get("architecture") != TEMPORAL_ARCHITECTURE
        or not _valid_sha(payload.get("code_revision"), 40)
    ):
        raise ValueError("JQ temporal checkpoint schema or architecture mismatch")
    scaler_row = dict(payload.get("scaler", {}))
    scaler_row["mean"] = tuple(scaler_row.get("mean", ()))
    scaler_row["scale"] = tuple(scaler_row.get("scale", ()))
    scaler = FeatureScaler(**scaler_row)
    if (
        len(scaler.mean) != len(FEATURE_NAMES) or len(scaler.scale) != len(FEATURE_NAMES)
        or not np.isfinite(np.asarray(scaler.mean)).all()
        or not np.isfinite(np.asarray(scaler.scale)).all()
        or any(value <= 0 for value in scaler.scale)
        or payload.get("outer_train_video_sha256") != scaler.fitted_video_sha256
        or payload.get("outer_train_group_sha256") != scaler.fitted_group_sha256
    ):
        raise ValueError("JQ temporal checkpoint scaler mismatch")
    model = TemporalJointQualityHead()
    model.load_state_dict(_deserialize_state_dict(payload["state_dict"]))
    model.module.eval()
    return model, scaler, payload


def save_temporal_checkpoint(checkpoint: Mapping[str, Any], path: Path | str) -> Path:
    # The common writer already verifies the same canonical payload checksum and
    # writes a file checksum sidecar; temporal loading applies the stricter schema.
    candidate = dict(checkpoint)
    claimed = candidate.pop("checkpoint_sha256", None)
    if claimed != hashlib.sha256(canonical_json(candidate)).hexdigest():
        raise ValueError("JQ temporal checkpoint payload SHA mismatch")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(dict(checkpoint), ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                   allow_nan=False) + "\n",
        encoding="utf-8",
    )
    destination.with_suffix(destination.suffix + ".sha256").write_text(
        hashlib.sha256(destination.read_bytes()).hexdigest() + "\n", encoding="ascii",
    )
    return destination


def select_temporal_joint_quality(
    baseline_row: Mapping[str, Any], snapshot: SnapshotReader,
    query_embedding: Sequence[float], extent_candidates: Sequence[FrameCandidate], *,
    checkpoint_path: Path | str, rho: float, budget_bytes: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run structural JQ inference or bit-exactly return Frozen V2 on any fault."""
    fallback = copy.deepcopy(dict(baseline_row))
    debug: dict[str, Any] = {
        "fallback": True, "fallback_reason": None,
        "pre_span": copy.deepcopy(fallback.get("final_span")),
        "final_span": copy.deepcopy(fallback.get("final_span")), "candidates": [],
    }
    try:
        if not isinstance(snapshot, SnapshotReader):
            raise ValueError("JQ inference requires a verified SnapshotReader")
        verified = SnapshotReader(snapshot.root)
        _reject_forbidden({
            "selector_input": {key: value for key, value in baseline_row.items() if key != "final_span"}
        })
        before = snapshot_fingerprint(verified)
        duration = float(verified.manifest.video_meta["duration_s"])
        candidates = build_joint_candidates(
            baseline_row["final_span"], extent_candidates, duration_s=duration,
            upper_bound_s=float(verified.manifest.t_q),
        )
        if len(candidates) == 1:
            raise ValueError("no valid X1 extent candidates")
        rows = compute_temporal_jq_features(
            candidates, verified.read_frame_metadata(), query_embedding,
            duration_s=duration, rho=rho, budget_bytes=budget_bytes,
            upper_bound_s=float(verified.manifest.t_q),
        )
        model, scaler, checkpoint = load_temporal_checkpoint(checkpoint_path)
        torch, _ = _torch()
        with torch.no_grad():
            outputs = model(tensorize_temporal_observations([rows], scaler, device="cpu"))
        logits = outputs["selection_logits"].cpu().numpy()
        if before != snapshot_fingerprint(SnapshotReader(verified.root)):
            raise ValueError("snapshot changed during JQ temporal inference")
        ids = [row.candidate.candidate_id for row in rows]
        selected_index = stable_argmax(logits, ids)
        selected = rows[selected_index].candidate
        ranking = sorted(range(len(rows)), key=lambda index: (-float(logits[index]), ids[index]))
        debug["candidates"] = [{
            "candidate_id": row.candidate.candidate_id,
            "span": list(row.candidate.span),
            "scalar_features": dict(zip(FEATURE_NAMES, row.scalar_values)),
            "token_counts": {
                "center": len(row.center_tokens), "left": len(row.left_tokens),
                "right": len(row.right_tokens),
            },
            "selection_logit": float(logits[index]),
            "quality": float(torch.sigmoid(outputs["quality_logits"][index]).item()),
            "predicted_gain": float(torch.tanh(outputs["gain_logits"][index]).item()),
            "left_quality": float(torch.sigmoid(outputs["left_quality_logits"][index]).item()),
            "right_quality": float(torch.sigmoid(outputs["right_quality_logits"][index]).item()),
            "rank": ranking.index(index) + 1, "selected": index == selected_index,
        } for index, row in enumerate(rows)]
        debug["checkpoint_sha256"] = checkpoint["checkpoint_sha256"]
        debug["selected_candidate_id"] = selected.candidate_id
        debug["fallback"] = bool(selected.is_frozen_baseline)
        debug["fallback_reason"] = (
            "model_selected_frozen_baseline" if selected.is_frozen_baseline else None
        )
        if selected.is_frozen_baseline:
            return fallback, debug
        result = copy.deepcopy(fallback)
        result["final_span"] = [selected.start_s, selected.end_s]
        result["jq01_selected_candidate_id"] = selected.candidate_id
        debug["final_span"] = list(selected.span)
        return result, debug
    except Exception as exc:
        debug["fallback_reason"] = f"{type(exc).__name__}: {exc}"
        return fallback, debug
