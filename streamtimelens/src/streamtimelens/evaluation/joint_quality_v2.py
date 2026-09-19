"""Nested JQ-01 OOF on Frozen Visual V2's original development benchmark."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from streamtimelens.evaluation.bootstrap import (
    BootstrapConfig,
    PairedMetricObservation,
    paired_video_bootstrap,
)
from streamtimelens.evaluation.joint_quality_oof import JQOOFObservation
from streamtimelens.retrieval.extent_tokens import (
    ExtentObservation,
    ExtentPipeline,
    TemporalCellToken,
    fit_extent_pipeline,
    predict_extent_candidates,
)
from streamtimelens.retrieval.joint_quality import (
    FEATURE_NAMES,
    TemporalCandidateFeatureRow,
    TemporalTrainingObservation,
    _identity_sha,
    _fit_temporal_scaler,
    _torch,
    build_joint_candidates,
    canonical_json,
    compute_jq_features,
    compute_temporal_jq_features,
    fit_temporal_joint_quality_fixed_epochs,
    make_temporal_checkpoint,
    make_temporal_training_observation,
    stable_argmax,
    tensorize_temporal_observations,
    train_temporal_joint_quality_head,
)


BASELINE = "frozen_visual_v2"
METHOD = "JQ-01"
GAIN_RIDGE_ALPHA = 0.01
MINIMUM_PREDICTED_GAIN = 0.03
MINIMUM_BASELINE_IOU = 0.30
R2_STANDARD_MIOU = 0.400079


@dataclass(frozen=True)
class V2SnapshotFeatures:
    duration_s: float
    upper_bound_s: float
    budget_bytes: int
    frame_metadata: Mapping[str, Mapping[str, Any]]
    tokens: tuple[TemporalCellToken, ...]


@dataclass(frozen=True)
class JQV2Observation:
    observation_id: str
    video_id: str
    group_id: str
    query_id: str
    content_sha256: str
    snapshot_id: str
    budget_bytes: int
    rho: float
    source: str
    duration_bucket: str
    query_embedding: tuple[float, ...]
    baseline_span: tuple[float, float]
    gt_span: tuple[float, float]
    good_baseline: bool

    def __post_init__(self) -> None:
        numeric = (*self.query_embedding, *self.baseline_span, *self.gt_span, self.rho)
        if (
            not all((self.observation_id, self.video_id, self.group_id, self.query_id, self.snapshot_id))
            or len(self.content_sha256) != 64
            or any(character not in "0123456789abcdef" for character in self.content_sha256)
            or self.budget_bytes <= 0 or not 0 < self.rho <= 1
            or not self.source or not self.duration_bucket or not self.query_embedding
            or not np.isfinite(np.asarray(numeric, dtype=np.float64)).all()
            or self.baseline_span[0] < 0 or self.baseline_span[1] < self.baseline_span[0]
            or self.gt_span[0] < 0 or self.gt_span[1] <= self.gt_span[0]
            or not isinstance(self.good_baseline, bool)
        ):
            raise ValueError("invalid Frozen V2 JQ observation")


@dataclass(frozen=True)
class TemporalJQV2Observation:
    """OOF metadata plus an inference-only candidate-local representation."""

    source: JQV2Observation
    rows: tuple[TemporalCandidateFeatureRow, ...]
    duration_s: float

    def training_view(self) -> TemporalTrainingObservation:
        return make_temporal_training_observation(
            self.source.observation_id, self.source.video_id, self.source.group_id,
            self.rows, self.source.gt_span, duration_s=self.duration_s,
        )


def _extent_observation(
    row: JQV2Observation, snapshots: Mapping[str, V2SnapshotFeatures], *, with_gt: bool,
) -> ExtentObservation:
    snapshot = snapshots[row.snapshot_id]
    return ExtentObservation(
        row.observation_id,
        row.group_id,
        row.query_embedding,
        snapshot.tokens,
        snapshot.upper_bound_s,
        row.gt_span if with_gt else None,
    )


def _fit_extent(
    rows: Sequence[JQV2Observation], snapshots: Mapping[str, V2SnapshotFeatures],
) -> ExtentPipeline:
    return fit_extent_pipeline([
        _extent_observation(row, snapshots, with_gt=True) for row in rows
    ])


def _materialize(
    rows: Sequence[JQV2Observation], snapshots: Mapping[str, V2SnapshotFeatures],
    pipeline: ExtentPipeline,
) -> list[JQOOFObservation]:
    result = []
    for row in sorted(rows, key=lambda item: item.observation_id):
        snapshot = snapshots[row.snapshot_id]
        extent_candidates, _ = predict_extent_candidates(
            pipeline, _extent_observation(row, snapshots, with_gt=False),
        )
        candidates = build_joint_candidates(
            row.baseline_span,
            extent_candidates,
            duration_s=snapshot.duration_s,
            upper_bound_s=snapshot.upper_bound_s,
        )
        features = compute_jq_features(
            candidates,
            snapshot.frame_metadata,
            row.query_embedding,
            duration_s=snapshot.duration_s,
            rho=row.rho,
            budget_bytes=row.budget_bytes,
            upper_bound_s=snapshot.upper_bound_s,
        )
        result.append(JQOOFObservation(
            row.observation_id,
            row.video_id,
            row.group_id,
            row.query_id,
            row.content_sha256,
            row.budget_bytes,
            row.rho,
            row.source,
            row.duration_bucket,
            candidates,
            features,
            row.gt_span,
            row.good_baseline,
        ))
    return result


def _materialize_temporal(
    rows: Sequence[JQV2Observation], snapshots: Mapping[str, V2SnapshotFeatures],
    pipeline: ExtentPipeline,
) -> list[TemporalJQV2Observation]:
    result = []
    for row in sorted(rows, key=lambda item: item.observation_id):
        snapshot = snapshots[row.snapshot_id]
        extent_candidates, _ = predict_extent_candidates(
            pipeline, _extent_observation(row, snapshots, with_gt=False),
        )
        candidates = build_joint_candidates(
            row.baseline_span, extent_candidates, duration_s=snapshot.duration_s,
            upper_bound_s=snapshot.upper_bound_s,
        )
        features = compute_temporal_jq_features(
            candidates, snapshot.frame_metadata, row.query_embedding,
            duration_s=snapshot.duration_s, rho=row.rho, budget_bytes=row.budget_bytes,
            upper_bound_s=snapshot.upper_bound_s,
        )
        result.append(TemporalJQV2Observation(row, features, snapshot.duration_s))
    return result


def _crossfit_extent(
    rows: Sequence[JQV2Observation], snapshots: Mapping[str, V2SnapshotFeatures],
    fold_by_group: Mapping[str, int],
) -> tuple[list[JQOOFObservation], list[dict[str, Any]]]:
    folds = sorted({fold_by_group[row.group_id] for row in rows})
    if len(folds) < 2:
        raise ValueError("X1 cross-fit requires at least two training folds")
    materialized, audits = [], []
    for heldout_fold in folds:
        train = [row for row in rows if fold_by_group[row.group_id] != heldout_fold]
        heldout = [row for row in rows if fold_by_group[row.group_id] == heldout_fold]
        pipeline = _fit_extent(train, snapshots)
        materialized.extend(_materialize(heldout, snapshots, pipeline))
        payload = asdict(pipeline)
        audits.append({
            "heldout_fold": heldout_fold,
            "training_groups_sha256": _identity_sha([row.group_id for row in train]),
            "heldout_groups_sha256": _identity_sha([row.group_id for row in heldout]),
            "pipeline_sha256": hashlib.sha256(canonical_json(payload)).hexdigest(),
        })
    if len(materialized) != len(rows) or len({row.observation_id for row in materialized}) != len(rows):
        raise RuntimeError("X1 cross-fit did not materialize every observation exactly once")
    return materialized, audits


def _crossfit_extent_temporal(
    rows: Sequence[JQV2Observation], snapshots: Mapping[str, V2SnapshotFeatures],
    fold_by_group: Mapping[str, int],
) -> tuple[list[TemporalJQV2Observation], list[dict[str, Any]]]:
    folds = sorted({fold_by_group[row.group_id] for row in rows})
    if len(folds) < 2:
        raise ValueError("X1 temporal cross-fit requires at least two training folds")
    materialized, audits = [], []
    for heldout_fold in folds:
        train = [row for row in rows if fold_by_group[row.group_id] != heldout_fold]
        heldout = [row for row in rows if fold_by_group[row.group_id] == heldout_fold]
        pipeline = _fit_extent(train, snapshots)
        materialized.extend(_materialize_temporal(heldout, snapshots, pipeline))
        payload = asdict(pipeline)
        audits.append({
            "heldout_fold": heldout_fold,
            "training_groups_sha256": _identity_sha([row.group_id for row in train]),
            "heldout_groups_sha256": _identity_sha([row.group_id for row in heldout]),
            "pipeline_sha256": hashlib.sha256(canonical_json(payload)).hexdigest(),
        })
    if (
        len(materialized) != len(rows)
        or len({row.source.observation_id for row in materialized}) != len(rows)
    ):
        raise RuntimeError("X1 temporal cross-fit did not materialize every observation exactly once")
    return materialized, audits


@dataclass(frozen=True)
class PairwiseGainRidge:
    feature_mean: tuple[float, ...]
    feature_scale: tuple[float, ...]
    coefficients: tuple[float, ...]
    intercept: float
    alpha: float = GAIN_RIDGE_ALPHA

    def predict(self, features: Sequence[float]) -> float:
        values = np.asarray(features, dtype=np.float64)
        mean = np.asarray(self.feature_mean, dtype=np.float64)
        scale = np.asarray(self.feature_scale, dtype=np.float64)
        if values.shape != mean.shape or values.shape != (2 * len(FEATURE_NAMES),):
            raise ValueError("pairwise JQ gain feature shape changed")
        return float(((values - mean) / scale).dot(self.coefficients) + self.intercept)


def fit_pairwise_gain_ridge(rows: Sequence[JQOOFObservation]) -> PairwiseGainRidge:
    """Regress candidate-minus-baseline IoU with video-equal training weights."""
    if not rows:
        raise ValueError("pairwise JQ gain training rows are empty")
    observations_per_video: dict[str, int] = defaultdict(int)
    for row in rows:
        observations_per_video[row.video_id] += 1
    features, targets, weights = [], [], []
    for row in sorted(rows, key=lambda item: item.observation_id):
        view = row.training_view()
        baseline_index = next(
            index for index, candidate in enumerate(row.candidates)
            if candidate.is_frozen_baseline
        )
        baseline_features = np.asarray(view.features[baseline_index], dtype=np.float64)
        successors = [index for index in range(len(row.candidates)) if index != baseline_index]
        if not successors:
            continue
        weight = 1.0 / (observations_per_video[row.video_id] * len(successors))
        for index in successors:
            candidate_features = np.asarray(view.features[index], dtype=np.float64)
            features.append(np.concatenate((candidate_features, baseline_features)))
            targets.append(float(view.targets[index] - view.targets[baseline_index]))
            weights.append(weight)
    matrix = np.asarray(features, dtype=np.float64)
    target = np.asarray(targets, dtype=np.float64)
    sample_weight = np.asarray(weights, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] < 2 or matrix.shape[1] != 2 * len(FEATURE_NAMES):
        raise ValueError("pairwise JQ gain training matrix is invalid")
    mean = np.average(matrix, axis=0, weights=sample_weight)
    scale = np.sqrt(np.average((matrix - mean) ** 2, axis=0, weights=sample_weight))
    scale[scale < 1e-8] = 1.0
    design = np.column_stack((np.ones(matrix.shape[0]), (matrix - mean) / scale))
    root_weight = np.sqrt(sample_weight)
    weighted_design = design * root_weight[:, None]
    penalty = np.eye(design.shape[1], dtype=np.float64) * GAIN_RIDGE_ALPHA
    penalty[0, 0] = 0.0
    coefficients = np.linalg.solve(
        weighted_design.T @ weighted_design + penalty,
        weighted_design.T @ (target * root_weight),
    )
    return PairwiseGainRidge(
        tuple(map(float, mean)), tuple(map(float, scale)),
        tuple(map(float, coefficients[1:])), float(coefficients[0]),
    )


def _select_pairwise_gain(
    row: JQOOFObservation, model: PairwiseGainRidge,
) -> tuple[int, list[float]]:
    view = row.training_view()
    baseline_index = next(
        index for index, candidate in enumerate(row.candidates)
        if candidate.is_frozen_baseline
    )
    baseline_features = view.features[baseline_index]
    predictions = [0.0] * len(row.candidates)
    successors = [index for index in range(len(row.candidates)) if index != baseline_index]
    for index in successors:
        predictions[index] = model.predict((*view.features[index], *baseline_features))
    best = min(successors, key=lambda index: (-predictions[index], view.candidate_ids[index]))
    baseline_iou_index = FEATURE_NAMES.index("iou_with_baseline")
    if (
        predictions[best] <= MINIMUM_PREDICTED_GAIN
        or view.features[best][baseline_iou_index] < MINIMUM_BASELINE_IOU
    ):
        return baseline_index, predictions
    return best, predictions


def run_v2_nested_oof(
    observations: Sequence[JQV2Observation], snapshots: Mapping[str, V2SnapshotFeatures], *,
    fold_by_group: Mapping[str, int], folds: int, seed: int, code_revision: str,
    device: str = "cpu",
) -> tuple[list[dict[str, Any]], dict[int, dict[str, Any]], dict[str, Any]]:
    """Nested X1 plus the fixed structural quality/gain/abstention selector."""
    if not observations or set(fold_by_group.values()) != set(range(folds)):
        raise ValueError("Frozen V2 JQ OOF needs complete fixed folds")
    if any(row.snapshot_id not in snapshots for row in observations):
        raise ValueError("Frozen V2 JQ observation is missing a verified snapshot")
    video_folds: dict[str, set[int]] = defaultdict(set)
    for row in observations:
        video_folds[row.video_id].add(int(fold_by_group[row.group_id]))
    if any(len(selected) != 1 for selected in video_folds.values()):
        raise ValueError("Frozen V2 video crosses outer folds")

    outputs: list[dict[str, Any]] = []
    checkpoints: dict[int, dict[str, Any]] = {}
    fold_audit: dict[str, Any] = {}
    torch, _ = _torch()
    for outer in range(folds):
        heldout = [row for row in observations if fold_by_group[row.group_id] == outer]
        outer_train = [row for row in observations if fold_by_group[row.group_id] != outer]
        inner_validation_fold = (outer + 1) % folds
        heldout_groups = sorted({row.group_id for row in heldout})
        heldout_videos = sorted({row.video_id for row in heldout})
        outer_train_jq, outer_x1_audit = _crossfit_extent_temporal(
            outer_train, snapshots, fold_by_group,
        )
        inner_train_raw = [
            row for row in outer_train if fold_by_group[row.group_id] != inner_validation_fold
        ]
        inner_validation_raw = [
            row for row in outer_train if fold_by_group[row.group_id] == inner_validation_fold
        ]
        inner_train_materialized, inner_x1_audit = _crossfit_extent_temporal(
            inner_train_raw, snapshots, fold_by_group,
        )
        inner_validation_pipeline = _fit_extent(inner_train_raw, snapshots)
        inner_validation_materialized = _materialize_temporal(
            inner_validation_raw, snapshots, inner_validation_pipeline,
        )
        inner_train = [row.training_view() for row in inner_train_materialized]
        inner_validation = [row.training_view() for row in inner_validation_materialized]
        _, _, selection = train_temporal_joint_quality_head(
            inner_train, inner_validation, seed=seed + outer, device=device,
            heldout_video_ids=heldout_videos, heldout_group_ids=heldout_groups,
        )
        outer_views = [row.training_view() for row in outer_train_jq]
        scaler = _fit_temporal_scaler(
            outer_views, heldout_video_ids=heldout_videos,
            heldout_group_ids=heldout_groups,
        )
        model = fit_temporal_joint_quality_fixed_epochs(
            outer_views, scaler, epochs=int(selection["best_epoch"]),
            seed=seed + outer, device=device,
        )
        outer_pipeline = _fit_extent(outer_train, snapshots)
        heldout_jq = _materialize_temporal(heldout, snapshots, outer_pipeline)
        train_videos = sorted({row.video_id for row in outer_train})
        train_groups = sorted({row.group_id for row in outer_train})
        checkpoint = make_temporal_checkpoint(
            model, scaler, outer_train_video_sha256=_identity_sha(train_videos),
            outer_train_group_sha256=_identity_sha(train_groups), code_revision=code_revision,
            training_metadata={**selection, "outer_fold": outer},
        )
        checkpoints[outer] = checkpoint
        pipeline_payload = asdict(outer_pipeline)
        inner_validation_pipeline_payload = asdict(inner_validation_pipeline)
        inner_train_groups = sorted({row.group_id for row in outer_train
                                     if fold_by_group[row.group_id] != inner_validation_fold})
        inner_validation_groups = sorted({row.group_id for row in outer_train
                                          if fold_by_group[row.group_id] == inner_validation_fold})
        fold_audit[str(outer)] = {
            "outer_train_groups": train_groups,
            "heldout_groups": heldout_groups,
            "heldout_videos": heldout_videos,
            "inner_train_groups": inner_train_groups,
            "inner_validation_groups": inner_validation_groups,
            "inner_validation_fold": inner_validation_fold,
            "outer_train_x1_crossfit": outer_x1_audit,
            "inner_train_x1_crossfit": inner_x1_audit,
            "inner_validation_x1_pipeline": inner_validation_pipeline_payload,
            "inner_validation_x1_pipeline_sha256": hashlib.sha256(
                canonical_json(inner_validation_pipeline_payload)
            ).hexdigest(),
            "heldout_x1_pipeline": pipeline_payload,
            "heldout_x1_pipeline_sha256": hashlib.sha256(
                canonical_json(pipeline_payload)
            ).hexdigest(),
            "selector_checkpoint_sha256": checkpoint["checkpoint_sha256"],
            "best_epoch": selection["best_epoch"],
        }
        inference_batch = tensorize_temporal_observations(
            [row.training_view() for row in heldout_jq], scaler, device="cpu",
        )
        with torch.no_grad():
            predictions = model(inference_batch)
        offset = 0
        for row in heldout_jq:
            source = row.source
            view = row.training_view()
            count = len(view.rows)
            selection_logits = predictions["selection_logits"][offset:offset + count].cpu().numpy()
            quality_predictions = torch.sigmoid(
                predictions["quality_logits"][offset:offset + count]
            ).cpu().numpy()
            gain_predictions = torch.tanh(
                predictions["gain_logits"][offset:offset + count]
            ).cpu().numpy()
            left_predictions = torch.sigmoid(
                predictions["left_quality_logits"][offset:offset + count]
            ).cpu().numpy()
            right_predictions = torch.sigmoid(
                predictions["right_quality_logits"][offset:offset + count]
            ).cpu().numpy()
            selected = stable_argmax(selection_logits, view.candidate_ids)
            baseline_index = view.baseline_index
            ranking = sorted(
                range(count),
                key=lambda index: (-float(selection_logits[index]), view.candidate_ids[index]),
            )
            outputs.append({
                    "training_or_evaluation_only": True,
                    "observation_id": source.observation_id,
                    "video_id": source.video_id,
                    "group_id": source.group_id,
                    "query_id": source.query_id,
                    "content_sha256": source.content_sha256,
                    "outer_fold": outer,
                    "budget_bytes": source.budget_bytes,
                    "rho": source.rho,
                    "source": source.source,
                    "duration_bucket": source.duration_bucket,
                    "good_baseline": source.good_baseline,
                    "baseline_candidate_id": view.rows[baseline_index].candidate.candidate_id,
                    "selected_candidate_id": view.rows[selected].candidate.candidate_id,
                    "pre_span": list(view.rows[baseline_index].candidate.span),
                    "final_span": list(view.rows[selected].candidate.span),
                    "fallback_reason": (
                        "model_selected_frozen_baseline" if selected == baseline_index else None
                    ),
                    "baseline_iou": view.quality_targets[baseline_index],
                    "jq_iou": view.quality_targets[selected],
                    "union_oracle_iou": max(view.quality_targets),
                    "candidate_rows": [
                        {
                            "candidate_id": candidate.candidate.candidate_id,
                            "span": list(candidate.candidate.span),
                            "features": dict(zip(FEATURE_NAMES, candidate.scalar_values)),
                            "temporal_tokens": {
                                "center": [list(token) for token in candidate.center_tokens],
                                "left": [list(token) for token in candidate.left_tokens],
                                "right": [list(token) for token in candidate.right_tokens],
                            },
                            "target": view.quality_targets[index],
                            "gain_target": view.gain_targets[index],
                            "left_boundary_target": view.left_boundary_targets[index],
                            "right_boundary_target": view.right_boundary_targets[index],
                            "selection_logit": float(selection_logits[index]),
                            "predicted_quality": float(quality_predictions[index]),
                            "predicted_gain": float(gain_predictions[index]),
                            "predicted_left_quality": float(left_predictions[index]),
                            "predicted_right_quality": float(right_predictions[index]),
                            "rank": ranking.index(index) + 1,
                            "selected": index == selected,
                        }
                        for index, candidate in enumerate(view.rows)
                    ],
            })
            offset += count
    if len(outputs) != len(observations) or len({row["observation_id"] for row in outputs}) != len(outputs):
        raise RuntimeError("Frozen V2 JQ OOF did not predict every observation exactly once")
    return sorted(outputs, key=lambda row: row["observation_id"]), checkpoints, {
        "schema_version": 1,
        "fold_by_group": dict(fold_by_group),
        "folds": fold_audit,
    }


def _video_delta(rows: Sequence[Mapping[str, Any]], left: str, right: str) -> float:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[str(row["video_id"])].append(float(row[left]) - float(row[right]))
    if not grouped:
        raise ValueError("empty Frozen V2 JQ metric slice")
    return float(np.mean([np.mean(values) for values in grouped.values()]))


def _bootstrap(
    rows: Sequence[Mapping[str, Any]], left: str, right: str, *, seed: int, resamples: int,
) -> dict[str, Any]:
    observations = []
    for row in rows:
        common = {
            "budget_bytes": int(row["budget_bytes"]),
            "video_id": str(row["video_id"]),
            "sample_id": str(row["observation_id"]),
        }
        observations.extend((
            PairedMetricObservation(METHOD, value=float(row[left]), **common),
            PairedMetricObservation(BASELINE, value=float(row[right]), **common),
        ))
    return paired_video_bootstrap(
        observations,
        method_a=METHOD,
        method_b=BASELINE,
        config=BootstrapConfig(seed=seed, resamples=resamples),
    )


def summarize_v2_nested_oof(
    rows: Sequence[Mapping[str, Any]], *, snapshots_unchanged: bool,
    byte_audit_complete: bool, fold_coverage_complete: bool,
    baseline_bit_exact: bool, provenance_complete: bool,
    seed: int = 20260911, resamples: int = 2000,
) -> dict[str, Any]:
    """Require a paired gain on the same Frozen V2 development observations."""
    rows = [dict(row) for row in rows]
    if not rows or len({row["observation_id"] for row in rows}) != len(rows):
        raise ValueError("Frozen V2 JQ summary rows are empty or duplicated")
    for row in rows:
        row["baseline_r07"] = float(row["baseline_iou"] >= 0.7)
        row["jq_r07"] = float(row["jq_iou"] >= 0.7)
    standard = {
        "baseline_miou": float(np.mean([float(row["baseline_iou"]) for row in rows])),
        "jq_miou": float(np.mean([float(row["jq_iou"]) for row in rows])),
        "delta_miou": float(np.mean([
            float(row["jq_iou"]) - float(row["baseline_iou"]) for row in rows
        ])),
        "baseline_r07": float(np.mean([row["baseline_r07"] for row in rows])),
        "jq_r07": float(np.mean([row["jq_r07"] for row in rows])),
        "delta_r07": float(np.mean([
            row["jq_r07"] - row["baseline_r07"] for row in rows
        ])),
        "observation_count": len(rows),
    }
    miou = _bootstrap(rows, "jq_iou", "baseline_iou", seed=seed, resamples=resamples)
    r07 = _bootstrap(rows, "jq_r07", "baseline_r07", seed=seed + 1, resamples=resamples)
    rho_deltas = {
        f"{rho:.2f}": _video_delta(
            [row for row in rows if abs(float(row["rho"]) - rho) < 1e-8],
            "jq_iou",
            "baseline_iou",
        )
        for rho in sorted({float(row["rho"]) for row in rows})
    }
    good = [row for row in rows if bool(row["good_baseline"])]
    good_delta = _video_delta(good, "jq_iou", "baseline_iou") if good else None
    recoverable = _video_delta(rows, "union_oracle_iou", "baseline_iou")
    transmitted = _video_delta(rows, "jq_iou", "baseline_iou")
    recovery_ratio = transmitted / recoverable if recoverable > 0 else 0.0
    checks = {
        "v2_standard_miou_above_frozen_and_r2": (
            standard["jq_miou"] > standard["baseline_miou"]
            and standard["jq_miou"] > R2_STANDARD_MIOU
        ),
        "paired_video_equal_miou_point_positive": miou["delta_a_minus_b"] > 0,
        "v2_standard_r07_nonnegative": standard["delta_r07"] >= 0,
        "rho_slices_above_floor": all(value >= -0.02 for value in rho_deltas.values()),
        "good_baseline_above_floor": good_delta is not None and good_delta >= -0.01,
        "recoverable_delta_transmission_at_least_10pct": recovery_ratio >= 0.10,
        "paired_video_equal_miou_ci_low_positive": miou["ci_low"] > 0,
        "snapshot_unchanged": bool(snapshots_unchanged),
        "byte_audit_complete": bool(byte_audit_complete),
        "fold_coverage_complete": bool(fold_coverage_complete),
        "baseline_bit_exact": bool(baseline_bit_exact),
        "provenance_complete": bool(provenance_complete),
    }
    diagnostics = {
        "paired_video_equal_miou_ci_low_positive": miou["ci_low"] > 0,
        "paired_video_equal_r07_nonnegative": r07["delta_a_minus_b"] >= 0,
        "full_jq_recoverable_delta_transmission_at_least_20pct": recovery_ratio >= 0.2,
    }
    passed = all(checks.values())
    return {
        "schema_version": 1,
        "stage": "jq01_v2_original_dev_nested_oof",
        "passed": passed,
        "decision": (
            "jq01_structural_revision_passed"
            if passed else "close_jq01_after_structural_revision"
        ),
        "gate_checks": checks,
        "diagnostic_checks": diagnostics,
        "v2_standard_metrics": standard,
        "miou_bootstrap": miou,
        "r07_bootstrap": r07,
        "rho_deltas": rho_deltas,
        "good_baseline_delta": good_delta,
        "union_oracle_recoverable_delta": recoverable,
        "final_transmitted_delta": transmitted,
        "recoverable_delta_transmission_ratio": recovery_ratio,
    }
