"""Group-isolated OOF execution and paired JQ-01 development gates."""

from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from streamtimelens.evaluation.bootstrap import (
    BootstrapConfig,
    PairedMetricObservation,
    paired_video_bootstrap,
)
from streamtimelens.retrieval.extent_tokens import temporal_iou
from streamtimelens.retrieval.joint_quality import (
    FEATURE_NAMES,
    CandidateFeatureRow,
    FeatureScaler,
    JointCandidate,
    JointQualityHead,
    QualityTrainingObservation,
    _identity_sha,
    _torch,
    fit_joint_quality_fixed_epochs,
    fit_feature_scaler,
    make_checkpoint,
    stable_argmax,
    train_joint_quality_head,
)


BASELINE = "frozen_visual_v2"
METHOD = "JQ-01"


@dataclass(frozen=True)
class JQOOFObservation:
    observation_id: str
    video_id: str
    group_id: str
    query_id: str
    content_sha256: str
    budget_bytes: int
    rho: float
    source: str
    duration_bucket: str
    candidates: tuple[JointCandidate, ...]
    feature_rows: tuple[CandidateFeatureRow, ...]
    gt_span: tuple[float, float]
    good_baseline: bool

    def __post_init__(self) -> None:
        gt_start, gt_end = map(float, self.gt_span)
        if (
            not self.observation_id or not self.video_id or not self.group_id or not self.query_id
            or len(self.content_sha256) != 64
            or any(character not in "0123456789abcdef" for character in self.content_sha256)
            or self.budget_bytes not in (1024 * 1024, 8 * 1024 * 1024)
            or not math.isfinite(self.rho) or not 0 < self.rho <= 1
            or not self.source or not self.duration_bucket
            or not 1 <= len(self.candidates) <= 9
            or tuple(row.candidate for row in self.feature_rows) != self.candidates
            or sum(candidate.is_frozen_baseline for candidate in self.candidates) != 1
            or not math.isfinite(gt_start) or not math.isfinite(gt_end)
            or gt_start < 0 or gt_end <= gt_start
        ):
            raise ValueError("invalid JQ OOF observation")

    def training_view(self) -> QualityTrainingObservation:
        return QualityTrainingObservation(
            self.observation_id, self.video_id, self.group_id,
            tuple(row.values for row in self.feature_rows),
            tuple(temporal_iou(candidate.span, self.gt_span) for candidate in self.candidates),
            tuple(candidate.candidate_id for candidate in self.candidates),
        )


def deterministic_group_folds(group_ids: Sequence[str], *, folds: int = 4, seed: int = 20260911) -> dict[str, int]:
    """Assign complete groups deterministically with balanced fold counts."""
    groups = sorted(set(map(str, group_ids)))
    if folds < 3 or len(groups) < folds:
        raise ValueError("JQ OOF requires at least three non-empty group folds")
    ordered = sorted(
        groups,
        key=lambda value: hashlib.sha256(f"{seed}:{value}".encode("utf-8")).hexdigest(),
    )
    return {group: index % folds for index, group in enumerate(ordered)}


def validate_fold_isolation(
    observations: Sequence[JQOOFObservation], fold_by_group: Mapping[str, int], *, folds: int,
) -> None:
    if not observations or folds < 3:
        raise ValueError("invalid JQ OOF fold request")
    video_folds: dict[str, set[int]] = defaultdict(set)
    group_folds: dict[str, set[int]] = defaultdict(set)
    for row in observations:
        if row.group_id not in fold_by_group:
            raise ValueError("JQ OOF group has no fold")
        fold = int(fold_by_group[row.group_id])
        if not 0 <= fold < folds:
            raise ValueError("JQ OOF fold index is invalid")
        video_folds[row.video_id].add(fold)
        group_folds[row.group_id].add(fold)
    if any(len(values) != 1 for values in video_folds.values()) or any(
        len(values) != 1 for values in group_folds.values()
    ):
        raise ValueError("JQ OOF video/group crosses outer folds")
    if set(fold_by_group.values()) != set(range(folds)):
        raise ValueError("JQ OOF contains an empty fold")


def _fit_outer_model(
    rows: Sequence[QualityTrainingObservation], scaler: FeatureScaler, *, epochs: int, seed: int,
) -> JointQualityHead:
    return fit_joint_quality_fixed_epochs(rows, scaler, epochs=epochs, seed=seed)


def run_joint_quality_oof(
    observations: Sequence[JQOOFObservation], *, folds: int = 4, seed: int = 20260911,
    code_revision: str, fold_by_group: Mapping[str, int] | None = None,
) -> tuple[list[dict[str, Any]], dict[int, dict[str, Any]], dict[str, Any]]:
    """Generate each held-out observation exactly once with nested epoch selection."""
    if not code_revision:
        raise ValueError("JQ OOF requires a code revision")
    assignment = dict(fold_by_group or deterministic_group_folds(
        [row.group_id for row in observations], folds=folds, seed=seed,
    ))
    validate_fold_isolation(observations, assignment, folds=folds)
    outputs: list[dict[str, Any]] = []
    checkpoints: dict[int, dict[str, Any]] = {}
    fold_audit: dict[str, Any] = {}
    torch, _ = _torch()
    for outer in range(folds):
        heldout = [row for row in observations if assignment[row.group_id] == outer]
        outer_train = [row for row in observations if assignment[row.group_id] != outer]
        inner_validation_fold = (outer + 1) % folds
        inner_validation = [
            row.training_view() for row in outer_train
            if assignment[row.group_id] == inner_validation_fold
        ]
        inner_train = [
            row.training_view() for row in outer_train
            if assignment[row.group_id] != inner_validation_fold
        ]
        heldout_videos = sorted({row.video_id for row in heldout})
        heldout_groups = sorted({row.group_id for row in heldout})
        _, _, selection = train_joint_quality_head(
            inner_train, inner_validation, seed=seed + outer, max_epochs=100, patience=10,
            heldout_video_ids=heldout_videos, heldout_group_ids=heldout_groups,
        )
        outer_views = [row.training_view() for row in outer_train]
        scaler = fit_feature_scaler(
            outer_views, heldout_video_ids=heldout_videos, heldout_group_ids=heldout_groups,
        )
        model = _fit_outer_model(
            outer_views, scaler, epochs=int(selection["best_epoch"]), seed=seed + outer,
        )
        train_videos = sorted({row.video_id for row in outer_train})
        train_groups = sorted({row.group_id for row in outer_train})
        inner_train_videos = sorted({
            row.video_id for row in outer_train
            if assignment[row.group_id] != inner_validation_fold
        })
        inner_validation_videos = sorted({
            row.video_id for row in outer_train
            if assignment[row.group_id] == inner_validation_fold
        })
        inner_train_groups = sorted({
            row.group_id for row in outer_train
            if assignment[row.group_id] != inner_validation_fold
        })
        inner_validation_groups = sorted({
            row.group_id for row in outer_train
            if assignment[row.group_id] == inner_validation_fold
        })
        checkpoint = make_checkpoint(
            model, scaler, outer_train_video_sha256=_identity_sha(train_videos),
            outer_train_group_sha256=_identity_sha(train_groups), code_revision=code_revision,
            training_metadata={**selection, "outer_fold": outer},
        )
        checkpoints[outer] = checkpoint
        fold_audit[str(outer)] = {
            "inner_train_videos": inner_train_videos,
            "inner_validation_videos": inner_validation_videos,
            "inner_train_groups": inner_train_groups,
            "inner_validation_groups": inner_validation_groups,
            "outer_train_videos": train_videos, "heldout_videos": heldout_videos,
            "outer_train_groups": train_groups, "heldout_groups": heldout_groups,
            "inner_validation_fold": inner_validation_fold,
            "inner_train_video_sha256": _identity_sha(inner_train_videos),
            "inner_validation_video_sha256": _identity_sha(inner_validation_videos),
            "heldout_video_sha256": _identity_sha(heldout_videos),
            "inner_train_group_sha256": _identity_sha(inner_train_groups),
            "inner_validation_group_sha256": _identity_sha(inner_validation_groups),
            "heldout_group_sha256": _identity_sha(heldout_groups),
            "outer_train_video_sha256": checkpoint["outer_train_video_sha256"],
            "outer_train_group_sha256": checkpoint["outer_train_group_sha256"],
            "best_epoch": selection["best_epoch"],
        }
        with torch.no_grad():
            for row in sorted(heldout, key=lambda item: item.observation_id):
                view = row.training_view()
                logits = model(torch.as_tensor(scaler.transform(view.features), dtype=torch.float32)).numpy()
                selected = stable_argmax(logits, view.candidate_ids)
                baseline_index = next(
                    index for index, candidate in enumerate(row.candidates)
                    if candidate.is_frozen_baseline
                )
                targets = list(view.targets)
                outputs.append({
                    "training_or_evaluation_only": True,
                    "observation_id": row.observation_id, "video_id": row.video_id,
                    "group_id": row.group_id, "query_id": row.query_id,
                    "content_sha256": row.content_sha256, "outer_fold": outer,
                    "budget_bytes": row.budget_bytes, "rho": row.rho,
                    "source": row.source, "duration_bucket": row.duration_bucket,
                    "good_baseline": row.good_baseline,
                    "baseline_candidate_id": row.candidates[baseline_index].candidate_id,
                    "selected_candidate_id": row.candidates[selected].candidate_id,
                    "pre_span": list(row.candidates[baseline_index].span),
                    "final_span": list(row.candidates[selected].span),
                    "fallback_reason": (
                        "model_selected_frozen_baseline"
                        if selected == baseline_index else None
                    ),
                    "baseline_iou": targets[baseline_index], "jq_iou": targets[selected],
                    "union_oracle_iou": max(targets),
                    "candidate_rows": [
                        {
                            "candidate_id": candidate.candidate_id, "span": list(candidate.span),
                            "features": dict(zip(FEATURE_NAMES, view.features[index])),
                            "target": targets[index], "logit": float(logits[index]),
                            "rank": sorted(
                                range(len(logits)),
                                key=lambda item: (-float(logits[item]), view.candidate_ids[item]),
                            ).index(index) + 1,
                            "selected": index == selected,
                        }
                        for index, candidate in enumerate(row.candidates)
                    ],
                })
    expected = {row.observation_id for row in observations}
    if len(outputs) != len(observations) or {row["observation_id"] for row in outputs} != expected:
        raise RuntimeError("JQ OOF did not predict every observation exactly once")
    return outputs, checkpoints, {
        "fold_by_group": assignment, "folds": fold_audit,
        "fold_assignment_sha256": hashlib.sha256(
            json_bytes(sorted(assignment.items())),
        ).hexdigest(),
    }


def json_bytes(value: Any) -> bytes:
    import json
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _video_equal_delta(rows: Sequence[Mapping[str, Any]], trial: str, baseline: str) -> float:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[str(row["video_id"])].append(float(row[trial]) - float(row[baseline]))
    if not grouped:
        raise ValueError("empty JQ gate slice")
    return float(np.mean([np.mean(values) for values in grouped.values()]))


def _bootstrap(
    rows: Sequence[Mapping[str, Any]], field: str, *, baseline_field: str = "baseline_iou",
    seed: int, resamples: int,
) -> dict[str, Any]:
    observations = []
    for row in rows:
        identity = str(row["observation_id"])
        common = {
            "budget_bytes": int(row["budget_bytes"]), "video_id": str(row["video_id"]),
            "sample_id": identity,
        }
        observations.extend((
            PairedMetricObservation(METHOD, value=float(row[field]), **common),
            PairedMetricObservation(BASELINE, value=float(row[baseline_field]), **common),
        ))
    return paired_video_bootstrap(
        observations, method_a=METHOD, method_b=BASELINE,
        config=BootstrapConfig(seed=seed, resamples=resamples),
    )


def summarize_jq01_oof(
    rows: Sequence[Mapping[str, Any]], *, snapshots_unchanged: bool,
    baseline_bit_exact: bool, byte_audit_complete: bool,
    provenance_complete: bool, fold_coverage_complete: bool,
    resamples: int = 2000, seed: int = 20260911, adequate_cell_videos: int = 12,
) -> dict[str, Any]:
    """Apply every preregistered J2 gate without exposing fields to the selector."""
    if len({str(row["observation_id"]) for row in rows}) != len(rows):
        raise ValueError("JQ OOF rows are duplicated")
    rows = [dict(row) for row in rows]
    by_budget = {
        budget: [row for row in rows if int(row["budget_bytes"]) == budget]
        for budget in (1024 * 1024, 8 * 1024 * 1024)
    }
    if any(not selected for selected in by_budget.values()):
        raise ValueError("JQ OOF must contain both frozen budgets")
    primary, secondary = by_budget[8 * 1024 * 1024], by_budget[1024 * 1024]
    for row in rows:
        row["baseline_recall_at_0_7"] = float(row["baseline_iou"] >= 0.7)
        row["jq_recall_at_0_7"] = float(row["jq_iou"] >= 0.7)
    primary_miou = _bootstrap(primary, "jq_iou", seed=seed, resamples=resamples)
    primary_r07 = _bootstrap(
        primary, "jq_recall_at_0_7", baseline_field="baseline_recall_at_0_7",
        seed=seed + 1, resamples=resamples,
    )
    secondary_miou = _bootstrap(secondary, "jq_iou", seed=seed + 2, resamples=resamples)
    secondary_r07 = _bootstrap(
        secondary, "jq_recall_at_0_7", baseline_field="baseline_recall_at_0_7",
        seed=seed + 3, resamples=resamples,
    )
    source_deltas = {
        source: _video_equal_delta(
            [row for row in primary if str(row["source"]) == source], "jq_iou", "baseline_iou",
        )
        for source in sorted({str(row["source"]) for row in primary})
    }
    cells = {}
    for row in primary:
        key = f"{row['source']}::{row['duration_bucket']}::rho={float(row['rho']):.6g}"
        cells.setdefault(key, []).append(row)
    adequate = {
        key: {
            "videos": len({str(row["video_id"]) for row in selected}),
            "delta": _video_equal_delta(selected, "jq_iou", "baseline_iou"),
        }
        for key, selected in cells.items()
        if len({str(row["video_id"]) for row in selected}) >= adequate_cell_videos
    }
    good = [row for row in primary if bool(row["good_baseline"])]
    good_delta = _video_equal_delta(good, "jq_iou", "baseline_iou") if good else None
    recoverable = _video_equal_delta(primary, "union_oracle_iou", "baseline_iou")
    transmitted = _video_equal_delta(primary, "jq_iou", "baseline_iou")
    recovery_ratio = transmitted / recoverable if recoverable > 0 else 0.0
    checks = {
        "primary_final_miou_ci_low_positive": primary_miou["ci_low"] > 0,
        "primary_final_r07_nonnegative": primary_r07["delta_a_minus_b"] >= 0
        and primary_r07["ci_high"] >= 0,
        "activitynet_and_charades_nonnegative": all(
            source_deltas.get(source, -math.inf) >= 0 for source in ("ActivityNet", "Charades")
        ),
        "adequate_cells_above_floor": bool(adequate)
        and all(row["delta"] >= -0.02 for row in adequate.values()),
        "good_baseline_above_floor": good_delta is not None and good_delta >= -0.01,
        "secondary_miou_above_floor": secondary_miou["delta_a_minus_b"] >= -0.005,
        "secondary_strict_recall_not_significantly_negative": secondary_r07["ci_high"] >= 0,
        "snapshot_unchanged": bool(snapshots_unchanged),
        "byte_audit_complete": bool(byte_audit_complete),
        "baseline_bit_exact": bool(baseline_bit_exact),
        "fold_coverage_complete": bool(fold_coverage_complete),
        "provenance_complete": bool(provenance_complete),
        "recoverable_delta_transmission_at_least_20pct": recovery_ratio >= 0.2,
    }
    passed = all(checks.values())
    return {
        "schema_version": 1, "stage": "jq01_development_oof", "passed": passed,
        "decision": "eligible_for_candidate_freeze" if passed else "iterate_development",
        "gate_checks": checks, "primary_miou_bootstrap": primary_miou,
        "primary_r07_bootstrap": primary_r07, "secondary_miou_bootstrap": secondary_miou,
        "secondary_r07_bootstrap": secondary_r07, "primary_source_deltas": source_deltas,
        "adequate_cell_deltas": adequate, "good_baseline_delta": good_delta,
        "union_oracle_recoverable_delta": recoverable, "final_transmitted_delta": transmitted,
        "recoverable_delta_transmission_ratio": recovery_ratio,
    }
