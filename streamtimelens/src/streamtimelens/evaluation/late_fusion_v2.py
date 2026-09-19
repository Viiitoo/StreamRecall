"""Leakage-free LF-01 OOF on the original Frozen Visual V2 development set."""

from __future__ import annotations

import hashlib
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from streamtimelens.evaluation.bootstrap import (
    BootstrapConfig,
    PairedMetricObservation,
    paired_video_bootstrap,
)
from streamtimelens.evaluation.joint_quality_v2 import (
    JQV2Observation,
    V2SnapshotFeatures,
    _extent_observation,
    _fit_extent,
)
from streamtimelens.retrieval.extent_tokens import ExtentPipeline, predict_extent_candidates
from streamtimelens.retrieval.late_fusion import (
    LOSS_WEIGHTS,
    LateFusionTrainingObservation,
    _canonical_json,
    _identity_sha,
    _torch,
    build_candidates,
    fit_late_fusion_fixed_epochs,
    make_checkpoint,
    make_training_observation,
    stable_argmax,
    temporal_tokens_from_metadata,
    tensorize,
    tensorize_training_observations,
    train_late_fusion_head,
)


BASELINE = "frozen_visual_v2"
METHOD = "LF-01"
R2_STANDARD_MIOU = 0.400079
MAX_READOUT_P95_MS = 250.0


@dataclass(frozen=True)
class LFV2Observation:
    source: JQV2Observation
    training: LateFusionTrainingObservation


def _materialize(
    rows: Sequence[JQV2Observation], snapshots: Mapping[str, V2SnapshotFeatures],
    pipeline: ExtentPipeline,
) -> list[LFV2Observation]:
    result = []
    for row in sorted(rows, key=lambda item: item.observation_id):
        snapshot = snapshots[row.snapshot_id]
        extent_candidates, _ = predict_extent_candidates(
            pipeline, _extent_observation(row, snapshots, with_gt=False),
        )
        candidates = build_candidates(
            row.baseline_span, extent_candidates, duration_s=snapshot.duration_s,
        )
        sequence, query = temporal_tokens_from_metadata(
            snapshot.frame_metadata, row.query_embedding,
            duration_s=snapshot.duration_s, t_q=snapshot.upper_bound_s,
        )
        training = make_training_observation(
            row.observation_id, row.video_id, row.group_id, sequence, query,
            candidates, row.gt_span,
        )
        result.append(LFV2Observation(row, training))
    return result


def _crossfit_materialize(
    rows: Sequence[JQV2Observation], snapshots: Mapping[str, V2SnapshotFeatures],
    fold_by_group: Mapping[str, int],
) -> tuple[list[LFV2Observation], list[dict[str, Any]]]:
    folds = sorted({int(fold_by_group[row.group_id]) for row in rows})
    if len(folds) < 2:
        raise ValueError("LF-01 X1 cross-fit needs at least two folds")
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
            "pipeline_sha256": hashlib.sha256(_canonical_json(payload)).hexdigest(),
        })
    if (
        len(materialized) != len(rows)
        or len({row.source.observation_id for row in materialized}) != len(rows)
    ):
        raise RuntimeError("LF-01 X1 cross-fit coverage failed")
    return sorted(materialized, key=lambda row: row.source.observation_id), audits


def _readout_latency_ms(model: Any, rows: Sequence[LFV2Observation], *, device: str) -> list[float]:
    torch, _ = _torch()
    model.module.to(device).eval()
    measurements = []
    with torch.no_grad():
        for row in rows:
            tensors = tensorize(
                row.training.sequence, row.training.query, row.training.candidates,
                device=device,
            )
            if str(device).startswith("cuda"):
                torch.cuda.synchronize(device)
            started = time.perf_counter()
            model(**tensors)
            if str(device).startswith("cuda"):
                torch.cuda.synchronize(device)
            measurements.append((time.perf_counter() - started) * 1000.0)
    model.module.to("cpu").eval()
    return measurements


def run_lf_v2_nested_oof(
    observations: Sequence[JQV2Observation], snapshots: Mapping[str, V2SnapshotFeatures], *,
    fold_by_group: Mapping[str, int], folds: int, seed: int, code_revision: str,
    device: str,
) -> tuple[list[dict[str, Any]], dict[int, dict[str, Any]], dict[str, Any]]:
    if not observations or set(fold_by_group.values()) != set(range(folds)):
        raise ValueError("LF-01 needs complete fixed outer folds")
    if any(row.snapshot_id not in snapshots for row in observations):
        raise ValueError("LF-01 observation lacks a verified snapshot")
    video_folds: dict[str, set[int]] = defaultdict(set)
    for row in observations:
        video_folds[row.video_id].add(int(fold_by_group[row.group_id]))
    if any(len(values) != 1 for values in video_folds.values()):
        raise ValueError("LF-01 video crosses outer folds")
    torch, _ = _torch()
    outputs: list[dict[str, Any]] = []
    checkpoints: dict[int, dict[str, Any]] = {}
    fold_audit: dict[str, Any] = {}
    for outer in range(folds):
        heldout = [row for row in observations if fold_by_group[row.group_id] == outer]
        outer_train = [row for row in observations if fold_by_group[row.group_id] != outer]
        validation_fold = (outer + 1) % folds
        inner_train_raw = [
            row for row in outer_train if fold_by_group[row.group_id] != validation_fold
        ]
        validation_raw = [
            row for row in outer_train if fold_by_group[row.group_id] == validation_fold
        ]
        heldout_videos = sorted({row.video_id for row in heldout})
        heldout_groups = sorted({row.group_id for row in heldout})
        inner_train_rows, inner_x1_audit = _crossfit_materialize(
            inner_train_raw, snapshots, fold_by_group,
        )
        validation_pipeline = _fit_extent(inner_train_raw, snapshots)
        validation_rows = _materialize(validation_raw, snapshots, validation_pipeline)
        _, selection = train_late_fusion_head(
            [row.training for row in inner_train_rows],
            [row.training for row in validation_rows],
            seed=seed + outer, device=device,
            heldout_video_ids=heldout_videos, heldout_group_ids=heldout_groups,
        )
        outer_train_rows, outer_x1_audit = _crossfit_materialize(
            outer_train, snapshots, fold_by_group,
        )
        model = fit_late_fusion_fixed_epochs(
            [row.training for row in outer_train_rows],
            epochs=int(selection["best_epoch"]), seed=seed + outer, device=device,
        )
        heldout_pipeline = _fit_extent(outer_train, snapshots)
        heldout_rows = _materialize(heldout, snapshots, heldout_pipeline)
        train_videos = sorted({row.video_id for row in outer_train})
        train_groups = sorted({row.group_id for row in outer_train})
        checkpoint = make_checkpoint(
            model, code_revision=code_revision,
            outer_train_video_sha256=_identity_sha(train_videos),
            outer_train_group_sha256=_identity_sha(train_groups),
        )
        checkpoints[outer] = checkpoint
        latency = _readout_latency_ms(model, heldout_rows, device=device)
        model.module.to(device).eval()
        batch = tensorize_training_observations(
            [row.training for row in heldout_rows], device=device,
        )
        with torch.no_grad():
            predictions = model.forward_batch(**batch["model"])
        quality = predictions["candidate_quality_logits"].detach().cpu().numpy()
        gain = predictions["candidate_gain_logits"].detach().cpu().numpy()
        selection_logits = predictions["selection_logits"].detach().cpu().numpy()
        saliency = predictions["saliency_logits"].detach().cpu().numpy()
        attention = predictions["attention_weights"].detach().cpu().numpy()
        model.module.to("cpu").eval()
        for index, row in enumerate(heldout_rows):
            source, view = row.source, row.training
            scores = selection_logits[index, :len(view.candidates)]
            selected = stable_argmax(scores, view.candidate_ids)
            ranking = sorted(
                range(len(view.candidates)),
                key=lambda offset: (-float(scores[offset]), view.candidate_ids[offset]),
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
                "baseline_candidate_id": view.candidate_ids[view.baseline_index],
                "selected_candidate_id": view.candidate_ids[selected],
                "pre_span": list(view.candidates[view.baseline_index].span),
                "final_span": list(view.candidates[selected].span),
                "baseline_iou": view.quality_targets[view.baseline_index],
                "lf_iou": view.quality_targets[selected],
                "union_oracle_iou": max(view.quality_targets),
                "readout_latency_ms": latency[index],
                "temporal_tokens": [
                    {
                        "frame_ref": frame_ref,
                        "timestamp_s": timestamp,
                        "saliency_target": view.saliency_targets[token_index],
                        "saliency_logit": float(saliency[index, token_index]),
                        "attention": float(attention[index, token_index]),
                    }
                    for token_index, (frame_ref, timestamp) in enumerate(zip(
                        view.sequence.frame_refs, view.sequence.timestamps_s,
                    ))
                ],
                "candidate_rows": [
                    {
                        "candidate_id": candidate.candidate_id,
                        "span": list(candidate.span),
                        "target": view.quality_targets[candidate_index],
                        "quality_logit": float(quality[index, candidate_index]),
                        "gain_logit": float(gain[index, candidate_index]),
                        "selection_logit": float(scores[candidate_index]),
                        "rank": ranking.index(candidate_index) + 1,
                        "selected": candidate_index == selected,
                    }
                    for candidate_index, candidate in enumerate(view.candidates)
                ],
            })
        heldout_payload = asdict(heldout_pipeline)
        validation_payload = asdict(validation_pipeline)
        fold_audit[str(outer)] = {
            "outer_train_groups": train_groups,
            "heldout_groups": heldout_groups,
            "heldout_videos": heldout_videos,
            "inner_train_groups": sorted({row.group_id for row in inner_train_raw}),
            "inner_validation_groups": sorted({row.group_id for row in validation_raw}),
            "inner_validation_fold": validation_fold,
            "inner_train_x1_crossfit": inner_x1_audit,
            "outer_train_x1_crossfit": outer_x1_audit,
            "inner_validation_x1_pipeline_sha256": hashlib.sha256(
                _canonical_json(validation_payload)
            ).hexdigest(),
            "heldout_x1_pipeline_sha256": hashlib.sha256(
                _canonical_json(heldout_payload)
            ).hexdigest(),
            "selector_checkpoint_sha256": checkpoint["checkpoint_sha256"],
            "best_epoch": selection["best_epoch"],
            "training": selection,
            "readout_latency_ms": {
                "count": len(latency), "mean": float(np.mean(latency)),
                "p95": float(np.percentile(latency, 95)), "max": float(np.max(latency)),
                "device": device,
            },
        }
    if len(outputs) != len(observations) or len({row["observation_id"] for row in outputs}) != len(outputs):
        raise RuntimeError("LF-01 OOF coverage failed")
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
        raise ValueError("empty LF-01 metric slice")
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
        observations, method_a=METHOD, method_b=BASELINE,
        config=BootstrapConfig(seed=seed, resamples=resamples),
    )


def summarize_lf_v2_oof(
    rows: Sequence[Mapping[str, Any]], *, snapshots_unchanged: bool,
    byte_audit_complete: bool, fold_coverage_complete: bool,
    baseline_bit_exact: bool, provenance_complete: bool,
    seed: int = 20260911, resamples: int = 2000,
) -> dict[str, Any]:
    rows = [dict(row) for row in rows]
    if not rows or len({row["observation_id"] for row in rows}) != len(rows):
        raise ValueError("LF-01 summary rows are empty or duplicated")
    for row in rows:
        row["baseline_r07"] = float(row["baseline_iou"] >= 0.7)
        row["lf_r07"] = float(row["lf_iou"] >= 0.7)
    standard = {
        "baseline_miou": float(np.mean([row["baseline_iou"] for row in rows])),
        "lf_miou": float(np.mean([row["lf_iou"] for row in rows])),
        "delta_miou": float(np.mean([row["lf_iou"] - row["baseline_iou"] for row in rows])),
        "baseline_r07": float(np.mean([row["baseline_r07"] for row in rows])),
        "lf_r07": float(np.mean([row["lf_r07"] for row in rows])),
        "delta_r07": float(np.mean([row["lf_r07"] - row["baseline_r07"] for row in rows])),
        "observation_count": len(rows),
    }
    miou = _bootstrap(rows, "lf_iou", "baseline_iou", seed=seed, resamples=resamples)
    r07 = _bootstrap(rows, "lf_r07", "baseline_r07", seed=seed + 1, resamples=resamples)
    rho = {
        f"{value:.2f}": _video_delta(
            [row for row in rows if abs(float(row["rho"]) - value) < 1e-8],
            "lf_iou", "baseline_iou",
        )
        for value in sorted({float(row["rho"]) for row in rows})
    }
    good = [row for row in rows if bool(row["good_baseline"])]
    good_delta = _video_delta(good, "lf_iou", "baseline_iou") if good else None
    recoverable = _video_delta(rows, "union_oracle_iou", "baseline_iou")
    transmitted = _video_delta(rows, "lf_iou", "baseline_iou")
    recovery_ratio = transmitted / recoverable if recoverable > 0 else 0.0
    p95_latency = float(np.percentile([row["readout_latency_ms"] for row in rows], 95))
    checks = {
        "standard_miou_above_frozen_and_jq_r2": (
            standard["lf_miou"] > standard["baseline_miou"]
            and standard["lf_miou"] > R2_STANDARD_MIOU
        ),
        "video_equal_miou_positive": miou["delta_a_minus_b"] > 0,
        "video_equal_miou_ci_low_positive": miou["ci_low"] > 0,
        "standard_r07_nonnegative": standard["delta_r07"] >= 0,
        "rho_slices_above_floor": all(value >= -0.02 for value in rho.values()),
        "good_baseline_above_floor": good_delta is not None and good_delta >= -0.01,
        "recoverable_delta_transmission_at_least_20pct": recovery_ratio >= 0.2,
        "readout_latency_p95_at_most_250ms": p95_latency <= MAX_READOUT_P95_MS,
        "snapshot_unchanged": bool(snapshots_unchanged),
        "byte_audit_complete": bool(byte_audit_complete),
        "fold_coverage_complete": bool(fold_coverage_complete),
        "baseline_bit_exact": bool(baseline_bit_exact),
        "provenance_complete": bool(provenance_complete),
    }
    return {
        "schema_version": 1,
        "stage": "lf01_v2_original_dev_nested_oof",
        "passed": all(checks.values()),
        "decision": "freeze_lf01_revision" if all(checks.values()) else "lf01_revision_not_frozen",
        "gate_checks": checks,
        "standard_metrics": standard,
        "miou_bootstrap": miou,
        "r07_bootstrap": r07,
        "rho_deltas": rho,
        "good_baseline_delta": good_delta,
        "union_oracle_recoverable_delta": recoverable,
        "final_transmitted_delta": transmitted,
        "recoverable_delta_transmission_ratio": recovery_ratio,
        "readout_latency_ms": {"p95": p95_latency, "limit": MAX_READOUT_P95_MS},
        "loss_weights": dict(LOSS_WEIGHTS),
    }
