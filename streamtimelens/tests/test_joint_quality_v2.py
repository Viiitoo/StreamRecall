import numpy as np
import pytest

from streamtimelens.evaluation.joint_quality_v2 import (
    JQV2Observation,
    V2SnapshotFeatures,
    run_v2_nested_oof,
    summarize_v2_nested_oof,
)
from streamtimelens.observer.clip_encoder import serialize_embedding
from streamtimelens.retrieval.extent_tokens import build_temporal_cell_tokens


def _snapshot(index):
    direction = np.asarray([1.0, 0.1 + index * 0.01], dtype=np.float32)
    metadata = {
        f"{cell:09d}.jpg": {
            "frame_index": cell,
            "timestamp_s": float(cell * 2 + 1),
            "clip_embedding": serialize_embedding(
                direction if cell == index % 3 else np.asarray([0.1, 1.0], dtype=np.float32)
            ),
        }
        for cell in range(3)
    }
    return V2SnapshotFeatures(
        6.0, 6.0, 1024 * 1024, metadata,
        build_temporal_cell_tokens(metadata, upper_bound_s=6.0),
    )


def test_nested_v2_oof_predicts_every_outer_video_without_candidate_leakage():
    snapshots = {f"snapshot-{index}": _snapshot(index) for index in range(8)}
    rows = []
    for index in range(8):
        cell = index % 3
        rows.append(JQV2Observation(
            f"observation-{index}", f"video-{index}", f"video-{index}",
            f"query-{index}", f"{index:064x}", f"snapshot-{index}", 1024 * 1024,
            1.0, "Charades", "short", (1.0, 0.1 + index * 0.01),
            (0.0, 6.0), (float(cell * 2), float(cell * 2 + 2)), False,
        ))
    folds = {f"video-{index}": index % 4 for index in range(8)}
    outputs, checkpoints, audit = run_v2_nested_oof(
        rows, snapshots, fold_by_group=folds, folds=4, seed=7,
        code_revision="a" * 40,
    )
    assert len(outputs) == 8
    assert len({row["observation_id"] for row in outputs}) == 8
    assert set(checkpoints) == {0, 1, 2, 3}
    for fold, details in audit["folds"].items():
        assert not set(details["outer_train_groups"]) & set(details["heldout_groups"])
        for x1_fold in details["outer_train_x1_crossfit"]:
            assert x1_fold["training_groups_sha256"] != x1_fold["heldout_groups_sha256"]
        assert all(
            row["outer_fold"] == int(fold)
            for row in outputs if row["group_id"] in details["heldout_groups"]
        )


def test_v2_summary_accepts_a_positive_same_observation_gain():
    rows = [
        {
            "observation_id": f"q-{index}", "video_id": f"v-{index}",
            "budget_bytes": 1024 * 1024, "rho": 1.0, "good_baseline": True,
            "baseline_iou": 0.6, "jq_iou": 0.8, "union_oracle_iou": 1.0,
        }
        for index in range(8)
    ]
    summary = summarize_v2_nested_oof(
        rows, snapshots_unchanged=True, byte_audit_complete=True,
        fold_coverage_complete=True, baseline_bit_exact=True,
        provenance_complete=True, resamples=100,
    )
    assert summary["passed"]
    assert summary["decision"] == "jq01_structural_revision_passed"
    assert summary["recoverable_delta_transmission_ratio"] == pytest.approx(0.5)
