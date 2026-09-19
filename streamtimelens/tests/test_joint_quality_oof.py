import numpy as np
import pytest

from streamtimelens.evaluation.joint_quality_oof import (
    JQOOFObservation,
    deterministic_group_folds,
    run_joint_quality_oof,
    summarize_jq01_oof,
)
from streamtimelens.retrieval.joint_quality import (
    FEATURE_NAMES,
    CandidateFeatureRow,
    JointCandidate,
)


def _observation(group, budget, source):
    baseline = JointCandidate("frozen-v2-final", 0, 1, True, 0)
    extent = JointCandidate("extent-a", 0, 2, False, 1, .8)
    first = np.zeros(len(FEATURE_NAMES), dtype=float)
    first[FEATURE_NAMES.index("is_frozen_baseline")] = 1
    second = np.zeros(len(FEATURE_NAMES), dtype=float)
    second[FEATURE_NAMES.index("pooled_cos")] = 1
    return JQOOFObservation(
        f"{group}-{budget}", f"video-{group}", group, f"query-{group}", "a" * 64,
        budget, .5, source, "short",
        (baseline, extent),
        (CandidateFeatureRow(baseline, tuple(first)), CandidateFeatureRow(extent, tuple(second))),
        (0, 2), True,
    )


def test_group_oof_predicts_each_heldout_observation_once_and_audits_folds():
    rows = [
        _observation(f"group-{index}", budget, "ActivityNet" if index % 2 == 0 else "Charades")
        for index in range(4) for budget in (1024 * 1024, 8 * 1024 * 1024)
    ]
    folds = {f"group-{index}": index for index in range(4)}
    outputs, checkpoints, audit = run_joint_quality_oof(
        rows, folds=4, seed=7, code_revision="a" * 40, fold_by_group=folds,
    )
    assert len(outputs) == len(rows)
    assert len({row["observation_id"] for row in outputs}) == len(rows)
    assert set(checkpoints) == {0, 1, 2, 3}
    for fold, details in audit["folds"].items():
        assert not set(details["outer_train_groups"]) & set(details["heldout_groups"])
        assert all(row["outer_fold"] == int(fold) for row in outputs if row["group_id"] in details["heldout_groups"])


def test_fold_assignment_is_order_independent():
    groups = ["d", "a", "c", "b"]
    assert deterministic_group_folds(groups, folds=4) == deterministic_group_folds(list(reversed(groups)), folds=4)


def test_j2_gate_summary_checks_both_budgets_sources_and_recovery_ratio():
    rows = []
    for budget in (1024 * 1024, 8 * 1024 * 1024):
        for index, source in enumerate(("ActivityNet", "Charades")):
            rows.append({
                "observation_id": f"{budget}-{index}", "video_id": f"v-{budget}-{index}",
                "budget_bytes": budget, "source": source, "duration_bucket": "short",
                "rho": .5, "good_baseline": True, "baseline_iou": .6, "jq_iou": .8,
                "union_oracle_iou": 1.0,
            })
    summary = summarize_jq01_oof(
        rows, snapshots_unchanged=True, baseline_bit_exact=True,
        byte_audit_complete=True, provenance_complete=True, fold_coverage_complete=True,
        adequate_cell_videos=1, resamples=100,
    )
    assert summary["passed"]
    assert summary["decision"] == "eligible_for_candidate_freeze"
    assert summary["recoverable_delta_transmission_ratio"] == pytest.approx(0.5)
    assert summary["primary_r07_bootstrap"]["delta_a_minus_b"] == 1.0


def test_j2_gate_fails_when_byte_audit_is_incomplete():
    rows = []
    for budget in (1024 * 1024, 8 * 1024 * 1024):
        for index, source in enumerate(("ActivityNet", "Charades")):
            rows.append({
                "observation_id": f"{budget}-{index}",
                "video_id": f"v-{budget}-{index}",
                "budget_bytes": budget,
                "source": source,
                "duration_bucket": "short",
                "rho": .5,
                "good_baseline": True,
                "baseline_iou": .6,
                "jq_iou": .8,
                "union_oracle_iou": 1.0,
            })
    summary = summarize_jq01_oof(
        rows, snapshots_unchanged=True, baseline_bit_exact=True,
        byte_audit_complete=False, provenance_complete=True,
        fold_coverage_complete=True, adequate_cell_videos=1, resamples=20,
    )
    assert not summary["passed"]
    assert not summary["gate_checks"]["byte_audit_complete"]
