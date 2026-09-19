from streamtimelens.evaluation.late_fusion_v2 import (
    MAX_READOUT_P95_MS,
    R2_STANDARD_MIOU,
    summarize_lf_v2_oof,
)


def _rows(delta=0.08, latency=3.0):
    rows = []
    for video in range(12):
        baseline = 0.36 + 0.005 * (video % 3)
        rows.append({
            "observation_id": f"o-{video}",
            "video_id": f"v-{video}",
            "budget_bytes": 1024 * 1024,
            "rho": (0.25, 0.5, 0.75, 1.0)[video % 4],
            "good_baseline": baseline >= 0.35,
            "baseline_iou": baseline,
            "lf_iou": baseline + delta,
            "union_oracle_iou": min(1.0, baseline + 0.20),
            "readout_latency_ms": latency,
        })
    return rows


def test_lf_v2_gate_freezes_effect_and_latency_thresholds():
    assert R2_STANDARD_MIOU == 0.400079
    assert MAX_READOUT_P95_MS == 250.0
    summary = summarize_lf_v2_oof(
        _rows(), snapshots_unchanged=True, byte_audit_complete=True,
        fold_coverage_complete=True, baseline_bit_exact=True,
        provenance_complete=True, resamples=200,
    )
    assert summary["passed"]
    assert summary["decision"] == "freeze_lf01_revision"
    assert all(summary["gate_checks"].values())


def test_lf_v2_gate_rejects_regression_and_slow_readout():
    summary = summarize_lf_v2_oof(
        _rows(delta=-0.01, latency=300.0), snapshots_unchanged=True,
        byte_audit_complete=True, fold_coverage_complete=True,
        baseline_bit_exact=True, provenance_complete=True, resamples=100,
    )
    assert not summary["passed"]
    assert summary["decision"] == "lf01_revision_not_frozen"
    assert not summary["gate_checks"]["standard_miou_above_frozen_and_jq_r2"]
    assert not summary["gate_checks"]["readout_latency_p95_at_most_250ms"]
