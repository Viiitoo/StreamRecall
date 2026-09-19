from streamtimelens.evaluation.hierarchical_event_v2 import (
    MAX_READOUT_P95_MS,
    summarize_hem_v2_development,
)


def _rows(delta=0.08, latency=2.0):
    return [
        {
            "observation_id": f"o-{index}",
            "video_id": f"v-{index}",
            "budget_bytes": 1024 * 1024,
            "rho": (0.25, 0.5, 0.75, 1.0)[index % 4],
            "good_baseline": True,
            "baseline_iou": 0.35,
            "hem_iou": 0.35 + delta,
            "hem_candidate_oracle_iou": 0.65,
            "readout_latency_ms": latency,
            "snapshot_state_bytes": 200000,
            "event_count": 80,
        }
        for index in range(12)
    ]


def _summary(rows, **overrides):
    gates = {
        "support_complete": True,
        "zero_seek": True,
        "snapshots_unchanged": True,
        "byte_audit_complete": True,
        "coverage_complete": True,
        "provenance_complete": True,
    }
    gates.update(overrides)
    return summarize_hem_v2_development(rows, resamples=200, **gates)


def test_hem_effect_gates_pass_only_complete_positive_result():
    assert MAX_READOUT_P95_MS == 50.0
    summary = _summary(_rows())
    assert summary["passed"]
    assert summary["decision"] == "freeze_hem01_r1"
    assert all(summary["gate_checks"].values())


def test_hem_effect_gates_reject_regression_latency_and_protocol_failure():
    summary = _summary(_rows(delta=-0.03, latency=60.0), zero_seek=False)
    assert not summary["passed"]
    assert summary["decision"] == "hem01_r1_not_frozen"
    assert not summary["gate_checks"]["video_equal_miou_ci_low_positive"]
    assert not summary["gate_checks"]["readout_latency_p95_at_most_50ms"]
    assert not summary["gate_checks"]["zero_seek"]
