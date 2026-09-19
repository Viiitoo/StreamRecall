"""Decision logic for the frozen 10-second, 16-frame visual refiner gate."""

from __future__ import annotations

import math
from typing import Any, Mapping


def disabled_visual_refiner_gate(reason: str) -> dict[str, Any]:
    if not reason.strip():
        raise ValueError("disabled visual refiner gate needs a reason")
    return {
        "schema_version": 1,
        "passed": False,
        "decision": "disable",
        "reason": reason.strip(),
        "frozen_setting": {
            "candidate_margin_s": 10.0, "sampling": "uniform", "max_frames": 16,
        },
        "checks": {"retrieved_candidate_refinement_feasible": False},
        "oracle_candidate": None,
        "retrieved_candidate": None,
    }


def build_visual_refiner_gate(
    oracle_candidate: Mapping[str, Any],
    retrieved_candidate: Mapping[str, Any],
    *,
    candidate_margin_s: float = 10.0,
    max_frames: int = 16,
) -> dict[str, Any]:
    dense_miou = float(oracle_candidate["dense_miou"])
    sparse_miou = float(oracle_candidate["sparse_miou"])
    interval_s = float(oracle_candidate["sampling_interval_s"])
    start_bias = float(oracle_candidate["start_signed_bias_s"])
    end_bias = float(oracle_candidate["end_signed_bias_s"])
    coarse_miou = float(retrieved_candidate["coarse_miou"])
    refined_miou = float(retrieved_candidate["refined_miou"])
    count = int(retrieved_candidate["count"])
    valid_count = int(retrieved_candidate["valid_count"])
    fallback_count = int(retrieved_candidate["fallback_count"])
    numeric = (
        dense_miou, sparse_miou, interval_s, start_bias, end_bias,
        coarse_miou, refined_miou,
    )
    if (
        any(not math.isfinite(value) for value in numeric)
        or interval_s <= 0 or candidate_margin_s <= 0 or max_frames <= 0
    ):
        raise ValueError("visual refiner gate metrics are invalid")
    if count <= 0 or min(valid_count, fallback_count) < 0 or valid_count + fallback_count > count:
        raise ValueError("visual refiner gate output counts are invalid")
    drop = dense_miou - sparse_miou
    delta = refined_miou - coarse_miou
    checks = {
        "oracle_sparse_drop_le_0.05": drop <= 0.05,
        "start_bias_within_sampling_interval": abs(start_bias) <= interval_s,
        "end_bias_within_sampling_interval": abs(end_bias) <= interval_s,
        "retrieved_candidate_delta_positive": delta > 0,
    }
    passed = all(checks.values())
    return {
        "schema_version": 1,
        "passed": passed,
        "decision": "enable" if passed else "disable",
        "frozen_setting": {
            "candidate_margin_s": candidate_margin_s,
            "sampling": "uniform", "max_frames": max_frames,
        },
        "checks": checks,
        "oracle_candidate": {
            **dict(oracle_candidate), "sparse_minus_dense_miou": sparse_miou - dense_miou,
        },
        "retrieved_candidate": {
            **dict(retrieved_candidate), "refined_minus_coarse_miou": delta,
            "valid_output_rate": valid_count / count,
            "fallback_rate": fallback_count / count,
        },
    }
