"""Layered G0 parity checks for the pinned SnAG bridge."""

from __future__ import annotations

import math
from typing import Mapping, Sequence

import numpy as np

from streamtimelens.baselines.snag_adapt import RankedSpan
from streamtimelens.baselines.vendor.snag_model import SnAGRawTrace, UPSTREAM_REVISION


def _arrays_close(
    reference: Sequence[np.ndarray], candidate: Sequence[np.ndarray],
    *, atol: float, rtol: float,
) -> tuple[bool, float]:
    if len(reference) != len(candidate):
        return False, float("inf")
    maximum = 0.0
    for left, right in zip(reference, candidate):
        if left.shape != right.shape:
            return False, float("inf")
        if left.size:
            maximum = max(maximum, float(np.max(np.abs(left.astype(float) - right.astype(float)))))
        if not np.allclose(left, right, atol=atol, rtol=rtol):
            return False, maximum
    return True, maximum


def compare_raw_traces(
    reference: SnAGRawTrace,
    candidate: SnAGRawTrace,
    *,
    atol: float = 1e-6,
    rtol: float = 1e-5,
) -> dict[str, object]:
    """Compare FPN contracts and pre-decode model outputs."""
    logits_ok, logits_error = _arrays_close(
        reference.logits, candidate.logits, atol=atol, rtol=rtol,
    )
    offsets_ok, offsets_error = _arrays_close(
        reference.offsets, candidate.offsets, atol=atol, rtol=rtol,
    )
    masks_ok, masks_error = _arrays_close(
        reference.masks, candidate.masks, atol=0.0, rtol=0.0,
    )
    checks = {
        "input_length": reference.input_length == candidate.input_length,
        "observed_length": reference.observed_length == candidate.observed_length,
        "fpn_shapes": reference.fpn_shapes == candidate.fpn_shapes,
        "mask_shapes": reference.mask_shapes == candidate.mask_shapes,
        "logits": logits_ok,
        "offsets": offsets_ok,
        "masks": masks_ok,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "maximum_absolute_error": {
            "logits": logits_error,
            "offsets": offsets_error,
            "masks": masks_error,
        },
        "atol": atol,
        "rtol": rtol,
    }


def compare_ranked_spans(
    reference: Sequence[RankedSpan],
    candidate: Sequence[RankedSpan],
    *,
    top_k: int = 5,
    time_atol_s: float = 1e-4,
    score_atol: float = 1e-6,
) -> dict[str, object]:
    """Compare the post-NMS Top-K contract without hiding count mismatches."""
    if top_k < 1:
        raise ValueError("top_k must be positive")
    left, right = tuple(reference[:top_k]), tuple(candidate[:top_k])
    count_ok = len(left) == len(right)
    time_error = score_error = 0.0
    if count_ok:
        for expected, actual in zip(left, right):
            time_error = max(
                time_error,
                abs(expected.start_s - actual.start_s),
                abs(expected.end_s - actual.end_s),
            )
            score_error = max(score_error, abs(expected.score - actual.score))
    checks = {
        "count": count_ok,
        "timestamps": count_ok and time_error <= time_atol_s,
        "scores": count_ok and score_error <= score_atol,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "top_k": top_k,
        "maximum_absolute_error": {
            "timestamps_s": time_error if count_ok else None,
            "scores": score_error if count_ok else None,
        },
        "reference": [vars(value) for value in left],
        "candidate": [vars(value) for value in right],
    }


def compare_official_metrics(
    expected: Mapping[str, float], actual: Mapping[str, float],
    *, absolute_tolerance_points: float = 0.25,
) -> dict[str, object]:
    """Compare published/evaluator percentage metrics using frozen keys."""
    if not expected or set(expected) != set(actual):
        return {
            "passed": False, "checks": {},
            "reason": "metric keys are empty or mismatched",
        }
    errors = {name: abs(float(actual[name]) - float(value)) for name, value in expected.items()}
    checks = {
        name: math.isfinite(error) and error <= absolute_tolerance_points
        for name, error in errors.items()
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "expected": dict(expected),
        "actual": dict(actual),
        "absolute_errors_points": errors,
        "absolute_tolerance_points": absolute_tolerance_points,
    }


def g0_gate_report(
    *,
    loader_provenance: Mapping[str, object],
    raw_parity: Mapping[str, object],
    post_nms_parity: Mapping[str, object] | None,
    official_metric_parity: Mapping[str, object] | None,
) -> dict[str, object]:
    """Only a real checkpoint plus all three parity layers can pass G0."""
    checkpoint_digest = str(loader_provenance.get("checkpoint_sha256", ""))
    option_digest = str(loader_provenance.get("option_sha256", ""))
    real_checkpoint = (
        loader_provenance.get("upstream_revision") == UPSTREAM_REVISION
        and len(checkpoint_digest) == 64
        and len(option_digest) == 64
        and all(character in "0123456789abcdef" for character in checkpoint_digest + option_digest)
    )
    checks = {
        "pinned_real_checkpoint_loaded": real_checkpoint,
        "fpn_logits_offsets_masks": bool(raw_parity.get("passed")),
        "post_nms_topk": bool(post_nms_parity and post_nms_parity.get("passed")),
        "official_protocol_metrics": bool(
            official_metric_parity and official_metric_parity.get("passed")
        ),
    }
    return {
        "schema_version": 1,
        "gate": "SnAG-G0-upstream-parity",
        "passed": all(checks.values()),
        "classification": "external-reference" if all(checks.values()) else "not-a-result",
        "checks": checks,
        "loader": dict(loader_provenance),
        "raw_parity": dict(raw_parity),
        "post_nms_parity": dict(post_nms_parity) if post_nms_parity else None,
        "official_metric_parity": (
            dict(official_metric_parity) if official_metric_parity else None
        ),
    }
