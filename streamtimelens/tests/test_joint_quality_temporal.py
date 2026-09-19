import math

import numpy as np
import pytest

from streamtimelens.observer.clip_encoder import serialize_embedding
from streamtimelens.retrieval.frame_candidates import FrameCandidate
from streamtimelens.retrieval.joint_quality import (
    BOUNDARY_TOKEN_CAP,
    CENTER_TOKEN_CAP,
    FEATURE_NAMES,
    TemporalJointQualityHead,
    _fit_temporal_scaler,
    build_joint_candidates,
    compute_temporal_jq_features,
    make_temporal_training_observation,
    stable_argmax,
    temporal_feature_schema,
    temporal_joint_quality_loss,
    tensorize_temporal_observations,
)


def _metadata(count=40):
    result = {}
    for index in range(count):
        cosine = 0.1 + 0.8 * index / max(1, count - 1)
        result[f"{index:09d}.jpg"] = {
            "blob": f"{index:09d}.jpg",
            "frame_index": index,
            "timestamp_s": float(index),
            "clip_embedding": serialize_embedding(np.asarray([
                cosine, math.sqrt(max(0.0, 1.0 - cosine * cosine)),
            ], dtype=np.float32)),
        }
    return result


def _observation(identifier="o", video="v", group="g"):
    candidates = build_joint_candidates(
        (8.0, 30.0),
        [
            FrameCandidate("extent-a", 10.0, 20.0, 0.9, (), ()),
            FrameCandidate("extent-b", 18.0, 28.0, 0.8, (), ()),
        ],
        duration_s=40.0,
    )
    rows = compute_temporal_jq_features(
        candidates, _metadata(), (1.0, 0.0), duration_s=40.0, rho=1.0,
        budget_bytes=1024 * 1024,
    )
    return make_temporal_training_observation(
        identifier, video, group, rows, (10.0, 20.0), duration_s=40.0,
    )


def test_temporal_schema_and_local_sequences_are_fixed_ordered_and_bounded():
    observation = _observation()
    schema = temporal_feature_schema()
    assert schema["center_token_cap"] == CENTER_TOKEN_CAP == 16
    assert schema["left_token_cap"] == BOUNDARY_TOKEN_CAP == 8
    assert schema["encoder"] == "independent-center-left-right-GRU"
    baseline = observation.rows[0]
    assert len(baseline.center_tokens) == CENTER_TOKEN_CAP
    assert len(baseline.left_tokens) == 2
    assert len(baseline.right_tokens) == 2
    assert all(len(token) == 5 for token in baseline.center_tokens)
    assert [token[2] for token in baseline.center_tokens] == sorted(
        token[2] for token in baseline.center_tokens
    )
    # Local rows contain no evaluation target; targets live in the wrapper only.
    assert not hasattr(baseline, "quality_targets")
    assert observation.gain_targets[observation.baseline_index] == pytest.approx(0.0)


def test_temporal_model_outputs_all_heads_and_loss_rewards_less_regret():
    import torch

    observation = _observation()
    scaler = _fit_temporal_scaler([observation])
    batch = tensorize_temporal_observations([observation], scaler, device="cpu")
    model = TemporalJointQualityHead(seed=9)
    outputs = model(batch)
    count = len(observation.rows)
    assert set(outputs) == {
        "selection_logits", "quality_logits", "gain_logits",
        "left_quality_logits", "right_quality_logits", "abstention_logits",
    }
    assert outputs["selection_logits"].shape == (count,)
    assert outputs["abstention_logits"].shape == (1,)
    loss, components = temporal_joint_quality_loss(outputs, [observation])
    assert torch.isfinite(loss)
    assert set(components) == {
        "quality", "gain", "left_boundary", "right_boundary", "listwise",
        "abstention", "relative_regret",
    }

    baseline = observation.baseline_index
    bad = min(
        (index for index, gain in enumerate(observation.gain_targets)
         if index != baseline and gain < 0),
        key=lambda index: observation.gain_targets[index],
    )
    safer = {name: value.clone() for name, value in outputs.items()}
    riskier = {name: value.clone() for name, value in outputs.items()}
    safer["selection_logits"][bad] = safer["selection_logits"][baseline] - 3
    riskier["selection_logits"][bad] = riskier["selection_logits"][baseline] + 3
    _, safe_components = temporal_joint_quality_loss(safer, [observation])
    _, risky_components = temporal_joint_quality_loss(riskier, [observation])
    assert safe_components["relative_regret"] < risky_components["relative_regret"]


def test_temporal_pool_and_loss_are_candidate_permutation_equivariant():
    observation = _observation()
    permutation = (2, 0, 1)
    permuted = type(observation)(
        observation.observation_id, observation.video_id, observation.group_id,
        tuple(observation.rows[index] for index in permutation),
        tuple(observation.quality_targets[index] for index in permutation),
        tuple(observation.gain_targets[index] for index in permutation),
        tuple(observation.left_boundary_targets[index] for index in permutation),
        tuple(observation.right_boundary_targets[index] for index in permutation),
    )
    scaler = _fit_temporal_scaler([observation])
    model = TemporalJointQualityHead(seed=11)
    original_outputs = model(tensorize_temporal_observations([observation], scaler, device="cpu"))
    permuted_outputs = model(tensorize_temporal_observations([permuted], scaler, device="cpu"))
    original_by_id = dict(zip(observation.candidate_ids, original_outputs["selection_logits"].tolist()))
    permuted_by_id = dict(zip(permuted.candidate_ids, permuted_outputs["selection_logits"].tolist()))
    assert original_by_id == pytest.approx(permuted_by_id, abs=1e-7)
    original_loss, _ = temporal_joint_quality_loss(original_outputs, [observation])
    permuted_loss, _ = temporal_joint_quality_loss(permuted_outputs, [permuted])
    assert float(original_loss) == pytest.approx(float(permuted_loss), abs=1e-7)
    assert stable_argmax([1, 1], ["b", "a"]) == 1
    assert len(observation.rows[0].scalar_values) == len(FEATURE_NAMES)
