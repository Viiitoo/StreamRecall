import math
import json
import hashlib
import tempfile
from pathlib import Path

import numpy as np
import pytest

from streamtimelens.observer.clip_encoder import serialize_embedding
from streamtimelens.retrieval.frame_candidates import FrameCandidate
from streamtimelens.retrieval.extent_tokens import (
    ExtentObservation,
    build_temporal_cell_tokens,
    fit_extent_pipeline,
    predict_extent_candidates,
)
from streamtimelens.retrieval.joint_quality import (
    FEATURE_NAMES,
    FeatureScaler,
    JointQualityHead,
    QualityTrainingObservation,
    build_joint_candidates,
    compute_jq_features,
    feature_schema,
    fit_feature_scaler,
    joint_quality_loss,
    load_checkpoint,
    make_checkpoint,
    save_checkpoint,
    stable_argmax,
)


def _embedding(score):
    return np.asarray([score, math.sqrt(max(0.0, 1.0 - score * score))], dtype=np.float32)


def _metadata():
    return {
        f"{index:09d}.jpg": {
            "blob": f"{index:09d}.jpg", "frame_index": index, "timestamp_s": float(index),
            "clip_embedding": serialize_embedding(_embedding(score)),
        }
        for index, score in enumerate((0.1, 0.2, 0.8, 0.9, 0.7, 0.3, 0.4, 0.1, 0.6, 0.5, 0.2))
    }


def _x1_metadata(shift=0.0):
    rows = {}
    for index in range(20):
        name = f"{index:09d}.jpg"
        score = 0.95 - 0.01 * abs(index - 5 - shift)
        rows[name] = {
            "blob": name, "frame_index": index, "timestamp_s": index * 2.0,
            "clip_embedding": serialize_embedding(
                np.asarray([score, math.sqrt(max(0.0, 1.0 - score * score)), 0, 0, 0, 0, 0, 0]),
            ),
        }
    return rows


def _pool(order=False):
    extent = [
        FrameCandidate("extent-b", 8.0, 10.0, 0.4, (), ()),
        FrameCandidate("extent-a", 2.0, 4.0, 0.8, (), ()),
    ]
    if order:
        extent.reverse()
    return build_joint_candidates((2.0, 4.0), extent, duration_s=12.0)


def test_schema_and_union_are_fixed_and_permutation_deterministic():
    assert feature_schema()["feature_names"] == list(FEATURE_NAMES)
    assert len(FEATURE_NAMES) == 34
    assert _pool(False) == _pool(True)
    pool = _pool()
    assert [row.candidate_id for row in pool] == ["frozen-v2-final", "extent-a", "extent-b"]
    assert [row.raw_rank for row in pool] == [0, 1, 2]
    # Geometrically equal baseline/extent candidates retain both identities.
    assert pool[0].span == pool[1].span


def test_frozen_x1_revision_fixture_is_numerically_equivalent():
    training = []
    for index in range(6):
        training.append(ExtentObservation(
            f"train-{index}", f"group-{index}", tuple(np.eye(8)[0]),
            build_temporal_cell_tokens(_x1_metadata(index % 2), upper_bound_s=40.0),
            40.0, (8.0, 16.0 + index),
        ))
    pipeline = fit_extent_pipeline(training)
    heldout = ExtentObservation(
        "held", "held", tuple(np.eye(8)[0]),
        build_temporal_cell_tokens(_x1_metadata(), upper_bound_s=40.0), 40.0, None,
    )
    candidates, debug = predict_extent_candidates(pipeline, heldout)
    # Numeric LAPACK implementations may differ in their final floating-point bits.
    # Freeze the ranked semantics and predictions rather than hashing raw float reprs.
    assert [row.frame_refs for row in candidates] == [
        ("000000006.jpg",), ("000000007.jpg",), ("000000005.jpg",),
        ("000000008.jpg",), ("000000004.jpg",), ("000000003.jpg",),
        ("000000002.jpg",), ("000000001.jpg",),
    ]
    np.testing.assert_allclose(
        [(row.start_s, row.end_s, row.score) for row in candidates],
        [
            (8.0053838895, 17.5621412632, 0.8764129226),
            (8.0485446431, 17.6830233605, 0.8613719145),
            (7.7724050193, 17.8943816627, 0.8490595105),
            (8.5195850270, 18.3632804457, 0.8402339508),
            (7.5486821064, 18.1269977895, 0.8248791442),
            (6.0, 40.0, 0.3065111794),
            (4.0, 40.0, 0.2920329955),
            (2.0, 40.0, 0.2633675814),
        ],
        rtol=0,
        atol=1e-6,
    )
    assert [row["cell_rank"] for row in debug] == [3, 5, 1, 7, 2, 4, 6, 8]


def test_jq_feature_v1_boundary_open_closed_empty_and_normalization():
    rows = compute_jq_features(
        _pool(), _metadata(), (1.0, 0.0), duration_s=12.0, rho=0.5,
        budget_bytes=8 * 1024 * 1024,
    )
    baseline = rows[0].as_mapping()
    # Inside is [2,4], left is [0,2), and right is (4,6].
    assert baseline["inside_mean"] == pytest.approx((0.8 + 0.9 + 0.7) / 3, abs=2e-3)
    assert baseline["left_mean"] == pytest.approx(0.15, abs=2e-3)
    assert baseline["right_mean"] == pytest.approx(0.35, abs=2e-3)
    assert baseline["start_norm"] == pytest.approx(2 / 12)
    assert baseline["end_norm"] == pytest.approx(4 / 12)
    assert baseline["support_norm"] == pytest.approx(3 / 11)
    assert baseline["is_frozen_baseline"] == 1
    assert baseline["raw_rank_norm"] == 0
    assert baseline["is_8mib"] == 1
    empty = compute_jq_features(
        build_joint_candidates((11.5, 12.0), (), duration_s=12.0), _metadata(), (1.0, 0.0),
        duration_s=12.0, rho=0.25, budget_bytes=1024 * 1024,
    )[0].as_mapping()
    assert empty["inside_missing"] == 1
    assert empty["inside_mean"] == empty["pooled_cos"] == 0
    assert empty["right_missing"] == 1


@pytest.mark.parametrize("width", [2.0, 8.0, 16.0, 32.0])
def test_synthetic_event_widths_remain_finite(width):
    duration = 40.0
    start = 4.0
    pool = build_joint_candidates(
        (start, start + width),
        [FrameCandidate("extent-near-tie", start + .25, start + width, .800001, (), ())],
        duration_s=duration,
    )
    features = compute_jq_features(
        pool, _metadata(), (1, 0), duration_s=duration, rho=.5,
        budget_bytes=8 * 1024 * 1024,
    )
    assert np.isfinite(np.asarray([row.values for row in features])).all()


def test_invalid_geometry_dimensions_and_forbidden_fields_fail_closed():
    with pytest.raises(ValueError, match="invalid"):
        build_joint_candidates((4, 2), (), duration_s=12)
    with pytest.raises(ValueError, match="invalid"):
        build_joint_candidates(
            (2, 4), [FrameCandidate("future", 8, 10, .5, (), ())],
            duration_s=12, upper_bound_s=6,
        )
    broken = _metadata()
    broken["x.jpg"] = {
        "timestamp_s": 1, "frame_index": 1,
        "clip_embedding": serialize_embedding(np.asarray([1.0, 0.0, 0.0])),
    }
    with pytest.raises(ValueError, match="dimensions"):
        compute_jq_features(_pool(), broken, (1, 0), duration_s=12, rho=.5, budget_bytes=1)
    leaked = _metadata()
    leaked["000000000.jpg"]["gt_span"] = [1, 2]
    with pytest.raises(ValueError, match="forbidden"):
        compute_jq_features(_pool(), leaked, (1, 0), duration_s=12, rho=.5, budget_bytes=1)


def test_scaler_excludes_binary_fields_and_heldout_identities():
    first = tuple(float(index) for index in range(len(FEATURE_NAMES)))
    second = tuple(float(index + 2) for index in range(len(FEATURE_NAMES)))
    row = QualityTrainingObservation("o", "v", "g", (first, second), (0.1, 0.9), ("a", "b"))
    scaler = fit_feature_scaler([row], heldout_video_ids=["held"], heldout_group_ids=["held"])
    for name in ("is_frozen_baseline", "is_8mib", "inside_missing", "left_missing", "right_missing"):
        index = FEATURE_NAMES.index(name)
        assert scaler.mean[index] == 0
        assert scaler.scale[index] == 1
    with pytest.raises(ValueError, match="held-out"):
        fit_feature_scaler([row], heldout_video_ids=["v"])


def test_loss_is_candidate_permutation_invariant_and_rewards_better_ordering():
    import torch

    targets = torch.tensor([0.1, 0.5, 0.9])
    logits = torch.tensor([-1.0, 0.0, 1.0])
    base = joint_quality_loss([logits], [targets])
    permutation = torch.tensor([2, 0, 1])
    permuted = joint_quality_loss([logits[permutation]], [targets[permutation]])
    worse = joint_quality_loss([torch.tensor([1.0, 0.0, -1.0])], [targets])
    assert float(base) == pytest.approx(float(permuted), abs=1e-7)
    assert float(base) < float(worse)
    assert stable_argmax([1, 1], ["candidate-b", "candidate-a"]) == 1


def test_checkpoint_schema_hash_and_state_are_verified():
    scaler = FeatureScaler(
        (0.0,) * len(FEATURE_NAMES), (1.0,) * len(FEATURE_NAMES), "a" * 64, "b" * 64,
    )
    checkpoint = make_checkpoint(
        JointQualityHead(), scaler, outer_train_video_sha256="a" * 64,
        outer_train_group_sha256="b" * 64, code_revision="c" * 40, training_metadata={},
    )
    with pytest.raises(ValueError, match="forbidden"):
        make_checkpoint(
            JointQualityHead(), scaler, outer_train_video_sha256="a" * 64,
            outer_train_group_sha256="b" * 64, code_revision="c" * 40,
            training_metadata={"source": "must-not-enter-checkpoint"},
        )
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / "checkpoint.json"
        save_checkpoint(checkpoint, path)
        _, loaded_scaler, payload = load_checkpoint(path)
        assert loaded_scaler == scaler
        assert payload["feature_schema"]["version"] == "jq_feature_v1"
        text = path.read_text().replace("jq_feature_v1", "jq_feature_v2")
        path.write_text(text)
        path.with_suffix(".json.sha256").write_text(__import__("hashlib").sha256(path.read_bytes()).hexdigest() + "\n")
        with pytest.raises(ValueError, match="payload SHA"):
            load_checkpoint(path)


def test_checkpoint_rejects_extra_or_forbidden_fields_even_with_valid_hashes(tmp_path):
    scaler = FeatureScaler(
        (0.0,) * len(FEATURE_NAMES), (1.0,) * len(FEATURE_NAMES), "a" * 64, "b" * 64,
    )
    checkpoint = make_checkpoint(
        JointQualityHead(), scaler, outer_train_video_sha256="a" * 64,
        outer_train_group_sha256="b" * 64, code_revision="c" * 40, training_metadata={},
    )
    checkpoint["unexpected"] = "value"
    payload = dict(checkpoint)
    payload.pop("checkpoint_sha256")
    checkpoint["checkpoint_sha256"] = hashlib.sha256(
        json.dumps(
            payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    path = tmp_path / "checkpoint.json"
    save_checkpoint(checkpoint, path)
    with pytest.raises(ValueError, match="fields changed"):
        load_checkpoint(path)
