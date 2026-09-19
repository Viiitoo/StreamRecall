import copy
import json
import math
from pathlib import Path

import numpy as np
import pytest

from streamtimelens.observer.clip_encoder import serialize_embedding
from streamtimelens.protocol.snapshot import SnapshotReader, SnapshotWriter
from streamtimelens.protocol.types import Budget, VideoMeta
from streamtimelens.retrieval.frame_candidates import FrameCandidate
from streamtimelens.retrieval.late_fusion import (
    ATTENTION_HEADS,
    HIDDEN_DIM,
    MAX_TEMPORAL_TOKENS,
    QueryTimeLateFusionHead,
    build_candidates,
    feature_schema,
    late_fusion_loss,
    load_temporal_tokens,
    make_checkpoint,
    make_training_observation,
    save_checkpoint,
    select_late_fusion,
    snapshot_fingerprint,
    stable_argmax,
    tensorize,
    tensorize_training_observations,
    train_late_fusion_head,
)


def _snapshot(root: Path, count: int = 12, *, future: bool = False) -> SnapshotReader:
    metadata = {}
    raw = []
    for index in reversed(range(count)):
        name = f"{index:09d}.jpg"
        cosine = 0.15 + 0.7 * index / max(count - 1, 1)
        metadata[name] = {
            "blob": name,
            "frame_index": index,
            "timestamp_s": float(30 if future and index == count - 1 else index * 2),
            "clip_embedding": serialize_embedding(np.asarray([
                cosine, math.sqrt(max(0.0, 1.0 - cosine * cosine)),
            ], dtype=np.float32)),
        }
        raw.append((name, f"lf01-{index}".encode("ascii")))
    SnapshotWriter(root, source_revision="lf01-test").write(
        name="snapshot", t_q=24.0,
        meta=VideoMeta("synthetic", 24.0, 1.0, 24),
        budget=Budget(1024 * 1024, 0), cards=[], raw_frames=raw,
        raw_metadata=metadata, writer_calls=0, config={"fixture": "lf01"},
    )
    return SnapshotReader(root / "snapshot")


def _extents(count: int = 2):
    return tuple(
        FrameCandidate(f"extent-{index}", float(index * 2), float(8 + index * 2),
                       0.9 - index * 0.01, (), ())
        for index in range(count)
    )


def _checkpoint(path: Path, embedding_dim: int = 2) -> Path:
    model = QueryTimeLateFusionHead(embedding_dim, seed=7)
    payload = make_checkpoint(
        model, code_revision="0" * 40,
        outer_train_video_sha256="1" * 64,
        outer_train_group_sha256="2" * 64,
    )
    return save_checkpoint(payload, path)


def _training(reader, identifier, video, group, gt=(4.0, 10.0), query_value=(1.0, 0.0)):
    sequence, query = load_temporal_tokens(reader, query_value)
    candidates = build_candidates((4.0, 10.0), _extents(), duration_s=24.0)
    return make_training_observation(
        identifier, video, group, sequence, query, candidates, gt,
    )


def test_schema_freezes_global_coverage_and_compute_cap():
    schema = feature_schema()
    assert schema["temporal_coverage"] == "all-query-visible-tokens-or-fallback"
    assert schema["max_temporal_tokens"] == MAX_TEMPORAL_TOKENS == 256
    assert schema["hidden_dim"] == HIDDEN_DIM == 64
    assert schema["attention_heads"] == ATTENTION_HEADS == 4
    assert schema["max_candidates"] == 9


def test_temporal_tokens_are_all_consumed_in_stable_time_order(tmp_path):
    reader = _snapshot(tmp_path)
    sequence, query = load_temporal_tokens(reader, (1.0, 0.0))
    assert len(sequence.frame_refs) == 12
    assert sequence.timestamps_s == tuple(float(index * 2) for index in range(12))
    assert len(query) == sequence.embedding_dim == 2
    assert max(sequence.timestamps_s) <= reader.manifest.t_q


def test_future_token_and_over_cap_snapshot_fail_closed(tmp_path, monkeypatch):
    reader = _snapshot(tmp_path / "future", future=True)
    with pytest.raises(ValueError, match="temporal token sequence"):
        load_temporal_tokens(reader, (1.0, 0.0))

    reader = _snapshot(tmp_path / "cap")
    metadata = reader.read_frame_metadata()
    one = next(iter(metadata.values()))
    expanded = {
        f"{index:09d}.jpg": {**one, "timestamp_s": float(index % 12)}
        for index in range(MAX_TEMPORAL_TOKENS + 1)
    }
    monkeypatch.setattr(reader, "read_frame_metadata", lambda: expanded)
    with pytest.raises(ValueError, match="above LF-01 cap"):
        load_temporal_tokens(reader, (1.0, 0.0))


def test_late_fusion_emits_token_saliency_and_candidate_quality(tmp_path):
    reader = _snapshot(tmp_path)
    sequence, query = load_temporal_tokens(reader, (1.0, 0.0))
    candidates = build_candidates((4.0, 10.0), _extents(), duration_s=24.0)
    tensors = tensorize(sequence, query, candidates)
    model = QueryTimeLateFusionHead(2, seed=11)
    outputs = model(**tensors)
    assert outputs["attention_weights"].shape == (12,)
    assert outputs["saliency_logits"].shape == (12,)
    assert outputs["candidate_quality_logits"].shape == (3,)
    assert outputs["candidate_gain_logits"].shape == (3,)
    assert outputs["selection_logits"].shape == (3,)
    assert float(outputs["selection_logits"][0]) == 0.0
    assert float(outputs["attention_weights"].sum()) == pytest.approx(1.0, abs=1e-6)
    assert all(bool(np.isfinite(value.detach().numpy()).all()) for value in outputs.values())


def test_candidate_scores_are_permutation_equivariant_and_ties_are_stable(tmp_path):
    reader = _snapshot(tmp_path)
    sequence, query = load_temporal_tokens(reader, (1.0, 0.0))
    left = build_candidates((4.0, 10.0), _extents(), duration_s=24.0)
    right = build_candidates((4.0, 10.0), tuple(reversed(_extents())), duration_s=24.0)
    assert [row.candidate_id for row in left] == [row.candidate_id for row in right]
    model = QueryTimeLateFusionHead(2, seed=13)
    left_scores = model(**tensorize(sequence, query, left))["candidate_quality_logits"]
    right_scores = model(**tensorize(sequence, query, right))["candidate_quality_logits"]
    assert left_scores.detach().numpy() == pytest.approx(right_scores.detach().numpy(), abs=1e-7)
    assert stable_argmax([1.0, 1.0], ["b", "a"]) == 1


def test_valid_inference_preserves_snapshot_and_records_every_token(tmp_path):
    reader = _snapshot(tmp_path / "data")
    checkpoint = _checkpoint(tmp_path / "checkpoint.json")
    baseline = {"observation_id": "o", "final_span": [4.0, 10.0], "status": "ok"}
    before = snapshot_fingerprint(reader)
    result, debug = select_late_fusion(
        baseline, reader, (1.0, 0.0), _extents(8), checkpoint_path=checkpoint,
    )
    assert not debug["fallback"]
    assert len(debug["temporal_tokens"]) == 12
    assert len(debug["candidates"]) == 9
    assert sum(row["selected"] for row in debug["candidates"]) == 1
    assert sorted(row["rank"] for row in debug["candidates"]) == list(range(1, 10))
    assert before == snapshot_fingerprint(SnapshotReader(reader.root))
    if debug["selected_candidate_id"] == "frozen-v2-final":
        assert result == baseline
    else:
        assert result["lf01_selected_candidate_id"] == debug["selected_candidate_id"]


def test_failures_and_forbidden_fields_return_bit_exact_baseline(tmp_path):
    reader = _snapshot(tmp_path / "data")
    checkpoint = _checkpoint(tmp_path / "checkpoint.json")
    baseline = {
        "observation_id": "o", "final_span": [4.0, 10.0], "status": "ok",
        "nested": {"preserve": [1, 2]},
    }
    original = copy.deepcopy(baseline)
    result, debug = select_late_fusion(
        baseline, reader, (1.0, 0.0), _extents(), checkpoint_path=tmp_path / "missing",
    )
    assert result == original and baseline == original and debug["fallback"]

    tampered = json.loads(checkpoint.read_text(encoding="utf-8"))
    tampered["schema"]["max_temporal_tokens"] = 1
    checkpoint.write_text(json.dumps(tampered), encoding="utf-8")
    result, debug = select_late_fusion(
        baseline, reader, (1.0, 0.0), _extents(), checkpoint_path=checkpoint,
    )
    assert result == original and "checkpoint contract mismatch" in debug["fallback_reason"]

    for field in ("gt_span", "good_baseline", "source", "video_path"):
        leaking = {**baseline, field: "forbidden"}
        result, debug = select_late_fusion(
            leaking, reader, (1.0, 0.0), _extents(), checkpoint_path=checkpoint,
        )
        assert result == leaking
        assert "forbidden" in debug["fallback_reason"]


def test_unverified_reader_and_nonfinite_query_fail_closed(tmp_path):
    baseline = {"final_span": [4.0, 10.0], "status": "ok"}
    result, debug = select_late_fusion(
        baseline, object(), (1.0, 0.0), _extents(), checkpoint_path="missing",
    )
    assert result == baseline and debug["fallback"]
    reader = _snapshot(tmp_path)
    result, debug = select_late_fusion(
        baseline, reader, (float("nan"), 0.0), _extents(), checkpoint_path="missing",
    )
    assert result == baseline and debug["fallback"]


def test_training_targets_stay_outside_inference_representation(tmp_path):
    row = _training(_snapshot(tmp_path), "o", "v", "g")
    assert not hasattr(row.sequence, "gt_span")
    assert not hasattr(row.candidates[0], "quality_target")
    assert row.quality_targets[row.baseline_index] == pytest.approx(1.0)
    assert sum(row.saliency_targets) == 4


def test_training_loss_uses_true_negative_video_queries_and_is_finite(tmp_path):
    rows = [
        _training(_snapshot(tmp_path / "a"), "a", "video-a", "group-a"),
        _training(
            _snapshot(tmp_path / "b"), "b", "video-b", "group-b", (8.0, 14.0),
            (0.0, 1.0),
        ),
    ]
    batch = tensorize_training_observations(rows)
    assert np.array_equal(
        batch["model"]["query"][1].numpy(), batch["negative_query"][0].numpy(),
    )
    model = QueryTimeLateFusionHead(2, seed=19)
    loss, components = late_fusion_loss(model, batch)
    assert bool(np.isfinite(float(loss.detach().numpy())))
    assert set(components) == {
        "positive_saliency", "negative_pair_saliency", "quality", "gain", "listwise",
        "relative_regret",
    }


def test_training_rejects_inner_or_outer_identity_leakage(tmp_path):
    train = _training(_snapshot(tmp_path / "a"), "a", "video-a", "group-a")
    validation = _training(_snapshot(tmp_path / "b"), "b", "video-b", "group-b")
    with pytest.raises(ValueError, match="outer-held-out"):
        train_late_fusion_head(
            [train], [validation], max_epochs=100, patience=10,
            heldout_video_ids=["video-b"],
        )
    overlapping = type(validation)(
        validation.observation_id, train.video_id, train.group_id,
        validation.sequence, validation.query, validation.candidates,
        validation.quality_targets, validation.saliency_targets,
    )
    with pytest.raises(ValueError, match="overlap"):
        train_late_fusion_head([train], [overlapping], max_epochs=100, patience=10)
