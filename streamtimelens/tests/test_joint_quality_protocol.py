import copy
import math
import tempfile
from pathlib import Path

import numpy as np

from streamtimelens.observer.clip_encoder import serialize_embedding
from streamtimelens.protocol.snapshot import SnapshotReader, SnapshotWriter
from streamtimelens.protocol.types import Budget, VideoMeta
from streamtimelens.retrieval.frame_candidates import FrameCandidate
from streamtimelens.retrieval.joint_quality import select_joint_quality, snapshot_fingerprint
from streamtimelens.retrieval.joint_quality import (
    JointQualityHead,
    QualityTrainingObservation,
    build_joint_candidates,
    compute_jq_features,
    fit_feature_scaler,
    make_checkpoint,
    save_checkpoint,
)


def _snapshot(root: Path) -> SnapshotReader:
    metadata = {}
    raw = []
    for index in range(6):
        name = f"{index:09d}.jpg"
        vector = np.asarray([0.9 - index * .05, math.sqrt(1 - (0.9 - index * .05) ** 2)])
        metadata[name] = {
            "blob": name, "frame_index": index, "timestamp_s": float(index * 2),
            "clip_embedding": serialize_embedding(vector),
        }
        raw.append((name, b"fixture" + bytes([index])))
    SnapshotWriter(root).write(
        name="snapshot", t_q=12, meta=VideoMeta("synthetic", 12, 1, 12),
        budget=Budget(1024 * 1024, 1), cards=[], raw_frames=raw,
        raw_metadata=metadata, writer_calls=0, config={"synthetic": True},
    )
    return SnapshotReader(root / "snapshot")


def test_missing_checkpoint_no_extent_nan_and_reader_failure_are_exact_fallbacks():
    baseline = {
        "observation_id": "synthetic", "final_span": [2.0, 6.0], "status": "ok",
        "candidate_ids": ["original"], "nested": {"keep": [1, 2]},
    }
    original = copy.deepcopy(baseline)
    with tempfile.TemporaryDirectory() as temporary:
        reader = _snapshot(Path(temporary))
        before = snapshot_fingerprint(reader)
        result, debug = select_joint_quality(
            baseline, reader, (1, 0),
            [FrameCandidate("extent-a", 1, 3, .8, (), ())],
            checkpoint_path=Path(temporary) / "missing.json", rho=.5,
            budget_bytes=1024 * 1024,
        )
        assert result == original and baseline == original and debug["fallback"]
        assert before == snapshot_fingerprint(SnapshotReader(reader.root))
        result, _ = select_joint_quality(
            baseline, reader, (1, 0), [], checkpoint_path="missing", rho=.5,
            budget_bytes=1024 * 1024,
        )
        assert result == original
        result, _ = select_joint_quality(
            baseline, reader, (1, 0),
            [FrameCandidate("extent-a", 1, 3, float("nan"), (), ())],
            checkpoint_path="missing", rho=.5, budget_bytes=1024 * 1024,
        )
        assert result == original


def test_gt_good_baseline_source_and_original_video_fields_are_rejected():
    with tempfile.TemporaryDirectory() as temporary:
        reader = _snapshot(Path(temporary))
        for forbidden in ("gt_span", "good_baseline", "source", "video_path"):
            baseline = {"final_span": [2.0, 6.0], forbidden: "must-not-enter-selector"}
            result, debug = select_joint_quality(
                baseline, reader, (1, 0),
                [FrameCandidate("extent-a", 1, 3, .8, (), ())],
                checkpoint_path="missing", rho=.5, budget_bytes=1024 * 1024,
            )
            assert result == baseline
            assert "forbidden" in debug["fallback_reason"]


def test_unverified_reader_failure_returns_exact_baseline():
    class BrokenReader:
        pass

    baseline = {"final_span": [2.0, 6.0], "status": "ok"}
    result, debug = select_joint_quality(
        baseline, BrokenReader(), (1, 0),
        [FrameCandidate("extent-a", 1, 3, .8, (), ())], checkpoint_path="missing",
        rho=.5, budget_bytes=1024 * 1024,
    )
    assert result == baseline
    assert debug["fallback"]


def test_valid_checkpoint_selects_successor_and_records_all_candidates():
    import torch

    with tempfile.TemporaryDirectory() as temporary:
        reader = _snapshot(Path(temporary))
        extent = FrameCandidate("extent-a", 0, 10, .8, (), ())
        candidates = build_joint_candidates((2, 6), [extent], duration_s=12)
        features = compute_jq_features(
            candidates, reader.read_frame_metadata(), (1, 0), duration_s=12,
            rho=.5, budget_bytes=1024 * 1024,
        )
        training = QualityTrainingObservation(
            "train", "video-train", "group-train", tuple(row.values for row in features),
            (0.1, 0.9), tuple(row.candidate.candidate_id for row in features),
        )
        scaler = fit_feature_scaler([training])
        model = JointQualityHead(seed=3)
        optimizer = torch.optim.AdamW(model.module.parameters(), lr=.03)
        matrix = torch.as_tensor(scaler.transform(training.features), dtype=torch.float32)
        targets = torch.tensor([-2.0, 2.0])
        for _ in range(100):
            optimizer.zero_grad()
            loss = ((model(matrix) - targets) ** 2).mean()
            loss.backward()
            optimizer.step()
        checkpoint = make_checkpoint(
            model, scaler, outer_train_video_sha256=scaler.fitted_video_sha256,
            outer_train_group_sha256=scaler.fitted_group_sha256,
            code_revision="c" * 40, training_metadata={},
        )
        checkpoint_path = Path(temporary) / "checkpoint.json"
        save_checkpoint(checkpoint, checkpoint_path)
        before = snapshot_fingerprint(reader)
        baseline = {"observation_id": "held", "final_span": [2, 6], "status": "ok"}
        result, debug = select_joint_quality(
            baseline, reader, (1, 0), [extent], checkpoint_path=checkpoint_path,
            rho=.5, budget_bytes=1024 * 1024,
        )
        assert result["final_span"] == [0.0, 10.0]
        assert result["jq01_selected_candidate_id"] == "extent-a"
        assert len(debug["candidates"]) == 2
        assert not debug["fallback"]
        assert before == snapshot_fingerprint(SnapshotReader(reader.root))
