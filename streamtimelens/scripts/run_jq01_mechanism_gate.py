#!/usr/bin/env python3
"""Run the effect-free JQ-01 protocol and deterministic regression gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import yaml

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT / "src"))
sys.path.insert(0, str(PACKAGE_ROOT.parent / "src"))

from baas.provenance import (
    git_metadata,
    versioned_result_path,
    write_artifact_manifest,
    write_provenance,
)
from streamtimelens.observer.clip_encoder import serialize_embedding
from streamtimelens.protocol.snapshot import SnapshotReader, SnapshotWriter
from streamtimelens.protocol.types import Budget, VideoMeta
from streamtimelens.retrieval.frame_candidates import FrameCandidate
from streamtimelens.retrieval.joint_quality import (
    BOUNDARY_TOKEN_CAP,
    CENTER_TOKEN_CAP,
    TEMPORAL_ARCHITECTURE,
    TEMPORAL_HIDDEN_DIM,
    TEMPORAL_LOSS_WEIGHTS,
    TEMPORAL_SCHEMA_VERSION,
    TEMPORAL_STATE_DIM,
    TemporalJointQualityHead,
    _fit_temporal_scaler,
    build_joint_candidates,
    compute_temporal_jq_features,
    make_temporal_checkpoint,
    make_temporal_training_observation,
    save_temporal_checkpoint,
    select_temporal_joint_quality,
    snapshot_fingerprint,
    validate_inference_payload,
    write_temporal_feature_schema,
)


IMMUTABLE_MECHANISM_CONFIG = {
    "schema_version": 2,
    "stage": "jq01_effect_free_mechanism_gate",
    "method": "JQ-01",
    "feature_schema": TEMPORAL_SCHEMA_VERSION,
    "architecture": TEMPORAL_ARCHITECTURE,
    "center_token_cap": CENTER_TOKEN_CAP,
    "boundary_token_cap": BOUNDARY_TOKEN_CAP,
    "temporal_hidden_dim": TEMPORAL_HIDDEN_DIM,
    "temporal_state_dim": TEMPORAL_STATE_DIM,
    "loss_weights": TEMPORAL_LOSS_WEIGHTS,
    "candidate_cap": 9,
    "extent_candidate_cap": 8,
    "boundary_window_s": 2.0,
    "budgets_bytes": [1048576, 8388608],
    "synthetic_event_widths_s": [2, 8, 16, 32],
    "require_snapshot_unchanged": True,
    "require_baseline_bit_exact": True,
    "forbid_effect_data": True,
    "formal_data_allowed": False,
}


def _validate_config(config: dict) -> None:
    if set(config) != set(IMMUTABLE_MECHANISM_CONFIG):
        raise ValueError(
            f"JQ-01 mechanism config fields changed: "
            f"{sorted(set(config) ^ set(IMMUTABLE_MECHANISM_CONFIG))}"
        )
    changed = {
        key: (config.get(key), expected)
        for key, expected in IMMUTABLE_MECHANISM_CONFIG.items()
        if config.get(key) != expected
    }
    if changed:
        raise ValueError(f"JQ-01 immutable mechanism config changed: {changed}")


def _synthetic_candidate_audit() -> dict:
    """Exercise and persist a full nine-candidate snapshot-only inference."""
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        metadata, raw_frames = {}, []
        for index in range(20):
            name = f"{index:09d}.jpg"
            score = 0.95 - 0.02 * abs(index - 7)
            vector = np.asarray([
                score,
                math.sqrt(max(0.0, 1.0 - score * score)),
            ], dtype=np.float32)
            metadata[name] = {
                "blob": name,
                "frame_index": index,
                "timestamp_s": float(index * 2),
                "clip_embedding": serialize_embedding(vector),
            }
            raw_frames.append((name, f"jq01-synthetic-{index}".encode("ascii")))
        SnapshotWriter(root, source_revision="jq01-synthetic").write(
            name="snapshot",
            t_q=40.0,
            meta=VideoMeta("jq01-synthetic", 40.0, 1.0, 40),
            budget=Budget(1024 * 1024, 0),
            cards=[],
            raw_frames=raw_frames,
            raw_metadata=metadata,
            writer_calls=0,
            config={"stage": "jq01_synthetic_mechanism"},
        )
        snapshot = SnapshotReader(root / "snapshot")
        before = snapshot_fingerprint(snapshot)
        extent = tuple(
            FrameCandidate(
                f"extent-{index:016x}",
                float(2 + index * 3),
                float(8 + index * 3),
                float(1.0 - index * 0.01),
                (),
                (),
            )
            for index in range(8)
        )
        candidates = build_joint_candidates((8.0, 16.0), extent, duration_s=40.0)
        features = compute_temporal_jq_features(
            candidates,
            snapshot.read_frame_metadata(),
            (1.0, 0.0),
            duration_s=40.0,
            rho=0.5,
            budget_bytes=1024 * 1024,
        )
        training = make_temporal_training_observation(
            "synthetic-training", "synthetic-video", "synthetic-group",
            features, (20.0, 28.0), duration_s=40.0,
        )
        scaler = _fit_temporal_scaler([training])
        checkpoint = make_temporal_checkpoint(
            TemporalJointQualityHead(seed=20260911),
            scaler,
            outer_train_video_sha256=scaler.fitted_video_sha256,
            outer_train_group_sha256=scaler.fitted_group_sha256,
            code_revision="0" * 40,
            training_metadata={"stage": "synthetic_mechanism"},
        )
        checkpoint_path = save_temporal_checkpoint(checkpoint, root / "checkpoint.json")
        baseline = {"final_span": [8.0, 16.0], "status": "ok"}
        result, debug = select_temporal_joint_quality(
            baseline,
            snapshot,
            (1.0, 0.0),
            extent,
            checkpoint_path=checkpoint_path,
            rho=0.5,
            budget_bytes=1024 * 1024,
        )
        after = snapshot_fingerprint(SnapshotReader(snapshot.root))
        ranks = sorted(int(row["rank"]) for row in debug["candidates"])
        if (
            len(debug["candidates"]) != 9
            or ranks != list(range(1, 10))
            or sum(bool(row["selected"]) for row in debug["candidates"]) != 1
            or before != after
        ):
            raise RuntimeError("synthetic JQ-01 nine-candidate audit failed")
        validate_inference_payload(debug)
        return {
            "schema_version": 2,
            "effect_data_read": False,
            "candidate_count": len(debug["candidates"]),
            "candidate_ids": [row["candidate_id"] for row in debug["candidates"]],
            "checkpoint_sha256": checkpoint["checkpoint_sha256"],
            "baseline_bit_exact_if_selected": (
                result == baseline if debug["selected_candidate_id"] == "frozen-v2-final" else True
            ),
            "snapshot_unchanged": before == after,
            "snapshot_state_bytes": snapshot.manifest.state_bytes,
            "snapshot_budget_bytes": snapshot.manifest.budget_bytes,
            "debug": debug,
        }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PACKAGE_ROOT / "configs/exploration/jq01_mechanism_gate.yaml")
    parser.add_argument("--output", type=Path, default=PACKAGE_ROOT.parent / "results/jq01/J1")
    args = parser.parse_args()
    metadata = git_metadata()
    if metadata["is_dirty"]:
        raise RuntimeError("JQ-01 mechanism gate requires a clean Git commit")
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    _validate_config(config)
    command = [
        sys.executable, "-m", "pytest", "-q", "tests/test_joint_quality.py",
        "tests/test_joint_quality_protocol.py", "tests/test_joint_quality_oof.py",
        "tests/test_joint_quality_mechanism.py", "tests/test_joint_quality_temporal.py",
        "tests/test_jq01_data_policy.py",
    ]
    completed = subprocess.run(command, cwd=PACKAGE_ROOT, text=True, capture_output=True, check=False)
    synthetic_error = None
    try:
        candidate_audit = _synthetic_candidate_audit()
    except Exception as exc:  # retain a failed J1 record instead of losing the run
        candidate_audit = None
        synthetic_error = f"{type(exc).__name__}: {exc}"
    final_metadata = git_metadata()
    if final_metadata["is_dirty"] or final_metadata["commit"] != metadata["commit"]:
        raise RuntimeError("Git state changed during JQ-01 mechanism gate")
    output = versioned_result_path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    if candidate_audit is not None:
        (output / "candidate_debug.json").write_text(
            json.dumps(candidate_audit, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    write_temporal_feature_schema(output / "feature_schema.json")
    passed = completed.returncode == 0 and synthetic_error is None
    report = {
        "schema_version": 2, "stage": "jq01_effect_free_mechanism_gate",
        "effect_data_read": False, "passed": passed,
        "returncode": completed.returncode, "stdout": completed.stdout, "stderr": completed.stderr,
        "synthetic_candidate_audit_error": synthetic_error,
        "assertions": {
            "automated_tests_passed": completed.returncode == 0,
            "nine_candidate_debug_persisted": candidate_audit is not None,
            "snapshot_unchanged": bool(candidate_audit and candidate_audit["snapshot_unchanged"]),
            "snapshot_within_budget": bool(
                candidate_audit
                and candidate_audit["snapshot_state_bytes"] <= candidate_audit["snapshot_budget_bytes"]
            ),
        },
    }
    (output / "mechanism_gate.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_provenance(
        output,
        configuration={
            **config,
            "config_source_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
            "test_command": command,
        },
        config_path=args.config,
        command=command,
        extra_metadata={
            "mechanism_gate_sha256": hashlib.sha256(
                (output / "mechanism_gate.json").read_bytes()
            ).hexdigest(),
            "candidate_debug_sha256": (
                hashlib.sha256((output / "candidate_debug.json").read_bytes()).hexdigest()
                if candidate_audit is not None else None
            ),
        },
    )
    write_artifact_manifest(output)
    print(json.dumps({"passed": report["passed"], "output": str(output)}, sort_keys=True))
    return 0 if passed else (completed.returncode or 2)


if __name__ == "__main__":
    raise SystemExit(main())
