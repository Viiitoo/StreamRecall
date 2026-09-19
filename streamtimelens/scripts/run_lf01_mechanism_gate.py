#!/usr/bin/env python3
"""Run the effect-free LF-01 T0/J1 protocol gate."""

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

from baas.provenance import (  # noqa: E402
    git_metadata,
    versioned_result_path,
    write_artifact_manifest,
    write_provenance,
)
from streamtimelens.observer.clip_encoder import serialize_embedding  # noqa: E402
from streamtimelens.protocol.snapshot import SnapshotReader, SnapshotWriter  # noqa: E402
from streamtimelens.protocol.types import Budget, VideoMeta  # noqa: E402
from streamtimelens.retrieval.frame_candidates import FrameCandidate  # noqa: E402
from streamtimelens.retrieval.late_fusion import (  # noqa: E402
    ATTENTION_HEADS,
    HIDDEN_DIM,
    LF_ARCHITECTURE,
    LF_SCHEMA_VERSION,
    MAX_CANDIDATES,
    MAX_EXTENT_CANDIDATES,
    MAX_TEMPORAL_TOKENS,
    QueryTimeLateFusionHead,
    make_checkpoint,
    save_checkpoint,
    select_late_fusion,
    snapshot_fingerprint,
    write_feature_schema,
)


IMMUTABLE_CONFIG = {
    "schema_version": 1,
    "stage": "lf01_effect_free_t0_j1",
    "method": "LF-01",
    "feature_schema": LF_SCHEMA_VERSION,
    "architecture": LF_ARCHITECTURE,
    "temporal_token_source": "verified_snapshot_frame_metadata",
    "max_temporal_tokens": MAX_TEMPORAL_TOKENS,
    "hidden_dim": HIDDEN_DIM,
    "attention_heads": ATTENTION_HEADS,
    "candidate_cap": MAX_CANDIDATES,
    "extent_candidate_cap": MAX_EXTENT_CANDIDATES,
    "require_all_visible_tokens": True,
    "require_snapshot_unchanged": True,
    "require_baseline_bit_exact_fallback": True,
    "forbid_effect_data": True,
    "formal_data_allowed": False,
    "device": "cpu",
    "seed": 20260911,
}


def _validate_config(config: dict) -> None:
    if config != IMMUTABLE_CONFIG:
        changed = {
            key: (config.get(key), expected)
            for key, expected in IMMUTABLE_CONFIG.items()
            if config.get(key) != expected
        }
        extra = sorted(set(config) - set(IMMUTABLE_CONFIG))
        missing = sorted(set(IMMUTABLE_CONFIG) - set(config))
        raise ValueError(
            f"LF-01 immutable mechanism config changed: {changed}; "
            f"extra={extra}; missing={missing}"
        )


def _synthetic_audit(seed: int) -> dict:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        metadata, frames = {}, []
        for index in range(20):
            name = f"{index:09d}.jpg"
            cosine = 0.2 + 0.7 * index / 19
            metadata[name] = {
                "blob": name,
                "frame_index": index,
                "timestamp_s": float(index * 2),
                "clip_embedding": serialize_embedding(np.asarray([
                    cosine, math.sqrt(max(0.0, 1.0 - cosine * cosine)),
                ], dtype=np.float32)),
            }
            frames.append((name, f"lf01-synthetic-{index}".encode("ascii")))
        SnapshotWriter(root, source_revision="lf01-effect-free").write(
            name="snapshot", t_q=40.0,
            meta=VideoMeta("lf01-synthetic", 40.0, 1.0, 40),
            budget=Budget(1024 * 1024, 0), cards=[], raw_frames=frames,
            raw_metadata=metadata, writer_calls=0,
            config={"stage": "lf01_effect_free_t0_j1"},
        )
        snapshot = SnapshotReader(root / "snapshot")
        before = snapshot_fingerprint(snapshot)
        extents = tuple(
            FrameCandidate(
                f"extent-{index:016x}", float(index * 3), float(8 + index * 3),
                float(1.0 - index * 0.01), (), (),
            )
            for index in range(MAX_EXTENT_CANDIDATES)
        )
        model = QueryTimeLateFusionHead(2, seed=seed)
        checkpoint = make_checkpoint(
            model, code_revision="0" * 40,
            outer_train_video_sha256="1" * 64,
            outer_train_group_sha256="2" * 64,
        )
        checkpoint_path = save_checkpoint(checkpoint, root / "checkpoint.json")
        baseline = {"observation_id": "synthetic", "final_span": [8.0, 16.0], "status": "ok"}
        result, debug = select_late_fusion(
            baseline, snapshot, (1.0, 0.0), extents,
            checkpoint_path=checkpoint_path,
        )
        after = snapshot_fingerprint(SnapshotReader(snapshot.root))
        if debug["fallback"]:
            raise RuntimeError(debug["fallback_reason"])
        if (
            len(debug["temporal_tokens"]) != 20
            or len(debug["candidates"]) != MAX_CANDIDATES
            or sorted(row["rank"] for row in debug["candidates"])
            != list(range(1, MAX_CANDIDATES + 1))
            or sum(bool(row["selected"]) for row in debug["candidates"]) != 1
            or before != after
        ):
            raise RuntimeError("synthetic LF-01 coverage audit failed")
        fallback, fallback_debug = select_late_fusion(
            baseline, snapshot, (1.0, 0.0), extents,
            checkpoint_path=root / "missing.json",
        )
        if fallback != baseline or not fallback_debug["fallback"]:
            raise RuntimeError("synthetic LF-01 bit-exact fallback audit failed")
        return {
            "schema_version": 1,
            "effect_data_read": False,
            "all_visible_tokens_consumed": len(debug["temporal_tokens"]) == len(metadata),
            "candidate_count": len(debug["candidates"]),
            "snapshot_unchanged": before == after,
            "fallback_bit_exact": fallback == baseline,
            "result": result,
            "debug": debug,
        }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path,
        default=PACKAGE_ROOT / "configs/exploration/lf01_mechanism_gate.yaml",
    )
    parser.add_argument(
        "--output", type=Path,
        default=PACKAGE_ROOT.parent / "results/lf01/T0_J1",
    )
    args = parser.parse_args()
    initial_git = git_metadata()
    if initial_git["is_dirty"]:
        raise RuntimeError("LF-01 T0/J1 requires a clean Git commit")
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    _validate_config(config)
    command = [
        sys.executable, "-m", "pytest", "-q",
        "tests/test_late_fusion.py", "tests/test_visual_route.py",
    ]
    completed = subprocess.run(
        command, cwd=PACKAGE_ROOT, text=True, capture_output=True, check=False,
    )
    audit, audit_error = None, None
    try:
        audit = _synthetic_audit(int(config["seed"]))
    except Exception as exc:
        audit_error = f"{type(exc).__name__}: {exc}"
    final_git = git_metadata()
    if final_git["is_dirty"] or final_git["commit"] != initial_git["commit"]:
        raise RuntimeError("Git state changed during LF-01 T0/J1")
    output = versioned_result_path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    if audit is not None:
        (output / "candidate_debug.json").write_text(
            json.dumps(audit, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    write_feature_schema(output / "feature_schema.json")
    passed = completed.returncode == 0 and audit_error is None
    report = {
        "schema_version": 1,
        "stage": "lf01_effect_free_t0_j1",
        "effect_data_read": False,
        "passed": passed,
        "test_returncode": completed.returncode,
        "test_stdout": completed.stdout,
        "test_stderr": completed.stderr,
        "synthetic_audit_error": audit_error,
        "assertions": {
            "automated_tests_passed": completed.returncode == 0,
            "all_visible_tokens_consumed": bool(audit and audit["all_visible_tokens_consumed"]),
            "nine_candidate_trace_persisted": bool(
                audit and audit["candidate_count"] == MAX_CANDIDATES
            ),
            "snapshot_unchanged": bool(audit and audit["snapshot_unchanged"]),
            "fallback_bit_exact": bool(audit and audit["fallback_bit_exact"]),
        },
    }
    report_path = output / "mechanism_gate.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    write_provenance(
        output,
        configuration={
            **config,
            "config_source_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
            "test_command": command,
        },
        config_path=args.config,
        command=[sys.executable, str(Path(__file__).resolve()), "--config", str(args.config)],
        extra_metadata={
            "effect_data_read": False,
            "mechanism_gate_sha256": hashlib.sha256(report_path.read_bytes()).hexdigest(),
        },
    )
    write_artifact_manifest(output)
    print(json.dumps({"passed": passed, "output": str(output)}, sort_keys=True))
    return 0 if passed else (completed.returncode or 2)


if __name__ == "__main__":
    raise SystemExit(main())
