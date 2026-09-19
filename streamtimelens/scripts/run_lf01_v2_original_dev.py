#!/usr/bin/env python3
"""Run the preregistered LF-01 r1 nested OOF development comparison."""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping

import yaml

# Deterministic cuBLAS must be configured before the query encoder or LF model
# can initialize a CUDA context. This value is also frozen in the resolved YAML.
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = PACKAGE_ROOT.parent
sys.path.insert(0, str(PACKAGE_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from baas.provenance import (  # noqa: E402
    git_metadata,
    versioned_result_path,
    write_artifact_manifest,
    write_provenance,
)
from run_jq01_v2_original_dev import (  # noqa: E402
    _encode_queries,
    _finish_snapshot_audit,
    _load_baselines,
    _load_jsonl,
    _load_snapshots,
    _prepare_observations,
    _resolve,
    _sha,
    _verified_path,
    _verify_clip_model,
)
from streamtimelens.evaluation.joint_quality_oof import deterministic_group_folds  # noqa: E402
from streamtimelens.evaluation.late_fusion_v2 import (  # noqa: E402
    MAX_READOUT_P95_MS,
    R2_STANDARD_MIOU,
    run_lf_v2_nested_oof,
    summarize_lf_v2_oof,
)
from streamtimelens.retrieval.late_fusion import (  # noqa: E402
    ATTENTION_HEADS,
    HIDDEN_DIM,
    LF_ARCHITECTURE,
    LF_SCHEMA_VERSION,
    LOSS_WEIGHTS,
    MAX_CANDIDATES,
    MAX_EXTENT_CANDIDATES,
    MAX_TEMPORAL_TOKENS,
    _canonical_json,
    write_feature_schema,
)


CONFIG_KEYS = {
    "schema_version", "stage", "method", "revision", "parent_registry_path",
    "parent_registry_sha256", "data_role", "dataset",
    "queries_path", "queries_sha256", "videos_path", "videos_sha256",
    "model_hashes_path", "model_hashes_sha256", "clip_model_path",
    "clip_model_revision", "clip_model_content_sha256", "clip_device",
    "clip_text_batch_size", "snapshot_root", "snapshot_source_revision",
    "baseline_source_revision", "baseline_source_config_id", "baseline_shards",
    "budget_bytes", "arrival_ratios", "eligibility", "expected_video_count",
    "expected_query_count", "expected_baseline_observation_count",
    "expected_eligible_observation_count", "expected_snapshot_count", "source",
    "duration_bucket", "outer_folds", "outer_unit", "inner_validation", "seed",
    "lf_device", "cublas_workspace_config", "candidate_generator", "x1_generator_source_revision",
    "x1_safety_selector_used", "candidate_cap", "extent_candidate_cap",
    "feature_schema", "architecture", "max_temporal_tokens", "hidden_dim",
    "attention_heads", "loss_weights", "negative_pairing", "optimizer",
    "learning_rate", "weight_decay", "max_epochs", "patience", "epoch_selection",
    "training_weighting", "acceptance_reference_jq_r2_miou",
    "minimum_recoverable_transmission", "maximum_readout_p95_ms",
    "bootstrap_resamples", "bootstrap_seed", "d_lock_required",
    "formal_data_allowed", "original_video_read_allowed",
}


def _validate_config(config: Mapping[str, Any]) -> None:
    if set(config) != CONFIG_KEYS:
        raise ValueError(f"LF-01 r1 config fields changed: {sorted(set(config) ^ CONFIG_KEYS)}")
    fixed = {
        "schema_version": 1,
        "stage": "lf01_v2_original_dev_nested_oof",
        "method": "LF-01",
        "revision": "v2-original-r2",
        "data_role": "development/consumed",
        "dataset": "charades_sta_derived_dev_v1",
        "budget_bytes": 1024 * 1024,
        "arrival_ratios": [0.25, 0.5, 0.75, 1.0],
        "eligibility": "gt_end_le_query_arrival",
        "source": "Charades",
        "duration_bucket": "short",
        "outer_folds": 4,
        "outer_unit": "video_id",
        "inner_validation": "next_outer_fold",
        "cublas_workspace_config": ":4096:8",
        "candidate_generator": "nested_x1_crossfit",
        "x1_safety_selector_used": False,
        "candidate_cap": MAX_CANDIDATES,
        "extent_candidate_cap": MAX_EXTENT_CANDIDATES,
        "feature_schema": LF_SCHEMA_VERSION,
        "architecture": LF_ARCHITECTURE,
        "max_temporal_tokens": MAX_TEMPORAL_TOKENS,
        "hidden_dim": HIDDEN_DIM,
        "attention_heads": ATTENTION_HEADS,
        "loss_weights": LOSS_WEIGHTS,
        "negative_pairing": "deterministic_true_cross_video",
        "optimizer": "AdamW",
        "learning_rate": 0.001,
        "weight_decay": 0.0001,
        "max_epochs": 100,
        "patience": 10,
        "epoch_selection": "video_equal_candidate_miou",
        "training_weighting": "video_equal",
        "acceptance_reference_jq_r2_miou": R2_STANDARD_MIOU,
        "minimum_recoverable_transmission": 0.2,
        "maximum_readout_p95_ms": MAX_READOUT_P95_MS,
        "d_lock_required": False,
        "formal_data_allowed": False,
        "original_video_read_allowed": False,
    }
    changed = {key: (config.get(key), value) for key, value in fixed.items() if config.get(key) != value}
    if changed:
        raise ValueError(f"LF-01 r1 immutable contract changed: {changed}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path,
        default=PACKAGE_ROOT / "configs/exploration/lf01_v2_original_dev.yaml",
    )
    parser.add_argument(
        "--hypothesis", required=True,
        help="Preregistered development hypothesis recorded verbatim in provenance.",
    )
    parser.add_argument(
        "--output", type=Path,
        default=PROJECT_ROOT / "results/lf01/v2_original_dev",
    )
    args = parser.parse_args()
    if not args.hypothesis.strip():
        raise ValueError("LF-01 r1 hypothesis is required")
    metadata = git_metadata()
    if metadata["is_dirty"]:
        raise RuntimeError("LF-01 r1 effect run requires a clean Git commit")
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    _validate_config(config)
    query_path = _verified_path(config, "queries_path", "queries_sha256")
    video_path = _verified_path(config, "videos_path", "videos_sha256")
    model_manifest_path = _verified_path(config, "model_hashes_path", "model_hashes_sha256")
    parent_registry_path = _verified_path(
        config, "parent_registry_path", "parent_registry_sha256",
    )
    clip_identity = _verify_clip_model(config, model_manifest_path)
    query_rows, video_rows = _load_jsonl(query_path), _load_jsonl(video_path)
    baseline_rows, baseline_hashes = _load_baselines(config)
    if (
        len(query_rows) != int(config["expected_query_count"])
        or len(video_rows) != int(config["expected_video_count"])
        or len(baseline_rows) != int(config["expected_baseline_observation_count"])
    ):
        raise ValueError("LF-01 fixed input counts changed")
    queries = {str(row["query_id"]): row for row in query_rows}
    eligible_keys = []
    for row in baseline_rows:
        query = queries[str(row["query_id"])]
        rho = round(float(row["rho_q"]), 2)
        if float(query["gt_span"][1]) <= float(query["duration_s"]) * rho + 1e-7:
            eligible_keys.append((str(row["video_id"]), rho))
    snapshots, raw_snapshot_audit = _load_snapshots(eligible_keys, config)
    if len(snapshots) != int(config["expected_snapshot_count"]):
        raise ValueError("LF-01 fixed snapshot count changed")
    embeddings, persisted_embeddings = _encode_queries(
        query_rows, model_path=_resolve(str(config["clip_model_path"])),
        device=str(config["clip_device"]), batch_size=int(config["clip_text_batch_size"]),
    )
    observations, baseline_payloads = _prepare_observations(
        query_rows, video_rows, baseline_rows, embeddings, snapshots, config,
    )
    if len(observations) != int(config["expected_eligible_observation_count"]):
        raise ValueError("LF-01 eligible observation count changed")
    folds = deterministic_group_folds(
        [row.group_id for row in observations], folds=int(config["outer_folds"]),
        seed=int(config["seed"]),
    )
    rows, checkpoints, fold_audit = run_lf_v2_nested_oof(
        observations, snapshots, fold_by_group=folds, folds=int(config["outer_folds"]),
        seed=int(config["seed"]), code_revision=metadata["commit"],
        device=str(config["lf_device"]),
    )
    snapshot_audit = _finish_snapshot_audit(raw_snapshot_audit)
    condition_rows, baseline_bit_exact = [], True
    for row in rows:
        baseline = baseline_payloads[row["observation_id"]]
        frozen, lf = copy.deepcopy(baseline), copy.deepcopy(baseline)
        if row["selected_candidate_id"] != row["baseline_candidate_id"]:
            lf["final_span"] = list(row["final_span"])
            lf["lf01_selected_candidate_id"] = row["selected_candidate_id"]
            lf["status"] = "lf01_selected"
        else:
            baseline_bit_exact &= _canonical_json(lf) == _canonical_json(baseline)
        condition_rows.extend((
            {"observation_id": row["observation_id"], "condition": "frozen_visual_v2", "prediction_row": frozen},
            {"observation_id": row["observation_id"], "condition": "LF-01", "prediction_row": lf},
        ))
    expected_folds = set(range(int(config["outer_folds"])))
    fold_complete = (
        len(rows) == len(observations)
        and {row["observation_id"] for row in rows} == {row.observation_id for row in observations}
        and set(folds.values()) == expected_folds
        and set(map(int, fold_audit["folds"])) == expected_folds
    )
    summary = summarize_lf_v2_oof(
        rows, snapshots_unchanged=bool(snapshot_audit["snapshots_unchanged"]),
        byte_audit_complete=bool(snapshot_audit["byte_audit_complete"]),
        fold_coverage_complete=fold_complete, baseline_bit_exact=baseline_bit_exact,
        provenance_complete=(not metadata["is_dirty"] and len(metadata["commit"]) == 40),
        seed=int(config["bootstrap_seed"]), resamples=int(config["bootstrap_resamples"]),
    )
    final_metadata = git_metadata()
    if final_metadata["is_dirty"] or final_metadata["commit"] != metadata["commit"]:
        raise RuntimeError("Git state changed during LF-01 r1 effect run")
    output = versioned_result_path(args.output / str(config["revision"]))
    output.mkdir(parents=True, exist_ok=False)
    input_hashes = {
        "queries_sha256": _sha(query_path),
        "videos_sha256": _sha(video_path),
        "model_hashes_sha256": _sha(model_manifest_path),
        "baseline_shards": baseline_hashes,
        "config_source_sha256": _sha(args.config),
        "parent_registry_sha256": _sha(parent_registry_path),
    }
    resolved = {
        **config, "hypothesis": args.hypothesis.strip(), "code_revision": metadata["commit"],
        "input_sha256": input_hashes, "clip_identity": clip_identity,
        "eligible_observation_count": len(observations),
        "eligible_snapshot_count": len(snapshots),
    }
    write_feature_schema(output / "feature_schema.json")
    (output / "query_embeddings.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in persisted_embeddings),
        encoding="utf-8",
    )
    (output / "oof_rows.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True, allow_nan=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    (output / "condition_rows.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True, allow_nan=False) + "\n" for row in condition_rows),
        encoding="utf-8",
    )
    for name, payload in (
        ("fold_audit.json", fold_audit), ("snapshot_audit.json", snapshot_audit),
        ("summary.json", summary),
    ):
        (output / name).write_text(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    checkpoint_hashes = {}
    for fold, checkpoint in checkpoints.items():
        fold_dir = output / f"fold-{fold}"
        fold_dir.mkdir()
        checkpoint_path = fold_dir / "checkpoint.json"
        checkpoint_path.write_text(
            json.dumps(checkpoint, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        checkpoint_hashes[str(fold)] = _sha(checkpoint_path)
    parent_registry = json.loads(parent_registry_path.read_text(encoding="utf-8"))
    if (
        parent_registry.get("schema_version") != 1
        or parent_registry.get("append_only") is not True
        or not isinstance(parent_registry.get("experiments"), list)
    ):
        raise ValueError("LF-01 parent experiment registry is invalid")
    registry = {
        "schema_version": 1,
        "append_only": True,
        "experiments": [*parent_registry["experiments"], {
            "revision": config["revision"], "hypothesis": args.hypothesis.strip(),
            "code_revision": metadata["commit"], "config_sha256": _sha(args.config),
            "passed": summary["passed"], "decision": summary["decision"],
        }],
    }
    (output / "experiment_registry.json").write_text(
        json.dumps(registry, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    write_provenance(
        output, configuration=resolved, config_path=args.config,
        extra_metadata={"input_sha256": input_hashes, "model_checkpoint_sha256": checkpoint_hashes},
    )
    write_artifact_manifest(output)
    print(json.dumps({
        "revision": config["revision"], "passed": summary["passed"],
        "decision": summary["decision"],
        "standard_miou_delta": summary["standard_metrics"]["delta_miou"],
        "video_equal_miou_delta": summary["miou_bootstrap"]["delta_a_minus_b"],
        "video_equal_miou_ci": [
            summary["miou_bootstrap"]["ci_low"], summary["miou_bootstrap"]["ci_high"],
        ],
        "r07_delta": summary["standard_metrics"]["delta_r07"],
        "readout_p95_ms": summary["readout_latency_ms"]["p95"],
        "output": str(output),
    }, sort_keys=True))
    return 0 if summary["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
