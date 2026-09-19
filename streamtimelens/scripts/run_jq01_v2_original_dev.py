#!/usr/bin/env python3
"""Run leakage-free JQ-01 OOF on Frozen Visual V2's original dev set."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

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
from streamtimelens.evaluation.joint_quality_oof import deterministic_group_folds  # noqa: E402
from streamtimelens.evaluation.joint_quality_v2 import (  # noqa: E402
    JQV2Observation,
    R2_STANDARD_MIOU,
    V2SnapshotFeatures,
    run_v2_nested_oof,
    summarize_v2_nested_oof,
)
from streamtimelens.observer.clip_encoder import FrozenCLIPEncoder  # noqa: E402
from streamtimelens.protocol.snapshot import SnapshotReader  # noqa: E402
from streamtimelens.retrieval.extent_tokens import (  # noqa: E402
    build_temporal_cell_tokens,
    temporal_iou,
)
from streamtimelens.retrieval.joint_quality import (  # noqa: E402
    BOUNDARY_TOKEN_CAP,
    CENTER_TOKEN_CAP,
    TEMPORAL_ARCHITECTURE,
    TEMPORAL_HIDDEN_DIM,
    TEMPORAL_LOSS_WEIGHTS,
    TEMPORAL_SCHEMA_VERSION,
    TEMPORAL_STATE_DIM,
    canonical_json,
    snapshot_fingerprint,
    write_temporal_feature_schema,
)


CONFIG_KEYS = {
    "schema_version", "stage", "method", "data_role", "dataset",
    "queries_path", "queries_sha256", "videos_path", "videos_sha256",
    "model_hashes_path", "model_hashes_sha256", "clip_model_path",
    "clip_model_revision", "clip_model_content_sha256", "clip_device",
    "clip_text_batch_size", "snapshot_root", "snapshot_source_revision",
    "baseline_source_revision", "baseline_source_config_id", "baseline_shards",
    "budget_bytes", "arrival_ratios", "eligibility", "expected_video_count",
    "expected_query_count", "expected_baseline_observation_count",
    "expected_eligible_observation_count", "expected_snapshot_count", "source",
    "duration_bucket", "outer_folds", "outer_unit", "seed", "jq_device",
    "candidate_generator", "x1_generator_source_revision",
    "x1_safety_selector_used", "candidate_cap", "feature_schema",
    "selector", "architecture", "center_token_cap", "boundary_token_cap",
    "temporal_hidden_dim", "temporal_state_dim", "loss_weights",
    "training_weighting", "acceptance_reference_r2_miou",
    "minimum_recoverable_transmission",
    "bootstrap_resamples", "bootstrap_seed", "repeated_revisions_allowed",
    "d_lock_required", "formal_data_allowed", "original_video_read_allowed",
}


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _safe_revision_id(value: str) -> str:
    allowed = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-"
    if not value or len(value) > 80 or Path(value).name != value or any(c not in allowed for c in value):
        raise ValueError("revision ID must be a safe filename component")
    return value


def _validate_config(config: Mapping[str, Any]) -> None:
    if set(config) != CONFIG_KEYS:
        raise ValueError(f"V2 original-dev config fields changed: {sorted(set(config) ^ CONFIG_KEYS)}")
    fixed = {
        "schema_version": 2,
        "stage": "jq01_v2_original_dev_nested_oof",
        "method": "JQ-01",
        "data_role": "development/consumed",
        "dataset": "charades_sta_derived_dev_v1",
        "budget_bytes": 1024 * 1024,
        "arrival_ratios": [0.25, 0.5, 0.75, 1.0],
        "eligibility": "gt_end_le_query_arrival",
        "source": "Charades",
        "duration_bucket": "short",
        "outer_folds": 4,
        "outer_unit": "video_id",
        "jq_device": "cpu",
        "candidate_generator": "nested_x1_crossfit",
        "x1_safety_selector_used": False,
        "candidate_cap": 9,
        "feature_schema": TEMPORAL_SCHEMA_VERSION,
        "selector": "candidate_local_temporal_quality_v1",
        "architecture": TEMPORAL_ARCHITECTURE,
        "center_token_cap": CENTER_TOKEN_CAP,
        "boundary_token_cap": BOUNDARY_TOKEN_CAP,
        "temporal_hidden_dim": TEMPORAL_HIDDEN_DIM,
        "temporal_state_dim": TEMPORAL_STATE_DIM,
        "loss_weights": TEMPORAL_LOSS_WEIGHTS,
        "training_weighting": "video_equal",
        "acceptance_reference_r2_miou": R2_STANDARD_MIOU,
        "minimum_recoverable_transmission": 0.1,
        "repeated_revisions_allowed": True,
        "d_lock_required": False,
        "formal_data_allowed": False,
        "original_video_read_allowed": False,
    }
    changed = {key: (config.get(key), value) for key, value in fixed.items() if config.get(key) != value}
    if changed:
        raise ValueError(f"V2 original-dev protocol changed: {changed}")


def _verified_path(config: Mapping[str, Any], path_key: str, sha_key: str) -> Path:
    path = _resolve(str(config[path_key]))
    if not path.is_file() or _sha(path) != config[sha_key]:
        raise ValueError(f"input identity mismatch: {path_key}")
    return path


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _verify_clip_model(config: Mapping[str, Any], manifest_path: Path) -> dict[str, Any]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    clip = dict(payload["models"]["clip"])
    root = _resolve(str(config["clip_model_path"]))
    if (
        clip["revision"] != config["clip_model_revision"]
        or clip["content_sha256"] != config["clip_model_content_sha256"]
        or not root.is_dir()
    ):
        raise ValueError("frozen CLIP identity changed")
    observed = []
    for expected in clip["files"]:
        path = root / expected["path"]
        row = {"bytes": path.stat().st_size, "path": expected["path"], "sha256": _sha(path)}
        if row != expected:
            raise ValueError(f"frozen CLIP file changed: {expected['path']}")
        observed.append(row)
    content_sha = hashlib.sha256(canonical_json(observed)).hexdigest()
    if content_sha != clip["content_sha256"]:
        raise ValueError("frozen CLIP content SHA changed")
    return {"path": str(root.resolve()), **{key: clip[key] for key in ("revision", "content_sha256", "file_count", "total_bytes")}}


def _encode_queries(
    query_rows: Sequence[Mapping[str, Any]], *, model_path: Path, device: str, batch_size: int,
) -> tuple[dict[str, tuple[float, ...]], list[dict[str, Any]]]:
    encoder = FrozenCLIPEncoder(model_path, device=device, batch_size=batch_size)
    embeddings = {}
    persisted = []
    for row in sorted(query_rows, key=lambda item: str(item["query_id"])):
        query_id = str(row["query_id"])
        vector = encoder.encode_text(str(row["query"]))
        embeddings[query_id] = tuple(float(value) for value in vector)
        persisted.append({"query_id": query_id, "embedding": encoder.persisted(vector)})
    return embeddings, persisted


def _load_baselines(config: Mapping[str, Any]) -> tuple[list[dict[str, Any]], dict[str, str]]:
    rows, hashes = [], {}
    for shard in config["baseline_shards"]:
        if set(shard) != {"path", "sha256"}:
            raise ValueError("baseline shard schema changed")
        path = _resolve(str(shard["path"]))
        digest = _sha(path)
        if digest != shard["sha256"]:
            raise ValueError(f"baseline shard changed: {path}")
        hashes[str(path.resolve())] = digest
        rows.extend(_load_jsonl(path))
    return rows, hashes


def _fingerprint_sha(rows: Sequence[tuple[str, int, str]]) -> str:
    return hashlib.sha256(canonical_json(list(rows))).hexdigest()


def _load_snapshots(
    keys: Sequence[tuple[str, float]], config: Mapping[str, Any],
) -> tuple[dict[str, V2SnapshotFeatures], dict[str, dict[str, Any]]]:
    root = _resolve(str(config["snapshot_root"]))
    snapshots, audit = {}, {}
    for video_id, rho in sorted(set(keys)):
        snapshot_id = f"{video_id}@{rho:.2f}"
        path = root / video_id / "snapshots" / video_id / f"rho_{rho:.2f}"
        reader = SnapshotReader(path)
        before = snapshot_fingerprint(reader)
        duration = float(reader.manifest.video_meta["duration_s"])
        if (
            reader.manifest.video_id != video_id
            or reader.manifest.budget_bytes != int(config["budget_bytes"])
            or abs(reader.manifest.t_q / duration - rho) > 1e-8
        ):
            raise ValueError(f"snapshot identity changed: {snapshot_id}")
        metadata = reader.read_frame_metadata()
        snapshots[snapshot_id] = V2SnapshotFeatures(
            duration, float(reader.manifest.t_q), int(reader.manifest.budget_bytes), metadata,
            build_temporal_cell_tokens(metadata, upper_bound_s=float(reader.manifest.t_q)),
        )
        audit[snapshot_id] = {
            "path": str(path.resolve()),
            "manifest_sha256": _sha(path / "manifest.json"),
            "before_fingerprint_sha256": _fingerprint_sha(before),
            "before_state_bytes": sum(size for _, size, _ in before),
            "manifest_state_bytes": int(reader.manifest.state_bytes),
            "budget_bytes": int(reader.manifest.budget_bytes),
        }
    return snapshots, audit


def _finish_snapshot_audit(audit: dict[str, dict[str, Any]]) -> dict[str, Any]:
    for row in audit.values():
        reader = SnapshotReader(row["path"])
        after = snapshot_fingerprint(reader)
        row["after_fingerprint_sha256"] = _fingerprint_sha(after)
        row["after_state_bytes"] = sum(size for _, size, _ in after)
        row["unchanged"] = row["after_fingerprint_sha256"] == row["before_fingerprint_sha256"]
        row["within_budget"] = (
            row["after_state_bytes"] == row["manifest_state_bytes"]
            and row["after_state_bytes"] <= row["budget_bytes"]
        )
    return {
        "schema_version": 1,
        "snapshot_count": len(audit),
        "snapshots_unchanged": bool(audit) and all(row["unchanged"] for row in audit.values()),
        "byte_audit_complete": bool(audit) and all(row["within_budget"] for row in audit.values()),
        "snapshots": audit,
    }


def _prepare_observations(
    query_rows: Sequence[Mapping[str, Any]], video_rows: Sequence[Mapping[str, Any]],
    baseline_rows: Sequence[Mapping[str, Any]], embeddings: Mapping[str, tuple[float, ...]],
    snapshots: Mapping[str, V2SnapshotFeatures], config: Mapping[str, Any],
) -> tuple[list[JQV2Observation], dict[str, dict[str, Any]]]:
    queries = {str(row["query_id"]): row for row in query_rows}
    videos = {str(row["video_id"]): row for row in video_rows}
    result, baseline_payloads = [], {}
    seen = set()
    for raw in baseline_rows:
        identity = str(raw["sample_id"])
        query_id, video_id = str(raw["query_id"]), str(raw["video_id"])
        rho = round(float(raw["rho_q"]), 2)
        if (
            identity in seen or query_id not in queries or video_id not in videos
            or raw["source_config_id"] != config["baseline_source_config_id"]
            or identity != f"{query_id}@{rho:.2f}"
        ):
            raise ValueError(f"invalid or duplicate Frozen V2 baseline row: {identity}")
        seen.add(identity)
        query, video = queries[query_id], videos[video_id]
        if query["video_id"] != video_id or abs(float(query["duration_s"]) - float(video["duration_s"])) > 1e-6:
            raise ValueError("query/video manifest identity changed")
        snapshot_id = f"{video_id}@{rho:.2f}"
        snapshot = snapshots.get(snapshot_id)
        if snapshot is None or float(query["gt_span"][1]) > snapshot.upper_bound_s + 1e-7:
            continue
        span = tuple(map(float, raw["coarse_span"]))
        gt = tuple(map(float, query["gt_span"]))
        if not 0 <= span[0] <= span[1] <= snapshot.upper_bound_s + 1e-6:
            raise ValueError("Frozen V2 baseline span is outside the snapshot")
        baseline_payloads[identity] = {
            "sample_id": identity, "query_id": query_id, "video_id": video_id,
            "rho_q": rho, "source_config_id": raw["source_config_id"],
            "final_span": list(span), "status": "frozen_visual_v2",
        }
        result.append(JQV2Observation(
            identity, video_id, video_id, query_id, str(video["sha256"]), snapshot_id,
            int(config["budget_bytes"]), rho, str(config["source"]),
            str(config["duration_bucket"]), embeddings[query_id], span, gt,
            temporal_iou(span, gt) >= 0.5,
        ))
    return sorted(result, key=lambda row: row.observation_id), baseline_payloads


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PACKAGE_ROOT / "configs/exploration/jq01_v2_original_dev.yaml")
    parser.add_argument("--revision-id", required=True)
    parser.add_argument("--hypothesis", required=True)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "results/jq01/v2_original_dev")
    args = parser.parse_args()
    revision_id = _safe_revision_id(args.revision_id)
    if not args.hypothesis.strip():
        raise ValueError("a development hypothesis is required")
    metadata = git_metadata()
    if metadata["is_dirty"]:
        raise RuntimeError("V2 original-dev JQ run requires a clean Git commit")
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    _validate_config(config)
    query_path = _verified_path(config, "queries_path", "queries_sha256")
    video_path = _verified_path(config, "videos_path", "videos_sha256")
    model_manifest_path = _verified_path(config, "model_hashes_path", "model_hashes_sha256")
    clip_identity = _verify_clip_model(config, model_manifest_path)
    query_rows, video_rows = _load_jsonl(query_path), _load_jsonl(video_path)
    baseline_rows, baseline_hashes = _load_baselines(config)
    if (
        len(query_rows) != int(config["expected_query_count"])
        or len(video_rows) != int(config["expected_video_count"])
        or len(baseline_rows) != int(config["expected_baseline_observation_count"])
    ):
        raise ValueError("Frozen V2 manifest or baseline count changed")
    eligible_keys = []
    queries = {str(row["query_id"]): row for row in query_rows}
    for row in baseline_rows:
        query = queries[str(row["query_id"])]
        rho = round(float(row["rho_q"]), 2)
        if float(query["gt_span"][1]) <= float(query["duration_s"]) * rho + 1e-7:
            eligible_keys.append((str(row["video_id"]), rho))
    snapshots, raw_snapshot_audit = _load_snapshots(eligible_keys, config)
    if len(snapshots) != int(config["expected_snapshot_count"]):
        raise ValueError("Frozen V2 eligible snapshot count changed")
    embeddings, persisted_embeddings = _encode_queries(
        query_rows, model_path=_resolve(str(config["clip_model_path"])),
        device=str(config["clip_device"]), batch_size=int(config["clip_text_batch_size"]),
    )
    observations, baseline_payloads = _prepare_observations(
        query_rows, video_rows, baseline_rows, embeddings, snapshots, config,
    )
    if len(observations) != int(config["expected_eligible_observation_count"]):
        raise ValueError("Frozen V2 eligible observation count changed")
    folds = deterministic_group_folds(
        [row.group_id for row in observations], folds=int(config["outer_folds"]),
        seed=int(config["seed"]),
    )
    rows, checkpoints, fold_audit = run_v2_nested_oof(
        observations, snapshots, fold_by_group=folds, folds=int(config["outer_folds"]),
        seed=int(config["seed"]), code_revision=metadata["commit"],
        device=str(config["jq_device"]),
    )
    snapshot_audit = _finish_snapshot_audit(raw_snapshot_audit)
    condition_rows, baseline_bit_exact = [], True
    for row in rows:
        baseline = baseline_payloads[row["observation_id"]]
        frozen = copy.deepcopy(baseline)
        jq = copy.deepcopy(baseline)
        if row["selected_candidate_id"] != row["baseline_candidate_id"]:
            jq["final_span"] = list(row["final_span"])
            jq["jq01_selected_candidate_id"] = row["selected_candidate_id"]
            jq["status"] = "jq01_selected"
        else:
            baseline_bit_exact &= canonical_json(jq) == canonical_json(baseline)
        condition_rows.extend((
            {"observation_id": row["observation_id"], "condition": "frozen_visual_v2", "prediction_row": frozen},
            {"observation_id": row["observation_id"], "condition": "JQ-01", "prediction_row": jq},
        ))
    expected_folds = set(range(int(config["outer_folds"])))
    fold_complete = (
        len(rows) == len(observations)
        and {row["observation_id"] for row in rows} == {row.observation_id for row in observations}
        and set(folds.values()) == expected_folds
        and set(map(int, fold_audit["folds"])) == expected_folds
    )
    input_hashes = {
        "queries_sha256": _sha(query_path), "videos_sha256": _sha(video_path),
        "model_hashes_sha256": _sha(model_manifest_path), "baseline_shards": baseline_hashes,
        "config_source_sha256": _sha(args.config),
    }
    summary = summarize_v2_nested_oof(
        rows, snapshots_unchanged=bool(snapshot_audit["snapshots_unchanged"]),
        byte_audit_complete=bool(snapshot_audit["byte_audit_complete"]),
        fold_coverage_complete=fold_complete, baseline_bit_exact=baseline_bit_exact,
        provenance_complete=(not metadata["is_dirty"] and len(metadata["commit"]) == 40),
        seed=int(config["bootstrap_seed"]), resamples=int(config["bootstrap_resamples"]),
    )
    final_metadata = git_metadata()
    if final_metadata["is_dirty"] or final_metadata["commit"] != metadata["commit"]:
        raise RuntimeError("Git state changed during V2 original-dev JQ run")
    output = versioned_result_path(args.output / revision_id)
    output.mkdir(parents=True, exist_ok=False)
    resolved = {
        **config, "revision_id": revision_id, "hypothesis": args.hypothesis.strip(),
        "code_revision": metadata["commit"], "input_sha256": input_hashes,
        "clip_identity": clip_identity, "eligible_observation_count": len(observations),
        "eligible_snapshot_count": len(snapshots),
    }
    write_temporal_feature_schema(output / "feature_schema.json")
    (output / "query_embeddings.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in persisted_embeddings), encoding="utf-8",
    )
    (output / "oof_rows.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True, allow_nan=False) + "\n" for row in rows), encoding="utf-8",
    )
    (output / "condition_rows.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True, allow_nan=False) + "\n" for row in condition_rows), encoding="utf-8",
    )
    for name, payload in (("fold_audit.json", fold_audit), ("snapshot_audit.json", snapshot_audit), ("summary.json", summary)):
        (output / name).write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    checkpoint_hashes = {}
    for fold, checkpoint in checkpoints.items():
        fold_dir = output / f"fold-{fold}"
        fold_dir.mkdir()
        claimed = checkpoint["checkpoint_sha256"]
        payload = dict(checkpoint)
        payload.pop("checkpoint_sha256")
        if claimed != hashlib.sha256(canonical_json(payload)).hexdigest():
            raise RuntimeError("temporal JQ checkpoint SHA changed")
        checkpoint_path = fold_dir / "checkpoint.json"
        checkpoint_path.write_text(
            json.dumps(checkpoint, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        checkpoint_hashes[str(fold)] = _sha(checkpoint_path)
        (fold_dir / "scaler.json").write_text(
            json.dumps({
                "feature_mean": checkpoint["scaler"]["mean"],
                "feature_scale": checkpoint["scaler"]["scale"],
            }, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    write_provenance(
        output, configuration=resolved, config_path=args.config,
        extra_metadata={"input_sha256": input_hashes, "model_checkpoint_sha256": checkpoint_hashes},
    )
    write_artifact_manifest(output)
    print(json.dumps({
        "revision_id": revision_id, "passed": summary["passed"],
        "decision": summary["decision"],
        "v2_standard_miou_delta": summary["v2_standard_metrics"]["delta_miou"],
        "video_equal_miou_delta": summary["miou_bootstrap"]["delta_a_minus_b"],
        "video_equal_miou_ci": [summary["miou_bootstrap"]["ci_low"], summary["miou_bootstrap"]["ci_high"]],
        "r07_delta": summary["r07_bootstrap"]["delta_a_minus_b"], "output": str(output),
    }, sort_keys=True))
    return 0 if summary["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
