#!/usr/bin/env python3
"""Run one revision on the fixed, reusable JQ-01 development benchmark."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

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
from streamtimelens.evaluation.joint_quality_oof import (
    JQOOFObservation,
    run_joint_quality_oof,
    summarize_jq01_oof,
)
from streamtimelens.protocol.snapshot import SnapshotReader
from streamtimelens.retrieval.frame_candidates import FrameCandidate
from streamtimelens.retrieval.joint_quality import (
    build_joint_candidates,
    canonical_json,
    compute_jq_features,
    save_checkpoint,
    snapshot_fingerprint,
    validate_inference_payload,
    write_feature_schema,
)


ALLOWED_INPUT_FIELDS = {
    "observation_id", "video_id", "group_id", "query_id", "content_sha256",
    "budget_bytes", "rho", "source",
    "duration_bucket", "good_baseline", "snapshot_path", "query_embedding",
    "baseline_row", "gt_span", "x1_candidates", "outer_fold",
}
DEVELOPMENT_ROLE = "development/consumed"
IMMUTABLE_OOF_CONFIG = {
    "schema_version": 2,
    "stage": "jq01_fixed_development_group_oof",
    "method": "JQ-01",
    "data_role": DEVELOPMENT_ROLE,
    "conditions": ["frozen_visual_v2", "JQ-01"],
    "budgets_bytes": [1048576, 8388608],
    "primary_budget_bytes": 8388608,
    "outer_folds": 4,
    "outer_unit": "group_id",
    "seed": 20260911,
    "device": "cpu",
    "deterministic_algorithms": True,
    "data_sort_key": "observation_id",
    "feature_schema": "jq_feature_v1",
    "candidate_cap": 9,
    "x1_generator_source_revision": "8f8c3f702146ab2f5a4c8a41877f5330a2973f90",
    "x1_safety_selector_used": False,
    "quality_bins": [[0.0, 0.3], [0.3, 0.7], [0.7, 1.0]],
    "quality_loss": "SmoothL1",
    "listwise_loss": "ListNet",
    "target_temperature": 0.1,
    "logit_temperature": 1.0,
    "quality_weight": 1.0,
    "ranking_weight": 1.0,
    "architecture": "layernorm-linear64-gelu-linear32-gelu-linear1",
    "optimizer": "AdamW",
    "learning_rate": 0.001,
    "weight_decay": 0.0001,
    "max_epochs": 100,
    "patience": 10,
    "bootstrap_resamples": 2000,
    "bootstrap_seed": 20260911,
    "formal_data_allowed": False,
    "original_video_read_allowed": False,
    "repeated_revisions_allowed": True,
    "require_complete_fixed_benchmark": True,
    "require_append_only_registry": True,
    "adequate_cell_minimum_videos": 12,
}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _validate_oof_config(config: dict) -> None:
    if set(config) != set(IMMUTABLE_OOF_CONFIG):
        raise ValueError(
            f"JQ-01 development config fields changed: "
            f"{sorted(set(config) ^ set(IMMUTABLE_OOF_CONFIG))}"
        )
    changed = {
        key: (config.get(key), expected)
        for key, expected in IMMUTABLE_OOF_CONFIG.items()
        if config.get(key) != expected
    }
    if changed:
        raise ValueError(f"JQ-01 immutable development comparison config changed: {changed}")


def _load_observations(
    path: Path,
) -> tuple[list[JQOOFObservation], dict[str, int], dict[str, Any], dict[str, dict]]:
    result, folds, baseline_rows = [], {}, {}
    snapshot_rows = []
    seen = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        raw = json.loads(line)
        if set(raw) != ALLOWED_INPUT_FIELDS:
            raise ValueError(f"JQ OOF input fields changed: {sorted(set(raw) ^ ALLOWED_INPUT_FIELDS)}")
        identity = str(raw["observation_id"])
        if identity in seen:
            raise ValueError("duplicate JQ OOF observation")
        seen.add(identity)
        snapshot = SnapshotReader(raw["snapshot_path"])
        before = snapshot_fingerprint(snapshot)
        if not isinstance(raw["budget_bytes"], int) or isinstance(raw["budget_bytes"], bool):
            raise ValueError("JQ OOF budget must be an integer")
        budget = int(raw["budget_bytes"])
        if snapshot.manifest.budget_bytes != budget:
            raise ValueError("JQ OOF budget disagrees with verified snapshot")
        if snapshot.manifest.video_id != str(raw["video_id"]):
            raise ValueError("JQ OOF video identity disagrees with verified snapshot")
        extent_rows = raw["x1_candidates"]
        if any(set(row) != {"candidate_id", "start_s", "end_s", "score", "frame_refs"} for row in extent_rows):
            raise ValueError("JQ OOF X1 candidate fields changed or contain leakage")
        for row in extent_rows:
            candidate_id = str(row["candidate_id"])
            suffix = candidate_id[len("extent-"):] if candidate_id.startswith("extent-") else ""
            if (
                not candidate_id.startswith("extent-") or len(suffix) != 16
                or any(character not in "0123456789abcdef" for character in suffix)
                or not isinstance(row["frame_refs"], list)
                or any(Path(str(ref)).name != str(ref) for ref in row["frame_refs"])
            ):
                raise ValueError("JQ OOF X1 candidate identity is not from the frozen generator")
        extent = tuple(
            FrameCandidate(
                str(row["candidate_id"]), float(row["start_s"]), float(row["end_s"]),
                float(row["score"]), tuple(map(str, row.get("frame_refs", []))),
                tuple(map(str, row.get("frame_refs", []))),
            )
            for row in extent_rows
        )
        duration = float(snapshot.manifest.video_meta["duration_s"])
        rho = float(raw["rho"])
        if abs(rho - float(snapshot.manifest.t_q) / duration) > 1e-8:
            raise ValueError("JQ OOF rho disagrees with verified snapshot arrival")
        if not isinstance(raw["good_baseline"], bool):
            raise ValueError("JQ OOF good_baseline must be boolean")
        gt_span = tuple(map(float, raw["gt_span"]))
        if len(gt_span) != 2 or not 0 <= gt_span[0] < gt_span[1] <= duration + 1e-6:
            raise ValueError("JQ OOF GT span is outside the declared video duration")
        baseline_row = raw["baseline_row"]
        if not isinstance(baseline_row, dict) or "final_span" not in baseline_row:
            raise ValueError("JQ OOF baseline row is invalid")
        validate_inference_payload({
            key: value for key, value in baseline_row.items() if key != "final_span"
        })
        baseline_rows[identity] = copy.deepcopy(baseline_row)
        candidates = build_joint_candidates(
            baseline_row["final_span"], extent, duration_s=duration,
            upper_bound_s=float(snapshot.manifest.t_q),
        )
        features = compute_jq_features(
            candidates, snapshot.read_frame_metadata(), raw["query_embedding"],
            duration_s=duration, rho=rho, budget_bytes=budget,
            upper_bound_s=float(snapshot.manifest.t_q),
        )
        after_reader = SnapshotReader(snapshot.root)
        after = snapshot_fingerprint(after_reader)
        state_bytes = sum(size for _, size, _ in after)
        snapshot_rows.append({
            "observation_id": identity,
            "snapshot_root": str(snapshot.root),
            "budget_bytes": budget,
            "manifest_state_bytes": int(after_reader.manifest.state_bytes),
            "fingerprint_state_bytes": state_bytes,
            "within_budget": state_bytes <= budget,
            "unchanged": before == after,
            "before": [
                {"path": name, "size_bytes": size, "sha256": digest}
                for name, size, digest in before
            ],
            "after": [
                {"path": name, "size_bytes": size, "sha256": digest}
                for name, size, digest in after
            ],
        })
        group_id = str(raw["group_id"])
        fold = int(raw["outer_fold"])
        if group_id in folds and folds[group_id] != fold:
            raise ValueError("JQ OOF group crosses preregistered folds")
        folds[group_id] = fold
        result.append(JQOOFObservation(
            identity, str(raw["video_id"]), group_id, str(raw["query_id"]),
            str(raw["content_sha256"]), budget, rho,
            str(raw["source"]), str(raw["duration_bucket"]), candidates, features,
            gt_span, raw["good_baseline"],
        ))
    snapshot_audit = {
        "schema_version": 1,
        "observation_count": len(snapshot_rows),
        "snapshots_unchanged": bool(snapshot_rows) and all(
            row["unchanged"] for row in snapshot_rows
        ),
        "byte_audit_complete": bool(snapshot_rows) and all(
            row["within_budget"]
            and row["manifest_state_bytes"] == row["fingerprint_state_bytes"]
            for row in snapshot_rows
        ),
        "observations": snapshot_rows,
    }
    return result, folds, snapshot_audit, baseline_rows


def _validate_j0_rows(
    raw_path: Path, observations: list[JQOOFObservation], fold_by_group: dict[str, int],
) -> None:
    selection = {}
    for line in raw_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            key = (str(row["video_id"]), str(row["group_id"]), str(row["query_id"]))
            if key in selection:
                raise ValueError("duplicate identity in JQ-01 J0 selection")
            selection[key] = row
    observed_keys = set()
    for observation in observations:
        key = (observation.video_id, observation.group_id, observation.query_id)
        observed_keys.add(key)
        admitted = selection.get(key)
        if (
            admitted is None or admitted.get("data_role") != DEVELOPMENT_ROLE
            or int(admitted.get("outer_fold", -1)) != fold_by_group[observation.group_id]
            or str(admitted.get("content_sha256")) != observation.content_sha256
        ):
            raise ValueError("JQ-01 OOF observation is not admitted by fixed development selection")
    expected_keys = {
        key for key, row in selection.items() if row.get("data_role") == DEVELOPMENT_ROLE
    }
    if observed_keys != expected_keys:
        raise ValueError("JQ-01 OOF does not cover the complete fixed development benchmark")


def _safe_revision_id(value: str) -> str:
    if (
        not value or len(value) > 80 or Path(value).name != value
        or any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-" for character in value)
    ):
        raise ValueError("JQ-01 revision ID must be a safe filename component")
    return value


def _load_parent_registry(path: Path | None) -> tuple[list[dict], str | None]:
    if path is None:
        return [], None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if set(payload) != {"schema_version", "records"} or payload["schema_version"] != 1:
        raise ValueError("unsupported JQ-01 experiment registry")
    records = payload["records"]
    if not isinstance(records, list):
        raise ValueError("JQ-01 experiment registry records must be a list")
    revisions = set()
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("invalid JQ-01 experiment record")
        candidate = dict(record)
        claimed = candidate.pop("record_sha256", None)
        if claimed != hashlib.sha256(canonical_json(candidate)).hexdigest():
            raise ValueError("JQ-01 parent experiment record SHA mismatch")
        revision = str(record.get("revision_id", ""))
        if not revision or revision in revisions:
            raise ValueError("JQ-01 experiment registry revision is empty or duplicated")
        revisions.add(revision)
    return records, _sha(path)


def build_experiment_registry(
    parent_records: list[dict], *, revision_id: str, hypothesis: str,
    code_revision: str, config_sha256: str, observations_sha256: str,
    j0_selection_sha256: str, result_path: str, summary: dict,
) -> dict:
    """Extend an immutable registry snapshot without modifying prior results."""
    revision_id = _safe_revision_id(revision_id)
    if not hypothesis.strip():
        raise ValueError("JQ-01 revision hypothesis is required")
    if revision_id in {str(record["revision_id"]) for record in parent_records}:
        raise ValueError("JQ-01 revision already exists in parent experiment registry")
    if any(
        record.get("observations_sha256") != observations_sha256
        or record.get("j0_selection_sha256") != j0_selection_sha256
        for record in parent_records
    ):
        raise ValueError("JQ-01 revision changed the fixed development benchmark")
    record = {
        "revision_id": revision_id, "hypothesis": hypothesis.strip(),
        "code_revision": code_revision, "config_sha256": config_sha256,
        "observations_sha256": observations_sha256,
        "j0_selection_sha256": j0_selection_sha256, "result_path": result_path,
        "passed": bool(summary["passed"]), "decision": str(summary["decision"]),
        "gate_checks": dict(summary["gate_checks"]),
    }
    record["record_sha256"] = hashlib.sha256(canonical_json(record)).hexdigest()
    return {"schema_version": 1, "records": [*parent_records, record]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observations", type=Path, required=True)
    parser.add_argument("--x1-provenance", type=Path, required=True)
    parser.add_argument("--j0-selection", type=Path, required=True)
    parser.add_argument("--revision-id", required=True)
    parser.add_argument("--hypothesis", required=True)
    history = parser.add_mutually_exclusive_group(required=True)
    history.add_argument("--first-revision", action="store_true")
    history.add_argument("--parent-registry", type=Path)
    parser.add_argument("--config", type=Path, default=PACKAGE_ROOT / "configs/exploration/jq01_oof.yaml")
    parser.add_argument("--output", type=Path, default=PACKAGE_ROOT.parent / "results/jq01/J2")
    args = parser.parse_args()
    revision_id = _safe_revision_id(args.revision_id)
    metadata = git_metadata()
    if metadata["is_dirty"]:
        raise RuntimeError("JQ-01 OOF requires a clean Git commit")
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    _validate_oof_config(config)
    parent_records, parent_registry_sha = _load_parent_registry(args.parent_registry)
    x1_provenance = json.loads(args.x1_provenance.read_text(encoding="utf-8"))
    expected_x1 = {
        "generator_source_revision": config["x1_generator_source_revision"],
        "artifact_sha256": _sha(args.observations), "top_k": 8,
        "safety_selector_used": False,
    }
    if x1_provenance != expected_x1:
        raise ValueError("JQ-01 X1 candidate provenance does not match the frozen generator")
    observations, fold_by_group, snapshot_audit, baseline_rows = _load_observations(
        args.observations,
    )
    _validate_j0_rows(args.j0_selection, observations, fold_by_group)
    rows, checkpoints, fold_audit = run_joint_quality_oof(
        observations, folds=int(config["outer_folds"]), seed=int(config["seed"]),
        code_revision=metadata["commit"], fold_by_group=fold_by_group,
    )
    condition_rows = []
    baseline_bit_exact = True
    for row in rows:
        baseline = copy.deepcopy(baseline_rows[row["observation_id"]])
        baseline_bytes = canonical_json(baseline_rows[row["observation_id"]])
        baseline_bit_exact &= canonical_json(baseline) == baseline_bytes
        jq_row = copy.deepcopy(baseline)
        selected = next(
            candidate for candidate in row["candidate_rows"] if candidate["selected"]
        )
        if row["selected_candidate_id"] != row["baseline_candidate_id"]:
            jq_row["final_span"] = selected["span"]
            jq_row["jq01_selected_candidate_id"] = row["selected_candidate_id"]
        else:
            baseline_bit_exact &= canonical_json(jq_row) == baseline_bytes
        condition_rows.extend((
            {"observation_id": row["observation_id"], "condition": "frozen_visual_v2", "prediction_row": baseline},
            {"observation_id": row["observation_id"], "condition": "JQ-01", "prediction_row": jq_row},
        ))
    expected_folds = set(range(int(config["outer_folds"])))
    fold_coverage_complete = (
        len(rows) == len(observations)
        and {row["observation_id"] for row in rows}
        == {row.observation_id for row in observations}
        and set(fold_by_group.values()) == expected_folds
        and set(map(int, fold_audit["folds"])) == expected_folds
    )
    input_hashes = {
        "observations_sha256": _sha(args.observations),
        "x1_provenance_sha256": _sha(args.x1_provenance),
        "j0_selection_sha256": _sha(args.j0_selection),
        "config_source_sha256": _sha(args.config),
    }
    provenance_complete = all(
        len(value) == 64 and all(character in "0123456789abcdef" for character in value)
        for value in input_hashes.values()
    ) and len(metadata["commit"]) == 40 and not metadata["is_dirty"]
    summary = summarize_jq01_oof(
        rows, snapshots_unchanged=bool(snapshot_audit["snapshots_unchanged"]),
        byte_audit_complete=bool(snapshot_audit["byte_audit_complete"]),
        baseline_bit_exact=baseline_bit_exact,
        provenance_complete=provenance_complete,
        fold_coverage_complete=fold_coverage_complete,
        resamples=int(config["bootstrap_resamples"]), seed=int(config["bootstrap_seed"]),
        adequate_cell_videos=int(config["adequate_cell_minimum_videos"]),
    )
    final_metadata = git_metadata()
    if final_metadata["is_dirty"] or final_metadata["commit"] != metadata["commit"]:
        raise RuntimeError("Git state changed during JQ-01 OOF")
    output = versioned_result_path(args.output / revision_id)
    output.mkdir(parents=True, exist_ok=False)
    resolved = {
        **config, "observations": str(args.observations.resolve()),
        **input_hashes, "code_revision": metadata["commit"],
        "x1_provenance": str(args.x1_provenance.resolve()),
        "j0_selection": str(args.j0_selection.resolve()),
        "revision_id": revision_id, "hypothesis": args.hypothesis.strip(),
        "parent_registry": (
            str(args.parent_registry.resolve()) if args.parent_registry else None
        ),
        "parent_registry_sha256": parent_registry_sha,
        "first_revision": bool(args.first_revision),
    }
    write_feature_schema(output / "feature_schema.json")
    (output / "oof_rows.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True, allow_nan=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    (output / "condition_rows.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True, allow_nan=False) + "\n" for row in condition_rows),
        encoding="utf-8",
    )
    (output / "fold_audit.json").write_text(json.dumps(fold_audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output / "snapshot_audit.json").write_text(
        json.dumps(snapshot_audit, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    checkpoint_files = {}
    for fold, checkpoint in checkpoints.items():
        fold_dir = output / f"fold-{fold}"
        fold_dir.mkdir()
        checkpoint_path = save_checkpoint(checkpoint, fold_dir / "checkpoint.json")
        checkpoint_files[str(fold)] = _sha(checkpoint_path)
        (fold_dir / "scaler.json").write_text(json.dumps(checkpoint["scaler"], indent=2, sort_keys=True) + "\n", encoding="utf-8")
    registry = build_experiment_registry(
        parent_records, revision_id=revision_id, hypothesis=args.hypothesis,
        code_revision=metadata["commit"],
        config_sha256=hashlib.sha256(canonical_json(resolved)).hexdigest(),
        observations_sha256=input_hashes["observations_sha256"],
        j0_selection_sha256=input_hashes["j0_selection_sha256"],
        result_path=str(output), summary=summary,
    )
    registry_path = output / "experiment_registry.json"
    registry_path.write_text(
        json.dumps(registry, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    write_provenance(
        output, configuration=resolved, config_path=args.config,
        extra_metadata={
            "input_sha256": input_hashes,
            "model_source_revision": config["x1_generator_source_revision"],
            "model_checkpoint_sha256": checkpoint_files,
            "experiment_registry_sha256": _sha(registry_path),
            "snapshot_audit_sha256": _sha(output / "snapshot_audit.json"),
        },
    )
    write_artifact_manifest(output)
    print(json.dumps({
        "revision_id": revision_id, "passed": summary["passed"],
        "decision": summary["decision"], "output": str(output),
    }, sort_keys=True))
    return 0 if summary["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
