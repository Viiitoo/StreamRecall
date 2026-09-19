"""Readiness audit for the four-dataset SnAG-adapt benchmark."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

import yaml


FORMAL_DATASETS = ("charades", "activitynet", "qvhighlights", "mad")
ARRIVAL_RATIOS = (0.25, 0.5, 0.75, 1.0)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class SnAGDatasetAssets:
    name: str
    annotation: Path | None
    video_root: Path | None
    feature_root: Path | None
    checkpoint: Path | None
    text_assets: Path | None
    upstream_recipe: bool


def _file_record(path: Path | None) -> dict[str, object]:
    if path is None:
        return {"path": None, "available": False, "sha256": None}
    resolved = path.expanduser().resolve(strict=False)
    return {
        "path": str(resolved),
        "available": resolved.is_file(),
        "sha256": sha256_file(resolved) if resolved.is_file() else None,
    }


def _directory_record(path: Path | None) -> dict[str, object]:
    if path is None:
        return {"path": None, "available": False, "file_count": 0}
    resolved = path.expanduser().resolve(strict=False)
    count = sum(child.is_file() for child in resolved.iterdir()) if resolved.is_dir() else 0
    return {"path": str(resolved), "available": resolved.is_dir(), "file_count": count}


def _annotation_counts(path: Path | None) -> dict[str, int]:
    if path is None or not path.is_file():
        return {"videos": 0, "queries": 0, "observations": 0, "eligible": 0}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("TimeLens formal annotation must be keyed by video ID")
    queries = observations = eligible = 0
    for value in payload.values():
        if not isinstance(value, Mapping):
            raise ValueError("invalid TimeLens annotation row")
        duration = float(value["duration"])
        spans = value["spans"]
        texts = value["queries"]
        if len(spans) != len(texts):
            raise ValueError("TimeLens spans and queries are misaligned")
        queries += len(texts)
        for span in spans:
            end = float(span[1])
            for rho in ARRIVAL_RATIOS:
                observations += 1
                eligible += end <= rho * duration
    return {
        "videos": len(payload), "queries": queries,
        "observations": observations, "eligible": eligible,
    }


def audit_snag_assets(assets: Iterable[SnAGDatasetAssets]) -> dict[str, object]:
    rows = []
    names = set()
    for asset in assets:
        if asset.name not in FORMAL_DATASETS or asset.name in names:
            raise ValueError("SnAG asset rows need unique formal dataset names")
        names.add(asset.name)
        annotation = _file_record(asset.annotation)
        videos = _directory_record(asset.video_root)
        features = _directory_record(asset.feature_root)
        checkpoint = _file_record(asset.checkpoint)
        text_assets = _directory_record(asset.text_assets)
        counts = _annotation_counts(asset.annotation)
        video_coverage = (
            int(videos["file_count"]) >= counts["videos"] > 0
            if bool(videos["available"]) else False
        )
        requirements = {
            "annotation": bool(annotation["available"]),
            "video_coverage": video_coverage,
            "sequential_features": bool(features["available"]),
            "checkpoint": bool(checkpoint["available"]),
            "text_assets": bool(text_assets["available"]),
            "dataset_specific_recipe": asset.upstream_recipe,
        }
        ready = all(requirements.values())
        rows.append({
            "dataset": asset.name,
            "counts": counts,
            "assets": {
                "annotation": annotation, "videos": videos, "features": features,
                "checkpoint": checkpoint, "text_assets": text_assets,
            },
            "requirements": requirements,
            "ready_for_formal_inference": ready,
            "status": "ready" if ready else "blocked_missing_assets_or_recipe",
        })
    missing_rows = sorted(set(FORMAL_DATASETS) - names)
    if missing_rows:
        raise ValueError(f"missing SnAG dataset audit rows: {', '.join(missing_rows)}")
    return {
        "schema_version": 1,
        "method": "SnAG-adapt",
        "evaluation_scope": "four-dataset-formal-readiness",
        "arrival_ratios": list(ARRIVAL_RATIOS),
        "datasets": rows,
        "ready_dataset_count": sum(bool(row["ready_for_formal_inference"]) for row in rows),
        "formal_metrics_emitted": False,
        "benchmark_table_update_allowed": all(bool(row["ready_for_formal_inference"]) for row in rows),
    }


def snag_protocol_audit(readiness: Mapping[str, object]) -> dict[str, object]:
    ready = bool(readiness.get("benchmark_table_update_allowed"))
    checks = {
        "online_ingest": "not_run",
        "late_query": "not_run",
        "query_blind_write": "unit_tested",
        "single_pass": "not_run",
        "snapshot_only_immutable": "unit_tested",
        "no_replay_no_future": "unit_tested",
        "actual_byte_budget": "unit_tested",
        "independent_queries": "unit_tested",
        "past_only_vtg": "cohort_counted",
        "revision_provenance": "recorded",
    }
    return {
        "schema_version": 1,
        "method": "SnAG-adapt",
        "passed": ready and all(value == "passed" for value in checks.values()),
        "checks": checks,
        "classification": "not-a-result",
        "reason": "formal inference has not run" if not ready else "runtime protocol audit pending",
    }


@dataclass(frozen=True)
class SnAGFormalStartAssets:
    name: str
    annotation: Path | None
    video_root: Path | None
    frozen_config: Path | None
    checkpoint: Path | None
    dev_effect_gate: Path | None
    dev_protocol_audit: Path | None
    g0_parity_gate: Path | None
    g1_upper_bound_gate: Path | None
    g2_diagnostic_gate: Path | None


def _passed_gate(path: Path | None) -> bool:
    if path is None or not path.is_file():
        return False
    value = json.loads(path.read_text(encoding="utf-8"))
    return bool(value.get("passed"))


def audit_snag_formal_start(
    assets: Iterable[SnAGFormalStartAssets], *, worktree_clean: bool,
) -> dict[str, object]:
    """Check whether each dataset may begin ingest without reading its metrics."""
    rows = []
    names = set()
    for asset in assets:
        if asset.name not in FORMAL_DATASETS or asset.name in names:
            raise ValueError("formal-start rows need unique known dataset names")
        names.add(asset.name)
        annotation = _file_record(asset.annotation)
        videos = _directory_record(asset.video_root)
        config_record = _file_record(asset.frozen_config)
        checkpoint = _file_record(asset.checkpoint)
        counts = _annotation_counts(asset.annotation)
        recipe_frozen = False
        checkpoint_matches = False
        if asset.frozen_config is not None and asset.frozen_config.is_file():
            config = yaml.safe_load(asset.frozen_config.read_text(encoding="utf-8"))
            reader = config.get("reader", {}) if isinstance(config, Mapping) else {}
            expected_path = reader.get("checkpoint") if isinstance(reader, Mapping) else None
            expected_hash = reader.get("checkpoint_sha256") if isinstance(reader, Mapping) else None
            recipe_frozen = (
                config.get("method") == "snag-adapt-pooled-B"
                and config.get("classification") == "strict"
                and bool(config.get("formal_frozen"))
                and bool(expected_path) and bool(expected_hash)
            )
            if recipe_frozen and asset.checkpoint is not None and asset.checkpoint.is_file():
                checkpoint_matches = (
                    Path(expected_path).expanduser().resolve() == asset.checkpoint.expanduser().resolve()
                    and sha256_file(asset.checkpoint) == expected_hash
                )
        requirements = {
            "annotation": bool(annotation["available"]),
            "video_coverage": bool(videos["available"]) and int(videos["file_count"]) >= counts["videos"] > 0,
            "frozen_strict_recipe": recipe_frozen,
            "checkpoint_hash_match": checkpoint_matches,
            "dev_effect_gate": _passed_gate(asset.dev_effect_gate),
            "dev_protocol_audit": _passed_gate(asset.dev_protocol_audit),
            "g0_upstream_parity": _passed_gate(asset.g0_parity_gate),
            "g1_full_store_upper_bound": _passed_gate(asset.g1_upper_bound_gate),
            "g2_three_row_diagnostic": _passed_gate(asset.g2_diagnostic_gate),
            "clean_worktree": worktree_clean,
        }
        ready = all(requirements.values())
        rows.append({
            "dataset": asset.name,
            "counts": counts,
            "assets": {
                "annotation": annotation, "videos": videos,
                "frozen_config": config_record, "checkpoint": checkpoint,
            },
            "requirements": requirements,
            "formal_test_can_start": ready,
            "status": "ready_to_start" if ready else "blocked",
        })
    missing = sorted(set(FORMAL_DATASETS) - names)
    if missing:
        raise ValueError(f"missing formal-start rows: {', '.join(missing)}")
    return {
        "schema_version": 1,
        "method": "SnAG-adapt-pooled-B",
        "scope": "per-dataset-formal-start",
        "datasets": rows,
        "ready_datasets": [row["dataset"] for row in rows if row["formal_test_can_start"]],
        "any_formal_test_can_start": any(row["formal_test_can_start"] for row in rows),
        "four_dataset_release_gate": all(row["formal_test_can_start"] for row in rows),
        "formal_metrics_read": False,
    }
