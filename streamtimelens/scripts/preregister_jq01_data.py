#!/usr/bin/env python3
"""Freeze a reusable consumed development benchmark and an unseen D-lock."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

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
from streamtimelens.retrieval.joint_quality import schema_sha256, write_feature_schema


BASE_FIELDS = {
    "video_id", "group_id", "query_id", "source", "duration_bucket", "content_sha256",
}
FORBIDDEN_WORDS = {
    "gt", "annotation", "span", "iou", "effect", "score", "metric", "good_baseline",
}
DEVELOPMENT_ROLE = "development/consumed"
D_LOCK_ROLE = "d_lock"
IMMUTABLE_REGISTRY_CONFIG = {
    "schema_version": 2,
    "stage": "jq01_fixed_development_and_d_lock",
    "method": "JQ-01",
    "minimum_d_lock_videos_per_source_duration_cell": 8,
    "development_outer_folds": 4,
    "identity_fields": ["video_id", "group_id", "query_id", "content_sha256"],
    "preserve_development_field": "outer_fold",
    "cell_fields": ["source", "duration_bucket"],
    "development_sources": ["C1", "F1", "X1", "T1", "E2"],
    "development_role": DEVELOPMENT_ROLE,
    "d_lock_role": D_LOCK_ROLE,
    "exclude_from_development": ["formal"],
    "selection_seed": 20260911,
    "development_annotation_read_allowed": True,
    "d_lock_annotation_read_allowed": False,
    "development_repeated_runs_allowed": True,
    "formal_data_allowed": False,
}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _valid_sha(value: object) -> bool:
    text = str(value)
    return len(text) == 64 and all(character in "0123456789abcdef" for character in text)


def _validate_registry_config(config: dict) -> None:
    if set(config) != set(IMMUTABLE_REGISTRY_CONFIG):
        raise ValueError(
            f"JQ-01 data-role config fields changed: "
            f"{sorted(set(config) ^ set(IMMUTABLE_REGISTRY_CONFIG))}"
        )
    changed = {
        key: (config.get(key), expected)
        for key, expected in IMMUTABLE_REGISTRY_CONFIG.items()
        if config.get(key) != expected
    }
    if changed:
        raise ValueError(f"JQ-01 immutable data-role config changed: {changed}")


def read_inventory(
    path: Path, *, label: str, require_outer_fold: bool = False,
) -> list[dict[str, Any]]:
    """Read identity-only inventory without opening annotations or effect fields."""
    rows = [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise ValueError(f"JQ-01 {label} inventory is empty")
    normalized = []
    identities = set()
    query_ids = set()
    video_contract: dict[str, tuple[str, str, str, str]] = {}
    content_contract: dict[str, str] = {}
    group_cells: dict[str, tuple[str, str]] = {}
    for raw in rows:
        allowed = BASE_FIELDS | ({"outer_fold"} if require_outer_fold else set())
        unknown = set(raw) - allowed
        if unknown or any(
            word in str(key).lower() for key in raw for word in FORBIDDEN_WORDS
        ):
            raise ValueError(f"effect-bearing or unknown {label} fields: {sorted(unknown)}")
        if require_outer_fold and "outer_fold" not in raw:
            raise ValueError("JQ-01 development inventory must preserve outer_fold")
        row: dict[str, Any] = {field: str(raw.get(field, "")) for field in BASE_FIELDS}
        if require_outer_fold:
            row["outer_fold"] = int(raw["outer_fold"])
        if any(not row[field] for field in BASE_FIELDS) or not _valid_sha(row["content_sha256"]):
            raise ValueError(f"JQ-01 {label} identity/cell/content SHA is invalid")
        identity = (row["video_id"], row["group_id"], row["query_id"])
        if identity in identities:
            raise ValueError(f"duplicate JQ-01 {label} query identity")
        identities.add(identity)
        if row["query_id"] in query_ids:
            raise ValueError(f"JQ-01 {label} query_id is not globally unique")
        query_ids.add(row["query_id"])
        video_value = (
            row["group_id"], row["source"], row["duration_bucket"], row["content_sha256"],
        )
        if row["video_id"] in video_contract and video_contract[row["video_id"]] != video_value:
            raise ValueError(f"JQ-01 {label} video contract changes across queries")
        video_contract[row["video_id"]] = video_value
        if (
            row["content_sha256"] in content_contract
            and content_contract[row["content_sha256"]] != row["video_id"]
        ):
            raise ValueError(f"JQ-01 {label} content SHA is reused by renamed videos")
        content_contract[row["content_sha256"]] = row["video_id"]
        cell = (row["source"], row["duration_bucket"])
        if row["group_id"] in group_cells and group_cells[row["group_id"]] != cell:
            raise ValueError(f"JQ-01 {label} group crosses source-duration cells")
        group_cells[row["group_id"]] = cell
        normalized.append(row)
    return normalized


def _identity_set(payload: Mapping[str, Any], name: str) -> set[str]:
    value = payload.get(name, [])
    if not isinstance(value, list):
        raise ValueError(f"exclusion {name} must be a list")
    return set(map(str, value))


def _exclusion_sets(payload: Mapping[str, Any]) -> dict[str, set[str]]:
    allowed = {"video_ids", "group_ids", "query_ids", "content_sha256"}
    if set(payload) != allowed:
        raise ValueError(f"JQ-01 exclusion fields changed: {sorted(set(payload) ^ allowed)}")
    result = {name: _identity_set(payload, name) for name in allowed}
    if any(not _valid_sha(value) for value in result["content_sha256"]):
        raise ValueError("JQ-01 excluded content SHA is invalid")
    return result


def _values(rows: Sequence[Mapping[str, str]], field: str) -> set[str]:
    return {row[field] for row in rows}


def _assert_disjoint(
    left: Sequence[Mapping[str, str]], right: Sequence[Mapping[str, str]], *, label: str,
) -> None:
    for field in ("video_id", "group_id", "query_id", "content_sha256"):
        overlap = _values(left, field) & _values(right, field)
        if overlap:
            raise ValueError(f"JQ-01 {label} overlaps on {field}: {len(overlap)}")


def _folds_for_development(
    rows: Sequence[Mapping[str, Any]], *, folds: int,
) -> dict[str, int]:
    assignment: dict[str, int] = {}
    for row in rows:
        fold = int(row.get("outer_fold", -1))
        if not 0 <= fold < folds:
            raise ValueError("JQ-01 development outer_fold is invalid")
        group = str(row["group_id"])
        if group in assignment and assignment[group] != fold:
            raise ValueError("JQ-01 development group crosses preserved outer folds")
        assignment[group] = fold
    if len(assignment) < folds or set(assignment.values()) != set(range(folds)):
        raise RuntimeError("JQ-01 development benchmark cannot form non-empty whole-group folds")
    return assignment


def build_data_registry(
    development: Sequence[Mapping[str, str]], d_lock_candidates: Sequence[Mapping[str, str]],
    exclusions: Mapping[str, Any], *, folds: int, minimum_d_lock_videos: int, seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build fixed roles; development is reused, while D-lock remains unseen."""
    if folds < 3 or minimum_d_lock_videos <= 0:
        raise ValueError("invalid JQ-01 registry fold or D-lock requirement")
    excluded = _exclusion_sets(exclusions)
    exclusion_field = {
        "video_id": "video_ids", "group_id": "group_ids", "query_id": "query_ids",
        "content_sha256": "content_sha256",
    }
    for field, exclusion_name in exclusion_field.items():
        if _values(development, field) & excluded[exclusion_name]:
            raise ValueError(f"formal/forbidden {field} entered JQ-01 development")
    _assert_disjoint(development, d_lock_candidates, label="development/D-lock candidates")
    folds_by_group = _folds_for_development(development, folds=folds)

    eligible_lock = [
        row for row in d_lock_candidates
        if row["video_id"] not in excluded["video_ids"]
        and row["group_id"] not in excluded["group_ids"]
        and row["query_id"] not in excluded["query_ids"]
        and row["content_sha256"] not in excluded["content_sha256"]
    ]
    videos: dict[str, tuple[str, tuple[str, str]]] = {}
    for row in eligible_lock:
        videos[row["video_id"]] = (
            row["group_id"], (row["source"], row["duration_bucket"]),
        )
    videos_by_cell: dict[tuple[str, str], list[str]] = defaultdict(list)
    for video_id, (_, cell) in videos.items():
        videos_by_cell[cell].append(video_id)
    selected_lock_groups = set()
    lock_cell_counts = {}
    candidate_cells = sorted({
        (row["source"], row["duration_bucket"]) for row in d_lock_candidates
    })
    for cell in candidate_cells:
        cell_videos = videos_by_cell.get(cell, [])
        groups: dict[str, list[str]] = defaultdict(list)
        for video_id in cell_videos:
            groups[videos[video_id][0]].append(video_id)
        ordered_groups = sorted(
            groups,
            key=lambda group: hashlib.sha256(
                f"{seed}:d-lock:{cell}:{group}".encode("utf-8"),
            ).hexdigest(),
        )
        count = 0
        for group in ordered_groups:
            if count >= minimum_d_lock_videos:
                break
            selected_lock_groups.add(group)
            count += len(groups[group])
        if count < minimum_d_lock_videos:
            raise RuntimeError(
                f"JQ-01 D-lock cell {cell} has {count} videos; "
                f"requires {minimum_d_lock_videos}"
            )
        lock_cell_counts["::".join(cell)] = count
    selected_lock = [row for row in eligible_lock if row["group_id"] in selected_lock_groups]
    _assert_disjoint(development, selected_lock, label="frozen development/D-lock")

    d_lock_candidate_audit = []
    for row in sorted(
        d_lock_candidates,
        key=lambda item: (item["source"], item["duration_bucket"], item["video_id"], item["query_id"]),
    ):
        reasons = [
            field for field, exclusion_name in exclusion_field.items()
            if row[field] in excluded[exclusion_name]
        ]
        selected = not reasons and row["group_id"] in selected_lock_groups
        d_lock_candidate_audit.append({
            **{field: row[field] for field in sorted(BASE_FIELDS)},
            "selected": selected,
            "decision": (
                "selected" if selected else
                "excluded_forbidden_identity" if reasons else
                "eligible_not_selected_after_cell_floor"
            ),
            "excluded_identity_fields": reasons,
        })

    selection = [
        {**row, "data_role": DEVELOPMENT_ROLE, "outer_fold": folds_by_group[row["group_id"]]}
        for row in development
    ] + [
        {**row, "data_role": D_LOCK_ROLE, "outer_fold": None}
        for row in selected_lock
    ]
    audit = {
        "schema_version": 2,
        "stage": "jq01_fixed_development_and_d_lock",
        "development_annotation_status": "consumed_allowed_for_repeated_development",
        "d_lock_annotation_opened": False,
        "development": {
            "rows": len(development), "videos": len(_values(development, "video_id")),
            "groups": len(_values(development, "group_id")), "outer_folds": folds,
        },
        "d_lock": {
            "candidate_rows": len(d_lock_candidates), "eligible_rows": len(eligible_lock),
            "selected_rows": len(selected_lock), "videos": len(_values(selected_lock, "video_id")),
            "groups": len(_values(selected_lock, "group_id")), "cell_video_counts": lock_cell_counts,
        },
        "excluded_counts": {name: len(values) for name, values in sorted(excluded.items())},
        "zero_overlap_fields": ["video_id", "group_id", "query_id", "content_sha256"],
        "d_lock_candidate_audit": d_lock_candidate_audit,
    }
    return selection, audit


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development-inventory", type=Path, required=True)
    parser.add_argument("--d-lock-inventory", type=Path, required=True)
    parser.add_argument(
        "--exclusions", type=Path, required=True,
        help="Formal and otherwise forbidden video/group/query/content identities.",
    )
    parser.add_argument(
        "--config", type=Path,
        default=PACKAGE_ROOT / "configs/exploration/jq01_preregistration.yaml",
    )
    parser.add_argument("--output", type=Path, default=PACKAGE_ROOT.parent / "results/jq01/J0")
    args = parser.parse_args()
    metadata = git_metadata()
    if metadata["is_dirty"]:
        raise RuntimeError("JQ-01 preregistration requires a clean Git commit")
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    _validate_registry_config(config)
    development = read_inventory(
        args.development_inventory, label="development", require_outer_fold=True,
    )
    d_lock = read_inventory(args.d_lock_inventory, label="D-lock")
    exclusions = json.loads(args.exclusions.read_text(encoding="utf-8"))
    selection, audit = build_data_registry(
        development, d_lock, exclusions, folds=int(config["development_outer_folds"]),
        minimum_d_lock_videos=int(config["minimum_d_lock_videos_per_source_duration_cell"]),
        seed=int(config["selection_seed"]),
    )
    final_metadata = git_metadata()
    if final_metadata["is_dirty"] or final_metadata["commit"] != metadata["commit"]:
        raise RuntimeError("Git state changed during JQ-01 preregistration")
    output = versioned_result_path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    resolved = {
        **config,
        "development_inventory": str(args.development_inventory.resolve()),
        "development_inventory_sha256": _sha(args.development_inventory),
        "d_lock_inventory": str(args.d_lock_inventory.resolve()),
        "d_lock_inventory_sha256": _sha(args.d_lock_inventory),
        "exclusions": str(args.exclusions.resolve()),
        "exclusions_sha256": _sha(args.exclusions),
        "feature_schema_sha256": schema_sha256(),
    }
    write_feature_schema(output / "feature_schema.json")
    (output / "selection.jsonl").write_text(
        "".join(
            json.dumps(row, sort_keys=True) + "\n"
            for row in sorted(
                selection,
                key=lambda item: (item["data_role"], item["video_id"], item["query_id"]),
            )
        ),
        encoding="utf-8",
    )
    (output / "selection_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    write_provenance(
        output,
        configuration=resolved,
        config_path=args.config,
        config_filename="config.resolved.yaml",
        extra_metadata={
            "input_sha256": {
                "development_inventory": resolved["development_inventory_sha256"],
                "d_lock_inventory": resolved["d_lock_inventory_sha256"],
                "exclusions": resolved["exclusions_sha256"],
            },
            "selection_sha256": _sha(output / "selection.jsonl"),
            "selection_audit_sha256": _sha(output / "selection_audit.json"),
            "feature_schema_sha256": _sha(output / "feature_schema.json"),
        },
    )
    write_artifact_manifest(output)
    print(json.dumps({
        "output": str(output), "development": audit["development"],
        "d_lock": audit["d_lock"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
