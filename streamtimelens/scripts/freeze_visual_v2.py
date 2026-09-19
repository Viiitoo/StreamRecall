#!/usr/bin/env python3
"""Freeze independent-dev candidate-margin winners as frozen_visual_v2."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import yaml

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT / "src"))

from streamtimelens.config import read_resolved_config
from streamtimelens.evaluation.freeze import freeze_visual_configuration


def _hash(value) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--protocol-audit", type=Path, required=True)
    parser.add_argument("--refiner-gate", type=Path, required=True)
    parser.add_argument("--matrix-root", type=Path, required=True)
    parser.add_argument(
        "--output", type=Path,
        default=PACKAGE_ROOT / "configs" / "frozen_visual_v2.yaml",
    )
    parser.add_argument(
        "--decision-record", type=Path,
        default=PACKAGE_ROOT / "docs" / "frozen_visual_v2_decision.md",
    )
    args = parser.parse_args()
    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    audit = json.loads(args.protocol_audit.read_text(encoding="utf-8"))
    refiner = json.loads(args.refiner_gate.read_text(encoding="utf-8"))
    if selection.get("selection_source") != "independent_dev_only":
        raise ValueError("visual v2 freeze requires an independent-dev-only selection")
    selected_rows = {str(row["config_id"]): row for row in selection.get("selected", [])}
    selected_ids = set(map(str, selection.get("selected_config_ids", ())))
    if set(selected_rows) != selected_ids or not selected_ids:
        raise ValueError("visual v2 selection rows and selected IDs disagree")
    source_rows = {str(row["source_config_id"]): row for row in selected_rows.values()}
    if len(source_rows) != len(selected_rows):
        raise ValueError("visual v2 selection reuses a source config")
    frozen_setting = refiner.get("frozen_setting", {})
    selected_margins = {float(row["candidate_margin_s"]) for row in selected_rows.values()}
    if selected_margins != {float(frozen_setting.get("candidate_margin_s", -1))}:
        raise ValueError("visual v2 selection and refiner gate use different candidate margins")

    resolved_configs = {}
    visual_parameters = {}
    for job_path in sorted(args.matrix_root.glob("**/config.resolved.json")):
        if job_path.parent.name in ("predictions", "snapshots"):
            continue
        job_root = job_path.parent
        job = json.loads(job_path.read_text(encoding="utf-8"))
        if not {"dataset", "method", "budget", "query_grid"} <= set(job):
            continue
        if "dev" not in str(job["dataset"]).lower():
            raise ValueError(f"visual v2 freeze source is not independent dev: {job['dataset']}")
        method = Path(job["method"]).stem
        budget = yaml.safe_load(Path(job["budget"]).read_text(encoding="utf-8"))
        budget_bytes = int(budget["memory_bytes"])
        ingest = json.loads(
            (job_root / "snapshots" / "config.resolved.json").read_text(encoding="utf-8")
        )
        for query_config in job["query_grid"]:
            descriptor = {"method": method, "budget_bytes": budget_bytes, **query_config}
            source_id = f"{method}-{budget_bytes}-{_hash(descriptor)[:12]}"
            selected = source_rows.get(source_id)
            if selected is None:
                continue
            config_id = str(selected["config_id"])
            if config_id in resolved_configs:
                continue
            chosen = selected["visual"]
            if (
                chosen.get("method") != method
                or int(chosen.get("budget_bytes", -1)) != budget_bytes
            ):
                raise ValueError(f"visual v2 descriptor disagrees with source: {config_id}")
            resolved_configs[config_id] = read_resolved_config(job_root / "query.resolved.yaml")
            visual_parameters[config_id] = {
                "writer": ingest["writer"], "clip_model": ingest["clip_model"],
                "clip_revision": ingest["clip_revision"],
                "clip_sha256": ingest["clip_sha256"],
                "jpeg_short_edge": ingest["jpeg_short_edge"],
                "jpeg_quality": ingest["jpeg_quality"],
                "embedding_precision": ingest["embedding_precision"],
                "anchor_fraction": ingest["anchor_fraction"],
                "top_k": int(chosen["top_k"]),
                "expand_neighbors": int(chosen["expand_neighbors"]),
                "merge_gap_s": float(chosen["merge_gap_s"]),
                "candidate_margin_s": float(selected["candidate_margin_s"]),
                "coarse_margin_s": float(selected["coarse_margin_s"]),
                "timelens_enabled": refiner.get("decision") == "enable",
                "refiner_sampling": frozen_setting.get("sampling"),
                "refiner_max_frames": int(frozen_setting.get("max_frames", 0)),
            }
    if set(resolved_configs) != selected_ids:
        missing = selected_ids - set(resolved_configs)
        raise ValueError(f"selected visual v2 configs are missing: {', '.join(sorted(missing))}")
    frozen = freeze_visual_configuration(
        resolved_configs, selection, audit, refiner, visual_parameters,
    )
    frozen["route"] = "semantic_visual_cache_candidate_margin_v2"
    frozen["selection_source"] = "independent_dev_only"
    if args.output.exists():
        raise FileExistsError(f"frozen visual v2 config already exists: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(yaml.safe_dump(frozen, sort_keys=True), encoding="utf-8")
    args.decision_record.parent.mkdir(parents=True, exist_ok=True)
    args.decision_record.write_text(
        "# frozen_visual_v2 decision\n\n"
        "Selection source: independent dev only\n\n"
        f"Selected configs: {', '.join(sorted(selected_ids))}\n\n"
        f"TimeLens refiner: {frozen['refiner_decision']}\n\n"
        f"Policy: {frozen['post_freeze_policy']}\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "output": str(args.output), "configs": sorted(selected_ids),
        "refiner_decision": frozen["refiner_decision"],
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
