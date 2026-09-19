#!/usr/bin/env python3
"""Freeze selected V2 visual configs directly from an audited matrix run."""

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
        default=PACKAGE_ROOT / "configs" / "frozen_visual_v1.yaml",
    )
    parser.add_argument(
        "--decision-record", type=Path,
        default=PACKAGE_ROOT / "docs" / "frozen_visual_v1_decision.md",
    )
    args = parser.parse_args()
    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    audit = json.loads(args.protocol_audit.read_text(encoding="utf-8"))
    refiner = json.loads(args.refiner_gate.read_text(encoding="utf-8"))
    selected = set(map(str, selection.get("selected_config_ids", ())))
    resolved_configs = {}
    visual_parameters = {}
    for job_path in sorted(args.matrix_root.glob("**/config.resolved.json")):
        if job_path.parent.name in ("predictions", "snapshots"):
            continue
        job_root = job_path.parent
        job = json.loads(job_path.read_text(encoding="utf-8"))
        if "method" not in job or "budget" not in job or "query_grid" not in job:
            continue
        method = Path(job["method"]).stem
        budget = yaml.safe_load(Path(job["budget"]).read_text(encoding="utf-8"))
        budget_bytes = int(budget["memory_bytes"])
        ingest = json.loads(
            (job_root / "snapshots" / "config.resolved.json").read_text(encoding="utf-8")
        )
        for query_config in job["query_grid"]:
            descriptor = {"method": method, "budget_bytes": budget_bytes, **query_config}
            config_id = f"{method}-{budget_bytes}-{_hash(descriptor)[:12]}"
            if config_id not in selected or config_id in resolved_configs:
                continue
            resolved_configs[config_id] = read_resolved_config(job_root / "query.resolved.yaml")
            visual_parameters[config_id] = {
                "writer": ingest["writer"], "clip_model": ingest["clip_model"],
                "clip_revision": ingest["clip_revision"],
                "clip_sha256": ingest["clip_sha256"],
                "jpeg_short_edge": ingest["jpeg_short_edge"],
                "jpeg_quality": ingest["jpeg_quality"],
                "embedding_precision": ingest["embedding_precision"],
                "anchor_fraction": ingest["anchor_fraction"],
                "top_k": int(query_config["top_k"]),
                "expand_neighbors": int(query_config["expand_neighbors"]),
                "merge_gap_s": float(query_config["merge_gap_s"]),
                "coarse_margin_s": float(query_config["coarse_margin_s"]),
                "timelens_enabled": refiner.get("decision") == "enable",
            }
    if set(resolved_configs) != selected:
        missing = selected - set(resolved_configs)
        raise ValueError(f"selected matrix configs are missing: {', '.join(sorted(missing))}")
    frozen = freeze_visual_configuration(
        resolved_configs, selection, audit, refiner, visual_parameters,
    )
    if args.output.exists():
        raise FileExistsError(f"frozen visual config already exists: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(yaml.safe_dump(frozen, sort_keys=True), encoding="utf-8")
    args.decision_record.parent.mkdir(parents=True, exist_ok=True)
    args.decision_record.write_text(
        "# frozen_visual_v1 decision\n\n"
        f"Selected method: {selection['selected_method']}\n\n"
        f"Selected configs: {', '.join(sorted(selected))}\n\n"
        f"TimeLens refiner: {frozen['refiner_decision']}\n\n"
        f"Policy: {frozen['post_freeze_policy']}\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "output": str(args.output), "selected_method": selection["selected_method"],
        "config_count": len(selected), "refiner_decision": frozen["refiner_decision"],
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
