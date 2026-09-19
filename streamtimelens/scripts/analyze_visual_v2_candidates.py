#!/usr/bin/env python3
"""Evaluate candidate-window expansion using only frozen independent-dev predictions."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import yaml

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT / "src"))
sys.path.insert(0, str(PACKAGE_ROOT.parent / "src"))

from baas.provenance import versioned_result_path, write_provenance
from streamtimelens.evaluation.visual_v2 import (
    select_candidate_margin_configs, summarize_candidate_margin,
)


def _hash(value) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix-root", type=Path, required=True)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--candidate-margins", default="1,2,3,4")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    margins = tuple(float(value) for value in args.candidate_margins.split(","))
    if not margins or any(value <= 0 for value in margins):
        raise ValueError("candidate margins must be positive")
    annotations = {
        row["query_id"]: row
        for row in (
            json.loads(line) for line in args.queries.read_text(encoding="utf-8").splitlines()
        )
    }
    grouped = defaultdict(list)
    descriptors = {}
    jobs = {}
    for path in sorted(args.matrix_root.glob("**/predictions/*.rho_*.qg_*.json")):
        job_root = path.parent.parent
        if job_root not in jobs:
            jobs[job_root] = json.loads(
                (job_root / "config.resolved.json").read_text(encoding="utf-8")
            )
        job = jobs[job_root]
        if "dev" not in str(job["dataset"]).lower():
            raise ValueError(f"visual v2 tuning source is not independent dev: {job['dataset']}")
        if Path(job["method"]).stem != "uniform_raw":
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        annotation = annotations.get(payload.get("query_id"))
        if annotation is None or annotation["video_id"] != payload.get("video_id"):
            raise ValueError(f"visual v2 annotation mismatch: {path}")
        rho = float(payload["rho_q"])
        upper_bound_s = rho * float(annotation["duration_s"])
        if float(annotation["gt_span"][1]) > upper_bound_s + 1e-6:
            continue
        variant = next(
            row for index, row in enumerate(job["query_grid"])
            if payload["query_variant"] == f"qg_{index:02d}_{_hash(row)[:8]}"
        )
        budget = int(yaml.safe_load(Path(job["budget"]).read_text())["memory_bytes"])
        descriptor = {"method": "uniform_raw", "budget_bytes": budget, **variant}
        source_id = f"uniform_raw-{budget}-{_hash(descriptor)[:12]}"
        descriptors[source_id] = descriptor
        grouped[source_id].append({
            "sample_id": f"{payload['query_id']}@{rho:.2f}",
            "query_id": payload["query_id"], "video_id": payload["video_id"],
            "budget_bytes": budget, "gt_span": annotation["gt_span"],
            "upper_bound_s": upper_bound_s,
            "candidate_spans": [
                [float(row["start_s"]), float(row["end_s"])]
                for row in payload.get("candidates", [])
            ],
        })
    summaries = []
    for source_id, rows in sorted(grouped.items()):
        descriptor = descriptors[source_id]
        for margin in margins:
            # Preserve the v1 coarse window while exposing a tighter, explicit
            # candidate prior to a future local refiner.
            coarse_margin = max(0.0, float(descriptor.get("coarse_margin_s", 0)) - margin)
            v2_descriptor = {
                **descriptor, "candidate_margin_s": margin,
                "coarse_margin_s": coarse_margin,
            }
            summary = summarize_candidate_margin(
                rows, margin_s=margin, coarse_margin_s=coarse_margin,
            )
            summaries.append({
                **summary, "source_config_id": source_id,
                "config_id": f"uniform_raw_v2-{descriptor['budget_bytes']}-{_hash(v2_descriptor)[:12]}",
                "budget_bytes": descriptor["budget_bytes"], "visual": v2_descriptor,
            })
    if not summaries:
        raise ValueError("no eligible independent-dev Uniform predictions were found")
    selection = select_candidate_margin_configs(summaries)
    output_root = versioned_result_path(args.output)
    output_root.mkdir(parents=True, exist_ok=True)
    write_provenance(
        output_root,
        configuration={
            "matrix_root": str(args.matrix_root.resolve()),
            "queries": str(args.queries.resolve()), "candidate_margins": margins,
            "formal_test_tuning": False, "bootstrap_resamples": 10000,
        },
        config_filename="config.resolved.json",
    )
    (output_root / "candidate_summaries.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in summaries),
        encoding="utf-8",
    )
    (output_root / "selection.json").write_text(
        json.dumps(selection, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    print(json.dumps({
        "summaries": len(summaries), "passed": selection["passed"],
        "selected_config_ids": selection["selected_config_ids"],
        "output_root": str(output_root),
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
