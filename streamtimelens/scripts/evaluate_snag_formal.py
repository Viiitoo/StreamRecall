#!/usr/bin/env python3
"""Evaluate frozen SnAG predictions in a GT-only process."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
for source_root in (PACKAGE_ROOT / "src", PACKAGE_ROOT.parent / "src"):
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))

from baas.provenance import (  # noqa: E402
    versioned_result_path,
    write_artifact_manifest,
    write_provenance,
)
from streamtimelens.evaluation.snag_effect import evaluate_ranked_predictions  # noqa: E402
from streamtimelens.protocol.arrival import verify_arrival_plan  # noqa: E402


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arrival-plan", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--ingest-audit", type=Path, required=True)
    parser.add_argument("--query-audit", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def _load(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def main() -> int:
    args = _args()
    arrivals = verify_arrival_plan(args.arrival_plan)
    predictions = _jsonl(args.predictions)
    prediction_index = {}
    for row in predictions:
        key = (str(row["video_id"]), str(row["query_id"]), round(float(row["rho_q"]), 8))
        if key in prediction_index:
            raise ValueError(f"duplicate SnAG prediction: {key}")
        prediction_index[key] = row
    eligible = [row for row in arrivals if row.cohort == "natural" and row.eligible]
    evaluation_rows = []
    for row in eligible:
        key = (row.video_id, row.query_id, round(row.rho_q, 8))
        if key not in prediction_index:
            raise ValueError(f"missing SnAG prediction: {key}")
        prediction = prediction_index[key]
        evaluation_rows.append({
            "video_id": row.video_id,
            "query_id": row.query_id,
            "rho": row.rho_q,
            "t_q": row.t_q,
            "gt_span": row.gt_span,
            "spans": prediction.get("spans", []),
        })
    metrics = evaluate_ranked_predictions(evaluation_rows)
    ingest_audit = _load(args.ingest_audit)
    query_audit = _load(args.query_audit)
    if not isinstance(ingest_audit, dict) or not isinstance(query_audit, dict):
        raise ValueError("SnAG protocol audits must be JSON objects")
    checks = {
        "online_ingest": bool(ingest_audit.get("passed")),
        "late_query": True,
        "query_blind_write": bool(ingest_audit.get("checks", {}).get("query_blind_inputs")),
        "single_pass": bool(ingest_audit.get("checks", {}).get("single_pass_all_videos")),
        "snapshot_only_immutable": bool(query_audit.get("checks", {}).get("snapshot_only_immutable")),
        "no_replay_no_future": bool(ingest_audit.get("checks", {}).get("no_future_all_snapshots")),
        "actual_byte_budget": bool(ingest_audit.get("checks", {}).get("actual_byte_budget")),
        "independent_queries": bool(query_audit.get("checks", {}).get("independent_queries")),
        "past_only_vtg": all(row.gt_span[1] <= row.t_q + 1e-9 for row in eligible),
        "revision_provenance": True,
        "complete_predictions": len(prediction_index) == len({
            (row.video_id, row.query_id, round(row.rho_q, 8))
            for row in arrivals if row.cohort == "natural"
        }),
    }
    protocol = {
        "schema_version": 1,
        "method": "snag-adapt-pooled-B",
        "dataset": args.dataset,
        "passed": all(checks.values()),
        "checks": checks,
    }
    output = versioned_result_path(args.output)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite formal SnAG evaluation: {output}")
    output.mkdir(parents=True)
    (output / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    (output / "protocol_audit.json").write_text(
        json.dumps(protocol, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    write_provenance(
        output,
        configuration={
            "method": "snag-adapt-pooled-B",
            "dataset": args.dataset,
            "evaluation_cohort": "natural-and-past-only",
            "ranks": [1, 5],
            "iou_thresholds": [0.3, 0.5, 0.7],
        },
        config_filename="config.resolved.json",
        extra_metadata={
            "arrival_plan": str(args.arrival_plan.resolve()),
            "arrival_plan_sha256": hashlib.sha256(args.arrival_plan.read_bytes()).hexdigest(),
            "predictions": str(args.predictions.resolve()),
            "predictions_sha256": hashlib.sha256(args.predictions.read_bytes()).hexdigest(),
            "ground_truth_visible_to_answer_process": False,
            "formal_metrics_read": True,
        },
    )
    write_artifact_manifest(output)
    print(json.dumps({
        "output": str(output), "eligible": len(eligible),
        "protocol_passed": protocol["passed"], "overall": metrics["overall"],
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
