#!/usr/bin/env python3
"""Join independent-dev GT after v2 refiner inference and render gate metrics."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import yaml

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT / "src"))
sys.path.insert(0, str(PACKAGE_ROOT.parent / "src"))

from baas.provenance import versioned_result_path, write_provenance
from streamtimelens.evaluation.bootstrap import PairedMetricObservation, paired_video_bootstrap
from streamtimelens.evaluation.metrics import temporal_iou
from streamtimelens.evaluation.visual_v2 import oracle_candidate_for_condition


def _hash(value) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix-root", type=Path, required=True)
    parser.add_argument("--source-config-id", required=True)
    parser.add_argument("--predictions", type=Path, action="append", required=True)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--oracle-metrics", type=Path, required=True)
    parser.add_argument("--oracle-condition", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    annotations = {row["query_id"]: row for row in _jsonl(args.queries)}
    predictions = [row for path in args.predictions for row in _jsonl(path)]
    by_sample = {str(row["sample_id"]): row for row in predictions}
    if len(by_sample) != len(predictions):
        raise ValueError("visual v2 refiner predictions contain duplicate sample IDs")
    expected = set()
    jobs = {}
    for path in sorted(args.matrix_root.glob("**/predictions/*.rho_*.qg_*.json")):
        job_root = path.parent.parent
        if job_root not in jobs:
            jobs[job_root] = json.loads(
                (job_root / "config.resolved.json").read_text(encoding="utf-8")
            )
        job = jobs[job_root]
        if "dev" not in str(job["dataset"]).lower():
            raise ValueError(f"visual v2 summary source is not independent dev: {job['dataset']}")
        if Path(job["method"]).stem != "uniform_raw":
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        variant = next(
            row for index, row in enumerate(job["query_grid"])
            if payload["query_variant"] == f"qg_{index:02d}_{_hash(row)[:8]}"
        )
        budget = int(yaml.safe_load(Path(job["budget"]).read_text())["memory_bytes"])
        descriptor = {"method": "uniform_raw", "budget_bytes": budget, **variant}
        source_id = f"uniform_raw-{budget}-{_hash(descriptor)[:12]}"
        if source_id == args.source_config_id:
            expected.add(f"{payload['query_id']}@{float(payload['rho_q']):.2f}")
    if set(by_sample) != expected:
        raise ValueError(
            f"visual v2 refiner matrix is incomplete: predictions={len(by_sample)}, "
            f"expected={len(expected)}"
        )
    rows = []
    for sample_id, prediction in sorted(by_sample.items()):
        annotation = annotations.get(str(prediction["query_id"]))
        if annotation is None or annotation["video_id"] != prediction["video_id"]:
            raise ValueError(f"visual v2 refiner annotation mismatch: {sample_id}")
        rho = float(prediction["rho_q"])
        if float(annotation["gt_span"][1]) > rho * float(annotation["duration_s"]) + 1e-6:
            continue
        gt = tuple(map(float, annotation["gt_span"]))
        coarse = tuple(prediction["coarse_span"]) if prediction.get("coarse_span") else None
        final = tuple(prediction["final_span"]) if prediction.get("final_span") else None
        rows.append({
            "sample_id": sample_id, "video_id": prediction["video_id"],
            "coarse_iou": temporal_iou(gt, coarse), "refined_iou": temporal_iou(gt, final),
            "valid": prediction.get("model_status") == "ok",
        })
    if not rows:
        raise ValueError("visual v2 refiner has no eligible independent-dev observations")
    paired = []
    for row in rows:
        paired.extend([
            PairedMetricObservation(
                "timelens_local_v2", 1, row["video_id"], row["sample_id"], row["refined_iou"],
            ),
            PairedMetricObservation(
                "coarse_visual_v2", 1, row["video_id"], row["sample_id"], row["coarse_iou"],
            ),
        ])
    retrieved = {
        "coarse_miou": sum(row["coarse_iou"] for row in rows) / len(rows),
        "refined_miou": sum(row["refined_iou"] for row in rows) / len(rows),
        "count": len(rows), "valid_count": sum(row["valid"] for row in rows),
        "fallback_count": sum(not row["valid"] for row in rows),
        "paired_bootstrap_refined_minus_coarse": paired_video_bootstrap(
            paired, method_a="timelens_local_v2", method_b="coarse_visual_v2",
        ),
    }
    oracle_report = json.loads(args.oracle_metrics.read_text(encoding="utf-8"))
    oracle = oracle_candidate_for_condition(oracle_report, args.oracle_condition)
    output_root = versioned_result_path(args.output)
    output_root.mkdir(parents=True, exist_ok=True)
    write_provenance(
        output_root,
        configuration={
            "matrix_root": str(args.matrix_root.resolve()),
            "source_config_id": args.source_config_id,
            "predictions": [str(path.resolve()) for path in args.predictions],
            "queries": str(args.queries.resolve()),
            "oracle_metrics": str(args.oracle_metrics.resolve()),
            "oracle_condition": args.oracle_condition,
            "sampling_interval_s": oracle["sampling_interval_s"],
            "ground_truth_joined_after_inference": True,
        },
        config_filename="config.resolved.json",
    )
    (output_root / "retrieved_metrics.json").write_text(
        json.dumps(retrieved, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    (output_root / "oracle_metrics.json").write_text(
        json.dumps(oracle, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    print(json.dumps({
        "eligible": len(rows), "retrieved_metrics": str(output_root / "retrieved_metrics.json"),
        "oracle_metrics": str(output_root / "oracle_metrics.json"),
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
