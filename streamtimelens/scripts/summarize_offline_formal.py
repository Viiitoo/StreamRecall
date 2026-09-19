#!/usr/bin/env python3
"""Join formal GT to sharded Offline TimeLens upper-bound predictions."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT / "src"))
sys.path.insert(0, str(PACKAGE_ROOT.parent / "src"))

from baas.provenance import versioned_result_path, write_provenance
from streamtimelens.evaluation.formal_visual import FormalVisualObservation, video_bootstrap_ci
from streamtimelens.evaluation.metrics import VTGMetricExample, evaluate_vtg


def _rows(paths: list[Path]) -> list[dict]:
    return [
        json.loads(line) for path in paths
        for line in path.read_text(encoding="utf-8").splitlines() if line
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, action="append", required=True)
    parser.add_argument("--annotations", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    annotations = {row["query_id"]: row for row in _rows(args.annotations)}
    predictions = _rows(args.predictions)
    by_id = {row["query_id"]: row for row in predictions}
    if len(by_id) != len(predictions):
        raise ValueError("offline formal predictions contain duplicate query IDs")
    if set(by_id) != set(annotations):
        raise ValueError(
            f"offline formal matrix is incomplete: predictions={len(by_id)}, annotations={len(annotations)}"
        )
    grouped = defaultdict(list)
    for query_id, annotation in annotations.items():
        grouped[annotation["dataset"]].append((annotation, by_id[query_id]))
    summaries = {}
    for dataset, rows in sorted(grouped.items()):
        metrics = evaluate_vtg([
            VTGMetricExample(
                annotation["query_id"], annotation["video_id"], tuple(annotation["gt_span"]),
                tuple(prediction["span"]) if prediction.get("span") is not None else None,
                prediction["status"],
            )
            for annotation, prediction in rows
        ]).to_dict()
        bootstrap_rows = [
            FormalVisualObservation(
                "offline_timelens_upper_bound", dataset, 1, annotation["video_id"],
                annotation["query_id"], 1.0, "natural", "[0,1)",
                float(annotation["duration_s"]), tuple(annotation["gt_span"]),
                tuple(prediction["span"]) if prediction.get("span") is not None else None,
                prediction["status"], (), (), 0, 0, 0,
                float(prediction["resource"].get("cuda_s") or 0),
                float(prediction["resource"].get("wall_s") or 0),
            )
            for annotation, prediction in rows
        ]
        summaries[dataset] = {
            **metrics, "protocol": "offline_upper_bound_not_streaming",
            "gpu_s": sum(float(row[1]["resource"].get("cuda_s") or 0) for row in rows),
            "query_latency_s": sum(
                float(row[1]["resource"].get("wall_s") or 0) for row in rows
            ) / len(rows),
            "bootstrap_miou": video_bootstrap_ci(bootstrap_rows),
        }
    output_root = versioned_result_path(args.output)
    output_root.mkdir(parents=True, exist_ok=True)
    write_provenance(
        output_root,
        configuration={
            "predictions": [str(path.resolve()) for path in args.predictions],
            "annotations": [str(path.resolve()) for path in args.annotations],
            "protocol": "offline_upper_bound_not_streaming",
        },
        config_filename="config.resolved.json",
    )
    (output_root / "offline_formal_summary.json").write_text(
        json.dumps(summaries, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    print(json.dumps({"datasets": len(summaries), "output_root": str(output_root)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
