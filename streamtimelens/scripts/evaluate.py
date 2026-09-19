#!/usr/bin/env python3
"""Evaluate GT-free predictions by joining a verified frozen arrival plan."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT / "src"))
sys.path.insert(0, str(PACKAGE_ROOT.parent / "src"))

from baas.provenance import versioned_result_path, write_provenance
from streamtimelens.evaluation.runner import evaluate_predictions
from streamtimelens.protocol.arrival import verify_arrival_plan


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query-plan", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plan = verify_arrival_plan(args.query_plan)
    predictions = [
        json.loads(line) for line in args.predictions.read_text(encoding="utf-8").splitlines()
    ]
    metrics = evaluate_predictions(plan, predictions)
    output_root = versioned_result_path(args.output)
    output_root.mkdir(parents=True, exist_ok=True)
    write_provenance(
        output_root,
        configuration={
            "query_plan": str(args.query_plan.resolve()),
            "query_plan_sha256": args.query_plan.with_suffix(args.query_plan.suffix + ".sha256").read_text().strip(),
            "predictions": str(args.predictions.resolve()),
        },
        config_filename="config.resolved.json",
    )
    (output_root / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    overall = metrics["overall_natural"]
    (output_root / "summary.md").write_text(
        "# Evaluation summary\n\n"
        f"- Count: {overall['count']}\n"
        f"- mIoU: {overall['miou']:.4f}\n"
        f"- R@1@0.3/0.5/0.7: {overall['recall_at_03']:.4f} / "
        f"{overall['recall_at_05']:.4f} / {overall['recall_at_07']:.4f}\n",
        encoding="utf-8",
    )
    print(json.dumps({"metrics": str(output_root / "metrics.json"), **overall}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
