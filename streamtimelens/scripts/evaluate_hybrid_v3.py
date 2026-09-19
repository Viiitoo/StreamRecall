#!/usr/bin/env python3
"""Evaluate Hybrid V3 predictions in a physically separate GT-visible step."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
WORK_ROOT = PACKAGE_ROOT.parent
sys.path.insert(0, str(PACKAGE_ROOT / "src"))
sys.path.insert(0, str(WORK_ROOT / "src"))

from baas.provenance import versioned_result_path, write_artifact_manifest, write_provenance
from streamtimelens.baselines.offline_resume import load_jsonl, sha256_file
from streamtimelens.evaluation.hybrid_v3 import summarize_hybrid_v3


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--budget-bytes", type=int, required=True)
    parser.add_argument("--bootstrap-resamples", type=int, default=10_000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize_hybrid_v3(
        load_jsonl(args.predictions), load_jsonl(args.annotations),
        budget_bytes=args.budget_bytes,
        bootstrap_resamples=args.bootstrap_resamples,
    )
    output_root = versioned_result_path(args.output)
    relative = output_root.relative_to((WORK_ROOT / "results").resolve())
    if len(relative.parts) < 3 or relative.parts[1] != "hybrid_v3":
        raise ValueError("Hybrid V3 evaluation output must be under results/hybrid_v3/<run-name>")
    output_root.mkdir(parents=True, exist_ok=True)
    configuration = {
        "protocol": "evaluation_only_past_visible",
        "predictions": str(args.predictions.resolve()),
        "predictions_sha256": sha256_file(args.predictions),
        "annotations": str(args.annotations.resolve()),
        "annotations_sha256": sha256_file(args.annotations),
        "budget_bytes": args.budget_bytes,
        "bootstrap_resamples": args.bootstrap_resamples,
        "formal_test_tuning": False,
    }
    write_provenance(
        output_root, configuration=configuration,
        config_filename="config.resolved.json",
    )
    (output_root / "accuracy_summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_artifact_manifest(output_root)
    print(json.dumps({
        "count": result["count"], "hybrid_miou": result["hybrid"]["miou"],
        "clip_miou": result["clip_coarse"]["miou"],
        "output_root": str(output_root),
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
