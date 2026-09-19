#!/usr/bin/env python3
"""Build the V3 enable/disable decision from oracle- and retrieved-candidate metrics."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT / "src"))
sys.path.insert(0, str(PACKAGE_ROOT.parent / "src"))

from baas.provenance import versioned_result_path, write_provenance
from streamtimelens.evaluation.visual_refiner_gate import (
    build_visual_refiner_gate, disabled_visual_refiner_gate,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oracle-metrics", type=Path)
    parser.add_argument("--retrieved-metrics", type=Path)
    parser.add_argument("--disabled-reason")
    parser.add_argument("--candidate-margin-s", type=float, default=10.0)
    parser.add_argument("--max-frames", type=int, default=16)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.disabled_reason:
        if args.oracle_metrics or args.retrieved_metrics:
            parser.error("--disabled-reason cannot be combined with metric inputs")
        gate = disabled_visual_refiner_gate(args.disabled_reason)
    else:
        if not args.oracle_metrics or not args.retrieved_metrics:
            parser.error("both metric inputs are required unless --disabled-reason is used")
        oracle = json.loads(args.oracle_metrics.read_text(encoding="utf-8"))
        retrieved = json.loads(args.retrieved_metrics.read_text(encoding="utf-8"))
        gate = build_visual_refiner_gate(
            oracle, retrieved, candidate_margin_s=args.candidate_margin_s,
            max_frames=args.max_frames,
        )
    output_root = versioned_result_path(args.output)
    output_root.mkdir(parents=True, exist_ok=True)
    write_provenance(
        output_root,
        configuration={
            "oracle_metrics": str(args.oracle_metrics.resolve()) if args.oracle_metrics else None,
            "retrieved_metrics": (
                str(args.retrieved_metrics.resolve()) if args.retrieved_metrics else None
            ),
            "disabled_reason": args.disabled_reason,
            "candidate_margin_s": args.candidate_margin_s,
            "sampling": "uniform", "max_frames": args.max_frames,
        },
        config_filename="config.resolved.json",
    )
    (output_root / "visual_refiner_gate.json").write_text(
        json.dumps(gate, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    print(json.dumps({"decision": gate["decision"], "output_root": str(output_root)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
