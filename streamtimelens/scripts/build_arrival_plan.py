#!/usr/bin/env python3
"""Freeze natural/fixed delayed-query cohorts from an annotation JSONL file."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT / "src"))

from streamtimelens.protocol.arrival import build_arrival_plan, write_arrival_plan


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True,
                        help="JSONL with video_id, query_id, query, gt_span, duration")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ratios", default=".25,.5,.75,1")
    parser.add_argument("--delays", default="", help="Optional event-end delays in seconds, e.g. 5,15")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing frozen plan explicitly")
    args = parser.parse_args()
    examples = [json.loads(line) for line in args.annotations.read_text(encoding="utf-8").splitlines() if line]
    rows = build_arrival_plan(examples, (float(value) for value in args.ratios.split(",")),
                              (float(value) for value in args.delays.split(",") if value))
    digest = write_arrival_plan(rows, args.output, overwrite=args.overwrite)
    print(json.dumps({"records": len(rows), "sha256": digest, "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
