#!/usr/bin/env python3
"""Validate frozen formal summaries and render accuracy/diagnostic/resource/Pareto tables."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT / "src"))
sys.path.insert(0, str(PACKAGE_ROOT.parent / "src"))

from baas.provenance import versioned_result_path, write_provenance
from streamtimelens.evaluation.formal_report import (
    FormalRunSummary, build_formal_report, formal_report_markdown,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summaries", type=Path, required=True)
    parser.add_argument("--methods", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = [
        FormalRunSummary(**json.loads(line))
        for line in args.summaries.read_text(encoding="utf-8").splitlines()
    ]
    methods = tuple(value.strip() for value in args.methods.split(",") if value.strip())
    report = build_formal_report(rows, expected_methods=methods)
    output_root = versioned_result_path(args.output)
    output_root.mkdir(parents=True, exist_ok=True)
    write_provenance(
        output_root,
        configuration={"summaries": str(args.summaries.resolve()), "methods": methods},
        config_filename="config.resolved.json",
    )
    (output_root / "report_tables.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    (output_root / "summary.md").write_text(formal_report_markdown(report), encoding="utf-8")
    print(output_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
