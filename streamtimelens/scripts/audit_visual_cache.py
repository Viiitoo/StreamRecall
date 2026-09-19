#!/usr/bin/env python3
"""Audit visual-cache run directories listed as JSONL VisualAuditRun records."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT / "src"))
sys.path.insert(0, str(PACKAGE_ROOT.parent / "src"))

from baas.provenance import versioned_result_path, write_provenance
from streamtimelens.evaluation.visual_audit import VisualAuditRun, audit_visual_cache_runs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--runs", type=Path)
    source.add_argument("--matrix-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.runs:
        rows = [
            VisualAuditRun(**json.loads(line))
            for line in args.runs.read_text(encoding="utf-8").splitlines()
        ]
    else:
        rows = []
        for marker in sorted(args.matrix_root.glob("**/.ingest.complete.json")):
            job_root = marker.parent
            job = json.loads((job_root / "config.resolved.json").read_text(encoding="utf-8"))
            budget = yaml.safe_load(Path(job["budget"]).read_text(encoding="utf-8"))
            rows.append(VisualAuditRun(
                Path(job["method"]).stem, int(budget["memory_bytes"]),
                str(job["video"]["video_id"]), str(job_root / "snapshots"),
            ))
    report = audit_visual_cache_runs(rows)
    output_root = versioned_result_path(args.output)
    output_root.mkdir(parents=True, exist_ok=True)
    write_provenance(
        output_root,
        configuration={
            "runs": str(args.runs.resolve()) if args.runs else None,
            "matrix_root": str(args.matrix_root.resolve()) if args.matrix_root else None,
            "expected_rhos": [.25, .5, .75, 1.0],
        },
        config_filename="config.resolved.json",
    )
    (output_root / "visual_protocol_audit.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    print(json.dumps({"passed": report["passed"], "output_root": str(output_root)}))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
