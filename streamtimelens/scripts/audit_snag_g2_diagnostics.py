#!/usr/bin/env python3
"""Audit the frozen three-row SnAG G2 diagnostic bundle."""

from __future__ import annotations

import argparse
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
from streamtimelens.evaluation.snag_diagnostics import (  # noqa: E402
    assemble_snag_g2_rows,
    audit_snag_g2_diagnostics,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--rows", type=Path)
    source.add_argument("--full-grid-run", type=Path)
    parser.add_argument("--pooled-frozen-run", type=Path)
    parser.add_argument("--pooled-trained-run", type=Path)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.rows is not None:
        if args.pooled_frozen_run is not None or args.pooled_trained_run is not None:
            parser.error("--rows cannot be combined with run directories")
        rows = json.loads(args.rows.read_text(encoding="utf-8"))
        if not isinstance(rows, list):
            raise ValueError("SnAG G2 rows must be a JSON list")
    else:
        if args.pooled_frozen_run is None or args.pooled_trained_run is None:
            parser.error("run assembly requires both pooled run directories")
        rows = assemble_snag_g2_rows(
            args.full_grid_run, args.pooled_frozen_run, args.pooled_trained_run,
        )
    report = audit_snag_g2_diagnostics(rows)
    output = versioned_result_path(args.output)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite G2 audit: {output}")
    write_provenance(output, configuration={"diagnostic_rows": rows})
    (output / "diagnostic_rows.json").write_text(
        json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    (output / "g2_diagnostic_gate.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    write_artifact_manifest(output)
    print(json.dumps({"output": str(output), "passed": report["passed"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
