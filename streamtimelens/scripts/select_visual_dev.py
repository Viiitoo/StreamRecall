#!/usr/bin/env python3
"""Apply the retrieval-first V2 gate and Pareto selection to visual dev summaries."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT / "src"))
sys.path.insert(0, str(PACKAGE_ROOT.parent / "src"))

from baas.provenance import versioned_result_path, write_provenance
from streamtimelens.evaluation.dev_selection import VisualDevRun, select_visual_dev_configs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = [
        VisualDevRun(**json.loads(line))
        for line in args.runs.read_text(encoding="utf-8").splitlines()
    ]
    selection = select_visual_dev_configs(rows)
    output_root = versioned_result_path(args.output)
    output_root.mkdir(parents=True, exist_ok=True)
    write_provenance(
        output_root, configuration={"runs": str(args.runs.resolve())},
        config_filename="config.resolved.json",
    )
    (output_root / "selection.json").write_text(
        json.dumps(selection, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    print(json.dumps({
        "passed": selection["passed"],
        "selected_config_ids": selection["selected_config_ids"],
        "output_root": str(output_root),
    }))
    return 0 if selection["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
