#!/usr/bin/env python3
"""Select at most two frozen Pareto configurations for V6 after V5 completes."""

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
from streamtimelens.evaluation.formal_visual import select_activitynet_configs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summaries", type=Path, action="append", required=True)
    parser.add_argument("--frozen", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = [
        json.loads(line) for path in args.summaries
        for line in path.read_text(encoding="utf-8").splitlines() if line
    ]
    frozen = yaml.safe_load(args.frozen.read_text(encoding="utf-8"))
    selection = select_activitynet_configs(frozen, rows)
    output_root = versioned_result_path(args.output)
    output_root.mkdir(parents=True, exist_ok=True)
    write_provenance(
        output_root,
        configuration={
            "summaries": [str(path.resolve()) for path in args.summaries],
            "frozen": str(args.frozen.resolve()), "maximum_configs": 2,
        },
        config_filename="config.resolved.json",
    )
    (output_root / "selection.json").write_text(
        json.dumps(selection, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    print(json.dumps({**selection, "output_root": str(output_root)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
