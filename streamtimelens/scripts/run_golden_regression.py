#!/usr/bin/env python3
"""Run and verify the deterministic regression required before formal evaluation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT / "src"))
sys.path.insert(0, str(PACKAGE_ROOT.parent / "src"))

from baas.provenance import versioned_result_path, write_provenance
from streamtimelens.evaluation.golden import assert_golden_matches, run_formal_golden


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--expected", type=Path,
        default=PACKAGE_ROOT / "tests" / "fixtures" / "formal_golden_v1.json",
    )
    args = parser.parse_args()
    output_root = versioned_result_path(args.output)
    output_root.mkdir(parents=True, exist_ok=True)
    write_provenance(
        output_root,
        configuration={"purpose": "pre_formal_golden_regression", "fixture": str(args.expected)},
        config_filename="config.resolved.json",
    )
    actual = run_formal_golden(output_root)
    actual_path = output_root / "formal_golden.actual.json"
    actual_path.write_text(json.dumps(actual, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    assert_golden_matches(actual, args.expected)
    print(json.dumps({"status": "passed", "output_root": str(output_root)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
