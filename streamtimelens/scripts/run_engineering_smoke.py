#!/usr/bin/env python3
"""Run the deterministic 20-video, six-method engineering smoke."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT / "src"))
sys.path.insert(0, str(PACKAGE_ROOT.parent / "src"))

from baas.provenance import versioned_result_path, write_provenance
from streamtimelens.evaluation.smoke import INTERNAL_METHODS, run_engineering_smoke


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--videos", type=int, default=20)
    parser.add_argument("--memory-bytes", type=int, default=256 * 1024)
    args = parser.parse_args()
    output_root = versioned_result_path(args.output)
    output_root.mkdir(parents=True, exist_ok=True)
    configuration = {
        "purpose": "engineering_only_no_model_selection", "videos": args.videos,
        "methods": INTERNAL_METHODS, "memory_bytes": args.memory_bytes,
        "duration_classes_s": [12, 120, 600], "arrival_ratios": [.25, .5, .75, 1.0],
        "deterministic_model_backends": True,
    }
    write_provenance(
        output_root, configuration=configuration, config_filename="config.resolved.json",
    )
    result = run_engineering_smoke(
        output_root, video_count=args.videos, memory_bytes=args.memory_bytes,
    )
    (output_root / "smoke.summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    print(json.dumps({**result, "output_root": str(output_root)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
