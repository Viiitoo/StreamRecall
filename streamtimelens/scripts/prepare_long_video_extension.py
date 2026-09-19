#!/usr/bin/env python3
"""Select frozen long-video runs and render memory-aging diagnostics."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT / "src"))
sys.path.insert(0, str(PACKAGE_ROOT.parent / "src"))

from baas.provenance import versioned_result_path, write_provenance
from streamtimelens.evaluation.long_video import (
    LongVideoConfig, MemoryAgingObservation, memory_aging_report,
    select_long_video_configs,
)


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configs", type=Path, required=True)
    parser.add_argument("--observations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = {
        "selection": select_long_video_configs(
            LongVideoConfig(**row) for row in _jsonl(args.configs)
        ),
        "memory_aging": memory_aging_report(
            MemoryAgingObservation(**row) for row in _jsonl(args.observations)
        ),
    }
    output_root = versioned_result_path(args.output)
    output_root.mkdir(parents=True, exist_ok=True)
    write_provenance(
        output_root,
        configuration={
            "configs": str(args.configs.resolve()),
            "observations": str(args.observations.resolve()),
        },
        config_filename="config.resolved.json",
    )
    (output_root / "long_video_extension.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    print(output_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
