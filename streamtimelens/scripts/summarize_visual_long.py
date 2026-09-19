#!/usr/bin/env python3
"""Render V6 ActivityNet memory-aging and multi-query amortization tables."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT / "src"))
sys.path.insert(0, str(PACKAGE_ROOT.parent / "src"))

from baas.provenance import versioned_result_path, write_provenance
from streamtimelens.evaluation.formal_visual import (
    FormalVisualObservation, summarize_long_video_observations,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observations", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    selected_ids = {row["config_id"] for row in selection["selected"]}
    observations = []
    for line in args.observations.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        row = json.loads(line)
        if row["config_id"] not in selected_ids:
            raise ValueError(f"ActivityNet observation is outside V6 selection: {row['config_id']}")
        for key in ("gt_span", "predicted_span", "retrieved_timestamps", "candidate_spans"):
            if row.get(key) is not None:
                row[key] = tuple(tuple(value) if isinstance(value, list) else value for value in row[key])
        observations.append(FormalVisualObservation(**row))
    report = summarize_long_video_observations(observations)
    output_root = versioned_result_path(args.output)
    output_root.mkdir(parents=True, exist_ok=True)
    write_provenance(
        output_root,
        configuration={
            "observations": str(args.observations.resolve()),
            "selection": str(args.selection.resolve()),
        },
        config_filename="config.resolved.json",
    )
    (output_root / "long_video_extension.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    print(json.dumps({"configs": len(report["configs"]), "output_root": str(output_root)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
