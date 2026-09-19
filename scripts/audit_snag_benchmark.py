#!/usr/bin/env python3
"""Audit whether SnAG-adapt formal benchmark inputs are runnable."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from baas.provenance import versioned_result_path, write_artifact_manifest, write_provenance
from streamtimelens.evaluation.snag_benchmark import (
    SnAGDatasetAssets,
    audit_snag_assets,
    snag_protocol_audit,
)


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "third_party/TimeLens/data/TimeLens-Bench"
SNAG_ASSETS = ROOT / "artifacts/snag"


def _assets() -> list[SnAGDatasetAssets]:
    return [
        SnAGDatasetAssets(
            "charades", DATA / "charades-timelens.json", DATA / "videos/charades",
            SNAG_ASSETS / "charades/features", SNAG_ASSETS / "charades/model.pth",
            SNAG_ASSETS / "charades/glove", True,
        ),
        SnAGDatasetAssets(
            "activitynet", DATA / "activitynet-timelens.json", DATA / "videos/activitynet",
            SNAG_ASSETS / "activitynet/features", SNAG_ASSETS / "activitynet/model.pth",
            SNAG_ASSETS / "activitynet/glove", True,
        ),
        SnAGDatasetAssets(
            "qvhighlights", DATA / "qvhighlights-timelens.json", DATA / "videos/qvhighlights",
            SNAG_ASSETS / "qvhighlights/features", SNAG_ASSETS / "qvhighlights/model.pth",
            SNAG_ASSETS / "qvhighlights/text", False,
        ),
        SnAGDatasetAssets(
            "mad", SNAG_ASSETS / "mad/annotations.json", SNAG_ASSETS / "mad/videos",
            SNAG_ASSETS / "mad/features", SNAG_ASSETS / "mad/model.pth",
            SNAG_ASSETS / "mad/text", True,
        ),
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="results/snag_adapt/formal-readiness-20260914")
    args = parser.parse_args()
    output = versioned_result_path(args.output)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite readiness audit: {output}")
    readiness = audit_snag_assets(_assets())
    protocol = snag_protocol_audit(readiness)
    configuration = {
        "method": "SnAG-adapt",
        "scope": "BENCHMARKS.md section 8.6",
        "datasets": [row.name for row in _assets()],
        "arrival_ratios": readiness["arrival_ratios"],
        "asset_root": str(SNAG_ASSETS.resolve()),
        "formal_metrics_enabled": False,
    }
    write_provenance(
        output,
        configuration=configuration,
        config_filename="config.resolved.json",
    )
    (output / "readiness_summary.json").write_text(
        json.dumps(readiness, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output / "protocol_audit.json").write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_artifact_manifest(output)
    print(json.dumps({
        "output": str(output),
        "ready_dataset_count": readiness["ready_dataset_count"],
        "formal_metrics_emitted": False,
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

