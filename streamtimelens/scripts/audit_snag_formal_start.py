#!/usr/bin/env python3
"""Audit the per-dataset gate for starting SnAG formal inference."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
WORK_ROOT = PACKAGE_ROOT.parent
for source_root in (PACKAGE_ROOT / "src", WORK_ROOT / "src"):
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))

from baas.provenance import (  # noqa: E402
    versioned_result_path,
    write_artifact_manifest,
    write_provenance,
)
from streamtimelens.evaluation.snag_benchmark import (  # noqa: E402
    SnAGFormalStartAssets,
    audit_snag_formal_start,
)


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen-config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dev-effect-gate", type=Path, required=True)
    parser.add_argument("--dev-protocol-audit", type=Path, required=True)
    parser.add_argument("--g0-parity-gate", type=Path, required=True)
    parser.add_argument("--g1-upper-bound-gate", type=Path, required=True)
    parser.add_argument("--g2-diagnostic-gate", type=Path, required=True)
    parser.add_argument("--output", default="results/snag_adapt/formal-start-audit")
    return parser.parse_args()


def main() -> int:
    args = _args()
    data = WORK_ROOT / "third_party/TimeLens/data/TimeLens-Bench"
    snag = WORK_ROOT / "artifacts/snag"
    common = {
        "frozen_config": args.frozen_config,
        "checkpoint": args.checkpoint,
        "dev_effect_gate": args.dev_effect_gate,
        "dev_protocol_audit": args.dev_protocol_audit,
        "g0_parity_gate": args.g0_parity_gate,
        "g1_upper_bound_gate": args.g1_upper_bound_gate,
        "g2_diagnostic_gate": args.g2_diagnostic_gate,
    }
    assets = [
        SnAGFormalStartAssets("charades", data / "charades-timelens.json", data / "videos/charades", **common),
        SnAGFormalStartAssets("activitynet", data / "activitynet-timelens.json", data / "videos/activitynet", **common),
        SnAGFormalStartAssets("qvhighlights", data / "qvhighlights-timelens.json", data / "videos/qvhighlights", **common),
        SnAGFormalStartAssets("mad", snag / "mad/annotations.json", snag / "mad/videos", **common),
    ]
    clean = not subprocess.check_output(
        ["git", "-C", str(WORK_ROOT), "status", "--porcelain=v1"], text=True,
    ).strip()
    report = audit_snag_formal_start(assets, worktree_clean=clean)
    output = versioned_result_path(args.output)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite formal-start audit: {output}")
    write_provenance(output, configuration={
        "method": "SnAG-adapt-pooled-B",
        "frozen_config": str(args.frozen_config.resolve()),
        "checkpoint": str(args.checkpoint.resolve()),
        "formal_metrics_enabled": False,
    })
    (output / "formal_start_audit.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    write_artifact_manifest(output)
    print(json.dumps({
        "output": str(output),
        "ready_datasets": report["ready_datasets"],
        "any_formal_test_can_start": report["any_formal_test_can_start"],
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
