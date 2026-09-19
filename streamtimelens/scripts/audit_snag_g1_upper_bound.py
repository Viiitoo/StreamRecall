#!/usr/bin/env python3
"""Audit a completed SnAG full-store PQ upper-bound run."""

from __future__ import annotations

import argparse
import json
import math
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
from streamtimelens.baselines.snag_config import (  # noqa: E402
    load_snag_config,
    resolved_snag_config_dict,
)


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _correlation(xs: list[float], ys: list[float]) -> float:
    if len(xs) != len(ys) or len(xs) < 2:
        return float("nan")
    mean_x, mean_y = sum(xs) / len(xs), sum(ys) / len(ys)
    numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    denominator = math.sqrt(
        sum((x - mean_x) ** 2 for x in xs) * sum((y - mean_y) ** 2 for y in ys)
    )
    return numerator / denominator if denominator else float("nan")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--snapshot-index", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--protocol-audit", type=Path, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = load_snag_config(args.config)
    snapshots = _jsonl(args.snapshot_index)
    metrics = json.loads(args.metrics.read_text(encoding="utf-8"))
    protocol = json.loads(args.protocol_audit.read_text(encoding="utf-8"))
    final = [row for row in snapshots if abs(float(row["rho"]) - 1.0) < 1e-8]
    correlation = _correlation(
        [float(row["t_q"]) for row in final],
        [float(row["state_bytes"]) for row in final],
    )
    overall = metrics.get("overall", {})
    checks = {
        "full_store_unbudgeted": config.writer.mode == "full" and config.budget_bytes is None,
        "upper_bound_classification": config.classification == "upper-bound",
        "runtime_protocol_passed": bool(protocol.get("passed")),
        "complete_snapshot_index": bool(final),
        "state_grows_with_duration": math.isfinite(correlation) and correlation > 0.9,
        "finite_non_degenerate_metrics": all(
            name in overall and math.isfinite(float(overall[name]))
            for name in ("miou", "r1_iou_0.3", "r5_iou_0.3")
        ) and float(overall.get("r5_iou_0.3", 0)) > 0,
    }
    report = {
        "schema_version": 1,
        "gate": "SnAG-G1-full-store-PQ-upper-bound",
        "passed": all(checks.values()),
        "checks": checks,
        "state_bytes_duration_correlation": correlation,
    }
    output = versioned_result_path(args.output)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite G1 audit: {output}")
    write_provenance(
        output, configuration=resolved_snag_config_dict(config),
        config_path=args.config, config_filename="config.resolved.yaml",
    )
    (output / "g1_upper_bound_gate.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    write_artifact_manifest(output)
    print(json.dumps({"output": str(output), "passed": report["passed"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
