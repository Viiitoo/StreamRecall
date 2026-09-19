#!/usr/bin/env python3
"""Write configs/frozen_v1.yaml only after all independent-dev gates pass."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT / "src"))

from streamtimelens.config import read_resolved_config
from streamtimelens.evaluation.freeze import freeze_dev_configuration


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--oracle-metrics", type=Path, required=True)
    parser.add_argument("--writer-metrics", type=Path, required=True)
    parser.add_argument("--resolved-config", action="append", required=True, metavar="ID=PATH")
    parser.add_argument("--output", type=Path, default=PACKAGE_ROOT / "configs" / "frozen_v1.yaml")
    parser.add_argument("--decision-record", type=Path, default=PACKAGE_ROOT / "docs" / "frozen_v1_decision.md")
    args = parser.parse_args()
    configs = {}
    for item in args.resolved_config:
        config_id, separator, raw_path = item.partition("=")
        if not separator or not config_id:
            parser.error("--resolved-config must use ID=PATH")
        configs[config_id] = read_resolved_config(raw_path)
    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    oracle = json.loads(args.oracle_metrics.read_text(encoding="utf-8"))
    writer = json.loads(args.writer_metrics.read_text(encoding="utf-8"))
    frozen = freeze_dev_configuration(configs, selection, oracle, writer)
    if args.output.exists():
        raise FileExistsError(f"frozen config already exists: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(yaml.safe_dump(frozen, sort_keys=True), encoding="utf-8")
    args.decision_record.parent.mkdir(parents=True, exist_ok=True)
    args.decision_record.write_text(
        "# frozen_v1 decision\n\n"
        f"Selected configs: {', '.join(sorted(frozen['configs']))}\n\n"
        f"Selected writer: {frozen['selected_writer']}\n\n"
        f"Policy: {frozen['post_freeze_policy']}\n",
        encoding="utf-8",
    )
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
