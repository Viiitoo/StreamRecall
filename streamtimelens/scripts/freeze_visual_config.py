#!/usr/bin/env python3
"""Write frozen_visual_v1.yaml after visual audit, selection, and refiner decision."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT / "src"))

from streamtimelens.config import read_resolved_config
from streamtimelens.evaluation.freeze import freeze_visual_configuration


def _read_mapping(path: Path) -> dict:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a mapping: {path}")
    return value


def _id_paths(values: list[str], option: str) -> dict[str, Path]:
    result = {}
    for item in values:
        config_id, separator, raw_path = item.partition("=")
        if not separator or not config_id or config_id in result:
            raise ValueError(f"{option} must use unique ID=PATH values")
        result[config_id] = Path(raw_path)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol-audit", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--refiner-gate", type=Path, required=True)
    parser.add_argument("--resolved-config", action="append", required=True, metavar="ID=PATH")
    parser.add_argument("--visual-config", action="append", required=True, metavar="ID=PATH")
    parser.add_argument(
        "--output", type=Path,
        default=PACKAGE_ROOT / "configs" / "frozen_visual_v1.yaml",
    )
    parser.add_argument(
        "--decision-record", type=Path,
        default=PACKAGE_ROOT / "docs" / "frozen_visual_v1_decision.md",
    )
    args = parser.parse_args()
    try:
        resolved_paths = _id_paths(args.resolved_config, "--resolved-config")
        visual_paths = _id_paths(args.visual_config, "--visual-config")
    except ValueError as exc:
        parser.error(str(exc))
    frozen = freeze_visual_configuration(
        {key: read_resolved_config(path) for key, path in resolved_paths.items()},
        _read_mapping(args.selection),
        _read_mapping(args.protocol_audit),
        _read_mapping(args.refiner_gate),
        {key: _read_mapping(path) for key, path in visual_paths.items()},
    )
    if args.output.exists():
        raise FileExistsError(f"frozen visual config already exists: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(yaml.safe_dump(frozen, sort_keys=True), encoding="utf-8")
    args.decision_record.parent.mkdir(parents=True, exist_ok=True)
    args.decision_record.write_text(
        "# frozen_visual_v1 decision\n\n"
        f"Selected configs: {', '.join(sorted(frozen['configs']))}\n\n"
        f"TimeLens refiner: {frozen['refiner_decision']}\n\n"
        f"Policy: {frozen['post_freeze_policy']}\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(args.output), "configs": sorted(frozen["configs"])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
