#!/usr/bin/env python3
"""Convert official OnVTG output into a separate query-known result table."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT / "src"))
sys.path.insert(0, str(PACKAGE_ROOT.parent / "src"))

from baas.provenance import versioned_result_path, write_provenance
from streamtimelens.baselines.onvtg_query_known import (
    QUERY_KNOWN_PROTOCOL, adapt_official_onvtg_rows, assert_protocol_table,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-predictions", type=Path, required=True)
    parser.add_argument("--repository-revision", required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--feature-revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source_rows = [
        json.loads(line) for line in args.official_predictions.read_text(encoding="utf-8").splitlines()
    ]
    rows = [item.to_dict() for item in adapt_official_onvtg_rows(source_rows)]
    assert_protocol_table(rows, QUERY_KNOWN_PROTOCOL)
    output_root = versioned_result_path(args.output)
    output_root.mkdir(parents=True, exist_ok=True)
    write_provenance(
        output_root,
        configuration={
            "protocol": QUERY_KNOWN_PROTOCOL,
            "source": str(args.official_predictions.resolve()),
            "repository_revision": args.repository_revision,
            "checkpoint_sha256": args.checkpoint_sha256,
            "feature_revision": args.feature_revision,
            "separate_from_delayed_query_main_table": True,
        },
        config_filename="config.resolved.json",
    )
    destination = output_root / "query_known_predictions.jsonl"
    destination.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8",
    )
    print(json.dumps({"protocol": QUERY_KNOWN_PROTOCOL, "count": len(rows), "output": str(destination)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
