#!/usr/bin/env python3
"""Prepare leakage-separated SnAG development manifests and arrival plan."""

from __future__ import annotations

import argparse
import hashlib
import json
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
from streamtimelens.baselines.snag_training import join_development_manifests  # noqa: E402
from streamtimelens.protocol.arrival import build_arrival_plan, write_arrival_plan  # noqa: E402


def _jsonl(path: Path) -> list[dict[str, object]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if not rows:
        raise ValueError(f"empty SnAG development manifest: {path}")
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--videos", type=Path, required=True)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--arrival-ratios", type=float, nargs="+", default=(0.25, 0.5, 0.75, 1.0),
    )
    args = parser.parse_args()
    videos = _jsonl(args.videos)
    queries = _jsonl(args.queries)
    annotations = _jsonl(args.annotations)
    joined = join_development_manifests(videos, queries, annotations)
    output = versioned_result_path(args.output)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite SnAG development preparation: {output}")
    output.mkdir(parents=True)
    (output / "training_queries.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in joined),
        encoding="utf-8",
    )
    gt_free = [{
        "video_id": row["video_id"], "query_id": row["query_id"], "query": row["query"],
    } for row in joined]
    (output / "queries.gt-free.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in gt_free),
        encoding="utf-8",
    )
    arrival_digest = write_arrival_plan(
        build_arrival_plan(joined, ratios=args.arrival_ratios), output / "arrival_plan.jsonl",
    )
    inputs = {name: {
        "path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    } for name, path in {
        "videos": args.videos, "queries": args.queries, "annotations": args.annotations,
    }.items()}
    write_provenance(
        output,
        configuration={
            "task": "snag-development-manifest-preparation",
            "arrival_ratios": list(args.arrival_ratios),
        },
        config_filename="config.resolved.json",
        extra_metadata={
            "inputs": inputs,
            "video_count": len(videos),
            "query_count": len(joined),
            "arrival_plan_sha256": arrival_digest,
        },
    )
    write_artifact_manifest(output)
    print(json.dumps({"output": str(output), "videos": len(videos), "queries": len(joined)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
