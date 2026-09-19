#!/usr/bin/env python3
"""Run the non-streaming official TimeLens reference with per-query video reloads."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT / "src"))
sys.path.insert(0, str(PACKAGE_ROOT.parent / "src"))

from baas.provenance import versioned_result_path, write_provenance
from streamtimelens.baselines.offline_timelens import OfflineQuery, OfflineTimeLensWrapper
from streamtimelens.baselines.offline_resume import (
    load_jsonl, queries_for_shard, sha256_file, sha256_model,
    validate_prediction_rows, validate_resume_source,
)
from streamtimelens.refiner.model_service import TimeLensModelService


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--resume-from", type=Path,
        help="Import a validated prediction prefix from another revision/result directory.",
    )
    args = parser.parse_args()
    if args.num_shards <= 0 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("offline shard index/count are invalid")
    if args.resume and args.resume_from is not None:
        raise ValueError("use either --resume or --resume-from, not both")
    manifest_rows = load_jsonl(args.queries)
    if len({str(row["query_id"]) for row in manifest_rows}) != len(manifest_rows):
        raise ValueError("offline query manifest contains duplicate query IDs")
    rows = list(queries_for_shard(manifest_rows, args.num_shards, args.shard_index))
    queries = [OfflineQuery(
        str(row["query_id"]), str(row["video_id"]), str(row["query"]),
        Path(row["video_path"]).expanduser().resolve(),
    ) for row in rows]
    query_manifest_sha256 = sha256_file(args.queries)
    model_sha256 = sha256_model(args.model)
    output_root = versioned_result_path(args.output)
    output_root.mkdir(parents=True, exist_ok=True)
    destination = output_root / "predictions.jsonl"
    imported_rows = []
    import_metadata = None
    if args.resume_from is not None:
        if destination.exists():
            raise FileExistsError(
                f"offline destination already exists; --resume-from only creates a new result: {destination}"
            )
        imported_rows, import_metadata = validate_resume_source(
            args.resume_from, model_sha256=model_sha256,
            query_manifest_sha256=query_manifest_sha256, queries=manifest_rows,
            num_shards=args.num_shards, shard_index=args.shard_index,
        )
        destination.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in imported_rows),
            encoding="utf-8",
        )
    configuration = {
        "protocol": "offline_upper_bound_not_streaming",
        "queries": str(args.queries.resolve()),
        "query_manifest_sha256": query_manifest_sha256,
        "model": str(args.model.resolve()), "model_sha256": model_sha256,
        "device_map": args.device_map, "per_query_full_video_reload": True,
        "num_shards": args.num_shards, "shard_index": args.shard_index,
        "resume": args.resume, "resume_from": import_metadata,
    }
    config_path = output_root / "config.resolved.json"
    if args.resume:
        if not destination.exists() or not config_path.exists():
            raise FileNotFoundError(f"offline result cannot resume without config and predictions: {output_root}")
        previous = json.loads(config_path.read_text(encoding="utf-8"))
        for key in ("protocol", "query_manifest_sha256", "model_sha256", "num_shards", "shard_index"):
            if previous.get(key) != configuration[key]:
                raise ValueError(f"offline result resume mismatch for {key}")
    else:
        write_provenance(
            output_root, configuration=configuration, config_filename="config.resolved.json",
            extra_metadata={"resume_import": import_metadata} if import_metadata else None,
        )
    completed = set()
    if destination.exists():
        if not (args.resume or args.resume_from is not None):
            raise FileExistsError(f"offline predictions already exist; use --resume: {destination}")
        completed = validate_prediction_rows(
            load_jsonl(destination), manifest_rows,
            num_shards=args.num_shards, shard_index=args.shard_index,
        )
    wrapper = OfflineTimeLensWrapper(
        TimeLensModelService.get(args.model, device_map=args.device_map),
    )
    executed = 0
    with destination.open("a", encoding="utf-8") as stream:
        for query in queries:
            if query.query_id in completed:
                continue
            prediction = wrapper.run([query])[0]
            stream.write(json.dumps(asdict(prediction), ensure_ascii=False) + "\n")
            stream.flush()
            executed += 1
    print(json.dumps({
        "queries": len(queries), "executed": executed, "resumed": len(completed),
        "imported": len(imported_rows), "predictions": str(destination),
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
