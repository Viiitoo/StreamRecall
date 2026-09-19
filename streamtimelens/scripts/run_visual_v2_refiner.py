#!/usr/bin/env python3
"""Run sharded TimeLens refinement on independent-dev candidate predictions without GT."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path

import yaml

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT / "src"))
sys.path.insert(0, str(PACKAGE_ROOT.parent / "src"))

from baas.provenance import versioned_result_path, write_provenance
from streamtimelens.baselines.offline_resume import load_jsonl, sha256_model
from streamtimelens.evaluation.visual_v2 import expanded_span
from streamtimelens.protocol.snapshot import SnapshotReader
from streamtimelens.refiner.model_service import TimeLensModelService
from streamtimelens.retrieval.frame_candidates import (
    FrameCandidate, RetrievedFrame, coarse_prediction, refine_frame_candidate,
)


def _hash(value) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _shard(sample_id: str, count: int) -> int:
    return int(hashlib.sha256(sample_id.encode("utf-8")).hexdigest(), 16) % count


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix-root", type=Path, required=True)
    parser.add_argument("--source-config-id", required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--candidate-margin-s", type=float, required=True)
    parser.add_argument("--coarse-margin-s", type=float, required=True)
    parser.add_argument("--max-frames", type=int, default=16)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (
        args.num_shards <= 0 or not 0 <= args.shard_index < args.num_shards
        or args.candidate_margin_s <= 0 or args.coarse_margin_s < 0
        or args.max_frames <= 0
    ):
        raise ValueError("visual v2 refiner parameters are invalid")
    jobs = {}
    query_cache = {}
    selected = []
    sample_ids = set()
    for path in sorted(args.matrix_root.glob("**/predictions/*.rho_*.qg_*.json")):
        job_root = path.parent.parent
        if job_root not in jobs:
            jobs[job_root] = json.loads(
                (job_root / "config.resolved.json").read_text(encoding="utf-8")
            )
        job = jobs[job_root]
        if "dev" not in str(job["dataset"]).lower():
            raise ValueError(f"visual v2 refiner source is not independent dev: {job['dataset']}")
        if Path(job["method"]).stem != "uniform_raw":
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        variant = next(
            row for index, row in enumerate(job["query_grid"])
            if payload["query_variant"] == f"qg_{index:02d}_{_hash(row)[:8]}"
        )
        budget = int(yaml.safe_load(Path(job["budget"]).read_text())["memory_bytes"])
        descriptor = {"method": "uniform_raw", "budget_bytes": budget, **variant}
        source_id = f"uniform_raw-{budget}-{_hash(descriptor)[:12]}"
        if source_id != args.source_config_id:
            continue
        if job_root not in query_cache:
            query_cache[job_root] = {
                str(row["query_id"]): str(row["query"])
                for row in load_jsonl(job_root / "queries.input.jsonl")
            }
        query_id = str(payload["query_id"])
        if query_id not in query_cache[job_root]:
            raise ValueError(f"source prediction query text is missing: {path}")
        sample_id = f"{query_id}@{float(payload['rho_q']):.2f}"
        if sample_id in sample_ids:
            raise ValueError(f"duplicate visual v2 refiner source sample: {sample_id}")
        sample_ids.add(sample_id)
        if _shard(sample_id, args.num_shards) == args.shard_index:
            selected.append((sample_id, path, job_root, payload, query_cache[job_root][query_id]))
    if not selected:
        raise ValueError("visual v2 refiner source config/shard has no predictions")
    model_sha256 = sha256_model(args.model)
    output_root = versioned_result_path(args.output)
    output_root.mkdir(parents=True, exist_ok=True)
    configuration = {
        "protocol": "independent_dev_candidate_refiner_gate",
        "ground_truth_visible": False, "matrix_root": str(args.matrix_root.resolve()),
        "source_config_id": args.source_config_id,
        "model": str(args.model.resolve()), "model_sha256": model_sha256,
        "candidate_margin_s": args.candidate_margin_s,
        "coarse_margin_s": args.coarse_margin_s, "max_frames": args.max_frames,
        "device_map": args.device_map,
        "num_shards": args.num_shards, "shard_index": args.shard_index,
    }
    destination = output_root / "predictions.jsonl"
    config_path = output_root / "config.resolved.json"
    completed = set()
    if destination.exists():
        if not args.resume or not config_path.exists():
            raise FileExistsError(f"visual v2 refiner output exists; use --resume: {destination}")
        previous = json.loads(config_path.read_text(encoding="utf-8"))
        for key, value in configuration.items():
            if previous.get(key) != value:
                raise ValueError(f"visual v2 refiner resume mismatch for {key}")
        existing = load_jsonl(destination)
        completed = {str(row.get("sample_id")) for row in existing}
        if len(completed) != len(existing) or not completed <= {row[0] for row in selected}:
            raise ValueError("visual v2 refiner resume predictions are duplicate or out of shard")
    else:
        write_provenance(
            output_root, configuration=configuration, config_filename="config.resolved.json",
        )
    service = TimeLensModelService.get(args.model, device_map=args.device_map)
    executed = 0
    with destination.open("a", encoding="utf-8") as stream:
        for sample_id, source_path, job_root, payload, query in selected:
            if sample_id in completed:
                continue
            snapshot = SnapshotReader(
                job_root / "snapshots" / str(payload["video_id"])
                / f"rho_{float(payload['rho_q']):.2f}"
            )
            candidates = payload.get("candidates", [])
            coarse = coarse_prediction([], upper_bound_s=snapshot.manifest.t_q)
            model_prediction = coarse
            trace = {"kind": "refiner_fallback", "reason": "no_candidate"}
            if candidates:
                row = candidates[0]
                start_s, end_s = expanded_span(
                    (float(row["start_s"]), float(row["end_s"])),
                    args.candidate_margin_s, snapshot.manifest.t_q,
                )
                candidate = FrameCandidate(
                    f"{row['candidate_id']}-margin-{args.candidate_margin_s:g}",
                    start_s, end_s, float(row["score"]),
                    tuple(map(str, row["frame_refs"])), tuple(map(str, row["hit_refs"])),
                    tuple(RetrievedFrame(**item) for item in row.get("retrieved_frames", [])),
                )
                coarse = coarse_prediction(
                    [candidate], margin_s=args.coarse_margin_s,
                    upper_bound_s=snapshot.manifest.t_q,
                )
                try:
                    model_prediction, trace = refine_frame_candidate(
                        query, snapshot, candidate, service, max_frames=args.max_frames,
                    )
                except Exception as exc:
                    trace = {
                        "kind": "refiner_fallback", "reason": type(exc).__name__,
                        "message": str(exc), "candidate_span": [start_s, end_s],
                    }
            final = model_prediction if model_prediction.status == "ok" else coarse
            result = {
                "sample_id": sample_id, "query_id": payload["query_id"],
                "video_id": payload["video_id"], "rho_q": payload["rho_q"],
                "source_config_id": args.source_config_id,
                "source_prediction": str(source_path.resolve()),
                "coarse_span": list(coarse.span) if coarse.span else None,
                "model_span": list(model_prediction.span) if model_prediction.span else None,
                "final_span": list(final.span) if final.span else None,
                "model_status": model_prediction.status, "final_status": final.status,
                "trace": trace, "prediction": asdict(final),
            }
            stream.write(json.dumps(result, ensure_ascii=False) + "\n")
            stream.flush()
            executed += 1
    print(json.dumps({
        "samples": len(selected), "executed": executed, "resumed": len(completed),
        "predictions": str(destination),
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
