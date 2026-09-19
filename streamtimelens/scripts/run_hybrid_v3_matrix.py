#!/usr/bin/env python3
"""Run a resumable, sharded Hybrid V3 matrix from query-safe manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import yaml

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
WORK_ROOT = PACKAGE_ROOT.parent
sys.path.insert(0, str(PACKAGE_ROOT / "src"))
sys.path.insert(0, str(WORK_ROOT / "src"))

from baas.provenance import versioned_result_path, write_artifact_manifest, write_provenance
from streamtimelens.baselines.offline_resume import load_jsonl, sha256_file
from streamtimelens.config import read_hybrid_v3_config
from streamtimelens.model_identity import verify_registered_model
from streamtimelens.observer.clip_encoder import FrozenCLIPEncoder
from streamtimelens.protocol.snapshot import SnapshotReader
from streamtimelens.refiner.model_service import TimeLensModelService
from streamtimelens.refiner.multicandidate_prompt import multicandidate_template_sha256
from streamtimelens.retrieval.clip_timelens_hybrid import answer_hybrid_snapshot


_FORBIDDEN = {"gt", "gt_span", "ground_truth", "ground_truth_span", "video_path", "video"}


def _shard(sample_id: str, count: int) -> int:
    return int(hashlib.sha256(sample_id.encode("utf-8")).hexdigest(), 16) % count


def _asset_path(value: str) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else WORK_ROOT / path).resolve()


def _output_root(path: Path) -> Path:
    resolved = versioned_result_path(path)
    relative = resolved.relative_to((WORK_ROOT / "results").resolve())
    if len(relative.parts) < 3 or relative.parts[1] != "hybrid_v3":
        raise ValueError("Hybrid V3 matrix output must be under results/hybrid_v3/<run-name>")
    return resolved


def _validate_queries(rows, rhos):
    jobs = []
    seen = set()
    for row in rows:
        forbidden = _FORBIDDEN & {str(key).lower() for key in row}
        if forbidden:
            raise ValueError(f"query manifest exposes forbidden fields: {sorted(forbidden)}")
        query_id = str(row.get("query_id", ""))
        video_id = str(row.get("video_id", ""))
        query = str(row.get("query", ""))
        if not query_id or not video_id or not query.strip():
            raise ValueError("query manifest rows need query_id, video_id, and query")
        row_rhos = (float(row["rho_q"]),) if "rho_q" in row else rhos
        for rho in row_rhos:
            if not 0 < rho <= 1:
                raise ValueError("query arrival ratios must be in (0,1]")
            sample_id = f"{query_id}@{rho:.2f}"
            if sample_id in seen:
                raise ValueError(f"duplicate Hybrid V3 sample: {sample_id}")
            seen.add(sample_id)
            jobs.append((sample_id, query_id, video_id, query, rho))
    return jobs


def _summaries(rows, max_frames):
    calls = 0
    generated = 0
    query_wall = 0.0
    total_query_wall = 0.0
    clip_wall = 0.0
    clip_gpu = 0.0
    peak = 0
    snapshot_bytes = []
    violations = []
    fallbacks = 0
    for row in rows:
        fallbacks += int(bool(row.get("fallback_used")))
        refs = row.get("selected_frame_refs", [])
        if len(refs) != len(set(refs)) or len(refs) > max_frames:
            violations.append({"sample_id": row.get("sample_id"), "kind": "frame_budget"})
        inference = [
            event for event in row.get("query_trace", [])
            if event.get("kind") in ("timelens_inference", "timelens_exception")
        ]
        configuration = next(
            (event for event in row.get("query_trace", []) if event.get("kind") == "hybrid_configuration"),
            {},
        )
        if configuration.get("snapshot_state_bytes") is not None:
            snapshot_bytes.append(int(configuration["snapshot_state_bytes"]))
        retrieval = next(
            (event for event in row.get("query_trace", []) if event.get("kind") == "clip_retrieval"),
            {},
        )
        clip_wall += float(retrieval.get("clip_query_wall_s") or 0)
        clip_gpu += sum(
            float(resource.get("cuda_s") or 0)
            for resource in retrieval.get("clip_resources", [])
        )
        completed_event = next(
            (event for event in row.get("query_trace", []) if event.get("kind") == "query_completed"),
            {},
        )
        total_query_wall += float(completed_event.get("wall_s") or 0)
        calls += sum(int(event.get("calls", 0)) for event in inference)
        if sum(int(event.get("calls", 0)) for event in inference) > 1:
            violations.append({"sample_id": row.get("sample_id"), "kind": "multiple_model_calls"})
        for event in inference:
            generated += int(event.get("generated_tokens") or 0)
            query_wall += float(event.get("wall_s") or 0)
            peak = max(peak, int(event.get("peak_cuda_bytes") or 0))
    audit = {
        "schema_version": 1, "samples": len(rows), "passed": not violations,
        "snapshot_only": True, "query_manifest_ground_truth_free": True,
        "max_one_model_call_per_query": not any(item["kind"] == "multiple_model_calls" for item in violations),
        "max_unique_frames": max_frames, "violations": violations,
    }
    cost = {
        "schema_version": 1, "samples": len(rows), "timelens_calls": calls,
        "generated_tokens": generated, "timelens_wall_s": query_wall,
        "mean_timelens_wall_s": query_wall / len(rows) if rows else 0.0,
        "clip_query_wall_s": clip_wall,
        "clip_query_gpu_s": clip_gpu,
        "total_query_wall_s": total_query_wall,
        "mean_total_query_wall_s": total_query_wall / len(rows) if rows else 0.0,
        "peak_cuda_bytes": peak, "fallback_count": fallbacks,
        "fallback_rate": fallbacks / len(rows) if rows else 0.0,
        "mean_snapshot_state_bytes": sum(snapshot_bytes) / len(snapshot_bytes) if snapshot_bytes else None,
        "max_snapshot_state_bytes": max(snapshot_bytes) if snapshot_bytes else None,
    }
    return audit, cost


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    snapshots = parser.add_mutually_exclusive_group(required=True)
    snapshots.add_argument("--snapshot-root", type=Path)
    snapshots.add_argument(
        "--snapshot-map", type=Path,
        help="JSON object mapping video_id to its directory containing rho_* snapshots.",
    )
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--rhos", default="0.25,0.50,0.75,1.00")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--clip-device")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.num_shards <= 0 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("invalid shard selection")
    rhos = tuple(float(value) for value in args.rhos.split(",") if value)
    rows = load_jsonl(args.queries)
    jobs = [
        job for job in _validate_queries(rows, rhos)
        if _shard(job[0], args.num_shards) == args.shard_index
    ]
    if not jobs:
        raise ValueError("Hybrid V3 shard contains no queries")
    config = read_hybrid_v3_config(args.config)
    frozen_path = PACKAGE_ROOT / "configs" / "frozen_visual_v2.yaml"
    if sha256_file(frozen_path) != config.snapshot_source.frozen_config_sha256:
        raise ValueError("registered frozen_visual_v2.yaml hash does not match the checkout")
    query_sha = sha256_file(args.queries)
    snapshot_map = None
    snapshot_map_sha = None
    if args.snapshot_map is not None:
        snapshot_map_sha = sha256_file(args.snapshot_map)
        snapshot_map = json.loads(args.snapshot_map.read_text(encoding="utf-8"))
        if not isinstance(snapshot_map, dict) or not snapshot_map:
            raise ValueError("snapshot map must be a non-empty JSON object")
        expected_videos = {job[2] for job in jobs}
        if set(snapshot_map) != expected_videos:
            raise ValueError("snapshot map video IDs do not exactly match this shard")
        snapshot_map = {
            str(video_id): str(Path(path).expanduser().resolve())
            for video_id, path in snapshot_map.items()
        }
    output_root = _output_root(args.output)
    output_root.mkdir(parents=True, exist_ok=True)
    destination = output_root / "predictions.jsonl"
    configuration = {
        **config.canonical_dict(),
        "runtime": {
            "snapshot_root": str(args.snapshot_root.resolve()) if args.snapshot_root else None,
            "snapshot_map": snapshot_map,
            "snapshot_map_source": str(args.snapshot_map.resolve()) if args.snapshot_map else None,
            "snapshot_map_sha256": snapshot_map_sha,
            "query_manifest": str(args.queries.resolve()),
            "query_manifest_sha256": query_sha,
            "rhos": list(rhos), "num_shards": args.num_shards,
            "shard_index": args.shard_index, "clip_device": args.clip_device,
            "device_map": args.device_map,
            "prompt_template_sha256": multicandidate_template_sha256(
                config.brain.prompt_version,
            ),
        },
    }
    completed = set()
    if destination.exists():
        if not args.resume or not (output_root / "config.resolved.yaml").is_file():
            raise FileExistsError(f"Hybrid V3 output exists; use --resume: {destination}")
        previous = yaml.safe_load((output_root / "config.resolved.yaml").read_text(encoding="utf-8"))
        if previous != configuration:
            raise ValueError("Hybrid V3 resume configuration mismatch")
        existing = load_jsonl(destination)
        completed = {str(row.get("sample_id", "")) for row in existing}
        if len(completed) != len(existing) or not completed <= {job[0] for job in jobs}:
            raise ValueError("Hybrid V3 resume predictions are duplicate or outside the shard")
    clip_path = _asset_path(config.retrieval.clip_model)
    model_path = _asset_path(config.brain.model)
    registry_path = WORK_ROOT / "artifacts" / "dev" / "model_hashes.json"
    clip_identity = verify_registered_model(
        registry_path, model_key="clip", model_root=clip_path,
        revision=config.retrieval.clip_revision,
        content_sha256=config.retrieval.clip_content_sha256, verify_files=True,
    )
    timelens_identity = verify_registered_model(
        registry_path, model_key=config.brain.model_registry_key, model_root=model_path,
        revision=config.brain.model_revision,
        content_sha256=config.brain.model_content_sha256, verify_files=False,
    )
    encoder = FrozenCLIPEncoder(clip_path, device=args.clip_device, batch_size=1)
    service = TimeLensModelService.get(model_path, device_map=args.device_map)
    if not destination.exists():
        write_provenance(
            output_root, configuration=configuration, config_path=args.config,
            config_filename="config.resolved.yaml",
            extra_metadata={
                "hybrid_config_sha256": config.sha256,
                "query_manifest_sha256": query_sha,
                "model_service_hashes": dict(service.hashes),
                "clip_identity": clip_identity,
                "timelens_identity": timelens_identity,
            },
        )
    executed = 0
    with destination.open("a", encoding="utf-8") as stream:
        for sample_id, query_id, video_id, query, rho in jobs:
            if sample_id in completed:
                continue
            snapshot_path = (
                Path(snapshot_map[video_id]) / f"rho_{rho:.2f}"
                if snapshot_map is not None
                else args.snapshot_root / video_id / f"rho_{rho:.2f}"
            )
            snapshot = SnapshotReader(snapshot_path)
            output = answer_hybrid_snapshot(
                query_id=query_id, query=query, snapshot=snapshot,
                encoder=encoder, service=service, config=config,
            )
            payload = {"sample_id": sample_id, **output.to_dict()}
            stream.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
            stream.flush()
            executed += 1
    completed_rows = load_jsonl(destination)
    audit, cost = _summaries(completed_rows, config.brain.max_unique_frames)
    (output_root / "protocol_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    (output_root / "cost_summary.json").write_text(
        json.dumps(cost, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    (output_root / "accuracy_summary.json").write_text(json.dumps({
        "schema_version": 1, "status": "not_evaluated",
        "reason": "query worker is ground-truth-free; run Hybrid V3 evaluation separately",
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_artifact_manifest(output_root)
    print(json.dumps({
        "samples": len(jobs), "executed": executed, "resumed": len(completed),
        "passed_protocol_audit": audit["passed"], "output_root": str(output_root),
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
