#!/usr/bin/env python3
"""Batch all visual dev queries/rhos/grid variants with one resident CLIP model."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT / "src"))
sys.path.insert(0, str(PACKAGE_ROOT.parent / "src"))

from baas.provenance import versioned_result_path, write_provenance
from streamtimelens.observer.clip_encoder import DEFAULT_CLIP_MODEL, FrozenCLIPEncoder
from streamtimelens.protocol.snapshot import SnapshotReader
from streamtimelens.retrieval.frame_candidates import build_frame_candidates, coarse_prediction


def _hash(value) -> str:
    text = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-root", type=Path, required=True)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--query-grid", type=Path, required=True)
    parser.add_argument("--rhos", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--clip-model", default=DEFAULT_CLIP_MODEL)
    parser.add_argument("--clip-revision")
    parser.add_argument("--clip-sha256")
    parser.add_argument("--clip-device")
    args = parser.parse_args()
    queries = [json.loads(line) for line in args.queries.read_text(encoding="utf-8").splitlines()]
    query_grid = json.loads(args.query_grid.read_text(encoding="utf-8"))
    rhos = tuple(float(value) for value in args.rhos.split(","))
    if not queries or not query_grid or not rhos:
        raise ValueError("visual query batch inputs must not be empty")
    encoder = FrozenCLIPEncoder(args.clip_model, device=args.clip_device, batch_size=1)
    embeddings = {}
    resources = {}
    for query in queries:
        embeddings[query["query_id"]] = encoder.encode_text(query["query"])
        resources[query["query_id"]] = encoder.resource_dicts()[-1]
    output_root = versioned_result_path(args.output)
    output_root.mkdir(parents=True, exist_ok=True)
    configuration = {
        "snapshot_root": str(args.snapshot_root.resolve()),
        "queries": str(args.queries.resolve()), "query_grid": query_grid, "rhos": rhos,
        "readout": "coarse_visual", "clip_model": args.clip_model,
        "clip_revision": args.clip_revision, "clip_sha256": args.clip_sha256,
    }
    write_provenance(output_root, configuration=configuration, config_filename="config.resolved.json")
    count = 0
    for rho in rhos:
        snapshot = SnapshotReader(args.snapshot_root / f"rho_{rho:.2f}")
        metadata = snapshot.read_frame_metadata()
        actual_rho = snapshot.manifest.t_q / float(snapshot.manifest.video_meta["duration_s"])
        for query in queries:
            for index, grid in enumerate(query_grid):
                candidates = build_frame_candidates(
                    embeddings[query["query_id"]], metadata,
                    top_k=int(grid["top_k"]), merge_gap_s=float(grid["merge_gap_s"]),
                    expand_neighbors=int(grid["expand_neighbors"]),
                    candidate_margin_s=float(grid.get("candidate_margin_s", 0)),
                    upper_bound_s=snapshot.manifest.t_q,
                )
                prediction = coarse_prediction(
                    candidates, margin_s=float(grid.get("coarse_margin_s", 0)),
                    upper_bound_s=snapshot.manifest.t_q,
                )
                variant = f"qg_{index:02d}_{_hash(grid)[:8]}"
                payload = {
                    **asdict(prediction), "query_id": query["query_id"],
                    "video_id": snapshot.manifest.video_id, "rho_q": actual_rho,
                    "evidence_ids": list(prediction.evidence_ids),
                    "candidate_ids": list(prediction.candidate_ids),
                    "candidates": [asdict(candidate) for candidate in candidates],
                    "query_trace": [{
                        "kind": "frame_retrieval", "candidate_count": len(candidates),
                        "retrieved_frames": [
                            asdict(hit) for candidate in candidates
                            for hit in candidate.retrieved_frames
                        ],
                        "clip_resource": [resources[query["query_id"]]],
                    }],
                    "readout": "coarse_visual",
                    "candidate_envelope": list(candidates[0].span) if candidates else None,
                    "final_selection_reason": (
                        "highest_scoring_temporal_cluster" if candidates else "no_candidate"
                    ),
                    "query_variant": variant,
                }
                path = output_root / f"{query['query_id']}.rho_{rho:.2f}.{variant}.json"
                path.write_text(json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8")
                count += 1
    print(json.dumps({"predictions": count, "output_root": str(output_root)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
