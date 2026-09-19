#!/usr/bin/env python3
"""Answer a query from a snapshot.  This command has deliberately no video option."""

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
from streamtimelens.protocol.snapshot import SnapshotReader
from streamtimelens.config import read_resolved_config
from streamtimelens.observer.clip_encoder import DEFAULT_CLIP_MODEL, FrozenCLIPEncoder
from streamtimelens.refiner.model_service import TimeLensModelService
from streamtimelens.retrieval.frame_candidates import (
    coarse_prediction, refine_frame_candidate, retrieve_frame_candidates,
)
from streamtimelens.retrieval.ranker import locate_lexically
from streamtimelens.retrieval.embedder import load_snapshot_query_embedder
from streamtimelens.retrieval.query_pipeline import answer_card_snapshot


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--query", required=True)
    parser.add_argument("--query-id", default="adhoc-query")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--retrieval", choices=("auto", "lexical", "frame", "hybrid"), default="auto")
    parser.add_argument("--resolved-config", type=Path)
    parser.add_argument("--text-embedder")
    parser.add_argument("--text-embedder-revision")
    parser.add_argument("--clip-model", default=DEFAULT_CLIP_MODEL)
    parser.add_argument("--clip-revision")
    parser.add_argument("--clip-sha256")
    parser.add_argument("--clip-device")
    parser.add_argument("--timelens-model", type=Path)
    parser.add_argument(
        "--readout", choices=("auto", "coarse_visual", "timelens_local"), default="auto",
        help="Explicit visual readout; auto preserves the legacy model-present behavior.",
    )
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--merge-gap-s", type=float, default=4.0)
    parser.add_argument("--expand-neighbors", type=int, default=1)
    parser.add_argument("--candidate-margin-s", type=float, default=0.0)
    parser.add_argument("--coarse-margin-s", type=float, default=0.0)
    parser.add_argument("--max-refine-frames", type=int, default=16)
    args = parser.parse_args()
    snapshot = SnapshotReader(args.snapshot)
    readout = args.readout
    if readout == "auto":
        readout = "timelens_local" if args.timelens_model else "coarse_visual"
    if readout == "timelens_local" and args.timelens_model is None:
        parser.error("timelens_local readout requires --timelens-model")
    retrieval = args.retrieval
    if retrieval == "auto":
        retrieval = "hybrid" if snapshot.manifest.embedder is not None else (
            "frame"
            if snapshot.manifest.method in ("uniform_raw", "semantic_reservoir") and not snapshot.manifest.pixel_only
            else "lexical"
        )
    if retrieval == "frame" and snapshot.manifest.pixel_only:
        parser.error("frame retrieval is unavailable for a pixel-only snapshot")
    candidates = []
    query_trace = []
    if retrieval == "hybrid":
        if args.resolved_config is None:
            parser.error("hybrid retrieval requires --resolved-config")
        resolved = read_resolved_config(args.resolved_config)
        embedder = load_snapshot_query_embedder(
            snapshot, model_name_or_path=args.text_embedder or resolved.model.text_embedder,
            revision=args.text_embedder_revision or resolved.model.text_embedder_revision,
            device=args.clip_device,
        )
        service = TimeLensModelService.get(args.timelens_model) if args.timelens_model else None
        output = answer_card_snapshot(
            query_id=args.query_id, query=args.query, snapshot=snapshot, embedder=embedder,
            protocol=resolved.protocol, budget=resolved.budget, refiner_service=service,
        )
        payload = output.to_dict()
    elif retrieval == "lexical":
        prediction = locate_lexically(args.query, snapshot)
    else:
        encoder = FrozenCLIPEncoder(args.clip_model, device=args.clip_device, batch_size=1)
        candidates = retrieve_frame_candidates(
            args.query, snapshot, encoder, top_k=args.top_k,
            merge_gap_s=args.merge_gap_s, expand_neighbors=args.expand_neighbors,
            candidate_margin_s=args.candidate_margin_s,
        )
        prediction = coarse_prediction(
            candidates, margin_s=args.coarse_margin_s,
            upper_bound_s=snapshot.manifest.t_q,
        )
        coarse = prediction
        query_trace.append({
            "kind": "frame_retrieval", "candidate_count": len(candidates),
            "retrieved_frames": [
                asdict(hit)
                for candidate in candidates for hit in candidate.retrieved_frames
            ],
            "clip_resource": encoder.resource_dicts(),
        })
        final_reason = "no_candidate" if not candidates else "highest_scoring_temporal_cluster"
        if candidates and readout == "timelens_local":
            try:
                prediction, refiner_trace = refine_frame_candidate(
                    args.query, snapshot, candidates[0],
                    TimeLensModelService.get(args.timelens_model),
                    max_frames=args.max_refine_frames,
                )
                if prediction.status != "ok":
                    prediction = coarse
                query_trace.append(refiner_trace)
                final_reason = (
                    "timelens_local_valid" if prediction.status == "ok"
                    else "timelens_local_parse_fallback_to_candidate"
                )
            except Exception as exc:
                # The coarse result is an explicit, trace-visible fallback.
                query_trace.append({
                    "kind": "refiner_fallback", "candidate_id": candidates[0].candidate_id,
                    "reason": type(exc).__name__, "message": str(exc),
                })
                final_reason = "timelens_local_exception_fallback_to_coarse_visual"
    if retrieval != "hybrid":
        payload = asdict(prediction)
        payload["query_id"] = args.query_id
        payload["video_id"] = snapshot.manifest.video_id
        payload["rho_q"] = snapshot.manifest.t_q / float(
            snapshot.manifest.video_meta["duration_s"]
        )
        payload["evidence_ids"] = list(prediction.evidence_ids)
        payload["candidate_ids"] = list(prediction.candidate_ids)
        payload["candidates"] = [asdict(candidate) for candidate in candidates]
        payload["query_trace"] = query_trace
        if retrieval == "frame":
            payload["readout"] = readout
            payload["candidate_envelope"] = (
                list(candidates[0].span) if candidates else None
            )
            payload["final_selection_reason"] = final_reason
    text = json.dumps(payload, ensure_ascii=False)
    if args.output:
        output_path = versioned_result_path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        write_provenance(
            output_path.parent,
            configuration={
                "snapshot": str(args.snapshot.resolve()), "query": args.query, "retrieval": retrieval,
                "query_id": args.query_id,
                "resolved_config": str(args.resolved_config.resolve()) if args.resolved_config else None,
                "clip_model": args.clip_model if retrieval == "frame" else None,
                "clip_revision": args.clip_revision if retrieval == "frame" else None,
                "clip_sha256": args.clip_sha256 if retrieval == "frame" else None,
                "timelens_model": str(args.timelens_model.resolve()) if args.timelens_model else None,
                "top_k": args.top_k, "merge_gap_s": args.merge_gap_s,
                "expand_neighbors": args.expand_neighbors,
                "candidate_margin_s": args.candidate_margin_s,
                "coarse_margin_s": args.coarse_margin_s,
                "readout": readout, "max_refine_frames": args.max_refine_frames,
            },
            config_filename="config.resolved.json",
        )
        output_path.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
