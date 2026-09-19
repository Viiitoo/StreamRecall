"""Snapshot-only CLIP retrieval plus one-shot TimeLens multi-candidate readout."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Any

from streamtimelens.config import HybridV3Config
from streamtimelens.protocol.snapshot import SnapshotReader
from streamtimelens.protocol.types import Prediction
from streamtimelens.refiner.frame_allocator import allocate_candidate_frames
from streamtimelens.refiner.multicandidate_parse import parse_multicandidate_output
from streamtimelens.refiner.multicandidate_prompt import (
    SPARSE_MULTICANDIDATE_ADAPTIVE_VERSION,
    build_multicandidate_prompt,
    candidate_labels,
    multicandidate_template_sha256,
)
from streamtimelens.refiner.prompts import video_messages
from streamtimelens.refiner.timelens_adapter import SparseFrame, prepare_sparse_video
from streamtimelens.retrieval.frame_candidates import build_frame_candidates, coarse_prediction
from streamtimelens.retrieval.multicandidate import select_diverse_candidates


def _prediction_dict(prediction: Prediction) -> dict[str, Any]:
    result = asdict(prediction)
    result["span"] = list(prediction.span) if prediction.span is not None else None
    result["evidence_ids"] = list(prediction.evidence_ids)
    result["candidate_ids"] = list(prediction.candidate_ids)
    result.pop("start_s")
    result.pop("end_s")
    return result


@dataclass(frozen=True)
class HybridQueryOutput:
    query_id: str
    video_id: str
    rho_q: float
    readout: str
    clip_coarse_prediction: dict[str, Any]
    candidate_ids: tuple[str, ...]
    candidates: tuple[dict[str, Any], ...]
    selected_candidate_id: str | None
    selected_frame_refs: tuple[str, ...]
    candidate_frame_refs: dict[str, tuple[str, ...]]
    timelens_prediction: dict[str, Any]
    final_prediction: dict[str, Any]
    final_selection_reason: str
    fallback_used: bool
    query_trace: tuple[dict[str, Any], ...]
    schema_version: int = 1

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["candidate_ids"] = list(self.candidate_ids)
        result["candidates"] = list(self.candidates)
        result["selected_frame_refs"] = list(self.selected_frame_refs)
        result["candidate_frame_refs"] = {
            key: list(value) for key, value in self.candidate_frame_refs.items()
        }
        result["query_trace"] = list(self.query_trace)
        return result


def _confidence(score: float) -> float:
    return max(0.0, min(1.0, (score + 1.0) / 2.0))


def _empty_timelens(status: str, reason: str) -> dict[str, Any]:
    return {
        "status": status, "span": None, "candidate_id": None,
        "raw_output": None, "parse_status": "not_called", "reason": reason,
    }


def answer_hybrid_snapshot(
    *,
    query_id: str,
    query: str,
    snapshot: SnapshotReader,
    encoder: Any,
    service: Any,
    config: HybridV3Config,
) -> HybridQueryOutput:
    """Answer independently from immutable snapshot state; no video path exists."""
    query_start = time.perf_counter()
    if not query_id or not query.strip():
        raise ValueError("query ID and text are required")
    if snapshot.manifest.pixel_only or snapshot.manifest.method != "uniform_raw":
        raise ValueError("Hybrid V3 requires a non-pixel-only frozen Uniform snapshot")
    if snapshot.manifest.config_hash != config.snapshot_source.snapshot_config_sha256:
        raise ValueError("snapshot config hash does not match the registered Frozen V2 source")
    observed_service_hashes = dict(getattr(service, "hashes", {}))
    if observed_service_hashes != config.brain.model_service_hashes:
        raise ValueError("loaded TimeLens service hashes do not match Hybrid V3 config")
    metadata = snapshot.read_frame_metadata()
    future_refs = [
        ref for ref, row in metadata.items()
        if float(row["timestamp_s"]) > snapshot.manifest.t_q + 1e-9
    ]
    if future_refs:
        raise ValueError(f"snapshot exposes frames after query arrival: {future_refs[:3]}")
    trace: list[dict[str, Any]] = []
    trace.append({
        "kind": "hybrid_configuration",
        "readout": config.brain.readout,
        "snapshot_config_sha256": snapshot.manifest.config_hash,
        "snapshot_state_bytes": snapshot.manifest.state_bytes,
        "snapshot_budget_bytes": snapshot.manifest.budget_bytes,
        "frozen_config_id": config.snapshot_source.frozen_config_id,
        "clip_model": config.retrieval.clip_model,
        "clip_revision": config.retrieval.clip_revision,
        "clip_content_sha256": config.retrieval.clip_content_sha256,
        "timelens_model": config.brain.model,
        "timelens_model_registry_key": config.brain.model_registry_key,
        "timelens_revision": config.brain.model_revision,
        "timelens_content_sha256": config.brain.model_content_sha256,
        "timelens_service_hashes": observed_service_hashes,
        "prompt_version": config.brain.prompt_version,
        "prompt_template_sha256": multicandidate_template_sha256(config.brain.prompt_version),
        "frame_allocator": config.brain.frame_allocator,
        "candidate_label_order": config.brain.candidate_label_order,
        "merge_gap_tolerance_s": config.retrieval.merge_gap_tolerance_s,
        "calls_per_query": config.brain.calls_per_query,
        "max_unique_frames": config.brain.max_unique_frames,
        "sparse_keep_max_frames": config.brain.sparse_keep_max_frames,
        "min_refined_candidate_ratio": config.brain.min_refined_candidate_ratio,
    })

    before_resources = len(getattr(encoder, "resource_records", ()))
    clip_start = time.perf_counter()
    query_embedding = encoder.encode_text(query)
    clip_wall_s = time.perf_counter() - clip_start
    encoder_resources = getattr(encoder, "resource_dicts", lambda: [])()
    clip_resources = encoder_resources[before_resources:]
    all_candidates = build_frame_candidates(
        query_embedding, metadata,
        top_k=config.retrieval.top_k,
        merge_gap_s=config.retrieval.merge_gap_s,
        expand_neighbors=config.retrieval.expand_neighbors,
        candidate_margin_s=config.retrieval.candidate_margin_s,
        upper_bound_s=snapshot.manifest.t_q,
        merge_gap_tolerance_s=config.retrieval.merge_gap_tolerance_s,
    )
    coarse = coarse_prediction(
        all_candidates, margin_s=config.retrieval.coarse_margin_s,
        upper_bound_s=snapshot.manifest.t_q,
    )
    retrieved = sorted(
        (asdict(hit) for candidate in all_candidates for hit in candidate.retrieved_frames),
        key=lambda row: (row["rank"], row["timestamp_s"], row["frame_ref"]),
    )
    trace.append({
        "kind": "clip_retrieval", "clip_query_wall_s": clip_wall_s,
        "clip_resources": clip_resources, "retrieved_frames": retrieved,
        "raw_candidate_count": len(all_candidates),
    })
    selection = select_diverse_candidates(
        all_candidates,
        max_candidates=config.retrieval.max_candidate_clusters,
        temporal_nms_iou=config.retrieval.temporal_nms_iou,
        min_cluster_separation_s=config.retrieval.min_cluster_separation_s,
    )
    trace.append({
        "kind": "candidate_selection", "decisions": list(selection.decisions),
        "clustered_candidates": [asdict(item) for item in all_candidates],
        "selected_candidate_ids": [item.candidate_id for item in selection.candidates],
    })
    if not selection.candidates:
        final = coarse
        trace.append({"kind": "query_completed", "wall_s": time.perf_counter() - query_start})
        return HybridQueryOutput(
            query_id, snapshot.manifest.video_id,
            snapshot.manifest.t_q / float(snapshot.manifest.video_meta["duration_s"]),
            config.brain.readout, _prediction_dict(coarse), (), (), None, (), {},
            _empty_timelens("not_called", "no_clip_candidate"),
            _prediction_dict(final), "no_clip_candidate", False, tuple(trace),
        )

    bundle = allocate_candidate_frames(
        selection.candidates, snapshot, max_frames=config.brain.max_unique_frames,
        strategy=config.brain.frame_allocator,
    )
    trace.append({
        "kind": "frame_allocation", "selected_frame_refs": list(bundle.selected_frame_refs),
        "selected_frames": [
            {
                "frame_ref": ref,
                "timestamp_s": float(metadata[ref]["timestamp_s"]),
                "frame_index": int(metadata[ref]["frame_index"]),
            }
            for ref in bundle.selected_frame_refs
        ],
        "candidate_frame_refs": {
            key: list(value) for key, value in bundle.candidate_frame_refs.items()
        },
        "allocation_reason": bundle.allocation_reason,
    })
    candidate_rows = tuple(asdict(candidate) for candidate in selection.candidates)
    common = dict(
        query_id=query_id, video_id=snapshot.manifest.video_id,
        rho_q=snapshot.manifest.t_q / float(snapshot.manifest.video_meta["duration_s"]),
        readout=config.brain.readout, clip_coarse_prediction=_prediction_dict(coarse),
        candidate_ids=tuple(item.candidate_id for item in selection.candidates),
        candidates=candidate_rows, selected_frame_refs=bundle.selected_frame_refs,
        candidate_frame_refs=bundle.candidate_frame_refs,
    )
    if not bundle.selected_frame_refs:
        trace.append({
            "kind": "fallback", "reason": "empty_frame_budget",
            "fallback_target": "clip_coarse",
        })
        trace.append({"kind": "query_completed", "wall_s": time.perf_counter() - query_start})
        return HybridQueryOutput(
            **common, selected_candidate_id=None,
            timelens_prediction=_empty_timelens("not_called", "empty_frame_budget"),
            final_prediction=_prediction_dict(coarse),
            final_selection_reason="empty_frame_budget_fallback_to_clip_coarse",
            fallback_used=True, query_trace=tuple(trace),
        )

    prompt = build_multicandidate_prompt(
        version=config.brain.prompt_version, query=query,
        candidates=bundle.candidates,
        candidate_frame_refs=bundle.candidate_frame_refs,
        frame_metadata=metadata, t_q=snapshot.manifest.t_q,
        candidate_label_order=config.brain.candidate_label_order,
        selected_frame_count=len(bundle.selected_frame_refs),
        sparse_keep_max_frames=config.brain.sparse_keep_max_frames,
    )
    labels = candidate_labels(
        bundle.candidates, order=config.brain.candidate_label_order,
    )
    trace.append({
        "kind": "timelens_prompt", "prompt_version": prompt.version,
        "prompt_sha256": prompt.sha256,
        "prompt_template_sha256": multicandidate_template_sha256(prompt.version),
        "candidate_labels": labels,
        "candidate_label_order": config.brain.candidate_label_order,
        "selected_frame_count": len(bundle.selected_frame_refs),
        "sparse_keep_max_frames": config.brain.sparse_keep_max_frames,
    })
    inference_start = time.perf_counter()
    model_called = False
    try:
        import numpy as np
        from PIL import Image

        sparse_frames = []
        for ref in bundle.selected_frame_refs:
            row = metadata[ref]
            with Image.open(snapshot.frame_path(ref)) as image:
                pixels = np.asarray(image.convert("RGB"))
            sparse_frames.append(SparseFrame(
                int(row["frame_index"]), float(row["timestamp_s"]), pixels,
            ))
        prepared = prepare_sparse_video(
            sparse_frames,
            original_fps=float(snapshot.manifest.video_meta["original_fps"]),
            total_num_frames=int(snapshot.manifest.video_meta["total_num_frames"]),
            k_frames=config.brain.max_unique_frames,
            total_visual_tokens=config.brain.total_visual_tokens,
        )
        model_called = True
        answer = service.generate(
            video_messages(prompt), [prepared.processor_video], config.brain.max_new_tokens,
        )
        inference_wall_s = time.perf_counter() - inference_start
        parsed = parse_multicandidate_output(
            answer, t_q=snapshot.manifest.t_q, candidate_labels=labels,
            candidate_spans={item.candidate_id: item.span for item in bundle.candidates},
            candidate_frame_refs=bundle.candidate_frame_refs,
            available_frame_refs=tuple(metadata),
            prompt_version=config.brain.prompt_version,
        )
        model_stats = dict(getattr(service, "last_call_stats", {}))
        trace.append({
            "kind": "timelens_inference", "model_path": config.brain.model,
            "model_revision": config.brain.model_revision,
            "model_content_sha256": config.brain.model_content_sha256,
            "service_hashes": dict(getattr(service, "hashes", {})),
            "calls": 1, "unique_input_frames": len(bundle.selected_frame_refs),
            "total_visual_tokens": config.brain.total_visual_tokens,
            "max_new_tokens": config.brain.max_new_tokens,
            "do_sample": config.brain.do_sample,
            "timestamp_audit": list(prepared.timestamp_audit),
            "raw_output": answer, "parse_status": parsed.status,
            "parse_reason": parsed.reason, "wall_s": inference_wall_s,
            "gpu_time_s": model_stats.get("gpu_time_s"),
            "generated_tokens": model_stats.get("generated_tokens"),
            "peak_cuda_bytes": model_stats.get("peak_cuda_bytes"),
            "model_stats": model_stats,
        })
        timelens_prediction = {
            "status": "ok" if parsed.status in ("ok", "keep") else (
                "NOT_FOUND" if parsed.not_found else "invalid"
            ),
            "span": list(parsed.span) if parsed.span is not None else None,
            "candidate_id": parsed.selected_candidate_id,
            "candidate_label": parsed.selected_label,
            "raw_output": answer, "parse_status": parsed.status,
            "reason": parsed.reason,
        }
        if parsed.status in ("ok", "keep"):
            assert parsed.span is not None and parsed.selected_candidate_id is not None
            selected_candidate = next(
                item for item in bundle.candidates
                if item.candidate_id == parsed.selected_candidate_id
            )
            sparse_gate = (
                config.brain.prompt_version == SPARSE_MULTICANDIDATE_ADAPTIVE_VERSION
                and len(bundle.selected_frame_refs) <= config.brain.sparse_keep_max_frames
            )
            refined_ratio = (
                (parsed.span[1] - parsed.span[0])
                / (selected_candidate.end_s - selected_candidate.start_s)
            )
            shrink_gate = (
                config.brain.prompt_version == SPARSE_MULTICANDIDATE_ADAPTIVE_VERSION
                and refined_ratio < config.brain.min_refined_candidate_ratio
            )
            accepted_span = (
                selected_candidate.span if sparse_gate or shrink_gate else parsed.span
            )
            final = Prediction(
                accepted_span[0], accepted_span[1], _confidence(selected_candidate.score), "ok",
                evidence_ids=bundle.candidate_frame_refs[parsed.selected_candidate_id],
                candidate_ids=(parsed.selected_candidate_id,),
            )
            final_reason = (
                "timelens_refinement_rejected_sparse_gate_keep"
                if sparse_gate and parsed.status == "ok"
                else "timelens_refinement_rejected_shrink_gate_keep"
                if shrink_gate and parsed.status == "ok"
                else "timelens_multicandidate_keep"
                if parsed.status == "keep"
                else "timelens_multicandidate_valid"
            )
            fallback_used = False
        elif parsed.not_found and config.fallback.accept_model_not_found:
            final = Prediction(None, None, 0.0, "NOT_FOUND")
            final_reason = "timelens_multicandidate_not_found_accepted"
            fallback_used = False
        else:
            final = coarse
            final_reason = (
                "timelens_not_found_rejected_fallback_to_clip_coarse"
                if parsed.not_found else "timelens_invalid_fallback_to_clip_coarse"
            )
            fallback_used = True
            trace.append({
                "kind": "fallback", "reason": parsed.status,
                "fallback_target": "clip_coarse",
            })
        selected_candidate_id = parsed.selected_candidate_id
    except Exception as exc:
        inference_wall_s = time.perf_counter() - inference_start
        final = coarse
        selected_candidate_id = None
        timelens_prediction = {
            **_empty_timelens("exception", type(exc).__name__),
            "message": str(exc),
        }
        final_reason = "timelens_exception_fallback_to_clip_coarse"
        fallback_used = True
        trace.append({
            "kind": "timelens_exception", "reason": type(exc).__name__,
            "message": str(exc), "wall_s": inference_wall_s,
            "fallback_target": "clip_coarse", "model_path": config.brain.model,
            "model_revision": config.brain.model_revision,
            "model_content_sha256": config.brain.model_content_sha256,
            "calls": int(model_called), "unique_input_frames": len(bundle.selected_frame_refs),
            "total_visual_tokens": config.brain.total_visual_tokens,
            "max_new_tokens": config.brain.max_new_tokens,
            "raw_output": None, "parse_status": "exception",
        })
    trace.append({"kind": "query_completed", "wall_s": time.perf_counter() - query_start})
    return HybridQueryOutput(
        **common, selected_candidate_id=selected_candidate_id,
        timelens_prediction=timelens_prediction,
        final_prediction=_prediction_dict(final), final_selection_reason=final_reason,
        fallback_used=fallback_used, query_trace=tuple(trace),
    )
