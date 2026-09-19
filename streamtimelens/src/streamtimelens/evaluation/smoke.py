"""Model-free 20-video engineering smoke across every internal method."""

from __future__ import annotations

import io
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image

from streamtimelens.baselines.oasis_adapt import OASISAdaptIngestor, OASISIngestConfig
from streamtimelens.baselines.vst_summary import VSTIngestConfig, VSTSummaryIngestor
from streamtimelens.config import BudgetConfig, ProtocolConfig
from streamtimelens.evaluation.cost import ComponentTimer
from streamtimelens.evaluation.resources import ResourceObservation, aggregate_resources
from streamtimelens.evaluation.runner import evaluate_predictions
from streamtimelens.observer.clip_encoder import FrozenCLIPEncoder
from streamtimelens.protocol.arrival import build_arrival_plan
from streamtimelens.protocol.snapshot import SnapshotReader, SnapshotWriter
from streamtimelens.protocol.types import Budget, FramePacket, VideoMeta
from streamtimelens.retrieval.embedder import CardTextEmbedder
from streamtimelens.retrieval.frame_candidates import coarse_prediction, retrieve_frame_candidates
from streamtimelens.retrieval.query_pipeline import answer_card_snapshot
from streamtimelens.stream.baseline_runner import BaselineIngestConfig, RawBaselineIngestor
from streamtimelens.stream.runner import IngestConfig, StreamingIngestor


INTERNAL_METHODS = (
    "uniform_raw", "semantic_reservoir", "vst_summary", "oasis_adapt",
    "evidence_fixed", "full",
)


class _TextBackend:
    def encode(self, texts: Sequence[str]) -> Any:
        return np.asarray([
            [1.0, 0.1] if "red" in text.lower() or "observation" in text.lower() else [0.1, 1.0]
            for text in texts
        ], dtype=np.float32)


class _CLIPBackend:
    def encode_images(self, images: Sequence[Any]) -> Any:
        vectors = []
        for image in images:
            pixels = np.asarray(image, dtype=np.float32)
            vectors.append([float(pixels[..., 0].mean()) + 1.0, float(pixels[..., 2].mean()) + 1.0])
        return np.asarray(vectors, dtype=np.float32)

    def encode_texts(self, texts: Sequence[str]) -> Any:
        return np.asarray([[1.0, 0.1] for _ in texts], dtype=np.float32)


def _jpeg(red: bool) -> bytes:
    output = io.BytesIO()
    Image.new("RGB", (16, 16), (240, 20, 20) if red else (20, 20, 240)).save(
        output, format="JPEG",
    )
    return output.getvalue()


def _video(video_index: int) -> tuple[VideoMeta, list[FramePacket], tuple[float, float]]:
    durations = (12.0, 120.0, 600.0)
    duration = durations[video_index % len(durations)]
    fps = 8.0 / duration
    video_id = f"smoke-{video_index:03d}"
    packets = [
        FramePacket(
            index / fps, index, _jpeg(3 <= index <= 4), 16, 16,
            source="synthetic_smoke", video_id=video_id,
        )
        for index in range(9)
    ]
    return VideoMeta(video_id, duration, fps, 9, 16, 16), packets, (3 / fps, 4 / fps)


def _ingestor(method: str, meta: VideoMeta, budget: Budget):
    text_embedder = CardTextEmbedder("deterministic-smoke-text", revision="v1", backend=_TextBackend())
    if method in ("uniform_raw", "semantic_reservoir"):
        clip = FrozenCLIPEncoder(
            "deterministic-smoke-clip", backend=_CLIPBackend(), batch_size=4,
            visual_fps=0.5, persistence="fp16",
        )
        return RawBaselineIngestor(
            meta, budget, BaselineIngestConfig(method, 8), clip_encoder=clip,
        ), None
    if method == "vst_summary":
        return VSTSummaryIngestor(
            meta, budget,
            VSTIngestConfig(segment_interval_s=meta.duration_s / 4, max_frames=8),
            text_embedder=text_embedder,
        ), text_embedder
    if method == "oasis_adapt":
        return OASISAdaptIngestor(
            meta, budget,
            OASISIngestConfig(
                segment_interval_s=meta.duration_s / 4, max_frames=8, max_roots=2,
            ),
            text_embedder=text_embedder,
        ), text_embedder
    trigger = "periodic" if method == "evidence_fixed" else "joint"
    return StreamingIngestor(
        meta, budget,
        IngestConfig(
            method_name=method, trigger_mode=trigger,
            decision_interval_s=meta.duration_s / 4,
            minimum_gap_s=0, max_gap_s=meta.duration_s / 4,
            max_active_frames=8, segment_bytes_per_frame=1024,
            forest_max_roots=2,
        ),
        text_embedder=text_embedder,
    ), text_embedder


def run_engineering_smoke(
    output_root: Path, *, video_count: int = 20,
    methods: Sequence[str] = INTERNAL_METHODS,
    memory_bytes: int = 256 * 1024,
) -> dict[str, Any]:
    if video_count <= 0 or memory_bytes <= 0 or any(method not in INTERNAL_METHODS for method in methods):
        raise ValueError("invalid engineering smoke matrix")
    budget = Budget(memory_bytes, 60, refine_calls_per_query=0, max_frames_per_refine=8)
    evaluation_budget = BudgetConfig(memory_bytes, 60, refine_calls_per_query=0, max_frames_per_refine=8)
    protocol = ProtocolConfig(not_found_threshold=0.0, max_active_frames=8)
    summary = {"videos": video_count, "methods": list(methods), "runs": 0, "snapshots": 0}
    for method in methods:
        method_predictions = []
        method_plan = []
        resources = []
        for video_index in range(video_count):
            meta, packets, gt_span = _video(video_index)
            ingestor, text_embedder = _ingestor(method, meta, budget)
            video_root = output_root / method / meta.video_id
            with ComponentTimer("engineering_smoke_ingest") as ingest_timer:
                manifests = ingestor.run(
                    packets, [ratio * meta.duration_s for ratio in (.25, .5, .75, 1.0)],
                    SnapshotWriter(video_root),
                    config={"method": method, "engineering_smoke": True},
                    snapshot_prefix="snapshots",
                )
            final_snapshot = video_root / "snapshots" / "rho_1.00"
            reader = SnapshotReader(final_snapshot)
            query_id = f"query-{meta.video_id}"
            with ComponentTimer("engineering_smoke_query") as query_timer:
                if text_embedder is not None:
                    output = answer_card_snapshot(
                        query_id=query_id, query="red visual event", snapshot=reader,
                        embedder=text_embedder, protocol=protocol, budget=evaluation_budget,
                    ).to_dict()
                else:
                    clip = FrozenCLIPEncoder(
                        "deterministic-smoke-clip", backend=_CLIPBackend(),
                        batch_size=1, visual_fps=.5,
                    )
                    candidates = retrieve_frame_candidates(
                        "red visual event", reader, clip, top_k=8,
                    )
                    coarse = coarse_prediction(candidates)
                    output = {
                        "schema_version": 1, "query_id": query_id,
                        "video_id": meta.video_id, "rho_q": 1.0,
                        "status": "not_found" if coarse.span is None else "fallback",
                        "span": list(coarse.span) if coarse.span is not None else None,
                        "confidence": coarse.confidence, "retrieved_cards": [],
                        "candidates": [item.candidate_id for item in candidates],
                        "raw_answers": [], "resource": {"clip": clip.resource_dicts()},
                        "diagnostics": {"candidate_count": len(candidates)},
                    }
            method_predictions.append(output)
            method_plan.extend(build_arrival_plan([{
                "video_id": meta.video_id, "query_id": query_id,
                "query": "red visual event", "gt_span": gt_span,
                "duration": meta.duration_s,
            }], [1.0]))
            resources.extend([
                ResourceObservation(
                    meta.video_id, "ingest", method, ingest_timer.record.wall_s,
                    cpu_s=ingest_timer.record.cpu_s,
                    state_peak_bytes=max(item.state_bytes for item in manifests.values()),
                    snapshot_bytes=reader.manifest.state_bytes,
                    stream_duration_s=meta.duration_s, max_backlog_s=0.0,
                ),
                ResourceObservation(
                    meta.video_id, "query", method, query_timer.record.wall_s,
                    cpu_s=query_timer.record.cpu_s, query_id=query_id,
                    cold_query=video_index == 0,
                ),
            ])
            trace_path = video_root / "ingest.trace.jsonl"
            trace_path.write_text(ingestor.trace.to_jsonl(), encoding="utf-8")
            summary["runs"] += 1
            summary["snapshots"] += len(manifests)
        method_root = output_root / method
        (method_root / "predictions.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in method_predictions),
            encoding="utf-8",
        )
        metrics = evaluate_predictions(method_plan, method_predictions)
        (method_root / "metrics.json").write_text(
            json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8",
        )
        (method_root / "sample_resources.jsonl").write_text(
            "".join(json.dumps(asdict(row), sort_keys=True) + "\n" for row in resources),
            encoding="utf-8",
        )
        (method_root / "resource_summary.json").write_text(
            json.dumps(aggregate_resources(resources), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return summary
