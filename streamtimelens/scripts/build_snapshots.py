#!/usr/bin/env python3
"""Build several delayed-query snapshots in one sequential decode pass."""

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
from streamtimelens.protocol.snapshot import SnapshotWriter
from streamtimelens.protocol.types import Budget
from streamtimelens.observer.clip_encoder import DEFAULT_CLIP_MODEL, FrozenCLIPEncoder
from streamtimelens.retrieval.embedder import DEFAULT_TEXT_EMBEDDER, CardTextEmbedder
from streamtimelens.stream.decoder import jsonl_packets, mp4_packets
from streamtimelens.stream.baseline_runner import BaselineIngestConfig, RawBaselineIngestor
from streamtimelens.memory.raw_cache import JPEG_QUALITY, JPEG_SHORT_EDGE
from streamtimelens.stream.runner import IngestConfig, StreamingIngestor
from streamtimelens.writer.prompts import WRITER_MAX_NEW_TOKENS, WRITER_PROMPT_VERSION, WRITER_TEMPERATURE
from streamtimelens.writer.timelens_writer import TimeLensEvidenceWriter
from streamtimelens.baselines.vst_summary import (
    VST_MAX_NEW_TOKENS, VST_PROMPT_VERSION, TimeLensFreeTextWriter,
    VSTIngestConfig, VSTSummaryIngestor,
)
from streamtimelens.refiner.model_service import TimeLensModelService
from streamtimelens.baselines.oasis_adapt import (
    OASISAdaptIngestor, OASISIngestConfig,
)


def _budget(value: str) -> int:
    normalized = value.lower().strip()
    units = {"k": 1024, "m": 1024 * 1024}
    if normalized[-1:] in units:
        return int(float(normalized[:-1]) * units[normalized[-1]])
    return int(normalized)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--video", type=Path, help="MP4 input; decoded exactly once")
    source.add_argument("--frames", type=Path, help="Synthetic JSONL FramePacket source")
    parser.add_argument("--video-id", help="Required with --frames")
    parser.add_argument("--duration", type=float, help="Required with --frames")
    parser.add_argument("--sample-fps", type=float, default=2.0)
    parser.add_argument("--arrival-ratios", default=".25,.5,.75,1")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--budget", default="1m")
    parser.add_argument("--writer-calls-per-minute", type=float, default=2.0)
    parser.add_argument("--trigger-mode", choices=("periodic", "cut-motion", "semantic", "joint"), default="joint")
    parser.add_argument("--trigger-threshold", type=float, default=0.65)
    parser.add_argument("--minimum-gap-s", type=float, default=1.0)
    parser.add_argument("--max-gap-s", type=float, default=60.0)
    parser.add_argument("--initial-writer-tokens", type=float, default=1.0)
    parser.add_argument("--segment-overlap-s", type=float, default=1.0)
    parser.add_argument("--segment-bytes-per-frame", type=int, default=16 * 1024)
    parser.add_argument("--source-revision", default="unknown")
    parser.add_argument(
        "--method", choices=(
            "full", "evidence_fixed", "uniform_raw", "semantic_reservoir",
            "vst_summary", "oasis_adapt",
        ),
        default="full",
    )
    parser.add_argument("--vst-policy", choices=("first_recent", "fifo"), default="first_recent")
    parser.add_argument("--fixed-segment-s", type=float, default=32.0)
    parser.add_argument("--oasis-max-roots", type=int, default=8)
    parser.add_argument("--capacity", type=int, help="Retained baseline frames; default derives from B_mem")
    parser.add_argument("--clip-model", default=DEFAULT_CLIP_MODEL)
    parser.add_argument("--clip-revision")
    parser.add_argument("--clip-sha256")
    parser.add_argument("--clip-device")
    parser.add_argument("--clip-batch-size", type=int, default=32)
    parser.add_argument("--clip-fps", type=float, default=0.5)
    parser.add_argument("--embedding-precision", choices=("fp16", "int8"), default="fp16")
    parser.add_argument("--pixel-only", action="store_true", help="Uniform ablation only; disables frame-query retrieval")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--anchor-fraction", type=float, default=0.25)
    parser.add_argument("--writer", choices=("fallback", "timelens"), default="fallback")
    parser.add_argument("--writer-model", type=Path)
    parser.add_argument("--writer-revision", default="frozen-local")
    parser.add_argument("--writer-device-map", default="auto")
    parser.add_argument("--writer-max-frames", choices=(8, 16, 32), type=int, default=32)
    parser.add_argument("--card-embedding", choices=("none", "minilm", "qwen3"), default="none")
    parser.add_argument("--text-embedder", default=DEFAULT_TEXT_EMBEDDER)
    parser.add_argument("--text-embedder-revision")
    parser.add_argument("--text-embedder-device")
    parser.add_argument("--boundary-window-s", type=float, default=2.0)
    parser.add_argument("--boundary-frames-per-side", type=int, default=4)
    parser.add_argument("--boundary-internal-frames", type=int, default=2)
    parser.add_argument("--forest-max-roots", type=int, default=8)
    parser.add_argument("--merge-semantic-weight", type=float, default=1.0)
    parser.add_argument("--merge-gap-weight", type=float, default=0.35)
    parser.add_argument("--merge-boundary-weight", type=float, default=0.35)
    parser.add_argument("--utility-novelty-weight", type=float, default=0.30)
    parser.add_argument("--utility-boundary-weight", type=float, default=0.30)
    parser.add_argument("--utility-inverse-density-weight", type=float, default=0.25)
    parser.add_argument("--utility-has-raw-weight", type=float, default=0.15)
    parser.add_argument("--coverage-bucket-base-s", type=float, default=1.0)
    parser.add_argument("--forest-debug", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.frames:
        if not args.video_id or args.duration is None:
            raise SystemExit("--video-id and --duration are required with --frames")
        meta, packets = jsonl_packets(args.frames)
        meta = type(meta)(args.video_id, args.duration, meta.original_fps, meta.total_num_frames)
    else:
        meta, packets = mp4_packets(args.video, args.sample_fps)
    budget = Budget(_budget(args.budget), args.writer_calls_per_minute)
    ratios = tuple(float(item) for item in args.arrival_ratios.split(","))
    arrival_times = [ratio * meta.duration_s for ratio in ratios]
    capacity = args.capacity or max(1, budget.memory_bytes // (20 * 1024))
    effective_text_embedder = args.text_embedder
    if args.card_embedding == "qwen3" and effective_text_embedder == DEFAULT_TEXT_EMBEDDER:
        effective_text_embedder = "Qwen/Qwen3-Embedding-0.6B"
    structured_method = args.method in ("full", "evidence_fixed")
    card_method = structured_method or args.method in ("vst_summary", "oasis_adapt")
    effective_card_embedding = args.card_embedding
    if args.method in ("vst_summary", "oasis_adapt") and effective_card_embedding == "none":
        effective_card_embedding = "minilm"
    ingest_config = None
    if structured_method:
        effective_trigger_mode = "periodic" if args.method == "evidence_fixed" else args.trigger_mode
        ingest_config = IngestConfig(
            method_name=args.method, trigger_mode=effective_trigger_mode,
            trigger_threshold=args.trigger_threshold,
            minimum_gap_s=args.minimum_gap_s, max_gap_s=args.max_gap_s,
            initial_writer_tokens=args.initial_writer_tokens,
            segment_overlap_s=args.segment_overlap_s,
            segment_bytes_per_frame=args.segment_bytes_per_frame,
            source_revision=args.source_revision,
            boundary_window_s=args.boundary_window_s,
            boundary_frames_per_side=args.boundary_frames_per_side,
            boundary_internal_frames=args.boundary_internal_frames,
            forest_max_roots=args.forest_max_roots,
            merge_semantic_weight=args.merge_semantic_weight,
            merge_gap_weight=args.merge_gap_weight,
            merge_boundary_weight=args.merge_boundary_weight,
            utility_novelty_weight=args.utility_novelty_weight,
            utility_boundary_weight=args.utility_boundary_weight,
            utility_inverse_density_weight=args.utility_inverse_density_weight,
            utility_has_raw_weight=args.utility_has_raw_weight,
            coverage_bucket_base_s=args.coverage_bucket_base_s,
            forest_debug=args.forest_debug,
        )
    config = {
        "method": args.method, "sample_fps": args.sample_fps, "arrival_ratios": ratios,
        "budget_bytes": budget.memory_bytes, "writer_calls_per_minute": budget.writer_calls_per_minute,
        "writer": args.writer if card_method else "none",
        "writer_revision": args.writer_revision if card_method else None,
        "writer_prompt_version": (
            WRITER_PROMPT_VERSION if structured_method else
            VST_PROMPT_VERSION if args.method in ("vst_summary", "oasis_adapt") else None
        ),
        "writer_max_new_tokens": (
            WRITER_MAX_NEW_TOKENS if structured_method else
            VST_MAX_NEW_TOKENS if args.method in ("vst_summary", "oasis_adapt") else None
        ),
        "writer_temperature": WRITER_TEMPERATURE if structured_method else None,
        "writer_max_frames": args.writer_max_frames if structured_method else None,
        "card_embedding": effective_card_embedding if card_method else "none",
        "text_embedder": effective_text_embedder if card_method and effective_card_embedding != "none" else None,
        "text_embedder_revision": args.text_embedder_revision if card_method and effective_card_embedding != "none" else None,
        "boundary_window_s": args.boundary_window_s if structured_method else None,
        "boundary_frames_per_side": args.boundary_frames_per_side if structured_method else None,
        "boundary_internal_frames": args.boundary_internal_frames if structured_method else None,
        "forest_max_roots": args.forest_max_roots if structured_method else None,
        "merge_semantic_weight": args.merge_semantic_weight if structured_method else None,
        "merge_gap_weight": args.merge_gap_weight if structured_method else None,
        "merge_boundary_weight": args.merge_boundary_weight if structured_method else None,
        "utility_novelty_weight": args.utility_novelty_weight if structured_method else None,
        "utility_boundary_weight": args.utility_boundary_weight if structured_method else None,
        "utility_inverse_density_weight": args.utility_inverse_density_weight if structured_method else None,
        "utility_has_raw_weight": args.utility_has_raw_weight if structured_method else None,
        "coverage_bucket_base_s": args.coverage_bucket_base_s if structured_method else None,
        "forest_debug": args.forest_debug if structured_method else None,
        "trigger_mode": ingest_config.trigger_mode if structured_method else None,
        "trigger_threshold": args.trigger_threshold if structured_method else None,
        "minimum_gap_s": args.minimum_gap_s if structured_method else None,
        "max_gap_s": args.max_gap_s if structured_method else None,
        "initial_writer_tokens": args.initial_writer_tokens if structured_method else None,
        "segment_overlap_s": args.segment_overlap_s if structured_method else None,
        "segment_bytes_per_frame": args.segment_bytes_per_frame if structured_method else None,
        "ingest_config": asdict(ingest_config) if ingest_config is not None else None,
        "capacity": capacity if not card_method else None,
        "clip_model": args.clip_model if not card_method and not args.pixel_only else None,
        "clip_revision": args.clip_revision if not card_method and not args.pixel_only else None,
        "clip_sha256": args.clip_sha256 if not card_method and not args.pixel_only else None,
        "clip_batch_size": args.clip_batch_size, "clip_fps": args.clip_fps,
        "jpeg_short_edge": JPEG_SHORT_EDGE, "jpeg_quality": JPEG_QUALITY,
        "embedding_precision": args.embedding_precision, "pixel_only": args.pixel_only,
        "seed": args.seed, "anchor_fraction": args.anchor_fraction,
        "vst_policy": args.vst_policy if args.method == "vst_summary" else None,
        "fixed_segment_s": args.fixed_segment_s if args.method in ("vst_summary", "oasis_adapt") else None,
        "oasis_max_roots": args.oasis_max_roots if args.method == "oasis_adapt" else None,
    }
    run_config = {
        **config, "source": str((args.frames or args.video).resolve()), "video_id": meta.video_id,
        "duration_s": meta.duration_s, "source_revision": args.source_revision,
        "writer_model": str(args.writer_model.resolve()) if args.writer_model else None,
        "writer_device_map": args.writer_device_map if args.writer == "timelens" else None,
        "text_embedder_device": args.text_embedder_device if args.card_embedding != "none" else None,
    }
    output_root = versioned_result_path(args.output)
    write_provenance(output_root, configuration=run_config, config_filename="config.resolved.json")
    if structured_method:
        if args.pixel_only:
            raise SystemExit("--pixel-only is only valid with --method uniform_raw")
        if args.writer == "timelens" and args.writer_model is None:
            raise SystemExit("--writer-model is required with --writer timelens")
        evidence_writer = None
        if args.writer == "timelens":
            evidence_writer = TimeLensEvidenceWriter(
                str(args.writer_model), writer_revision=args.writer_revision,
                device_map=args.writer_device_map, k_frames=args.writer_max_frames,
            )
        text_embedder = None
        if args.card_embedding != "none":
            text_embedder = CardTextEmbedder(
                effective_text_embedder, revision=args.text_embedder_revision,
                device=args.text_embedder_device,
            )
        ingestor = StreamingIngestor(
            meta, budget, ingest_config, evidence_writer=evidence_writer,
            text_embedder=text_embedder,
        )
    elif args.method in ("vst_summary", "oasis_adapt"):
        if args.pixel_only:
            raise SystemExit("--pixel-only is not valid with summary baselines")
        if args.writer == "timelens" and args.writer_model is None:
            raise SystemExit("--writer-model is required with --writer timelens")
        text_embedder = CardTextEmbedder(
            effective_text_embedder, revision=args.text_embedder_revision,
            device=args.text_embedder_device,
        )
        summary_writer = None
        if args.writer == "timelens":
            summary_writer = TimeLensFreeTextWriter(
                TimeLensModelService.get(args.writer_model, device_map=args.writer_device_map),
                max_frames=args.writer_max_frames,
            )
        if args.method == "vst_summary":
            ingestor = VSTSummaryIngestor(
                meta, budget,
                VSTIngestConfig(
                    policy=args.vst_policy, segment_interval_s=args.fixed_segment_s,
                    max_frames=args.writer_max_frames,
                    source_revision=args.source_revision,
                ),
                text_embedder=text_embedder, writer=summary_writer,
            )
        else:
            ingestor = OASISAdaptIngestor(
                meta, budget,
                OASISIngestConfig(
                    segment_interval_s=args.fixed_segment_s,
                    max_frames=args.writer_max_frames, max_roots=args.oasis_max_roots,
                    source_revision=args.source_revision,
                ),
                text_embedder=text_embedder, writer=summary_writer,
            )
    else:
        if args.method == "semantic_reservoir" and args.pixel_only:
            raise SystemExit("semantic_reservoir requires CLIP embeddings")
        encoder = None if args.pixel_only else FrozenCLIPEncoder(
            args.clip_model, device=args.clip_device, batch_size=args.clip_batch_size,
            visual_fps=args.clip_fps, persistence=args.embedding_precision,
        )
        ingestor = RawBaselineIngestor(
            meta, budget,
            BaselineIngestConfig(
                args.method, capacity, pixel_only=args.pixel_only,
                embedding_precision=args.embedding_precision, seed=args.seed,
                anchor_fraction=args.anchor_fraction, source_revision=args.source_revision,
            ),
            clip_encoder=encoder,
        )
    result = ingestor.run(packets, arrival_times, SnapshotWriter(output_root, source_revision=args.source_revision),
                          config=config, snapshot_prefix=meta.video_id)
    trace_path = output_root / meta.video_id / "ingest.trace.jsonl"
    trace_path.write_text(ingestor.trace.to_jsonl(), encoding="utf-8")
    # Trace is intentionally beside snapshots, never in allowed_files.
    print(json.dumps({"video_id": meta.video_id, "snapshots": {str(key): value.state_bytes for key, value in result.items()},
                      "method": args.method, "writer_calls": ingestor.ledger.writer_calls,
                      "trace": str(trace_path), "output_root": str(output_root)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
