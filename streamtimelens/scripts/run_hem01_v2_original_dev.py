#!/usr/bin/env python3
"""Build HEM-01 snapshots and run fixed-reader paired development evaluation."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Mapping

import yaml

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = PACKAGE_ROOT.parent
sys.path.insert(0, str(PACKAGE_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from baas.provenance import (  # noqa: E402
    git_metadata,
    versioned_result_path,
    write_artifact_manifest,
    write_provenance,
)
from run_jq01_v2_original_dev import (  # noqa: E402
    _load_baselines,
    _load_jsonl,
    _resolve,
    _sha,
    _verified_path,
    _verify_clip_model,
)
from streamtimelens.evaluation.hierarchical_event_v2 import (  # noqa: E402
    MAX_READOUT_P95_MS,
    summarize_hem_v2_development,
)
from streamtimelens.evaluation.metrics import temporal_iou  # noqa: E402
from streamtimelens.memory.hierarchical_event import (  # noqa: E402
    BOUNDARY_COSINE,
    EMBEDDING_PRECISION,
    HEM_SCHEMA_VERSION,
    LEVEL_CAPACITIES,
    MAX_FINE_EVENT_DURATION_S,
    load_event_memory,
)
from streamtimelens.observer.clip_encoder import FrozenCLIPEncoder  # noqa: E402
from streamtimelens.protocol.snapshot import SnapshotReader, SnapshotWriter  # noqa: E402
from streamtimelens.protocol.types import Budget, VideoMeta  # noqa: E402
from streamtimelens.retrieval.event_candidates import (  # noqa: E402
    EVENT_CANDIDATE_MARGIN_S,
    EVENT_EXPAND_NEIGHBORS,
    EVENT_MERGE_GAP_S,
    EVENT_TOP_K,
    locate_event_memory,
)
from streamtimelens.stream.decoder import mp4_packets  # noqa: E402
from streamtimelens.stream.hierarchical_event_runner import (  # noqa: E402
    HierarchicalEventIngestor,
    hem_ingest_config,
)


CONFIG_KEYS = {
    "schema_version", "stage", "method", "revision", "data_role", "dataset",
    "queries_path", "queries_sha256", "videos_path", "videos_sha256",
    "model_hashes_path", "model_hashes_sha256", "clip_model_path",
    "clip_model_revision", "clip_model_content_sha256", "clip_device",
    "clip_batch_size", "clip_visual_fps", "decoder_packet_fps",
    "baseline_source_revision", "baseline_source_config_id", "baseline_shards",
    "budget_bytes", "arrival_ratios", "eligibility", "expected_video_count",
    "expected_query_count", "expected_baseline_observation_count",
    "expected_eligible_observation_count", "expected_generated_snapshot_count",
    "event_schema", "level_capacities", "boundary_cosine",
    "max_fine_event_duration_s", "embedding_precision", "reader_top_k",
    "reader_merge_gap_s", "reader_expand_neighbors", "reader_candidate_margin_s",
    "readout", "maximum_readout_p95_ms", "bootstrap_resamples", "bootstrap_seed",
    "require_single_pass", "require_zero_seek", "require_support_conservation",
    "require_snapshot_unchanged", "formal_data_allowed", "d_lock_required",
}


def _validate_config(config: Mapping[str, Any]) -> None:
    if set(config) != CONFIG_KEYS:
        raise ValueError(f"HEM-01 effect config fields changed: {sorted(set(config) ^ CONFIG_KEYS)}")
    fixed = {
        "schema_version": 1,
        "stage": "hem01_v2_original_development",
        "method": "HEM-01",
        "revision": "v2-original-r1",
        "data_role": "development/consumed",
        "dataset": "charades_sta_derived_dev_v1",
        "clip_visual_fps": 0.5,
        "decoder_packet_fps": 2.0,
        "budget_bytes": 1024 * 1024,
        "arrival_ratios": [0.25, 0.5, 0.75, 1.0],
        "eligibility": "gt_end_le_query_arrival",
        "event_schema": HEM_SCHEMA_VERSION,
        "level_capacities": list(LEVEL_CAPACITIES),
        "boundary_cosine": BOUNDARY_COSINE,
        "max_fine_event_duration_s": MAX_FINE_EVENT_DURATION_S,
        "embedding_precision": EMBEDDING_PRECISION,
        "reader_top_k": EVENT_TOP_K,
        "reader_merge_gap_s": EVENT_MERGE_GAP_S,
        "reader_expand_neighbors": EVENT_EXPAND_NEIGHBORS,
        "reader_candidate_margin_s": EVENT_CANDIDATE_MARGIN_S,
        "readout": "highest_scoring_event_group",
        "maximum_readout_p95_ms": MAX_READOUT_P95_MS,
        "require_single_pass": True,
        "require_zero_seek": True,
        "require_support_conservation": True,
        "require_snapshot_unchanged": True,
        "formal_data_allowed": False,
        "d_lock_required": False,
    }
    changed = {key: (config.get(key), value) for key, value in fixed.items() if config.get(key) != value}
    if changed:
        raise ValueError(f"HEM-01 immutable effect contract changed: {changed}")


def _fingerprint(reader: SnapshotReader) -> tuple[tuple[str, int, str], ...]:
    return tuple(
        (name, reader.path(name).stat().st_size, _sha(reader.path(name)))
        for name in sorted(reader.manifest.allowed_files)
    )


def _build_snapshots(
    videos: list[dict[str, Any]], config: Mapping[str, Any], output: Path,
    encoder: FrozenCLIPEncoder, source_revision: str,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    budget = Budget(int(config["budget_bytes"]), 0)
    ingest_rows = []
    snapshot_rows: dict[str, dict[str, Any]] = {}
    for raw in sorted(videos, key=lambda row: str(row["video_id"])):
        video_id = str(raw["video_id"])
        path = Path(str(raw["path"]))
        if (
            not path.is_file() or path.stat().st_size != int(raw["bytes"])
            or _sha(path) != raw["sha256"]
        ):
            raise ValueError(f"HEM-01 video identity changed: {video_id}")
        decoded_meta, packets = mp4_packets(path, float(config["decoder_packet_fps"]))
        if decoded_meta.video_id != video_id:
            raise ValueError(f"HEM-01 decoder video identity changed: {video_id}")
        meta = VideoMeta(
            video_id, float(raw["duration_s"]), float(raw["fps"]),
            int(raw["total_num_frames"]),
        )
        writer = SnapshotWriter(output / video_id, source_revision=source_revision)
        ingestor = HierarchicalEventIngestor(
            meta, budget, clip_encoder=encoder, source_revision=source_revision,
        )
        arrivals = [float(rho) * meta.duration_s for rho in config["arrival_ratios"]]
        manifests = ingestor.run(
            packets, arrivals, writer, config=hem_ingest_config(), snapshot_prefix="snapshots",
        )
        if packets.decode_count != int(raw["total_num_frames"]) or packets.seek_count != 0:
            raise RuntimeError(f"HEM-01 single-pass decoder audit failed: {video_id}")
        ingest_rows.append({
            "video_id": video_id,
            "video_sha256": raw["sha256"],
            "decode_count": packets.decode_count,
            "emitted_packet_count": packets.emitted_packet_count,
            "seek_count": packets.seek_count,
            "timestamp_mode": packets.timestamp_mode,
            "pts_fallback_count": packets.pts_fallback_count,
            "final_support_count": ingestor.memory.seen_count,
        })
        for t_q, manifest in manifests.items():
            rho = round(t_q / meta.duration_s, 2)
            snapshot_id = f"{video_id}@{rho:.2f}"
            snapshot_path = output / video_id / "snapshots" / f"rho_{rho:.2f}"
            reader = SnapshotReader(snapshot_path)
            events = load_event_memory(reader)
            support = sum(event.support_count for event in events)
            snapshot_rows[snapshot_id] = {
                "snapshot_id": snapshot_id,
                "path": str(snapshot_path.resolve()),
                "rho": rho,
                "t_q": t_q,
                "event_count": len(events),
                "level_counts": [sum(event.scale == scale for event in events) for scale in range(3)],
                "support_count": support,
                "state_bytes": manifest.state_bytes,
                "budget_bytes": manifest.budget_bytes,
                "before_fingerprint": _fingerprint(reader),
            }
    return ingest_rows, snapshot_rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path,
        default=PACKAGE_ROOT / "configs/exploration/hem01_v2_original_dev.yaml",
    )
    parser.add_argument(
        "--hypothesis", required=True,
        help="Preregistered HEM-01 development hypothesis.",
    )
    parser.add_argument(
        "--output", type=Path,
        default=PROJECT_ROOT / "results/hem01/v2_original_dev",
    )
    args = parser.parse_args()
    if not args.hypothesis.strip():
        raise ValueError("HEM-01 hypothesis is required")
    git = git_metadata()
    if git["is_dirty"]:
        raise RuntimeError("HEM-01 effect run requires a clean Git commit")
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    _validate_config(config)
    query_path = _verified_path(config, "queries_path", "queries_sha256")
    video_path = _verified_path(config, "videos_path", "videos_sha256")
    model_path = _verified_path(config, "model_hashes_path", "model_hashes_sha256")
    clip_identity = _verify_clip_model(config, model_path)
    queries, videos = _load_jsonl(query_path), _load_jsonl(video_path)
    baselines, baseline_hashes = _load_baselines(config)
    if (
        len(queries) != int(config["expected_query_count"])
        or len(videos) != int(config["expected_video_count"])
        or len(baselines) != int(config["expected_baseline_observation_count"])
    ):
        raise ValueError("HEM-01 fixed development input counts changed")
    output = versioned_result_path(args.output / str(config["revision"]))
    output.mkdir(parents=True, exist_ok=False)
    resolved = {
        **config, "hypothesis": args.hypothesis.strip(), "code_revision": git["commit"],
        "clip_identity": clip_identity,
    }
    write_provenance(output, configuration=resolved, config_path=args.config)
    (output / "run_state.json").write_text(
        json.dumps({"status": "ingesting", "effect_rows_written": 0}, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    encoder = FrozenCLIPEncoder(
        _resolve(str(config["clip_model_path"])), device=str(config["clip_device"]),
        batch_size=int(config["clip_batch_size"]), visual_fps=float(config["clip_visual_fps"]),
        persistence=str(config["embedding_precision"]),
    )
    ingestion, snapshots = _build_snapshots(videos, config, output, encoder, git["commit"])
    if len(snapshots) != int(config["expected_generated_snapshot_count"]):
        raise RuntimeError("HEM-01 generated snapshot count changed")
    query_by_id = {str(row["query_id"]): row for row in queries}
    query_embeddings = {}
    persisted_queries = []
    for row in sorted(queries, key=lambda item: str(item["query_id"])):
        query_id = str(row["query_id"])
        embedding = encoder.encode_text(str(row["query"]))
        query_embeddings[query_id] = embedding
        persisted_queries.append({"query_id": query_id, "embedding": encoder.persisted(embedding)})
    effect_rows, condition_rows = [], []
    for baseline in sorted(baselines, key=lambda row: str(row["sample_id"])):
        query_id = str(baseline["query_id"])
        query = query_by_id[query_id]
        rho = round(float(baseline["rho_q"]), 2)
        if float(query["gt_span"][1]) > float(query["duration_s"]) * rho + 1e-7:
            continue
        video_id = str(baseline["video_id"])
        identity = str(baseline["sample_id"])
        snapshot_row = snapshots[f"{video_id}@{rho:.2f}"]
        reader = SnapshotReader(snapshot_row["path"])
        started = time.perf_counter()
        prediction, debug = locate_event_memory(query_embeddings[query_id], reader)
        latency_ms = (time.perf_counter() - started) * 1000.0
        if prediction.start_s is None or prediction.end_s is None:
            hem_span = (0.0, min(1e-3, reader.manifest.t_q))
        else:
            hem_span = (float(prediction.start_s), float(prediction.end_s))
        gt_span = tuple(map(float, query["gt_span"]))
        baseline_span = tuple(map(float, baseline["coarse_span"]))
        candidate_ious = [
            temporal_iou(gt_span, (float(row["start_s"]), float(row["end_s"])))
            for row in debug["candidates"][:5]
        ]
        baseline_iou = temporal_iou(gt_span, baseline_span)
        row = {
            "training_or_evaluation_only": True,
            "observation_id": identity,
            "video_id": video_id,
            "query_id": query_id,
            "budget_bytes": int(config["budget_bytes"]),
            "rho": rho,
            "good_baseline": baseline_iou >= 0.5,
            "gt_span": list(gt_span),
            "baseline_span": list(baseline_span),
            "hem_span": list(hem_span),
            "baseline_iou": baseline_iou,
            "hem_iou": temporal_iou(gt_span, hem_span),
            "hem_candidate_oracle_iou": max(candidate_ious, default=0.0),
            "readout_latency_ms": latency_ms,
            "event_count": int(snapshot_row["event_count"]),
            "level_counts": snapshot_row["level_counts"],
            "snapshot_state_bytes": int(snapshot_row["state_bytes"]),
            "candidate_debug": debug,
        }
        effect_rows.append(row)
        frozen_payload = {
            "sample_id": identity, "query_id": query_id, "video_id": video_id,
            "rho_q": rho, "final_span": list(baseline_span), "status": "frozen_visual_v2",
        }
        condition_rows.extend((
            {"observation_id": identity, "condition": "frozen_visual_v2", "prediction_row": frozen_payload},
            {"observation_id": identity, "condition": "HEM-01", "prediction_row": {
                **frozen_payload, "final_span": list(hem_span), "status": "hem01_event_readout",
            }},
        ))
    snapshot_unchanged = True
    byte_complete = True
    support_complete = True
    snapshot_audit_rows = []
    final_support = {row["video_id"]: int(row["final_support_count"]) for row in ingestion}
    support_timeline: dict[str, list[tuple[float, int]]] = {}
    for row in snapshots.values():
        reader = SnapshotReader(row["path"])
        after = _fingerprint(reader)
        unchanged = tuple(map(tuple, row["before_fingerprint"])) == after
        within_budget = (
            int(row["state_bytes"]) == reader.manifest.state_bytes
            and reader.manifest.state_bytes <= reader.manifest.budget_bytes
        )
        support_valid = int(row["support_count"]) == sum(
            event.support_count for event in load_event_memory(reader)
        )
        snapshot_unchanged &= unchanged
        byte_complete &= within_budget
        support_complete &= support_valid
        video_id = str(row["snapshot_id"]).rsplit("@", 1)[0]
        support_timeline.setdefault(video_id, []).append((float(row["rho"]), int(row["support_count"])))
        snapshot_audit_rows.append({
            **{key: value for key, value in row.items() if key != "before_fingerprint"},
            "unchanged": unchanged, "within_budget": within_budget,
            "support_conserved": support_valid,
        })
    for video_id, timeline in support_timeline.items():
        ordered = sorted(timeline)
        support_complete &= all(
            left[1] <= right[1] for left, right in zip(ordered, ordered[1:])
        )
        support_complete &= ordered[-1][1] == final_support[video_id]
    zero_seek = all(row["seek_count"] == 0 for row in ingestion)
    coverage = (
        len(effect_rows) == int(config["expected_eligible_observation_count"])
        and len({row["observation_id"] for row in effect_rows}) == len(effect_rows)
    )
    summary = summarize_hem_v2_development(
        effect_rows, support_complete=support_complete, zero_seek=zero_seek,
        snapshots_unchanged=snapshot_unchanged, byte_audit_complete=byte_complete,
        coverage_complete=coverage, provenance_complete=True,
        seed=int(config["bootstrap_seed"]), resamples=int(config["bootstrap_resamples"]),
    )
    final_git = git_metadata()
    if final_git["is_dirty"] or final_git["commit"] != git["commit"]:
        raise RuntimeError("Git state changed during HEM-01 effect run")
    for name, rows in (
        ("ingestion_audit.jsonl", ingestion),
        ("snapshot_audit.jsonl", snapshot_audit_rows),
        ("query_embeddings.jsonl", persisted_queries),
        ("effect_rows.jsonl", effect_rows),
        ("condition_rows.jsonl", condition_rows),
    ):
        (output / name).write_text(
            "".join(json.dumps(row, sort_keys=True, allow_nan=False) + "\n" for row in rows),
            encoding="utf-8",
        )
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    input_hashes = {
        "queries_sha256": _sha(query_path), "videos_sha256": _sha(video_path),
        "model_hashes_sha256": _sha(model_path), "baseline_shards": baseline_hashes,
        "config_source_sha256": _sha(args.config),
    }
    resolved["input_sha256"] = input_hashes
    resolved["generated_snapshot_count"] = len(snapshots)
    resolved["eligible_observation_count"] = len(effect_rows)
    write_provenance(
        output, configuration=resolved, config_path=args.config,
        extra_metadata={"input_sha256": input_hashes},
    )
    (output / "run_state.json").write_text(
        json.dumps({
            "status": "complete", "effect_rows_written": len(effect_rows),
            "passed": summary["passed"], "decision": summary["decision"],
        }, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_artifact_manifest(output)
    print(json.dumps({
        "passed": summary["passed"], "decision": summary["decision"],
        "standard_miou_delta": summary["standard_metrics"]["delta_miou"],
        "video_equal_miou_delta": summary["miou_bootstrap"]["delta_a_minus_b"],
        "video_equal_miou_ci": [
            summary["miou_bootstrap"]["ci_low"], summary["miou_bootstrap"]["ci_high"],
        ],
        "r07_delta": summary["standard_metrics"]["delta_r07"],
        "output": str(output),
    }, sort_keys=True))
    return 0 if summary["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
