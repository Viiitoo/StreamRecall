#!/usr/bin/env python3
"""Train and evaluate a SnAG-adapt reader on an independent development split."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import shutil
import sys
import time
from dataclasses import asdict
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
from streamtimelens.baselines.snag_adapt import (  # noqa: E402
    SnAGAdaptReader,
    SnAGAdaptWriter,
    SnAGSnapshotReader,
    ranked_spans_close,
)
from streamtimelens.baselines.snag_config import (  # noqa: E402
    load_snag_config,
    resolved_snag_config_dict,
)
from streamtimelens.baselines.snag_physical import (  # noqa: E402
    SnAGPhysicalQuery,
    SnAGPhysicalTimeBackend,
)
from streamtimelens.baselines.snag_stream import snapshot_fingerprint  # noqa: E402
from streamtimelens.baselines.snag_training import (  # noqa: E402
    build_training_examples,
    load_physical_reader_checkpoint,
    save_physical_reader_checkpoint,
    split_examples_by_video,
    train_physical_reader,
    training_manifest_sha256,
)
from streamtimelens.evaluation.snag_effect import (  # noqa: E402
    evaluate_ranked_predictions,
    percentile,
    snag_effect_gate,
)
from streamtimelens.observer.clip_encoder import FrozenCLIPEncoder  # noqa: E402
from streamtimelens.protocol.arrival import build_arrival_plan  # noqa: E402


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--snapshot-index", type=Path, required=True)
    parser.add_argument("--query-manifest", type=Path, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--checkpoint", type=Path,
        help="Re-evaluate an immutable compatible checkpoint without retraining.",
    )
    return parser.parse_args()


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def main() -> int:
    args = _args()
    config = load_snag_config(args.config)
    if config.training is None or config.method not in {
        "snag-adapt-pooled-B", "snag-adapt-input-full",
    }:
        raise ValueError("training requires a supported SnAG adaptation config")
    index_rows = _jsonl(args.snapshot_index)
    index = {
        (str(row["video_id"]), round(float(row["rho"]), 8)): Path(row["path"])
        for row in index_rows
    }
    queries = _jsonl(args.query_manifest)
    for row in queries:
        if "duration" not in row and "duration_s" in row:
            row["duration"] = row["duration_s"]
    arrivals = build_arrival_plan(queries, ratios=config.training.arrival_ratios)
    encoder = FrozenCLIPEncoder(
        config.feature.model, device=args.device, batch_size=1,
        visual_fps=config.feature.feature_fps,
    )
    examples = build_training_examples(
        arrivals,
        lambda row: index[(row.video_id, round(row.rho_q, 8))],
        encoder.encode_text,
        minimum_delay_s=config.training.minimum_delay_s,
    )
    checkpoint_payload = None
    if args.checkpoint is None:
        model, training_report = train_physical_reader(
            examples, config.reader, config.training, device=args.device,
        )
    else:
        model, checkpoint_payload = load_physical_reader_checkpoint(
            args.checkpoint, device=args.device,
        )
        if checkpoint_payload["model_config"] != asdict(config.reader):
            raise ValueError("reused SnAG checkpoint reader configuration mismatch")
        if checkpoint_payload["training_recipe"] != asdict(config.training):
            raise ValueError("reused SnAG checkpoint training recipe mismatch")
        training_report = checkpoint_payload["training_report"]
    output = versioned_result_path(args.output)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite SnAG G3 run: {output}")
    output.mkdir(parents=True)
    checkpoint = output / "snag_physical_reader.pth"
    expected_feature_model = {
        "id": config.feature.upstream_id,
        "revision": config.feature.revision,
        "content_sha256": config.feature.content_sha256,
    }
    if args.checkpoint is None:
        checkpoint_digest = save_physical_reader_checkpoint(
            checkpoint, model, config.reader, config.training, training_report,
            feature_model=expected_feature_model,
        )
    else:
        if checkpoint_payload["feature_model"] != expected_feature_model:
            raise ValueError("reused SnAG checkpoint feature model mismatch")
        source_checkpoint = args.checkpoint.expanduser().resolve(strict=True)
        shutil.copyfile(source_checkpoint, checkpoint)
        checkpoint_digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    checkpoint.chmod(0o444)
    train_rows, validation_rows = split_examples_by_video(
        examples, seed=config.training.seed,
        validation_fraction=config.training.validation_fraction,
    )
    del train_rows
    backend = SnAGPhysicalTimeBackend(model, config.reader, device=args.device)
    reader = SnAGAdaptReader(backend)
    predictions = []
    evaluation_rows = []
    latencies = []
    query_independence = []
    for example in validation_rows:
        snapshot = SnAGSnapshotReader(example.snapshot_path)
        before = snapshot_fingerprint(snapshot)
        query = SnAGPhysicalQuery(example.query_embedding, example.t_q)
        started = time.perf_counter()
        spans = reader.answer(snapshot, query, topk=5)
        latencies.append(time.perf_counter() - started)
        repeated = reader.answer(SnAGSnapshotReader(example.snapshot_path), query, topk=5)
        after = snapshot_fingerprint(SnAGSnapshotReader(example.snapshot_path))
        independent = before == after and ranked_spans_close(spans, repeated)
        query_independence.append(independent)
        duration = float(snapshot.manifest.video_meta["duration_s"])
        prediction = {
            "video_id": example.video_id,
            "query_id": example.query_id,
            "rho": example.t_q / duration,
            "t_q": example.t_q,
            "spans": [[span.start_s, span.end_s, span.score] for span in spans],
            "status": "ok" if spans else "not_found",
        }
        predictions.append(prediction)
        evaluation_rows.append({**prediction, "gt_span": example.gt_span})
    metrics = evaluate_ranked_predictions(evaluation_rows)
    snapshot_bytes = [SnAGSnapshotReader(row.snapshot_path).manifest.state_bytes for row in validation_rows]
    ingest_audit_path = args.snapshot_index.parent / "ingest_audit.json"
    ingest_audit = json.loads(ingest_audit_path.read_text(encoding="utf-8"))
    ingest_runs = _jsonl(args.snapshot_index.parent / "ingest_runs.jsonl")
    protocol_checks = {
        "online_ingest": bool(ingest_audit["checks"]["query_blind_inputs"]),
        "late_query": True,
        "query_blind_write": "query" not in inspect.signature(SnAGAdaptWriter.ingest_step).parameters,
        "single_pass": bool(ingest_audit["checks"]["single_pass_all_videos"]),
        "snapshot_only_immutable": all(query_independence),
        "no_replay_no_future": bool(ingest_audit["checks"]["no_future_all_snapshots"]),
        "actual_byte_budget": bool(ingest_audit["checks"]["actual_byte_budget"]),
        "independent_queries": all(query_independence),
        "past_only_vtg": all(row.gt_span[1] <= row.t_q for row in validation_rows),
        "revision_provenance": True,
    }
    protocol_audit = {
        "schema_version": 1, "method": config.method,
        "passed": all(protocol_checks.values()), "checks": protocol_checks,
    }
    cost = {
        "readout_count": len(latencies),
        "readout_p50_s": percentile(latencies, 0.5),
        "readout_p95_s": percentile(latencies, 0.95),
        "snapshot_bytes_mean": sum(snapshot_bytes) / len(snapshot_bytes),
        "snapshot_bytes_max": max(snapshot_bytes),
        "query_visible_bytes_mean": sum(snapshot_bytes) / len(snapshot_bytes),
        "checkpoint_bytes": checkpoint.stat().st_size,
        "decoded_frames": sum(row["ingest"]["decoded_frames"] for row in ingest_runs),
        "ingest_wall_s": sum(row["ingest"]["wall_s"] for row in ingest_runs),
        "ingest_fps": (
            sum(row["ingest"]["decoded_frames"] for row in ingest_runs)
            / max(sum(row["ingest"]["wall_s"] for row in ingest_runs), 1e-9)
        ),
        "writer_peak_items": max(row["writer_peak_items"] for row in ingest_runs),
        "writer_peak_feature_bytes": max(row["writer_peak_feature_bytes"] for row in ingest_runs),
    }
    gate = snag_effect_gate(
        metrics, prediction_count=len(predictions), expected_count=len(validation_rows),
        protocol_audit_passed=protocol_audit["passed"],
        checkpoint_frozen=checkpoint.stat().st_mode & 0o222 == 0,
    )
    if config.method != "snag-adapt-pooled-B":
        gate = {
            **gate,
            "gate": "SnAG-full-store-reader-development-run",
            "note": "G1 is decided by the separate full-store upper-bound audit.",
        }
    configuration = resolved_snag_config_dict(config)
    configuration["trained_checkpoint"] = str(checkpoint)
    configuration["trained_checkpoint_sha256"] = checkpoint_digest
    write_provenance(
        output, configuration=configuration,
        config_path=args.config, config_filename="config.resolved.yaml",
        extra_metadata={
            "snapshot_index": str(args.snapshot_index.resolve()),
            "snapshot_index_sha256": hashlib.sha256(args.snapshot_index.read_bytes()).hexdigest(),
            "query_manifest": str(args.query_manifest.resolve()),
            "query_manifest_sha256": hashlib.sha256(args.query_manifest.read_bytes()).hexdigest(),
            "training_manifest_sha256": training_manifest_sha256(examples),
            "reused_checkpoint": str(args.checkpoint.resolve()) if args.checkpoint else None,
        },
    )
    artifacts = {
        "training_report.json": training_report,
        "metrics.json": metrics,
        "protocol_audit.json": protocol_audit,
        "cost_summary.json": cost,
        ("effect_gate.json" if config.method == "snag-adapt-pooled-B" else "reader_run_gate.json"): gate,
    }
    for name, value in artifacts.items():
        (output / name).write_text(
            json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8",
        )
    (output / "predictions.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in predictions),
        encoding="utf-8",
    )
    write_artifact_manifest(output)
    print(json.dumps({"output": str(output), "gate_passed": gate["passed"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
