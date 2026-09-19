#!/usr/bin/env python3
"""Join GT outside query processes and summarize V2 visual retrieval configurations."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

import yaml

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT / "src"))
sys.path.insert(0, str(PACKAGE_ROOT.parent / "src"))

from baas.provenance import versioned_result_path, write_provenance
from streamtimelens.evaluation.dev_selection import VisualDevRun
from streamtimelens.evaluation.bootstrap import PairedMetricObservation, paired_video_bootstrap
from streamtimelens.evaluation.diagnostics import VisualDiagnosticExample, evaluate_visual_diagnostics
from streamtimelens.evaluation.metrics import temporal_iou
from streamtimelens.protocol.snapshot import SnapshotReader


def _hash(value) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _resources(records) -> tuple[float, float]:
    unique = {
        json.dumps(record, sort_keys=True, separators=(",", ":")): record
        for record in records
    }
    gpu = sum(float(row.get("cuda_s") or 0) for row in unique.values())
    wall = sum(float(row.get("wall_s") or 0) for row in unique.values())
    return gpu, wall


def _variant(query_grid: list[dict], name: str) -> dict:
    for index, row in enumerate(query_grid):
        if name == f"qg_{index:02d}_{_hash(row)[:8]}":
            return row
    raise ValueError(f"prediction has unknown query-grid variant: {name}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix-root", type=Path, required=True)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    annotations = {
        row["query_id"]: row
        for row in (
            json.loads(line) for line in args.queries.read_text(encoding="utf-8").splitlines()
        )
    }
    grouped = defaultdict(list)
    ingest_resources = {}
    job_cache = {}
    snapshot_bytes_cache = {}
    ingest_trace_cache = {}
    for prediction_path in sorted(args.matrix_root.glob("**/predictions/*.rho_*.qg_*.json")):
        job_root = prediction_path.parent.parent
        if job_root not in job_cache:
            job_cache[job_root] = json.loads(
                (job_root / "config.resolved.json").read_text(encoding="utf-8")
            )
        job = job_cache[job_root]
        payload = json.loads(prediction_path.read_text(encoding="utf-8"))
        annotation = annotations.get(payload.get("query_id"))
        if annotation is None or annotation["video_id"] != payload.get("video_id"):
            raise ValueError(f"prediction annotation identity mismatch: {prediction_path}")
        variant_name = prediction_path.stem.split(".")[-1]
        query_config = _variant(job["query_grid"], variant_name)
        method = Path(job["method"]).stem
        budget_values = yaml.safe_load(Path(job["budget"]).read_text(encoding="utf-8"))
        budget_bytes = int(budget_values["memory_bytes"])
        descriptor = {"method": method, "budget_bytes": budget_bytes, **query_config}
        config_id = f"{method}-{budget_bytes}-{_hash(descriptor)[:12]}"
        rho = float(payload["rho_q"])
        duration = float(annotation["duration_s"])
        if float(annotation["gt_span"][1]) > rho * duration + 1e-6:
            continue
        candidates = payload.get("candidates", [])
        retrieved = {}
        for candidate in candidates:
            for frame in candidate.get("retrieved_frames", []):
                previous = retrieved.get(frame["frame_ref"])
                if previous is None or int(frame["rank"]) < int(previous["rank"]):
                    retrieved[frame["frame_ref"]] = frame
        span = None
        if payload.get("start_s") is not None:
            span = (float(payload["start_s"]), float(payload["end_s"]))
        snapshot_root = job_root / "snapshots" / payload["video_id"] / f"rho_{rho:.2f}"
        if snapshot_root not in snapshot_bytes_cache:
            snapshot_bytes_cache[snapshot_root] = SnapshotReader(snapshot_root).manifest.state_bytes
        snapshot_bytes = snapshot_bytes_cache[snapshot_root]
        clip_records = []
        for trace in payload.get("query_trace", []):
            clip_records.extend(trace.get("clip_resource", []))
        query_gpu_s, query_wall_s = _resources(clip_records)
        ingest_key = (config_id, payload["video_id"])
        if ingest_key not in ingest_resources:
            trace_path = job_root / "snapshots" / payload["video_id"] / "ingest.trace.jsonl"
            if trace_path not in ingest_trace_cache:
                trace_rows = [
                    json.loads(line)
                    for line in trace_path.read_text(encoding="utf-8").splitlines()
                ]
                resource_rows = list({
                    row.get(
                        "clip_call_index", json.dumps(row["resource"], sort_keys=True)
                    ): row["resource"]
                    for row in trace_rows
                    if row.get("kind") == "clip_encoded"
                    and isinstance(row.get("resource"), dict)
                }.values())
                ingest_trace_cache[trace_path] = _resources(resource_rows)[0]
            ingest_resources[ingest_key] = ingest_trace_cache[trace_path]
        grouped[config_id].append({
            "descriptor": descriptor, "query_id": payload["query_id"],
            "video_id": payload["video_id"], "rho": rho, "duration": duration,
            "gt_span": tuple(map(float, annotation["gt_span"])), "span": span,
            "retrieved": tuple(retrieved.values()),
            "candidates": tuple(
                (float(item["start_s"]), float(item["end_s"])) for item in candidates
            ),
            "snapshot_bytes": snapshot_bytes, "query_gpu_s": query_gpu_s,
            "query_latency_s": query_wall_s,
            "candidate_hit_at_5": float(max(
                (temporal_iou(tuple(map(float, annotation["gt_span"])), span)
                 for span in (
                     (float(item["start_s"]), float(item["end_s"]))
                     for item in candidates[:5]
                 )),
                default=0.0,
            ) >= .5),
        })
    summaries = []
    diagnostic_payload = {}
    for config_id, rows in sorted(grouped.items()):
        diagnostics = evaluate_visual_diagnostics([
            VisualDiagnosticExample(
                row["query_id"] + f"@{row['rho']:.2f}", row["gt_span"],
                row["retrieved"], row["candidates"], row["span"], None,
            ) for row in rows
        ])
        ious = [temporal_iou(row["gt_span"], row["span"]) for row in rows]
        fixed = {
            f"{rho:.2f}": [
                temporal_iou(row["gt_span"], row["span"])
                for row in rows
                if row["rho"] == rho and row["gt_span"][1] <= .25 * row["duration"]
            ]
            for rho in (.25, .5, .75, 1.0)
        }
        descriptor = rows[0]["descriptor"]
        summary = VisualDevRun(
            config_id=config_id, method=descriptor["method"],
            budget_bytes=descriptor["budget_bytes"],
            frame_recall_at_1=diagnostics["frame_recall"]["1"],
            frame_recall_at_5=diagnostics["frame_recall"]["5"],
            frame_recall_at_8=diagnostics["frame_recall"]["8"],
            candidate_recall_at_1=diagnostics["candidate_recall"]["1"],
            candidate_recall_at_5=diagnostics["candidate_recall"]["5"],
            candidate_oracle_miou=diagnostics["candidate_oracle_miou"],
            left_boundary_hit_rate=diagnostics["left_boundary_hit_rate"],
            right_boundary_hit_rate=diagnostics["right_boundary_hit_rate"],
            coarse_miou=sum(ious) / len(ious),
            coarse_recall_at_1=sum(value >= .5 for value in ious) / len(ious),
            snapshot_bytes=sum(row["snapshot_bytes"] for row in rows) / len(rows),
            ingest_gpu_s=sum(
                value for (key, _), value in ingest_resources.items() if key == config_id
            ),
            query_gpu_s=sum(row["query_gpu_s"] for row in rows),
            query_latency_s=sum(row["query_latency_s"] for row in rows) / len(rows),
            fixed_c025_by_rho={
                key: (sum(values) / len(values) if values else 0.0)
                for key, values in fixed.items()
            },
        )
        summaries.append(summary)
        diagnostic_payload[config_id] = diagnostics
    by_descriptor = {
        (row.method, row.budget_bytes, tuple(sorted(
            (key, value) for key, value in grouped[row.config_id][0]["descriptor"].items()
            if key not in ("method", "budget_bytes")
        ))): row
        for row in summaries
    }
    updated = []
    for summary in summaries:
        if summary.method != "semantic_reservoir":
            updated.append(summary)
            continue
        descriptor_key = tuple(sorted(
            (key, value) for key, value in grouped[summary.config_id][0]["descriptor"].items()
            if key not in ("method", "budget_bytes")
        ))
        uniform = by_descriptor.get(("uniform_raw", summary.budget_bytes, descriptor_key))
        if uniform is None:
            updated.append(summary)
            continue
        observations = {"coarse_iou": [], "candidate_recall_at_5": []}
        for method_summary in (summary, uniform):
            for row in grouped[method_summary.config_id]:
                sample_id = f"{row['query_id']}@{row['rho']:.2f}"
                observations["coarse_iou"].append(PairedMetricObservation(
                    method_summary.method, summary.budget_bytes, row["video_id"], sample_id,
                    temporal_iou(row["gt_span"], row["span"]),
                ))
                observations["candidate_recall_at_5"].append(PairedMetricObservation(
                    method_summary.method, summary.budget_bytes, row["video_id"], sample_id,
                    row["candidate_hit_at_5"],
                ))
        evidence = {
            metric: paired_video_bootstrap(
                values, method_a="semantic_reservoir", method_b="uniform_raw",
            )
            for metric, values in observations.items()
        }
        updated.append(replace(summary, paired_bootstrap_vs_uniform=evidence))
    summaries = updated
    if not summaries:
        raise ValueError("no eligible visual dev predictions were found")
    output_root = versioned_result_path(args.output)
    output_root.mkdir(parents=True, exist_ok=True)
    write_provenance(
        output_root,
        configuration={
            "matrix_root": str(args.matrix_root.resolve()),
            "queries": str(args.queries.resolve()), "candidate_iou_threshold": .5,
        },
        config_filename="config.resolved.json",
    )
    (output_root / "visual_dev_summaries.jsonl").write_text(
        "".join(json.dumps(row.__dict__, sort_keys=True) + "\n" for row in summaries),
        encoding="utf-8",
    )
    (output_root / "visual_diagnostics.json").write_text(
        json.dumps(diagnostic_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    print(json.dumps({"config_count": len(summaries), "output_root": str(output_root)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
