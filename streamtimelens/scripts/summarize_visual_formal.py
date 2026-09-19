#!/usr/bin/env python3
"""Join formal GT after query execution and render frozen visual V5 tables."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import yaml

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT / "src"))
sys.path.insert(0, str(PACKAGE_ROOT.parent / "src"))

from baas.provenance import versioned_result_path, write_provenance
from streamtimelens.evaluation.formal_visual import (
    FormalVisualObservation, canonical_formal_rho, canonical_sha256, formal_observation_dict,
    paired_formal_bootstrap, summarize_formal_group,
)
from streamtimelens.protocol.arrival import verify_arrival_plan
from streamtimelens.protocol.snapshot import SnapshotReader


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _resources(rows: list[dict]) -> tuple[float, float]:
    unique = {
        json.dumps(row, sort_keys=True, separators=(",", ":")): row for row in rows
    }
    return (
        sum(float(row.get("cuda_s") or 0) for row in unique.values()),
        sum(float(row.get("wall_s") or 0) for row in unique.values()),
    )


def _variant(query_grid: list[dict], name: str) -> dict:
    for index, row in enumerate(query_grid):
        expected = f"qg_{index:02d}_{canonical_sha256(row)[:8]}"
        if name == expected:
            return row
    raise ValueError(f"prediction has unknown frozen query variant: {name}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix-root", type=Path, action="append", required=True)
    parser.add_argument("--annotations", type=Path, action="append", required=True)
    parser.add_argument("--arrival-plan", type=Path, action="append", required=True)
    parser.add_argument("--audit", type=Path, action="append", required=True)
    parser.add_argument("--frozen", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    frozen = yaml.safe_load(args.frozen.read_text(encoding="utf-8"))
    frozen_ids = set(frozen["configs"])
    for audit_path in args.audit:
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        if not audit.get("passed"):
            raise ValueError(f"formal visual audit failed: {audit_path}")
    annotations = {}
    for path in args.annotations:
        for row in _jsonl(path):
            if row["query_id"] in annotations:
                raise ValueError(f"duplicate formal query ID: {row['query_id']}")
            annotations[row["query_id"]] = row
    arrival_by_key = defaultdict(list)
    for path in args.arrival_plan:
        for row in verify_arrival_plan(path):
            if row.eligible:
                arrival_by_key[(row.video_id, row.query_id, round(row.rho_q, 8))].append(row)

    observations = []
    job_cache = {}
    snapshot_cache = {}
    ingest_cache = {}
    query_count_cache = {}
    for matrix_root in args.matrix_root:
        paths = sorted(matrix_root.glob("**/predictions/*.rho_*.qg_*.json"))
        if not paths:
            raise ValueError(f"formal matrix contains no predictions: {matrix_root}")
        for prediction_path in paths:
            job_root = prediction_path.parent.parent
            if job_root not in job_cache:
                job_cache[job_root] = json.loads(
                    (job_root / "config.resolved.json").read_text(encoding="utf-8")
                )
            job = job_cache[job_root]
            payload = json.loads(prediction_path.read_text(encoding="utf-8"))
            variant = _variant(job["query_grid"], str(payload["query_variant"]))
            config_id = str(variant["config_id"])
            if config_id not in frozen_ids:
                raise ValueError(f"formal prediction is not frozen: {config_id}")
            annotation = annotations.get(str(payload["query_id"]))
            if annotation is None or annotation["video_id"] != payload["video_id"]:
                raise ValueError(f"formal annotation identity mismatch: {prediction_path}")
            rho = float(payload["rho_q"])
            arrivals = arrival_by_key.get((payload["video_id"], payload["query_id"], round(rho, 8)))
            if not arrivals:
                continue
            snapshot_root = job_root / "snapshots" / payload["video_id"] / f"rho_{rho:.2f}"
            if snapshot_root not in snapshot_cache:
                reader = SnapshotReader(snapshot_root)
                metadata = reader.read_frame_metadata()
                timestamps = tuple(float(row["timestamp_s"]) for row in metadata.values())
                snapshot_cache[snapshot_root] = (
                    reader.manifest.state_bytes, timestamps, len(metadata),
                )
            if job_root not in query_count_cache:
                query_count_cache[job_root] = len(_jsonl(job_root / "queries.input.jsonl"))
            if job_root not in ingest_cache:
                trace_path = job_root / "snapshots" / payload["video_id"] / "ingest.trace.jsonl"
                trace = _jsonl(trace_path)
                resources = list({
                    row.get("clip_call_index", index): row["resource"]
                    for index, row in enumerate(trace)
                    if row.get("kind") == "clip_encoded" and isinstance(row.get("resource"), dict)
                }.values())
                ingest_cache[job_root] = _resources(resources)
            query_resources = []
            for trace in payload.get("query_trace", []):
                query_resources.extend(trace.get("clip_resource", []))
            query_gpu, query_wall = _resources(query_resources)
            retrieved = {}
            candidates = payload.get("candidates", [])
            for candidate in candidates:
                for frame in candidate.get("retrieved_frames", []):
                    ref = str(frame["frame_ref"])
                    if ref not in retrieved or int(frame["rank"]) < int(retrieved[ref]["rank"]):
                        retrieved[ref] = frame
            ordered = sorted(retrieved.values(), key=lambda row: (int(row["rank"]), row["frame_ref"]))
            span = None
            if payload.get("start_s") is not None:
                span = (float(payload["start_s"]), float(payload["end_s"]))
            budget = int(yaml.safe_load(Path(job["budget"]).read_text())["memory_bytes"])
            for arrival in arrivals:
                observations.append(FormalVisualObservation(
                    config_id=config_id, dataset=str(job["dataset"]), budget_bytes=budget,
                    # Measured t_q / duration can differ from the frozen ratio
                    # by a few ulps. Keep formal grouping on the canonical plan.
                    video_id=arrival.video_id, query_id=arrival.query_id,
                    rho_q=canonical_formal_rho(rho, arrival.rho_q),
                    cohort=arrival.cohort, lag_bin=arrival.lag_bin,
                    duration_s=float(annotation["duration_s"]), gt_span=arrival.gt_span,
                    predicted_span=span, status=str(payload.get("status", "error")),
                    retrieved_timestamps=tuple(float(row["timestamp_s"]) for row in ordered),
                    candidate_spans=tuple(
                        (float(row["start_s"]), float(row["end_s"])) for row in candidates
                    ),
                    snapshot_bytes=int(snapshot_cache[snapshot_root][0]),
                    ingest_gpu_s=ingest_cache[job_root][0],
                    ingest_wall_s=ingest_cache[job_root][1],
                    query_gpu_s=query_gpu, query_latency_s=query_wall,
                    earliest_anchor_retained=(
                        min(snapshot_cache[snapshot_root][1], default=float("inf")) <= 1e-6
                    ),
                    retained_frame_count=int(snapshot_cache[snapshot_root][2]),
                    queries_per_snapshot=query_count_cache[job_root],
                ))
    if not observations:
        raise ValueError("no eligible formal observations were joined")
    grouped = defaultdict(list)
    lag_grouped = defaultdict(list)
    for row in observations:
        grouped[(row.config_id, row.dataset, row.rho_q, row.cohort)].append(row)
        lag_grouped[(row.config_id, row.dataset, row.cohort, row.lag_bin)].append(row)
    summaries = []
    reference_groups = defaultdict(list)
    for key in grouped:
        reference_groups[(key[1], grouped[key][0].budget_bytes, key[2], key[3])].append(key[0])
    for key, rows in sorted(grouped.items()):
        reference_id = sorted(reference_groups[(key[1], rows[0].budget_bytes, key[2], key[3])])[0]
        reference_rows = grouped[(reference_id, key[1], key[2], key[3])]
        summaries.append({
            **summarize_formal_group(rows),
            "paired_bootstrap_reference_config_id": reference_id,
            "paired_bootstrap_vs_budget_reference": paired_formal_bootstrap(rows, reference_rows),
        })
    lag_summaries = [
        {**summarize_formal_group(rows), "rho_q": None, "lag_bin": key[-1]}
        for key, rows in sorted(lag_grouped.items())
    ]
    output_root = versioned_result_path(args.output)
    output_root.mkdir(parents=True, exist_ok=True)
    write_provenance(
        output_root,
        configuration={
            "matrix_roots": [str(path.resolve()) for path in args.matrix_root],
            "annotations": [str(path.resolve()) for path in args.annotations],
            "arrival_plans": [str(path.resolve()) for path in args.arrival_plan],
            "audits": [str(path.resolve()) for path in args.audit],
            "frozen": str(args.frozen.resolve()), "bootstrap_resamples": 10000,
        },
        config_filename="config.resolved.json",
    )
    (output_root / "formal_observations.jsonl").write_text(
        "".join(json.dumps(formal_observation_dict(row), sort_keys=True) + "\n" for row in observations),
        encoding="utf-8",
    )
    (output_root / "formal_summaries.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in summaries), encoding="utf-8",
    )
    (output_root / "lag_summaries.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in lag_summaries), encoding="utf-8",
    )
    lines = ["# Frozen visual V5 formal evaluation", ""]
    for row in summaries:
        lines.append(
            f"- {row['dataset']} {row['config_id']} rho={row['rho_q']:.2f} "
            f"{row['cohort']}: mIoU={row['miou']:.3f}, R@1@0.5={row['recall_at_05']:.3f}, "
            f"n={row['count']}"
        )
    (output_root / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    result = {
        "observations": len(observations), "summary_rows": len(summaries),
        "lag_rows": len(lag_summaries), "output_root": str(output_root),
    }
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
