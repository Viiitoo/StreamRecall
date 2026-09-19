#!/usr/bin/env python3
"""Materialize GT-isolated TimeLens-Bench manifests and exact frozen V5 matrices."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT / "src"))
sys.path.insert(0, str(PACKAGE_ROOT.parent / "src"))

from baas.provenance import versioned_result_path, write_provenance
from streamtimelens.evaluation.formal_visual import frozen_visual_grids, timelens_bench_records
from streamtimelens.protocol.arrival import write_arrival_plan


def _jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--video-root", type=Path, required=True)
    parser.add_argument("--frozen", type=Path, required=True)
    parser.add_argument(
        "--config-ids", default="",
        help="Optional comma-separated frozen subset, used only after V5 for V6.",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    official = json.loads(args.annotations.read_text(encoding="utf-8"))
    frozen = yaml.safe_load(args.frozen.read_text(encoding="utf-8"))
    videos, queries, ground_truth, arrival = timelens_bench_records(
        args.dataset, official, args.video_root,
    )
    output_root = versioned_result_path(args.output)
    output_root.mkdir(parents=True, exist_ok=True)
    write_provenance(
        output_root,
        configuration={
            "dataset": args.dataset, "annotations": str(args.annotations.resolve()),
            "video_root": str(args.video_root.resolve()), "frozen": str(args.frozen.resolve()),
            "formal_test_access": True, "ground_truth_visible_to_query_process": False,
            "config_ids": args.config_ids,
        },
        config_path=args.frozen, config_filename="config.resolved.json",
    )
    video_manifest = output_root / "videos.jsonl"
    query_manifest = output_root / "queries.gt_free.jsonl"
    annotations = output_root / "annotations.evaluation_only.jsonl"
    offline_queries = output_root / "offline_queries.jsonl"
    _jsonl(video_manifest, videos)
    _jsonl(query_manifest, queries)
    _jsonl(annotations, ground_truth)
    paths = {row["video_id"]: row["path"] for row in videos}
    _jsonl(offline_queries, [{**row, "video_path": paths[row["video_id"]]} for row in queries])
    arrival_path = output_root / "arrival_plan.jsonl"
    arrival_sha256 = write_arrival_plan(arrival, arrival_path)

    grids = frozen_visual_grids(frozen)
    requested = {value.strip() for value in args.config_ids.split(",") if value.strip()}
    if requested:
        available = {row["config_id"] for rows in grids.values() for row in rows}
        if not requested <= available:
            raise ValueError(f"requested configurations are not frozen: {sorted(requested - available)}")
        grids = {
            budget: [row for row in rows if row["config_id"] in requested]
            for budget, rows in grids.items()
            if any(row["config_id"] in requested for row in rows)
        }
    visual_rows = [row["visual"] for row in frozen["configs"].values()]
    common = visual_rows[0]
    immutable_ingest = (
        "clip_model", "clip_revision", "clip_sha256", "embedding_precision", "anchor_fraction",
    )
    if any(any(row[key] != common[key] for key in immutable_ingest) for row in visual_rows):
        raise ValueError("frozen configurations disagree on immutable ingest parameters")
    budget_paths = {
        262144: "streamtimelens/configs/budgets/256k.yaml",
        1048576: "streamtimelens/configs/budgets/1m.yaml",
        4194304: "streamtimelens/configs/budgets/4m.yaml",
    }
    matrices = []
    for budget, query_grid in sorted(grids.items()):
        if budget not in budget_paths:
            raise ValueError(f"no executable budget config for {budget}")
        matrix = {
            "methods": ["streamtimelens/configs/methods/uniform_raw.yaml"],
            "budgets": [budget_paths[budget]],
            "protocol_config": "streamtimelens/configs/protocol.yaml",
            "model_config": "streamtimelens/configs/models.visual_local.yaml",
            "rhos": [.25, .5, .75, 1.0], "gpu_ids": [0, 1, 2, 3],
            "run_id": f"frozen-visual-v1-formal-{args.dataset}-{budget}",
            "query_mode": "per_video_grid", "query_grid": query_grid,
            "datasets": [{
                "name": args.dataset, "video_manifest": str(video_manifest.resolve()),
                "query_manifest": str(query_manifest.resolve()),
            }],
            "ingest_command": [
                "python3", "streamtimelens/scripts/build_snapshots.py",
                "--video", "{video_path}", "--method", "{method_name}",
                "--budget", "{budget_bytes}", "--arrival-ratios", "{rhos}",
                "--sample-fps", "2", "--clip-fps", "0.5",
                "--clip-model", str(common["clip_model"]),
                "--clip-revision", str(common["clip_revision"]),
                "--clip-sha256", str(common["clip_sha256"]),
                "--embedding-precision", str(common["embedding_precision"]),
                "--anchor-fraction", str(common["anchor_fraction"]),
                "--output", "{job_dir}/snapshots",
            ],
            "query_command": [
                "python3", "streamtimelens/scripts/answer_visual_grid.py",
                "--snapshot-root", "{snapshot_root}", "--queries", "{queries}",
                "--query-grid", "{query_grid}", "--rhos", "{rhos}",
                "--clip-model", str(common["clip_model"]),
                "--clip-revision", str(common["clip_revision"]),
                "--clip-sha256", str(common["clip_sha256"]),
                "--output", "{predictions_dir}",
            ],
        }
        matrix_path = output_root / f"matrix.{budget}.yaml"
        matrix_path.write_text(yaml.safe_dump(matrix, sort_keys=False), encoding="utf-8")
        matrices.append(str(matrix_path))
    summary = {
        "dataset": args.dataset, "videos": len(videos), "queries": len(queries),
        "arrival_records": len(arrival), "arrival_sha256": arrival_sha256,
        "matrices": matrices, "frozen_configs": sum(map(len, grids.values())),
        "output_root": str(output_root),
    }
    (output_root / "preparation.summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
