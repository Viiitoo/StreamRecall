#!/usr/bin/env python3
"""Expand and execute a resumable multi-GPU experiment matrix."""

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
from streamtimelens.evaluation.matrix import (
    MatrixDataset, MatrixQuery, MatrixSpec, MatrixVideo, execute_matrix, expand_matrix,
)


def _load_spec(path: Path) -> MatrixSpec:
    row = yaml.safe_load(path.read_text(encoding="utf-8"))
    datasets = []
    for dataset in row["datasets"]:
        video_rows = dataset.get("videos")
        if video_rows is None:
            video_rows = [
                json.loads(line)
                for line in Path(dataset["video_manifest"]).read_text(encoding="utf-8").splitlines()
            ]
        query_rows = dataset.get("queries")
        if query_rows is None and dataset.get("query_manifest"):
            query_rows = [
                json.loads(line)
                for line in Path(dataset["query_manifest"]).read_text(encoding="utf-8").splitlines()
            ]
        limit = int(dataset.get("video_limit", len(video_rows)))
        video_rows = video_rows[:limit]
        if dataset.get("video_root"):
            video_root = Path(dataset["video_root"])
            video_rows = [
                {**video, "path": str(video_root / Path(str(video["path"])).name)}
                for video in video_rows
            ]
        allowed_video_ids = {str(video["video_id"]) for video in video_rows}
        query_rows = [
            query for query in (query_rows or [])
            if str(query["video_id"]) in allowed_video_ids
        ]
        datasets.append(MatrixDataset(
            str(dataset["name"]),
            tuple(MatrixVideo(str(video["video_id"]), str(video["path"])) for video in video_rows),
            tuple(MatrixQuery(
                str(query["query_id"]), str(query["video_id"]), str(query["query"]),
            ) for query in query_rows),
        ))
    return MatrixSpec(
        tuple(map(str, row["methods"])), tuple(map(str, row["budgets"])), tuple(datasets),
        tuple(map(float, row["rhos"])), tuple(map(str, row.get("gpu_ids", [0, 1, 2, 3]))),
        tuple(map(str, row["ingest_command"])), tuple(map(str, row["query_command"])),
        protocol_config=str(row.get("protocol_config", "")),
        model_config=str(row.get("model_config", "")),
        run_id=str(row.get("run_id", "run-001")),
        query_grid=tuple(dict(value) for value in row.get("query_grid", [{}])),
        query_mode=str(row.get("query_mode", "per_query")),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    spec = _load_spec(args.matrix)
    output_root = versioned_result_path(args.output)
    output_root.mkdir(parents=True, exist_ok=True)
    write_provenance(
        output_root, configuration=yaml.safe_load(args.matrix.read_text(encoding="utf-8")),
        config_path=args.matrix, config_filename="matrix.resolved.yaml",
    )
    jobs = expand_matrix(spec, output_root)
    result = execute_matrix(jobs)
    summary = {**result, "jobs": len(jobs), "output_root": str(output_root)}
    (output_root / "matrix.summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
