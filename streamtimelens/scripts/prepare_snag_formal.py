#!/usr/bin/env python3
"""Materialize GT-isolated manifests for a frozen SnAG formal run."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
for source_root in (PACKAGE_ROOT / "src", PACKAGE_ROOT.parent / "src"):
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))

from baas.provenance import versioned_result_path, write_artifact_manifest, write_provenance  # noqa: E402
from streamtimelens.baselines.snag_config import load_snag_config, resolved_snag_config_dict  # noqa: E402
from streamtimelens.evaluation.formal_visual import timelens_bench_records  # noqa: E402
from streamtimelens.protocol.arrival import write_arrival_plan  # noqa: E402


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--video-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def _write(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def main() -> int:
    import cv2

    args = _args()
    config = load_snag_config(args.config)
    raw = config.raw
    if raw.get("classification") != "strict" or not raw.get("formal_frozen"):
        raise ValueError("formal preparation requires the frozen strict SnAG config")
    official = json.loads(args.annotations.read_text(encoding="utf-8"))
    videos, queries, evaluation, arrivals = timelens_bench_records(
        args.dataset, official, args.video_root,
    )
    durations = {row["video_id"]: float(row["duration_s"]) for row in evaluation}
    probed = []
    for video in videos:
        capture = cv2.VideoCapture(video["path"])
        if not capture.isOpened():
            raise ValueError(f"cannot probe formal video: {video['path']}")
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        capture.release()
        if fps <= 0 or frames <= 0:
            raise ValueError(f"invalid formal video metadata: {video['video_id']}")
        probed.append({
            **video, "fps": fps, "total_num_frames": frames,
            "duration_s": durations[video["video_id"]],
        })
    for query in queries:
        query["duration_s"] = durations[query["video_id"]]
    output = versioned_result_path(args.output)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite formal preparation: {output}")
    output.mkdir(parents=True)
    _write(output / "videos.jsonl", probed)
    _write(output / "queries.gt_free.jsonl", queries)
    _write(output / "annotations.evaluation_only.jsonl", evaluation)
    arrival_digest = write_arrival_plan(arrivals, output / "arrival_plan.jsonl")
    write_provenance(
        output, configuration=resolved_snag_config_dict(config),
        config_path=args.config, config_filename="config.resolved.yaml",
        extra_metadata={
            "dataset": args.dataset,
            "formal_test_access": True,
            "ground_truth_visible_to_query_process": False,
            "arrival_plan_sha256": arrival_digest,
        },
    )
    write_artifact_manifest(output)
    print(json.dumps({
        "output": str(output), "videos": len(videos), "queries": len(queries),
        "arrivals": len(arrivals), "arrival_sha256": arrival_digest,
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
