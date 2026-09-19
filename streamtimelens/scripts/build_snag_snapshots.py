#!/usr/bin/env python3
"""Build SnAG snapshots with one query-blind sequential video pass."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
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
from streamtimelens.baselines.snag_config import (  # noqa: E402
    load_snag_config,
    resolved_snag_config_dict,
)
from streamtimelens.baselines.snag_features import SequentialCLIPFeatureExtractor  # noqa: E402
from streamtimelens.baselines.snag_stream import SnAGSequentialIngestor  # noqa: E402
from streamtimelens.observer.clip_encoder import FrozenCLIPEncoder  # noqa: E402
from streamtimelens.protocol.types import VideoMeta  # noqa: E402


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--video-manifest", type=Path, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--video-id", action="append", dest="video_ids")
    return parser.parse_args()


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if not rows:
        raise ValueError("empty SnAG video manifest")
    return rows


def _video_meta(row: dict[str, object]) -> VideoMeta:
    """Load manifest metadata, probing the video when frame fields are absent."""
    duration_s = float(row["duration_s"])
    fps_value = row.get("fps")
    frame_value = row.get("total_num_frames")
    if fps_value is None or frame_value is None:
        import cv2

        capture = cv2.VideoCapture(str(row["path"]))
        try:
            if not capture.isOpened():
                raise RuntimeError(f"cannot open SnAG input video: {row['path']}")
            probed_fps = float(capture.get(cv2.CAP_PROP_FPS))
            probed_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        finally:
            capture.release()
        if probed_fps <= 0 or probed_frames < 1:
            raise ValueError(f"invalid probed video metadata: {row['path']}")
        fps_value = probed_fps if fps_value is None else fps_value
        frame_value = probed_frames if frame_value is None else frame_value
    return VideoMeta(
        str(row["video_id"]), duration_s, float(fps_value), int(frame_value),
    )


def main() -> int:
    args = _args()
    config = load_snag_config(args.config)
    videos = _read_jsonl(args.video_manifest)
    if args.video_ids:
        requested = set(args.video_ids)
        videos = [row for row in videos if str(row["video_id"]) in requested]
        found = {str(row["video_id"]) for row in videos}
        if found != requested:
            raise ValueError(f"unknown requested video IDs: {sorted(requested - found)}")
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("limit must be positive")
        videos = videos[:args.limit]
    output = versioned_result_path(args.output)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite SnAG snapshot run: {output}")
    output.mkdir(parents=True)
    encoder = FrozenCLIPEncoder(
        config.feature.model, device=args.device,
        batch_size=1, visual_fps=config.feature.feature_fps,
    )
    extractor = SequentialCLIPFeatureExtractor(
        encoder, feature_fps=config.feature.feature_fps,
    )
    ingestor = SnAGSequentialIngestor(extractor, config.writer)
    ratios = tuple(config.training.arrival_ratios) if config.training else (0.25, 0.5, 0.75, 1.0)
    index_rows = []
    run_rows = []
    for video in videos:
        meta = _video_meta(video)
        times = tuple(ratio * meta.duration_s for ratio in ratios)
        run = ingestor.run(
            str(video["path"]), meta, times, output / "snapshots" / meta.video_id,
        )
        run_rows.append(asdict(run))
        for ratio, record in zip(ratios, run.snapshots):
            index_rows.append({
                "video_id": meta.video_id, "rho": ratio, "t_q": record.t_q,
                "path": record.path, "state_bytes": record.state_bytes,
                "token_count": record.token_count,
            })
    (output / "snapshot_index.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in index_rows),
        encoding="utf-8",
    )
    (output / "ingest_runs.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in run_rows),
        encoding="utf-8",
    )
    checks = {
        "query_blind_inputs": True,
        "single_pass_all_videos": all(row["ingest"]["decode_passes"] == 1 for row in run_rows),
        "no_future_all_snapshots": all(
            record["latest_evidence_end_s"] is None
            or record["latest_evidence_end_s"] <= record["t_q"] + 1e-9
            for row in run_rows for record in row["snapshots"]
        ),
        "actual_byte_budget": all(
            config.budget_bytes is None or row["state_bytes"] <= config.budget_bytes
            for row in index_rows
        ),
    }
    (output / "ingest_audit.json").write_text(
        json.dumps({
            "schema_version": 1, "passed": all(checks.values()), "checks": checks,
            "classification": "ingest-only-not-a-result",
        }, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest_digest = hashlib.sha256(args.video_manifest.read_bytes()).hexdigest()
    write_provenance(
        output, configuration=resolved_snag_config_dict(config),
        config_path=args.config, config_filename="config.resolved.yaml",
        extra_metadata={
            "input_video_manifest": str(args.video_manifest.resolve()),
            "input_video_manifest_sha256": manifest_digest,
            "video_count": len(videos),
        },
    )
    write_artifact_manifest(output)
    print(json.dumps({"output": str(output), "videos": len(videos), "checks": checks}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
