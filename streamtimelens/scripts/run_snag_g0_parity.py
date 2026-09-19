#!/usr/bin/env python3
"""Run the checkpoint-backed, snapshot-roundtrip layer of SnAG G0 parity."""

from __future__ import annotations

import argparse
import json
import math
import sys
import tempfile
from pathlib import Path

import numpy as np

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
    SnAGAdaptConfig,
    SnAGAdaptWriter,
    SnAGSnapshotReader,
)
from streamtimelens.baselines.vendor.snag_model import (  # noqa: E402
    SnAGOfficialGridBackend,
    SnAGTorchBackend,
)
from streamtimelens.baselines.vendor.snag_upstream import (  # noqa: E402
    load_precomputed_text_feature,
    load_snag_model,
    load_upstream_nms_extension,
    sha256_file,
)
from streamtimelens.evaluation.snag_parity import (  # noqa: E402
    compare_official_metrics,
    compare_ranked_spans,
    compare_raw_traces,
    g0_gate_report,
)
from streamtimelens.protocol.types import VideoMeta  # noqa: E402


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream-root", required=True)
    parser.add_argument("--opt", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--video-feature", required=True)
    parser.add_argument("--text-feature", required=True)
    parser.add_argument("--video-id", required=True)
    parser.add_argument("--duration", required=True, type=float)
    parser.add_argument("--fps", required=True, type=float)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--checkpoint-key", default="model_ema")
    parser.add_argument("--expected-official-metrics", type=Path, required=True)
    parser.add_argument("--actual-official-metrics", type=Path, required=True)
    parser.add_argument("--metric-tolerance-points", type=float, default=0.25)
    parser.add_argument("--output", default="results/snag_adapt/g0-parity")
    return parser.parse_args()


def _video_feature(path: Path, data_opt: dict[str, object]) -> np.ndarray:
    value = np.load(path, allow_pickle=False).astype(np.float32)
    if value.ndim != 2 or not len(value) or not np.isfinite(value).all():
        raise ValueError("video feature must be a finite [time,channels] array")
    value = value[::int(data_opt.get("downsample_rate", 1))]
    if bool(data_opt.get("normalize_vid", False)):
        norm = np.linalg.norm(value, axis=1, keepdims=True)
        value = value / np.maximum(norm, np.finfo(np.float32).eps)
    return np.ascontiguousarray(value)


def main() -> int:
    args = _arguments()
    loaded = load_snag_model(
        args.upstream_root, args.opt, args.checkpoint,
        device=args.device, checkpoint_key=args.checkpoint_key,
    )
    data_opt = dict(loaded.option["eval"]["data"])
    features = _video_feature(Path(args.video_feature), data_opt)
    text = load_precomputed_text_feature(
        args.text_feature,
        normalize=bool(data_opt.get("normalize_text", False)),
    )
    query = (text, np.ones(text.shape[1], dtype=bool))
    backend = SnAGTorchBackend.from_upstream_option(
        loaded.model, loaded.option, device=args.device,
    )
    reference = backend.raw_predict(features, query)

    clip_stride = int(data_opt["clip_stride"]) * int(data_opt.get("downsample_rate", 1))
    clip_size = int(data_opt["clip_size"])
    meta = VideoMeta(
        args.video_id, args.duration, args.fps,
        max(1, int(math.ceil(args.duration * args.fps))),
    )
    writer = SnAGAdaptWriter(meta, SnAGAdaptConfig(mode="full", storage_dtype="float32"))
    for index, feature in enumerate(features):
        start = index * clip_stride / args.fps
        end = min((index * clip_stride + clip_size) / args.fps, args.duration)
        if end <= start:
            raise ValueError("feature provenance extends beyond declared video duration")
        writer.ingest_step(feature, start, end, f"{args.video_id}:clip:{index}")

    with tempfile.TemporaryDirectory(prefix="snag-g0-") as temporary:
        snapshot_root = Path(temporary) / "snapshot"
        manifest = writer.freeze(args.duration, snapshot_root)
        snapshot = SnAGSnapshotReader(snapshot_root)
        candidate = backend.raw_predict(snapshot.features(), query)
    raw_parity = compare_raw_traces(reference, candidate)
    official = SnAGOfficialGridBackend(
        loaded.model, loaded.option, load_upstream_nms_extension(args.upstream_root),
        fps=args.fps,
        clip_size=clip_size,
        clip_stride=clip_stride,
        duration_s=args.duration,
        device=args.device,
    )
    reference_spans = official.predict(features, (), query)
    candidate_spans = official.predict(snapshot.features(), snapshot.items, query)
    post_nms_parity = compare_ranked_spans(reference_spans, candidate_spans)
    expected_metrics = json.loads(args.expected_official_metrics.read_text(encoding="utf-8"))
    actual_metrics = json.loads(args.actual_official_metrics.read_text(encoding="utf-8"))
    official_metric_parity = compare_official_metrics(
        expected_metrics, actual_metrics,
        absolute_tolerance_points=args.metric_tolerance_points,
    )
    gate = g0_gate_report(
        loader_provenance=loaded.provenance(),
        raw_parity=raw_parity,
        post_nms_parity=post_nms_parity,
        official_metric_parity=official_metric_parity,
    )
    configuration = {
        "gate": "SnAG-G0-upstream-parity",
        "stage": "real-model-snapshot-roundtrip",
        "upstream_root": str(Path(args.upstream_root).resolve()),
        "option": str(Path(args.opt).resolve()),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_key": args.checkpoint_key,
        "video_feature": str(Path(args.video_feature).resolve()),
        "text_feature": str(Path(args.text_feature).resolve()),
        "video_id": args.video_id,
        "duration": args.duration,
        "fps": args.fps,
        "device": args.device,
        "storage_dtype": "float32",
        "expected_official_metrics": str(args.expected_official_metrics.resolve()),
        "actual_official_metrics": str(args.actual_official_metrics.resolve()),
        "metric_tolerance_points": args.metric_tolerance_points,
    }
    output = versioned_result_path(args.output)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite G0 parity artifact: {output}")
    write_provenance(
        output,
        configuration=configuration,
        config_filename="config.resolved.json",
        extra_metadata={
            "snag": loaded.provenance(),
            "inputs": {
                "video_feature_sha256": sha256_file(Path(args.video_feature)),
                "text_feature_sha256": sha256_file(Path(args.text_feature)),
                "expected_official_metrics_sha256": sha256_file(args.expected_official_metrics),
                "actual_official_metrics_sha256": sha256_file(args.actual_official_metrics),
            },
        },
    )
    (output / "g0_gate.json").write_text(
        json.dumps(gate, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    (output / "snapshot_manifest_summary.json").write_text(
        json.dumps({
            "state_bytes": manifest.state_bytes,
            "token_count": manifest.token_count,
            "feature_dim": manifest.feature_dim,
            "storage_dtype": manifest.storage_dtype,
        }, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_artifact_manifest(output)
    print(json.dumps({"output": str(output), "g0_passed": gate["passed"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
