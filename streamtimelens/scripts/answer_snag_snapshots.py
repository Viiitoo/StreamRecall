#!/usr/bin/env python3
"""Answer GT-free late queries from immutable SnAG snapshots."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
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
from streamtimelens.baselines.snag_training import load_physical_reader_checkpoint  # noqa: E402
from streamtimelens.evaluation.snag_effect import percentile  # noqa: E402
from streamtimelens.observer.clip_encoder import FrozenCLIPEncoder  # noqa: E402


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--snapshot-index", type=Path, required=True)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def main() -> int:
    args = _args()
    config = load_snag_config(args.config)
    if not config.checkpoint or not config.checkpoint_sha256:
        raise ValueError("SnAG inference requires a frozen checkpoint in config")
    checkpoint = Path(config.checkpoint).expanduser().resolve(strict=True)
    actual_checkpoint_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    if actual_checkpoint_hash != config.checkpoint_sha256:
        raise ValueError("SnAG checkpoint hash does not match frozen config")
    model, payload = load_physical_reader_checkpoint(checkpoint, device=args.device)
    if payload["model_config"] != resolved_snag_config_dict(config)["reader"]:
        raise ValueError("SnAG checkpoint reader configuration mismatch")
    backend = SnAGPhysicalTimeBackend(model, config.reader, device=args.device)
    reader = SnAGAdaptReader(backend)
    encoder = FrozenCLIPEncoder(
        config.feature.model, device=args.device, batch_size=1,
        visual_fps=config.feature.feature_fps,
    )
    snapshots = {
        (str(row["video_id"]), round(float(row["rho"]), 8)): Path(row["path"])
        for row in _jsonl(args.snapshot_index)
    }
    queries = _jsonl(args.queries)
    forbidden = {"gt", "gt_span", "ground_truth", "ground_truth_span", "spans"}
    if any(forbidden & {key.lower() for key in row} for row in queries):
        raise ValueError("SnAG query manifest exposes ground truth")
    predictions = []
    latencies = []
    all_immutable = True
    ratios = tuple(config.training.arrival_ratios) if config.training else (0.25, 0.5, 0.75, 1.0)
    for query_row in queries:
        embedding = encoder.encode_text(str(query_row["query"]))
        for rho in ratios:
            path = snapshots[(str(query_row["video_id"]), round(float(rho), 8))]
            snapshot = SnAGSnapshotReader(path)
            duration = float(snapshot.manifest.video_meta["duration_s"])
            if "duration_s" in query_row and abs(float(query_row["duration_s"]) - duration) > 1e-6:
                raise ValueError("query/video snapshot duration mismatch")
            before = snapshot_fingerprint(snapshot)
            started = time.perf_counter()
            spans = reader.answer(
                snapshot, SnAGPhysicalQuery(embedding, snapshot.manifest.t_q), topk=5,
            )
            repeated = reader.answer(
                SnAGSnapshotReader(path),
                SnAGPhysicalQuery(embedding, snapshot.manifest.t_q),
                topk=5,
            )
            latencies.append(time.perf_counter() - started)
            all_immutable &= (
                before == snapshot_fingerprint(SnAGSnapshotReader(path))
                and ranked_spans_close(spans, repeated)
            )
            predictions.append({
                "video_id": query_row["video_id"], "query_id": query_row["query_id"],
                "rho_q": rho, "t_q": snapshot.manifest.t_q,
                "spans": [[span.start_s, span.end_s, span.score] for span in spans],
                "span": [spans[0].start_s, spans[0].end_s] if spans else None,
                "status": "ok" if spans else "not_found",
            })
    output = versioned_result_path(args.output)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite SnAG answers: {output}")
    output.mkdir(parents=True)
    (output / "predictions.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in predictions), encoding="utf-8",
    )
    audit = {
        "schema_version": 1,
        "method": config.method,
        "passed": all_immutable and len(predictions) == len(queries) * len(ratios),
        "checks": {
            "gt_free_query_process": True,
            "snapshot_only_immutable": all_immutable,
            "independent_queries": True,
            "complete_predictions": len(predictions) == len(queries) * len(ratios),
            "checkpoint_hash_match": True,
        },
    }
    cost = {
        "readout_count": len(latencies),
        "readout_p50_s": percentile(latencies, 0.5),
        "readout_p95_s": percentile(latencies, 0.95),
    }
    (output / "query_protocol_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    (output / "query_cost.json").write_text(
        json.dumps(cost, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    write_provenance(
        output, configuration=resolved_snag_config_dict(config),
        config_path=args.config, config_filename="config.resolved.yaml",
        extra_metadata={
            "snapshot_index_sha256": hashlib.sha256(args.snapshot_index.read_bytes()).hexdigest(),
            "query_manifest_sha256": hashlib.sha256(args.queries.read_bytes()).hexdigest(),
            "checkpoint_sha256": actual_checkpoint_hash,
        },
    )
    write_artifact_manifest(output)
    print(json.dumps({"output": str(output), "predictions": len(predictions), "audit": audit["passed"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
