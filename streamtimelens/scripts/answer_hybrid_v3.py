#!/usr/bin/env python3
"""Answer one query with Hybrid V3 from an immutable snapshot (no video option)."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
WORK_ROOT = PACKAGE_ROOT.parent
sys.path.insert(0, str(PACKAGE_ROOT / "src"))
sys.path.insert(0, str(WORK_ROOT / "src"))

from baas.provenance import versioned_result_path, write_provenance
from streamtimelens.config import read_hybrid_v3_config
from streamtimelens.model_identity import verify_registered_model
from streamtimelens.observer.clip_encoder import FrozenCLIPEncoder
from streamtimelens.protocol.snapshot import SnapshotReader
from streamtimelens.refiner.model_service import TimeLensModelService
from streamtimelens.refiner.multicandidate_prompt import (
    multicandidate_template_sha256,
)
from streamtimelens.retrieval.clip_timelens_hybrid import answer_hybrid_snapshot


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _asset_path(value: str) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else WORK_ROOT / path).resolve()


def _hybrid_output_root(path: Path) -> Path:
    resolved = versioned_result_path(path)
    relative = resolved.relative_to((WORK_ROOT / "results").resolve())
    if len(relative.parts) < 3 or relative.parts[1] != "hybrid_v3":
        raise ValueError("Hybrid V3 output must be under results/hybrid_v3/<run-name>")
    return resolved


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--query", required=True)
    parser.add_argument("--query-id", required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--clip-device")
    parser.add_argument("--device-map", default="auto")
    args = parser.parse_args()

    config = read_hybrid_v3_config(args.config)
    frozen_path = PACKAGE_ROOT / "configs" / "frozen_visual_v2.yaml"
    observed_frozen_sha = _sha256(frozen_path)
    if observed_frozen_sha != config.snapshot_source.frozen_config_sha256:
        raise ValueError("registered frozen_visual_v2.yaml hash does not match the checkout")
    snapshot = SnapshotReader(args.snapshot)
    clip_path = _asset_path(config.retrieval.clip_model)
    model_path = _asset_path(config.brain.model)
    registry_path = WORK_ROOT / "artifacts" / "dev" / "model_hashes.json"
    clip_identity = verify_registered_model(
        registry_path, model_key="clip", model_root=clip_path,
        revision=config.retrieval.clip_revision,
        content_sha256=config.retrieval.clip_content_sha256, verify_files=True,
    )
    timelens_identity = verify_registered_model(
        registry_path, model_key=config.brain.model_registry_key, model_root=model_path,
        revision=config.brain.model_revision,
        content_sha256=config.brain.model_content_sha256, verify_files=False,
    )
    encoder = FrozenCLIPEncoder(clip_path, device=args.clip_device, batch_size=1)
    service = TimeLensModelService.get(model_path, device_map=args.device_map)
    output = answer_hybrid_snapshot(
        query_id=args.query_id, query=args.query, snapshot=snapshot,
        encoder=encoder, service=service, config=config,
    )
    payload = output.to_dict()
    prompt_trace = next(
        (event for event in output.query_trace if event.get("kind") == "timelens_prompt"),
        {},
    )
    if Path(args.query_id).name != args.query_id or not args.query_id.replace("-", "").replace("_", "").isalnum():
        raise ValueError("query ID must be a safe filename component")
    output_root = _hybrid_output_root(args.output)
    destination = output_root / "predictions" / f"{args.query_id}.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    query_manifest_sha256 = hashlib.sha256(json.dumps(
        {"query_id": args.query_id, "query": args.query},
        sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode("utf-8")).hexdigest()
    configuration = {
        **config.canonical_dict(),
        "runtime": {
            "snapshot": str(args.snapshot.resolve()),
            "snapshot_manifest_config_sha256": snapshot.manifest.config_hash,
            "snapshot_state_bytes": snapshot.manifest.state_bytes,
            "query_id": args.query_id,
            "query_manifest_sha256": query_manifest_sha256,
            "clip_device": args.clip_device,
            "device_map": args.device_map,
            "prompt_template_sha256": multicandidate_template_sha256(
                config.brain.prompt_version,
            ),
        },
    }
    write_provenance(
        output_root, configuration=configuration,
        config_path=args.config, config_filename="config.resolved.yaml",
        extra_metadata={
            "hybrid_config_sha256": config.sha256,
            "frozen_config_observed_sha256": observed_frozen_sha,
            "query_manifest_sha256": query_manifest_sha256,
            "model_service_hashes": dict(service.hashes),
            "clip_identity": clip_identity,
            "timelens_identity": timelens_identity,
            "rendered_prompt_sha256": prompt_trace.get("prompt_sha256"),
        },
    )
    destination.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    calls = sum(
        int(event.get("calls", 0)) for event in output.query_trace
        if event.get("kind") in ("timelens_inference", "timelens_exception")
    )
    audit = {
        "schema_version": 1, "samples": 1,
        "passed": calls <= 1 and len(output.selected_frame_refs) <= config.brain.max_unique_frames,
        "snapshot_only": True, "query_manifest_ground_truth_free": True,
        "model_calls": calls, "unique_frames": len(output.selected_frame_refs),
        "max_unique_frames": config.brain.max_unique_frames,
    }
    inference = next((
        event for event in output.query_trace
        if event.get("kind") in ("timelens_inference", "timelens_exception")
    ), {})
    clip = next((event for event in output.query_trace if event.get("kind") == "clip_retrieval"), {})
    completed = next((event for event in output.query_trace if event.get("kind") == "query_completed"), {})
    cost = {
        "schema_version": 1, "samples": 1,
        "clip_query_wall_s": float(clip.get("clip_query_wall_s") or 0),
        "timelens_wall_s": float(inference.get("wall_s") or 0),
        "total_query_wall_s": float(completed.get("wall_s") or 0),
        "generated_tokens": int(inference.get("generated_tokens") or 0),
        "peak_cuda_bytes": inference.get("peak_cuda_bytes"),
        "snapshot_state_bytes": snapshot.manifest.state_bytes,
        "fallback_count": int(output.fallback_used),
    }
    (output_root / "protocol_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    (output_root / "cost_summary.json").write_text(
        json.dumps(cost, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    (output_root / "accuracy_summary.json").write_text(json.dumps({
        "schema_version": 1, "status": "not_evaluated",
        "reason": "query worker is ground-truth-free; run Hybrid V3 evaluation separately",
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(destination),
        "final_selection_reason": output.final_selection_reason,
        "fallback_used": output.fallback_used,
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
