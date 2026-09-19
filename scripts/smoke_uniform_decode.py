#!/usr/bin/env python3
"""CPU-only decode smoke test for the fixed 20 Charades audit samples."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from baas.dataset import decode_selected_frames, read_video_metadata
from baas.provenance import versioned_result_path, write_provenance
from baas.sampling import uniform_plan


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", default="artifacts/p1_charades_20.json")
    parser.add_argument("--video-root", required=True)
    parser.add_argument("--budget", type=int, default=2048)
    parser.add_argument("--frame-token-cap", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--output", required=True)
    parser.add_argument("--container-image", default=os.environ.get("BAAS_CONTAINER_IMAGE"))
    parser.add_argument("--container-image-id", default=os.environ.get("BAAS_CONTAINER_IMAGE_ID"))
    args = parser.parse_args()
    if args.budget <= 0:
        parser.error("--budget must be positive")
    if args.frame_token_cap <= 0:
        parser.error("--frame-token-cap must be positive")
    if args.limit <= 0:
        parser.error("--limit must be positive")
    output = versioned_result_path(args.output)
    if output.parent.exists():
        raise FileExistsError(f"refusing to overwrite existing smoke run: {output.parent}")
    output.parent.mkdir(parents=True, exist_ok=False)
    samples_path = Path(args.samples).resolve()
    samples_sha256 = hashlib.sha256(samples_path.read_bytes()).hexdigest()
    write_provenance(
        output.parent,
        configuration={
            "samples": str(samples_path), "samples_sha256": samples_sha256,
            "video_root": str(Path(args.video_root).resolve()),
            "budget": args.budget, "frame_token_cap": args.frame_token_cap, "seed": args.seed,
            "limit": args.limit, "output": str(output), "decoder": "pyav",
            "metadata_include_vfr_timestamps": True, "timestamp_mode": "normalized_frame_pts",
            "sampling_method": "uniform", "endpoint_interval": "[0,duration)",
            "container_image": args.container_image, "container_image_id": args.container_image_id,
        },
        config_filename="config.resolved.json",
    )
    samples = json.loads(samples_path.read_text(encoding="utf-8"))[: args.limit]
    records = []
    for sample in samples:
        path = Path(args.video_root) / f"{sample['video_id']}.mp4"
        metadata = read_video_metadata(path)
        plan = uniform_plan(
            metadata, args.budget, args.seed, video_id=sample["video_id"],
            max_frames=max(1, args.budget // args.frame_token_cap),
        )
        frames = decode_selected_frames(path, plan.frame_indices)
        if len(frames) != len(plan.frame_indices):
            raise RuntimeError(f"decoded frame count mismatch for {path}")
        records.append({"video_id": sample["video_id"], "video_path": str(path), "plan": plan.to_record()})
    output.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in records), encoding="utf-8")
    print(f"decoded {len(records)} fixed samples -> {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
