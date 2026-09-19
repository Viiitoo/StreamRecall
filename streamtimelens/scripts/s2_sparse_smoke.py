#!/usr/bin/env python3
"""Repeat one fixed sparse TimeLens sample and persist deterministic smoke evidence."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

WORK_ROOT = Path(__file__).resolve().parents[2]
for source in (WORK_ROOT / "src", WORK_ROOT / "streamtimelens" / "src", WORK_ROOT / "third_party" / "TimeLens"):
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))

import decord

from baas.provenance import versioned_result_path, write_provenance
from streamtimelens.refiner.model_service import TimeLensModelService
from streamtimelens.refiner.prompts import SPARSE_LOCAL_VERSION, build_grounding_prompt, video_messages
from streamtimelens.refiner.timelens_adapter import SparseFrame, prepare_sparse_video


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--samples", required=True, type=Path)
    parser.add_argument("--video-root", required=True, type=Path)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--k-frames", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    sample = json.loads(args.samples.read_text(encoding="utf-8"))[args.sample_index]
    video_path = args.video_root / f"{sample['video_id']}.mp4"
    reader = decord.VideoReader(str(video_path))
    fps, total = float(reader.get_avg_fps()), len(reader)
    indices = sorted(set(round(i * (total - 1) / max(1, args.k_frames - 1)) for i in range(args.k_frames)))
    images = reader.get_batch(indices).asnumpy()
    prepared = prepare_sparse_video(
        [SparseFrame(index, index / fps, image) for index, image in zip(indices, images)],
        original_fps=fps, total_num_frames=total, k_frames=args.k_frames,
    )
    prompt = build_grounding_prompt(
        SPARSE_LOCAL_VERSION, query=sample["query"], candidate=(0.0, float(sample["duration"])), card_summary=None,
    )
    service = TimeLensModelService.get(args.model)
    answers = [service.generate(video_messages(prompt), [prepared.processor_video], args.max_new_tokens) for _ in range(2)]
    result = {
        "schema_version": 1, "video_id": sample["video_id"], "indices": indices,
        "timestamp_audit": prepared.timestamp_audit, "answers": answers,
        "deterministic": answers[0] == answers[1], "model_hashes": service.hashes,
        "last_call_stats": service.last_call_stats,
    }
    output_dir = versioned_result_path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "smoke.json").write_text(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    write_provenance(output_dir, configuration={
        "model": str(args.model.resolve()), "samples": str(args.samples.resolve()),
        "video_root": str(args.video_root.resolve()), "sample_index": args.sample_index,
        "k_frames": args.k_frames, "max_new_tokens": args.max_new_tokens,
        "prompt_version": prompt.version, "prompt_sha256": prompt.sha256,
    }, config_filename="config.resolved.json", extra_metadata={"model_hashes": service.hashes})
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result["deterministic"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
