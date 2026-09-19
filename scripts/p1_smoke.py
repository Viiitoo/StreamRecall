#!/usr/bin/env python3
"""Run one fixed Charades-TimeLens sample through the unmodified TimeLens path."""

import argparse
import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from baas.provenance import versioned_result_path, write_provenance
from evaluation.utils import GroundingDataset
from timelens.utils import extract_time


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--samples", required=True)
    parser.add_argument("--video-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--min-tokens", type=int, default=64)
    parser.add_argument("--total-tokens", type=int, default=4096)
    parser.add_argument("--fps", type=int, default=2)
    return parser.parse_args()


def main():
    args = parse_args()
    samples = json.loads(Path(args.samples).read_text())
    sample = samples[args.sample_index]
    video_path = Path(args.video_root) / f"{sample['video_id']}.mp4"
    if not video_path.is_file():
        raise FileNotFoundError(video_path)

    anno = {
        "video_path": str(video_path),
        "duration": sample["duration"],
        "query": sample["query"],
        "span": [sample["span"]],
    }
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_path,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto",
    ).eval()
    processor = AutoProcessor.from_pretrained(
        args.model_path,
        padding_side="left",
        do_resize=False,
        trust_remote_code=True,
    )
    inputs = GroundingDataset([anno], processor, args)[0]["inputs"].to("cuda")
    output_ids = model.generate(
        **inputs,
        do_sample=False,
        temperature=None,
        top_p=None,
        top_k=None,
        max_new_tokens=512,
    )
    answer = processor.batch_decode(
        [output_ids[0][len(inputs.input_ids[0]) :]],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]
    result = {
        "sample_index": args.sample_index,
        "video_id": sample["video_id"],
        "video_path": str(video_path),
        "duration": sample["duration"],
        "query": sample["query"],
        "ground_truth": sample["span"],
        "sampling": {
            "min_tokens": args.min_tokens,
            "total_tokens": args.total_tokens,
            "fps": args.fps,
        },
        "answer": answer,
        "timestamps": extract_time(answer),
    }
    output_path = versioned_result_path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_provenance(
        output_path.parent,
        configuration={
            "model_path": str(Path(args.model_path).resolve()),
            "samples": str(Path(args.samples).resolve()),
            "video_root": str(Path(args.video_root).resolve()),
            "sample_index": args.sample_index,
            "min_tokens": args.min_tokens,
            "total_tokens": args.total_tokens,
            "fps": args.fps,
        },
        config_filename="config.resolved.json",
    )
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
