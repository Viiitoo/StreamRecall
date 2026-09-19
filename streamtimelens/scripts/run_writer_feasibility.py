#!/usr/bin/env python3
"""Aggregate the fixed TimeLens-100K-dev writer feasibility comparison."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT / "src"))
sys.path.insert(0, str(PACKAGE_ROOT.parent / "src"))

from baas.provenance import versioned_result_path, write_provenance
from streamtimelens.evaluation.writer_feasibility import (
    evaluate_writer_records, feasibility_decision, semantic_reservoir_oracle_coverage,
)
from streamtimelens.writer.prompts import WRITER_PROMPT_VERSION, build_writer_prompt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--records", type=Path, required=True, nargs="+",
        help="One or more paired per-model raw writer JSONL files",
    )
    parser.add_argument("--output", type=Path, required=True, help="Path below work/results")
    parser.add_argument("--expected-chunks", type=int, default=100)
    parser.add_argument("--semantic-oracle-coverage", type=float)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = [
        json.loads(line) for path in args.records
        for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    metrics = evaluate_writer_records(rows, expected_chunks=args.expected_chunks)
    semantic_coverage = (
        args.semantic_oracle_coverage
        if args.semantic_oracle_coverage is not None
        else semantic_reservoir_oracle_coverage(rows)
    )
    decision = feasibility_decision(metrics, semantic_oracle_coverage=semantic_coverage)
    prompt = build_writer_prompt(segment=(0, 1), sampled_timestamps=(0, 1))
    configuration = {
        "records": [str(path.resolve()) for path in args.records],
        "record_sha256": {
            str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in args.records
        },
        "expected_chunks": args.expected_chunks,
        "semantic_oracle_coverage": semantic_coverage,
        "semantic_oracle_source": (
            "cli_override" if args.semantic_oracle_coverage is not None
            else "best_span_from_fixed_sampled_timestamp_pairs"
        ),
        "prompt_version": WRITER_PROMPT_VERSION, "prompt_template_sha256": prompt.template_sha256,
    }
    output = versioned_result_path(args.output)
    write_provenance(output, configuration=configuration, config_filename="config.resolved.json")
    payload = {**metrics, **decision, "prompt_version": WRITER_PROMPT_VERSION,
               "prompt_template_sha256": prompt.template_sha256}
    (output / "metrics.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# Writer feasibility report", "", f"Decision: **{decision['decision']}**",
        f"Selected writer: **{decision['selected_writer']}**", "",
        "| model | JSON valid | schema usable | events/chunk | GT coverage IoU | endpoint error (s) | GPU s |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for model, item in sorted(metrics["models"].items()):
        lines.append(
            f"| {model} | {item['json_valid_rate']:.3f} | {item['schema_usable_rate']:.3f} | "
            f"{item['mean_events_per_chunk']:.3f} | {item['mean_gt_coverage_iou'] or 0:.3f} | "
            f"{item['mean_endpoint_abs_error_s'] or 0:.3f} | {item['gpu_s']:.3f} |"
        )
    (output / "writer_feasibility_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), **decision}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
