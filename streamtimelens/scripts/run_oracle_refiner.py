#!/usr/bin/env python3
"""Build or execute the S2 Oracle local-refiner experiment."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import sys
from pathlib import Path

WORK_ROOT = Path(__file__).resolve().parents[2]
for source in (WORK_ROOT / "src", WORK_ROOT / "streamtimelens" / "src"):
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))

from baas.provenance import versioned_result_path, write_provenance
from streamtimelens.protocol.oracle import (
    OraclePrediction, build_oracle_examples, evaluate_oracle_conditions,
    evaluate_oracle_predictions, oracle_jsonl, oracle_gate_decision,
)


def _jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _prediction(item: dict[str, object]) -> OraclePrediction:
    span = item.get("predicted_span")
    return OraclePrediction(
        example_id=str(item["example_id"]), mode=str(item["mode"]),  # type: ignore[arg-type]
        predicted_span=None if span is None else tuple(map(float, span)),  # type: ignore[arg-type]
        status=str(item["status"]), raw_answer=str(item.get("raw_answer", "")),
    )


def _prediction_json(prediction: OraclePrediction) -> str:
    return json.dumps(
        {**prediction.__dict__, "predicted_span": prediction.predicted_span},
        ensure_ascii=False, sort_keys=True,
    ) + "\n"


def _load_runner(spec: str):
    module_name, separator, attribute = spec.partition(":")
    if not separator:
        raise ValueError("runner must use module:factory syntax")
    factory = getattr(importlib.import_module(module_name), attribute)
    return factory()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", required=True, type=Path, help="Dev JSONL with query/video/GT/FPS metadata")
    parser.add_argument("--output", required=True, type=Path, help="Path below this checkout's results directory")
    parser.add_argument("--margins", nargs="+", type=float, default=[2, 5, 10])
    parser.add_argument("--frame-counts", nargs="+", type=int, default=[16, 32])
    parser.add_argument("--model", type=Path, help="Use the built-in TimeLens runner with this checkpoint")
    parser.add_argument("--video-root", type=Path, help="Directory containing <video_id>.mp4 for the built-in runner")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--runner", help="module:factory returning an OracleRunner backed by real inference")
    group.add_argument(
        "--predictions", type=Path, nargs="+",
        help="Evaluate and merge one or more existing prediction JSONL shards",
    )
    parser.add_argument("--device-map", default="auto", help="Transformers device map for built-in inference")
    parser.add_argument("--num-shards", type=int, default=1, help="Stable query-level inference shard count")
    parser.add_argument("--shard-index", type=int, default=0, help="Zero-based query-level inference shard index")
    parser.add_argument("--resume", action="store_true", help="Resume the built-in runner from predictions.jsonl")
    parser.add_argument(
        "--resume-from", type=Path, nargs="+", default=[],
        help="Seed this shard from prediction JSONLs produced by an earlier partitioning",
    )
    args = parser.parse_args()

    if args.num_shards <= 0 or not 0 <= args.shard_index < args.num_shards:
        parser.error("--shard-index must be in [0, --num-shards)")
    if args.predictions and (
        args.num_shards != 1 or args.shard_index != 0 or args.resume or args.resume_from
    ):
        parser.error("prediction aggregation cannot be combined with sharding or --resume")
    if args.resume_from and not args.model:
        parser.error("--resume-from requires the built-in --model runner")
    all_examples = build_oracle_examples(
        _jsonl(args.annotations), margins_s=args.margins, frame_counts=args.frame_counts,
    )
    examples = all_examples
    if args.num_shards > 1:
        examples = [
            example for example in all_examples
            if int(hashlib.sha256(example.query_id.encode("utf-8")).hexdigest(), 16)
            % args.num_shards == args.shard_index
        ]
        if not examples:
            parser.error("selected Oracle shard is empty")
    output_dir = versioned_result_path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "oracle_examples.jsonl").write_text(oracle_jsonl(examples), encoding="utf-8")

    predictions = []
    metrics = None
    model_hashes = None
    if args.model:
        if args.runner or args.predictions:
            parser.error("--model cannot be combined with --runner or --predictions")
        if args.video_root is None:
            parser.error("--video-root is required with --model")
        import decord
        from streamtimelens.refiner.model_service import TimeLensModelService
        from streamtimelens.refiner.oracle_inference import TimeLensOracleRunner

        readers = {}
        def load_frames(example, indices):
            if example.video_id not in readers:
                readers[example.video_id] = decord.VideoReader(str(args.video_root / f"{example.video_id}.mp4"))
            return list(readers[example.video_id].get_batch(list(indices)).asnumpy())
        service = TimeLensModelService.get(args.model, device_map=args.device_map)
        prediction_path = output_dir / "predictions.jsonl"
        existing = []
        if args.resume and prediction_path.is_file():
            existing = [_prediction(item) for item in _jsonl(prediction_path)]
        elif prediction_path.exists():
            parser.error(f"predictions already exist; pass --resume or choose another output: {prediction_path}")
        example_ids = {example.example_id for example in examples}
        existing_by_key = {}
        for item in existing:
            key = (item.example_id, item.mode)
            if key in existing_by_key:
                parser.error(f"duplicate local resume prediction for {item.example_id}/{item.mode}")
            existing_by_key[key] = item
        for resume_path in args.resume_from:
            for item in map(_prediction, _jsonl(resume_path)):
                if item.example_id not in example_ids:
                    continue
                key = (item.example_id, item.mode)
                previous = existing_by_key.setdefault(key, item)
                if previous != item:
                    parser.error(f"conflicting resume prediction for {item.example_id}/{item.mode}")
        existing = list(existing_by_key.values())
        with prediction_path.open("a", encoding="utf-8") as stream:
            def checkpoint(prediction: OraclePrediction) -> None:
                stream.write(_prediction_json(prediction))
                stream.flush()

            predictions, metrics = TimeLensOracleRunner(service, load_frames).run(
                examples, existing_predictions=existing, prediction_callback=checkpoint,
            )
        model_hashes = service.hashes
    elif args.runner:
        predictions, metrics = _load_runner(args.runner).run(examples)
    elif args.predictions:
        predictions = [
            _prediction(item) for path in args.predictions for item in _jsonl(path)
        ]
        metrics = evaluate_oracle_predictions(examples, predictions)
        shard_hashes = []
        for path in args.predictions:
            provenance_path = path.parent / "provenance.json"
            if provenance_path.is_file():
                hashes = json.loads(provenance_path.read_text(encoding="utf-8")).get("model_hashes")
                if hashes:
                    shard_hashes.append(hashes)
        if shard_hashes:
            canonical = {json.dumps(item, sort_keys=True) for item in shard_hashes}
            if len(canonical) != 1:
                raise ValueError("prediction shards were produced by different model hashes")
            model_hashes = shard_hashes[0]
    if predictions:
        predictions = sorted(predictions, key=lambda item: (item.example_id, item.mode))
        (output_dir / "predictions.jsonl").write_text(
            "".join(_prediction_json(prediction) for prediction in predictions), encoding="utf-8",
        )
    if metrics is not None:
        metrics["conditions"] = evaluate_oracle_conditions(examples, predictions)
        (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        decision = oracle_gate_decision(metrics)
        gate = metrics["p0_gate"]
        condition_lines = [
            "", "| condition | dense mIoU | sparse mIoU | drop | start bias | end bias | passed |",
            "| --- | ---: | ---: | ---: | ---: | ---: | :---: |",
        ]
        for name, condition in sorted(metrics["conditions"].items()):
            condition_gate = condition["p0_gate"]
            dense = condition["metrics"]["dense_crop"]
            sparse = condition["metrics"]["sparse_adapter"]
            condition_lines.append(
                f"| {name} | {dense['miou']:.4f} | {sparse['miou']:.4f} | "
                f"{condition_gate['miou_drop']:.4f} | {sparse['signed_start_bias_s']:.4f} | "
                f"{sparse['signed_end_bias_s']:.4f} | {condition_gate['passed']} |"
            )
        (output_dir / "oracle_refiner_report.md").write_text(
            "# Oracle local-refiner P0 gate\n\n"
            f"Decision: **{decision['decision']}**\n\n"
            f"Reason: {decision['reason']}\n\n"
            f"- sparse-vs-dense mIoU drop: {gate['miou_drop']:.6f}\n"
            f"- allowed mIoU drop: {gate['allowed_miou_drop']:.6f}\n"
            f"- signed bias tolerance: {gate['bias_tolerance_s']:.6f}s\n"
            + "\n".join(condition_lines) + "\n",
            encoding="utf-8",
        )

    write_provenance(output_dir, configuration={
        "annotations": str(args.annotations.resolve()), "margins_s": args.margins,
        "frame_counts": args.frame_counts, "strategies": ["uniform", "boundary-heavy"],
        "runner": args.runner,
        "predictions": [str(path.resolve()) for path in args.predictions] if args.predictions else None,
        "model": str(args.model.resolve()) if args.model else None,
        "video_root": str(args.video_root.resolve()) if args.video_root else None,
        "device_map": args.device_map, "num_shards": args.num_shards,
        "shard_index": args.shard_index, "resume": args.resume,
        "resume_from": [str(path.resolve()) for path in args.resume_from],
        "annotation_sha256": _sha256(args.annotations), "example_count": len(examples),
    }, config_filename="config.resolved.json", extra_metadata={
        "model_hashes": model_hashes,
        "input_hashes": {
            "annotations": _sha256(args.annotations),
            "prediction_shards": (
                {str(path.resolve()): _sha256(path) for path in args.predictions}
                if args.predictions else {}
            ),
            "resume_prediction_shards": {
                str(path.resolve()): _sha256(path) for path in args.resume_from
            },
        },
    })
    print(output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
