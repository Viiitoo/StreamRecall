"""Single-process runner for auditable uniform TimeLens baselines."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import sys
import traceback
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any, Iterable

import yaml

from .dataset import prepare_inputs, read_video_metadata
from .provenance import versioned_result_path, write_provenance
from .sampling import SamplingError, SamplingPlan, uniform_plan, with_actual_visual_tokens


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TIMELENS_ROOT = PROJECT_ROOT / "third_party" / "TimeLens"


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"not JSON serializable: {type(value)!r}")


def _write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True, default=_json_default) + "\n")


def _video_id(anno: dict[str, Any]) -> str:
    return str(anno.get("video_id") or Path(anno["video_path"]).stem)


def _prediction_key(anno: dict[str, Any]) -> str:
    span = anno["span"]
    if isinstance(span[0], (list, tuple)):
        span = span[0]
    return f"{Path(anno['video_path']).name}>>>{anno['query']}>>>{span}"


def _load_annos(dataset: str, split: str) -> tuple[list[dict[str, Any]], Any]:
    try:
        from timelens.dataset.timelens_data import DATASET_DICT
    except ImportError as exc:  # pragma: no cover - requires TimeLens environment
        raise RuntimeError("set PYTHONPATH to include third_party/TimeLens") from exc
    if dataset not in DATASET_DICT:
        raise ValueError(f"unknown dataset {dataset!r}; choices: {', '.join(sorted(DATASET_DICT))}")
    dataset_class = DATASET_DICT[dataset]
    # Official dataset classes intentionally use paths relative to the TimeLens
    # repository. Load there, then freeze absolute video paths for this runner.
    previous_cwd = Path.cwd()
    try:
        os.chdir(TIMELENS_ROOT)
        annos = dataset_class.load_annos(split=split)
    finally:
        os.chdir(previous_cwd)
    for anno in annos:
        video_path = Path(anno["video_path"])
        if not video_path.is_absolute():
            anno["video_path"] = str(TIMELENS_ROOT / video_path)
    annos.sort(key=lambda item: item["duration"], reverse=True)
    return annos, dataset_class


def _span_tuple(anno: dict[str, Any]) -> tuple[float, float]:
    span = anno["span"]
    if isinstance(span[0], (list, tuple)):
        span = span[0]
    return float(span[0]), float(span[1])


def _filter_fixed_samples(
    annos: list[dict[str, Any]], samples_file: Path | None
) -> list[dict[str, Any]]:
    """Select an ordered, exact audit set without consulting labels for sampling."""
    if samples_file is None:
        return annos
    requested = json.loads(samples_file.read_text(encoding="utf-8"))
    lookup: dict[tuple[str, str, tuple[float, float]], list[dict[str, Any]]] = {}
    for anno in annos:
        key = (_video_id(anno), str(anno["query"]), _span_tuple(anno))
        lookup.setdefault(key, []).append(anno)
    selected: list[dict[str, Any]] = []
    for item in requested:
        key = (
            str(item["video_id"]),
            str(item["query"]).strip().strip(".").strip(),
            (float(item["span"][0]), float(item["span"][1])),
        )
        matches = lookup.get(key, [])
        if len(matches) != 1:
            raise ValueError(f"fixed sample must match exactly one annotation, got {len(matches)}: {key}")
        selected.append(matches[0])
    return selected


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _select_annos(
    annos: list[dict[str, Any]], *, limit: int | None, chunk: int, index: int
) -> list[dict[str, Any]]:
    """Apply a global sample limit before deterministic round-robin sharding."""
    selected = annos if limit is None else annos[:limit]
    return selected[index::chunk]


def _load_model_and_processor(config: dict[str, Any]) -> tuple[Any, Any]:
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    model_path = Path(config["model_path"])
    if not model_path.is_absolute():
        model_path = PROJECT_ROOT / model_path
    model_path = str(model_path)
    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        attn_implementation=config.get("attn_implementation", "flash_attention_2"),
        device_map=config.get("device", "auto"),
    ).eval()
    processor = AutoProcessor.from_pretrained(
        model_path,
        padding_side="left",
        # Official video decoding is resized by qwen_vl_utils before a processor
        # configured with do_resize=False. We decode raw frames ourselves, so the
        # equivalent Qwen resize/patch alignment must happen in the processor.
        do_resize=bool(config.get("input", {}).get("processor_do_resize", True)),
        use_fast=bool(config.get("input", {}).get("processor_use_fast", True)),
        trust_remote_code=True,
    )
    return model, processor


def _set_random_seed(seed: int) -> None:
    """Freeze the same RNG families used by the official evaluation path."""
    import torch

    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:  # pragma: no cover - inference image includes numpy
        pass
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _move_to_cuda(inputs: Any) -> Any:
    # BatchFeature.to("cuda") is what the official evaluator uses. Keeping this
    # isolated makes unit tests independent of torch and allows CPU diagnostics.
    return inputs.to("cuda", non_blocking=True) if hasattr(inputs, "to") else inputs


def _make_plan_and_inputs(anno: dict[str, Any], config: dict[str, Any], processor: Any) -> tuple[Any, SamplingPlan, Any]:
    """Fit a uniform frame count to the *measured* token budget deterministically."""
    metadata_start = perf_counter()
    metadata = read_video_metadata(
        anno["video_path"],
        include_vfr_timestamps=bool(config["input"]["metadata_include_vfr_timestamps"]),
    )
    metadata_latency_ms = (perf_counter() - metadata_start) * 1000
    budget = int(config["budget"])
    frame_token_cap = int(config["input"].get("frame_token_cap", 128))
    if frame_token_cap <= 0:
        raise ValueError("input.frame_token_cap must be positive")
    count = min(metadata.frame_count, max(1, budget // frame_token_cap))
    last_error: Exception | None = None
    # The conservative cap normally succeeds immediately. A processor can still
    # choose a larger grid for unusual aspect ratios, so replan uniformly rather
    # than trimming a biased prefix.
    for _ in range(count):
        plan = uniform_plan(
            metadata,
            budget,
            int(config["seed"]),
            video_id=_video_id(anno),
            max_frames=count,
        )
        try:
            inputs, stats = prepare_inputs(
                anno["video_path"],
                anno["query"],
                plan,
                processor,
                max_pixels=config["input"].get("max_pixels"),
                min_pixels=config["input"].get("min_pixels"),
            )
            # PTS extraction is decoding work too, even though it happens before
            # the selected-frame pass; include it in the reported decode latency.
            stats = replace(stats, decode_latency_ms=stats.decode_latency_ms + metadata_latency_ms)
            return inputs, with_actual_visual_tokens(plan, stats.actual_visual_tokens), stats
        except SamplingError as exc:
            if "exceed budget" not in str(exc) or count == 1:
                raise
            last_error = exc
            # Monotonic image-token grids make this a deterministic, bounded fit.
            count = max(1, min(count - 1, (count * budget) // max(budget + 1, budget)))
    raise SamplingError(f"unable to fit visual tokens into budget {budget}: {last_error}")


def run_one(
    anno: dict[str, Any],
    config: dict[str, Any],
    model: Any | None = None,
    processor: Any | None = None,
    dataset_class: Any | None = None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Run one annotation, optionally reusing caller-owned model resources.

    The two-argument form is the public P0 interface.  The evaluation loop passes
    already loaded objects so thousands of samples do not reload the checkpoint.
    """
    import torch
    from timelens.utils import extract_time

    if model is None or processor is None:
        model, processor = _load_model_and_processor(config)
    if dataset_class is None:
        from timelens.dataset.timelens_data import DATASET_DICT

        dataset_name = config.get("dataset", {}).get("name") or config.get("execution", {}).get("dataset")
        if dataset_name not in DATASET_DICT:
            raise ValueError("config must identify a known dataset when run_one is called without context")
        dataset_class = DATASET_DICT[dataset_name]

    inputs, plan, token_stats = _make_plan_and_inputs(anno, config, processor)
    torch.cuda.reset_peak_memory_stats()
    generate_start = perf_counter()
    inputs = _move_to_cuda(inputs)
    output_ids = model.generate(
        **inputs,
        do_sample=False,
        temperature=None,
        top_p=None,
        top_k=None,
        max_new_tokens=int(config.get("max_new_tokens", 512)),
    )
    generate_latency_ms = (perf_counter() - generate_start) * 1000
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, output_ids)
    ]
    answer = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    timestamps = extract_time(answer) or [[float(anno["duration"]) + 10, float(anno["duration"]) + 20]]
    unit = getattr(dataset_class, "UNIT", 1.0)
    timestamps = [[round(start / unit) * unit, round(end / unit) * unit] for start, end in timestamps]
    prediction = {
        _prediction_key(anno): {"timestamps": timestamps, "answers": answer, "duration": anno["duration"]}
    }
    sample_id = _prediction_key(anno)
    sampling_record = {
        "sample_id": sample_id,
        "video_path": anno["video_path"],
        "query": anno["query"],
        "plan": plan.to_record(),
        "actual_visual_tokens": plan.actual_visual_tokens,
        "image_grid_thw": token_stats.image_grid_thw,
        "pixel_value_patches": token_stats.pixel_value_patches,
    }
    resource_record = {
        "sample_id": sample_id,
        "video_id": _video_id(anno),
        "frame_count": len(plan.frame_indices),
        "actual_visual_tokens": plan.actual_visual_tokens,
        "decode_latency_ms": round(token_stats.decode_latency_ms, 3),
        "preprocess_latency_ms": round(token_stats.preprocess_latency_ms, 3),
        "generate_latency_ms": round(generate_latency_ms, 3),
        "gpu_peak_memory_bytes": int(torch.cuda.max_memory_allocated()),
    }
    # Keep the complete per-sample audit self-contained as required by the P0
    # protocol; resources.csv is a convenient tabular projection of these fields.
    sampling_record.update({
        key: resource_record[key]
        for key in (
            "decode_latency_ms", "preprocess_latency_ms", "generate_latency_ms", "gpu_peak_memory_bytes"
        )
    })
    return prediction, sampling_record, resource_record


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--chunk", type=int, default=1)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--samples-file")
    parser.add_argument("--run-id")
    parser.add_argument("--gpu-label")
    parser.add_argument("--all-gpus")
    parser.add_argument("--launcher-use-docker")
    parser.add_argument("--container-image")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.chunk <= 0 or not 0 <= args.index < args.chunk:
        raise ValueError("index must be in [0, chunk)")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("limit must be positive")
    config_path = Path(args.config).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["budget"] = int(config["budget"])
    config.setdefault("seed", 42)
    config.setdefault("input", {})
    config["input"].setdefault("frame_token_cap", 128)
    config["input"].setdefault("processor_do_resize", True)
    config["input"].setdefault("processor_use_fast", True)
    config["input"].setdefault("metadata_include_vfr_timestamps", True)
    config.setdefault("max_new_tokens", 512)
    model_path = Path(config["model_path"])
    config["model_path"] = str(model_path if model_path.is_absolute() else (PROJECT_ROOT / model_path).resolve())
    samples_file = Path(args.samples_file).resolve() if args.samples_file else None
    annos, dataset_class = _load_annos(args.dataset, args.split)
    annos = _filter_fixed_samples(annos, samples_file)
    annos = _select_annos(annos, limit=args.limit, chunk=args.chunk, index=args.index)
    annotation_source = getattr(
        dataset_class,
        "ANNO_PATH_TEST" if args.split == "test" else "ANNO_PATH_TRAIN",
        None,
    )
    annotation_path = (TIMELENS_ROOT / annotation_source).resolve() if annotation_source else None
    video_root = getattr(dataset_class, "VIDEO_ROOT", None)
    config["execution"] = {
        "dataset": args.dataset,
        "split": args.split,
        "run_id": args.run_id,
        "all_gpus": args.all_gpus,
        "gpu_label": args.gpu_label,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "launcher_use_docker": args.launcher_use_docker,
        "container_image": args.container_image,
        "chunk_count": args.chunk,
        "chunk_index": args.index,
        "limit_total_before_sharding": args.limit,
        "samples_file": str(samples_file) if samples_file else None,
        "samples_file_sha256": _sha256_file(samples_file) if samples_file else None,
        "samples_assigned": len(annos),
        "output_dir": str(Path(args.output_dir).resolve(strict=False)),
        "decoder": "pyav",
        "visual_token_counting": "sum(image_grid_thw)/spatial_merge_size^2",
    }
    config["dataset"] = {
        "name": args.dataset,
        "split": args.split,
        "annotation_path": str(annotation_path) if annotation_path else None,
        "annotation_sha256": _sha256_file(annotation_path) if annotation_path and annotation_path.is_file() else None,
        "video_root": str((TIMELENS_ROOT / video_root).resolve()) if video_root else None,
    }
    output_dir = versioned_result_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    config["execution"]["output_dir"] = str(output_dir)
    write_provenance(output_dir, configuration=config, config_path=config_path, command=sys.argv)
    _set_random_seed(int(config["seed"]))
    model, processor = _load_model_and_processor(config)
    predictions: list[dict[str, Any]] = []
    sampling_records: list[dict[str, Any]] = []
    resources: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for position, anno in enumerate(annos):
        try:
            prediction, sampling, resource = run_one(
                anno,
                config,
                model=model,
                processor=processor,
                dataset_class=dataset_class,
            )
            predictions.append(prediction)
            sampling_records.append(sampling)
            resources.append(resource)
            print(f"[{position + 1}/{len(annos)}] {sampling['sample_id']} tokens={sampling['actual_visual_tokens']}", flush=True)
        except Exception as exc:
            failures.append({
                "sample_id": _prediction_key(anno), "video_path": anno.get("video_path"),
                "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc(),
            })
            print(f"FAILED {failures[-1]['sample_id']}: {failures[-1]['error']}", file=sys.stderr, flush=True)
    _write_jsonl(output_dir / "predictions.jsonl", predictions)
    _write_jsonl(output_dir / "sampling.jsonl", sampling_records)
    _write_jsonl(output_dir / "failures.jsonl", failures)
    fields = [
        "sample_id", "video_id", "frame_count", "actual_visual_tokens", "decode_latency_ms",
        "preprocess_latency_ms", "generate_latency_ms", "gpu_peak_memory_bytes",
    ]
    with (output_dir / "resources.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(resources)
    summary = {
        "dataset": args.dataset, "split": args.split, "chunk": args.chunk, "index": args.index,
        "samples_assigned": len(annos), "samples_succeeded": len(predictions), "samples_failed": len(failures),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "provenance_file": "provenance.json",
    }
    (output_dir / "chunk_manifest.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    # A failure must make the experimental command non-zero. There is deliberately
    # no fallback to another sampling policy.
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
