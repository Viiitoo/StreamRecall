#!/usr/bin/env bash
# Run an auditable P0 uniform baseline.  Outputs are never written into official
# TimeLens logs and a failed decode/token audit makes this command fail.
set -euo pipefail

usage() {
  echo "Usage: $0 --dataset NAME --split NAME --gpus 0,1 --run-id ID --budget {2048|4096|8192|14336} [--limit N] [--samples-file FILE]" >&2
}

dataset=""
split="test"
gpus=""
run_id=""
budget=""
limit=""
samples_file=""
original_args=("$@")
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset) dataset="$2"; shift 2 ;;
    --split) split="$2"; shift 2 ;;
    --gpus) gpus="$2"; shift 2 ;;
    --run-id) run_id="$2"; shift 2 ;;
    --budget) budget="$2"; shift 2 ;;
    --limit) limit="$2"; shift 2 ;;
    --samples-file) samples_file="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 2 ;;
  esac
done
[[ -n "$dataset" && -n "$gpus" && -n "$run_id" && -n "$budget" ]] || { usage; exit 2; }
case "$budget" in 2048|4096|8192|14336) ;; *) echo "Unsupported budget: $budget" >&2; exit 2;; esac
[[ "$run_id" != *"/"* && "$run_id" != "." && "$run_id" != ".." ]] || { echo "run-id must be a directory name" >&2; exit 2; }
[[ -z "$limit" || "$limit" =~ ^[1-9][0-9]*$ ]] || { echo "limit must be a positive integer" >&2; exit 2; }

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
config_rel="configs/baas/uniform_${budget}.yaml"
config_path="${project_root}/${config_rel}"
if [[ -n "$samples_file" ]]; then
  samples_file="$(realpath "$samples_file")"
  [[ -f "$samples_file" ]] || { echo "Samples file does not exist: $samples_file" >&2; exit 2; }
  [[ "$samples_file" == "$project_root"/* ]] || { echo "Samples file must be inside the project for container runs" >&2; exit 2; }
fi
configured_model_path="$(python3 -c 'import sys, yaml; print(yaml.safe_load(open(sys.argv[1], encoding="utf-8"))["model_path"])' "$config_path")"
if [[ "$configured_model_path" = /* ]]; then
  resolved_model_path="$configured_model_path"
else
  resolved_model_path="${project_root}/${configured_model_path}"
fi
IFS=',' read -r -a gpu_list <<< "$gpus"
[[ ${#gpu_list[@]} -gt 0 ]] || { echo "No GPUs specified" >&2; exit 2; }
code_version="$(PYTHONPATH="${project_root}/src" python3 -m baas.provenance --print-code-version)"
output_dir="${project_root}/results/${code_version}/baas/uniform/${dataset}/${budget}/${run_id}"
[[ ! -e "$output_dir" ]] || { echo "Refusing to overwrite existing run: $output_dir" >&2; exit 2; }
mkdir -p "$output_dir"
use_docker="${BAAS_USE_DOCKER:-1}"
timelens_image="${TIMELENS_IMAGE:-vlm-timelens:0.1}"
container_image_id=""
if [[ "$use_docker" == "1" ]]; then
  container_image_id="$(docker image inspect "$timelens_image" --format '{{.Id}}' 2>/dev/null || true)"
fi
annotation_rel=""
video_root_rel=""
case "$dataset" in
  charades-timelens)
    annotation_rel="third_party/TimeLens/data/TimeLens-Bench/charades-timelens.json"
    video_root_rel="third_party/TimeLens/data/TimeLens-Bench/videos/charades" ;;
  qvhighlights-timelens)
    annotation_rel="third_party/TimeLens/data/TimeLens-Bench/qvhighlights-timelens.json"
    video_root_rel="third_party/TimeLens/data/TimeLens-Bench/videos/qvhighlights" ;;
  activitynet-timelens)
    annotation_rel="third_party/TimeLens/data/TimeLens-Bench/activitynet-timelens.json"
    video_root_rel="third_party/TimeLens/data/TimeLens-Bench/videos/activitynet" ;;
  timelens-100k)
    annotation_rel="third_party/TimeLens/data/TimeLens-100K/timelens-100k.jsonl"
    video_root_rel="third_party/TimeLens/data/TimeLens-100K/videos" ;;
esac
annotation_sha256=""
[[ -z "$annotation_rel" || ! -f "${project_root}/${annotation_rel}" ]] || annotation_sha256="$(sha256sum "${project_root}/${annotation_rel}" | awk '{print $1}')"
samples_sha256=""
[[ -z "$samples_file" ]] || samples_sha256="$(sha256sum "$samples_file" | awk '{print $1}')"
PYTHONPATH="${project_root}/src" python3 -m baas.provenance \
  --output-dir "$output_dir" --config "$config_path" \
  --override-string "execution.dataset=${dataset}" \
  --override-string "execution.split=${split}" \
  --override-string "execution.gpus=${gpus}" \
  --override-string "execution.run_id=${run_id}" \
  --override "execution.limit=${limit:-null}" \
  --override-string "execution.samples_file=${samples_file}" \
  --override-string "execution.samples_file_sha256=${samples_sha256}" \
  --override "execution.chunk_count=${#gpu_list[@]}" \
  --override "execution.use_docker=${use_docker}" \
  --override-string "execution.container_image=${timelens_image}" \
  --override-string "execution.container_image_id=${container_image_id}" \
  --override-string "paths.project_root=${project_root}" \
  --override-string "paths.output_dir=${output_dir}" \
  --override-string "paths.model_path_resolved=${resolved_model_path}" \
  --override-string "paths.timelens_root=${project_root}/third_party/TimeLens" \
  --override-string "dataset.name=${dataset}" \
  --override-string "dataset.split=${split}" \
  --override-string "dataset.annotation_path=${project_root}/${annotation_rel}" \
  --override-string "dataset.annotation_sha256=${annotation_sha256}" \
  --override-string "dataset.video_root=${project_root}/${video_root_rel}" \
  -- "${BASH_SOURCE[0]}" "${original_args[@]}"

run_chunk() {
  local gpu="$1" index="$2" chunk_dir="$3"
  if [[ "${BAAS_USE_DOCKER:-1}" == "1" ]]; then
    local container_chunk_dir="/workspace${chunk_dir#"$project_root"}"
    local common=(python3 -m baas.runner --config "/workspace/${config_rel}" --dataset "$dataset" --split "$split" --output-dir "$container_chunk_dir" --chunk "${#gpu_list[@]}" --index "$index" --run-id "$run_id" --gpu-label "$gpu" --all-gpus "$gpus" --launcher-use-docker "$use_docker" --container-image "$timelens_image")
    [[ -z "$limit" ]] || common+=(--limit "$limit")
    [[ -z "$samples_file" ]] || common+=(--samples-file "/workspace${samples_file#"$project_root"}")
    docker run --rm --gpus "device=${gpu}" \
      -v "${project_root}:/workspace" -w /workspace \
      -e PYTHONPATH=/workspace/src:/workspace/third_party/TimeLens \
      -e CUDA_VISIBLE_DEVICES=0 "${TIMELENS_IMAGE:-vlm-timelens:0.1}" "${common[@]}"
  else
    local local_common=(python3 -m baas.runner --config "$config_path" --dataset "$dataset" --split "$split" --output-dir "$chunk_dir" --chunk "${#gpu_list[@]}" --index "$index" --run-id "$run_id" --gpu-label "$gpu" --all-gpus "$gpus" --launcher-use-docker "$use_docker" --container-image "$timelens_image")
    [[ -z "$limit" ]] || local_common+=(--limit "$limit")
    [[ -z "$samples_file" ]] || local_common+=(--samples-file "$samples_file")
    (
      cd "$project_root"
      CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH="${project_root}/src:${project_root}/third_party/TimeLens${PYTHONPATH:+:${PYTHONPATH}}" "${local_common[@]}"
    )
  fi
}

pids=()
for index in "${!gpu_list[@]}"; do
  chunk_dir="${output_dir}/chunks/${index}"
  mkdir -p "${output_dir}/chunks"
  run_chunk "${gpu_list[$index]}" "$index" "$chunk_dir" >"${output_dir}/chunk_${index}.stdout.log" 2>"${output_dir}/chunk_${index}.stderr.log" &
  pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do
  wait "$pid" || failed=1
done

for filename in predictions.jsonl sampling.jsonl failures.jsonl; do
  : > "${output_dir}/${filename}"
  for index in "${!gpu_list[@]}"; do
    chunk_file="${output_dir}/chunks/${index}/${filename}"
    [[ ! -f "$chunk_file" ]] || cat "$chunk_file" >> "${output_dir}/${filename}"
  done
done
echo 'sample_id,video_id,frame_count,actual_visual_tokens,decode_latency_ms,preprocess_latency_ms,generate_latency_ms,gpu_peak_memory_bytes' > "${output_dir}/resources.csv"
for index in "${!gpu_list[@]}"; do
  chunk_resources="${output_dir}/chunks/${index}/resources.csv"
  [[ ! -f "$chunk_resources" ]] || tail -n +2 "$chunk_resources" >> "${output_dir}/resources.csv"
done

python3 - "$output_dir" "$dataset" "$split" "$budget" "$run_id" "$gpus" "$code_version" "$failed" <<'PY'
import csv, json, math, sys
from pathlib import Path
out, dataset, split, budget, run_id, gpus, code_version, process_failed = map(str, sys.argv[1:])
rows = list(csv.DictReader((Path(out) / "resources.csv").open()))
tokens = sorted(int(row["actual_visual_tokens"]) for row in rows)
def percentile(values, q):
    if not values: return None
    return values[max(0, math.ceil(len(values) * q) - 1)]
failure_count = sum(1 for line in (Path(out) / "failures.jsonl").open() if line.strip())
manifest = {"method": "uniform", "dataset": dataset, "split": split, "budget": int(budget), "run_id": run_id,
            "gpus": gpus, "sample_count": len(rows), "failure_count": failure_count,
            "process_failed": bool(int(process_failed)),
            "code_version": code_version, "provenance_file": "provenance.json",
            "actual_visual_tokens": {"mean": sum(tokens) / len(tokens) if tokens else None,
                                     "p95": percentile(tokens, .95), "max": max(tokens) if tokens else None}}
(Path(out) / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
if any(value > int(budget) for value in tokens): raise SystemExit("token audit failed")
PY
if (( failed )); then
  echo "One or more chunks failed; merged audit files are in ${output_dir}" >&2
  exit 1
fi

PYTHONPATH="${project_root}/third_party/TimeLens" python3 "${project_root}/third_party/TimeLens/evaluation/compute_metrics.py" \
  -f "${output_dir}/predictions.jsonl" | tee "${output_dir}/metrics.log"
echo "Completed: $output_dir"
