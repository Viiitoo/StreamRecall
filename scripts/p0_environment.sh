#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
image_tag="${TIMELENS_IMAGE:-vlm-timelens:0.1}"
mkdir -p "${project_root}/logs" "${project_root}/.cache/huggingface"

docker run --rm --gpus all \
  -v "${project_root}:/workspace" \
  -w /workspace/third_party/TimeLens \
  "${image_tag}" \
  bash -lc '
    set -euo pipefail
    python --version
    python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))"
    python -c "import transformers; print(transformers.__version__)"
    python -c "import flash_attn; print(flash_attn.__version__)"
    python -c "import importlib.metadata as m, qwen_vl_utils, decord, av; print(m.version('qwen-vl-utils')); print('qwen_vl_utils/decord/av imports: ok')"
    nvidia-smi
    ffmpeg -version | head -n 1
  ' | tee "${project_root}/logs/p0_environment.txt"
