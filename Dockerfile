# TimeLens inference environment. This PyTorch/CUDA image is already present on
# the server and supplies Python 3.11 plus the compiler required by flash-attn.
FROM docker.1ms.run/pytorch/pytorch:2.4.1-cuda12.1-cudnn9-devel

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg git build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/timelens
COPY environment/requirements.inference.txt /tmp/requirements.inference.txt

# Install the CUDA wheels through their exact index. Passing the CUDA wheel page
# as a generic ``-f`` source caused pip's dependency resolution to stall here.
# The host driver supports CUDA 12.4 wheels; flash-attn is required upstream.
RUN python -m pip install --upgrade pip \
    && python -m pip install --no-cache-dir \
        torch==2.6.0+cu124 torchvision==0.21.0+cu124 \
        --index-url https://download.pytorch.org/whl/cu124 \
    && python -m pip install --no-cache-dir -r /tmp/requirements.inference.txt \
    && python -m pip install --no-cache-dir flash-attn==2.7.4.post1 \
        --no-build-isolation

ENV PYTHONUNBUFFERED=1 \
    HF_HOME=/workspace/.cache/huggingface \
    TRANSFORMERS_CACHE=/workspace/.cache/huggingface

CMD ["bash"]
