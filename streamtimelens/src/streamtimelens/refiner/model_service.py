"""Process-wide, immutable service for the frozen TimeLens checkpoint."""

from __future__ import annotations

import hashlib
import json
import threading
import time
from pathlib import Path
from typing import Any, Sequence


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hash_group(model_path: Path, names: Sequence[str]) -> str:
    digest = hashlib.sha256()
    found = False
    for name in sorted(names):
        path = model_path / name
        if path.is_file():
            found = True
            digest.update(name.encode("utf-8") + b"\0" + bytes.fromhex(_sha256_file(path)))
    if not found:
        raise FileNotFoundError(f"none of the hash inputs exist under {model_path}: {names}")
    return digest.hexdigest()


class TimeLensModelService:
    """Load TimeLens once per resolved checkpoint path and expose greedy generation.

    Heavy dependencies are imported only when ``get`` first constructs a service,
    keeping CPU-only unit tests and protocol workers free of model side effects.
    """

    _instances: dict[str, "TimeLensModelService"] = {}
    _instances_lock = threading.Lock()

    @classmethod
    def get(cls, model_path: str | Path, *, device_map: str = "auto") -> "TimeLensModelService":
        key = str(Path(model_path).expanduser().resolve())
        with cls._instances_lock:
            if key not in cls._instances:
                cls._instances[key] = cls(Path(key), device_map=device_map)
            return cls._instances[key]

    def __init__(self, model_path: Path, *, device_map: str) -> None:
        if not model_path.is_dir():
            raise FileNotFoundError(model_path)
        try:
            import torch
            from transformers import AutoModelForImageTextToText, AutoProcessor
        except ImportError as exc:
            raise RuntimeError("TimeLens inference dependencies are unavailable") from exc

        self.model_path = model_path
        self.device_map = device_map
        self.torch = torch
        self.hashes = self._compute_hashes(model_path)
        self.processor = AutoProcessor.from_pretrained(
            str(model_path), padding_side="left", do_resize=False, use_fast=False,
            trust_remote_code=True,
        )
        self.model = AutoModelForImageTextToText.from_pretrained(
            str(model_path), dtype=torch.bfloat16, attn_implementation="flash_attention_2", device_map=device_map,
        ).eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self._lock = threading.Lock()
        self.last_call_stats: dict[str, Any] = {}

    @staticmethod
    def _compute_hashes(model_path: Path) -> dict[str, str]:
        index_path = model_path / "model.safetensors.index.json"
        model_files: list[str]
        if index_path.is_file():
            index = json.loads(index_path.read_text(encoding="utf-8"))
            model_files = ["model.safetensors.index.json"] + sorted(set(index.get("weight_map", {}).values()))
        else:
            model_files = [path.name for path in model_path.glob("*.safetensors")]
        return {
            "model_sha256": _hash_group(model_path, model_files),
            "config_sha256": _hash_group(model_path, [
                "chat_template.json", "config.json", "generation_config.json",
                "preprocessor_config.json", "processing_timelens.py",
            ]),
            "tokenizer_sha256": _hash_group(model_path, [
                "added_tokens.json", "merges.txt", "special_tokens_map.json",
                "tokenizer.json", "tokenizer_config.json", "vocab.json",
            ]),
        }

    def generate(self, messages: Sequence[dict[str, Any]], videos: Sequence[Any], max_new_tokens: int) -> str:
        """Generate greedily; callers cannot override frozen inference parameters."""
        if not messages or not videos or max_new_tokens <= 0:
            raise ValueError("messages, videos, and a positive token limit are required")
        import torch

        with self._lock, torch.inference_mode():
            self.last_call_stats = {}
            wall_start = time.perf_counter()
            gpu_start = gpu_end = None
            if torch.cuda.is_available():
                for device_index in range(torch.cuda.device_count()):
                    torch.cuda.reset_peak_memory_stats(device_index)
                torch.cuda.synchronize()
                gpu_start = torch.cuda.Event(enable_timing=True)
                gpu_end = torch.cuda.Event(enable_timing=True)
                gpu_start.record()
            text = self.processor.apply_chat_template(list(messages), tokenize=False, add_generation_prompt=True)
            inputs = self.processor(text=[text], images=None, videos=list(videos), padding=True, return_tensors="pt")
            target = "cuda" if torch.cuda.is_available() else next(self.model.parameters()).device
            inputs = inputs.to(target)
            output_ids = self.model.generate(
                **inputs, do_sample=False, temperature=None, top_p=None, top_k=None,
                max_new_tokens=int(max_new_tokens),
            )
            prompt_length = int(inputs.input_ids.shape[1])
            answer = self.processor.batch_decode(
                [output_ids[0][prompt_length:]], skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]
            gpu_time_s = None
            if gpu_start is not None and gpu_end is not None:
                gpu_end.record()
                torch.cuda.synchronize()
                gpu_time_s = gpu_start.elapsed_time(gpu_end) / 1000.0
            peaks = ([int(torch.cuda.max_memory_allocated(index)) for index in range(torch.cuda.device_count())]
                     if torch.cuda.is_available() else [])
            self.last_call_stats = {
                "prompt_tokens": prompt_length,
                "generated_tokens": int(output_ids.shape[1]) - prompt_length,
                "peak_cuda_bytes": sum(peaks),
                "peak_cuda_bytes_by_device": peaks,
                "wall_time_s": time.perf_counter() - wall_start,
                "gpu_time_s": gpu_time_s,
            }
            return answer
