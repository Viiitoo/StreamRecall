"""Frozen CLIP ViT-B/32 encoding and compact embedding persistence."""

from __future__ import annotations

import base64
from dataclasses import asdict
from pathlib import Path
from typing import Any, Literal, Sequence

from streamtimelens.evaluation.cost import ComponentTimer


EmbeddingPrecision = Literal["fp16", "int8"]
DEFAULT_CLIP_MODEL = "openai/clip-vit-base-patch32"


def l2_normalize(values: Any) -> Any:
    import numpy as np

    array = np.asarray(values, dtype=np.float32)
    if array.ndim == 1:
        array = array[None, :]
        squeeze = True
    elif array.ndim == 2:
        squeeze = False
    else:
        raise ValueError("embeddings must be one- or two-dimensional")
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    if not np.isfinite(array).all() or (norms <= 0).any():
        raise ValueError("embeddings must contain finite non-zero vectors")
    result = array / norms
    return result[0] if squeeze else result


def serialize_embedding(values: Any, precision: EmbeddingPrecision = "fp16") -> dict[str, Any]:
    """Serialize one normalized vector without JSON float expansion."""
    import numpy as np

    vector = l2_normalize(values)
    if vector.ndim != 1:
        raise ValueError("only one embedding may be serialized per frame")
    if precision == "fp16":
        stored = vector.astype("<f2")
        scale = None
    elif precision == "int8":
        scale = float(np.max(np.abs(vector)) / 127.0)
        if scale <= 0:
            raise ValueError("cannot quantize a zero embedding")
        stored = np.clip(np.rint(vector / scale), -127, 127).astype(np.int8)
    else:
        raise ValueError(f"unsupported embedding precision: {precision}")
    return {
        "format_version": 1,
        "dtype": precision,
        "length": int(vector.shape[0]),
        "scale": scale,
        "data_b64": base64.b64encode(stored.tobytes()).decode("ascii"),
    }


def deserialize_embedding(record: dict[str, Any]) -> Any:
    import numpy as np

    if record.get("format_version") != 1:
        raise ValueError("unsupported embedding format")
    length = int(record.get("length", 0))
    if length <= 0:
        raise ValueError("embedding length must be positive")
    try:
        payload = base64.b64decode(record["data_b64"], validate=True)
    except (KeyError, ValueError) as exc:
        raise ValueError("invalid embedding payload") from exc
    if record.get("dtype") == "fp16":
        vector = np.frombuffer(payload, dtype="<f2").astype(np.float32)
    elif record.get("dtype") == "int8":
        scale = float(record.get("scale", 0))
        if not scale > 0:
            raise ValueError("int8 embedding scale must be positive")
        vector = np.frombuffer(payload, dtype=np.int8).astype(np.float32) * scale
    else:
        raise ValueError("unsupported persisted embedding dtype")
    if vector.size != length:
        raise ValueError("embedding payload length mismatch")
    return l2_normalize(vector)


class _TransformersCLIPBackend:
    def __init__(self, model_name_or_path: str | Path, device: str | None) -> None:
        try:
            import torch
            from transformers import CLIPModel, CLIPProcessor
        except ImportError as exc:  # pragma: no cover - inference environment dependent
            raise RuntimeError("CLIP inference requires torch and transformers") from exc
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = CLIPProcessor.from_pretrained(str(model_name_or_path))
        dtype = torch.float16 if self.device.startswith("cuda") else torch.float32
        self.model = CLIPModel.from_pretrained(str(model_name_or_path), torch_dtype=dtype)
        self.model.to(self.device).eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    def encode_images(self, images: Sequence[Any]) -> Any:
        inputs = self.processor(images=list(images), return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with self.torch.inference_mode():
            return self.model.get_image_features(**inputs).float().cpu().numpy()

    def encode_texts(self, texts: Sequence[str]) -> Any:
        inputs = self.processor(text=list(texts), padding=True, truncation=True, return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with self.torch.inference_mode():
            return self.model.get_text_features(**inputs).float().cpu().numpy()


class FrozenCLIPEncoder:
    """Batched visual ingestion and single-query text encoding.

    A small backend interface is accepted for deterministic CPU unit tests.
    Production construction always loads the frozen Hugging Face ViT-B/32.
    """

    def __init__(
        self,
        model_name_or_path: str | Path = DEFAULT_CLIP_MODEL,
        *,
        device: str | None = None,
        batch_size: int = 32,
        visual_fps: float = 0.5,
        persistence: EmbeddingPrecision = "fp16",
        backend: Any | None = None,
    ) -> None:
        if batch_size <= 0 or visual_fps <= 0:
            raise ValueError("CLIP batch size and visual FPS must be positive")
        if persistence not in ("fp16", "int8"):
            raise ValueError("CLIP persistence must be fp16 or int8")
        self.model_name_or_path = str(model_name_or_path)
        self.batch_size = batch_size
        self.visual_fps = visual_fps
        self.persistence = persistence
        self.backend = backend or _TransformersCLIPBackend(model_name_or_path, device)
        self.resource_records: list[Any] = []
        self._last_visual_timestamp: float | None = None

    def due(self, timestamp_s: float) -> bool:
        if timestamp_s < 0:
            raise ValueError("visual timestamp must be non-negative")
        interval = 1.0 / self.visual_fps
        if self._last_visual_timestamp is None or timestamp_s + 1e-9 >= self._last_visual_timestamp + interval:
            self._last_visual_timestamp = timestamp_s
            return True
        return False

    def reset_visual_clock(self) -> None:
        """Start a new video while reusing the same frozen model weights."""
        self._last_visual_timestamp = None

    def encode_images(self, images: Sequence[Any]) -> Any:
        import numpy as np

        if not images:
            return np.empty((0, 0), dtype=np.float32)
        chunks = []
        for start in range(0, len(images), self.batch_size):
            batch = images[start:start + self.batch_size]
            torch_module = getattr(self.backend, "torch", None)
            with ComponentTimer("clip_visual_ingest", torch_module=torch_module) as timer:
                chunks.append(l2_normalize(self.backend.encode_images(batch)))
            self.resource_records.append(timer.record)
        return l2_normalize(np.concatenate(chunks, axis=0))

    def encode_text(self, query: str) -> Any:
        if not query.strip():
            raise ValueError("CLIP query must not be empty")
        torch_module = getattr(self.backend, "torch", None)
        with ComponentTimer("clip_text_query", torch_module=torch_module) as timer:
            result = l2_normalize(self.backend.encode_texts([query.strip()]))[0]
        self.resource_records.append(timer.record)
        return result

    def persisted(self, embedding: Any) -> dict[str, Any]:
        return serialize_embedding(embedding, self.persistence)

    def resource_dicts(self) -> list[dict[str, Any]]:
        return [asdict(record) for record in self.resource_records if record is not None]
