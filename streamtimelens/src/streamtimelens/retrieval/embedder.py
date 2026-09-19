"""Frozen text embeddings for evidence cards and later query retrieval."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Sequence

from streamtimelens.evaluation.cost import ComponentTimer
from streamtimelens.protocol.snapshot import SnapshotReader
from streamtimelens.memory.cards import EvidenceCard
from streamtimelens.observer.clip_encoder import l2_normalize, serialize_embedding


DEFAULT_TEXT_EMBEDDER = "sentence-transformers/all-MiniLM-L6-v2"
QWEN3_TEXT_EMBEDDER = "Qwen/Qwen3-Embedding-0.6B"
EMBEDDING_TEMPLATE_VERSION = "evidence_text_v1"
_CARD_TEMPLATE = "Represent this video evidence for retrieval: {text}"
_QUERY_TEMPLATE = "Represent this video evidence request for retrieval: {text}"
EMBEDDING_TEMPLATE_SHA256 = hashlib.sha256(
    (_CARD_TEMPLATE + "\n" + _QUERY_TEMPLATE).encode("utf-8")
).hexdigest()


def card_embedding_text(card: EvidenceCard) -> str:
    text = card.normalized_text.strip() or (
        f"summary: {card.summary} | actors: {', '.join(card.actors) or 'none'} | "
        f"actions: {', '.join(card.actions) or 'none'} | "
        f"objects: {', '.join(card.objects) or 'none'} | scene: {card.scene}"
    )
    return _CARD_TEMPLATE.format(text=text)


def query_embedding_text(text: str) -> str:
    normalized = " ".join(text.split())
    if not normalized:
        raise ValueError("embedding request text must not be empty")
    return _QUERY_TEMPLATE.format(text=normalized)


class _TransformersTextBackend:
    def __init__(self, model_name_or_path: str, revision: str | None, device: str | None) -> None:
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - inference environment dependent
            raise RuntimeError("text embedding requires torch and transformers") from exc
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, revision=revision, trust_remote_code=True,
        )
        self.model = AutoModel.from_pretrained(
            model_name_or_path, revision=revision, trust_remote_code=True,
        ).to(self.device).eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    def encode(self, texts: Sequence[str]) -> Any:
        inputs = self.tokenizer(
            list(texts), padding=True, truncation=True, max_length=512, return_tensors="pt",
        )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with self.torch.inference_mode():
            hidden = self.model(**inputs).last_hidden_state
            mask = inputs["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
        return pooled.float().cpu().numpy()


@dataclass(frozen=True)
class TextEmbeddingMetadata:
    model_name: str
    revision: str
    template_version: str = EMBEDDING_TEMPLATE_VERSION
    template_sha256: str = EMBEDDING_TEMPLATE_SHA256
    dtype: str = "fp16"


class CardTextEmbedder:
    """L2-normalized fp16 persistence with an injectable deterministic backend."""

    def __init__(
        self,
        model_name_or_path: str = DEFAULT_TEXT_EMBEDDER,
        *,
        revision: str | None = None,
        device: str | None = None,
        backend: Any | None = None,
    ) -> None:
        if not model_name_or_path:
            raise ValueError("text embedding model is required")
        self.model_name_or_path = str(model_name_or_path)
        self.revision = revision or "unspecified"
        self.backend = backend or _TransformersTextBackend(model_name_or_path, revision, device)
        self.metadata = TextEmbeddingMetadata(self.model_name_or_path, self.revision)

    def _encode(self, texts: Sequence[str]) -> Any:
        if not texts:
            raise ValueError("at least one embedding input is required")
        return l2_normalize(self.backend.encode(list(texts)))

    def embed_cards(self, cards: Sequence[EvidenceCard]) -> Any:
        if not cards:
            return []
        vectors = self._encode([card_embedding_text(card) for card in cards])
        metadata = self.metadata.__dict__.copy()
        for card, vector in zip(cards, vectors):
            card.text_embedding = serialize_embedding(vector, "fp16")
            card.writer_provenance["text_embedding"] = metadata
            card.serializable()
        return vectors

    def embed_request(self, text: str) -> Any:
        return self._encode([query_embedding_text(text)])[0]

    # Compatibility name used by retrieval callers while preserving the same
    # template family as card ingestion.
    def encode_query(self, text: str) -> Any:
        return self.embed_request(text)


@dataclass(frozen=True)
class QueryEmbeddingResult:
    vector: Any
    resource: dict[str, Any]


def load_snapshot_query_embedder(
    snapshot: SnapshotReader,
    *,
    model_name_or_path: str | None = None,
    revision: str | None = None,
    device: str | None = None,
    backend: Any | None = None,
) -> CardTextEmbedder:
    """Load exactly the card embedder declared by the immutable snapshot."""
    declared = snapshot.manifest.embedder
    if not isinstance(declared, dict):
        raise ValueError("snapshot does not declare a text embedder")
    required = {
        "model_name", "revision", "template_version", "template_sha256", "dtype",
    }
    if set(declared) != required:
        raise ValueError("snapshot text embedder metadata is incomplete")
    if declared["template_version"] != EMBEDDING_TEMPLATE_VERSION or (
        declared["template_sha256"] != EMBEDDING_TEMPLATE_SHA256
    ):
        raise ValueError("snapshot uses an incompatible text embedding template")
    requested_model = model_name_or_path or str(declared["model_name"])
    requested_revision = revision or str(declared["revision"])
    if requested_model != declared["model_name"] or requested_revision != declared["revision"]:
        raise ValueError("query embedder must exactly match snapshot model and revision")
    embedder = CardTextEmbedder(
        requested_model, revision=requested_revision, device=device, backend=backend,
    )
    if vars(embedder.metadata) != declared:
        raise ValueError("loaded query embedder metadata does not match snapshot")
    return embedder


def timed_query_embedding(embedder: CardTextEmbedder, query: str) -> QueryEmbeddingResult:
    torch_module = getattr(getattr(embedder, "backend", None), "torch", None)
    with ComponentTimer("query_embedding", torch_module=torch_module) as timer:
        vector = embedder.encode_query(query)
    return QueryEmbeddingResult(vector=vector, resource=timer.as_dict())
