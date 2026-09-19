"""Strict, hash-stable configuration for SnAG-adapt runs."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

from streamtimelens.baselines.snag_adapt import SnAGAdaptConfig
from streamtimelens.baselines.snag_physical import SnAGPhysicalReaderConfig
from streamtimelens.baselines.snag_training import SnAGTrainingRecipe


@dataclass(frozen=True)
class SnAGFeatureConfig:
    model: str
    upstream_id: str
    revision: str
    content_sha256: str
    feature_dim: int
    feature_fps: float
    image_batch_size: int
    normalized: bool

    def __post_init__(self) -> None:
        if not all((self.model, self.upstream_id, self.revision)):
            raise ValueError("SnAG feature model identity is incomplete")
        if len(self.content_sha256) != 64:
            raise ValueError("SnAG feature model needs a SHA-256 digest")
        if self.feature_dim < 1 or self.feature_fps <= 0 or self.image_batch_size != 1:
            raise ValueError("strict SnAG feature extraction requires batch size one")
        if not self.normalized:
            raise ValueError("SnAG v1 recipe requires normalized CLIP embeddings")


@dataclass(frozen=True)
class ResolvedSnAGConfig:
    schema_version: int
    method: str
    classification: str
    writer: SnAGAdaptConfig
    feature: SnAGFeatureConfig
    reader: SnAGPhysicalReaderConfig
    training: SnAGTrainingRecipe | None
    budget_bytes: int | None
    checkpoint: str | None
    checkpoint_sha256: str | None
    raw: Mapping[str, Any]

    @property
    def sha256(self) -> str:
        payload = json.dumps(self.raw, sort_keys=True, separators=(",", ":"), allow_nan=False)
        return hashlib.sha256(payload.encode()).hexdigest()


def load_snag_config(path: Path | str) -> ResolvedSnAGConfig:
    source = Path(path).expanduser().resolve(strict=True)
    raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping) or raw.get("schema_version") != 1:
        raise ValueError("unsupported SnAG config")
    writer_raw = dict(raw["writer"])
    budget = raw.get("budget_bytes")
    writer = SnAGAdaptConfig(
        mode=writer_raw["mode"], storage_dtype=writer_raw["storage_dtype"],
        budget_bytes=int(budget) if budget is not None else None,
        near_capacity=int(writer_raw.get("near_capacity", 64)),
        far_capacity=int(writer_raw.get("far_capacity", 192)),
        max_source_ids=int(writer_raw.get("max_source_ids", 3)),
    )
    feature = SnAGFeatureConfig(**raw["feature"])
    reader_raw = dict(raw["reader"])
    checkpoint = reader_raw.pop("checkpoint", None)
    checkpoint_sha256 = reader_raw.pop("checkpoint_sha256", None)
    reader_raw.pop("offset_normalization", None)
    reader = SnAGPhysicalReaderConfig(**reader_raw)
    if reader.feature_dim != feature.feature_dim:
        raise ValueError("SnAG writer/reader feature dimensions disagree")
    training_raw = dict(raw["training"]) if "training" in raw else None
    if training_raw is not None and "arrival_ratios" in training_raw:
        training_raw["arrival_ratios"] = tuple(float(value) for value in training_raw["arrival_ratios"])
    training = SnAGTrainingRecipe(**training_raw) if training_raw is not None else None
    if raw["method"] == "snag-adapt-pooled-B" and training is None:
        raise ValueError("pooled SnAG config requires a training recipe")
    if checkpoint_sha256 is not None and len(str(checkpoint_sha256)) != 64:
        raise ValueError("invalid SnAG reader checkpoint digest")
    return ResolvedSnAGConfig(
        1, str(raw["method"]), str(raw["classification"]), writer, feature,
        reader, training, int(budget) if budget is not None else None,
        str(checkpoint) if checkpoint else None,
        str(checkpoint_sha256) if checkpoint_sha256 else None,
        raw,
    )


def resolved_snag_config_dict(config: ResolvedSnAGConfig) -> dict[str, Any]:
    return {
        "schema_version": config.schema_version,
        "method": config.method,
        "classification": config.classification,
        "writer": asdict(config.writer),
        "feature": asdict(config.feature),
        "reader": asdict(config.reader),
        "training": asdict(config.training) if config.training else None,
        "budget_bytes": config.budget_bytes,
        "checkpoint": config.checkpoint,
        "checkpoint_sha256": config.checkpoint_sha256,
        "config_sha256": config.sha256,
    }
