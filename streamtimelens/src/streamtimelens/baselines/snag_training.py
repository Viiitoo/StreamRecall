"""Leakage-safe training utilities for the pooled physical-time reader."""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np

from streamtimelens.baselines.snag_adapt import SnAGSnapshotReader
from streamtimelens.baselines.snag_physical import (
    SnAGPhysicalReaderConfig,
    SnAGPhysicalTimeModel,
    item_metadata,
    physical_time_loss,
)
from streamtimelens.protocol.arrival import ArrivalRecord


@dataclass(frozen=True)
class SnAGTrainingRecipe:
    seed: int = 20260914
    epochs: int = 20
    batch_size: int = 16
    learning_rate: float = 3e-4
    weight_decay: float = 0.05
    validation_fraction: float = 0.2
    gradient_clip_norm: float = 1.0
    arrival_ratios: tuple[float, ...] = (0.25, 0.5, 0.75, 1.0)
    minimum_delay_s: float = 0.0

    def __post_init__(self) -> None:
        if min(self.epochs, self.batch_size) < 1 or self.learning_rate <= 0:
            raise ValueError("invalid SnAG training schedule")
        if self.weight_decay < 0 or not 0 < self.validation_fraction < 1:
            raise ValueError("invalid SnAG regularization/split")
        if self.gradient_clip_norm <= 0 or self.minimum_delay_s < 0:
            raise ValueError("invalid gradient clip or minimum delay")


@dataclass(frozen=True)
class SnAGTrainingExample:
    video_id: str
    query_id: str
    query: str
    gt_span: tuple[float, float]
    t_q: float
    snapshot_path: str
    query_embedding: np.ndarray

    def __post_init__(self) -> None:
        embedding = np.asarray(self.query_embedding, dtype=np.float32)
        if embedding.ndim != 1 or not embedding.size or not np.isfinite(embedding).all():
            raise ValueError("training query embedding must be finite")
        if self.gt_span[1] > self.t_q + 1e-9:
            raise ValueError("training target is not a past event at query time")
        object.__setattr__(self, "query_embedding", embedding.copy())


def join_development_manifests(
    videos: Sequence[dict[str, object]],
    queries: Sequence[dict[str, object]],
    annotations: Sequence[dict[str, object]],
) -> list[dict[str, object]]:
    """Join split P5 manifests into the GT-bearing reader-training format."""
    video_index: dict[str, dict[str, object]] = {}
    for row in videos:
        video_id = str(row["video_id"])
        if video_id in video_index:
            raise ValueError(f"duplicate video id: {video_id}")
        duration = float(row["duration_s"])
        if duration <= 0:
            raise ValueError(f"invalid duration for video: {video_id}")
        video_index[video_id] = row
    query_index: dict[str, dict[str, object]] = {}
    for row in queries:
        query_id = str(row["query_id"])
        if query_id in query_index:
            raise ValueError(f"duplicate query id: {query_id}")
        query_index[query_id] = row
    annotation_index: dict[str, dict[str, object]] = {}
    for row in annotations:
        query_id = str(row["query_id"])
        if query_id in annotation_index:
            raise ValueError(f"duplicate annotation query id: {query_id}")
        annotation_index[query_id] = row
    if set(query_index) != set(annotation_index):
        raise ValueError("query and annotation IDs do not match exactly")
    output = []
    for query_id in sorted(query_index):
        query = query_index[query_id]
        annotation = annotation_index[query_id]
        video_id = str(query["video_id"])
        if video_id != str(annotation["video_id"]):
            raise ValueError(f"query/annotation video mismatch: {query_id}")
        if video_id not in video_index:
            raise ValueError(f"query references missing video: {video_id}")
        duration = float(video_index[video_id]["duration_s"])
        start, end = map(float, annotation["gt_span"])
        if not 0 <= start < end <= duration + 1e-6:
            raise ValueError(f"invalid ground-truth span: {query_id}")
        output.append({
            "video_id": video_id,
            "query_id": query_id,
            "query": str(query["query"]),
            "gt_span": [start, min(end, duration)],
            "duration": duration,
        })
    return output


def build_training_examples(
    arrivals: Iterable[ArrivalRecord],
    snapshot_path: Callable[[ArrivalRecord], Path | str],
    encode_query: Callable[[str], np.ndarray],
    *,
    minimum_delay_s: float = 0.0,
) -> list[SnAGTrainingExample]:
    """Join query/GT only after opening an already-frozen query-blind state."""
    examples = []
    embeddings: dict[str, np.ndarray] = {}
    for row in arrivals:
        if row.cohort != "natural" or not row.eligible:
            continue
        if row.t_q - row.gt_span[1] + 1e-9 < minimum_delay_s:
            continue
        path = Path(snapshot_path(row)).expanduser().resolve(strict=True)
        snapshot = SnAGSnapshotReader(path)
        if snapshot.manifest.video_id != row.video_id or abs(snapshot.manifest.t_q - row.t_q) > 1e-6:
            raise ValueError("training snapshot does not match frozen arrival")
        if any(item.t_end_s > row.t_q + 1e-9 for item in snapshot.items):
            raise ValueError("training snapshot contains future evidence")
        embedding = embeddings.get(row.query_id)
        if embedding is None:
            embedding = np.asarray(encode_query(row.query), dtype=np.float32)
            embeddings[row.query_id] = embedding
        examples.append(SnAGTrainingExample(
            row.video_id, row.query_id, row.query, row.gt_span, row.t_q,
            str(path), embedding,
        ))
    if not examples:
        raise ValueError("no eligible SnAG pooled-reader training examples")
    return examples


def split_examples_by_video(
    examples: Sequence[SnAGTrainingExample], *, seed: int, validation_fraction: float,
) -> tuple[list[SnAGTrainingExample], list[SnAGTrainingExample]]:
    videos = sorted({row.video_id for row in examples})
    ranked = sorted(
        videos,
        key=lambda value: hashlib.sha256(f"{seed}\0{value}".encode()).hexdigest(),
    )
    validation_count = max(1, min(len(ranked) - 1, round(len(ranked) * validation_fraction)))
    validation_videos = set(ranked[:validation_count])
    train = [row for row in examples if row.video_id not in validation_videos]
    validation = [row for row in examples if row.video_id in validation_videos]
    if not train or not validation:
        raise ValueError("training requires at least two videos")
    return train, validation


def _collate(rows: Sequence[SnAGTrainingExample], device: str) -> tuple[Any, ...]:
    import torch

    snapshots = [SnAGSnapshotReader(row.snapshot_path) for row in rows]
    maximum = max(len(snapshot.items) for snapshot in snapshots)
    dimension = snapshots[0].manifest.feature_dim
    features = torch.zeros((len(rows), maximum, dimension), dtype=torch.float32)
    metadata = torch.zeros((len(rows), maximum, 6), dtype=torch.float32)
    masks = torch.zeros((len(rows), maximum), dtype=torch.bool)
    for index, snapshot in enumerate(snapshots):
        length = len(snapshot.items)
        if snapshot.manifest.feature_dim != dimension or length < 1:
            raise ValueError("training snapshots must be non-empty with one feature dimension")
        features[index, :length] = torch.from_numpy(snapshot.features().copy())
        metadata[index, :length] = torch.from_numpy(item_metadata(snapshot.items))
        masks[index, :length] = True
    query = torch.from_numpy(np.stack([row.query_embedding for row in rows]))
    t_q = torch.tensor([row.t_q for row in rows], dtype=torch.float32)
    gt = torch.tensor([row.gt_span for row in rows], dtype=torch.float32)
    return tuple(value.to(device) for value in (features, metadata, masks, query, t_q, gt))


def _mean_loss(model: Any, rows: Sequence[SnAGTrainingExample], batch_size: int, device: str) -> float:
    import torch

    model.eval()
    values = []
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            features, metadata, masks, query, t_q, gt = _collate(rows[start:start + batch_size], device)
            logits, offsets, levels = model(features, metadata, masks, query, t_q)
            loss, _ = physical_time_loss(logits, offsets, levels, masks, gt)
            values.append(float(loss.cpu()))
    return sum(values) / len(values)


def train_physical_reader(
    examples: Sequence[SnAGTrainingExample],
    model_config: SnAGPhysicalReaderConfig,
    recipe: SnAGTrainingRecipe,
    *,
    device: str = "cuda",
) -> tuple[Any, dict[str, object]]:
    import torch

    random.seed(recipe.seed)
    np.random.seed(recipe.seed)
    torch.manual_seed(recipe.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(recipe.seed)
    train, validation = split_examples_by_video(
        examples, seed=recipe.seed, validation_fraction=recipe.validation_fraction,
    )
    model = SnAGPhysicalTimeModel.build(model_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=recipe.learning_rate, weight_decay=recipe.weight_decay,
    )
    generator = random.Random(recipe.seed)
    history = []
    best_state = None
    best_validation = float("inf")
    for epoch in range(recipe.epochs):
        model.train()
        shuffled = list(train)
        generator.shuffle(shuffled)
        training_losses = []
        for start in range(0, len(shuffled), recipe.batch_size):
            batch = shuffled[start:start + recipe.batch_size]
            features, metadata, masks, query, t_q, gt = _collate(batch, device)
            optimizer.zero_grad(set_to_none=True)
            logits, offsets, levels = model(features, metadata, masks, query, t_q)
            loss, _ = physical_time_loss(logits, offsets, levels, masks, gt)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), recipe.gradient_clip_norm)
            optimizer.step()
            training_losses.append(float(loss.detach().cpu()))
        validation_loss = _mean_loss(model, validation, recipe.batch_size, device)
        history.append({
            "epoch": epoch + 1,
            "training_loss": sum(training_losses) / len(training_losses),
            "validation_loss": validation_loss,
        })
        if validation_loss < best_validation:
            best_validation = validation_loss
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    assert best_state is not None
    model.load_state_dict(best_state)
    model.to(device).eval()
    report = {
        "schema_version": 1,
        "model_config": asdict(model_config),
        "recipe": asdict(recipe),
        "train_video_count": len({row.video_id for row in train}),
        "validation_video_count": len({row.video_id for row in validation}),
        "train_example_count": len(train),
        "validation_example_count": len(validation),
        "best_validation_loss": best_validation,
        "history": history,
    }
    return model, report


def save_physical_reader_checkpoint(
    path: Path | str,
    model: Any,
    model_config: SnAGPhysicalReaderConfig,
    recipe: SnAGTrainingRecipe,
    report: dict[str, object],
    *,
    feature_model: dict[str, str],
) -> str:
    import torch

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": 1,
        "architecture": model_config.architecture_name,
        "model_config": asdict(model_config),
        "training_recipe": asdict(recipe),
        "feature_model": dict(feature_model),
        "training_report": report,
        "model": {name: value.detach().cpu() for name, value in model.state_dict().items()},
    }
    torch.save(payload, destination)
    return hashlib.sha256(destination.read_bytes()).hexdigest()


def load_physical_reader_checkpoint(path: Path | str, *, device: str = "cpu") -> tuple[Any, dict[str, Any]]:
    import torch

    source = Path(path).expanduser().resolve(strict=True)
    payload = torch.load(source, map_location="cpu")
    if payload.get("format_version") != 1 or "model" not in payload:
        raise ValueError("unsupported SnAG physical reader checkpoint")
    config = SnAGPhysicalReaderConfig(**payload["model_config"])
    if payload.get("architecture") != config.architecture_name:
        raise ValueError("SnAG checkpoint architecture mismatch")
    model = SnAGPhysicalTimeModel.build(config)
    model.load_state_dict(payload["model"], strict=True)
    return model.to(device).eval(), payload


def training_manifest_sha256(examples: Sequence[SnAGTrainingExample]) -> str:
    rows = [{
        "video_id": row.video_id,
        "query_id": row.query_id,
        "gt_span": row.gt_span,
        "t_q": row.t_q,
        "snapshot_path": row.snapshot_path,
        "query_embedding_sha256": hashlib.sha256(row.query_embedding.tobytes()).hexdigest(),
    } for row in examples]
    payload = json.dumps(rows, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()
