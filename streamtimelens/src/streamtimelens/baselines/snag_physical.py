"""Learned physical-time late-fusion reader for pooled SnAG snapshots."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Sequence

import numpy as np

from streamtimelens.baselines.snag_adapt import RankedSpan, SnAGStateItem


@dataclass(frozen=True)
class SnAGPhysicalReaderConfig:
    feature_dim: int = 512
    query_dim: int = 512
    hidden_dim: int = 256
    num_heads: int = 4
    num_layers: int = 3
    fpn_levels: int = 4
    dropout: float = 0.1
    score_threshold: float = 0.001
    nms_iou_threshold: float = 0.5
    pre_nms_topk: int = 2000
    max_output_spans: int = 5
    architecture_name: str = "snag-adapt-physical-time-v1"

    def __post_init__(self) -> None:
        if min(
            self.feature_dim, self.query_dim, self.hidden_dim, self.num_heads,
            self.num_layers, self.fpn_levels, self.pre_nms_topk,
            self.max_output_spans,
        ) < 1:
            raise ValueError("SnAG physical reader dimensions/counts must be positive")
        if self.hidden_dim % self.num_heads:
            raise ValueError("hidden_dim must be divisible by num_heads")
        if not 0 <= self.dropout < 1 or not 0 <= self.score_threshold <= 1:
            raise ValueError("invalid dropout or score threshold")
        if not 0 <= self.nms_iou_threshold <= 1 or not self.architecture_name:
            raise ValueError("invalid NMS threshold or architecture name")


@dataclass(frozen=True)
class SnAGPhysicalQuery:
    embedding: np.ndarray
    t_q: float

    def __post_init__(self) -> None:
        value = np.asarray(self.embedding, dtype=np.float32)
        if value.ndim != 1 or not value.size or not np.isfinite(value).all():
            raise ValueError("query embedding must be a finite vector")
        if not math.isfinite(self.t_q) or self.t_q <= 0:
            raise ValueError("physical query time must be positive")
        object.__setattr__(self, "embedding", value.copy())


def item_metadata(items: Sequence[SnAGStateItem]) -> np.ndarray:
    """Return center, support width, aggregation, level and uncertainty."""
    rows = []
    for item in items:
        rows.append((
            (item.t_start_s + item.t_end_s) / 2,
            item.t_end_s - item.t_start_s,
            float(item.aggregation_count),
            float(item.level_or_scale),
            item.left_uncertainty_s,
            item.right_uncertainty_s,
        ))
    return np.asarray(rows, dtype=np.float32).reshape(-1, 6)


def _torch_modules() -> tuple[Any, Any]:
    try:
        import torch
        import torch.nn as nn
    except ImportError as exc:  # pragma: no cover - experiment dependency
        raise RuntimeError("torch is required for the learned SnAG reader") from exc
    return torch, nn


class SnAGPhysicalTimeModel:
    """Factory wrapper keeping torch optional during protocol-only imports."""

    @staticmethod
    def build(config: SnAGPhysicalReaderConfig) -> Any:
        torch, nn = _torch_modules()

        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.config = config
                self.feature_projection = nn.Linear(config.feature_dim, config.hidden_dim)
                self.time_projection = nn.Sequential(
                    nn.Linear(6, config.hidden_dim), nn.GELU(),
                    nn.Linear(config.hidden_dim, config.hidden_dim),
                )
                layer = nn.TransformerEncoderLayer(
                    config.hidden_dim, config.num_heads,
                    dim_feedforward=config.hidden_dim * 4,
                    dropout=config.dropout, activation="gelu", batch_first=True,
                )
                self.video_encoder = nn.TransformerEncoder(layer, config.num_layers)
                self.query_projection = nn.Linear(config.query_dim, config.hidden_dim)
                self.fusion = nn.Sequential(
                    nn.Linear(config.hidden_dim * 3, config.hidden_dim),
                    nn.GELU(), nn.Dropout(config.dropout),
                )
                self.classifier = nn.Linear(config.hidden_dim, 1)
                self.regressor = nn.Linear(config.hidden_dim, 2)

            @staticmethod
            def _normalize_metadata(metadata: Any, t_q: Any) -> Any:
                scale = t_q[:, None, None].clamp_min(1e-6)
                normalized = metadata.clone()
                normalized[..., 0:2] = normalized[..., 0:2] / scale
                normalized[..., 2] = torch.log2(normalized[..., 2].clamp_min(1.0)) / 16.0
                normalized[..., 3] = normalized[..., 3] / 16.0
                normalized[..., 4:6] = normalized[..., 4:6] / scale
                return normalized

            @staticmethod
            def _pool(values: Any, masks: Any, metadata: Any) -> tuple[Any, Any, Any]:
                length = values.shape[1]
                if length <= 1:
                    return values, masks, metadata
                if length % 2:
                    values = torch.cat((values, values[:, -1:]), dim=1)
                    masks = torch.cat((masks, torch.zeros_like(masks[:, -1:])), dim=1)
                    metadata = torch.cat((metadata, metadata[:, -1:]), dim=1)
                pair_mask = masks.reshape(masks.shape[0], -1, 2)
                weights = pair_mask.to(values.dtype)[..., None]
                pair_values = values.reshape(values.shape[0], -1, 2, values.shape[-1])
                pooled = (pair_values * weights).sum(2) / weights.sum(2).clamp_min(1.0)
                pair_metadata = metadata.reshape(metadata.shape[0], -1, 2, 6)
                meta = (pair_metadata * weights).sum(2) / weights.sum(2).clamp_min(1.0)
                # Center/width represent the union support, not an average width.
                starts = pair_metadata[..., 0] - pair_metadata[..., 1] / 2
                ends = pair_metadata[..., 0] + pair_metadata[..., 1] / 2
                start = torch.where(pair_mask, starts, torch.inf).amin(2)
                end = torch.where(pair_mask, ends, -torch.inf).amax(2)
                valid = pair_mask.any(2)
                meta[..., 0] = torch.where(valid, (start + end) / 2, 0.0)
                meta[..., 1] = torch.where(valid, end - start, 1.0)
                return pooled, valid, meta

            def forward(
                self, features: Any, metadata: Any, masks: Any,
                query: Any, t_q: Any,
            ) -> tuple[tuple[Any, ...], tuple[Any, ...], tuple[Any, ...]]:
                if features.ndim != 3 or metadata.shape[:2] != features.shape[:2]:
                    raise ValueError("invalid SnAG physical reader tensor shapes")
                time_values = self._normalize_metadata(metadata, t_q)
                video = self.feature_projection(features) + self.time_projection(time_values)
                video = self.video_encoder(video, src_key_padding_mask=~masks)
                query_value = self.query_projection(query)[:, None]
                logits: list[Any] = []
                offsets: list[Any] = []
                level_metadata: list[Any] = []
                values, level_masks, level_meta = video, masks, metadata
                for level in range(config.fpn_levels):
                    expanded = query_value.expand(-1, values.shape[1], -1)
                    fused = self.fusion(torch.cat((values, expanded, values * expanded), dim=-1))
                    logits.append(self.classifier(fused).squeeze(-1).masked_fill(~level_masks, -80.0))
                    offsets.append(torch.nn.functional.softplus(self.regressor(fused)))
                    level_metadata.append(level_meta)
                    if level + 1 < config.fpn_levels:
                        values, level_masks, level_meta = self._pool(values, level_masks, level_meta)
                return tuple(logits), tuple(offsets), tuple(level_metadata)

        return Model()


def physical_time_loss(
    logits: Sequence[Any], offsets: Sequence[Any], metadata: Sequence[Any],
    masks: Any, gt_spans: Any,
) -> tuple[Any, dict[str, float]]:
    torch, _ = _torch_modules()
    classification = torch.zeros((), device=gt_spans.device)
    regression = torch.zeros((), device=gt_spans.device)
    level_masks = masks
    positive_count = 0
    for level_logits, level_offsets, level_meta in zip(logits, offsets, metadata):
        centers = level_meta[..., 0]
        widths = level_meta[..., 1].clamp_min(1e-6)
        positives = (
            (centers >= gt_spans[:, None, 0])
            & (centers <= gt_spans[:, None, 1])
            & level_masks
        )
        # Every example needs at least one regression anchor.
        missing = ~positives.any(dim=1)
        if missing.any():
            gt_center = gt_spans.mean(dim=1)
            distance = (centers - gt_center[:, None]).abs().masked_fill(~level_masks, torch.inf)
            nearest = distance.argmin(dim=1)
            positives[missing, nearest[missing]] = True
        targets = positives.to(level_logits.dtype)
        classification = classification + torch.nn.functional.binary_cross_entropy_with_logits(
            level_logits[level_masks], targets[level_masks], reduction="mean",
        )
        target_offsets = torch.stack((
            (centers - gt_spans[:, None, 0]) / widths,
            (gt_spans[:, None, 1] - centers) / widths,
        ), dim=-1).clamp_min(0.0)
        regression = regression + torch.nn.functional.smooth_l1_loss(
            level_offsets[positives], target_offsets[positives], reduction="mean",
        )
        positive_count += int(positives.sum().detach().cpu())
        if level_masks.shape[1] > 1:
            level_masks = level_masks[:, ::2]
    loss = classification + regression
    return loss, {
        "loss": float(loss.detach().cpu()),
        "classification_loss": float(classification.detach().cpu()),
        "regression_loss": float(regression.detach().cpu()),
        "positive_points": positive_count,
    }


class SnAGPhysicalTimeBackend:
    def __init__(self, model: Any, config: SnAGPhysicalReaderConfig, *, device: str = "cpu") -> None:
        self.model = model.to(device).eval()
        self.config = config
        self.device = device

    def predict(
        self, features: np.ndarray, items: Sequence[SnAGStateItem], query: Any,
    ) -> tuple[RankedSpan, ...]:
        torch, _ = _torch_modules()
        if not items:
            return ()
        if not isinstance(query, SnAGPhysicalQuery):
            raise TypeError("physical-time backend requires SnAGPhysicalQuery")
        values = torch.as_tensor(features, dtype=torch.float32, device=self.device)[None]
        metadata = torch.as_tensor(item_metadata(items), dtype=torch.float32, device=self.device)[None]
        masks = torch.ones((1, len(items)), dtype=torch.bool, device=self.device)
        query_value = torch.as_tensor(
            query.embedding, dtype=torch.float32, device=self.device,
        ).reshape(1, -1)
        # The model consumes float32, but protocol clipping must retain the
        # snapshot's Python-float boundary. At long timestamps, rounding t_q
        # through float32 can move a span several microseconds into the future.
        boundary_t_q = float(query.t_q)
        t_q = torch.as_tensor([boundary_t_q], dtype=torch.float32, device=self.device)
        with torch.no_grad():
            logits, offsets, levels = self.model(values, metadata, masks, query_value, t_q)
        candidates: list[RankedSpan] = []
        for level_logits, level_offsets, level_meta in zip(logits, offsets, levels):
            scores = torch.sigmoid(level_logits[0])
            centers, widths = level_meta[0, :, 0], level_meta[0, :, 1]
            spans = torch.stack((
                centers - level_offsets[0, :, 0] * widths,
                centers + level_offsets[0, :, 1] * widths,
            ), dim=-1)
            for span, score in zip(spans.cpu().numpy(), scores.cpu().numpy()):
                start, end = max(0.0, float(span[0])), min(boundary_t_q, float(span[1]))
                if score > self.config.score_threshold and end > start:
                    candidates.append(RankedSpan(start, end, float(score)))
        candidates.sort(key=lambda value: value.score, reverse=True)
        kept: list[RankedSpan] = []
        for candidate in candidates[:self.config.pre_nms_topk]:
            if all(_iou(candidate, previous) <= self.config.nms_iou_threshold for previous in kept):
                kept.append(candidate)
                if len(kept) >= self.config.max_output_spans:
                    break
        return tuple(kept)


def _iou(left: RankedSpan, right: RankedSpan) -> float:
    intersection = max(0.0, min(left.end_s, right.end_s) - max(left.start_s, right.start_s))
    union = max(left.end_s, right.end_s) - min(left.start_s, right.start_s)
    return intersection / union if union > 0 else 0.0


def reader_config_dict(config: SnAGPhysicalReaderConfig) -> dict[str, Any]:
    return asdict(config)
