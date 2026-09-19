"""Minimal bridge to a SnAG-compatible late-fusion model.

Adapted from fmu2/snag_release at commit
44dd90eea9a65b64f7088974eae352c1e26ef6e3.  This module intentionally does
not import the checkout under ``ref/``.  A checkpoint-compatible PtTransformer
can be supplied by the experiment environment as long as it exposes
``encode_video``, ``encode_text`` and ``fuse_and_predict``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from streamtimelens.baselines.snag_adapt import (
    ProvenancePointGenerator,
    RankedSpan,
    SnAGStateItem,
)


UPSTREAM_REVISION = "44dd90eea9a65b64f7088974eae352c1e26ef6e3"


@dataclass(frozen=True)
class SnAGRawTrace:
    fpn_shapes: tuple[tuple[int, ...], ...]
    mask_shapes: tuple[tuple[int, ...], ...]
    logits: tuple[np.ndarray, ...]
    offsets: tuple[np.ndarray, ...]
    masks: tuple[np.ndarray, ...]
    input_length: int
    observed_length: int


class SnAGTorchBackend:
    """Run an official-compatible SnAG head with provenance-aware decoding.

    ``query`` is ``(tokens, token_mask)`` in the format expected by the pinned
    SnAG text network.  Pooled snapshots require a reader head trained with
    physical-time offsets; set ``offset_mode='token_width'`` for that head.
    """

    def __init__(
        self,
        model: Any,
        *,
        device: str = "cpu",
        offset_mode: str = "official_stride",
        score_threshold: float = 0.001,
        segment_length_threshold_s: float = 0.0,
        pre_nms_topk: int = 2000,
        nms_iou_threshold: float = 0.5,
        input_video_length: int | None = None,
        minimum_chunk_size: int = 1,
    ) -> None:
        if offset_mode not in ("official_stride", "token_width"):
            raise ValueError("unsupported SnAG offset mode")
        if not 0 <= score_threshold <= 1 or not 0 <= nms_iou_threshold <= 1:
            raise ValueError("invalid SnAG score/NMS threshold")
        self.model = model
        self.device = device
        self.offset_mode = offset_mode
        self.score_threshold = score_threshold
        self.segment_length_threshold_s = segment_length_threshold_s
        self.pre_nms_topk = pre_nms_topk
        self.nms_iou_threshold = nms_iou_threshold
        if input_video_length is not None and input_video_length < 1:
            raise ValueError("input_video_length must be positive")
        if minimum_chunk_size < 1:
            raise ValueError("minimum_chunk_size must be positive")
        self.input_video_length = input_video_length
        self.minimum_chunk_size = minimum_chunk_size
        self.point_generator = ProvenancePointGenerator()

    @classmethod
    def from_upstream_option(cls, model: Any, option: Any, **kwargs: Any) -> "SnAGTorchBackend":
        """Build the bridge with the padding contract used by Evaluator.predict."""
        model_opt = option["model"]
        max_video_length = int(model_opt["max_vid_len"])
        video_stride = int(model_opt.get("vid_stride", 1))
        input_length = max_video_length * video_stride
        minimum_chunk_size = 1
        for level in range(int(model_opt["num_fpn_levels"])):
            stride = 2 ** level
            window = int(model_opt["mha_win_size"])
            if window > 0:
                stride *= (window // 2) * 2
            minimum_chunk_size = max(minimum_chunk_size, stride * video_stride)
        eval_opt = option["eval"]
        return cls(
            model,
            input_video_length=input_length,
            minimum_chunk_size=minimum_chunk_size,
            score_threshold=float(eval_opt["pre_nms_thresh"]),
            segment_length_threshold_s=0.0,
            pre_nms_topk=int(eval_opt["pre_nms_topk"]),
            **kwargs,
        )

    def raw_predict(
        self, features: np.ndarray, query: Any,
    ) -> SnAGRawTrace:
        """Run the real split model API and retain tensors needed by G0 parity."""
        try:
            import torch
            import torch.nn.functional as functional
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("torch is required for the SnAG model bridge") from exc
        values = np.asarray(features)
        if values.ndim != 2 or not len(values):
            raise ValueError("features must be a non-empty [time,channels] array")
        observed_length = len(values)
        input_length = self.input_video_length or observed_length
        if observed_length > input_length:
            input_length = (
                (observed_length + self.minimum_chunk_size - 1)
                // self.minimum_chunk_size * self.minimum_chunk_size
            )
        tokens, token_masks = query
        video = torch.as_tensor(values.T, dtype=torch.float32, device=self.device)
        video = functional.pad(video, (0, input_length - observed_length))[None]
        video_masks = (
            torch.arange(input_length, device=self.device).view(1, 1, -1)
            < observed_length
        )
        tokens = torch.as_tensor(tokens, dtype=torch.float32, device=self.device)
        token_masks = torch.as_tensor(token_masks, dtype=torch.bool, device=self.device)
        if tokens.ndim == 2:
            tokens = tokens[None]
        if token_masks.ndim == 1:
            token_masks = token_masks[None, None]
        elif token_masks.ndim == 2:
            token_masks = token_masks[:, None]
        with torch.no_grad():
            text, text_masks = self.model.encode_text(tokens, token_masks)
            fpn, fpn_masks = self.model.encode_video(video, video_masks)
            logits, offsets, masks = self.model.fuse_and_predict(
                fpn, fpn_masks, text, text_masks,
            )
        def numpy(value: Any) -> np.ndarray:
            return value.detach().cpu().numpy().copy()

        return SnAGRawTrace(
            fpn_shapes=tuple(tuple(value.shape) for value in fpn),
            mask_shapes=tuple(tuple(value.shape) for value in fpn_masks),
            logits=tuple(numpy(value) for value in logits),
            offsets=tuple(numpy(value) for value in offsets),
            masks=tuple(numpy(value) for value in masks),
            input_length=input_length,
            observed_length=observed_length,
        )

    def predict(
        self, features: np.ndarray, items: Sequence[SnAGStateItem], query: Any,
    ) -> tuple[RankedSpan, ...]:
        if len(features) == 0:
            return ()
        trace = self.raw_predict(features, query)
        # Valid positions are a prefix at every FPN level; padded positions are
        # never candidates in the upstream evaluator.
        valid_lengths = [int(value[0].reshape(-1).sum()) for value in trace.masks]
        points_by_level = self.point_generator(items, valid_lengths)
        candidates = []
        for level_points, level_logits, level_offsets, level_mask in zip(
            points_by_level, trace.logits, trace.offsets, trace.masks,
        ):
            length = len(level_points)
            raw_scores = np.clip(level_logits[0].reshape(-1)[:length], -80.0, 80.0)
            scores = 1.0 / (1.0 + np.exp(-raw_scores))
            values = level_offsets[0].reshape(-1, 2)[:length]
            valid = level_mask[0].reshape(-1)[:length].astype(bool)
            decoded = self.point_generator.decode(
                level_points,
                values,
                normalized_by_width=self.offset_mode == "token_width",
                duration_s=max(item.t_end_s for item in items),
            )
            for span, score, keep in zip(decoded, scores, valid):
                if keep and score > self.score_threshold and span[1] - span[0] > self.segment_length_threshold_s:
                    candidates.append(RankedSpan(float(span[0]), float(span[1]), float(score)))
        candidates.sort(key=lambda value: value.score, reverse=True)
        return self._hard_nms(candidates[:self.pre_nms_topk])

    def _hard_nms(self, spans: Sequence[RankedSpan]) -> tuple[RankedSpan, ...]:
        kept = []
        for candidate in spans:
            if all(self._iou(candidate, previous) <= self.nms_iou_threshold for previous in kept):
                kept.append(candidate)
        return tuple(kept)

    @staticmethod
    def _iou(left: RankedSpan, right: RankedSpan) -> float:
        intersection = max(0.0, min(left.end_s, right.end_s) - max(left.start_s, right.start_s))
        union = max(left.end_s, right.end_s) - min(left.start_s, right.start_s)
        return intersection / union if union > 0 else 0.0


class SnAGOfficialGridBackend(SnAGTorchBackend):
    """Exact single-window upstream point decode and soft-NMS for G0/full-store."""

    def __init__(
        self, model: Any, option: Any, nms_extension: Any, *,
        fps: float, clip_size: int, clip_stride: int, duration_s: float,
        device: str = "cpu",
    ) -> None:
        eval_opt = option["eval"]
        super().__init__(
            model, device=device,
            input_video_length=int(option["model"]["max_vid_len"]) * int(option["model"].get("vid_stride", 1)),
            minimum_chunk_size=_minimum_chunk_size(option),
            score_threshold=float(eval_opt["pre_nms_thresh"]),
            segment_length_threshold_s=0.0,
            pre_nms_topk=int(eval_opt["pre_nms_topk"]),
        )
        if min(fps, clip_size, clip_stride, duration_s) <= 0:
            raise ValueError("official SnAG temporal conversion values must be positive")
        self.option = option
        self.nms_extension = nms_extension
        self.fps = float(fps)
        self.clip_size = int(clip_size)
        self.clip_stride = int(clip_stride)
        self.duration_s = float(duration_s)

    def predict(
        self, features: np.ndarray, items: Sequence[SnAGStateItem], query: Any,
    ) -> tuple[RankedSpan, ...]:
        del items
        import torch

        trace = self.raw_predict(features, query)
        points_values = []
        score_values = []
        for level, (logits, offsets, masks) in enumerate(zip(trace.logits, trace.offsets, trace.masks)):
            stride = 2 ** level
            length = logits.shape[-1]
            centers = torch.arange(length, dtype=torch.float32) * stride
            if self.option["pt_gen"].get("use_offset", False):
                centers += 0.5 * stride
            scores = torch.sigmoid(torch.from_numpy(logits[0].reshape(-1)))
            valid = torch.from_numpy(masks[0].reshape(-1).astype(bool))
            keep = (scores > self.score_threshold) & valid
            level_offsets = torch.from_numpy(offsets[0].reshape(-1, 2))
            points_values.append((centers[keep], level_offsets[keep], stride))
            score_values.append(scores[keep])
        segments = []
        scores = []
        for (centers, offsets, stride), level_scores in zip(points_values, score_values):
            left = centers - offsets[:, 0] * stride
            right = centers + offsets[:, 1] * stride
            keep = right - left > float(self.option["eval"]["seg_len_thresh"])
            segments.append(torch.stack((left[keep], right[keep]), dim=-1))
            scores.append(level_scores[keep])
        if not segments or sum(len(value) for value in segments) == 0:
            return ()
        segments_value = torch.cat(segments)
        scores_value = torch.cat(scores)
        order = scores_value.argsort(descending=True)[:self.pre_nms_topk]
        segments_value, scores_value = _upstream_soft_nms(
            segments_value[order], scores_value[order],
            self.option["eval"]["nms"], self.nms_extension,
        )
        video_stride = int(self.option["model"].get("vid_stride", 1))
        segments_value *= video_stride
        segments_value = (segments_value * self.clip_stride + 0.5 * self.clip_size) / self.fps
        segments_value = segments_value.clamp(min=0.0, max=self.duration_s)
        return tuple(
            RankedSpan(float(span[0]), float(span[1]), float(score))
            for span, score in zip(segments_value.numpy(), scores_value.numpy())
        )


def _minimum_chunk_size(option: Any) -> int:
    model = option["model"]
    result = 1
    for level in range(int(model["num_fpn_levels"])):
        stride = 2 ** level
        window = int(model["mha_win_size"])
        if window > 0:
            stride *= (window // 2) * 2
        result = max(result, stride * int(model.get("vid_stride", 1)))
    return result


def _upstream_soft_nms(segments: Any, scores: Any, option: Any, extension: Any) -> tuple[Any, Any]:
    import torch

    segments = segments.float().contiguous().cpu()
    scores = scores.float().contiguous().cpu()
    output = segments.new_empty((len(segments), 3), device="cpu")
    indices = extension.softnms(
        segments, scores, output,
        iou_thresh=float(option["iou_thresh"]), sigma=float(option["sigma"]),
        min_score=float(option["min_score"]), method=2,
    )
    count = min(len(indices), int(option["max_num_segs"]))
    nms_segments = output[:count, :2].contiguous()
    nms_scores = output[:count, 2].contiguous()
    voting = float(option.get("voting_thresh", 0.0))
    if voting > 0 and len(nms_segments):
        left = torch.maximum(nms_segments[:, None, 0], segments[None, :, 0])
        right = torch.minimum(nms_segments[:, None, 1], segments[None, :, 1])
        overlap = (right - left).clamp(min=0)
        union = (
            (nms_segments[:, 1] - nms_segments[:, 0])[:, None]
            + (segments[:, 1] - segments[:, 0])[None, :] - overlap
        )
        weights = (overlap / union >= voting).float() * scores[None]
        weights /= weights.sum(dim=1, keepdim=True)
        nms_segments = weights @ segments
    order = nms_scores.argsort(descending=True)[:count]
    return nms_segments[order], nms_scores[order]
