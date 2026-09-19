"""Prepare non-contiguous frames for TimeLens' timestamp-aware processor."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence


@dataclass(frozen=True)
class SparseFrame:
    """A decoded frame retaining its position in the original video."""

    frame_index: int
    timestamp_s: float
    image: Any

    def __post_init__(self) -> None:
        if self.frame_index < 0 or self.timestamp_s < 0:
            raise ValueError("frame index and timestamp must be non-negative")


@dataclass(frozen=True)
class PreparedSparseVideo:
    """Processor input plus an exact human-readable timestamp audit."""

    video: Any
    metadata: dict[str, Any]
    unique_indices: tuple[int, ...]
    timestamps_s: tuple[float, ...]
    timestamp_audit: tuple[str, ...]

    @property
    def processor_video(self) -> tuple[Any, dict[str, Any]]:
        return self.video, self.metadata


def _stack_and_repeat(images: Sequence[Any]) -> Any:
    """Produce the processor's TCHW [f0,f0,f1,f1,...] representation."""
    try:
        import torch

        if all(isinstance(image, torch.Tensor) for image in images):
            stacked = torch.stack(list(images), dim=0)
            if stacked.ndim == 4 and stacked.shape[-1] in (1, 3, 4):
                stacked = stacked.permute(0, 3, 1, 2)
            return stacked.repeat_interleave(2, dim=0)
    except ImportError:
        pass
    try:
        import numpy as np

        stacked = np.stack(images, axis=0)
        if stacked.ndim == 4 and stacked.shape[-1] in (1, 3, 4):
            stacked = stacked.transpose(0, 3, 1, 2)
        return np.repeat(stacked, 2, axis=0)
    except (ImportError, ValueError, TypeError) as exc:
        raise TypeError("images must be consistently shaped numpy arrays or torch tensors") from exc


def prepare_sparse_video(
    frames: Iterable[SparseFrame],
    *,
    original_fps: float,
    total_num_frames: int,
    k_frames: int,
    timestamp_tolerance_s: float | None = None,
    min_visual_tokens: int = 64,
    total_visual_tokens: int = 4096,
) -> PreparedSparseVideo:
    """Sort/deduplicate sparse frames while preserving original global indices.

    TimeLens consumes timestamps from ``frames_indices[::2] / fps``. Both pixels
    and indices are therefore duplicated in adjacent pairs. Supplied timestamps
    are checked against that official calculation so audit text cannot disagree
    with the timestamps actually shown to the model.
    """
    if original_fps <= 0 or total_num_frames <= 0 or k_frames <= 0:
        raise ValueError("fps, total frame count, and K_frames must be positive")
    if min_visual_tokens <= 0 or total_visual_tokens < min_visual_tokens:
        raise ValueError("invalid TimeLens visual token budget")
    ordered = sorted(frames, key=lambda frame: (frame.timestamp_s, frame.frame_index))
    unique: list[SparseFrame] = []
    seen_indices: set[int] = set()
    seen_timestamps: set[float] = set()
    for frame in ordered:
        if frame.frame_index >= total_num_frames:
            raise ValueError("frame index exceeds original video metadata")
        if frame.frame_index in seen_indices or frame.timestamp_s in seen_timestamps:
            continue
        unique.append(frame)
        seen_indices.add(frame.frame_index)
        seen_timestamps.add(frame.timestamp_s)
    if not unique:
        raise ValueError("at least one unique sparse frame is required")
    if len(unique) > k_frames:
        raise ValueError(f"unique sparse frame count {len(unique)} exceeds K_frames={k_frames}")

    official_times = tuple(frame.frame_index / float(original_fps) for frame in unique)
    tolerance = timestamp_tolerance_s if timestamp_tolerance_s is not None else 0.5 / float(original_fps) + 1e-6
    for frame, official_time in zip(unique, official_times):
        if abs(frame.timestamp_s - official_time) > tolerance:
            raise ValueError("frame timestamp disagrees with original index/fps metadata")

    indices = tuple(frame.frame_index for frame in unique)
    repeated_indices = [index for index in indices for _ in range(2)]
    metadata = {
        "fps": float(original_fps),
        "frames_indices": repeated_indices,
        "total_num_frames": int(total_num_frames),
    }
    video = _stack_and_repeat([frame.image for frame in unique])
    # The official loader resizes before a processor configured with do_resize=False.
    # Sparse in-memory inputs must do the same, including divisibility by 28.
    if video.shape[-2] >= 28 and video.shape[-1] >= 28:
        try:
            import torch
            from qwen_vl_utils.vision_process import smart_resize

            if not isinstance(video, torch.Tensor):
                video = torch.from_numpy(video)
            frame_factor = 28
            nframes = int(video.shape[0])
            min_pixels = min_visual_tokens * frame_factor * frame_factor
            max_pixels = max(
                min(768 * frame_factor * frame_factor, total_visual_tokens * frame_factor * frame_factor / nframes * 2),
                int(min_pixels * 1.05),
            )
            height, width = smart_resize(
                int(video.shape[-2]), int(video.shape[-1]), factor=frame_factor,
                min_pixels=min_pixels, max_pixels=max_pixels,
            )
            if (height, width) != tuple(video.shape[-2:]):
                video = torch.nn.functional.interpolate(
                    video.float(), size=(height, width), mode="bicubic", align_corners=False, antialias=True,
                )
            else:
                video = video.float()
        except ImportError as exc:
            raise RuntimeError("qwen-vl-utils and torch are required to spatially prepare TimeLens frames") from exc
    return PreparedSparseVideo(
        video=video,
        metadata=metadata,
        unique_indices=indices,
        timestamps_s=official_times,
        timestamp_audit=tuple(f"frame_index={index} timestamp={timestamp:.1f}s" for index, timestamp in zip(indices, official_times)),
    )
