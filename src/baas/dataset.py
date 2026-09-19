"""Explicit-frame TimeLens input adapter.

The official evaluator delegates video sampling to ``qwen_vl_utils``.  This
adapter decodes the frames selected by :class:`SamplingPlan` itself and sends them
as interleaved images with their original timestamps, so non-uniform policies can
use exactly the same model prompt and image processor.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Iterable, Sequence

from .sampling import SamplingError, SamplingPlan, VideoMetadata


GROUNDER_PROMPT = (
    "Please find the visual event described by the sentence '{}', determining its starting and ending times. "
    "The format should be: 'The event happens in <start time> - <end time> seconds'."
)
GROUNDER_PROMPT_TEXT_TIMESTAMP = (
    "You are given a video with multiple frames. "
    "The numbers before each video frame indicate its sampling timestamp (in seconds). "
) + GROUNDER_PROMPT


class DecodeError(RuntimeError):
    """A video could not be read exactly according to its sampling plan."""


@dataclass(frozen=True)
class TokenStats:
    actual_visual_tokens: int
    image_grid_thw: tuple[tuple[int, int, int], ...]
    pixel_value_patches: int
    decode_latency_ms: float
    preprocess_latency_ms: float


def read_video_metadata(video_path: str | Path, *, include_vfr_timestamps: bool = True) -> VideoMetadata:
    """Read video timing via PyAV, retaining exact PTS values when practical."""
    try:
        import av
    except ImportError as exc:  # pragma: no cover - exercised in inference image
        raise DecodeError("PyAV is required for explicit frame decoding; install av") from exc

    path = str(video_path)
    try:
        with av.open(path) as container:
            stream = container.streams.video[0]
            fps = float(stream.average_rate or stream.base_rate or 0)
            if fps <= 0:
                raise DecodeError(f"video has no usable FPS: {path}")
            duration = float(stream.duration * stream.time_base) if stream.duration else None
            frame_count = int(stream.frames or 0)
            timestamps: tuple[float, ...] | None = None
            # Container metadata often has no frame count. Decode once in that case;
            # this also gives a correct VFR lookup table rather than silently using
            # average FPS.
            if include_vfr_timestamps or frame_count <= 0:
                values = [float(frame.time) for frame in container.decode(stream) if frame.time is not None]
                if not values:
                    raise DecodeError(f"no decodable video frames: {path}")
                # Ground-truth timestamps are relative to video start, while some
                # containers expose a non-zero first PTS.
                origin = values[0]
                timestamps = tuple(value - origin for value in values)
                frame_count = len(timestamps)
                duration = max(duration or 0.0, timestamps[-1] + 1.0 / fps)
            if duration is None or duration <= 0:
                duration = frame_count / fps
    except DecodeError:
        raise
    except Exception as exc:
        raise DecodeError(f"failed to read metadata for {path}: {exc}") from exc
    return VideoMetadata(fps=fps, frame_count=frame_count, duration_sec=duration, frame_timestamps_sec=timestamps)


def decode_selected_frames(video_path: str | Path, frame_indices: Sequence[int]) -> list[Any]:
    """Decode exactly the selected frame indices into RGB PIL images."""
    try:
        import av
    except ImportError as exc:  # pragma: no cover
        raise DecodeError("PyAV is required for explicit frame decoding; install av") from exc
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover
        raise DecodeError("Pillow is required for explicit frame decoding") from exc

    targets = tuple(int(index) for index in frame_indices)
    if not targets or tuple(sorted(set(targets))) != targets:
        raise DecodeError("frame_indices must be non-empty, unique, and increasing")
    found: dict[int, Any] = {}
    try:
        with av.open(str(video_path)) as container:
            stream = container.streams.video[0]
            wanted = set(targets)
            for index, frame in enumerate(container.decode(stream)):
                if index in wanted:
                    array = frame.to_ndarray(format="rgb24")
                    found[index] = Image.fromarray(array)
                    if len(found) == len(targets):
                        break
    except Exception as exc:
        raise DecodeError(f"failed to decode {video_path}: {exc}") from exc
    missing = [index for index in targets if index not in found]
    if missing:
        raise DecodeError(f"decoder did not yield requested frame indices: {missing}")
    return [found[index] for index in targets]


def build_messages(query: str, plan: SamplingPlan, frames: Sequence[Any]) -> list[dict[str, Any]]:
    """Build the TimeLens-7B chat content with one timestamp before every frame."""
    if len(frames) != len(plan.timestamps_sec):
        raise SamplingError("number of decoded frames does not match sampling plan")
    content: list[dict[str, Any]] = []
    for timestamp, frame in zip(plan.timestamps_sec, frames):
        content.append({"type": "text", "text": f"{timestamp:.6f} seconds: "})
        content.append({"type": "image", "image": frame})
    content.append({"type": "text", "text": GROUNDER_PROMPT_TEXT_TIMESTAMP.format(query)})
    return [{"role": "user", "content": content}]


def _as_int_grid(value: Any) -> tuple[tuple[int, int, int], ...]:
    if value is None:
        return ()
    if hasattr(value, "detach"):
        value = value.detach().cpu().tolist()
    elif hasattr(value, "tolist"):
        value = value.tolist()
    grid = tuple(tuple(int(axis) for axis in row) for row in value)
    if any(len(row) != 3 or any(axis <= 0 for axis in row) for row in grid):
        raise SamplingError("image_grid_thw must contain positive THW rows")
    return grid


def count_visual_tokens(inputs: Any, processor: Any) -> tuple[int, tuple[tuple[int, int, int], ...], int]:
    """Measure LLM visual tokens from the processor output, not a nominal budget."""
    grid = _as_int_grid(inputs.get("image_grid_thw") if hasattr(inputs, "get") else inputs["image_grid_thw"])
    pixel_values = inputs.get("pixel_values") if hasattr(inputs, "get") else inputs["pixel_values"]
    if not grid or pixel_values is None:
        raise SamplingError("processor output lacks image_grid_thw or pixel_values")
    patches = int(pixel_values.shape[0])
    raw_patches = sum(time * height * width for time, height, width in grid)
    if raw_patches != patches:
        raise SamplingError(
            f"image_grid_thw describes {raw_patches} patches but pixel_values contains {patches}"
        )
    image_processor = getattr(processor, "image_processor", processor)
    merge_size = int(getattr(image_processor, "merge_size", 2))
    merge_area = merge_size * merge_size
    if raw_patches % merge_area:
        raise SamplingError("image patch count is not divisible by the visual merge area")
    return raw_patches // merge_area, grid, patches


def prepare_inputs(
    video_path: str | Path,
    query: str,
    plan: SamplingPlan,
    processor: Any,
    *,
    frames: Sequence[Any] | None = None,
    max_pixels: int | None = None,
    min_pixels: int | None = None,
) -> tuple[Any, TokenStats]:
    """Decode selected frames, create explicit timestamp inputs, and audit tokens."""
    decode_start = perf_counter()
    if frames is None:
        frames = decode_selected_frames(video_path, plan.frame_indices)
    decode_latency_ms = (perf_counter() - decode_start) * 1000
    messages = build_messages(query, plan, frames)
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    preprocess_start = perf_counter()
    kwargs: dict[str, Any] = {
        "text": [text],
        "images": list(frames),
        "padding": True,
        "return_tensors": "pt",
    }
    if max_pixels is not None:
        kwargs["max_pixels"] = int(max_pixels)
    if min_pixels is not None:
        kwargs["min_pixels"] = int(min_pixels)
    inputs = processor(**kwargs)
    actual, grid, patches = count_visual_tokens(inputs, processor)
    preprocess_latency_ms = (perf_counter() - preprocess_start) * 1000
    if actual > plan.requested_budget:
        raise SamplingError(
            f"actual visual tokens ({actual}) exceed budget ({plan.requested_budget})"
        )
    return inputs, TokenStats(actual, grid, patches, decode_latency_ms, preprocess_latency_ms)
