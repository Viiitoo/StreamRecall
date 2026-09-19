"""Real, query-blind generation for the fixed writer-feasibility chunks."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

from streamtimelens.evaluation.cost import ComponentTimer
from streamtimelens.refiner.timelens_adapter import SparseFrame, prepare_sparse_video
from streamtimelens.writer.prompts import (
    WRITER_MAX_NEW_TOKENS, build_writer_prompt, writer_video_messages,
)


@dataclass(frozen=True)
class WriterChunk:
    chunk_id: str
    video_id: str
    video_path: str
    segment: tuple[float, float]
    sampled_timestamps: tuple[float, ...]
    gt_spans: tuple[tuple[float, float], ...]

    @classmethod
    def from_mapping(cls, row: Mapping[str, Any]) -> "WriterChunk":
        segment = tuple(map(float, row.get("segment", ())))
        sampled = tuple(map(float, row.get("sampled_timestamps", ())))
        gt_spans = tuple(tuple(map(float, span)) for span in row.get("gt_spans", ()))
        chunk = cls(
            str(row.get("chunk_id", "")).strip(), str(row.get("video_id", "")).strip(),
            str(row.get("video_path", "")).strip(), segment, sampled, gt_spans,
        )
        chunk.validate()
        return chunk

    def validate(self) -> None:
        if not self.chunk_id or not self.video_id or not self.video_path:
            raise ValueError("writer chunk identifiers and video path are required")
        if len(self.segment) != 2 or not 0 <= self.segment[0] < self.segment[1]:
            raise ValueError(f"writer chunk {self.chunk_id} has an invalid segment")
        if (
            len(self.sampled_timestamps) < 2
            or tuple(sorted(set(self.sampled_timestamps))) != self.sampled_timestamps
            or self.sampled_timestamps[0] < self.segment[0] - 1e-6
            or self.sampled_timestamps[-1] > self.segment[1] + 1e-6
        ):
            raise ValueError(f"writer chunk {self.chunk_id} has invalid sampled timestamps")
        if any(
            len(span) != 2
            or not all(math.isfinite(value) for value in span)
            or not self.segment[0] <= span[0] < span[1] <= self.segment[1]
            for span in self.gt_spans
        ):
            raise ValueError(f"writer chunk {self.chunk_id} has an invalid GT span")


@dataclass(frozen=True)
class LoadedChunkFrames:
    images: tuple[Any, ...]
    frame_indices: tuple[int, ...]
    original_fps: float
    total_num_frames: int


ChunkFrameLoader = Callable[[WriterChunk], LoadedChunkFrames]


def generate_writer_record(
    chunk: WriterChunk,
    *,
    model_id: str,
    model_revision: str,
    service: Any,
    frame_loader: ChunkFrameLoader,
) -> dict[str, Any]:
    """Generate one raw record without exposing query or GT to the model."""
    if not model_id or not model_revision:
        raise ValueError("writer model ID and revision are required")
    loaded = frame_loader(chunk)
    if len(loaded.images) != len(chunk.sampled_timestamps) or len(
        loaded.frame_indices
    ) != len(chunk.sampled_timestamps):
        raise ValueError("writer frame loader returned the wrong number of frames")
    prepared = prepare_sparse_video(
        [
            SparseFrame(index, timestamp, image)
            for index, timestamp, image in zip(
                loaded.frame_indices, chunk.sampled_timestamps, loaded.images,
            )
        ],
        original_fps=loaded.original_fps,
        total_num_frames=loaded.total_num_frames,
        k_frames=len(chunk.sampled_timestamps),
    )
    prompt = build_writer_prompt(
        segment=chunk.segment, sampled_timestamps=chunk.sampled_timestamps,
    )
    # TimeLens' Qwen2-VL processor consumes the (pixels, global metadata)
    # extension used by the timestamp adapter.  The untouched Qwen2.5-VL base
    # processor accepts pixels only; its global seconds remain explicit in the
    # identical writer prompt and are therefore still auditable.
    processor_name = type(getattr(service, "processor", None)).__name__
    video_input = (
        prepared.video if processor_name == "Qwen2_5_VLProcessor" else prepared.processor_video
    )
    torch_module = getattr(service, "torch", None)
    with ComponentTimer("writer_feasibility_generation", torch_module=torch_module) as timer:
        raw_output = service.generate(
            writer_video_messages(prompt), [video_input], WRITER_MAX_NEW_TOKENS,
        )
    resource = timer.as_dict()
    gpu_s = resource["cuda_s"] if resource["cuda_s"] is not None else resource["wall_s"]
    return {
        "model": model_id,
        "model_revision": model_revision,
        "chunk_id": chunk.chunk_id,
        "video_id": chunk.video_id,
        "segment": list(chunk.segment),
        "sampled_timestamps": list(chunk.sampled_timestamps),
        "raw_output": str(raw_output),
        "gt_spans": [list(span) for span in chunk.gt_spans],
        "gpu_s": float(gpu_s),
        "resource": resource,
        "model_stats": dict(getattr(service, "last_call_stats", {})),
        "timestamp_audit": list(prepared.timestamp_audit),
        "processor_input_adapter": (
            "pixels_only_with_prompt_timestamps"
            if video_input is prepared.video else "timelens_global_metadata"
        ),
        "prompt_version": prompt.version,
        "prompt_template_sha256": prompt.template_sha256,
        "rendered_prompt_sha256": prompt.sha256,
    }


def decord_frame_loader(
    chunk: WriterChunk, *, video_root: Path | None = None,
) -> LoadedChunkFrames:
    """Decode exactly the fixed timestamps from one chunk's frozen local video."""
    try:
        import decord
    except ImportError as exc:  # pragma: no cover - real inference dependency
        raise RuntimeError("decord is required for writer feasibility generation") from exc
    video_path = (
        Path(chunk.video_path) if video_root is None else Path(video_root) / f"{chunk.video_id}.mp4"
    )
    if not video_path.is_file():
        raise FileNotFoundError(video_path)
    reader = decord.VideoReader(str(video_path))
    fps = float(reader.get_avg_fps())
    total = len(reader)
    indices = tuple(min(total - 1, max(0, round(timestamp * fps))) for timestamp in chunk.sampled_timestamps)
    if len(set(indices)) != len(indices):
        raise ValueError(f"writer chunk {chunk.chunk_id} timestamps map to duplicate frames")
    images = tuple(reader.get_batch(list(indices)).asnumpy())
    return LoadedChunkFrames(images, indices, fps, total)
