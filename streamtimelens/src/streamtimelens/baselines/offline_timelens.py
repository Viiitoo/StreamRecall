"""Frozen Offline TimeLens wrapper, explicitly outside the streaming protocol."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

from streamtimelens.evaluation.cost import ComponentTimer
from streamtimelens.refiner.parse import parse_refiner_output
from streamtimelens.refiner.prompts import OFFICIAL_CROP_VERSION, build_grounding_prompt, video_messages
from streamtimelens.refiner.timelens_adapter import SparseFrame, prepare_sparse_video


@dataclass(frozen=True)
class OfflineQuery:
    query_id: str
    video_id: str
    query: str
    video_path: Path


@dataclass(frozen=True)
class LoadedOfflineVideo:
    processor_video: Any
    duration_s: float
    visual_frames: int
    visual_tokens: int


@dataclass(frozen=True)
class OfflinePrediction:
    query_id: str
    video_id: str
    span: tuple[float, float] | None
    status: str
    raw_answer: str
    resource: dict[str, Any]
    visual_frames: int
    visual_tokens: int


def load_offline_video(video_path: Path, *, sample_fps: float = 2.0) -> LoadedOfflineVideo:
    """Decode a full video anew for one query, matching the frozen adapter."""
    if sample_fps <= 0 or not video_path.is_file():
        raise ValueError("offline video path/sample FPS is invalid")
    try:
        import decord
    except ImportError as exc:  # pragma: no cover - inference environment
        raise RuntimeError("Decord is required for Offline TimeLens") from exc
    reader = decord.VideoReader(str(video_path), ctx=decord.cpu(0), num_threads=1)
    fps = float(reader.get_avg_fps())
    total = len(reader)
    if fps <= 0 or total <= 0:
        raise ValueError("offline video metadata is invalid")
    step = max(1, round(fps / sample_fps))
    indices = list(range(0, total, step))
    if indices[-1] != total - 1:
        indices.append(total - 1)
    images = reader.get_batch(indices).asnumpy()
    prepared = prepare_sparse_video(
        [SparseFrame(index, index / fps, image) for index, image in zip(indices, images)],
        original_fps=fps, total_num_frames=total, k_frames=len(indices),
    )
    stats = prepared.metadata
    visual_tokens = int(stats.get("total_visual_tokens", 4096))
    return LoadedOfflineVideo(
        prepared.processor_video, total / fps, len(indices), visual_tokens,
    )


class OfflineTimeLensWrapper:
    """Each query reloads and re-decodes the full video; no cost is amortized."""

    def __init__(
        self, service: Any, *, loader: Callable[[Path], LoadedOfflineVideo] = load_offline_video,
        max_new_tokens: int = 128,
    ) -> None:
        if max_new_tokens <= 0:
            raise ValueError("offline generation limit must be positive")
        self.service = service
        self.loader = loader
        self.max_new_tokens = max_new_tokens

    def run(self, queries: Sequence[OfflineQuery]) -> list[OfflinePrediction]:
        predictions = []
        for item in queries:
            if not item.query_id or not item.video_id or not item.query.strip():
                raise ValueError("offline query fields are required")
            with ComponentTimer(
                "offline_timelens_query", torch_module=getattr(self.service, "torch", None),
            ) as timer:
                loaded = self.loader(item.video_path)
                prompt = build_grounding_prompt(OFFICIAL_CROP_VERSION, query=item.query)
                raw = str(self.service.generate(
                    video_messages(prompt), [loaded.processor_video], self.max_new_tokens,
                ))
                parsed = parse_refiner_output(
                    raw, t_q=loaded.duration_s, candidate=(0.0, loaded.duration_s),
                )
            resource = timer.as_dict()
            resource.update({
                "visual_frames": loaded.visual_frames,
                "visual_tokens": loaded.visual_tokens,
                "full_video_reloads": 1,
            })
            predictions.append(OfflinePrediction(
                item.query_id, item.video_id, parsed.span, parsed.status, raw,
                resource, loaded.visual_frames, loaded.visual_tokens,
            ))
        return predictions
