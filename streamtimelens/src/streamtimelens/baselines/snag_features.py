"""Single-pass, query-blind clip feature extraction for SnAG-adapt."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
from PIL import Image

from streamtimelens.observer.clip_encoder import FrozenCLIPEncoder
from streamtimelens.protocol.types import VideoMeta


@dataclass(frozen=True)
class SnAGFeatureObservation:
    feature: np.ndarray
    t_start_s: float
    t_end_s: float
    source_id: str
    frame_index: int


@dataclass(frozen=True)
class SnAGIngestStats:
    decoded_frames: int
    emitted_features: int
    decode_passes: int
    wall_s: float
    video_duration_s: float

    @property
    def ingest_fps(self) -> float:
        return self.decoded_frames / self.wall_s if self.wall_s > 0 else 0.0


def sampling_cell(timestamp_s: float, interval_s: float, duration_s: float) -> tuple[float, float]:
    """Map a sampled frame to its deterministic feature-grid support cell."""
    if min(timestamp_s, interval_s, duration_s) < 0 or interval_s <= 0 or duration_s <= 0:
        raise ValueError("invalid sampling-cell inputs")
    cell_index = math.floor((timestamp_s + 1e-9) / interval_s)
    start_s = min(cell_index * interval_s, max(0.0, duration_s - 1e-9))
    end_s = min(duration_s, max(start_s + 1e-9, (cell_index + 1) * interval_s))
    return start_s, end_s


class SequentialCLIPFeatureExtractor:
    """Decode a video once and emit features immediately in temporal order.

    Formal runs intentionally use one image-encoder call per observation.  A
    batch spanning an arrival boundary could decode future frames before a
    snapshot is frozen, so batching is not exposed by this strict extractor.
    """

    def __init__(self, encoder: FrozenCLIPEncoder, *, feature_fps: float = 2.0) -> None:
        if not math.isfinite(feature_fps) or feature_fps <= 0:
            raise ValueError("feature_fps must be positive and finite")
        self.encoder = encoder
        self.feature_fps = float(feature_fps)
        self.last_stats: SnAGIngestStats | None = None

    def iter_video(
        self, video_path: Path | str, meta: VideoMeta,
    ) -> Iterator[SnAGFeatureObservation]:
        try:
            import cv2
        except ImportError as exc:  # pragma: no cover - declared dependency
            raise RuntimeError("OpenCV is required for sequential video ingest") from exc
        path = Path(video_path).expanduser().resolve(strict=True)
        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            raise ValueError(f"cannot open video: {path}")
        self.encoder.reset_visual_clock()
        interval = 1.0 / self.feature_fps
        next_sample_s = 0.0
        decoded = emitted = 0
        started = time.perf_counter()
        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                frame_index = decoded
                decoded += 1
                timestamp_s = frame_index / meta.fps
                if timestamp_s + 1e-9 < next_sample_s:
                    continue
                # Assign the actual sampled frame to its fixed feature-grid
                # cell. Small frame-rate jitter then cannot create artificial
                # gaps between otherwise adjacent 2 fps observations. If a
                # damaged stream skips a complete cell, the gap stays explicit.
                start_s, end_s = sampling_cell(timestamp_s, interval, meta.duration_s)
                next_sample_s = end_s
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                feature = self.encoder.encode_images([Image.fromarray(rgb)])[0]
                emitted += 1
                yield SnAGFeatureObservation(
                    np.asarray(feature, dtype=np.float32), start_s, end_s,
                    f"{meta.video_id}:frame:{frame_index}", frame_index,
                )
        finally:
            capture.release()
            self.last_stats = SnAGIngestStats(
                decoded, emitted, 1, time.perf_counter() - started, meta.duration_s,
            )


class FixtureFeatureExtractor:
    """Ordered fixture extractor used to audit orchestration without video IO."""

    def __init__(self, observations: list[SnAGFeatureObservation]) -> None:
        self.observations = list(observations)
        self.last_stats: SnAGIngestStats | None = None

    def iter_video(self, video_path: Path | str, meta: VideoMeta) -> Iterator[SnAGFeatureObservation]:
        del video_path
        previous = -1.0
        started = time.perf_counter()
        for observation in self.observations:
            if observation.t_start_s < previous or observation.t_end_s > meta.duration_s:
                raise ValueError("fixture features violate sequential provenance")
            previous = observation.t_start_s
            yield observation
        self.last_stats = SnAGIngestStats(
            len(self.observations), len(self.observations), 1,
            time.perf_counter() - started, meta.duration_s,
        )
