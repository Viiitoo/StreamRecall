"""Deterministic, budget-audited frame sampling primitives.

This module intentionally has no model or video-decoder dependency.  That keeps
the sampling decision reproducible from the metadata saved in ``sampling.jsonl``.
"""

from __future__ import annotations

from bisect import bisect_left
from dataclasses import asdict, dataclass
from math import floor, isfinite
from typing import Any, Mapping, Sequence


class SamplingError(ValueError):
    """Raised when a sampling plan would be ambiguous or invalid."""


@dataclass(frozen=True)
class VideoMetadata:
    """Video timing information used to map requested seconds to frame indices.

    ``frame_timestamps_sec`` is optional but should be supplied for VFR assets.
    When it is available, its values—not an average FPS—are used for both frame
    lookup and the timestamps presented to the model.
    """

    fps: float
    frame_count: int
    duration_sec: float | None = None
    frame_timestamps_sec: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        if not isfinite(self.fps) or self.fps <= 0:
            raise SamplingError("fps must be a positive finite number")
        if self.frame_count <= 0:
            raise SamplingError("frame_count must be positive")
        duration = self.duration_sec if self.duration_sec is not None else self.frame_count / self.fps
        if not isfinite(duration) or duration <= 0:
            raise SamplingError("duration_sec must be a positive finite number")
        object.__setattr__(self, "duration_sec", float(duration))
        if self.frame_timestamps_sec is not None:
            timestamps = tuple(float(value) for value in self.frame_timestamps_sec)
            if len(timestamps) != self.frame_count:
                raise SamplingError("frame_timestamps_sec length must equal frame_count")
            if any(not isfinite(value) or value < 0 for value in timestamps):
                raise SamplingError("frame timestamps must be finite and non-negative")
            if any(right <= left for left, right in zip(timestamps, timestamps[1:])):
                raise SamplingError("frame timestamps must be strictly increasing")
            object.__setattr__(self, "frame_timestamps_sec", timestamps)

    @classmethod
    def from_mapping(cls, value: "VideoMetadata | Mapping[str, Any]") -> "VideoMetadata":
        if isinstance(value, cls):
            return value
        return cls(
            fps=float(value["fps"]),
            frame_count=int(value.get("frame_count", value.get("total_frames"))),
            duration_sec=value.get("duration_sec", value.get("duration")),
            frame_timestamps_sec=(
                tuple(value["frame_timestamps_sec"])
                if value.get("frame_timestamps_sec") is not None
                else None
            ),
        )

    def index_for_timestamp(self, timestamp_sec: float) -> int:
        """Return the first frame at/after ``timestamp_sec`` within ``[0, duration)``."""
        timestamp_sec = min(max(float(timestamp_sec), 0.0), self.duration_sec - 1e-12)
        if self.frame_timestamps_sec is not None:
            return min(bisect_left(self.frame_timestamps_sec, timestamp_sec), self.frame_count - 1)
        return min(max(floor(timestamp_sec * self.fps), 0), self.frame_count - 1)

    def timestamp_for_index(self, index: int) -> float:
        if not 0 <= index < self.frame_count:
            raise SamplingError(f"frame index {index} outside [0, {self.frame_count})")
        if self.frame_timestamps_sec is not None:
            return self.frame_timestamps_sec[index]
        # Clamp protects against a container duration that is shorter than N / FPS.
        return min(index / self.fps, self.duration_sec - 1e-12)


@dataclass(frozen=True)
class SamplingPlan:
    """An immutable sampling decision and its post-preprocessing token audit."""

    video_id: str
    duration_sec: float
    timestamps_sec: tuple[float, ...]
    frame_indices: tuple[int, ...]
    requested_budget: int
    actual_visual_tokens: int | None
    stage: str
    seed: int

    def __post_init__(self) -> None:
        if not self.video_id:
            raise SamplingError("video_id must be non-empty")
        if self.requested_budget <= 0:
            raise SamplingError("requested_budget must be positive")
        if len(self.timestamps_sec) != len(self.frame_indices) or not self.frame_indices:
            raise SamplingError("timestamps and frame indices must be equally sized and non-empty")
        if any(not isfinite(value) or value < 0 or value >= self.duration_sec for value in self.timestamps_sec):
            raise SamplingError("timestamps must be finite and inside [0, duration)")
        if any(right <= left for left, right in zip(self.timestamps_sec, self.timestamps_sec[1:])):
            raise SamplingError("timestamps must be strictly increasing")
        if any(right <= left for left, right in zip(self.frame_indices, self.frame_indices[1:])):
            raise SamplingError("frame indices must be strictly increasing")
        if self.actual_visual_tokens is not None:
            if self.actual_visual_tokens < 0:
                raise SamplingError("actual_visual_tokens cannot be negative")
            if self.actual_visual_tokens > self.requested_budget:
                raise SamplingError("actual visual tokens exceed requested budget")

    def to_record(self) -> dict[str, Any]:
        """Return a JSON-safe, deterministic representation."""
        return asdict(self)


def _uniform_indices(metadata: VideoMetadata, count: int) -> tuple[int, ...]:
    """Map a uniform `[0, duration)` grid to unique frame indices and fill gaps."""
    count = min(max(1, int(count)), metadata.frame_count)
    desired = [metadata.duration_sec * position / count for position in range(count)]
    selected = sorted({metadata.index_for_timestamp(timestamp) for timestamp in desired})
    if len(selected) == count:
        return tuple(selected)

    # A VFR stream or a metadata duration mismatch can collapse a grid. Replenish
    # by repeatedly selecting the frame furthest from already selected timestamps;
    # unlike index-space filling this preserves temporal coverage for VFR assets.
    selected = sorted(set(selected))
    available = set(range(metadata.frame_count)) - set(selected)
    while len(selected) < count and available:
        def distance_to_selection(index: int) -> float:
            timestamp = metadata.timestamp_for_index(index)
            return min(abs(timestamp - metadata.timestamp_for_index(other)) for other in selected)

        # ``max`` keeps the lower index on exact ties via the negative index key.
        chosen = max(available, key=lambda index: (distance_to_selection(index), -index))
        selected.append(chosen)
        selected.sort()
        available.remove(chosen)
    return tuple(sorted(selected)[:count])


def uniform_plan(
    video_metadata: VideoMetadata | Mapping[str, Any],
    budget: int,
    seed: int,
    *,
    video_id: str = "unknown",
    stage: str = "uniform",
    max_frames: int | None = None,
) -> SamplingPlan:
    """Build a deterministic uniform plan.

    ``budget`` is recorded as a visual-token limit and is never interpreted as an
    observed token count.  ``max_frames`` is the caller's conservative allocation
    before preprocessing; the runner subsequently measures the actual total.
    """
    if budget <= 0:
        raise SamplingError("budget must be positive")
    metadata = VideoMetadata.from_mapping(video_metadata)
    count = metadata.frame_count if max_frames is None else max_frames
    indices = _uniform_indices(metadata, count)
    timestamps = tuple(metadata.timestamp_for_index(index) for index in indices)
    return SamplingPlan(
        video_id=str(video_id),
        duration_sec=metadata.duration_sec,
        timestamps_sec=timestamps,
        frame_indices=indices,
        requested_budget=int(budget),
        actual_visual_tokens=None,
        stage=stage,
        seed=int(seed),
    )


def build_plan(
    video_metadata: VideoMetadata | Mapping[str, Any],
    budget: int,
    seed: int,
    **kwargs: Any,
) -> SamplingPlan:
    """P0 public plan-building interface; currently selects uniform sampling."""
    return uniform_plan(video_metadata, budget, seed, **kwargs)


def with_actual_visual_tokens(plan: SamplingPlan, actual_visual_tokens: int) -> SamplingPlan:
    """Create the audited version of a plan, rejecting budget violations."""
    return SamplingPlan(**{**plan.to_record(), "actual_visual_tokens": int(actual_visual_tokens)})


def stable_plan_json(plan: SamplingPlan) -> str:
    """Canonical JSON used by determinism tests and manifests."""
    import json

    return json.dumps(plan.to_record(), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
