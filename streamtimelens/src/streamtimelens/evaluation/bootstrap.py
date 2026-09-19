"""Video-level paired bootstrap confidence intervals."""

from __future__ import annotations

import math
import random
from dataclasses import asdict, dataclass
from typing import Iterable


@dataclass(frozen=True)
class PairedMetricObservation:
    method: str
    budget_bytes: int
    video_id: str
    sample_id: str
    value: float

    def __post_init__(self) -> None:
        if not self.method or not self.video_id or not self.sample_id or self.budget_bytes <= 0:
            raise ValueError("invalid paired metric observation identity/budget")
        if not math.isfinite(self.value):
            raise ValueError("paired metric value must be finite")


@dataclass(frozen=True)
class BootstrapConfig:
    seed: int = 20260830
    resamples: int = 10_000
    confidence_level: float = 0.95
    unit: str = "video"

    def __post_init__(self) -> None:
        if self.resamples <= 0 or not 0 < self.confidence_level < 1 or self.unit != "video":
            raise ValueError("invalid paired bootstrap configuration")


def _percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def paired_video_bootstrap(
    observations: Iterable[PairedMetricObservation],
    *,
    method_a: str,
    method_b: str,
    config: BootstrapConfig = BootstrapConfig(),
) -> dict:
    if not method_a or not method_b or method_a == method_b:
        raise ValueError("paired bootstrap needs two distinct methods")
    rows = [row for row in observations if row.method in (method_a, method_b)]
    methods = {row.method for row in rows}
    if methods != {method_a, method_b}:
        raise ValueError("paired bootstrap is missing one method")
    budgets = {row.budget_bytes for row in rows}
    if len(budgets) != 1:
        raise ValueError("paired bootstrap only compares the same byte budget")
    by_method = {
        method: {(row.video_id, row.sample_id): row.value for row in rows if row.method == method}
        for method in (method_a, method_b)
    }
    if len(by_method[method_a]) != sum(row.method == method_a for row in rows) or (
        len(by_method[method_b]) != sum(row.method == method_b for row in rows)
    ):
        raise ValueError("paired bootstrap contains duplicate sample keys")
    if set(by_method[method_a]) != set(by_method[method_b]):
        raise ValueError("paired bootstrap samples are not paired")
    video_differences: dict[str, list[float]] = {}
    for key, left in by_method[method_a].items():
        video_differences.setdefault(key[0], []).append(left - by_method[method_b][key])
    videos = sorted(video_differences)
    if not videos:
        raise ValueError("paired bootstrap has no videos")
    per_video = {
        video: sum(values) / len(values) for video, values in video_differences.items()
    }
    observed = sum(per_video.values()) / len(per_video)
    generator = random.Random(config.seed)
    draws = []
    for _ in range(config.resamples):
        sample = [per_video[generator.choice(videos)] for _ in videos]
        draws.append(sum(sample) / len(sample))
    alpha = (1.0 - config.confidence_level) / 2.0
    return {
        "method_a": method_a, "method_b": method_b,
        "budget_bytes": next(iter(budgets)), "video_count": len(videos),
        "sample_count": len(by_method[method_a]), "delta_a_minus_b": observed,
        "ci_low": _percentile(draws, alpha),
        "ci_high": _percentile(draws, 1.0 - alpha),
        "configuration": asdict(config),
    }
