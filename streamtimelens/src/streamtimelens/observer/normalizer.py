"""Past-only online normalization for observer signals."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class NormalizedSignal:
    value: float
    z_score: float | None
    ready: bool
    active: bool
    reference_mean: float | None
    reference_variance: float | None


@dataclass
class _Moment:
    count: int = 0
    mean: float = 0.0
    variance: float = 0.0


class OnlineNormalizer:
    """EWMA normalizer whose output at ``t`` only uses state before ``t``."""

    format_version = 1

    def __init__(
        self,
        *,
        alpha: float = 0.05,
        warmup_samples: int = 8,
        epsilon: float = 1e-6,
        fixed_thresholds: Mapping[str, float] | None = None,
        z_threshold: float = 2.0,
    ) -> None:
        if not 0 < alpha <= 1:
            raise ValueError("EWMA alpha must be in (0,1]")
        if warmup_samples < 1 or epsilon <= 0 or not math.isfinite(z_threshold):
            raise ValueError("invalid online normalizer configuration")
        thresholds = dict(fixed_thresholds or {})
        if any(not math.isfinite(float(value)) for value in thresholds.values()):
            raise ValueError("fixed thresholds must be finite")
        self.alpha = float(alpha)
        self.warmup_samples = int(warmup_samples)
        self.epsilon = float(epsilon)
        self.fixed_thresholds = {str(key): float(value) for key, value in thresholds.items()}
        self.z_threshold = float(z_threshold)
        self._moments: dict[str, _Moment] = {}

    def observe_one(self, name: str, value: float) -> NormalizedSignal:
        numeric = float(value)
        if not name or not math.isfinite(numeric):
            raise ValueError("normalizer inputs must have a name and finite value")
        moment = self._moments.setdefault(name, _Moment())
        ready = moment.count >= self.warmup_samples
        reference_mean = moment.mean if moment.count else None
        reference_variance = moment.variance if moment.count else None
        z_score = None
        if ready:
            z_score = (numeric - moment.mean) / math.sqrt(moment.variance + self.epsilon)
            active = z_score >= self.z_threshold
        else:
            threshold = self.fixed_thresholds.get(name, math.inf)
            active = numeric >= threshold

        # Updating happens strictly after the current output has been formed.
        if moment.count == 0:
            moment.mean = numeric
            moment.variance = 0.0
        else:
            delta = numeric - moment.mean
            moment.mean += self.alpha * delta
            moment.variance = (1.0 - self.alpha) * (
                moment.variance + self.alpha * delta * delta
            )
        moment.count += 1
        return NormalizedSignal(
            numeric, z_score, ready, active, reference_mean, reference_variance,
        )

    def observe(self, values: Mapping[str, float | None]) -> dict[str, NormalizedSignal]:
        return {
            name: self.observe_one(name, value)
            for name, value in values.items()
            if value is not None
        }

    def snapshot(self) -> dict[str, object]:
        return {
            "format_version": self.format_version,
            "alpha": self.alpha,
            "warmup_samples": self.warmup_samples,
            "epsilon": self.epsilon,
            "fixed_thresholds": dict(sorted(self.fixed_thresholds.items())),
            "z_threshold": self.z_threshold,
            "moments": {
                name: {"count": state.count, "mean": state.mean, "variance": state.variance}
                for name, state in sorted(self._moments.items())
            },
        }

    @classmethod
    def restore(cls, snapshot: Mapping[str, object]) -> "OnlineNormalizer":
        if snapshot.get("format_version") != cls.format_version:
            raise ValueError("unsupported normalizer snapshot version")
        normalizer = cls(
            alpha=float(snapshot["alpha"]),
            warmup_samples=int(snapshot["warmup_samples"]),
            epsilon=float(snapshot["epsilon"]),
            fixed_thresholds=dict(snapshot.get("fixed_thresholds", {})),
            z_threshold=float(snapshot["z_threshold"]),
        )
        raw_moments = snapshot.get("moments", {})
        if not isinstance(raw_moments, Mapping):
            raise ValueError("normalizer moments must be a mapping")
        for name, raw in raw_moments.items():
            if not isinstance(raw, Mapping):
                raise ValueError("invalid normalizer moment")
            moment = _Moment(int(raw["count"]), float(raw["mean"]), float(raw["variance"]))
            if moment.count < 0 or not math.isfinite(moment.mean) or not math.isfinite(moment.variance) or moment.variance < 0:
                raise ValueError("invalid normalizer moment")
            normalizer._moments[str(name)] = moment
        return normalizer
