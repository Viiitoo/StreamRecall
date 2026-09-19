"""Frozen rule-based confidence and NOT_FOUND calibration."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal


ParseFeature = Literal["ok", "fallback", "failure", "not_attempted"]


@dataclass(frozen=True)
class ConfidenceFeatures:
    top1_score: float
    top1_top2_margin: float
    raw_completeness: float
    parse_status: ParseFeature

    def __post_init__(self) -> None:
        if not math.isfinite(self.top1_score) or not math.isfinite(self.top1_top2_margin):
            raise ValueError("confidence scores must be finite")
        if self.top1_top2_margin < 0 or not 0 <= self.raw_completeness <= 1:
            raise ValueError("confidence margin/completeness is invalid")
        if self.parse_status not in ("ok", "fallback", "failure", "not_attempted"):
            raise ValueError("unknown parse status")


@dataclass(frozen=True)
class ConfidenceConfig:
    top1_weight: float = 1.0
    margin_weight: float = 0.5
    raw_weight: float = 0.5
    parse_weight: float = 0.75
    bias: float = -0.5
    temperature: float = 1.0
    not_found_threshold: float = 0.20

    def __post_init__(self) -> None:
        values = (
            self.top1_weight, self.margin_weight, self.raw_weight,
            self.parse_weight, self.bias, self.temperature, self.not_found_threshold,
        )
        if any(not math.isfinite(value) for value in values):
            raise ValueError("confidence configuration must be finite")
        if min(self.top1_weight, self.margin_weight, self.raw_weight, self.parse_weight) < 0:
            raise ValueError("confidence feature weights cannot be negative")
        if self.temperature <= 0 or not 0 <= self.not_found_threshold <= 1:
            raise ValueError("confidence temperature/threshold is invalid")


@dataclass(frozen=True)
class ConfidenceResult:
    confidence: float
    not_found: bool
    logit: float
    features: ConfidenceFeatures


def calibrate_confidence(
    features: ConfidenceFeatures,
    config: ConfidenceConfig,
) -> ConfidenceResult:
    parse_value = {
        "ok": 1.0, "fallback": 0.0, "not_attempted": 0.0, "failure": -1.0,
    }[features.parse_status]
    logit = (
        config.bias + config.top1_weight * features.top1_score
        + config.margin_weight * features.top1_top2_margin
        + config.raw_weight * features.raw_completeness
        + config.parse_weight * parse_value
    ) / config.temperature
    if logit >= 0:
        confidence = 1.0 / (1.0 + math.exp(-logit))
    else:
        exponential = math.exp(logit)
        confidence = exponential / (1.0 + exponential)
    return ConfidenceResult(confidence, confidence < config.not_found_threshold, logit, features)
