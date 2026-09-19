"""Causal trigger proposals with explicit ablation and rule provenance."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal


TriggerMode = Literal["periodic", "cut-motion", "semantic", "joint"]


@dataclass(frozen=True)
class TriggerProposal:
    timestamp_s: float
    score: float
    reasons: tuple[str, ...]

    @property
    def t(self) -> float:
        return self.timestamp_s


class JointTrigger:
    """Combine lite, semantic and age signals without invoking a writer.

    ``periodic`` reads age only; ``cut-motion`` reads lite+age;
    ``semantic`` reads semantic+age; and ``joint`` reads all three.  Max-gap
    is an unconditional safety rule, while hard-cut and minimum-gap remain
    independently auditable decisions.
    """

    def __init__(
        self,
        mode: TriggerMode = "joint",
        *,
        threshold: float = 0.65,
        periodic_interval_s: float = 4.0,
        minimum_gap_s: float = 1.0,
        max_gap_s: float = 60.0,
        lite_weight: float = 0.45,
        semantic_weight: float = 0.35,
        age_weight: float = 0.20,
    ) -> None:
        if mode not in ("periodic", "cut-motion", "semantic", "joint"):
            raise ValueError(f"unsupported trigger mode: {mode}")
        numeric = (threshold, periodic_interval_s, minimum_gap_s, max_gap_s,
                   lite_weight, semantic_weight, age_weight)
        if any(not math.isfinite(float(value)) for value in numeric):
            raise ValueError("trigger configuration must be finite")
        if threshold < 0 or periodic_interval_s <= 0 or minimum_gap_s < 0 or max_gap_s <= 0:
            raise ValueError("invalid trigger timing or threshold")
        if max_gap_s < minimum_gap_s or min(lite_weight, semantic_weight, age_weight) < 0:
            raise ValueError("invalid trigger gaps or weights")
        if mode == "joint" and lite_weight + semantic_weight + age_weight <= 0:
            raise ValueError("joint trigger needs a positive signal weight")
        self.mode = mode
        self.threshold = float(threshold)
        self.periodic_interval_s = float(periodic_interval_s)
        self.minimum_gap_s = float(minimum_gap_s)
        self.max_gap_s = float(max_gap_s)
        self.lite_weight = float(lite_weight)
        self.semantic_weight = float(semantic_weight)
        self.age_weight = float(age_weight)
        self._first_timestamp: float | None = None
        self._last_timestamp: float | None = None
        self._last_proposal: float | None = None
        self._last_write: float | None = None
        self.last_decision_reasons: tuple[str, ...] = ()

    @staticmethod
    def _signal(name: str, value: object | None) -> float:
        if value is None:
            return 0.0
        numeric = float(value)
        if not math.isfinite(numeric):
            raise ValueError(f"{name} trigger signal must be finite")
        return max(0.0, numeric)

    def propose(
        self,
        timestamp_s: float,
        *,
        lite_score: object | None = None,
        semantic_score: object | None = None,
        hard_cut: bool = False,
    ) -> TriggerProposal | None:
        timestamp = float(timestamp_s)
        if not math.isfinite(timestamp) or timestamp < 0:
            raise ValueError("trigger timestamp must be finite and non-negative")
        if self._last_timestamp is not None and timestamp < self._last_timestamp:
            raise ValueError("trigger timestamps must be monotonic")
        self._last_timestamp = timestamp
        if self._first_timestamp is None:
            self._first_timestamp = timestamp

        age_origin = self._last_write if self._last_write is not None else self._first_timestamp
        age = timestamp - age_origin
        if age + 1e-9 >= self.max_gap_s:
            proposal = TriggerProposal(timestamp, 1.0, ("rule:max_gap", "signal:age"))
            self._last_proposal = timestamp
            self.last_decision_reasons = proposal.reasons
            return proposal

        since_proposal = math.inf if self._last_proposal is None else timestamp - self._last_proposal
        if since_proposal + 1e-9 < self.minimum_gap_s:
            self.last_decision_reasons = ("rule:minimum_gap",)
            return None

        if self.mode == "periodic":
            due = self._last_proposal is None or since_proposal + 1e-9 >= self.periodic_interval_s
            if not due:
                self.last_decision_reasons = ("rule:periodic_not_due",)
                return None
            proposal = TriggerProposal(timestamp, min(1.0, age / self.max_gap_s), ("rule:periodic", "signal:age"))
        else:
            # Convert only the signals enabled by this ablation.
            lite = self._signal("lite", lite_score) if self.mode in ("cut-motion", "joint") else 0.0
            semantic = self._signal("semantic", semantic_score) if self.mode in ("semantic", "joint") else 0.0
            age_score = min(1.0, max(0.0, age / self.max_gap_s))
            if self.mode == "cut-motion":
                score, reasons = lite, ("signal:lite",)
            elif self.mode == "semantic":
                score, reasons = semantic, ("signal:semantic",)
            else:
                weighted = [(lite, self.lite_weight, "signal:lite"), (age_score, self.age_weight, "signal:age")]
                if semantic_score is not None:
                    weighted.insert(1, (semantic, self.semantic_weight, "signal:semantic"))
                weight = sum(item[1] for item in weighted)
                score = sum(value * signal_weight for value, signal_weight, _ in weighted) / weight
                reasons = tuple(reason for _, _, reason in weighted)
            hard_cut_enabled = self.mode in ("cut-motion", "joint") and bool(hard_cut)
            if not hard_cut_enabled and score < self.threshold:
                self.last_decision_reasons = ("rule:below_threshold", *reasons)
                return None
            proposal_reasons = (("rule:hard_cut",) if hard_cut_enabled else ()) + reasons
            proposal = TriggerProposal(timestamp, max(1.0, score) if hard_cut_enabled else score, proposal_reasons)

        self._last_proposal = timestamp
        self.last_decision_reasons = proposal.reasons
        return proposal

    def mark_written(self, timestamp_s: float) -> None:
        timestamp = float(timestamp_s)
        if self._last_timestamp is not None and timestamp > self._last_timestamp + 1e-9:
            raise ValueError("a write cannot be marked in the observer's future")
        if self._last_write is not None and timestamp < self._last_write:
            raise ValueError("write timestamps must be monotonic")
        self._last_write = timestamp


class PeriodicTrigger:
    """Backward-compatible adapter for the original P0 runner API."""

    def __init__(self, interval_s: float = 4.0, max_gap_s: float = 60.0) -> None:
        self._trigger = JointTrigger(
            "periodic", periodic_interval_s=interval_s, minimum_gap_s=0,
            max_gap_s=max_gap_s,
        )

    def propose(self, timestamp_s: float) -> str | None:
        proposal = self._trigger.propose(timestamp_s)
        if proposal is None:
            return None
        return "max_gap" if "rule:max_gap" in proposal.reasons else "periodic"

    def mark_written(self, timestamp_s: float) -> None:
        self._trigger.mark_written(timestamp_s)
