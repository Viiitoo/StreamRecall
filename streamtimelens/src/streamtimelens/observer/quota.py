"""Causal token-bucket admission for expensive writer calls."""

from __future__ import annotations

import math


class WriterQuota:
    """Capacity-one bucket plus an explicit short-video call ceiling.

    The ceiling uses ``max(1, floor(t / 60 * B_write) + 1)``.  A proposal is
    never queued: a failed call remains failed even if tokens accrue later.
    """

    capacity = 1.0

    def __init__(self, calls_per_minute: float, *, initial_tokens: float | None = None) -> None:
        calls = float(calls_per_minute)
        if not math.isfinite(calls) or calls < 0:
            raise ValueError("writer calls per minute must be finite and non-negative")
        if initial_tokens is None:
            initial_tokens = 1.0 if calls > 0 else 0.0
        initial = float(initial_tokens)
        if not math.isfinite(initial) or not 0 <= initial <= self.capacity:
            raise ValueError("initial writer tokens must be in [0,1]")
        if calls == 0 and initial > 0:
            raise ValueError("zero-rate writer quota cannot start with a token")
        self.calls_per_minute = calls
        self.rate = calls / 60.0
        self.initial_tokens = initial
        self.tokens = initial
        self.last_timestamp: float | None = None
        self.calls_consumed = 0
        self.last_reason = "not_attempted"

    def max_calls(self, duration_seen_s: float) -> int:
        duration = float(duration_seen_s)
        if not math.isfinite(duration) or duration < 0:
            raise ValueError("duration seen must be finite and non-negative")
        if self.calls_per_minute == 0:
            return 0
        return max(1, math.floor(duration / 60.0 * self.calls_per_minute) + 1)

    def _refill(self, timestamp_s: float) -> None:
        timestamp = float(timestamp_s)
        if not math.isfinite(timestamp) or timestamp < 0:
            raise ValueError("quota timestamp must be finite and non-negative")
        if self.last_timestamp is not None and timestamp < self.last_timestamp:
            raise ValueError("quota timestamps must be monotonic")
        if self.last_timestamp is not None:
            self.tokens = min(
                self.capacity,
                self.tokens + (timestamp - self.last_timestamp) * self.rate,
            )
        self.last_timestamp = timestamp

    def try_consume(self, timestamp_s: float) -> bool:
        self._refill(timestamp_s)
        if self.calls_consumed >= self.max_calls(timestamp_s):
            self.last_reason = "call_ceiling"
            return False
        if self.tokens + 1e-12 < 1.0:
            self.last_reason = "token_bucket"
            return False
        self.tokens = max(0.0, self.tokens - 1.0)
        self.calls_consumed += 1
        self.last_reason = "consumed"
        return True

    def consume(self, timestamp_s: float) -> bool:
        """Compatibility alias for the P0 runner."""
        return self.try_consume(timestamp_s)
