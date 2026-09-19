"""Budget-aware raw-frame allocation around evidence-card boundaries."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable

from streamtimelens.memory.cards import EvidenceCard
from streamtimelens.memory.raw_cache import CachedFrame, RawFrameCache
from streamtimelens.protocol.types import FramePacket


@dataclass(frozen=True)
class BoundaryCandidate:
    frame_id: str
    timestamp_s: float
    novelty: float = 0.0

    def __post_init__(self) -> None:
        if (not self.frame_id or not math.isfinite(self.timestamp_s) or self.timestamp_s < 0 or
                not math.isfinite(self.novelty) or self.novelty < 0):
            raise ValueError("invalid boundary-frame candidate")


@dataclass(frozen=True)
class BoundaryAllocation:
    selected_frame_ids: tuple[str, ...]
    left_frame_ids: tuple[str, ...]
    right_frame_ids: tuple[str, ...]
    internal_frame_ids: tuple[str, ...]
    rejected_frame_ids: tuple[str, ...]
    bytes_admitted: int

    @property
    def left_hit(self) -> bool:
        return bool(self.left_frame_ids)

    @property
    def right_hit(self) -> bool:
        return bool(self.right_frame_ids)

    @property
    def both_hit(self) -> bool:
        return self.left_hit and self.right_hit


def _as_candidate(value: Any) -> BoundaryCandidate:
    if isinstance(value, BoundaryCandidate):
        return value
    ref = getattr(value, "ref", value)
    if isinstance(ref, CachedFrame):
        return BoundaryCandidate(
            ref.frame_id, ref.timestamp_s, float(getattr(value, "novelty", 0.0)),
        )
    if isinstance(ref, FramePacket):
        return BoundaryCandidate(
            f"{ref.frame_index:09d}.jpg", ref.timestamp_s, float(getattr(value, "novelty", 0.0)),
        )
    raise TypeError("boundary candidates must be cached frames, packets, or segment frames")


class BoundaryFrameAllocator:
    """Select <=4 frames per endpoint and <=2 novel interior frames."""

    def __init__(
        self,
        store: RawFrameCache,
        *,
        boundary_window_s: float = 2.0,
        max_per_boundary: int = 4,
        max_internal: int = 2,
    ) -> None:
        if boundary_window_s < 0 or max_per_boundary < 0 or max_internal < 0:
            raise ValueError("boundary allocator limits must be non-negative")
        self.store = store
        self.boundary_window_s = float(boundary_window_s)
        self.max_per_boundary = int(max_per_boundary)
        self.max_internal = int(max_internal)

    def allocate(
        self,
        card: EvidenceCard,
        candidates: Iterable[Any],
        *,
        budget_bytes: int | None = None,
    ) -> BoundaryAllocation:
        if budget_bytes is not None and budget_bytes < 0:
            raise ValueError("boundary allocation budget must be non-negative")
        by_id = {item.frame_id: item for item in map(_as_candidate, candidates)}
        for frame_id in by_id:
            if frame_id not in self.store:
                raise KeyError(f"boundary candidate is absent from shared store: {frame_id}")
        values = list(by_id.values())
        left = sorted(
            (item for item in values if abs(item.timestamp_s - card.t_start) <= self.boundary_window_s + 1e-9),
            key=lambda item: (abs(item.timestamp_s - card.t_start), -item.novelty, item.timestamp_s, item.frame_id),
        )[:self.max_per_boundary]
        right = sorted(
            (item for item in values if abs(item.timestamp_s - card.t_end) <= self.boundary_window_s + 1e-9),
            key=lambda item: (abs(item.timestamp_s - card.t_end), -item.novelty, item.timestamp_s, item.frame_id),
        )[:self.max_per_boundary]
        boundary_ids = {item.frame_id for item in (*left, *right)}
        internal = sorted(
            (item for item in values if card.t_start <= item.timestamp_s <= card.t_end and
             item.frame_id not in boundary_ids),
            key=lambda item: (-item.novelty, item.timestamp_s, item.frame_id),
        )[:self.max_internal]
        proposed = {item.frame_id: item for item in (*left, *right, *internal)}

        def coverage_gap(item: BoundaryCandidate) -> float:
            others = [abs(item.timestamp_s - other.timestamp_s) for other in values if other.frame_id != item.frame_id]
            return min(others) if others else float("inf")

        roles = {
            frame_id: {
                "left": any(item.frame_id == frame_id for item in left),
                "right": any(item.frame_id == frame_id for item in right),
                "internal": any(item.frame_id == frame_id for item in internal),
            }
            for frame_id in proposed
        }
        ranked = sorted(
            proposed.values(),
            key=lambda item: (
                -max(
                    card.left_uncertainty_s if roles[item.frame_id]["left"] else 0.0,
                    card.right_uncertainty_s if roles[item.frame_id]["right"] else 0.0,
                ),
                -item.novelty,
                -coverage_gap(item),
                item.timestamp_s,
                item.frame_id,
            ),
        )
        limit = math.inf if budget_bytes is None else int(budget_bytes)
        admitted: list[str] = []
        rejected: list[str] = []
        admitted_content: set[str] = set()
        used = 0
        metadata = self.store.metadata(set(proposed))
        for item in ranked:
            content_id = str(metadata[item.frame_id]["sha256"])
            already_card_owned = any(
                str(owner).startswith("card:") for owner in metadata[item.frame_id].get("owners", [])
            )
            cost = 0 if content_id in admitted_content or already_card_owned else len(self.store.get(item.frame_id))
            if used + cost <= limit:
                admitted.append(item.frame_id)
                admitted_content.add(content_id)
                used += cost
                self.store.retain(item.frame_id, f"card:{card.id}")
            else:
                rejected.append(item.frame_id)

        admitted_set = set(admitted)
        left_ids = tuple(item.frame_id for item in left if item.frame_id in admitted_set)
        right_ids = tuple(item.frame_id for item in right if item.frame_id in admitted_set)
        internal_ids = tuple(item.frame_id for item in internal if item.frame_id in admitted_set)
        ordered_ids = tuple(sorted(admitted, key=lambda frame_id: (by_id[frame_id].timestamp_s, frame_id)))
        card.raw_ref_ids = list(ordered_ids)
        card.raw_ref_status = {
            frame_id: ("available" if frame_id in admitted_set else "budget_rejected")
            for frame_id in proposed
        }
        card.boundary_cache = {
            "left_frame_ids": list(left_ids), "right_frame_ids": list(right_ids),
            "internal_frame_ids": list(internal_ids), "left_hit": bool(left_ids),
            "right_hit": bool(right_ids), "both_hit": bool(left_ids and right_ids),
            "budget_bytes": None if budget_bytes is None else int(budget_bytes),
            "bytes_admitted": used,
        }
        card.serializable()
        return BoundaryAllocation(
            ordered_ids, left_ids, right_ids, internal_ids, tuple(sorted(rejected)), used,
        )

    def release_card(self, card: EvidenceCard, *, status: str = "evicted") -> None:
        owner = f"card:{card.id}"
        for frame_id in tuple(card.raw_ref_ids):
            self.store.release(frame_id, owner)
            card.mark_raw_ref(frame_id, status)
        card.serializable()
