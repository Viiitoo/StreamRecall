"""Deterministic cross-candidate sparse-frame allocation for Hybrid V3."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from streamtimelens.protocol.snapshot import SnapshotReader
from streamtimelens.retrieval.frame_candidates import FrameCandidate


HIERARCHICAL_ALLOCATOR_V1 = "hierarchical_4_4_3_3_1_1_v1"
BOUNDARY_ANCHORED_ALLOCATOR_V2 = "boundary_anchored_v2"
_BOUNDARY_TOLERANCE_S = 5e-4 + 1e-9


@dataclass(frozen=True)
class MultiCandidateBundle:
    candidates: tuple[FrameCandidate, ...]
    selected_frame_refs: tuple[str, ...]
    candidate_frame_refs: dict[str, tuple[str, ...]]
    allocation_reason: dict[str, object]

    def __post_init__(self) -> None:
        if len(set(self.selected_frame_refs)) != len(self.selected_frame_refs):
            raise ValueError("selected frame refs must be unique")
        selected = set(self.selected_frame_refs)
        candidate_ids = {candidate.candidate_id for candidate in self.candidates}
        if set(self.candidate_frame_refs) - candidate_ids:
            raise ValueError("frame allocation contains an unknown candidate")
        if any(not set(refs) <= selected for refs in self.candidate_frame_refs.values()):
            raise ValueError("candidate frame refs must be selected frame refs")


def _target_counts(candidate_count: int, budget: int) -> list[int]:
    """Produce the documented 4/4/3/3/1/1 layout at six/16."""
    targets = [0] * candidate_count
    remaining = budget
    for index in range(candidate_count):
        if remaining == 0:
            return targets
        targets[index] = 1
        remaining -= 1
    for index in range(min(4, candidate_count)):
        for _ in range(2):
            if remaining == 0:
                return targets
            targets[index] += 1
            remaining -= 1
    for index in range(min(2, candidate_count)):
        if remaining == 0:
            return targets
        targets[index] += 1
        remaining -= 1
    return targets


def _best_ref(candidate: FrameCandidate) -> str:
    if candidate.retrieved_frames:
        return min(candidate.retrieved_frames, key=lambda item: (item.rank, item.frame_ref)).frame_ref
    if candidate.hit_refs:
        return candidate.hit_refs[0]
    return candidate.frame_refs[0]


def _role_refs(
    candidate: FrameCandidate,
    metadata: dict[str, dict[str, Any]],
    target: int,
) -> list[tuple[str, str]]:
    ordered = sorted(
        candidate.frame_refs,
        key=lambda ref: (float(metadata[ref]["timestamp_s"]), int(metadata[ref]["frame_index"]), ref),
    )
    middle_s = (candidate.start_s + candidate.end_s) / 2.0
    middle = min(
        ordered,
        key=lambda ref: (abs(float(metadata[ref]["timestamp_s"]) - middle_s), ref),
    )
    best = _best_ref(candidate)
    roles = {
        1: [(best, "best")],
        2: [(best, "best"), (middle, "middle")],
        3: [(ordered[0], "left"), (best, "best"), (ordered[-1], "right")],
    }.get(target, [
        (ordered[0], "left"), (best, "best"), (middle, "middle"),
        (ordered[-1], "right"),
    ])
    result = []
    seen = set()
    for ref, role in roles:
        if ref not in seen:
            result.append((ref, role))
            seen.add(ref)
    return result


def _next_uncovered_ref(
    candidate: FrameCandidate,
    metadata: dict[str, dict[str, Any]],
    chosen: set[str],
) -> str | None:
    available = [ref for ref in candidate.frame_refs if ref not in chosen]
    if not available:
        return None
    selected_times = [float(metadata[ref]["timestamp_s"]) for ref in chosen]
    if not selected_times:
        return _best_ref(candidate)
    return min(
        available,
        key=lambda ref: (
            -min(abs(float(metadata[ref]["timestamp_s"]) - value) for value in selected_times),
            float(metadata[ref]["timestamp_s"]), ref,
        ),
    )


def allocate_candidate_frames(
    candidates: Iterable[FrameCandidate],
    snapshot: SnapshotReader,
    *,
    max_frames: int,
    hard_limit: int = 16,
    strategy: str = HIERARCHICAL_ALLOCATOR_V1,
) -> MultiCandidateBundle:
    """Allocate snapshot-owned, past-only frames across candidate clusters."""
    if max_frames < 0 or hard_limit <= 0 or max_frames > hard_limit:
        raise ValueError(f"frame budget must be between 0 and hard limit {hard_limit}")
    if strategy not in (HIERARCHICAL_ALLOCATOR_V1, BOUNDARY_ANCHORED_ALLOCATOR_V2):
        raise ValueError(f"unknown frame allocation strategy: {strategy}")
    selected_candidates = tuple(candidates)
    metadata = snapshot.read_frame_metadata()
    t_q = float(snapshot.manifest.t_q)
    for candidate in selected_candidates:
        if not candidate.frame_refs:
            raise ValueError("candidate has no frame refs")
        for ref in candidate.frame_refs:
            if ref not in metadata:
                raise ValueError(f"candidate references a frame outside snapshot metadata: {ref}")
            timestamp_s = float(metadata[ref]["timestamp_s"])
            if timestamp_s > t_q + 1e-9:
                raise ValueError(f"candidate references a future frame: {ref}")

    assigned: dict[str, list[str]] = {candidate.candidate_id: [] for candidate in selected_candidates}
    roles: dict[str, dict[str, str]] = {candidate.candidate_id: {} for candidate in selected_candidates}
    globally_selected: set[str] = set()
    global_anchors: list[str] = []
    if strategy == BOUNDARY_ANCHORED_ALLOCATOR_V2 and max_frames:
        visible_refs = sorted(
            (
                ref for ref, row in metadata.items()
                if float(row["timestamp_s"]) <= t_q + 1e-9
            ),
            key=lambda ref: (
                float(metadata[ref]["timestamp_s"]), int(metadata[ref]["frame_index"]), ref,
            ),
        )
        if visible_refs:
            global_anchors.append(visible_refs[0])
            if max_frames > 1 and visible_refs[-1] != visible_refs[0]:
                global_anchors.append(visible_refs[-1])
        for ref in global_anchors:
            snapshot.frame_path(ref)
            globally_selected.add(ref)
        for candidate in selected_candidates:
            for ref, role in zip(global_anchors, ("global_earliest", "global_latest")):
                timestamp = float(metadata[ref]["timestamp_s"])
                if (
                    candidate.start_s - _BOUNDARY_TOLERANCE_S
                    <= timestamp
                    <= candidate.end_s + _BOUNDARY_TOLERANCE_S
                ):
                    assigned[candidate.candidate_id].append(ref)
                    roles[candidate.candidate_id][ref] = role

    targets = _target_counts(
        len(selected_candidates), max(0, max_frames - len(globally_selected)),
    )
    for candidate, target in zip(selected_candidates, targets):
        remaining_target = max(0, target - len(assigned[candidate.candidate_id]))
        for ref, role in _role_refs(candidate, metadata, remaining_target) if remaining_target else ():
            if len(globally_selected) >= max_frames:
                break
            if ref not in globally_selected:
                snapshot.frame_path(ref)  # Enforce the SnapshotReader allow-list now.
                globally_selected.add(ref)
            if ref not in assigned[candidate.candidate_id]:
                assigned[candidate.candidate_id].append(ref)
                roles[candidate.candidate_id][ref] = role

    # Missing/duplicate role slots transfer by score rank. Each turn chooses the
    # frame that covers the largest currently unrepresented local time gap.
    while len(globally_selected) < max_frames:
        progressed = False
        for candidate in selected_candidates:
            ref = _next_uncovered_ref(candidate, metadata, globally_selected)
            if ref is None:
                continue
            snapshot.frame_path(ref)
            globally_selected.add(ref)
            assigned[candidate.candidate_id].append(ref)
            roles[candidate.candidate_id][ref] = "transferred_uncovered_duration"
            progressed = True
            if len(globally_selected) >= max_frames:
                break
        if not progressed:
            break

    if strategy == BOUNDARY_ANCHORED_ALLOCATOR_V2:
        # The model receives one global chronological bundle. Preserve that fact
        # in each candidate's textual evidence mapping for shared local frames.
        for candidate in selected_candidates:
            local = set(candidate.frame_refs)
            for ref in globally_selected:
                if ref in local and ref not in assigned[candidate.candidate_id]:
                    assigned[candidate.candidate_id].append(ref)
                    roles[candidate.candidate_id][ref] = "shared_selected"

    ordered_refs = tuple(sorted(
        globally_selected,
        key=lambda ref: (float(metadata[ref]["timestamp_s"]), int(metadata[ref]["frame_index"]), ref),
    ))
    position = {ref: index for index, ref in enumerate(ordered_refs)}
    mapped = {
        candidate.candidate_id: tuple(sorted(refs, key=position.__getitem__))
        for candidate, refs in (
            (candidate, assigned[candidate.candidate_id]) for candidate in selected_candidates
        )
        if refs
    }
    allocation = {
        "strategy": strategy,
        "global_anchor_refs": global_anchors,
        "max_frames": max_frames,
        "hard_limit": hard_limit,
        "target_counts": {
            candidate.candidate_id: target
            for candidate, target in zip(selected_candidates, targets)
        },
        "actual_counts": {key: len(value) for key, value in mapped.items()},
        "roles": roles,
        "unused_budget": max_frames - len(ordered_refs),
    }
    return MultiCandidateBundle(selected_candidates, ordered_refs, mapped, allocation)
