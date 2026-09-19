"""Turn hierarchy-masked retrieval hits into bounded temporal candidates."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

from streamtimelens.retrieval.mmr import MMRChoice


@dataclass(frozen=True)
class RetrievalCandidate:
    candidate_id: str
    start_s: float
    end_s: float
    score: float
    contributing_card_ids: tuple[str, ...]
    raw_completeness: float
    diagnostics: dict[str, Any]

    @property
    def span(self) -> tuple[float, float]:
        return self.start_s, self.end_s


def _raw_completeness(rows: Sequence[dict[str, Any]]) -> float:
    available = 0
    expected = 0
    for row in rows:
        statuses = row.get("raw_ref_status") or {}
        refs = row.get("raw_ref_ids") or []
        expected += len(refs)
        available += sum(statuses.get(frame_id) == "available" for frame_id in refs)
    return available / expected if expected else 0.0


def _candidate_id(card_ids: Sequence[str], start_s: float, end_s: float) -> str:
    canonical = json.dumps(
        {"card_ids": list(card_ids), "start_s": start_s, "end_s": end_s},
        sort_keys=True, separators=(",", ":"), allow_nan=False,
    )
    return "candidate-" + hashlib.sha1(canonical.encode("utf-8")).hexdigest()[:16]


def _contiguous_group(anchor: dict[str, Any], rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    ordered = sorted(rows, key=lambda row: (float(row["t_start"]), float(row["t_end"]), str(row["id"])))
    selected = {str(anchor["id"])}
    anchor_index = next(index for index, row in enumerate(ordered) if row["id"] == anchor["id"])
    left = anchor_index - 1
    cursor_start = float(anchor["t_start"])
    while left >= 0:
        row = ordered[left]
        duration = max(1e-6, float(row["t_end"]) - float(row["t_start"]))
        gap = cursor_start - float(row["t_end"])
        if gap >= max(2.0, 0.1 * duration):
            break
        selected.add(str(row["id"]))
        cursor_start = float(row["t_start"])
        left -= 1
    right = anchor_index + 1
    cursor_end = float(anchor["t_end"])
    while right < len(ordered):
        row = ordered[right]
        duration = max(1e-6, float(row["t_end"]) - float(row["t_start"]))
        gap = float(row["t_start"]) - cursor_end
        if gap >= max(2.0, 0.1 * duration):
            break
        selected.add(str(row["id"]))
        cursor_end = max(cursor_end, float(row["t_end"]))
        right += 1
    return [row for row in ordered if str(row["id"]) in selected]


def expand_temporal_candidates(
    choices: Iterable[MMRChoice],
    card_rows: Iterable[dict[str, Any]],
    *,
    t_q: float,
) -> list[RetrievalCandidate]:
    if t_q <= 0:
        raise ValueError("query arrival time must be positive")
    rows = list(card_rows)
    by_id = {str(row.get("id", "")): row for row in rows}
    if "" in by_id or len(by_id) != len(rows):
        raise ValueError("candidate card rows need unique non-empty IDs")
    visible = [
        row for row in rows
        if float(row["t_start"]) < t_q and float(row["t_end"]) <= t_q + 1e-6
    ]
    timeline = sorted(
        visible, key=lambda row: (float(row["t_start"]), float(row["t_end"]), str(row["id"])),
    )
    positions = {str(row["id"]): index for index, row in enumerate(timeline)}
    candidates: list[RetrievalCandidate] = []
    seen: set[tuple[str, ...]] = set()
    for choice in choices:
        anchor = by_id.get(choice.card_id)
        if anchor is None:
            raise ValueError(f"selected card is missing: {choice.card_id}")
        if choice.card_id not in positions:
            continue
        child_ids = tuple(str(value) for value in anchor.get("child_ids") or [])
        if child_ids:
            components = [by_id[child_id] for child_id in child_ids if child_id in positions]
            components = components or [anchor]
            if anchor in components:
                expanded = components
            else:
                expanded = []
                for component in components:
                    expanded.extend(_contiguous_group(component, components))
                expanded = list({str(row["id"]): row for row in expanded}.values())
        else:
            index = positions[choice.card_id]
            neighborhood = timeline[max(0, index - 1):min(len(timeline), index + 2)]
            expanded = _contiguous_group(anchor, neighborhood)
        expanded = sorted(
            expanded, key=lambda row: (float(row["t_start"]), float(row["t_end"]), str(row["id"])),
        )
        identifiers = tuple(str(row["id"]) for row in expanded)
        if not identifiers or identifiers in seen:
            continue
        seen.add(identifiers)
        start = min(float(row["t_start"]) for row in expanded)
        end = min(t_q, max(float(row["t_end"]) for row in expanded))
        if start >= end:
            continue
        completeness = _raw_completeness(expanded)
        candidates.append(RetrievalCandidate(
            _candidate_id(identifiers, start, end), start, end, choice.mmr_score,
            identifiers, completeness,
            {
                "source_card_id": choice.card_id,
                "rank_score": choice.rank_score,
                "mmr_score": choice.mmr_score,
                "semantic_redundancy": choice.semantic_redundancy,
                "temporal_redundancy": choice.temporal_redundancy,
            },
        ))
    return sorted(candidates, key=lambda item: (-item.score, item.start_s, item.candidate_id))
