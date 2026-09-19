"""Strictly byte-bounded event forest with soft and hard compaction."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any, Iterable

from streamtimelens.memory.budget import BudgetLedger
from streamtimelens.memory.card_store import CardStore
from streamtimelens.memory.cards import EvidenceCard
from streamtimelens.memory.eviction import UtilityEvictor
from streamtimelens.memory.merge_policy import MergeCandidate, MergeCandidateQueue
from streamtimelens.memory.parent_builder import build_parent_card
from streamtimelens.memory.raw_cache import RawFrameCache


def _json_size(value: Any) -> int:
    return len(json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8"))


@dataclass(frozen=True)
class MergeRecord:
    parent_id: str
    child_ids: tuple[str, ...]
    mode: str
    reason: dict[str, float]
    nodes_removed: int
    state_bytes_before: int
    state_bytes_after: int


class EventForest:
    """Own every query-visible card node under one exact logical budget."""

    def __init__(
        self, limit_bytes: int, *, raw_cache: RawFrameCache | None = None,
        ledger: BudgetLedger | None = None, merge_queue: MergeCandidateQueue | None = None,
        evictor: UtilityEvictor | None = None, debug: bool = False,
        require_embeddings: bool = False,
    ) -> None:
        if limit_bytes <= 0:
            raise ValueError("forest limit must be positive")
        self.limit_bytes = int(limit_bytes)
        self.raw = raw_cache if raw_cache is not None else RawFrameCache()
        self.ledger = ledger
        self.store = CardStore(self.raw)
        self.merge_queue = merge_queue or MergeCandidateQueue()
        self.evictor = evictor or UtilityEvictor()
        self.debug = bool(debug)
        self.require_embeddings = bool(require_embeddings)
        self._root_ids: list[str] = []
        self._current_stream_time = 0.0
        self.merge_trace: list[dict[str, Any]] = []
        self.eviction_trace: list[dict[str, Any]] = []

    @property
    def roots(self) -> tuple[EvidenceCard, ...]:
        return tuple(self.store.get(card_id) for card_id in self._ordered_root_ids())

    def _ordered_root_ids(self) -> list[str]:
        return sorted(self._root_ids, key=lambda card_id: CardStore.key(self.store.get(card_id)))

    @property
    def cards(self) -> tuple[EvidenceCard, ...]:
        return tuple(self.store)

    def _topology_bytes(self) -> int:
        return _json_size({
            "root_ids": self._ordered_root_ids(),
            "root_count": len(self._root_ids),
            "leaf_count": sum(card.level == 0 for card in self.store),
            "parent_count": sum(card.level > 0 for card in self.store),
        })

    def accounting_components(self) -> dict[str, int]:
        accounting = self.store.accounting()
        return {
            "card_payloads": accounting.card_payload_bytes,
            "card_embeddings": accounting.embedding_bytes,
            "card_raw_refs": accounting.raw_ref_index_bytes,
            "card_index": accounting.temporal_index_bytes,
            "forest_topology": self._topology_bytes(),
            "raw_frames": self.raw.byte_size,
        }

    @property
    def state_bytes(self) -> int:
        return sum(self.accounting_components().values())

    def _sync_ledger(self) -> None:
        if self.ledger is not None:
            other = {
                key: value for key, value in self.ledger.components.items()
                if key not in self.accounting_components()
            }
            self.ledger.replace_components({**other, **self.accounting_components()})

    def insert(self, card: EvidenceCard) -> None:
        self.store.insert(card)
        self._root_ids.append(card.id)
        self._current_stream_time = max(self._current_stream_time, card.t_end)
        self.merge_queue.invalidate(card.id)

    def _subtree_ids(self, card_id: str) -> set[str]:
        result: set[str] = set()
        pending = [card_id]
        while pending:
            current = pending.pop()
            if current in result:
                raise RuntimeError("parent/child cycle detected while compacting")
            result.add(current)
            pending.extend(self.store.get(current).child_ids)
        return result

    def _retain_parent_raw(self, parent: EvidenceCard) -> None:
        retained: list[str] = []
        for frame_id in parent.raw_ref_ids:
            if frame_id in self.raw and parent.raw_ref_status.get(frame_id) == "available":
                self.raw.retain(frame_id, f"card:{parent.id}")
                retained.append(frame_id)
            else:
                parent.raw_ref_status[frame_id] = "evicted"
        parent.raw_ref_ids = retained
        parent.serializable()
        self.store.touch(parent.id)

    def _delete_subtrees(self, child_ids: Iterable[str]) -> int:
        targets: set[str] = set()
        for child_id in child_ids:
            targets.update(self._subtree_ids(child_id))
        # Parent payloads go first; raw refcounts remain valid because the hard
        # parent retained every available blob before this deletion starts.
        for card_id in sorted(targets, key=lambda value: self.store.get(value).level, reverse=True):
            self.merge_queue.invalidate(card_id)
            self.store.delete(card_id)
        return len(targets)

    def delete_root(self, card_id: str) -> int:
        if card_id not in self._root_ids:
            raise ValueError(f"card is not a forest root: {card_id}")
        removed = self._delete_subtrees([card_id])
        self._root_ids.remove(card_id)
        return removed

    def _best_candidate(self) -> MergeCandidate | None:
        roots = self._ordered_root_ids()
        self.merge_queue.rebuild(self.store, roots)
        return self.merge_queue.pop_best(self.store, roots)

    def merge_best(self, *, force_hard: bool = False) -> MergeRecord | None:
        candidate = self._best_candidate()
        if candidate is None:
            return None
        children = [self.store.get(candidate.left_id), self.store.get(candidate.right_id)]
        before = self.state_bytes
        soft_parent = build_parent_card(children, hard=False)
        self.store.insert(soft_parent)
        soft_fits = not force_hard and self.state_bytes <= self.limit_bytes
        self.store.delete(soft_parent.id)
        if soft_fits:
            parent = soft_parent
            nodes_removed = 0
            self.store.insert(parent)
            self._retain_parent_raw(parent)
        else:
            targets = set().union(*(self._subtree_ids(child.id) for child in children))
            parent = build_parent_card(
                children, hard=True, compacted_child_count=len(targets),
            )
            self.store.insert(parent)
            self._retain_parent_raw(parent)
            nodes_removed = self._delete_subtrees(child.id for child in children)
        self._root_ids = [
            card_id for card_id in self._root_ids
            if card_id not in (candidate.left_id, candidate.right_id)
        ]
        self._root_ids.append(parent.id)
        self.merge_queue.invalidate(parent.id)
        after = self.state_bytes
        record = MergeRecord(
            parent.id, (candidate.left_id, candidate.right_id), parent.merge_mode,
            asdict(candidate.reason), nodes_removed, before, after,
        )
        self.merge_trace.append(asdict(record))
        if after <= self.limit_bytes:
            self._sync_ledger()
        if self.debug:
            self.assert_invariants(
                current_stream_time=self._current_stream_time,
                enforce_budget=after <= self.limit_bytes,
            )
        return record

    def enforce_budget(self, *, current_stream_time: float, external_bytes: int = 0) -> None:
        if external_bytes < 0 or external_bytes >= self.limit_bytes:
            raise ValueError("external forest bytes are invalid")
        effective_limit = self.limit_bytes - external_bytes
        # Compact topology before discarding evidence. A forced hard merge is
        # only retained when it actually reduces state.
        while self.state_bytes > effective_limit and len(self._root_ids) >= 2:
            before = self.state_bytes
            record = self.merge_best(force_hard=True)
            if record is None or self.state_bytes >= before:
                break
        # Within the eviction phase, frame evidence is always attempted before
        # deleting a card subtree.
        while self.state_bytes > effective_limit:
            record = self.evictor.evict_raw(self)
            if record is not None:
                self.eviction_trace.append(asdict(record))
                continue
            record = self.evictor.evict_card(self)
            if record is None:
                raise MemoryError("forest cannot satisfy its byte budget")
            self.eviction_trace.append(asdict(record))
        self._sync_ledger()
        self.assert_invariants(current_stream_time=current_stream_time)

    def restore(self, rows: Iterable[dict[str, Any]]) -> None:
        self.store = CardStore.from_rows(rows, raw_cache=self.raw)
        retained = {child_id for card in self.store for child_id in card.child_ids}
        self._root_ids = [card.id for card in self.store if card.id not in retained]
        self._current_stream_time = max((card.t_end for card in self.store), default=0.0)
        self.merge_queue = MergeCandidateQueue(
            semantic_weight=self.merge_queue.semantic_weight,
            gap_weight=self.merge_queue.gap_weight,
            boundary_weight=self.merge_queue.boundary_weight,
        )

    def assert_invariants(
        self, *, current_stream_time: float, enforce_budget: bool = True,
        allowed_raw_refs: set[str] | None = None,
    ) -> None:
        if enforce_budget and self.state_bytes > self.limit_bytes:
            raise AssertionError(
                f"forest state is {self.state_bytes} bytes; limit is {self.limit_bytes}"
            )
        ids = set(self.store.ids)
        if len(self._root_ids) != len(set(self._root_ids)) or not set(self._root_ids) <= ids:
            raise AssertionError("forest roots are invalid")
        available = set(self.raw.metadata())
        if allowed_raw_refs is not None:
            available &= set(allowed_raw_refs)
        child_parent_counts: dict[str, int] = {}
        embedding_count = 0
        for card in self.store:
            if card.t_start < 0 or card.t_end < card.t_start or card.t_end > current_stream_time + 1e-6:
                raise AssertionError(f"card span is outside observed history: {card.id}")
            missing_raw = set(card.raw_ref_ids) - available
            if missing_raw:
                raise AssertionError(f"card has unavailable raw refs: {card.id}")
            missing_children = set(card.child_ids) - ids
            if missing_children:
                raise AssertionError(f"card has dangling children: {card.id}")
            for child_id in card.child_ids:
                child_parent_counts[child_id] = child_parent_counts.get(child_id, 0) + 1
            embedding_count += card.text_embedding is not None
        if any(count != 1 for count in child_parent_counts.values()):
            raise AssertionError("a retained child has multiple parents")
        if self.require_embeddings and embedding_count != len(self.store):
            raise AssertionError("embedding count does not match query-visible nodes")
        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(card_id: str) -> None:
            if card_id in visiting:
                raise AssertionError("parent/child cycle detected")
            if card_id in visited:
                return
            visiting.add(card_id)
            for child_id in self.store.get(card_id).child_ids:
                visit(child_id)
            visiting.remove(card_id)
            visited.add(card_id)

        for card_id in ids:
            visit(card_id)
        if set(self._root_ids) != ids - set(child_parent_counts):
            raise AssertionError("forest roots do not match retained child topology")
