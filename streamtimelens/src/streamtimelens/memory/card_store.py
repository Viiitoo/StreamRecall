"""Deterministic temporal storage and accounting for evidence-card nodes."""

from __future__ import annotations

import bisect
import json
from dataclasses import dataclass, fields
from typing import Any, Iterable, Iterator

from streamtimelens.memory.cards import EvidenceCard
from streamtimelens.memory.raw_cache import RawFrameCache


def _json_bytes(value: Any) -> int:
    return len(json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8"))


@dataclass(frozen=True)
class StoreAccounting:
    card_payload_bytes: int
    embedding_bytes: int
    raw_ref_index_bytes: int
    temporal_index_bytes: int

    @property
    def total_bytes(self) -> int:
        return (
            self.card_payload_bytes + self.embedding_bytes +
            self.raw_ref_index_bytes + self.temporal_index_bytes
        )


class CardStore:
    """ID lookup plus a stable ``(t_start, t_end, id)`` temporal index.

    Raw JPEG payloads remain owned by :class:`RawFrameCache`; this store only
    accounts for the card-side references. Deletion releases the card owner so
    shared blobs disappear exactly when their final owner does.
    """

    def __init__(self, raw_cache: RawFrameCache | None = None) -> None:
        self.raw_cache = raw_cache
        self._cards: dict[str, EvidenceCard] = {}
        self._keys: list[tuple[float, float, str]] = []
        self._accounting: dict[str, tuple[int, int, int]] = {}
        self._temporal_index_bytes = _json_bytes({"card_ids": []})

    @staticmethod
    def _card_accounting(card: EvidenceCard) -> tuple[int, int, int]:
        row = card.serializable()
        serialized = card.byte_size + 1
        embedding_row = row.get("text_embedding")
        embedding_size = 0 if embedding_row is None else _json_bytes(embedding_row)
        raw_size = _json_bytes({
            "raw_ref_ids": row.get("raw_ref_ids", []),
            "raw_ref_status": row.get("raw_ref_status", {}),
            "boundary_cache": row.get("boundary_cache", {}),
        })
        return max(0, serialized - embedding_size - raw_size), embedding_size, raw_size

    def _refresh_index_bytes(self) -> None:
        self._temporal_index_bytes = _json_bytes({"card_ids": list(self.ids)})

    @staticmethod
    def key(card: EvidenceCard) -> tuple[float, float, str]:
        return (float(card.t_start), float(card.t_end), str(card.id))

    def insert(self, card: EvidenceCard) -> None:
        if not card.id:
            raise ValueError("card ID must not be empty")
        if card.id in self._cards:
            raise ValueError(f"duplicate card ID: {card.id}")
        key = self.key(card)
        bisect.insort(self._keys, key)
        self._cards[card.id] = card
        self._accounting[card.id] = self._card_accounting(card)
        self._refresh_index_bytes()

    def touch(self, card_id: str) -> None:
        """Refresh exact bytes after an in-place card metadata update."""
        self._accounting[card_id] = self._card_accounting(self.get(card_id))

    def get(self, card_id: str) -> EvidenceCard:
        return self._cards[card_id]

    def delete(self, card_id: str) -> EvidenceCard:
        card = self._cards.pop(card_id)
        index = bisect.bisect_left(self._keys, self.key(card))
        if index >= len(self._keys) or self._keys[index] != self.key(card):
            raise RuntimeError(f"temporal index lost card: {card_id}")
        self._keys.pop(index)
        del self._accounting[card_id]
        self._refresh_index_bytes()
        if self.raw_cache is not None:
            owner = f"card:{card.id}"
            for frame_id in tuple(card.raw_ref_ids):
                self.raw_cache.release(frame_id, owner)
        return card

    def temporal_neighbors(
        self, card_id: str,
    ) -> tuple[EvidenceCard | None, EvidenceCard | None]:
        card = self.get(card_id)
        index = bisect.bisect_left(self._keys, self.key(card))
        left = self._cards[self._keys[index - 1][2]] if index else None
        right = self._cards[self._keys[index + 1][2]] if index + 1 < len(self._keys) else None
        return left, right

    def rows(self) -> list[dict[str, Any]]:
        return [card.serializable() for card in self]

    @classmethod
    def from_rows(
        cls, rows: Iterable[dict[str, Any]], *, raw_cache: RawFrameCache | None = None,
    ) -> "CardStore":
        accepted = {item.name for item in fields(EvidenceCard)}
        store = cls(raw_cache)
        for row in rows:
            unknown = set(row) - accepted
            if unknown:
                raise ValueError(f"unknown evidence-card fields: {', '.join(sorted(unknown))}")
            store.insert(EvidenceCard(**row))
        return store

    def accounting(self) -> StoreAccounting:
        payload = sum(value[0] for value in self._accounting.values())
        embedding = sum(value[1] for value in self._accounting.values())
        raw_refs = sum(value[2] for value in self._accounting.values())
        return StoreAccounting(payload, embedding, raw_refs, self._temporal_index_bytes)

    def __contains__(self, card_id: object) -> bool:
        return card_id in self._cards

    def __len__(self) -> int:
        return len(self._cards)

    def __iter__(self) -> Iterator[EvidenceCard]:
        for _, _, card_id in self._keys:
            yield self._cards[card_id]

    @property
    def ids(self) -> tuple[str, ...]:
        return tuple(key[2] for key in self._keys)
