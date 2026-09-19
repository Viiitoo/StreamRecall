"""Transactional accounting for every query-visible byte of stream state."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class LedgerEntry:
    object_id: str
    kind: str
    logical_bytes: int


@dataclass
class BudgetLedger:
    """Logical state budget plus separately measured snapshot filesystem bytes.

    Components model aggregate structures. Reservations model individually
    owned payloads and are idempotent for the same object, so shared frame
    stores cannot accidentally be double-counted.
    """

    limit_bytes: int
    writer_calls: int = 0
    components: dict[str, int] = field(default_factory=dict)
    entries: dict[str, LedgerEntry] = field(default_factory=dict)
    snapshot_filesystem_bytes: int | None = None

    def __post_init__(self) -> None:
        if self.limit_bytes <= 0:
            raise ValueError("limit_bytes must be positive")

    def set_bytes(self, component: str, byte_size: int) -> None:
        if not component or byte_size < 0:
            raise ValueError("component name and non-negative byte size are required")
        projected = self.logical_bytes - self.components.get(component, 0) + byte_size
        if projected > self.limit_bytes:
            raise MemoryError(f"reserving {component} would exceed {self.limit_bytes} bytes")
        self.components[component] = byte_size

    def replace_components(self, components: dict[str, int]) -> None:
        """Atomically replace aggregate accounting after eviction decisions."""
        if any(not name or size < 0 for name, size in components.items()):
            raise ValueError("component names and byte sizes must be valid")
        projected = sum(components.values()) + self.reserved_bytes
        if projected > self.limit_bytes:
            raise MemoryError(f"component state would exceed {self.limit_bytes} bytes")
        self.components = dict(components)

    def reserve(self, object_id: str, kind: str, nbytes: int) -> bool:
        """Atomically reserve an object; false denotes an identical duplicate."""
        if not object_id or not kind or nbytes < 0:
            raise ValueError("object_id, kind and non-negative nbytes are required")
        existing = self.entries.get(object_id)
        candidate = LedgerEntry(object_id, kind, nbytes)
        if existing:
            if existing == candidate:
                return False
            raise ValueError(f"object already reserved with different payload: {object_id}")
        if self.logical_bytes + nbytes > self.limit_bytes:
            raise MemoryError(f"reserving {object_id} would exceed {self.limit_bytes} bytes")
        self.entries[object_id] = candidate
        return True

    def release(self, object_id: str) -> bool:
        """Release an owned object; false means it was already absent."""
        return self.entries.pop(object_id, None) is not None

    @property
    def reserved_bytes(self) -> int:
        return sum(entry.logical_bytes for entry in self.entries.values())

    @property
    def logical_bytes(self) -> int:
        return sum(self.components.values()) + self.reserved_bytes

    @property
    def state_bytes(self) -> int:
        """Compatibility name for query-visible logical-state accounting."""
        return self.logical_bytes

    def reconcile(self, snapshot_dir: Path | str) -> int:
        """Measure serialization bytes without conflating them with payload bytes."""
        root = Path(snapshot_dir)
        if not root.is_dir():
            raise ValueError(f"snapshot directory does not exist: {root}")
        total = 0
        for entry in root.rglob("*"):
            if entry.is_symlink():
                raise ValueError(f"cannot reconcile symlinked snapshot file: {entry}")
            if entry.is_file():
                total += entry.stat().st_size
        self.snapshot_filesystem_bytes = total
        return total

    def assert_within_budget(self) -> None:
        if self.logical_bytes > self.limit_bytes:
            raise MemoryError(f"query state is {self.logical_bytes} bytes; budget is {self.limit_bytes}")
