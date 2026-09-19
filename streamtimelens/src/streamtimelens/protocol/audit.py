"""Filesystem gate used by the query process.

Keeping this check in one small module makes it easy to integration-test that
queries cannot recover the original video by following manifest paths.
"""

from __future__ import annotations

import builtins
import io
import os
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator


class SnapshotAccessError(PermissionError):
    pass


def checked_snapshot_path(root: Path, relative_path: str, allowed_files: set[str]) -> Path:
    if relative_path not in allowed_files:
        raise SnapshotAccessError(f"manifest does not authorize {relative_path!r}")
    candidate = (root / relative_path).resolve(strict=False)
    try:
        candidate.relative_to(root.resolve())
    except ValueError as exc:
        raise SnapshotAccessError(f"path escapes snapshot root: {relative_path!r}") from exc
    return candidate


@dataclass
class ReadAudit:
    """Captured file reads for one query invocation."""

    allowed_paths: set[Path]
    reads: list[Path] = field(default_factory=list)

    def record(self, path: os.PathLike | str | int, mode: str) -> None:
        if isinstance(path, int) or any(flag in mode for flag in ("w", "a", "x", "+")):
            return
        resolved = Path(path).resolve(strict=False)
        if resolved not in self.allowed_paths:
            raise SnapshotAccessError(f"query attempted to read unauthorized path: {resolved}")
        self.reads.append(resolved)


@contextmanager
def audit_query_reads(snapshot_root: Path | str, allowed_files: set[str], *, model_files: set[Path] | None = None) -> Iterator[ReadAudit]:
    """Deny direct query reads outside manifest files (plus explicit model files).

    The guard intentionally instruments both `builtins.open` and `io.open`, as
    `pathlib.Path.read_*` uses the latter.  It is an integration-test/audit
    guard; production isolation still relies on mounting only the snapshot.
    """
    root = Path(snapshot_root).resolve()
    allowed = {checked_snapshot_path(root, relative, allowed_files) for relative in allowed_files}
    allowed.update(path.resolve() for path in (model_files or set()))
    audit = ReadAudit(allowed)
    original_builtin_open, original_io_open = builtins.open, io.open

    def audited_open(path, mode="r", *args, **kwargs):
        audit.record(path, mode)
        return original_builtin_open(path, mode, *args, **kwargs)

    def audited_io_open(path, mode="r", *args, **kwargs):
        audit.record(path, mode)
        return original_io_open(path, mode, *args, **kwargs)

    builtins.open, io.open = audited_open, audited_io_open
    try:
        yield audit
    finally:
        builtins.open, io.open = original_builtin_open, original_io_open
