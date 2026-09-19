"""Self-contained, byte-accounted query snapshots."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from .audit import checked_snapshot_path
from .types import Budget, VideoMeta


MANIFEST_NAME = "manifest.json"
MANIFEST_HASH_NAME = "manifest.sha256"
_FORBIDDEN_QUERY_STATE_KEYS = {
    "query", "queries", "query_id", "query_path", "query_text",
    "gt", "gt_path", "gt_span", "ground_truth", "ground_truth_span",
    "video_path", "video_root", "trace", "prediction", "predictions",
}


def directory_bytes(directory: Path) -> int:
    return sum(entry.stat().st_size for entry in directory.rglob("*") if entry.is_file())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _file_table(root: Path) -> dict[str, dict[str, object]]:
    """Return metadata for regular query-visible files, never following links."""
    result: dict[str, dict[str, object]] = {}
    for entry in root.rglob("*"):
        if entry.is_symlink():
            raise ValueError(f"snapshot cannot contain a symlink: {entry}")
        if entry.is_file():
            relative = str(entry.relative_to(root))
            result[relative] = {"size_bytes": entry.stat().st_size, "sha256": _sha256(entry)}
    return result


def _assert_query_safe(value: Any, path: str = "") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if str(key).lower() in _FORBIDDEN_QUERY_STATE_KEYS:
                raise ValueError(f"query snapshot cannot contain {path}{key}")
            _assert_query_safe(child, f"{path}{key}.")
    elif isinstance(value, (list, tuple)):
        for child in value:
            _assert_query_safe(child, path)


def _safe_snapshot_target(output_root: Path, name: str) -> Path:
    """Resolve a nested snapshot name without permitting root escape."""
    if not name or Path(name).is_absolute() or ".." in Path(name).parts:
        raise ValueError("snapshot name must be a safe relative path")
    root = output_root.expanduser().resolve(strict=False)
    target = (root / name).resolve(strict=False)
    try:
        target.relative_to(root)
    except ValueError as exc:
        raise ValueError("snapshot target escapes output root") from exc
    if target == root:
        raise ValueError("snapshot target must be below output root")
    return target


def _json_text(value: Any, *, indent: int | None = None) -> str:
    """Serialize strict JSON so NaN/Infinity cannot enter persisted state."""
    return json.dumps(value, sort_keys=True, indent=indent, allow_nan=False)


def _manifest_payload(manifest: "SnapshotManifest") -> dict[str, Any]:
    """Keep the pre-HEM default manifest byte-exact when no event state exists."""
    payload = asdict(manifest)
    if payload.get("event_memory") is None:
        payload.pop("event_memory")
    return payload


@dataclass(frozen=True)
class SnapshotManifest:
    format_version: int
    video_id: str
    t_q: float
    budget_bytes: int
    state_bytes: int
    cards: str
    index: str
    raw_cache_dir: str
    raw_metadata: str
    allowed_files: tuple[str, ...]
    files: dict[str, dict[str, object]]
    writer_calls: int
    config_hash: str
    source_revision: str
    video_meta: dict[str, Any]
    method: str = "full"
    pixel_only: bool = False
    embedder: dict[str, Any] | None = None
    event_memory: str | None = None


class SnapshotWriter:
    """Atomically materialize only objects that query code may read."""

    def __init__(self, output_root: Path, *, source_revision: str = "unknown") -> None:
        self.output_root = output_root
        self.source_revision = source_revision

    def write(
        self,
        *,
        name: str,
        t_q: float,
        meta: VideoMeta,
        budget: Budget,
        cards: Iterable[dict[str, Any]],
        raw_frames: Iterable[tuple[str, bytes]],
        raw_metadata: dict[str, dict[str, object]] | None = None,
        writer_calls: int,
        config: dict[str, Any],
        method: str = "full",
        pixel_only: bool = False,
        embedder: dict[str, Any] | None = None,
        event_memory: Iterable[dict[str, Any]] | None = None,
    ) -> SnapshotManifest:
        if not math.isfinite(float(t_q)) or not 0 <= float(t_q) <= meta.duration_s + 1e-6:
            raise ValueError("snapshot time must be finite and within video duration")
        if not isinstance(writer_calls, int) or isinstance(writer_calls, bool) or writer_calls < 0:
            raise ValueError("writer_calls must be a non-negative integer")
        target = _safe_snapshot_target(self.output_root, name)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}.", dir=target.parent))
        try:
            _assert_query_safe(config)
            _assert_query_safe(embedder)
            cards_path = temporary / "cards.jsonl"
            card_rows = list(cards)
            _assert_query_safe(card_rows)
            cards_path.write_text(
                "".join(_json_text(card) + "\n" for card in card_rows), encoding="utf-8"
            )
            # JSON index stays dependency-free in P0 and is strictly query-visible.
            index_path = temporary / "index.json"
            index_path.write_text(
                _json_text({"card_ids": [str(row["id"]) for row in card_rows]}), encoding="utf-8"
            )
            frames = temporary / "frames"
            frames.mkdir()
            for frame_id, payload in raw_frames:
                if Path(frame_id).name != frame_id:
                    raise ValueError("frame identifiers must be simple filenames")
                if frame_id.lower().startswith(("trace", "prediction")):
                    raise ValueError("trace and prediction files cannot enter query state")
                (frames / frame_id).write_bytes(payload)
            metadata_path = temporary / "frame_metadata.json"
            metadata_rows = raw_metadata or {}
            _assert_query_safe(metadata_rows)
            metadata_path.write_text(_json_text(metadata_rows), encoding="utf-8")
            event_memory_path = None
            if event_memory is not None:
                event_rows = list(event_memory)
                _assert_query_safe(event_rows)
                event_memory_path = temporary / "event_memory.json"
                event_memory_path.write_text(
                    _json_text({"format_version": 1, "events": event_rows}),
                    encoding="utf-8",
                )
            files = _file_table(temporary)
            allowed = tuple(sorted((*files, MANIFEST_NAME, MANIFEST_HASH_NAME)))
            config_hash = hashlib.sha256(
                json.dumps(config, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
            ).hexdigest()
            # Budget includes the manifest itself.  The manifest has a fixed-width
            # state_bytes representation so its final serialization is stable.
            if not method or not method.replace("_", "").replace("-", "").isalnum():
                raise ValueError("snapshot method name is invalid")
            provisional = SnapshotManifest(
                format_version=2, video_id=meta.video_id, t_q=t_q, budget_bytes=budget.memory_bytes,
                state_bytes=0, cards="cards.jsonl", index="index.json", raw_cache_dir="frames", raw_metadata="frame_metadata.json",
                allowed_files=allowed, writer_calls=writer_calls,
                files=files, config_hash=config_hash, source_revision=self.source_revision,
                video_meta=asdict(meta), method=method, pixel_only=bool(pixel_only),
                embedder=dict(embedder) if embedder is not None else None,
                event_memory=(event_memory_path.name if event_memory_path is not None else None),
            )
            manifest_path = temporary / MANIFEST_NAME
            manifest_hash_path = temporary / MANIFEST_HASH_NAME
            manifest = provisional
            for _ in range(4):
                manifest_path.write_text(
                    _json_text(_manifest_payload(manifest), indent=2) + "\n", encoding="utf-8"
                )
                manifest_hash_path.write_text(_sha256(manifest_path) + "\n", encoding="ascii")
                actual = directory_bytes(temporary)
                if actual == manifest.state_bytes:
                    break
                manifest = SnapshotManifest(**{**asdict(manifest), "state_bytes": actual})
            else:  # pragma: no cover - integer width convergence guard
                raise RuntimeError("snapshot manifest size failed to converge")
            actual = directory_bytes(temporary)
            if actual > budget.memory_bytes:
                raise MemoryError(f"snapshot requires {actual} bytes, budget is {budget.memory_bytes}")
            if target.exists():
                shutil.rmtree(target)
            os.replace(temporary, target)
            # mkdtemp creates mode 0700. Snapshots may be produced inside the
            # pinned container and audited by the host user, so keep the
            # immutable files private-by-default while making the snapshot
            # directory traversable/readable (subject to file permissions).
            os.chmod(target, 0o755)
            return manifest
        except Exception:
            shutil.rmtree(temporary, ignore_errors=True)
            raise


class SnapshotReader:
    """The only reader handed to `QueryMethod`; it has no video path field."""

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root).resolve()
        manifest_path = self.root / MANIFEST_NAME
        manifest_hash_path = self.root / MANIFEST_HASH_NAME
        if not self.root.is_dir() or manifest_path.is_symlink() or manifest_hash_path.is_symlink():
            raise ValueError("snapshot root or manifest is invalid")
        if not manifest_hash_path.is_file() or _sha256(manifest_path) != manifest_hash_path.read_text(encoding="ascii").strip():
            raise ValueError("snapshot manifest integrity check failed")
        self.manifest = SnapshotManifest(**json.loads(manifest_path.read_text(encoding="utf-8")))
        self._allowed = set(self.manifest.allowed_files)
        if MANIFEST_NAME not in self._allowed:
            raise ValueError("manifest must authorize itself")
        if set(self.manifest.files) | {MANIFEST_NAME, MANIFEST_HASH_NAME} != self._allowed:
            raise ValueError("manifest file table does not match allowed_files")
        actual_files = _file_table(self.root)
        if set(actual_files) != self._allowed:
            raise ValueError("snapshot contains missing or unlisted files")
        for relative, expected in self.manifest.files.items():
            actual = actual_files.get(relative)
            if actual != expected:
                raise ValueError(f"snapshot integrity check failed: {relative}")
        actual_bytes = directory_bytes(self.root)
        if actual_bytes != self.manifest.state_bytes:
            raise ValueError("snapshot state_bytes does not match filesystem")
        if actual_bytes > self.manifest.budget_bytes:
            raise ValueError("snapshot exceeds its declared byte budget")

    def path(self, relative_path: str) -> Path:
        return checked_snapshot_path(self.root, relative_path, self._allowed)

    def read_cards(self) -> list[dict[str, Any]]:
        return [json.loads(line) for line in self.path(self.manifest.cards).read_text(encoding="utf-8").splitlines()]

    def frame_paths(self) -> list[Path]:
        return [self.path(name) for name in self.manifest.allowed_files if name.startswith(self.manifest.raw_cache_dir + "/")]

    def read_frame_metadata(self) -> dict[str, dict[str, Any]]:
        rows = json.loads(self.path(self.manifest.raw_metadata).read_text(encoding="utf-8"))
        if not isinstance(rows, dict):
            raise ValueError("frame metadata must be an object")
        prefix = self.manifest.raw_cache_dir + "/"
        for frame_ref, row in rows.items():
            if Path(frame_ref).name != frame_ref or not isinstance(row, dict):
                raise ValueError("invalid frame metadata record")
            blob = row.get("blob", frame_ref)
            if not isinstance(blob, str) or Path(blob).name != blob:
                raise ValueError("invalid frame blob reference")
            relative = prefix + blob
            if relative not in self._allowed:
                raise ValueError(f"frame metadata references an unavailable blob: {blob}")
            expected_hash = row.get("sha256")
            if expected_hash is not None and expected_hash != self.manifest.files[relative]["sha256"]:
                raise ValueError(f"frame metadata hash disagrees with blob: {blob}")
        return rows

    def frame_path(self, frame_ref: str) -> Path:
        metadata = self.read_frame_metadata()
        if frame_ref not in metadata:
            raise KeyError(frame_ref)
        blob = str(metadata[frame_ref].get("blob", frame_ref))
        return self.path(f"{self.manifest.raw_cache_dir}/{blob}")

    def read_event_memory(self) -> list[dict[str, Any]]:
        """Read the optional query-independent hierarchical event state."""
        relative = self.manifest.event_memory
        if relative is None:
            return []
        payload = json.loads(self.path(relative).read_text(encoding="utf-8"))
        if payload.get("format_version") != 1 or not isinstance(payload.get("events"), list):
            raise ValueError("invalid event memory payload")
        rows = payload["events"]
        _assert_query_safe(rows)
        if any(not isinstance(row, dict) for row in rows):
            raise ValueError("event memory rows must be objects")
        return rows
