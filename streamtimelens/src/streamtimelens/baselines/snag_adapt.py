"""Query-blind streaming stores for the SnAG late-fusion reader.

This is a StreamTimeLens adaptation, not an upstream SnAG entry point.  The
writer deliberately knows nothing about text queries and the reader can only
open files listed in an immutable snapshot manifest.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Literal, Protocol, Sequence

import numpy as np

from streamtimelens.protocol.types import VideoMeta


SNAG_UPSTREAM_REVISION = "44dd90eea9a65b64f7088974eae352c1e26ef6e3"
_MANIFEST = "manifest.json"
_MANIFEST_HASH = "manifest.sha256"
_FEATURES = "features.npy"
_SCALES = "scales.npy"
_ITEMS = "items.jsonl"


def _finite(name: str, value: float, *, non_negative: bool = False) -> float:
    result = float(value)
    if not math.isfinite(result) or (non_negative and result < 0):
        raise ValueError(f"{name} must be finite and valid")
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _directory_bytes(root: Path) -> int:
    return sum(path.stat().st_size for path in root.iterdir() if path.is_file())


def _json(value: Any, *, indent: int | None = None) -> str:
    return json.dumps(value, sort_keys=True, indent=indent, allow_nan=False)


@dataclass(frozen=True)
class SnAGStateItem:
    """A feature token and its absolute-time provenance."""

    feature: np.ndarray
    t_start_s: float
    t_end_s: float
    source_ids: tuple[str, ...]
    level_or_scale: int = 0
    aggregation_count: int = 1
    left_uncertainty_s: float = 0.0
    right_uncertainty_s: float = 0.0

    def __post_init__(self) -> None:
        feature = np.asarray(self.feature)
        start = _finite("t_start_s", self.t_start_s, non_negative=True)
        end = _finite("t_end_s", self.t_end_s, non_negative=True)
        if feature.ndim != 1 or not feature.size or not np.isfinite(feature).all():
            raise ValueError("feature must be a finite, non-empty vector")
        if start >= end:
            raise ValueError("state item interval must be ordered")
        if not self.source_ids or any(not value for value in self.source_ids):
            raise ValueError("state item needs non-empty source IDs")
        if self.level_or_scale < 0 or self.aggregation_count < 1:
            raise ValueError("invalid state item aggregation metadata")
        _finite("left_uncertainty_s", self.left_uncertainty_s, non_negative=True)
        _finite("right_uncertainty_s", self.right_uncertainty_s, non_negative=True)
        object.__setattr__(self, "feature", feature.astype(np.float32, copy=True))

    def metadata(self) -> dict[str, Any]:
        value = asdict(self)
        value.pop("feature")
        value["source_ids"] = list(self.source_ids)
        return value


@dataclass(frozen=True)
class SnAGAdaptConfig:
    mode: Literal["full", "pooled"] = "full"
    storage_dtype: Literal["float16", "float32", "int8"] = "float16"
    budget_bytes: int | None = None
    near_capacity: int = 64
    far_capacity: int = 192
    max_source_ids: int = 3
    source_revision: str = SNAG_UPSTREAM_REVISION

    def __post_init__(self) -> None:
        if self.mode not in ("full", "pooled"):
            raise ValueError("SnAG store mode must be full or pooled")
        if self.storage_dtype not in ("float16", "float32", "int8"):
            raise ValueError("unsupported SnAG storage dtype")
        if self.mode == "pooled" and (self.budget_bytes is None or self.budget_bytes <= 0):
            raise ValueError("pooled SnAG store needs a positive serialized-byte budget")
        if self.mode == "full" and self.budget_bytes is not None:
            raise ValueError("full SnAG store is an unbudgeted upper bound")
        if self.near_capacity < 1 or self.far_capacity < 1 or self.max_source_ids < 1:
            raise ValueError("SnAG store capacities must be positive")
        if not self.source_revision:
            raise ValueError("SnAG upstream revision must be recorded")


@dataclass(frozen=True)
class SnAGSnapshotManifest:
    format_version: int
    method: str
    mode: str
    video_id: str
    t_q: float
    state_bytes: int
    budget_bytes: int | None
    token_count: int
    feature_dim: int
    storage_dtype: str
    features: str
    scales: str | None
    items: str
    files: dict[str, dict[str, object]]
    allowed_files: tuple[str, ...]
    source_revision: str
    video_meta: dict[str, Any]
    config: dict[str, Any]


@dataclass(frozen=True)
class ProvenancePoint:
    center_s: float
    stride_s: float
    support_start_s: float
    support_end_s: float
    level_or_scale: int
    left_uncertainty_s: float
    right_uncertainty_s: float


@dataclass(frozen=True)
class RankedSpan:
    start_s: float
    end_s: float
    score: float


def ranked_spans_close(
    left: Sequence[RankedSpan], right: Sequence[RankedSpan], *, absolute_tolerance: float = 1e-5,
) -> bool:
    """Compare repeated GPU readouts within a frozen numerical tolerance."""
    if absolute_tolerance < 0 or len(left) != len(right):
        return False
    return all(
        math.isclose(a.start_s, b.start_s, rel_tol=1e-6, abs_tol=absolute_tolerance)
        and math.isclose(a.end_s, b.end_s, rel_tol=1e-6, abs_tol=absolute_tolerance)
        and math.isclose(a.score, b.score, rel_tol=1e-6, abs_tol=absolute_tolerance)
        for a, b in zip(left, right)
    )


class SnAGPredictionBackend(Protocol):
    """Boundary used by :class:`SnAGAdaptReader` for real or fixture heads."""

    def predict(
        self,
        features: np.ndarray,
        items: Sequence[SnAGStateItem],
        query: Any,
    ) -> Iterable[RankedSpan | tuple[float, float, float]]:
        ...


class ProvenancePointGenerator:
    """Generate physical-time FPN points without packed-index decoding."""

    @staticmethod
    def level(items: Sequence[SnAGStateItem], stride: int, level: int) -> tuple[ProvenancePoint, ...]:
        if stride < 1 or level < 0:
            raise ValueError("stride and level must be non-negative")
        base_centers = np.asarray([
            (item.t_start_s + item.t_end_s) / 2 for item in items
        ], dtype=np.float64)
        center_steps = np.diff(base_centers)
        positive_steps = center_steps[center_steps > 0]
        base_step = (
            float(np.median(positive_steps))
            if positive_steps.size
            else (items[0].t_end_s - items[0].t_start_s if items else 1.0)
        )
        points = []
        for offset in range(0, len(items), stride):
            group = items[offset:offset + stride]
            if not group:
                continue
            start = min(item.t_start_s for item in group)
            end = max(item.t_end_s for item in group)
            anchor = group[0]
            points.append(ProvenancePoint(
                # Upstream PtGenerator anchors level-l points at input indices
                # 0, 2**l, ...; this is not the receptive-field midpoint.
                center_s=(anchor.t_start_s + anchor.t_end_s) / 2,
                stride_s=max(base_step * stride, np.finfo(np.float32).eps),
                support_start_s=start,
                support_end_s=end,
                level_or_scale=level,
                left_uncertainty_s=max(item.left_uncertainty_s for item in group),
                right_uncertainty_s=max(item.right_uncertainty_s for item in group),
            ))
        return tuple(points)

    def __call__(
        self, items: Sequence[SnAGStateItem], fpn_lengths: Sequence[int],
    ) -> tuple[tuple[ProvenancePoint, ...], ...]:
        levels = []
        for level, length in enumerate(fpn_lengths):
            candidates = self.level(items, 2 ** level, level)
            if length < 0 or length > len(candidates):
                raise ValueError("FPN length is incompatible with snapshot provenance")
            levels.append(candidates[:length])
        return tuple(levels)

    @staticmethod
    def decode(
        points: Sequence[ProvenancePoint], offsets: np.ndarray,
        *, normalized_by_width: bool = False, duration_s: float | None = None,
    ) -> np.ndarray:
        values = np.asarray(offsets, dtype=np.float64)
        if values.shape != (len(points), 2) or not np.isfinite(values).all() or (values < 0).any():
            raise ValueError("offsets must have shape [points,2] and be finite/non-negative")
        result = np.empty_like(values)
        for index, (point, offset) in enumerate(zip(points, values)):
            scale = point.support_end_s - point.support_start_s if normalized_by_width else point.stride_s
            result[index] = (point.center_s - offset[0] * scale, point.center_s + offset[1] * scale)
        result[:, 0] = np.maximum(result[:, 0], 0.0)
        if duration_s is not None:
            result[:, 1] = np.minimum(result[:, 1], float(duration_s))
        return result


class SnAGAdaptWriter:
    """Append-only query-blind writer with deterministic adjacent pooling."""

    def __init__(self, meta: VideoMeta, config: SnAGAdaptConfig) -> None:
        self.meta = meta
        self.config = config
        self._near: list[SnAGStateItem] = []
        self._far: list[SnAGStateItem] = []
        self._feature_dim: int | None = None
        self._last_start = -1.0
        self._last_end = -1.0

    @property
    def items(self) -> tuple[SnAGStateItem, ...]:
        return tuple((*self._far, *self._near))

    def ingest_step(
        self,
        feature: np.ndarray,
        t_start_s: float,
        t_end_s: float,
        source_ids: str | Iterable[str],
        *,
        level_or_scale: int = 0,
        left_uncertainty_s: float = 0.0,
        right_uncertainty_s: float = 0.0,
    ) -> None:
        """Consume one sequential clip.  No query/GT argument exists by design."""
        ids = (source_ids,) if isinstance(source_ids, str) else tuple(source_ids)
        item = SnAGStateItem(
            feature, t_start_s, t_end_s, ids, level_or_scale, 1,
            left_uncertainty_s, right_uncertainty_s,
        )
        if item.t_start_s + 1e-9 < self._last_start or item.t_end_s + 1e-9 < self._last_end:
            raise ValueError("clip observations must arrive in temporal order")
        if item.t_end_s > self.meta.duration_s + 1e-6:
            raise ValueError("clip provenance exceeds video duration")
        if self._feature_dim is not None and item.feature.size != self._feature_dim:
            raise ValueError("all SnAG clip features must have the same dimension")
        previous = (
            list(self._near), list(self._far), self._feature_dim,
            self._last_start, self._last_end,
        )
        try:
            if self._feature_dim is None:
                self._feature_dim = item.feature.size
            self._last_start, self._last_end = item.t_start_s, item.t_end_s
            self._near.append(item)
            if self.config.mode == "pooled":
                while len(self._near) > self.config.near_capacity:
                    self._far.append(self._near.pop(0))
                while len(self._far) > self.config.far_capacity:
                    self._merge_closest_far()
                self._enforce_serialized_budget()
        except Exception:
            (
                self._near, self._far, self._feature_dim,
                self._last_start, self._last_end,
            ) = previous
            raise

    def _bounded_sources(self, left: SnAGStateItem, right: SnAGStateItem) -> tuple[str, ...]:
        source_ids = (*left.source_ids, *right.source_ids)
        if len(source_ids) <= self.config.max_source_ids:
            return source_ids
        digest = hashlib.sha256("\0".join(source_ids).encode("utf-8")).hexdigest()[:16]
        if self.config.max_source_ids == 1:
            return (f"aggregate-sha256:{digest}",)
        if self.config.max_source_ids == 2:
            return (source_ids[0], f"aggregate-sha256:{digest}")
        return (source_ids[0], f"aggregate-sha256:{digest}", source_ids[-1])

    def _merge(self, left: SnAGStateItem, right: SnAGStateItem) -> SnAGStateItem:
        count = left.aggregation_count + right.aggregation_count
        feature = (
            left.feature * left.aggregation_count + right.feature * right.aggregation_count
        ) / count
        return SnAGStateItem(
            feature=feature,
            t_start_s=min(left.t_start_s, right.t_start_s),
            t_end_s=max(left.t_end_s, right.t_end_s),
            source_ids=self._bounded_sources(left, right),
            level_or_scale=max(left.level_or_scale, right.level_or_scale) + 1,
            aggregation_count=count,
            left_uncertainty_s=max(left.left_uncertainty_s, right.left_uncertainty_s),
            right_uncertainty_s=max(left.right_uncertainty_s, right.right_uncertainty_s),
        )

    def _merge_closest_far(self) -> None:
        if len(self._far) < 2:
            raise MemoryError("cannot compact a single far token further")
        gaps = [max(0.0, right.t_start_s - left.t_end_s) for left, right in zip(self._far, self._far[1:])]
        compatible = [index for index, gap in enumerate(gaps) if gap <= 1e-6]
        if not compatible:
            raise MemoryError(
                "cannot pool temporally disjoint evidence without inventing continuous support"
            )
        index = min(compatible, key=lambda value: (gaps[value], value))
        self._far[index:index + 2] = [self._merge(self._far[index], self._far[index + 1])]

    def _compact_one(self) -> bool:
        if len(self._far) >= 2:
            self._merge_closest_far()
            return True
        if len(self._near) >= 2:
            self._far.append(self._near.pop(0))
            if len(self._far) >= 2:
                self._merge_closest_far()
            return True
        return False

    def _enforce_serialized_budget(self) -> None:
        assert self.config.budget_bytes is not None
        while self._estimated_snapshot_bytes(self._last_end) > self.config.budget_bytes:
            if not self._compact_one():
                required = self._estimated_snapshot_bytes(self._last_end)
                raise MemoryError(
                    f"SnAG snapshot minimum is {required} bytes, budget is {self.config.budget_bytes}"
                )

    def _estimated_snapshot_bytes(self, t_q: float) -> int:
        temporary = Path(tempfile.mkdtemp(prefix=".snag-budget."))
        try:
            manifest = self._write_files(temporary, t_q)
            return manifest.state_bytes
        finally:
            shutil.rmtree(temporary, ignore_errors=True)

    def freeze(self, t_q: float, target: Path | str) -> SnAGSnapshotManifest:
        """Serialize a self-contained immutable snapshot at the current frontier."""
        query_time = _finite("t_q", t_q, non_negative=True)
        if query_time + 1e-9 < self._last_end:
            raise ValueError("cannot freeze behind already ingested observations")
        if query_time > self.meta.duration_s + 1e-6:
            raise ValueError("snapshot time exceeds video duration")
        destination = Path(target).expanduser().resolve(strict=False)
        if destination.exists():
            raise FileExistsError(f"refusing to overwrite snapshot: {destination}")
        if self.config.budget_bytes is not None:
            while self._estimated_snapshot_bytes(query_time) > self.config.budget_bytes:
                if not self._compact_one():
                    required = self._estimated_snapshot_bytes(query_time)
                    raise MemoryError(
                        f"SnAG snapshot minimum is {required} bytes, "
                        f"budget is {self.config.budget_bytes}"
                    )
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
        try:
            manifest = self._write_files(temporary, query_time)
            if self.config.budget_bytes is not None and manifest.state_bytes > self.config.budget_bytes:
                raise MemoryError("serialized SnAG snapshot exceeds its byte budget")
            os.replace(temporary, destination)
            os.chmod(destination, 0o555)
            for child in destination.iterdir():
                child.chmod(0o444)
            return manifest
        except Exception:
            shutil.rmtree(temporary, ignore_errors=True)
            raise

    def _encoded(self) -> tuple[np.ndarray, np.ndarray | None]:
        if not self.items:
            dimension = self._feature_dim or 0
            dtype = np.int8 if self.config.storage_dtype == "int8" else np.dtype(self.config.storage_dtype)
            return np.empty((0, dimension), dtype=dtype), None
        features = np.stack([item.feature for item in self.items])
        if self.config.storage_dtype != "int8":
            return features.astype(self.config.storage_dtype), None
        scales = np.maximum(np.max(np.abs(features), axis=1), np.finfo(np.float32).eps) / 127.0
        quantized = np.clip(np.rint(features / scales[:, None]), -127, 127).astype(np.int8)
        return quantized, scales.astype(np.float32)

    def _write_files(self, root: Path, t_q: float) -> SnAGSnapshotManifest:
        root.mkdir(parents=True, exist_ok=True)
        encoded, scales = self._encoded()
        np.save(root / _FEATURES, encoded, allow_pickle=False)
        if scales is not None:
            np.save(root / _SCALES, scales, allow_pickle=False)
        (root / _ITEMS).write_text(
            "".join(_json(item.metadata()) + "\n" for item in self.items), encoding="utf-8"
        )
        payload_files = (_FEATURES, _ITEMS) if scales is None else (_FEATURES, _SCALES, _ITEMS)
        files = {
            name: {"size_bytes": (root / name).stat().st_size, "sha256": _sha256(root / name)}
            for name in payload_files
        }
        allowed = tuple(sorted((*payload_files, _MANIFEST, _MANIFEST_HASH)))
        config = asdict(self.config)
        provisional = SnAGSnapshotManifest(
            format_version=1,
            method="snag-adapt-input-full" if self.config.mode == "full" else "snag-adapt-pooled-B",
            mode=self.config.mode,
            video_id=self.meta.video_id,
            t_q=t_q,
            state_bytes=0,
            budget_bytes=self.config.budget_bytes,
            token_count=len(self.items),
            feature_dim=self._feature_dim or 0,
            storage_dtype=self.config.storage_dtype,
            features=_FEATURES,
            scales=_SCALES if scales is not None else None,
            items=_ITEMS,
            files=files,
            allowed_files=allowed,
            source_revision=self.config.source_revision,
            video_meta=asdict(self.meta),
            config=config,
        )
        manifest = provisional
        for _ in range(4):
            (root / _MANIFEST).write_text(_json(asdict(manifest), indent=2) + "\n", encoding="utf-8")
            (root / _MANIFEST_HASH).write_text(_sha256(root / _MANIFEST) + "\n", encoding="ascii")
            actual = _directory_bytes(root)
            if actual == manifest.state_bytes:
                return manifest
            manifest = SnAGSnapshotManifest(**{**asdict(manifest), "state_bytes": actual})
        raise RuntimeError("SnAG manifest byte count failed to converge")


class SnAGSnapshotReader:
    """Integrity-checking reader with no access to source video or writer state."""

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root).expanduser().resolve()
        if not self.root.is_dir() or any(path.is_symlink() for path in self.root.iterdir()):
            raise ValueError("invalid SnAG snapshot directory")
        manifest_path = self.root / _MANIFEST
        hash_path = self.root / _MANIFEST_HASH
        if not hash_path.is_file() or _sha256(manifest_path) != hash_path.read_text(encoding="ascii").strip():
            raise ValueError("SnAG manifest integrity check failed")
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
        raw["allowed_files"] = tuple(raw["allowed_files"])
        self.manifest = SnAGSnapshotManifest(**raw)
        if (
            self.manifest.format_version != 1
            or self.manifest.mode not in ("full", "pooled")
            or self.manifest.storage_dtype not in ("float16", "float32", "int8")
            or not self.manifest.source_revision
            or self.manifest.token_count < 0
            or self.manifest.feature_dim < 0
        ):
            raise ValueError("unsupported SnAG snapshot manifest")
        snapshot_meta = VideoMeta(**self.manifest.video_meta)
        if snapshot_meta.video_id != self.manifest.video_id:
            raise ValueError("SnAG video metadata disagrees with manifest")
        actual_names = {path.name for path in self.root.iterdir() if path.is_file()}
        if actual_names != set(self.manifest.allowed_files):
            raise ValueError("SnAG snapshot contains missing or unauthorized files")
        for name, expected in self.manifest.files.items():
            path = self._path(name)
            actual = {"size_bytes": path.stat().st_size, "sha256": _sha256(path)}
            if actual != expected:
                raise ValueError(f"SnAG snapshot integrity check failed: {name}")
        actual_bytes = _directory_bytes(self.root)
        if actual_bytes != self.manifest.state_bytes:
            raise ValueError("SnAG state_bytes does not match serialized files")
        if self.manifest.budget_bytes is not None and actual_bytes > self.manifest.budget_bytes:
            raise ValueError("SnAG snapshot exceeds its declared budget")
        self._items = self._read_items()
        encoded = np.load(self._path(self.manifest.features), mmap_mode="r", allow_pickle=False)
        if encoded.shape != (self.manifest.token_count, self.manifest.feature_dim):
            raise ValueError("SnAG feature shape disagrees with manifest")
        if len(self._items) != self.manifest.token_count:
            raise ValueError("SnAG provenance count disagrees with feature count")

    def _path(self, name: str) -> Path:
        if Path(name).name != name or name not in self.manifest.allowed_files:
            raise ValueError("unauthorized SnAG snapshot path")
        return self.root / name

    def _read_items(self) -> tuple[SnAGStateItem, ...]:
        rows = [json.loads(line) for line in self._path(self.manifest.items).read_text(encoding="utf-8").splitlines()]
        # Placeholder features are replaced after the encoded matrix is loaded.
        result = []
        for row in rows:
            result.append(row)
        encoded = np.load(self._path(self.manifest.features), mmap_mode="r", allow_pickle=False)
        scales = None
        if self.manifest.scales is not None:
            scales = np.load(self._path(self.manifest.scales), mmap_mode="r", allow_pickle=False)
            if scales.shape != (len(rows),):
                raise ValueError("SnAG int8 scale shape is invalid")
        items = []
        for index, row in enumerate(result):
            feature = np.asarray(encoded[index], dtype=np.float32)
            if scales is not None:
                feature = feature * float(scales[index])
            row["source_ids"] = tuple(row["source_ids"])
            item = SnAGStateItem(feature=feature, **row)
            item.feature.setflags(write=False)
            items.append(item)
        return tuple(items)

    @property
    def items(self) -> tuple[SnAGStateItem, ...]:
        return self._items

    def features(self) -> np.ndarray:
        if not self._items:
            return np.empty((0, self.manifest.feature_dim), dtype=np.float32)
        return np.stack([item.feature for item in self._items]).astype(np.float32, copy=False)


class SnAGAdaptReader:
    """Snapshot-only query API.  A new backend call is made for every query."""

    def __init__(self, backend: SnAGPredictionBackend) -> None:
        self.backend = backend

    def answer(self, snapshot: SnAGSnapshotReader, query: Any, *, topk: int = 5) -> tuple[RankedSpan, ...]:
        if topk < 1:
            raise ValueError("topk must be positive")
        raw = self.backend.predict(snapshot.features(), snapshot.items, query)
        spans = []
        for value in raw:
            span = value if isinstance(value, RankedSpan) else RankedSpan(*map(float, value))
            if (
                not all(math.isfinite(number) for number in (span.start_s, span.end_s, span.score))
                or span.start_s < 0
                or span.start_s >= span.end_s
                or not 0 <= span.score <= 1
                or span.end_s > snapshot.manifest.t_q + 1e-6
            ):
                raise ValueError("SnAG backend returned an invalid or future span")
            spans.append(span)
        return tuple(sorted(spans, key=lambda value: value.score, reverse=True)[:topk])
