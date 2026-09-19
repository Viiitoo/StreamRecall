"""Shared, content-addressed JPEG storage for every in-memory frame owner."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from io import BytesIO
from typing import Any

from streamtimelens.protocol.types import FramePacket


JPEG_SHORT_EDGE = 224
JPEG_QUALITY = 85


@dataclass(frozen=True)
class CachedFrame:
    """A timestamped reference to a shared immutable JPEG blob."""

    frame_id: str
    content_id: str
    timestamp_s: float
    frame_index: int
    video_id: str
    width: int
    height: int

    @property
    def sha256(self) -> str:
        return self.content_id


@dataclass
class _Blob:
    payload: bytes
    owner_count: int = 0


def _resample_filter() -> int:
    from PIL import Image

    return getattr(getattr(Image, "Resampling", Image), "LANCZOS")


def compress_rgb_image(
    image: Any, *, short_edge: int = JPEG_SHORT_EDGE, quality: int = JPEG_QUALITY
) -> tuple[bytes, int, int]:
    """Resize an RGB image by its short edge and encode a deterministic JPEG."""
    if short_edge <= 0 or not 1 <= quality <= 100:
        raise ValueError("short_edge and JPEG quality must be positive and valid")
    try:
        from PIL import Image
        import numpy as np
    except ImportError as exc:  # pragma: no cover - runtime dependencies
        raise RuntimeError("Pillow and numpy are required for RGB frame compression") from exc

    if isinstance(image, Image.Image):
        rgb = image.convert("RGB")
    else:
        array = np.asarray(image)
        if array.ndim != 3 or array.shape[2] not in (3, 4):
            raise ValueError("RGB input must have shape HxWx3 or HxWx4")
        if array.dtype != np.uint8:
            if not np.issubdtype(array.dtype, np.number) or not np.isfinite(array).all():
                raise ValueError("RGB input values must be finite numeric values")
            array = np.clip(array, 0, 255).astype(np.uint8)
        rgb = Image.fromarray(array[..., :3], mode="RGB")
    width, height = rgb.size
    if width <= 0 or height <= 0:
        raise ValueError("RGB input dimensions must be positive")
    scale = short_edge / float(min(width, height))
    resized = rgb.resize(
        (max(1, round(width * scale)), max(1, round(height * scale))),
        _resample_filter(),
    )
    output = BytesIO()
    resized.save(
        output, format="JPEG", quality=quality, optimize=False,
        progressive=False, subsampling=2,
    )
    return output.getvalue(), resized.width, resized.height


def _packet_jpeg(packet: FramePacket) -> tuple[bytes, int, int]:
    """Decode normal image packets; retain opaque synthetic fixtures verbatim."""
    try:
        from PIL import Image

        with Image.open(BytesIO(packet.image)) as image:
            image.load()
            return compress_rgb_image(image)
    except (OSError, ValueError):
        # Protocol fixtures intentionally use opaque byte strings. Real decoded
        # video packets always take the normalized RGB/JPEG path above.
        return packet.image, packet.width, packet.height


class RawFrameCache:
    """Content blobs plus independently owned timestamp references.

    ``ring``, ``active_segment`` and ``persistent`` may retain the same frame
    without duplicating JPEG bytes. Failed additions never mutate the store.
    """

    def __init__(self, max_bytes: int | None = None) -> None:
        if max_bytes is not None and max_bytes <= 0:
            raise ValueError("max_bytes must be positive")
        self.max_bytes = max_bytes
        self._blobs: dict[str, _Blob] = {}
        self._frames: dict[str, CachedFrame] = {}
        self._owners: dict[str, set[str]] = {}

    @staticmethod
    def _frame_id(packet: FramePacket) -> str:
        return f"{packet.frame_index:09d}.jpg"

    def _insert(
        self, packet: FramePacket, payload: bytes, width: int, height: int, *, owner: str
    ) -> CachedFrame:
        if not owner:
            raise ValueError("frame owner must not be empty")
        digest = hashlib.sha256(payload).hexdigest()
        frame_id = self._frame_id(packet)
        existing = self._frames.get(frame_id)
        if existing is not None:
            if existing.content_id != digest or existing.timestamp_s != packet.timestamp_s:
                raise ValueError(f"frame reference collision: {frame_id}")
            if owner not in self._owners[frame_id]:
                self._owners[frame_id].add(owner)
                self._blobs[digest].owner_count += 1
            return existing

        extra_bytes = 0 if digest in self._blobs else len(payload)
        if self.max_bytes is not None and self.byte_size + extra_bytes > self.max_bytes:
            raise MemoryError(
                f"adding {frame_id} would exceed shared frame budget {self.max_bytes} bytes"
            )
        cached = CachedFrame(
            frame_id, digest, float(packet.timestamp_s), int(packet.frame_index),
            packet.video_id, int(width), int(height),
        )
        if digest not in self._blobs:
            self._blobs[digest] = _Blob(payload)
        self._frames[frame_id] = cached
        self._owners[frame_id] = {owner}
        self._blobs[digest].owner_count += 1
        return cached

    def add(self, packet: FramePacket, *, owner: str = "persistent") -> CachedFrame:
        payload, width, height = _packet_jpeg(packet)
        return self._insert(packet, payload, width, height, owner=owner)

    def add_rgb(self, packet: FramePacket, rgb_image: Any, *, owner: str = "persistent") -> CachedFrame:
        payload, width, height = compress_rgb_image(rgb_image)
        return self._insert(packet, payload, width, height, owner=owner)

    def retain(self, frame_id: str, owner: str) -> CachedFrame:
        if frame_id not in self._frames:
            raise KeyError(frame_id)
        if not owner:
            raise ValueError("frame owner must not be empty")
        if owner not in self._owners[frame_id]:
            self._owners[frame_id].add(owner)
            self._blobs[self._frames[frame_id].content_id].owner_count += 1
        return self._frames[frame_id]

    def release(self, frame_id: str, owner: str | None = None) -> bool:
        cached = self._frames.get(frame_id)
        if cached is None:
            return False
        owners = self._owners[frame_id]
        removed = set(owners) if owner is None else ({owner} if owner in owners else set())
        if not removed:
            return False
        owners.difference_update(removed)
        blob = self._blobs[cached.content_id]
        blob.owner_count -= len(removed)
        if not owners:
            del self._owners[frame_id]
            del self._frames[frame_id]
        if blob.owner_count == 0:
            del self._blobs[cached.content_id]
        if blob.owner_count < 0:  # pragma: no cover - invariant guard
            raise RuntimeError("shared frame refcount became negative")
        return True

    def evict_oldest(self, *, exclude_owner_prefix: str | None = None) -> CachedFrame | None:
        candidates = [
            frame_id for frame_id, owners in self._owners.items()
            if exclude_owner_prefix is None or
            not any(owner.startswith(exclude_owner_prefix) for owner in owners)
        ]
        if not candidates:
            return None
        key = min(candidates, key=lambda value: (self._frames[value].timestamp_s, value))
        cached = self._frames[key]
        self.release(key)
        return cached

    def bytes_owned_by_prefix(self, prefix: str) -> int:
        if not prefix:
            raise ValueError("owner prefix must not be empty")
        content_ids = {
            self._frames[frame_id].content_id for frame_id, owners in self._owners.items()
            if any(owner.startswith(prefix) for owner in owners)
        }
        return sum(len(self._blobs[content_id].payload) for content_id in content_ids)

    def get(self, frame_id: str) -> bytes:
        cached = self._frames[frame_id]
        payload = self._blobs[cached.content_id].payload
        if hashlib.sha256(payload).hexdigest() != cached.content_id:
            raise ValueError(f"shared frame content hash mismatch: {frame_id}")
        return payload

    def items(self) -> list[tuple[str, bytes]]:
        """Return each content blob once; metadata maps frame refs to blobs."""
        return [(f"{digest}.jpg", self._blobs[digest].payload) for digest in sorted(self._blobs)]

    def metadata(self, frame_ids: set[str] | None = None) -> dict[str, dict[str, object]]:
        selected = self._frames if frame_ids is None else {
            key: value for key, value in self._frames.items() if key in frame_ids
        }
        return {
            key: {
                "blob": f"{value.content_id}.jpg",
                "timestamp_s": value.timestamp_s,
                "frame_index": value.frame_index,
                "video_id": value.video_id,
                "width": value.width,
                "height": value.height,
                "sha256": value.sha256,
                "owners": sorted(self._owners[key]),
            }
            for key, value in sorted(selected.items())
        }

    def owners(self, frame_id: str) -> tuple[str, ...]:
        return tuple(sorted(self._owners.get(frame_id, ())))

    def content_refcount(self, content_id: str) -> int:
        blob = self._blobs.get(content_id)
        return 0 if blob is None else blob.owner_count

    def __contains__(self, frame_id: str) -> bool:
        return frame_id in self._frames

    def __len__(self) -> int:
        return len(self._frames)

    @property
    def byte_size(self) -> int:
        return sum(len(blob.payload) for blob in self._blobs.values())
