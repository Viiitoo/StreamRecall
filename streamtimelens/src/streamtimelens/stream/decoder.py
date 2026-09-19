"""Single-pass frame sources.  Neither implementation supports seeking."""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Callable, Iterator

from streamtimelens.protocol.types import FramePacket, VideoMeta


class SinglePassPackets:
    """Decoder packet stream with auditable single-consumption semantics."""

    def __init__(self, factory: Callable[[], Iterator[FramePacket]]) -> None:
        self._factory = factory
        self._consumed = False
        self.decode_count = 0
        self.emitted_packet_count = 0
        self.last_frame_index: int | None = None
        self.seek_count = 0
        self.timestamp_mode = "unknown"
        self.pts_fallback_count = 0

    def __iter__(self) -> Iterator[FramePacket]:
        if self._consumed:
            raise RuntimeError("frame stream has already been consumed; second decode is forbidden")
        self._consumed = True
        for packet in self._factory():
            self.emitted_packet_count += 1
            self.last_frame_index = packet.frame_index
            yield packet

    def record_source_decode(self) -> None:
        self.decode_count += 1


def jsonl_packets(path: Path) -> tuple[VideoMeta, SinglePassPackets]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    if not rows:
        raise ValueError("frame stream is empty")
    meta = VideoMeta(str(rows[0].get("video_id", path.stem)), float(rows[-1]["timestamp_s"]),
                     float(rows[0].get("fps", 1.0)), len(rows))
    holder: dict[str, SinglePassPackets] = {}
    def iterate() -> Iterator[FramePacket]:
        for index, row in enumerate(rows):
            holder["stream"].record_source_decode()
            yield FramePacket(float(row["timestamp_s"]), int(row.get("frame_index", index)),
                              base64.b64decode(row.get("image_b64", "")), int(row.get("width", 0)), int(row.get("height", 0)),
                              "jsonl", str(row.get("video_id", meta.video_id)))
    stream = SinglePassPackets(iterate)
    holder["stream"] = stream
    stream.timestamp_mode = "jsonl_timestamp"
    return meta, stream


def mp4_packets(path: Path, sample_fps: float) -> tuple[VideoMeta, SinglePassPackets]:
    """Prefer Decord's sequential decoder; OpenCV remains a compatibility path."""
    try:
        import decord  # noqa: F401
    except ImportError:
        return _opencv_packets(path, sample_fps)
    return decord_packets(path, sample_fps)


def _jpeg_rgb(frame) -> bytes:
    from io import BytesIO
    from PIL import Image
    image = Image.fromarray(frame)
    output = BytesIO()
    image.save(output, format="JPEG", quality=85)
    return output.getvalue()


def decord_packets(path: Path, sample_fps: float) -> tuple[VideoMeta, SinglePassPackets]:
    """Sequential Decord decoder retaining original frame indices and PTS."""
    try:
        import decord
    except ImportError as exc:  # pragma: no cover - host dependent
        raise RuntimeError("MP4 decoding requires decord (or OpenCV compatibility backend)") from exc
    reader = decord.VideoReader(str(path), ctx=decord.cpu(0))
    fps, count = float(reader.get_avg_fps()), len(reader)
    if fps <= 0 or count <= 0:
        raise ValueError(f"video reports invalid metadata: {path}")
    try:
        final_timestamp = reader.get_frame_timestamp(count - 1)
        duration = float(final_timestamp[-1] if hasattr(final_timestamp, "__len__") else final_timestamp)
    except Exception:
        duration = count / fps
    meta = VideoMeta(path.stem, max(duration, count / fps), fps, count)
    stride, holder = max(1, round(fps / sample_fps)), {}

    def iterate() -> Iterator[FramePacket]:
        previous_timestamp = -1.0
        for index in range(count):
            # `next()` advances the decoder without accurate seeks or get_batch.
            frame = reader.next().asnumpy()
            holder["stream"].record_source_decode()
            if index % stride:
                continue
            try:
                pts = reader.get_frame_timestamp(index)
                timestamp_s = float(pts[0] if hasattr(pts, "__len__") else pts)
                if timestamp_s < previous_timestamp:
                    raise ValueError("non-monotonic PTS")
                holder["stream"].timestamp_mode = "decord_pts"
            except Exception:
                timestamp_s = index / fps
                holder["stream"].pts_fallback_count += 1
                holder["stream"].timestamp_mode = "frame_index_fallback"
            previous_timestamp = timestamp_s
            height, width = frame.shape[:2]
            yield FramePacket(timestamp_s, index, _jpeg_rgb(frame), width, height, "decord", meta.video_id)

    stream = SinglePassPackets(iterate)
    holder["stream"] = stream
    return meta, stream


def _opencv_packets(path: Path, sample_fps: float) -> tuple[VideoMeta, SinglePassPackets]:
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - host dependent
        raise RuntimeError("MP4 decoding requires opencv-python") from exc
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"cannot open video: {path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps <= 0:
        capture.release()
        raise ValueError(f"video reports invalid FPS: {path}")
    meta = VideoMeta(path.stem, count / fps, fps, count)
    stride = max(1, round(fps / sample_fps))
    holder: dict[str, SinglePassPackets] = {}
    def iterate() -> Iterator[FramePacket]:
        index = 0
        previous_timestamp = -1.0
        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                holder["stream"].record_source_decode()
                if index % stride == 0:
                    ok, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                    if not ok:
                        raise RuntimeError("JPEG encoding failed")
                    height, width = frame.shape[:2]
                    pts_s = float(capture.get(cv2.CAP_PROP_POS_MSEC)) / 1000.0
                    if pts_s > previous_timestamp:
                        timestamp_s = pts_s
                        holder["stream"].timestamp_mode = "container_pts"
                    else:
                        timestamp_s = index / fps
                        holder["stream"].pts_fallback_count += 1
                        holder["stream"].timestamp_mode = "mixed_pts_fallback" if holder["stream"].timestamp_mode == "container_pts" else "frame_index_fallback"
                    previous_timestamp = timestamp_s
                    yield FramePacket(timestamp_s, index, jpeg.tobytes(), width, height, "mp4", meta.video_id)
                index += 1
        finally:
            capture.release()
    stream = SinglePassPackets(iterate)
    holder["stream"] = stream
    stream.timestamp_mode = "frame_index_fallback"
    return meta, stream
