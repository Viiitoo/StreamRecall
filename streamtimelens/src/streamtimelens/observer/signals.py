"""Causal, query-blind visual change signals for the streaming observer."""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from typing import Any, Callable


@dataclass(frozen=True)
class LiteVisualSignals:
    """Signals comparing the current frame with the immediately prior frame.

    Missing measurements are represented by ``None`` and accompanied by an
    explicit status.  This keeps dependency and input failures out of the
    numerical trigger path instead of silently treating them as no motion.
    """

    timestamp_s: float
    hsv_distance: float | None
    ssim_change: float | None
    flow_mean: float | None
    flow_p90: float | None
    hard_cut: bool
    statuses: tuple[str, ...] = ()

    @property
    def available(self) -> bool:
        return self.hsv_distance is not None and self.ssim_change is not None


def _rgb_array(image: Any) -> Any:
    import numpy as np
    from PIL import Image

    if image is None:
        raise ValueError("missing_frame")
    if isinstance(image, bytes):
        if not image:
            raise ValueError("missing_frame")
        try:
            with Image.open(BytesIO(image)) as decoded:
                decoded.load()
                return np.asarray(decoded.convert("RGB"), dtype=np.uint8)
        except OSError as exc:
            raise ValueError("decode_error") from exc
    if isinstance(image, Image.Image):
        return np.asarray(image.convert("RGB"), dtype=np.uint8)
    array = np.asarray(image)
    if array.ndim != 3 or array.shape[2] not in (3, 4) or array.size == 0:
        raise ValueError("invalid_frame")
    if not np.issubdtype(array.dtype, np.number) or not np.isfinite(array).all():
        raise ValueError("invalid_frame")
    return np.clip(array[..., :3], 0, 255).astype(np.uint8)


def _resize_rgb(image: Any, shape: tuple[int, int]) -> Any:
    import numpy as np
    from PIL import Image

    height, width = shape
    resized = Image.fromarray(image, mode="RGB").resize((width, height), Image.BILINEAR)
    return np.asarray(resized, dtype=np.uint8)


def _rgb_to_hsv(image: Any) -> Any:
    """Vectorized RGB-to-HSV with all channels normalized to [0, 1]."""
    import numpy as np

    rgb = image.astype(np.float32) / 255.0
    maximum = rgb.max(axis=2)
    minimum = rgb.min(axis=2)
    delta = maximum - minimum
    hue = np.zeros_like(maximum)
    nonzero = delta > 1e-12
    red = nonzero & (maximum == rgb[..., 0])
    green = nonzero & (maximum == rgb[..., 1])
    blue = nonzero & (maximum == rgb[..., 2])
    hue[red] = ((rgb[..., 1][red] - rgb[..., 2][red]) / delta[red]) % 6.0
    hue[green] = (rgb[..., 2][green] - rgb[..., 0][green]) / delta[green] + 2.0
    hue[blue] = (rgb[..., 0][blue] - rgb[..., 1][blue]) / delta[blue] + 4.0
    hue /= 6.0
    saturation = np.divide(delta, maximum, out=np.zeros_like(delta), where=maximum > 1e-12)
    return np.stack((hue, saturation, maximum), axis=2)


def hsv_histogram_distance(previous: Any, current: Any) -> float:
    """Hellinger distance between normalized 16x4x4 HSV histograms."""
    import numpy as np

    bins = (16, 4, 4)
    ranges = ((0.0, 1.0),) * 3
    left, _ = np.histogramdd(_rgb_to_hsv(previous).reshape(-1, 3), bins=bins, range=ranges)
    right, _ = np.histogramdd(_rgb_to_hsv(current).reshape(-1, 3), bins=bins, range=ranges)
    left /= max(float(left.sum()), 1.0)
    right /= max(float(right.sum()), 1.0)
    coefficient = float(np.sqrt(left * right).sum())
    return float(np.sqrt(max(0.0, 1.0 - min(coefficient, 1.0))))


def ssim_change(previous: Any, current: Any, *, data_range: float = 255.0) -> float:
    """Return one minus global luminance SSIM, bounded to [0, 2]."""
    import numpy as np

    weights = np.asarray((0.299, 0.587, 0.114), dtype=np.float64)
    left = np.tensordot(previous.astype(np.float64), weights, axes=([2], [0]))
    right = np.tensordot(current.astype(np.float64), weights, axes=([2], [0]))
    mean_left, mean_right = float(left.mean()), float(right.mean())
    variance_left, variance_right = float(left.var()), float(right.var())
    covariance = float(((left - mean_left) * (right - mean_right)).mean())
    c1, c2 = (0.01 * data_range) ** 2, (0.03 * data_range) ** 2
    numerator = (2 * mean_left * mean_right + c1) * (2 * covariance + c2)
    denominator = (mean_left ** 2 + mean_right ** 2 + c1) * (variance_left + variance_right + c2)
    similarity = numerator / denominator if denominator else 1.0
    return float(min(2.0, max(0.0, 1.0 - similarity)))


def _farneback(previous_gray: Any, current_gray: Any) -> Any:
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - depends on inference image
        raise RuntimeError("opencv_unavailable") from exc
    return cv2.calcOpticalFlowFarneback(
        previous_gray, current_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0,
    )


class LiteSignalObserver:
    """Compute lite visual signals once per observed frame, without lookahead."""

    def __init__(
        self,
        *,
        hard_cut_hsv_threshold: float = 0.72,
        hard_cut_ssim_threshold: float = 0.65,
        black_luma_threshold: float = 2.0,
        flow_backend: Callable[[Any, Any], Any] | None = None,
    ) -> None:
        if not 0 <= hard_cut_hsv_threshold <= 1 or not 0 <= hard_cut_ssim_threshold <= 2:
            raise ValueError("hard-cut thresholds are outside signal ranges")
        if black_luma_threshold < 0:
            raise ValueError("black luma threshold must be non-negative")
        self.hard_cut_hsv_threshold = hard_cut_hsv_threshold
        self.hard_cut_ssim_threshold = hard_cut_ssim_threshold
        self.black_luma_threshold = black_luma_threshold
        self.flow_backend = flow_backend or _farneback
        self._previous: Any | None = None
        self._last_timestamp: float | None = None

    def observe(self, image: Any, timestamp_s: float) -> LiteVisualSignals:
        import numpy as np

        if timestamp_s < 0 or (self._last_timestamp is not None and timestamp_s < self._last_timestamp):
            raise ValueError("signal timestamps must be non-negative and monotonic")
        self._last_timestamp = float(timestamp_s)
        statuses: list[str] = []
        try:
            current = _rgb_array(image)
        except ValueError as exc:
            # A missing/corrupt current frame must not replace the last valid
            # frame, so the next valid comparison remains meaningful.
            return LiteVisualSignals(float(timestamp_s), None, None, None, None, False, (str(exc),))

        if float(current.mean()) <= self.black_luma_threshold:
            statuses.append("black_frame")
        if self._previous is None:
            self._previous = current
            statuses.append("initial_frame")
            return LiteVisualSignals(float(timestamp_s), None, None, None, None, False, tuple(statuses))

        previous = self._previous
        self._previous = current
        if previous.shape[:2] != current.shape[:2]:
            statuses.append("resolution_change")
            previous = _resize_rgb(previous, current.shape[:2])
        histogram = hsv_histogram_distance(previous, current)
        structural = ssim_change(previous, current)
        gray_weights = np.asarray((0.299, 0.587, 0.114), dtype=np.float32)
        previous_gray = np.tensordot(previous, gray_weights, axes=([2], [0])).astype(np.uint8)
        current_gray = np.tensordot(current, gray_weights, axes=([2], [0])).astype(np.uint8)
        flow_mean: float | None = None
        flow_p90: float | None = None
        try:
            flow = np.asarray(self.flow_backend(previous_gray, current_gray), dtype=np.float32)
            if flow.shape != (*current_gray.shape, 2) or not np.isfinite(flow).all():
                raise ValueError("invalid_flow")
            magnitude = np.linalg.norm(flow, axis=2)
            flow_mean = float(magnitude.mean())
            flow_p90 = float(np.percentile(magnitude, 90))
        except RuntimeError as exc:
            statuses.append(str(exc))
        except ValueError as exc:
            statuses.append(str(exc))
        hard_cut = histogram >= self.hard_cut_hsv_threshold and structural >= self.hard_cut_ssim_threshold
        return LiteVisualSignals(
            float(timestamp_s), histogram, structural, flow_mean, flow_p90, hard_cut, tuple(statuses),
        )
