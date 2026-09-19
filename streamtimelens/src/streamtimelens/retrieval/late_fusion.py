"""Snapshot-only query-time late fusion for LF-01.

This module is deliberately independent from the closed JQ-01 implementation.
It reads every persisted temporal embedding (up to the frozen compute cap),
fuses the query with that immutable sequence, and scores a fixed candidate pool.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from streamtimelens.observer.clip_encoder import deserialize_embedding, l2_normalize
from streamtimelens.protocol.snapshot import SnapshotReader
from streamtimelens.retrieval.frame_candidates import FrameCandidate


LF_SCHEMA_VERSION = "lf_query_time_v2"
LF_ARCHITECTURE = "query-token-mha64h4-pairwise-gain-v2"
MAX_TEMPORAL_TOKENS = 256
MAX_EXTENT_CANDIDATES = 8
MAX_CANDIDATES = 1 + MAX_EXTENT_CANDIDATES
HIDDEN_DIM = 64
ATTENTION_HEADS = 4
BASELINE_CANDIDATE_ID = "frozen-v2-final"
LOSS_WEIGHTS = {
    "positive_saliency": 0.5,
    "negative_pair_saliency": 0.5,
    "quality": 1.0,
    "gain": 1.0,
    "listwise": 1.0,
    "relative_regret": 2.0,
}

GEOMETRY_FEATURES = (
    "start_norm",
    "end_norm",
    "center_norm",
    "width_norm",
    "iou_with_baseline",
    "center_distance_norm",
    "log_width_ratio",
    "is_frozen_baseline",
)

_FORBIDDEN_INFERENCE_KEYS = {
    "annotation", "annotations", "dataset", "dataset_name", "failure_layer",
    "good_baseline", "ground_truth", "ground_truth_span", "gt", "gt_span",
    "oracle", "oracle_iou", "source", "trace", "video_path", "video_root",
}


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")


def feature_schema() -> dict[str, Any]:
    """Return the frozen effect-free LF-01 reader contract."""
    return {
        "version": LF_SCHEMA_VERSION,
        "architecture": LF_ARCHITECTURE,
        "temporal_token_source": "verified-snapshot-frame-metadata-clip-embedding",
        "temporal_order": "timestamp-then-frame-ref",
        "temporal_coverage": "all-query-visible-tokens-or-fallback",
        "max_temporal_tokens": MAX_TEMPORAL_TOKENS,
        "hidden_dim": HIDDEN_DIM,
        "attention_heads": ATTENTION_HEADS,
        "candidate_pool": "frozen-v2-final-plus-frozen-x1-top8",
        "max_candidates": MAX_CANDIDATES,
        "geometry_features": list(GEOMETRY_FEATURES),
        "selection": "pairwise-gain-vs-zero-baseline-stable-argmax",
        "fallback": "bit-exact-frozen-v2-row",
    }


def schema_sha256(schema: Mapping[str, Any] | None = None) -> str:
    return hashlib.sha256(_canonical_json(dict(schema or feature_schema()))).hexdigest()


def write_feature_schema(path: Path | str) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(feature_schema(), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return destination


def _reject_forbidden(value: Any, path: str = "") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            lowered = str(key).lower()
            if lowered in _FORBIDDEN_INFERENCE_KEYS:
                raise ValueError(f"forbidden LF-01 inference field: {path}{key}")
            _reject_forbidden(child, f"{path}{key}.")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _reject_forbidden(child, f"{path}{index}.")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def snapshot_fingerprint(snapshot: SnapshotReader) -> tuple[tuple[str, int, str], ...]:
    """Hash the complete allow-listed snapshot without following external paths."""
    if not isinstance(snapshot, SnapshotReader):
        raise TypeError("LF-01 requires a verified SnapshotReader")
    rows = []
    for relative in sorted(snapshot.manifest.allowed_files):
        path = snapshot.path(relative)
        rows.append((relative, path.stat().st_size, _sha256(path)))
    return tuple(rows)


@dataclass(frozen=True)
class TemporalTokenSequence:
    frame_refs: tuple[str, ...]
    timestamps_s: tuple[float, ...]
    embeddings: tuple[tuple[float, ...], ...]
    duration_s: float
    t_q: float

    def __post_init__(self) -> None:
        count = len(self.frame_refs)
        dimensions = {len(row) for row in self.embeddings}
        if (
            not 1 <= count <= MAX_TEMPORAL_TOKENS
            or len(self.timestamps_s) != count
            or len(self.embeddings) != count
            or len(set(self.frame_refs)) != count
            or len(dimensions) != 1
            or next(iter(dimensions), 0) <= 0
            or not math.isfinite(self.duration_s)
            or self.duration_s <= 0
            or not math.isfinite(self.t_q)
            or not 0 <= self.t_q <= self.duration_s + 1e-6
            or not np.isfinite(np.asarray(self.timestamps_s, dtype=np.float64)).all()
            or not np.isfinite(np.asarray(self.embeddings, dtype=np.float64)).all()
            or any(t < 0 or t > self.t_q + 1e-6 for t in self.timestamps_s)
            or list(zip(self.timestamps_s, self.frame_refs))
            != sorted(zip(self.timestamps_s, self.frame_refs))
        ):
            raise ValueError("invalid LF-01 temporal token sequence")

    @property
    def embedding_dim(self) -> int:
        return len(self.embeddings[0])


@dataclass(frozen=True)
class LateFusionCandidate:
    candidate_id: str
    start_s: float
    end_s: float
    raw_score: float
    is_frozen_baseline: bool = False

    @property
    def span(self) -> tuple[float, float]:
        return self.start_s, self.end_s


def _validate_span(span: Sequence[float], duration_s: float, label: str) -> tuple[float, float]:
    if len(span) != 2:
        raise ValueError(f"{label} span must contain two values")
    start, end = map(float, span)
    if (
        not math.isfinite(start) or not math.isfinite(end)
        or start < 0 or end < start or end > duration_s + 1e-6
    ):
        raise ValueError(f"invalid {label} span")
    return start, end


def load_temporal_tokens(
    snapshot: SnapshotReader,
    query_embedding: Sequence[float],
) -> tuple[TemporalTokenSequence, tuple[float, ...]]:
    """Load all visible embeddings; never truncate an over-cap snapshot silently."""
    if not isinstance(snapshot, SnapshotReader):
        raise TypeError("LF-01 requires a verified SnapshotReader")
    return temporal_tokens_from_metadata(
        snapshot.read_frame_metadata(), query_embedding,
        duration_s=float(snapshot.manifest.video_meta["duration_s"]),
        t_q=float(snapshot.manifest.t_q),
    )


def temporal_tokens_from_metadata(
    metadata: Mapping[str, Mapping[str, Any]],
    query_embedding: Sequence[float],
    *,
    duration_s: float,
    t_q: float,
) -> tuple[TemporalTokenSequence, tuple[float, ...]]:
    """Materialize the same inference representation from verified metadata."""
    query = np.asarray(query_embedding, dtype=np.float32)
    if query.ndim != 1 or query.size == 0 or not np.isfinite(query).all():
        raise ValueError("invalid LF-01 query embedding")
    query = np.asarray(l2_normalize(query), dtype=np.float32)
    rows = []
    for frame_ref, raw in metadata.items():
        if "clip_embedding" not in raw:
            raise ValueError(f"snapshot token lacks clip embedding: {frame_ref}")
        timestamp = float(raw["timestamp_s"])
        embedding = np.asarray(
            deserialize_embedding(dict(raw["clip_embedding"])), dtype=np.float32,
        )
        if embedding.ndim != 1 or embedding.shape != query.shape or not np.isfinite(embedding).all():
            raise ValueError("snapshot/query embedding dimension mismatch")
        rows.append((timestamp, str(frame_ref), np.asarray(l2_normalize(embedding), dtype=np.float32)))
    rows.sort(key=lambda row: (row[0], row[1]))
    if not rows:
        raise ValueError("LF-01 needs at least one persisted temporal token")
    if len(rows) > MAX_TEMPORAL_TOKENS:
        raise ValueError(
            f"snapshot has {len(rows)} tokens, above LF-01 cap {MAX_TEMPORAL_TOKENS}"
        )
    sequence = TemporalTokenSequence(
        tuple(row[1] for row in rows),
        tuple(row[0] for row in rows),
        tuple(tuple(map(float, row[2])) for row in rows),
        duration_s,
        t_q,
    )
    return sequence, tuple(map(float, query))


@dataclass(frozen=True)
class LateFusionTrainingObservation:
    """Evaluation-owned targets around an inference-only LF representation."""

    observation_id: str
    video_id: str
    group_id: str
    sequence: TemporalTokenSequence
    query: tuple[float, ...]
    candidates: tuple[LateFusionCandidate, ...]
    quality_targets: tuple[float, ...]
    saliency_targets: tuple[float, ...]

    def __post_init__(self) -> None:
        if (
            not self.observation_id or not self.video_id or not self.group_id
            or len(self.query) != self.sequence.embedding_dim
            or not 1 <= len(self.candidates) <= MAX_CANDIDATES
            or len(self.quality_targets) != len(self.candidates)
            or len(self.saliency_targets) != len(self.sequence.frame_refs)
            or sum(candidate.is_frozen_baseline for candidate in self.candidates) != 1
            or len({candidate.candidate_id for candidate in self.candidates}) != len(self.candidates)
            or not np.isfinite(np.asarray(self.query, dtype=np.float64)).all()
            or not np.isfinite(np.asarray(self.quality_targets, dtype=np.float64)).all()
            or not np.isfinite(np.asarray(self.saliency_targets, dtype=np.float64)).all()
            or any(not 0 <= value <= 1 for value in self.quality_targets)
            or any(value not in (0.0, 1.0) for value in self.saliency_targets)
        ):
            raise ValueError("invalid LF-01 training observation")

    @property
    def candidate_ids(self) -> tuple[str, ...]:
        return tuple(row.candidate_id for row in self.candidates)

    @property
    def baseline_index(self) -> int:
        return next(i for i, row in enumerate(self.candidates) if row.is_frozen_baseline)


def make_training_observation(
    observation_id: str,
    video_id: str,
    group_id: str,
    sequence: TemporalTokenSequence,
    query: Sequence[float],
    candidates: Sequence[LateFusionCandidate],
    gt_span: Sequence[float],
) -> LateFusionTrainingObservation:
    """Create GT supervision only at the controlled training/evaluation boundary."""
    gt = _validate_span(gt_span, sequence.t_q, "ground-truth")
    candidates = tuple(candidates)
    quality = tuple(_temporal_iou(candidate.span, gt) for candidate in candidates)
    saliency = tuple(float(gt[0] <= timestamp <= gt[1]) for timestamp in sequence.timestamps_s)
    return LateFusionTrainingObservation(
        observation_id, video_id, group_id, sequence, tuple(map(float, query)),
        candidates, quality, saliency,
    )


def build_candidates(
    baseline_span: Sequence[float],
    extent_candidates: Sequence[FrameCandidate],
    *,
    duration_s: float,
) -> tuple[LateFusionCandidate, ...]:
    """Build a deterministic fixed pool without changing the frozen generator."""
    start, end = _validate_span(baseline_span, duration_s, "baseline")
    if len(extent_candidates) > MAX_EXTENT_CANDIDATES:
        raise ValueError("LF-01 received more than the frozen Top-8 extents")
    result = [LateFusionCandidate(BASELINE_CANDIDATE_ID, start, end, 0.0, True)]
    seen = {BASELINE_CANDIDATE_ID}
    ordered = sorted(extent_candidates, key=lambda row: (-float(row.score), str(row.candidate_id)))
    for raw in ordered:
        candidate_id = str(raw.candidate_id)
        if not candidate_id or candidate_id in seen:
            raise ValueError("LF-01 candidate IDs must be unique")
        candidate_start, candidate_end = _validate_span(raw.span, duration_s, candidate_id)
        score = float(raw.score)
        if not math.isfinite(score):
            raise ValueError("LF-01 candidate score must be finite")
        result.append(LateFusionCandidate(
            candidate_id, candidate_start, candidate_end, score, False,
        ))
        seen.add(candidate_id)
    return tuple(result)


def _temporal_iou(left: Sequence[float], right: Sequence[float]) -> float:
    intersection = max(0.0, min(left[1], right[1]) - max(left[0], right[0]))
    union = max(left[1], right[1]) - min(left[0], right[0])
    return intersection / union if union > 0 else 0.0


def candidate_geometry(
    candidates: Sequence[LateFusionCandidate], *, duration_s: float,
) -> np.ndarray:
    baseline = next(row for row in candidates if row.is_frozen_baseline)
    baseline_width = max(baseline.end_s - baseline.start_s, 1e-6)
    values = []
    for candidate in candidates:
        width = candidate.end_s - candidate.start_s
        center = (candidate.start_s + candidate.end_s) / 2
        baseline_center = (baseline.start_s + baseline.end_s) / 2
        values.append([
            candidate.start_s / duration_s,
            candidate.end_s / duration_s,
            center / duration_s,
            width / duration_s,
            _temporal_iou(candidate.span, baseline.span),
            abs(center - baseline_center) / duration_s,
            math.log((width + 1e-6) / (baseline_width + 1e-6)),
            float(candidate.is_frozen_baseline),
        ])
    matrix = np.asarray(values, dtype=np.float32)
    if matrix.shape != (len(candidates), len(GEOMETRY_FEATURES)) or not np.isfinite(matrix).all():
        raise ValueError("invalid LF-01 candidate geometry")
    return matrix


def candidate_mask(
    sequence: TemporalTokenSequence, candidates: Sequence[LateFusionCandidate],
) -> np.ndarray:
    timestamps = np.asarray(sequence.timestamps_s, dtype=np.float32)
    return np.asarray([
        (timestamps >= candidate.start_s) & (timestamps <= candidate.end_s)
        for candidate in candidates
    ], dtype=np.bool_)


def _torch():
    try:
        import torch
        from torch import nn
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("LF-01 requires the optional torch dependency") from exc
    return torch, nn


class QueryTimeLateFusionHead:
    """Bounded query-to-token attention plus token saliency and candidate quality."""

    def __init__(self, embedding_dim: int, *, seed: int = 20260911) -> None:
        if not isinstance(embedding_dim, int) or embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")
        torch, nn = _torch()
        torch.manual_seed(seed)
        self.embedding_dim = embedding_dim
        self.module = nn.ModuleDict({
            "video": nn.Linear(embedding_dim, HIDDEN_DIM),
            "query": nn.Linear(embedding_dim, HIDDEN_DIM),
            "time": nn.Sequential(nn.Linear(2, 16), nn.GELU(), nn.Linear(16, HIDDEN_DIM)),
            "attention": nn.MultiheadAttention(
                HIDDEN_DIM, ATTENTION_HEADS, dropout=0.0, batch_first=True,
            ),
            "token_norm": nn.LayerNorm(HIDDEN_DIM),
            "token_fusion": nn.Sequential(
                nn.Linear(3 * HIDDEN_DIM, HIDDEN_DIM), nn.GELU(), nn.LayerNorm(HIDDEN_DIM),
            ),
            "saliency": nn.Linear(HIDDEN_DIM, 1),
            "missing_candidate": nn.Embedding(1, HIDDEN_DIM),
            "quality": nn.Sequential(
                nn.Linear(2 * HIDDEN_DIM + len(GEOMETRY_FEATURES), HIDDEN_DIM),
                nn.GELU(), nn.Linear(HIDDEN_DIM, 1),
            ),
            "gain": nn.Sequential(
                nn.Linear(5 * HIDDEN_DIM + len(GEOMETRY_FEATURES), HIDDEN_DIM),
                nn.GELU(), nn.Linear(HIDDEN_DIM, 1),
            ),
        })

    def state_dict(self) -> dict[str, Any]:
        return self.module.state_dict()

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self.module.load_state_dict(state, strict=True)

    def __call__(
        self,
        embeddings: Any,
        query: Any,
        time_features: Any,
        masks: Any,
        geometry: Any,
    ) -> dict[str, Any]:
        torch, _ = _torch()
        if embeddings.ndim != 2 or query.ndim != 1 or embeddings.shape[1] != self.embedding_dim:
            raise ValueError("invalid LF-01 model tensors")
        batch = self.forward_batch(
            embeddings.unsqueeze(0), query.unsqueeze(0), time_features.unsqueeze(0),
            torch.ones((1, embeddings.shape[0]), dtype=torch.bool, device=embeddings.device),
            masks.unsqueeze(0), geometry.unsqueeze(0),
            torch.ones((1, geometry.shape[0]), dtype=torch.bool, device=embeddings.device),
        )
        result = {
            "attention_weights": batch["attention_weights"][0, :embeddings.shape[0]],
            "saliency_logits": batch["saliency_logits"][0, :embeddings.shape[0]],
            "candidate_quality_logits": batch["candidate_quality_logits"][0, :geometry.shape[0]],
        }
        for name in ("candidate_gain_logits", "selection_logits"):
            result[name] = batch[name][0, :geometry.shape[0]]
        return result

    def forward_batch(
        self, embeddings: Any, query: Any, time_features: Any, token_valid: Any,
        masks: Any, geometry: Any, candidate_valid: Any,
    ) -> dict[str, Any]:
        """Vectorized padded forward used by leakage-free whole-video training."""
        torch, _ = _torch()
        if (
            embeddings.ndim != 3 or query.ndim != 2 or time_features.ndim != 3
            or token_valid.ndim != 2 or masks.ndim != 3 or geometry.ndim != 3
            or candidate_valid.ndim != 2 or embeddings.shape[0] != query.shape[0]
            or embeddings.shape[2] != self.embedding_dim
        ):
            raise ValueError("invalid LF-01 batch tensors")
        video = self.module["video"](embeddings) + self.module["time"](time_features)
        query_state = self.module["query"](query).unsqueeze(1)
        attended, attention = self.module["attention"](
            query_state, video, video, key_padding_mask=~token_valid, need_weights=True,
        )
        global_state = attended[:, 0]
        repeated_query = query_state.expand(-1, video.shape[1], -1)
        repeated_global = global_state.unsqueeze(1).expand(-1, video.shape[1], -1)
        token_state = self.module["token_norm"](video) + self.module["token_fusion"](
            torch.cat((video, repeated_query, repeated_global), dim=2)
        )
        saliency = self.module["saliency"](token_state).squeeze(2)
        saliency = saliency.masked_fill(~token_valid, -1e4)
        masked_saliency = saliency.unsqueeze(1).masked_fill(~masks, -1e4)
        weights = torch.softmax(masked_saliency, dim=2) * masks.to(saliency.dtype)
        weights = weights / weights.sum(dim=2, keepdim=True).clamp_min(1e-12)
        pooled = torch.einsum("bct,bth->bch", weights, token_state)
        has_token = masks.any(dim=2)
        missing = self.module["missing_candidate"](
            torch.zeros(1, dtype=torch.long, device=embeddings.device)
        )[0].reshape(1, 1, HIDDEN_DIM)
        pooled = torch.where(has_token.unsqueeze(2), pooled, missing)
        repeated_global = global_state.unsqueeze(1).expand(-1, pooled.shape[1], -1)
        quality = self.module["quality"](
            torch.cat((pooled, repeated_global, geometry), dim=2)
        ).squeeze(2).masked_fill(~candidate_valid, -1e4)
        baseline_indices = geometry[:, :, -1].argmax(dim=1)
        batch_indices = torch.arange(geometry.shape[0], device=geometry.device)
        baseline = pooled[batch_indices, baseline_indices]
        repeated_baseline = baseline.unsqueeze(1).expand(-1, pooled.shape[1], -1)
        gain = self.module["gain"](torch.cat((
            pooled, repeated_baseline, pooled - repeated_baseline,
            pooled * repeated_baseline, repeated_global, geometry,
        ), dim=2)).squeeze(2).masked_fill(~candidate_valid, -1e4)
        selection = gain.clone()
        selection[batch_indices, baseline_indices] = 0.0
        return {
            "attention_weights": attention[:, 0],
            "saliency_logits": saliency,
            "candidate_quality_logits": quality,
            "candidate_gain_logits": gain,
            "selection_logits": selection,
        }


def tensorize(
    sequence: TemporalTokenSequence,
    query: Sequence[float],
    candidates: Sequence[LateFusionCandidate],
    *, device: str = "cpu",
) -> dict[str, Any]:
    torch, _ = _torch()
    embeddings = np.asarray(sequence.embeddings, dtype=np.float32)
    timestamps = np.asarray(sequence.timestamps_s, dtype=np.float32)
    denominator = max(sequence.t_q, 1e-6)
    time_features = np.stack((
        timestamps / sequence.duration_s,
        (sequence.t_q - timestamps) / denominator,
    ), axis=1).astype(np.float32)
    return {
        "embeddings": torch.as_tensor(embeddings, dtype=torch.float32, device=device),
        "query": torch.as_tensor(query, dtype=torch.float32, device=device),
        "time_features": torch.as_tensor(time_features, dtype=torch.float32, device=device),
        "masks": torch.as_tensor(candidate_mask(sequence, candidates), dtype=torch.bool, device=device),
        "geometry": torch.as_tensor(
            candidate_geometry(candidates, duration_s=sequence.duration_s),
            dtype=torch.float32, device=device,
        ),
    }


def _identity_sha(values: Sequence[str]) -> str:
    return hashlib.sha256(_canonical_json(sorted(set(map(str, values))))).hexdigest()


def tensorize_training_observations(
    observations: Sequence[LateFusionTrainingObservation], *, device: str = "cpu",
) -> dict[str, Any]:
    """Pad a fixed observation set and construct deterministic true negative pairs."""
    if not observations:
        raise ValueError("LF-01 training observations are empty")
    observations = tuple(observations)
    dimensions = {row.sequence.embedding_dim for row in observations}
    if len(dimensions) != 1:
        raise ValueError("LF-01 training embedding dimensions differ")
    batch_size = len(observations)
    token_count = max(len(row.sequence.frame_refs) for row in observations)
    candidate_count = max(len(row.candidates) for row in observations)
    embedding_dim = next(iter(dimensions))
    embeddings = np.zeros((batch_size, token_count, embedding_dim), dtype=np.float32)
    queries = np.zeros((batch_size, embedding_dim), dtype=np.float32)
    times = np.zeros((batch_size, token_count, 2), dtype=np.float32)
    token_valid = np.zeros((batch_size, token_count), dtype=np.bool_)
    masks = np.zeros((batch_size, candidate_count, token_count), dtype=np.bool_)
    geometry = np.zeros(
        (batch_size, candidate_count, len(GEOMETRY_FEATURES)), dtype=np.float32,
    )
    candidate_valid = np.zeros((batch_size, candidate_count), dtype=np.bool_)
    quality_targets = np.zeros((batch_size, candidate_count), dtype=np.float32)
    saliency_targets = np.zeros((batch_size, token_count), dtype=np.float32)
    baseline_indices = np.zeros(batch_size, dtype=np.int64)
    observations_per_video: dict[str, int] = defaultdict(int)
    for row in observations:
        observations_per_video[row.video_id] += 1
    observation_weights = np.asarray([
        1.0 / (len(observations_per_video) * observations_per_video[row.video_id])
        for row in observations
    ], dtype=np.float32)
    for index, row in enumerate(observations):
        tokens = len(row.sequence.frame_refs)
        candidates = len(row.candidates)
        embeddings[index, :tokens] = np.asarray(row.sequence.embeddings, dtype=np.float32)
        queries[index] = np.asarray(row.query, dtype=np.float32)
        timestamps = np.asarray(row.sequence.timestamps_s, dtype=np.float32)
        times[index, :tokens, 0] = timestamps / row.sequence.duration_s
        times[index, :tokens, 1] = (
            (row.sequence.t_q - timestamps) / max(row.sequence.t_q, 1e-6)
        )
        token_valid[index, :tokens] = True
        masks[index, :candidates, :tokens] = candidate_mask(row.sequence, row.candidates)
        geometry[index, :candidates] = candidate_geometry(
            row.candidates, duration_s=row.sequence.duration_s,
        )
        candidate_valid[index, :candidates] = True
        quality_targets[index, :candidates] = row.quality_targets
        saliency_targets[index, :tokens] = row.saliency_targets
        baseline_indices[index] = row.baseline_index
    negative_indices = []
    for index, row in enumerate(observations):
        choices = [
            offset for offset, other in enumerate(observations)
            if other.group_id != row.group_id and other.video_id != row.video_id
        ]
        if not choices:
            raise ValueError("LF-01 needs a true cross-video negative query")
        negative_indices.append(choices[index % len(choices)])
    torch, _ = _torch()
    def tensor(value: Any, dtype: Any) -> Any:
        return torch.as_tensor(value, dtype=dtype, device=device)
    query_tensor = tensor(queries, torch.float32)
    return {
        "model": {
            "embeddings": tensor(embeddings, torch.float32),
            "query": query_tensor,
            "time_features": tensor(times, torch.float32),
            "token_valid": tensor(token_valid, torch.bool),
            "masks": tensor(masks, torch.bool),
            "geometry": tensor(geometry, torch.float32),
            "candidate_valid": tensor(candidate_valid, torch.bool),
        },
        "negative_query": query_tensor[tensor(negative_indices, torch.long)],
        "quality_targets": tensor(quality_targets, torch.float32),
        "saliency_targets": tensor(saliency_targets, torch.float32),
        "baseline_indices": tensor(baseline_indices, torch.long),
        "observation_weights": tensor(observation_weights, torch.float32),
    }


def late_fusion_loss(
    model: QueryTimeLateFusionHead, batch: Mapping[str, Any],
) -> tuple[Any, dict[str, float]]:
    """Fixed positive/negative saliency, quality, listwise and regret objective."""
    torch, _ = _torch()
    outputs = model.forward_batch(**batch["model"])
    negative_inputs = {**batch["model"], "query": batch["negative_query"]}
    negative = model.forward_batch(**negative_inputs)
    token_valid = batch["model"]["token_valid"]
    candidate_valid = batch["model"]["candidate_valid"]
    weights = batch["observation_weights"]
    components: dict[str, list[Any]] = defaultdict(list)
    for index in range(token_valid.shape[0]):
        valid_tokens = token_valid[index]
        target_saliency = batch["saliency_targets"][index, valid_tokens]
        positive_logits = outputs["saliency_logits"][index, valid_tokens]
        parts = []
        for target_value in (1.0, 0.0):
            selected = target_saliency == target_value
            if bool(selected.any()):
                parts.append(torch.nn.functional.binary_cross_entropy_with_logits(
                    positive_logits[selected], target_saliency[selected], reduction="mean",
                ))
        components["positive_saliency"].append(torch.stack(parts).mean())
        components["negative_pair_saliency"].append(
            torch.nn.functional.binary_cross_entropy_with_logits(
                negative["saliency_logits"][index, valid_tokens],
                torch.zeros_like(target_saliency), reduction="mean",
            )
        )
        valid_candidates = candidate_valid[index]
        quality_logits = outputs["candidate_quality_logits"][index, valid_candidates]
        gain_logits = outputs["candidate_gain_logits"][index, valid_candidates]
        selection_logits = outputs["selection_logits"][index, valid_candidates]
        targets = batch["quality_targets"][index, valid_candidates]
        components["quality"].append(torch.nn.functional.smooth_l1_loss(
            torch.sigmoid(quality_logits), targets, reduction="mean",
        ))
        baseline = int(batch["baseline_indices"][index].item())
        gain_targets = targets - targets[baseline]
        components["gain"].append(torch.nn.functional.smooth_l1_loss(
            torch.tanh(gain_logits), gain_targets, reduction="mean",
        ))
        target_distribution = torch.softmax(targets / 0.1, dim=0)
        components["listwise"].append(
            -(target_distribution * torch.log_softmax(selection_logits, dim=0)).sum()
        )
        worse = gain_targets < 0
        worse[baseline] = False
        if bool(worse.any()):
            regret = (-gain_targets[worse]) * torch.nn.functional.softplus(
                selection_logits[worse] - selection_logits[baseline]
            )
            components["relative_regret"].append(regret.mean())
        else:
            components["relative_regret"].append(selection_logits.sum() * 0.0)
    reduced = {
        name: torch.stack(values).dot(weights) for name, values in components.items()
    }
    total = sum(LOSS_WEIGHTS[name] * reduced[name] for name in LOSS_WEIGHTS)
    return total, {
        name: float(value.detach().cpu().item()) for name, value in reduced.items()
    }


def _selected_video_equal_miou(
    model: QueryTimeLateFusionHead,
    observations: Sequence[LateFusionTrainingObservation], *, device: str,
) -> float:
    batch = tensorize_training_observations(observations, device=device)
    outputs = model.forward_batch(**batch["model"])
    scores = outputs["selection_logits"].detach().cpu().numpy()
    by_video: dict[str, list[float]] = defaultdict(list)
    for index, row in enumerate(observations):
        selected = stable_argmax(scores[index, :len(row.candidates)], row.candidate_ids)
        by_video[row.video_id].append(row.quality_targets[selected])
    return float(np.mean([np.mean(values) for values in by_video.values()]))


def train_late_fusion_head(
    training: Sequence[LateFusionTrainingObservation],
    validation: Sequence[LateFusionTrainingObservation], *,
    seed: int = 20260911, device: str = "cpu", max_epochs: int = 100,
    patience: int = 10, learning_rate: float = 1e-3, weight_decay: float = 1e-4,
    heldout_video_ids: Sequence[str] = (), heldout_group_ids: Sequence[str] = (),
) -> tuple[QueryTimeLateFusionHead, dict[str, Any]]:
    if (
        not training or not validation or max_epochs != 100 or patience != 10
        or learning_rate != 1e-3 or weight_decay != 1e-4
    ):
        raise ValueError("LF-01 training settings changed")
    train_videos = {row.video_id for row in training}
    val_videos = {row.video_id for row in validation}
    train_groups = {row.group_id for row in training}
    val_groups = {row.group_id for row in validation}
    if train_videos & val_videos or train_groups & val_groups:
        raise ValueError("LF-01 inner train and validation identities overlap")
    if (train_videos | val_videos) & set(heldout_video_ids) or (
        (train_groups | val_groups) & set(heldout_group_ids)
    ):
        raise ValueError("LF-01 outer-held-out identity entered training")
    dimensions = {row.sequence.embedding_dim for row in (*training, *validation)}
    if len(dimensions) != 1:
        raise ValueError("LF-01 embedding dimension changed")
    torch, _ = _torch()
    torch.manual_seed(seed)
    if hasattr(torch, "use_deterministic_algorithms"):
        torch.use_deterministic_algorithms(True)
    model = QueryTimeLateFusionHead(next(iter(dimensions)), seed=seed)
    model.module.to(device)
    optimizer = torch.optim.AdamW(
        model.module.parameters(), lr=learning_rate, weight_decay=weight_decay,
    )
    train_batch = tensorize_training_observations(training, device=device)
    best_state, best_score, best_epoch, stale = None, -math.inf, 0, 0
    history = []
    for epoch in range(1, max_epochs + 1):
        model.module.train()
        optimizer.zero_grad(set_to_none=True)
        loss, components = late_fusion_loss(model, train_batch)
        loss.backward()
        optimizer.step()
        model.module.eval()
        with torch.no_grad():
            score = _selected_video_equal_miou(model, validation, device=device)
        history.append({
            "epoch": epoch, "train_loss": float(loss.item()),
            "validation_candidate_miou": score, "loss_components": components,
        })
        if score > best_score + 1e-12:
            best_state = {
                name: value.detach().cpu().clone() for name, value in model.state_dict().items()
            }
            best_score, best_epoch, stale = score, epoch, 0
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state is None:
        raise RuntimeError("LF-01 failed to select an epoch")
    model.load_state_dict(best_state)
    model.module.to("cpu").eval()
    return model, {
        "best_epoch": best_epoch,
        "best_validation_candidate_miou": best_score,
        "epochs_ran": len(history),
        "history": history,
        "seed": seed,
        "device": device,
        "deterministic_algorithms": True,
        "optimizer": "AdamW",
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "max_epochs": max_epochs,
        "patience": patience,
        "training_weighting": "video_equal",
        "loss_weights": dict(LOSS_WEIGHTS),
    }


def fit_late_fusion_fixed_epochs(
    training: Sequence[LateFusionTrainingObservation], *, epochs: int,
    seed: int = 20260911, device: str = "cpu", learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
) -> QueryTimeLateFusionHead:
    if (
        not training or not 1 <= int(epochs) <= 100
        or learning_rate != 1e-3 or weight_decay != 1e-4
    ):
        raise ValueError("invalid fixed-epoch LF-01 training request")
    torch, _ = _torch()
    torch.manual_seed(seed)
    if hasattr(torch, "use_deterministic_algorithms"):
        torch.use_deterministic_algorithms(True)
    model = QueryTimeLateFusionHead(training[0].sequence.embedding_dim, seed=seed)
    model.module.to(device)
    optimizer = torch.optim.AdamW(
        model.module.parameters(), lr=learning_rate, weight_decay=weight_decay,
    )
    batch = tensorize_training_observations(training, device=device)
    for _ in range(int(epochs)):
        optimizer.zero_grad(set_to_none=True)
        loss, _ = late_fusion_loss(model, batch)
        loss.backward()
        optimizer.step()
    model.module.to("cpu").eval()
    return model


def stable_argmax(values: Sequence[float], candidate_ids: Sequence[str]) -> int:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or len(array) != len(candidate_ids) or not np.isfinite(array).all():
        raise ValueError("invalid LF-01 ranking values")
    best = float(array.max())
    return min(
        (index for index, value in enumerate(array) if float(value) == best),
        key=lambda index: str(candidate_ids[index]),
    )


def _serialize_state_dict(state: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value.detach().cpu().tolist() for key, value in state.items()}


def _checkpoint_digest(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(dict(payload))).hexdigest()


def make_checkpoint(
    model: QueryTimeLateFusionHead,
    *, code_revision: str,
    outer_train_video_sha256: str,
    outer_train_group_sha256: str,
) -> dict[str, Any]:
    payload = {
        "format_version": 1,
        "state_dict": _serialize_state_dict(model.state_dict()),
        "schema": feature_schema(),
        "schema_sha256": schema_sha256(),
        "architecture": LF_ARCHITECTURE,
        "embedding_dim": model.embedding_dim,
        "code_revision": str(code_revision),
        "outer_train_video_sha256": str(outer_train_video_sha256),
        "outer_train_group_sha256": str(outer_train_group_sha256),
    }
    return {**payload, "checkpoint_sha256": _checkpoint_digest(payload)}


def save_checkpoint(checkpoint: Mapping[str, Any], path: Path | str) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(dict(checkpoint), sort_keys=True, allow_nan=False) + "\n", encoding="utf-8",
    )
    return destination


def load_checkpoint(
    path: Path | str, *, expected_embedding_dim: int,
) -> tuple[QueryTimeLateFusionHead, dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    required = {
        "format_version", "state_dict", "schema", "schema_sha256", "architecture",
        "embedding_dim", "code_revision", "outer_train_video_sha256",
        "outer_train_group_sha256", "checkpoint_sha256",
    }
    if set(payload) != required:
        raise ValueError("LF-01 checkpoint fields mismatch")
    digest_payload = {key: value for key, value in payload.items() if key != "checkpoint_sha256"}
    if (
        payload["format_version"] != 1
        or payload["schema"] != feature_schema()
        or payload["schema_sha256"] != schema_sha256()
        or payload["architecture"] != LF_ARCHITECTURE
        or int(payload["embedding_dim"]) != expected_embedding_dim
        or payload["checkpoint_sha256"] != _checkpoint_digest(digest_payload)
    ):
        raise ValueError("LF-01 checkpoint contract mismatch")
    torch, _ = _torch()
    model = QueryTimeLateFusionHead(int(payload["embedding_dim"]))
    current = model.state_dict()
    serialized = payload["state_dict"]
    if set(serialized) != set(current):
        raise ValueError("LF-01 checkpoint state keys mismatch")
    state = {
        key: torch.as_tensor(serialized[key], dtype=current[key].dtype).reshape(current[key].shape)
        for key in current
    }
    model.load_state_dict(state)
    model.module.eval()
    return model, payload


def select_late_fusion(
    baseline_row: Mapping[str, Any],
    snapshot: SnapshotReader,
    query_embedding: Sequence[float],
    extent_candidates: Sequence[FrameCandidate],
    *, checkpoint_path: Path | str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run LF-01 or return an untouched deep copy of Frozen V2 on any error."""
    fallback = copy.deepcopy(dict(baseline_row))
    debug: dict[str, Any] = {
        "method": "LF-01", "schema_version": LF_SCHEMA_VERSION,
        "fallback": True, "fallback_reason": None,
    }
    try:
        _reject_forbidden(baseline_row)
        if not isinstance(snapshot, SnapshotReader):
            raise TypeError("LF-01 requires a verified SnapshotReader")
        before = snapshot_fingerprint(snapshot)
        sequence, query = load_temporal_tokens(snapshot, query_embedding)
        baseline_span = baseline_row.get("final_span")
        if not isinstance(baseline_span, (list, tuple)):
            raise ValueError("Frozen V2 row lacks final_span")
        candidates = build_candidates(
            baseline_span, extent_candidates, duration_s=sequence.duration_s,
        )
        model, checkpoint = load_checkpoint(
            checkpoint_path, expected_embedding_dim=sequence.embedding_dim,
        )
        tensors = tensorize(sequence, query, candidates)
        torch, _ = _torch()
        with torch.no_grad():
            outputs = model(**tensors)
        quality = outputs["candidate_quality_logits"].detach().cpu().numpy()
        gain = outputs["candidate_gain_logits"].detach().cpu().numpy()
        selection_logits = outputs["selection_logits"].detach().cpu().numpy()
        selected = stable_argmax(selection_logits, [row.candidate_id for row in candidates])
        after = snapshot_fingerprint(SnapshotReader(snapshot.root))
        if before != after:
            raise RuntimeError("LF-01 mutated the immutable snapshot")
        chosen = candidates[selected]
        result = fallback if chosen.is_frozen_baseline else copy.deepcopy(fallback)
        if not chosen.is_frozen_baseline:
            result["final_span"] = [float(chosen.start_s), float(chosen.end_s)]
            result["lf01_selected_candidate_id"] = chosen.candidate_id
        saliency = outputs["saliency_logits"].detach().cpu().tolist()
        attention = outputs["attention_weights"].detach().cpu().tolist()
        ranks = sorted(
            range(len(candidates)),
            key=lambda i: (-float(selection_logits[i]), candidates[i].candidate_id),
        )
        rank_by_index = {index: rank + 1 for rank, index in enumerate(ranks)}
        debug.update({
            "fallback": False,
            "checkpoint_sha256": checkpoint["checkpoint_sha256"],
            "temporal_token_count": len(sequence.frame_refs),
            "temporal_tokens": [
                {
                    "frame_ref": frame_ref,
                    "timestamp_s": timestamp,
                    "attention": float(attention[index]),
                    "saliency_logit": float(saliency[index]),
                }
                for index, (frame_ref, timestamp) in enumerate(
                    zip(sequence.frame_refs, sequence.timestamps_s)
                )
            ],
            "candidates": [
                {
                    **asdict(candidate),
                    "quality_logit": float(quality[index]),
                    "gain_logit": float(gain[index]),
                    "selection_logit": float(selection_logits[index]),
                    "rank": rank_by_index[index],
                    "selected": index == selected,
                }
                for index, candidate in enumerate(candidates)
            ],
            "selected_candidate_id": chosen.candidate_id,
            "snapshot_unchanged": True,
        })
        _reject_forbidden(debug)
        return result, debug
    except Exception as exc:
        debug["fallback_reason"] = f"{type(exc).__name__}: {exc}"
        return fallback, debug
