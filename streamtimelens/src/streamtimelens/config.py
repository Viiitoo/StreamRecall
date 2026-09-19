"""Typed, hash-stable configuration loading for protocol runs."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping

import yaml


_FORBIDDEN_KEYS = {
    "query", "queries", "query_id", "query_path", "query_text",
    "gt", "gt_path", "gt_span", "ground_truth", "ground_truth_span",
    "video_path", "video_root",
}


@dataclass(frozen=True)
class ProtocolConfig:
    decode_fps: float = 2.0
    decision_interval_s: float = 4.0
    ring_window_s: float = 12.0
    max_active_frames: int = 32
    trigger_threshold: float = 0.65
    minimum_gap_s: float = 1.0
    max_gap_s: float = 60.0
    lite_weight: float = 0.45
    semantic_weight: float = 0.35
    age_weight: float = 0.20
    initial_writer_tokens: float = 1.0
    segment_overlap_s: float = 1.0
    segment_bytes_per_frame: int = 16384
    hard_cut_hsv_threshold: float = 0.72
    hard_cut_ssim_threshold: float = 0.65
    black_luma_threshold: float = 2.0
    normalizer_alpha: float = 0.05
    normalizer_warmup_samples: int = 8
    normalizer_epsilon: float = 1e-6
    normalizer_z_threshold: float = 2.0
    warmup_hsv_threshold: float = 0.45
    warmup_ssim_threshold: float = 0.35
    warmup_flow_threshold: float = 1.5
    boundary_window_s: float = 2.0
    boundary_frames_per_side: int = 4
    boundary_internal_frames: int = 2
    forest_max_roots: int = 8
    merge_semantic_weight: float = 1.0
    merge_gap_weight: float = 0.35
    merge_boundary_weight: float = 0.35
    utility_novelty_weight: float = 0.30
    utility_boundary_weight: float = 0.30
    utility_inverse_density_weight: float = 0.25
    utility_has_raw_weight: float = 0.15
    coverage_bucket_base_s: float = 1.0
    forest_debug: bool = False
    rank_text_weight: float = 1.0
    rank_visual_weight: float = 0.0
    rank_boundary_bonus: float = 0.10
    rank_parent_penalty: float = 0.05
    mmr_semantic_penalty: float = 0.20
    mmr_temporal_penalty: float = 0.20
    confidence_top1_weight: float = 1.0
    confidence_margin_weight: float = 0.5
    confidence_raw_weight: float = 0.5
    confidence_parse_weight: float = 0.75
    confidence_bias: float = -0.5
    confidence_temperature: float = 1.0
    not_found_threshold: float = 0.20
    arrival_ratios: tuple[float, ...] = (0.25, 0.5, 0.75, 1.0)
    snapshot_format_version: int = 1
    query_visible_trace: bool = False

    def __post_init__(self) -> None:
        numeric = (
            self.decode_fps, self.decision_interval_s, self.ring_window_s,
            self.max_active_frames,
            self.trigger_threshold, self.minimum_gap_s, self.max_gap_s,
            self.lite_weight, self.semantic_weight, self.age_weight,
            self.initial_writer_tokens, self.segment_overlap_s,
            self.segment_bytes_per_frame,
            self.hard_cut_hsv_threshold, self.hard_cut_ssim_threshold,
            self.black_luma_threshold, self.normalizer_alpha,
            self.normalizer_warmup_samples, self.normalizer_epsilon,
            self.normalizer_z_threshold,
            self.warmup_hsv_threshold, self.warmup_ssim_threshold,
            self.warmup_flow_threshold, self.boundary_window_s,
            self.boundary_frames_per_side, self.boundary_internal_frames,
            self.forest_max_roots, self.merge_semantic_weight,
            self.merge_gap_weight, self.merge_boundary_weight,
            self.utility_novelty_weight, self.utility_boundary_weight,
            self.utility_inverse_density_weight, self.utility_has_raw_weight,
            self.coverage_bucket_base_s,
            self.rank_text_weight, self.rank_visual_weight,
            self.rank_boundary_bonus, self.rank_parent_penalty,
            self.mmr_semantic_penalty, self.mmr_temporal_penalty,
            self.confidence_top1_weight, self.confidence_margin_weight,
            self.confidence_raw_weight, self.confidence_parse_weight,
            self.confidence_bias, self.confidence_temperature,
            self.not_found_threshold,
            *self.arrival_ratios,
            self.snapshot_format_version,
        )
        if any(not math.isfinite(float(value)) for value in numeric):
            raise ValueError("protocol numeric values must be finite")
        if self.decode_fps <= 0 or self.decision_interval_s <= 0 or self.ring_window_s <= 0:
            raise ValueError("protocol times and decode_fps must be positive")
        if self.max_active_frames not in (8, 16, 32) or self.segment_bytes_per_frame <= 0:
            raise ValueError("active segment capacity/byte estimate is invalid")
        if self.minimum_gap_s < 0 or self.max_gap_s < self.minimum_gap_s:
            raise ValueError("trigger gaps are invalid")
        if self.trigger_threshold < 0 or min(self.lite_weight, self.semantic_weight, self.age_weight) < 0:
            raise ValueError("trigger threshold/weights are invalid")
        if not 0 <= self.initial_writer_tokens <= 1 or self.segment_overlap_s < 0:
            raise ValueError("quota initial tokens or segment overlap is invalid")
        if not 0 <= self.hard_cut_hsv_threshold <= 1 or not 0 <= self.hard_cut_ssim_threshold <= 2:
            raise ValueError("hard-cut thresholds are invalid")
        if self.black_luma_threshold < 0 or not 0 < self.normalizer_alpha <= 1:
            raise ValueError("visual/normalizer configuration is invalid")
        if self.normalizer_warmup_samples < 1 or self.normalizer_epsilon <= 0:
            raise ValueError("normalizer warmup/epsilon is invalid")
        if (self.boundary_window_s < 0 or self.boundary_frames_per_side < 0 or
                self.boundary_internal_frames < 0):
            raise ValueError("boundary allocator configuration is invalid")
        forest_weights = (
            self.merge_semantic_weight, self.merge_gap_weight, self.merge_boundary_weight,
            self.utility_novelty_weight, self.utility_boundary_weight,
            self.utility_inverse_density_weight, self.utility_has_raw_weight,
        )
        if (self.forest_max_roots < 1 or min(forest_weights) < 0 or
                sum(forest_weights[3:]) <= 0 or self.coverage_bucket_base_s <= 0):
            raise ValueError("forest merge/utility configuration is invalid")
        if min(
            self.rank_text_weight, self.rank_visual_weight,
            self.rank_boundary_bonus, self.rank_parent_penalty,
            self.mmr_semantic_penalty, self.mmr_temporal_penalty,
        ) < 0 or self.rank_text_weight + self.rank_visual_weight <= 0:
            raise ValueError("retrieval rank weights are invalid")
        if min(
            self.confidence_top1_weight, self.confidence_margin_weight,
            self.confidence_raw_weight, self.confidence_parse_weight,
        ) < 0 or self.confidence_temperature <= 0 or not 0 <= self.not_found_threshold <= 1:
            raise ValueError("confidence calibration configuration is invalid")
        if not self.arrival_ratios or any(value <= 0 or value > 1 for value in self.arrival_ratios):
            raise ValueError("arrival_ratios must be in (0, 1]")
        if tuple(sorted(self.arrival_ratios)) != self.arrival_ratios or len(set(self.arrival_ratios)) != len(self.arrival_ratios):
            raise ValueError("arrival_ratios must be strictly increasing")


@dataclass(frozen=True)
class BudgetConfig:
    memory_bytes: int
    writer_calls_per_minute: float
    refine_calls_per_query: int = 1
    max_frames_per_refine: int = 16

    def __post_init__(self) -> None:
        numeric = (
            self.memory_bytes, self.writer_calls_per_minute,
            self.refine_calls_per_query, self.max_frames_per_refine,
        )
        if any(not math.isfinite(float(value)) for value in numeric):
            raise ValueError("budget numeric values must be finite")
        if self.memory_bytes <= 0 or self.writer_calls_per_minute <= 0:
            raise ValueError("memory_bytes and writer_calls_per_minute must be positive")
        if self.refine_calls_per_query < 0 or self.max_frames_per_refine <= 0:
            raise ValueError("invalid refinement budget")


@dataclass(frozen=True)
class MethodConfig:
    method: str
    trigger: str
    writer: str
    retrieval: str
    raw_cache: str


@dataclass(frozen=True)
class ModelConfig:
    writer_model: str | None = None
    writer_revision: str | None = None
    text_embedder: str = "sentence-transformers/all-MiniLM-L6-v2"
    text_embedder_revision: str | None = None
    clip_model: str | None = None


@dataclass(frozen=True)
class ResolvedConfig:
    protocol: ProtocolConfig
    budget: BudgetConfig
    method: MethodConfig
    model: ModelConfig = field(default_factory=ModelConfig)

    def canonical_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def sha256(self) -> str:
        payload = json.dumps(
            self.canonical_dict(), sort_keys=True, separators=(",", ":"),
            ensure_ascii=False, allow_nan=False,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _require_sha256(name: str, value: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")


@dataclass(frozen=True)
class HybridSnapshotSourceConfig:
    frozen_config_id: str
    frozen_config_sha256: str
    snapshot_config_sha256: str
    reuse_frozen_snapshot: bool

    def __post_init__(self) -> None:
        if not self.frozen_config_id or not self.reuse_frozen_snapshot:
            raise ValueError("Hybrid V3 must identify and reuse a frozen snapshot")
        _require_sha256("frozen_config_sha256", self.frozen_config_sha256)
        _require_sha256("snapshot_config_sha256", self.snapshot_config_sha256)


@dataclass(frozen=True)
class HybridRetrievalConfig:
    clip_model: str
    clip_revision: str
    clip_content_sha256: str
    top_k: int
    merge_gap_s: float
    expand_neighbors: int
    candidate_margin_s: float
    coarse_margin_s: float
    max_candidate_clusters: int
    temporal_nms_iou: float
    min_cluster_separation_s: float
    merge_gap_tolerance_s: float = 0.0

    def __post_init__(self) -> None:
        numeric = (
            self.top_k, self.merge_gap_s, self.expand_neighbors,
            self.candidate_margin_s, self.coarse_margin_s,
            self.max_candidate_clusters, self.temporal_nms_iou,
            self.min_cluster_separation_s, self.merge_gap_tolerance_s,
        )
        if any(not math.isfinite(float(value)) for value in numeric):
            raise ValueError("Hybrid V3 retrieval values must be finite")
        if not self.clip_model or not self.clip_revision:
            raise ValueError("Hybrid V3 CLIP identity is required")
        _require_sha256("clip_content_sha256", self.clip_content_sha256)
        if (
            self.top_k <= 0 or self.merge_gap_s < 0 or self.expand_neighbors < 0
            or self.candidate_margin_s < 0 or self.coarse_margin_s < 0
            or self.max_candidate_clusters <= 0
            or not 0 <= self.temporal_nms_iou <= 1
            or self.min_cluster_separation_s < 0
            or self.merge_gap_tolerance_s < 0
        ):
            raise ValueError("Hybrid V3 retrieval parameters are invalid")


@dataclass(frozen=True)
class HybridBrainConfig:
    model: str
    model_registry_key: str
    model_revision: str
    model_content_sha256: str
    model_service_hashes: dict[str, str]
    readout: str
    prompt_version: str
    max_unique_frames: int
    max_new_tokens: int
    do_sample: bool
    calls_per_query: int
    total_visual_tokens: int
    frame_allocator: str = "hierarchical_4_4_3_3_1_1_v1"
    candidate_label_order: str = "score_v1"
    sparse_keep_max_frames: int = 7
    min_refined_candidate_ratio: float = 0.0

    def __post_init__(self) -> None:
        if not self.model or not self.model_registry_key or not self.model_revision:
            raise ValueError("Hybrid V3 brain identity is required")
        _require_sha256("model_content_sha256", self.model_content_sha256)
        if set(self.model_service_hashes) != {"model_sha256", "config_sha256", "tokenizer_sha256"}:
            raise ValueError("Hybrid V3 model service hashes are incomplete")
        for name, value in self.model_service_hashes.items():
            _require_sha256(name, value)
        supported_pairs = {
            ("timelens_multicandidate_v1", "sparse_multicandidate_v1"),
            ("timelens_multicandidate_keep_v2", "sparse_multicandidate_keep_v2"),
            ("timelens_multicandidate_adaptive_v3", "sparse_multicandidate_adaptive_v3"),
        }
        if (self.readout, self.prompt_version) not in supported_pairs:
            raise ValueError("Hybrid V3 readout/prompt version is unsupported")
        if self.frame_allocator not in (
            "hierarchical_4_4_3_3_1_1_v1", "boundary_anchored_v2",
        ):
            raise ValueError("Hybrid V3 frame allocator is unsupported")
        if self.candidate_label_order not in ("score_v1", "reverse_score_v1"):
            raise ValueError("Hybrid V3 candidate label order is unsupported")
        if (
            self.max_unique_frames < 0 or self.max_unique_frames > 16
            or self.sparse_keep_max_frames < 0 or self.sparse_keep_max_frames > 16
            or not math.isfinite(self.min_refined_candidate_ratio)
            or not 0 <= self.min_refined_candidate_ratio <= 1
            or self.max_new_tokens <= 0 or self.do_sample
            or self.calls_per_query != 1 or self.total_visual_tokens <= 0
        ):
            raise ValueError("Hybrid V3 brain budget must be deterministic and single-call")


@dataclass(frozen=True)
class HybridFallbackConfig:
    invalid_output: str
    exception: str
    accept_model_not_found: bool

    def __post_init__(self) -> None:
        if self.invalid_output != "clip_coarse" or self.exception != "clip_coarse":
            raise ValueError("Hybrid V3 invalid output and exceptions must fall back to CLIP coarse")


@dataclass(frozen=True)
class HybridV3Config:
    schema_version: int
    method: str
    snapshot_source: HybridSnapshotSourceConfig
    retrieval: HybridRetrievalConfig
    brain: HybridBrainConfig
    fallback: HybridFallbackConfig

    def __post_init__(self) -> None:
        if self.schema_version != 1 or self.method != "clip_timelens_hybrid_v3":
            raise ValueError("unsupported Hybrid V3 configuration")

    def canonical_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def sha256(self) -> str:
        payload = json.dumps(
            self.canonical_dict(), sort_keys=True, separators=(",", ":"),
            ensure_ascii=False, allow_nan=False,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _read_mapping(path: Path | str) -> dict[str, Any]:
    loaded = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"configuration must be a mapping: {path}")
    return loaded


def _reject_forbidden(value: Any, path: str = "") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if str(key).lower() in _FORBIDDEN_KEYS:
                raise ValueError(f"forbidden protocol key: {path}{key}")
            _reject_forbidden(child, f"{path}{key}.")
    elif isinstance(value, list):
        for child in value:
            _reject_forbidden(child, path)


def _construct(cls: type, data: Mapping[str, Any], label: str):
    accepted = set(cls.__dataclass_fields__)
    unknown = set(data) - accepted
    if unknown:
        raise ValueError(f"unknown {label} fields: {', '.join(sorted(unknown))}")
    normalized = dict(data)
    if cls is ProtocolConfig and "arrival_ratios" in normalized:
        normalized["arrival_ratios"] = tuple(float(item) for item in normalized["arrival_ratios"])
    return cls(**normalized)


def resolve_config(
    *, protocol_path: Path | str, budget_path: Path | str, method_path: Path | str,
    model_path: Path | str | None = None, overrides: Mapping[str, Mapping[str, Any]] | None = None,
) -> ResolvedConfig:
    """Load strict YAML layers and apply typed overrides before hashing."""
    layers: dict[str, dict[str, Any]] = {
        "protocol": _read_mapping(protocol_path), "budget": _read_mapping(budget_path), "method": _read_mapping(method_path),
        "model": _read_mapping(model_path) if model_path else {},
    }
    if overrides:
        for section, values in overrides.items():
            if section not in layers or not isinstance(values, Mapping):
                raise ValueError(f"unknown configuration section: {section}")
            layers[section].update(values)
    _reject_forbidden(layers["method"])
    return ResolvedConfig(
        protocol=_construct(ProtocolConfig, layers["protocol"], "protocol"),
        budget=_construct(BudgetConfig, layers["budget"], "budget"),
        method=_construct(MethodConfig, layers["method"], "method"),
        model=_construct(ModelConfig, layers["model"], "model"),
    )


def write_resolved_config(config: ResolvedConfig, path: Path | str) -> Path:
    """Persist the canonical config used for a run; the hash is reproducible."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(yaml.safe_dump(config.canonical_dict(), allow_unicode=True, sort_keys=True), encoding="utf-8")
    return destination


def read_resolved_config(path: Path | str) -> ResolvedConfig:
    data = _read_mapping(path)
    if set(data) != {"protocol", "budget", "method", "model"}:
        raise ValueError("resolved configuration has invalid sections")
    return ResolvedConfig(
        protocol=_construct(ProtocolConfig, data["protocol"], "protocol"),
        budget=_construct(BudgetConfig, data["budget"], "budget"),
        method=_construct(MethodConfig, data["method"], "method"),
        model=_construct(ModelConfig, data["model"], "model"),
    )


def read_hybrid_v3_config(path: Path | str) -> HybridV3Config:
    """Read the standalone, fully resolved experimental Hybrid V3 config."""
    data = _read_mapping(path)
    expected = {"schema_version", "method", "snapshot_source", "retrieval", "brain", "fallback"}
    if set(data) != expected:
        unknown = sorted(set(data) - expected)
        missing = sorted(expected - set(data))
        raise ValueError(f"Hybrid V3 configuration sections differ: missing={missing}, unknown={unknown}")
    for section in ("snapshot_source", "retrieval", "brain", "fallback"):
        if not isinstance(data[section], Mapping):
            raise ValueError(f"Hybrid V3 {section} must be a mapping")
    _reject_forbidden(data)
    return HybridV3Config(
        schema_version=int(data["schema_version"]), method=str(data["method"]),
        snapshot_source=_construct(
            HybridSnapshotSourceConfig, data["snapshot_source"], "Hybrid V3 snapshot_source",
        ),
        retrieval=_construct(
            HybridRetrievalConfig, data["retrieval"], "Hybrid V3 retrieval",
        ),
        brain=_construct(HybridBrainConfig, data["brain"], "Hybrid V3 brain"),
        fallback=_construct(HybridFallbackConfig, data["fallback"], "Hybrid V3 fallback"),
    )
