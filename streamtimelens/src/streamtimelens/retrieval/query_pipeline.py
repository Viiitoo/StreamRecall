"""End-to-end card retrieval, local refinement, and prediction serialization."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Literal

from streamtimelens.config import BudgetConfig, ProtocolConfig
from streamtimelens.protocol.snapshot import SnapshotReader
from streamtimelens.refiner.pipeline import refine_candidates
from streamtimelens.retrieval.candidates import expand_temporal_candidates
from streamtimelens.retrieval.confidence import (
    ConfidenceConfig, ConfidenceFeatures, calibrate_confidence,
)
from streamtimelens.retrieval.embedder import CardTextEmbedder, timed_query_embedding
from streamtimelens.retrieval.mmr import hierarchy_mmr
from streamtimelens.retrieval.ranker import RankWeights, hybrid_rank_cards


QueryStatus = Literal["ok", "fallback", "not_found", "error"]
_FORBIDDEN_OUTPUT_KEYS = {"gt", "gt_span", "ground_truth", "ground_truth_span"}


def _assert_no_ground_truth(value: Any) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if str(key).lower() in _FORBIDDEN_OUTPUT_KEYS:
                raise ValueError("prediction values cannot contain ground truth")
            _assert_no_ground_truth(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            _assert_no_ground_truth(child)


@dataclass(frozen=True)
class QueryOutput:
    query_id: str
    video_id: str
    rho_q: float
    status: QueryStatus
    span: tuple[float, float] | None
    confidence: float
    retrieved_cards: tuple[str, ...]
    candidates: tuple[str, ...]
    raw_answers: tuple[str, ...]
    resource: dict[str, Any]
    diagnostics: dict[str, Any]
    schema_version: int = 1

    def __post_init__(self) -> None:
        if not self.query_id or not self.video_id or self.schema_version != 1:
            raise ValueError("prediction identifiers/schema are invalid")
        if not 0 < self.rho_q <= 1 or not 0 <= self.confidence <= 1:
            raise ValueError("prediction rho/confidence is invalid")
        if self.status == "not_found" and self.span is not None:
            raise ValueError("NOT_FOUND prediction cannot contain a span")
        if self.status in ("ok", "fallback") and self.span is None:
            raise ValueError("successful/fallback prediction needs a span")
        if self.span is not None:
            start, end = self.span
            if not all(math.isfinite(value) for value in self.span) or start < 0 or start >= end:
                raise ValueError("prediction span is invalid")
        _assert_no_ground_truth(self.resource)
        _assert_no_ground_truth(self.diagnostics)

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["span"] = list(self.span) if self.span is not None else None
        result["retrieved_cards"] = list(self.retrieved_cards)
        result["candidates"] = list(self.candidates)
        result["raw_answers"] = list(self.raw_answers)
        _assert_no_ground_truth(result)
        return result


def _confidence_config(config: ProtocolConfig) -> ConfidenceConfig:
    return ConfidenceConfig(
        top1_weight=config.confidence_top1_weight,
        margin_weight=config.confidence_margin_weight,
        raw_weight=config.confidence_raw_weight,
        parse_weight=config.confidence_parse_weight,
        bias=config.confidence_bias,
        temperature=config.confidence_temperature,
        not_found_threshold=config.not_found_threshold,
    )


def answer_card_snapshot(
    *,
    query_id: str,
    query: str,
    snapshot: SnapshotReader,
    embedder: CardTextEmbedder,
    protocol: ProtocolConfig,
    budget: BudgetConfig,
    refiner_service: Any | None = None,
) -> QueryOutput:
    if not query_id or not query.strip():
        raise ValueError("query ID and text are required")
    embedding = timed_query_embedding(embedder, query)
    cards = snapshot.read_cards()
    ranked = hybrid_rank_cards(
        embedding.vector, cards,
        weights=RankWeights(
            text=protocol.rank_text_weight, visual=protocol.rank_visual_weight,
            boundary_bonus=protocol.rank_boundary_bonus,
            parent_penalty=protocol.rank_parent_penalty,
        ),
        upper_bound_s=snapshot.manifest.t_q,
    )
    rho_q = snapshot.manifest.t_q / float(snapshot.manifest.video_meta["duration_s"])
    if not ranked:
        return QueryOutput(
            query_id, snapshot.manifest.video_id, rho_q, "not_found", None, 0.0,
            (), (), (), {"query_embedding": embedding.resource},
            {"ranked_cards": [], "reason": "empty_snapshot"},
        )
    mmr = hierarchy_mmr(
        ranked, cards, semantic_penalty=protocol.mmr_semantic_penalty,
        temporal_penalty=protocol.mmr_temporal_penalty,
    )
    candidates = expand_temporal_candidates(
        mmr.choices, cards, t_q=snapshot.manifest.t_q,
    )
    refinement = refine_candidates(
        query, snapshot, candidates, refiner_service,
        max_calls=budget.refine_calls_per_query,
        max_frames=budget.max_frames_per_refine,
    )
    top1 = candidates[0].score if candidates else ranked[0].score
    top2 = candidates[1].score if len(candidates) > 1 else top1
    parse_feature = "not_attempted"
    if refinement.attempts:
        if any(attempt.status == "ok" for attempt in refinement.attempts):
            parse_feature = "ok"
        elif any(attempt.status == "parse_failure" for attempt in refinement.attempts):
            parse_feature = "failure"
        else:
            parse_feature = "fallback"
    features = ConfidenceFeatures(
        top1, max(0.0, top1 - top2),
        candidates[0].raw_completeness if candidates else 0.0,
        parse_feature,  # type: ignore[arg-type]
    )
    confidence = calibrate_confidence(features, _confidence_config(protocol))
    not_found = not candidates or confidence.not_found
    status: QueryStatus = "not_found" if not_found else refinement.status
    span = None if not_found else refinement.span
    attempts = [asdict(attempt) for attempt in refinement.attempts]
    resources = {
        "query_embedding": embedding.resource,
        "refiner": [attempt.resource for attempt in refinement.attempts],
        "refiner_model_calls": refinement.model_calls,
    }
    diagnostics = {
        "ranked_cards": [asdict(item) for item in ranked],
        "mmr": [asdict(item) for item in mmr.choices],
        "prefixes": {str(key): list(value) for key, value in mmr.prefixes.items()},
        "candidate_details": [asdict(item) for item in candidates],
        "refinement_attempts": attempts,
        "confidence": asdict(confidence),
    }
    return QueryOutput(
        query_id, snapshot.manifest.video_id, rho_q, status, span,
        confidence.confidence, tuple(choice.card_id for choice in mmr.choices),
        tuple(candidate.candidate_id for candidate in candidates),
        tuple(attempt.raw_answer for attempt in refinement.attempts), resources, diagnostics,
    )
