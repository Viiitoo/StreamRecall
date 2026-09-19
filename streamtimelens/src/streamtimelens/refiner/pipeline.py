"""Budgeted multi-candidate sparse local TimeLens refinement."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Literal, Sequence

from streamtimelens.evaluation.cost import ComponentTimer
from streamtimelens.protocol.snapshot import SnapshotReader
from streamtimelens.refiner.frame_assembler import FrameAssembly, assemble_refinement_frames
from streamtimelens.refiner.parse import parse_refiner_output
from streamtimelens.refiner.prompts import SPARSE_LOCAL_VERSION, build_grounding_prompt, video_messages
from streamtimelens.retrieval.candidates import RetrievalCandidate
from streamtimelens.retrieval.mmr import temporal_iou


AttemptStatus = Literal["ok", "parse_failure", "NO_VISUAL_EVIDENCE", "model_error"]
PipelineStatus = Literal["ok", "fallback", "not_found"]


@dataclass(frozen=True)
class RefinementAttempt:
    candidate_id: str
    candidate_span: tuple[float, float]
    status: AttemptStatus
    raw_answer: str
    parsed_span: tuple[float, float] | None
    parse_status: str | None
    evidence_overlap: float
    frame_refs: tuple[str, ...]
    resource: dict[str, Any]
    diagnostics: dict[str, Any]


@dataclass(frozen=True)
class RefinementResult:
    span: tuple[float, float] | None
    status: PipelineStatus
    selected_candidate_id: str | None
    attempts: tuple[RefinementAttempt, ...]
    model_calls: int


FrameAssembler = Callable[[SnapshotReader, RetrievalCandidate], FrameAssembly]


def refine_candidates(
    query: str,
    snapshot: SnapshotReader,
    candidates: Sequence[RetrievalCandidate],
    service: Any | None,
    *,
    max_calls: int,
    max_frames: int,
    max_new_tokens: int = 128,
    assembler: Callable[..., FrameAssembly] = assemble_refinement_frames,
) -> RefinementResult:
    if not query.strip() or max_calls < 0 or max_frames <= 0 or max_new_tokens <= 0:
        raise ValueError("invalid refinement request or budget")
    if not candidates:
        return RefinementResult(None, "not_found", None, (), 0)
    top = candidates[0]
    if max_calls == 0 or service is None:
        return RefinementResult(top.span, "fallback", top.candidate_id, (), 0)
    cards = {str(row["id"]): row for row in snapshot.read_cards()}
    attempts: list[RefinementAttempt] = []
    valid: list[tuple[RetrievalCandidate, tuple[float, float], float]] = []
    model_calls = 0
    for candidate in candidates:
        if model_calls >= max_calls:
            break
        assembly = assembler(snapshot, candidate, max_frames=max_frames)
        if assembly.status != "ok" or assembly.prepared is None:
            attempts.append(RefinementAttempt(
                candidate.candidate_id, candidate.span, "NO_VISUAL_EVIDENCE", "", None,
                None, 0.0, (), {}, dict(assembly.diagnostics),
            ))
            continue
        summaries = [
            str(cards[card_id].get("normalized_text") or cards[card_id].get("summary", ""))
            for card_id in candidate.contributing_card_ids if card_id in cards
        ]
        prompt = build_grounding_prompt(
            SPARSE_LOCAL_VERSION, query=query, candidate=candidate.span,
            card_summary=" | ".join(value for value in summaries if value) or None,
        )
        raw_answer = ""
        parsed_span = None
        parse_status = None
        status: AttemptStatus
        diagnostics = dict(assembly.diagnostics)
        torch_module = getattr(service, "torch", None)
        try:
            with ComponentTimer("timelens_local_refiner", torch_module=torch_module) as timer:
                model_calls += 1
                raw_answer = str(service.generate(
                    video_messages(prompt), [assembly.prepared.processor_video], max_new_tokens,
                ))
            parsed = parse_refiner_output(
                raw_answer, t_q=snapshot.manifest.t_q, candidate=candidate.span,
            )
            parsed_span = parsed.span
            parse_status = parsed.status
            diagnostics.update({
                "multiple_span": parsed.multiple_span,
                "model_stats": dict(getattr(service, "last_call_stats", {})),
            })
            if parsed_span is None:
                status = "parse_failure"
                overlap = 0.0
            else:
                status = "ok"
                overlap = temporal_iou(parsed_span, candidate.span)
                valid.append((candidate, parsed_span, overlap))
        except Exception as exc:
            status = "model_error"
            overlap = 0.0
            diagnostics.update({"error_type": type(exc).__name__, "error": str(exc)})
        attempts.append(RefinementAttempt(
            candidate.candidate_id, candidate.span, status, raw_answer, parsed_span,
            parse_status, overlap, assembly.frame_refs, timer.as_dict(), diagnostics,
        ))
    if not valid:
        return RefinementResult(top.span, "fallback", top.candidate_id, tuple(attempts), model_calls)
    selected_candidate, span, _ = min(
        valid, key=lambda item: (-item[0].score, -item[2], item[0].start_s, item[0].candidate_id),
    )
    return RefinementResult(
        span, "ok", selected_candidate.candidate_id, tuple(attempts), model_calls,
    )
