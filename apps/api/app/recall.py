"""Stable recall boundary shared by deterministic and model-backed workers."""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Protocol

from .demo import answer_query, candidate_payload


WORK_ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class BackendResult:
    answer: Dict[str, Any]
    candidates: List[Dict[str, Any]]


class RecallBackend(Protocol):
    name: str

    def run(
        self,
        *,
        session: Dict[str, Any],
        query_id: str,
        message: str,
        context: Dict[str, Any],
    ) -> BackendResult:
        ...


class DeterministicDemoBackend:
    """Model-free backend for UI development, CI and the public demo fixture."""

    name = "deterministic-demo"

    def run(
        self,
        *,
        session: Dict[str, Any],
        query_id: str,
        message: str,
        context: Dict[str, Any],
    ) -> BackendResult:
        del session, query_id
        answer = answer_query(message, context)
        return BackendResult(answer, candidate_payload(answer))


def _distributed(values: List[str], limit: int = 3) -> List[str]:
    if len(values) <= limit:
        return values
    indices = [round(index * (len(values) - 1) / (limit - 1)) for index in range(limit)]
    return [values[index] for index in indices]


def _trace(output: Any, kind: str) -> Dict[str, Any]:
    return next((dict(row) for row in output.query_trace if row.get("kind") == kind), {})


class HybridV3Backend:
    """Lazy, snapshot-only adapter around the frozen Hybrid V3 implementation."""

    name = "hybrid-v3"

    def __init__(self) -> None:
        self.config_path = Path(os.environ.get(
            "STREAMRECALL_HYBRID_CONFIG",
            WORK_ROOT / "streamtimelens/configs/exploration/clip_timelens_hybrid_v3.yaml",
        )).resolve()
        self.clip_device = os.environ.get("STREAMRECALL_CLIP_DEVICE")
        self.device_map = os.environ.get("STREAMRECALL_DEVICE_MAP", "auto")
        self._lock = threading.Lock()
        self._inference_lock = threading.Lock()
        self._config: Any = None
        self._encoder: Any = None
        self._service: Any = None

    def _load(self) -> None:
        if self._service is not None:
            return
        with self._lock:
            if self._service is not None:
                return
            from streamtimelens.config import read_hybrid_v3_config
            from streamtimelens.model_identity import verify_registered_model
            from streamtimelens.observer.clip_encoder import FrozenCLIPEncoder
            from streamtimelens.refiner.model_service import TimeLensModelService

            config = read_hybrid_v3_config(self.config_path)
            clip_path = (WORK_ROOT / config.retrieval.clip_model).resolve()
            model_path = (WORK_ROOT / config.brain.model).resolve()
            registry_path = WORK_ROOT / "artifacts/dev/model_hashes.json"
            verify_registered_model(
                registry_path, model_key="clip", model_root=clip_path,
                revision=config.retrieval.clip_revision,
                content_sha256=config.retrieval.clip_content_sha256,
                verify_files=True,
            )
            verify_registered_model(
                registry_path, model_key=config.brain.model_registry_key,
                model_root=model_path, revision=config.brain.model_revision,
                content_sha256=config.brain.model_content_sha256,
                verify_files=False,
            )
            self._config = config
            self._encoder = FrozenCLIPEncoder(clip_path, device=self.clip_device, batch_size=1)
            self._service = TimeLensModelService.get(model_path, device_map=self.device_map)

    def run(
        self,
        *,
        session: Dict[str, Any],
        query_id: str,
        message: str,
        context: Dict[str, Any],
    ) -> BackendResult:
        if session.get("source_type") == "demo":
            return DeterministicDemoBackend().run(
                session=session, query_id=query_id, message=message, context=context,
            )
        del context
        snapshot_path = session.get("snapshot_path_internal")
        if not snapshot_path:
            raise ValueError("Hybrid V3 requires a registered snapshot session")
        self._load()

        from streamtimelens.protocol.snapshot import SnapshotReader
        from streamtimelens.retrieval.clip_timelens_hybrid import answer_hybrid_snapshot

        snapshot = SnapshotReader(snapshot_path)
        started = time.perf_counter()
        # The frozen process-wide model service is not assumed to be re-entrant.
        # Queue GPU turns while allowing session and evidence APIs to stay concurrent.
        with self._inference_lock:
            output = answer_hybrid_snapshot(
                query_id=query_id, query=message, snapshot=snapshot,
                encoder=self._encoder, service=self._service, config=self._config,
            )
        total_ms = round((time.perf_counter() - started) * 1000)
        prediction = output.final_prediction
        span = prediction.get("span")
        inference = _trace(output, "timelens_inference")
        clip = _trace(output, "clip_retrieval")
        used_refs = list(prediction.get("evidence_ids") or output.selected_frame_refs)
        metadata = snapshot.read_frame_metadata()
        evidence = [
            {
                "frame_ref": ref,
                "timestamp_s": float(metadata[ref]["timestamp_s"]),
                "role": role,
            }
            for ref, role in zip(_distributed(used_refs), ("onset", "action", "offset"))
            if ref in metadata
        ]
        if span is None:
            status = "NOT_FOUND"
            message_text = "当前视觉记忆中没有找到足够可靠的对应事件。"
        else:
            status = "FOUND_WITH_FALLBACK" if output.fallback_used else "FOUND"
            message_text = f"事件发生在 {span[0]:.1f}s–{span[1]:.1f}s。"
        candidates = [
            {
                "candidate_id": row["candidate_id"],
                "start_s": float(row["start_s"]),
                "end_s": float(row["end_s"]),
                "score": float(row["score"]),
                "selected": row["candidate_id"] == output.selected_candidate_id,
            }
            for row in output.candidates
        ]
        alternatives = [
            {key: row[key] for key in ("candidate_id", "start_s", "end_s", "score")}
            for row in candidates if not row["selected"]
        ]
        calls = sum(
            int(row.get("calls", 0)) for row in output.query_trace
            if row.get("kind") in ("timelens_inference", "timelens_exception")
        )
        answer = {
            "status": status,
            "message": message_text,
            "resolved_query": message,
            "span": None if span is None else {"start_s": float(span[0]), "end_s": float(span[1])},
            "confidence": float(prediction.get("confidence") or 0.0),
            "evidence": evidence,
            "alternatives": alternatives,
            "reason": output.final_selection_reason,
            "limitations": ["答案只基于不可变快照中保留的视觉证据。"],
            "audit": {
                "snapshot_only": True,
                "future_frames_used": False,
                "model_calls": calls,
                "unique_frames_used": len(output.selected_frame_refs),
                "actual_byte_budget": snapshot.manifest.state_bytes <= snapshot.manifest.budget_bytes,
            },
            "cost": {
                "search_ms": round(float(clip.get("clip_query_wall_s") or 0) * 1000),
                "refine_ms": round(float(inference.get("wall_s") or 0) * 1000),
                "total_ms": total_ms,
                "generated_tokens": int(inference.get("generated_tokens") or 0),
            },
            "engine": {
                "backend": self.name,
                "readout": output.readout,
                "fallback_used": output.fallback_used,
            },
        }
        return BackendResult(answer, candidates)


def configured_backend() -> RecallBackend:
    selected = os.environ.get("STREAMRECALL_BACKEND", "demo").strip().lower()
    if selected in ("demo", "deterministic-demo"):
        return DeterministicDemoBackend()
    if selected in ("hybrid", "hybrid-v3"):
        return HybridV3Backend()
    raise ValueError(f"unsupported STREAMRECALL_BACKEND: {selected}")
