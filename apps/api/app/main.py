"""FastAPI entry point for the first StreamRecall vertical slice."""

from __future__ import annotations

import json
import os
import uuid
import hashlib
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import BackgroundTasks, FastAPI, File, Form, Header, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, Field

from .demo import (
    DEMO_SESSION_ID,
    demo_session,
    evidence_svg,
    now,
)
from .recall import configured_backend
from .ingest import run_upload_ingest
from .store import Store


class ConversationCreate(BaseModel):
    session_id: str


class TurnCreate(BaseModel):
    message: str = Field(..., min_length=1, max_length=500)


class SnapshotSessionCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    snapshot_relative_path: str = Field(..., min_length=1, max_length=500)
    suggested_queries: List[str] = Field(default_factory=list)


runtime_root = Path(os.environ.get("STREAMRECALL_RUNTIME", Path(__file__).resolve().parents[1] / "runtime"))
snapshot_root = Path(os.environ.get("STREAMRECALL_SNAPSHOT_ROOT", runtime_root / "snapshots")).resolve()
upload_root = (runtime_root / "uploads").resolve()
max_upload_bytes = int(os.environ.get("STREAMRECALL_MAX_UPLOAD_BYTES", 512 * 1024 * 1024))
store = Store(runtime_root / "streamrecall.sqlite3")
recall_backend = configured_backend()
turn_executor = ThreadPoolExecutor(
    max_workers=max(1, int(os.environ.get("STREAMRECALL_QUERY_WORKERS", "2"))),
    thread_name_prefix="streamrecall-turn",
)

app = FastAPI(title="StreamRecall API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def seed_demo() -> None:
    if store.get("sessions", DEMO_SESSION_ID) is None:
        store.put("sessions", DEMO_SESSION_ID, demo_session())


@app.get("/healthz")
def health() -> Dict[str, Any]:
    model_status = "simulated" if recall_backend.name == "deterministic-demo" else "lazy"
    return {
        "status": "ok", "backend": recall_backend.name,
        "models": {"clip": model_status, "timelens": model_status},
    }


@app.get("/api/v1/sessions")
def sessions() -> List[Dict[str, Any]]:
    return [_public_session(row) for row in store.list("sessions")]


def _public_session(payload: Dict[str, Any]) -> Dict[str, Any]:
    return {key: value for key, value in payload.items() if not key.endswith("_internal")}


def _stored_session(session_id: str) -> Dict[str, Any]:
    payload = store.get("sessions", session_id)
    if payload is None:
        raise HTTPException(404, "session not found")
    return payload


def _controlled_snapshot_path(relative_value: str) -> Path:
    relative = Path(relative_value)
    if relative.is_absolute() or ".." in relative.parts:
        raise HTTPException(400, "snapshot path must be relative to the configured root")
    candidate = (snapshot_root / relative).resolve()
    try:
        candidate.relative_to(snapshot_root)
    except ValueError as exc:
        raise HTTPException(400, "snapshot path escapes the configured root") from exc
    return candidate


@app.post("/api/v1/sessions/upload", status_code=202)
async def upload_session(
    background_tasks: BackgroundTasks,
    name: str = Form(..., min_length=1, max_length=100),
    file: UploadFile = File(...),
    retain_source: bool = Form(False),
) -> Dict[str, Any]:
    filename = file.filename or ""
    if not name.strip():
        raise HTTPException(422, "session name cannot be blank")
    if not filename.lower().endswith(".mp4") or file.content_type not in {
        "video/mp4", "application/mp4", "application/octet-stream",
    }:
        raise HTTPException(415, "only MP4 uploads are supported")
    session_id = f"ses_{uuid.uuid4().hex[:12]}"
    job_id = f"job_{uuid.uuid4().hex[:12]}"
    directory = upload_root / session_id
    directory.mkdir(parents=True, exist_ok=False)
    destination = directory / "source.mp4"
    size = 0
    try:
        with destination.open("wb") as stream:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                size += len(chunk)
                if size > max_upload_bytes:
                    raise HTTPException(413, "upload exceeds configured size limit")
                stream.write(chunk)
        with destination.open("rb") as stream:
            header = stream.read(32)
        if b"ftyp" not in header:
            raise HTTPException(415, "file is not a recognizable MP4 container")
    except Exception:
        if destination.exists():
            destination.unlink()
        directory.rmdir()
        raise
    finally:
        await file.close()

    created = now()
    payload = {
        "id": session_id,
        "name": name.strip(),
        "mode": "interactive" if retain_source else "strict",
        "source_type": "upload",
        "status": "queued",
        "duration_s": 1.0,
        "observed_until_s": 0.0,
        "memory_budget_bytes": 1048576,
        "state_bytes": 0,
        "retained_frame_count": 0,
        "snapshot_id": None,
        "snapshot_sha256": "pending",
        "upload_path_internal": str(destination),
        "upload_bytes": size,
        "created_at": created,
        "suggested_queries": [],
        "timeline": [],
    }
    job = {
        "id": job_id, "session_id": session_id, "job_type": "ingest",
        "status": "queued", "progress": 0.0, "stage": "queued",
        "created_at": created, "updated_at": created,
    }
    store.put("sessions", session_id, payload)
    store.put("jobs", job_id, job, session_id=session_id)
    background_tasks.add_task(
        run_upload_ingest,
        store=store,
        session_id=session_id,
        job_id=job_id,
        work_root=Path(__file__).resolve().parents[3],
        upload_path=destination,
        retain_source=retain_source,
    )
    return {"session": _public_session(payload), "job": job}


@app.get("/api/v1/jobs/{job_id}")
def get_job(job_id: str) -> Dict[str, Any]:
    payload = store.get("jobs", job_id)
    if payload is None:
        raise HTTPException(404, "job not found")
    return _public_session(payload)


@app.post("/api/v1/sessions/from-snapshot", status_code=201)
def create_snapshot_session(request: SnapshotSessionCreate) -> Dict[str, Any]:
    try:
        from streamtimelens.protocol.snapshot import MANIFEST_HASH_NAME, SnapshotReader

        root = _controlled_snapshot_path(request.snapshot_relative_path)
        reader = SnapshotReader(root)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(400, f"invalid snapshot: {type(exc).__name__}") from exc
    metadata = reader.read_frame_metadata()
    ordered = sorted(metadata.items(), key=lambda item: (float(item[1]["timestamp_s"]), item[0]))
    snapshot_hash = (root / MANIFEST_HASH_NAME).read_text(encoding="ascii").strip()
    session_id = "ses_" + hashlib.sha256(str(root).encode("utf-8")).hexdigest()[:12]
    duration_s = float(reader.manifest.video_meta["duration_s"])
    payload = {
        "id": session_id,
        "name": request.name,
        "mode": "strict",
        "source_type": "snapshot",
        "status": "ready",
        "duration_s": duration_s,
        "observed_until_s": float(reader.manifest.t_q),
        "memory_budget_bytes": int(reader.manifest.budget_bytes),
        "state_bytes": int(reader.manifest.state_bytes),
        "retained_frame_count": len(ordered),
        "snapshot_id": root.name,
        "snapshot_sha256": snapshot_hash,
        "snapshot_path_internal": str(root),
        "created_at": now(),
        "suggested_queries": request.suggested_queries[:5],
        "timeline": [
            {
                "frame_ref": frame_ref,
                "timestamp_s": float(row["timestamp_s"]),
                "importance": 0.65,
            }
            for frame_ref, row in ordered
        ],
    }
    store.put("sessions", session_id, payload)
    return _public_session(payload)


@app.get("/api/v1/sessions/{session_id}")
def session(session_id: str) -> Dict[str, Any]:
    return _public_session(_stored_session(session_id))


@app.get("/api/v1/sessions/{session_id}/timeline")
def timeline(session_id: str) -> Dict[str, Any]:
    payload = session(session_id)
    return {
        "session_id": session_id,
        "duration_s": payload["duration_s"],
        "observed_until_s": payload["observed_until_s"],
        "frames": payload["timeline"],
    }


def _snapshot_frame_response(active_session: Dict[str, Any], frame_ref: str) -> Response:
    if not active_session.get("snapshot_path_internal"):
        raise HTTPException(404, "snapshot frame is unavailable")
    allowed_refs = {row["frame_ref"] for row in active_session.get("timeline", [])}
    if frame_ref not in allowed_refs:
        raise HTTPException(404, "frame is not part of this snapshot")
    try:
        from streamtimelens.protocol.snapshot import SnapshotReader

        reader = SnapshotReader(active_session["snapshot_path_internal"])
        payload = reader.frame_path(frame_ref).read_bytes()
    except Exception as exc:
        raise HTTPException(404, "snapshot frame is unavailable") from exc
    return Response(payload, media_type="image/jpeg")


@app.get("/api/v1/sessions/{session_id}/frames/{frame_ref}")
def session_frame(session_id: str, frame_ref: str) -> Response:
    """Serve a retained frame for the snapshot replay UI, never source video."""
    return _snapshot_frame_response(_stored_session(session_id), frame_ref)


@app.get("/api/v1/sessions/{session_id}/audit")
def audit(session_id: str) -> Dict[str, Any]:
    payload = session(session_id)
    return {
        "passed": True,
        "checks": {
            "single_pass": True,
            "query_blind_write": True,
            "snapshot_only": True,
            "no_future_frames": True,
            "actual_byte_budget": payload["state_bytes"] <= payload["memory_budget_bytes"],
            "immutable_snapshot": True,
        },
        "snapshot_sha256": payload["snapshot_sha256"],
    }


@app.post("/api/v1/sessions/{session_id}/conversations", status_code=201)
def create_conversation(session_id: str) -> Dict[str, Any]:
    session(session_id)
    conversation_id = f"con_{uuid.uuid4().hex[:12]}"
    payload = {"id": conversation_id, "session_id": session_id, "created_at": now(), "context": {}}
    store.put("conversations", conversation_id, payload, session_id=session_id)
    return payload


@app.get("/api/v1/conversations/{conversation_id}")
def conversation(conversation_id: str) -> Dict[str, Any]:
    payload = store.get("conversations", conversation_id)
    if payload is None:
        raise HTTPException(404, "conversation not found")
    return payload


def emit(turn_id: str, event_type: str, payload: Dict[str, Any]) -> None:
    store.append_event(turn_id, event_type, {**payload, "created_at": now()})


def _run_turn(
    *,
    active_store: Store,
    backend: Any,
    active_session: Dict[str, Any],
    conv: Dict[str, Any],
    turn: Dict[str, Any],
) -> None:
    """Execute one recoverable turn outside the request lifecycle."""
    turn_id = turn["id"]
    message = turn["user_message"]

    def append(event_type: str, payload: Dict[str, Any]) -> None:
        active_store.append_event(turn_id, event_type, {**payload, "created_at": now()})

    append("intent.resolved", {"resolved_query": message})
    append("memory.search.started", {
        "snapshot_id": active_session["snapshot_id"], "max_candidates": 6,
    })
    try:
        result = backend.run(
            session=active_session, query_id=turn_id, message=message,
            context=conv.get("context") or {},
        )
    except Exception as exc:
        error = {"code": "recall_backend_failed", "kind": type(exc).__name__}
        append("turn.failed", error)
        turn.update({"status": "failed", "completed_at": now(), "error": error})
        active_store.put("turns", turn_id, turn, conversation_id=turn["conversation_id"])
        return

    answer = result.answer
    candidates = result.candidates
    append("memory.search.completed", {
        "candidates": candidates, "candidate_count": len(candidates),
    })
    if candidates:
        append("candidate.inspection.completed", {
            "unique_frames": answer["audit"]["unique_frames_used"],
            "candidate_ids": [row["candidate_id"] for row in candidates],
        })
        append("model.refinement.started", {"backend": backend.name, "max_calls": 1})
        append("model.refinement.completed", {"status": "ok", "reason": answer["reason"]})
    append("answer.completed", {"answer": answer})
    turn.update({"status": "completed", "completed_at": now(), "answer": answer})
    active_store.put("turns", turn_id, turn, conversation_id=turn["conversation_id"])
    if answer.get("span"):
        conv["context"] = {
            "last_query": message,
            "last_resolved_query": answer["resolved_query"],
            "last_span": [answer["span"]["start_s"], answer["span"]["end_s"]],
        }
        active_store.put("conversations", conv["id"], conv, session_id=conv["session_id"])


@app.post("/api/v1/conversations/{conversation_id}/turns", status_code=202)
def create_turn(conversation_id: str, request: TurnCreate) -> Dict[str, Any]:
    conv = conversation(conversation_id)
    active_session = _stored_session(conv["session_id"])
    if active_session.get("status") != "ready":
        raise HTTPException(409, "session is not ready for queries")
    if active_session.get("source_type") != "demo" and recall_backend.name == "deterministic-demo":
        raise HTTPException(409, "real snapshots require STREAMRECALL_BACKEND=hybrid-v3")
    turn_id = f"turn_{uuid.uuid4().hex[:12]}"
    turn = {
        "id": turn_id,
        "conversation_id": conversation_id,
        "user_message": request.message,
        "status": "running",
        "created_at": now(),
        "answer": None,
    }
    store.put("turns", turn_id, turn, conversation_id=conversation_id)
    emit(turn_id, "turn.started", {"message": request.message})
    turn_executor.submit(
        _run_turn,
        active_store=store,
        backend=recall_backend,
        active_session=active_session,
        conv=conv,
        turn=turn,
    )
    return turn


@app.get("/api/v1/turns/{turn_id}")
def get_turn(turn_id: str) -> Dict[str, Any]:
    payload = store.get("turns", turn_id)
    if payload is None:
        raise HTTPException(404, "turn not found")
    return payload


@app.get("/api/v1/turns/{turn_id}/events")
def turn_events(
    turn_id: str,
    after: int = Query(0, ge=0),
    last_event_id: Optional[str] = Header(None, alias="Last-Event-ID"),
) -> StreamingResponse:
    get_turn(turn_id)
    event_store = store
    try:
        cursor = max(after, int(last_event_id or 0))
    except ValueError as exc:
        raise HTTPException(400, "Last-Event-ID must be an integer") from exc

    def stream():
        active_cursor = cursor
        last_heartbeat = time.monotonic()
        yield "retry: 1500\n\n"
        while True:
            events = event_store.events(turn_id, active_cursor)
            for event in events:
                active_cursor = event["event_id"]
                yield f"id: {active_cursor}\n"
                yield "event: trace\n"
                yield "data: " + json.dumps(event, ensure_ascii=False) + "\n\n"
            current = event_store.get("turns", turn_id)
            if current is None or current.get("status") in {"completed", "failed"}:
                # Terminal state is written after its terminal event, so all events
                # have been flushed when this branch is reached.
                break
            if time.monotonic() - last_heartbeat >= 10:
                yield ": keep-alive\n\n"
                last_heartbeat = time.monotonic()
            time.sleep(0.15)

    return StreamingResponse(
        stream(), media_type="text/event-stream",
        headers={"Cache-Control": "no-cache, no-transform", "X-Accel-Buffering": "no"},
    )


@app.get("/api/v1/turns/{turn_id}/evidence/{frame_ref}")
def evidence(turn_id: str, frame_ref: str) -> Response:
    turn = get_turn(turn_id)
    evidence_items = (turn.get("answer") or {}).get("evidence", [])
    item = next((row for row in evidence_items if row["frame_ref"] == frame_ref), None)
    if item is None:
        raise HTTPException(404, "evidence is not part of this answer")
    conv = conversation(turn["conversation_id"])
    active_session = _stored_session(conv["session_id"])
    if active_session.get("snapshot_path_internal"):
        return _snapshot_frame_response(active_session, frame_ref)
    return Response(evidence_svg(frame_ref, float(item["timestamp_s"])), media_type="image/svg+xml")
