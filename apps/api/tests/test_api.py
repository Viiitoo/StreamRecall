import threading
import time
from pathlib import Path

from fastapi.testclient import TestClient

from app import main
from app.recall import BackendResult
from app.store import Store
from streamtimelens.protocol.snapshot import SnapshotWriter
from streamtimelens.protocol.types import Budget, VideoMeta


def client(tmp_path: Path) -> TestClient:
    main.store = Store(tmp_path / "test.sqlite3")
    return TestClient(main.app)


def wait_turn(api: TestClient, turn_id: str, timeout: float = 3) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        turn = api.get(f"/api/v1/turns/{turn_id}").json()
        if turn["status"] in {"completed", "failed"}:
            return turn
        time.sleep(0.01)
    raise AssertionError(f"turn {turn_id} did not finish")


def test_demo_recall_flow(tmp_path):
    with client(tmp_path) as api:
        sessions = api.get("/api/v1/sessions")
        assert sessions.status_code == 200
        session = sessions.json()[0]
        assert session["state_bytes"] <= session["memory_budget_bytes"]

        conversation = api.post(f"/api/v1/sessions/{session['id']}/conversations").json()
        turn = api.post(
            f"/api/v1/conversations/{conversation['id']}/turns",
            json={"message": "什么时候有人把杯子放到桌上？"},
        )
        assert turn.status_code == 202
        submitted = turn.json()
        assert submitted["status"] == "running"
        answer = wait_turn(api, submitted["id"])["answer"]
        assert answer["status"] == "FOUND"
        assert answer["audit"]["snapshot_only"] is True
        assert answer["audit"]["unique_frames_used"] <= 16

        events = api.get(f"/api/v1/turns/{submitted['id']}/events")
        assert events.status_code == 200
        assert "answer.completed" in events.text


def test_unknown_evidence_is_rejected(tmp_path):
    with client(tmp_path) as api:
        conversation = api.post("/api/v1/sessions/ses_kitchen_demo/conversations").json()
        submitted = api.post(
            f"/api/v1/conversations/{conversation['id']}/turns",
            json={"message": "open cabinet"},
        ).json()
        turn = wait_turn(api, submitted["id"])
        response = api.get(f"/api/v1/turns/{turn['id']}/evidence/not-allowed.jpg")
        assert response.status_code == 404


def test_not_found_is_explicit(tmp_path):
    with client(tmp_path) as api:
        conversation = api.post("/api/v1/sessions/ses_kitchen_demo/conversations").json()
        submitted = api.post(
            f"/api/v1/conversations/{conversation['id']}/turns",
            json={"message": "什么时候出现了一辆汽车？"},
        ).json()
        turn = wait_turn(api, submitted["id"])
        assert turn["answer"]["status"] == "NOT_FOUND"
        assert turn["answer"]["audit"]["model_calls"] == 0


def test_registers_only_controlled_valid_snapshots_and_serves_evidence(tmp_path):
    snapshot_root = tmp_path / "snapshots"
    writer = SnapshotWriter(snapshot_root, source_revision="test")
    writer.write(
        name="video/rho_1.00",
        t_q=10,
        meta=VideoMeta("video", 10, 1, 10),
        budget=Budget(1024 * 1024, 0),
        cards=[],
        raw_frames=[("frame.jpg", b"jpeg-evidence")],
        raw_metadata={"frame.jpg": {"blob": "frame.jpg", "timestamp_s": 4.0, "frame_index": 4}},
        writer_calls=0,
        config={"fixture": "product-api"},
        method="uniform_raw",
        pixel_only=False,
    )
    main.snapshot_root = snapshot_root.resolve()
    with client(tmp_path) as api:
        escaped = api.post(
            "/api/v1/sessions/from-snapshot",
            json={"name": "bad", "snapshot_relative_path": "../outside"},
        )
        assert escaped.status_code == 400

        registered = api.post(
            "/api/v1/sessions/from-snapshot",
            json={"name": "real snapshot", "snapshot_relative_path": "video/rho_1.00"},
        )
        assert registered.status_code == 201
        session = registered.json()
        assert session["retained_frame_count"] == 1
        assert session["state_bytes"] <= session["memory_budget_bytes"]
        assert "snapshot_path_internal" not in session

        conversation_id = "con_snapshot"
        turn_id = "turn_snapshot"
        main.store.put(
            "conversations", conversation_id,
            {"id": conversation_id, "session_id": session["id"], "context": {}},
            session_id=session["id"],
        )
        main.store.put(
            "turns", turn_id,
            {
                "id": turn_id,
                "conversation_id": conversation_id,
                "answer": {"evidence": [{"frame_ref": "frame.jpg", "timestamp_s": 4.0}]},
            },
            conversation_id=conversation_id,
        )
        evidence = api.get(f"/api/v1/turns/{turn_id}/evidence/frame.jpg")
        assert evidence.status_code == 200
        assert evidence.content == b"jpeg-evidence"
        snapshot_frame = api.get(f"/api/v1/sessions/{session['id']}/frames/frame.jpg")
        assert snapshot_frame.status_code == 200
        assert snapshot_frame.content == b"jpeg-evidence"
        assert api.get(f"/api/v1/sessions/{session['id']}/frames/other.jpg").status_code == 404

        unavailable = api.post(
            f"/api/v1/conversations/{conversation_id}/turns",
            json={"message": "person opens a door"},
        )
        assert unavailable.status_code == 409


def test_upload_streams_to_generated_path_and_schedules_ingest(tmp_path, monkeypatch):
    main.upload_root = (tmp_path / "uploads").resolve()
    captured = {}

    def fake_ingest(**kwargs):
        captured.update(kwargs)
        job = kwargs["store"].get("jobs", kwargs["job_id"])
        job.update({"status": "completed", "progress": 1.0, "stage": "ready"})
        kwargs["store"].put("jobs", job["id"], job, session_id=job["session_id"])

    monkeypatch.setattr(main, "run_upload_ingest", fake_ingest)
    payload = b"\x00\x00\x00\x18ftypmp42" + b"product-video"
    with client(tmp_path) as api:
        response = api.post(
            "/api/v1/sessions/upload",
            data={"name": "Uploaded stream", "retain_source": "false"},
            files={"file": ("camera.mp4", payload, "video/mp4")},
        )
        assert response.status_code == 202
        body = response.json()
        assert body["session"]["status"] == "queued"
        assert "upload_path_internal" not in body["session"]
        assert captured["upload_path"].read_bytes() == payload
        assert captured["upload_path"].name == "source.mp4"
        job = api.get(f"/api/v1/jobs/{body['job']['id']}").json()
        assert job["status"] == "completed"


def test_upload_rejects_non_mp4(tmp_path):
    main.upload_root = (tmp_path / "uploads").resolve()
    with client(tmp_path) as api:
        response = api.post(
            "/api/v1/sessions/upload",
            data={"name": "Invalid stream"},
            files={"file": ("notes.txt", b"not a video", "text/plain")},
        )
        assert response.status_code == 415


def test_turn_failure_is_persisted_and_streamed(tmp_path, monkeypatch):
    class FailingBackend:
        name = "deterministic-demo"

        def run(self, **kwargs):
            del kwargs
            raise RuntimeError("private model error")

    monkeypatch.setattr(main, "recall_backend", FailingBackend())
    with client(tmp_path) as api:
        conversation = api.post("/api/v1/sessions/ses_kitchen_demo/conversations").json()
        submitted = api.post(
            f"/api/v1/conversations/{conversation['id']}/turns",
            json={"message": "trigger failure"},
        )
        assert submitted.status_code == 202
        turn = wait_turn(api, submitted.json()["id"])
        assert turn["status"] == "failed"
        assert turn["error"] == {"code": "recall_backend_failed", "kind": "RuntimeError"}
        events = api.get(f"/api/v1/turns/{turn['id']}/events")
        assert "turn.failed" in events.text
        assert "private model error" not in events.text


def test_event_stream_resumes_after_cursor(tmp_path):
    with client(tmp_path) as api:
        conversation = api.post("/api/v1/sessions/ses_kitchen_demo/conversations").json()
        submitted = api.post(
            f"/api/v1/conversations/{conversation['id']}/turns",
            json={"message": "open the cabinet"},
        ).json()
        wait_turn(api, submitted["id"])
        persisted = main.store.events(submitted["id"])
        cursor = persisted[1]["event_id"]

        query_resume = api.get(f"/api/v1/turns/{submitted['id']}/events?after={cursor}")
        assert f"id: {cursor}\n" not in query_resume.text
        assert "answer.completed" in query_resume.text

        header_resume = api.get(
            f"/api/v1/turns/{submitted['id']}/events?after=0",
            headers={"Last-Event-ID": str(cursor)},
        )
        assert f"id: {cursor}\n" not in header_resume.text
        assert "answer.completed" in header_resume.text


def test_turn_submission_stays_responsive_with_concurrent_workers(tmp_path, monkeypatch):
    release = threading.Event()
    both_started = threading.Event()
    lock = threading.Lock()
    started = 0

    class BlockingBackend:
        name = "deterministic-demo"

        def run(self, **kwargs):
            nonlocal started
            with lock:
                started += 1
                if started == 2:
                    both_started.set()
            assert release.wait(2)
            return BackendResult(
                answer={
                    "status": "NOT_FOUND", "message": "none", "resolved_query": kwargs["message"],
                    "span": None, "confidence": 0.0, "evidence": [], "alternatives": [],
                    "reason": "not_found", "limitations": [],
                    "audit": {"snapshot_only": True, "future_frames_used": False,
                              "model_calls": 0, "unique_frames_used": 0,
                              "actual_byte_budget": True},
                    "cost": {"search_ms": 0, "refine_ms": 0, "total_ms": 0,
                             "generated_tokens": 0},
                },
                candidates=[],
            )

    monkeypatch.setattr(main, "recall_backend", BlockingBackend())
    with client(tmp_path) as api:
        conversations = [
            api.post("/api/v1/sessions/ses_kitchen_demo/conversations").json()["id"]
            for _ in range(2)
        ]
        started_at = time.monotonic()
        responses = [
            api.post(f"/api/v1/conversations/{conversation_id}/turns", json={"message": f"q{index}"})
            for index, conversation_id in enumerate(conversations)
        ]
        assert time.monotonic() - started_at < 0.5
        assert all(response.status_code == 202 for response in responses)
        assert both_started.wait(1)
        release.set()
        for response in responses:
            assert wait_turn(api, response.json()["id"])["status"] == "completed"
