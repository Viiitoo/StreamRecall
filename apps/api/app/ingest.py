"""Background single-pass upload ingestion through the frozen snapshot builder."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict

from .demo import now
from .store import Store


CLIP_REVISION = "3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268"
CLIP_SHA256 = "16d265fbd9c648b096d830bbeaf83a4c197f071a9b6e4c6685533b5f6028b20b"


def _update_job(store: Store, job: Dict[str, Any], **changes: Any) -> None:
    job.update(changes, updated_at=now())
    store.put("jobs", job["id"], job, session_id=job["session_id"])


def _update_session(store: Store, session: Dict[str, Any], **changes: Any) -> None:
    session.update(changes, updated_at=now())
    store.put("sessions", session["id"], session)


def run_upload_ingest(
    *,
    store: Store,
    session_id: str,
    job_id: str,
    work_root: Path,
    upload_path: Path,
    retain_source: bool,
) -> None:
    """Create all frozen arrival snapshots in one CLI-enforced decode pass."""
    session = store.get("sessions", session_id)
    job = store.get("jobs", job_id)
    if session is None or job is None:
        return
    _update_session(store, session, status="ingesting")
    _update_job(store, job, status="running", progress=0.05, stage="loading_video")
    output = work_root / "results" / "streamrecall_uploads" / session_id
    command = [
        sys.executable,
        "streamtimelens/scripts/build_snapshots.py",
        "--video", str(upload_path),
        "--method", "uniform_raw",
        "--budget", "1048576",
        "--arrival-ratios", ".25,.5,.75,1",
        "--sample-fps", "2",
        "--clip-fps", "0.5",
        "--clip-model", "models/clip-vit-base-patch32",
        "--clip-revision", CLIP_REVISION,
        "--clip-sha256", CLIP_SHA256,
        "--embedding-precision", "fp16",
        "--output", str(output),
    ]
    _update_job(store, job, progress=0.15, stage="single_pass_ingest", command_internal=command)
    try:
        completed = subprocess.run(
            command, cwd=str(work_root), text=True, capture_output=True, check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(completed.stderr[-2000:] or f"ingest exited {completed.returncode}")
        lines = [line for line in completed.stdout.splitlines() if line.strip()]
        summary = json.loads(lines[-1])
        snapshot_path = Path(summary["output_root"]) / summary["video_id"] / "rho_1.00"

        from streamtimelens.protocol.snapshot import MANIFEST_HASH_NAME, SnapshotReader

        reader = SnapshotReader(snapshot_path)
        metadata = reader.read_frame_metadata()
        ordered = sorted(metadata.items(), key=lambda item: (float(item[1]["timestamp_s"]), item[0]))
        snapshot_hash = (snapshot_path / MANIFEST_HASH_NAME).read_text(encoding="ascii").strip()
        duration_s = float(reader.manifest.video_meta["duration_s"])
        _update_session(
            store,
            session,
            status="ready",
            duration_s=duration_s,
            observed_until_s=float(reader.manifest.t_q),
            state_bytes=int(reader.manifest.state_bytes),
            retained_frame_count=len(ordered),
            snapshot_id=snapshot_path.name,
            snapshot_sha256=snapshot_hash,
            snapshot_path_internal=str(snapshot_path.resolve()),
            timeline=[
                {
                    "frame_ref": frame_ref,
                    "timestamp_s": float(row["timestamp_s"]),
                    "importance": 0.65,
                }
                for frame_ref, row in ordered
            ],
            source_retained=bool(retain_source),
        )
        if not retain_source and upload_path.is_file():
            upload_path.unlink()
            session.pop("upload_path_internal", None)
            store.put("sessions", session_id, session)
        _update_job(
            store, job, status="completed", progress=1.0, stage="ready",
            snapshot_id=snapshot_path.name,
        )
    except Exception as exc:
        error = {
            "code": "ingest_failed",
            "kind": type(exc).__name__,
            "detail": "Snapshot ingest failed. Confirm the file is a decodable MP4 and retry.",
        }
        diagnostic = {"kind": type(exc).__name__, "detail": str(exc)[-4000:]}
        _update_session(store, session, status="failed", error=error, diagnostic_internal=diagnostic)
        _update_job(
            store, job, status="failed", stage="failed", error=error,
            diagnostic_internal=diagnostic,
        )
