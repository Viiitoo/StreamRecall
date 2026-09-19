"""Protocol audit for query-blind Uniform/Semantic visual-cache runs."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from streamtimelens.protocol.snapshot import SnapshotReader


@dataclass(frozen=True)
class VisualAuditRun:
    method: str
    budget_bytes: int
    video_id: str
    run_dir: str

    def __post_init__(self) -> None:
        if self.method not in ("uniform_raw", "semantic_reservoir"):
            raise ValueError("visual audit method is invalid")
        if self.budget_bytes <= 0 or not self.video_id or not self.run_dir:
            raise ValueError("visual audit identity is invalid")


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def audit_visual_cache_runs(
    runs: Iterable[VisualAuditRun],
    *,
    expected_rhos: tuple[float, ...] = (.25, .5, .75, 1.0),
) -> dict[str, Any]:
    rows = list(runs)
    identities = [(row.method, row.budget_bytes, row.video_id) for row in rows]
    if not rows or len(identities) != len(set(identities)):
        raise ValueError("visual audit needs unique method/budget/video runs")
    results = []
    all_passed = True
    for row in rows:
        root = Path(row.run_dir)
        trace_path = root / row.video_id / "ingest.trace.jsonl"
        provenance_path = root / "provenance.json"
        revision_path = root / "git_revision.txt"
        config_paths = tuple(root.glob("config.resolved.*"))
        failures = []
        if not trace_path.is_file():
            failures.append("missing_ingest_trace")
            traces = []
        else:
            traces = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()]
        if not provenance_path.is_file() or not revision_path.is_file() or not config_paths:
            failures.append("incomplete_provenance")
        config = _load_json(config_paths[0]) if config_paths and config_paths[0].suffix == ".json" else {}
        if config.get("writer") != "none":
            failures.append("writer_not_none")
        if any(item.get("kind") == "writer_called" for item in traces):
            failures.append("writer_loaded_or_called")
        seen_times = [float(item["t"]) for item in traces if item.get("kind") == "frame_seen"]
        if not seen_times or seen_times != sorted(seen_times):
            failures.append("decode_not_single_pass_monotonic")
        mutation_kinds = {"clip_encoded", "frame_admitted", "frame_replaced", "frame_evicted"}
        if any(
            "state_bytes_before" not in item or "state_bytes_after" not in item
            for item in traces if item.get("kind") in mutation_kinds
        ):
            failures.append("mutation_byte_trace_incomplete")
        snapshots = []
        for rho in expected_rhos:
            snapshot_root = root / row.video_id / f"rho_{rho:.2f}"
            try:
                reader = SnapshotReader(snapshot_root)
                metadata = reader.read_frame_metadata()
            except Exception as exc:
                failures.append(f"snapshot_{rho:.2f}_{type(exc).__name__}")
                continue
            if reader.manifest.method != row.method:
                failures.append(f"snapshot_{rho:.2f}_method_mismatch")
            if reader.manifest.budget_bytes != row.budget_bytes:
                failures.append(f"snapshot_{rho:.2f}_budget_mismatch")
            if reader.manifest.writer_calls != 0:
                failures.append(f"snapshot_{rho:.2f}_writer_calls")
            if any(float(item["timestamp_s"]) > reader.manifest.t_q + 1e-6 for item in metadata.values()):
                failures.append(f"snapshot_{rho:.2f}_future_frame")
            snapshots.append({
                "rho": rho, "t_q": reader.manifest.t_q,
                "state_bytes": reader.manifest.state_bytes,
                "budget_bytes": reader.manifest.budget_bytes,
                "frame_count": len(metadata),
            })
        if len(snapshots) != len(expected_rhos):
            failures.append("incomplete_rho_snapshots")
        passed = not failures
        all_passed = all_passed and passed
        results.append({
            **asdict(row), "passed": passed, "failures": failures,
            "frame_seen": len(seen_times), "snapshots": snapshots,
        })
    return {
        "schema_version": 1,
        "passed": all_passed,
        "run_count": len(rows),
        "expected_rhos": list(expected_rhos),
        "runs": results,
    }
