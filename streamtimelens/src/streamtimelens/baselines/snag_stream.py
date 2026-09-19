"""Strict snapshot orchestration and runtime auditing for SnAG-adapt."""

from __future__ import annotations

import hashlib
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Protocol

from streamtimelens.baselines.snag_adapt import (
    SnAGAdaptConfig,
    SnAGAdaptWriter,
    SnAGSnapshotReader,
)
from streamtimelens.baselines.snag_features import SnAGFeatureObservation, SnAGIngestStats
from streamtimelens.protocol.types import VideoMeta


class FeatureExtractor(Protocol):
    last_stats: SnAGIngestStats | None

    def iter_video(
        self, video_path: Path | str, meta: VideoMeta,
    ) -> Iterable[SnAGFeatureObservation]: ...


@dataclass(frozen=True)
class SnAGFreezeRecord:
    t_q: float
    path: str
    state_bytes: int
    token_count: int
    latest_evidence_end_s: float | None
    freeze_wall_s: float


@dataclass(frozen=True)
class SnAGStreamRun:
    video_id: str
    snapshots: tuple[SnAGFreezeRecord, ...]
    ingest: SnAGIngestStats
    writer_peak_items: int
    writer_peak_feature_bytes: int


def snapshot_fingerprint(snapshot: SnAGSnapshotReader) -> tuple[tuple[str, int, str], ...]:
    rows = []
    for path in sorted(snapshot.root.iterdir()):
        if not path.is_file():
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        rows.append((path.name, path.stat().st_size, digest))
    return tuple(rows)


class SnAGSequentialIngestor:
    """Freeze each requested time before consuming any later observation."""

    def __init__(self, extractor: FeatureExtractor, config: SnAGAdaptConfig) -> None:
        self.extractor = extractor
        self.config = config

    def run(
        self,
        video_path: Path | str,
        meta: VideoMeta,
        query_times: Iterable[float],
        output_root: Path | str,
    ) -> SnAGStreamRun:
        times = tuple(sorted(set(map(float, query_times))))
        if not times or times[0] < 0 or times[-1] > meta.duration_s + 1e-6:
            raise ValueError("query times must be non-empty and within the video")
        root = Path(output_root).expanduser().resolve(strict=False)
        root.mkdir(parents=True, exist_ok=True)
        writer = SnAGAdaptWriter(meta, self.config)
        pending = 0
        records: list[SnAGFreezeRecord] = []
        peak_items = peak_bytes = 0

        def freeze(t_q: float) -> None:
            nonlocal records
            target = root / f"tq_{t_q:.9f}"
            started = time.perf_counter()
            manifest = writer.freeze(t_q, target)
            elapsed = time.perf_counter() - started
            latest = max((item.t_end_s for item in writer.items), default=None)
            records.append(SnAGFreezeRecord(
                t_q, str(target), manifest.state_bytes, manifest.token_count,
                latest, elapsed,
            ))

        for observation in self.extractor.iter_video(video_path, meta):
            while pending < len(times) and times[pending] + 1e-9 < observation.t_end_s:
                freeze(times[pending])
                pending += 1
            writer.ingest_step(
                observation.feature, observation.t_start_s, observation.t_end_s,
                observation.source_id,
            )
            peak_items = max(peak_items, len(writer.items))
            peak_bytes = max(peak_bytes, sum(item.feature.nbytes for item in writer.items))
        while pending < len(times):
            freeze(times[pending])
            pending += 1
        stats = self.extractor.last_stats
        if stats is None or stats.decode_passes != 1:
            raise RuntimeError("feature extractor did not certify one completed decode pass")
        return SnAGStreamRun(meta.video_id, tuple(records), stats, peak_items, peak_bytes)


def snag_runtime_protocol_audit(
    run: SnAGStreamRun,
    *,
    query_blind_signature: bool,
    independent_query_checks: Iterable[Mapping[str, Any]],
) -> dict[str, object]:
    snapshots = [SnAGSnapshotReader(row.path) for row in run.snapshots]
    no_future = all(
        row.latest_evidence_end_s is None or row.latest_evidence_end_s <= row.t_q + 1e-9
        for row in run.snapshots
    )
    byte_budget = all(
        snap.manifest.budget_bytes is None
        or snap.manifest.state_bytes <= snap.manifest.budget_bytes
        for snap in snapshots
    )
    immutable = all(
        os.stat(snap.root).st_mode & 0o222 == 0
        and all(os.stat(path).st_mode & 0o222 == 0 for path in snap.root.iterdir())
        for snap in snapshots
    )
    query_checks = list(independent_query_checks)
    independent = bool(query_checks) and all(bool(row.get("passed")) for row in query_checks)
    checks = {
        "online_ingest": run.ingest.emitted_features > 0,
        "late_query": len(run.snapshots) > 0,
        "query_blind_write": query_blind_signature,
        "single_pass": run.ingest.decode_passes == 1,
        "snapshot_only_immutable": immutable,
        "no_replay_no_future": no_future,
        "actual_byte_budget": byte_budget,
        "independent_queries": independent,
        "revision_provenance": all(snap.manifest.source_revision for snap in snapshots),
    }
    return {
        "schema_version": 1,
        "method": snapshots[0].manifest.method if snapshots else "SnAG-adapt",
        "passed": all(checks.values()),
        "checks": checks,
        "run": asdict(run),
        "independent_query_checks": query_checks,
    }
