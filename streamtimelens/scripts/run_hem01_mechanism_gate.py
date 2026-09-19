#!/usr/bin/env python3
"""Run the effect-free HEM-01 T0/J1 mechanism and persistence gate."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import yaml

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT / "src"))
sys.path.insert(0, str(PACKAGE_ROOT.parent / "src"))

from baas.provenance import (  # noqa: E402
    git_metadata,
    versioned_result_path,
    write_artifact_manifest,
    write_provenance,
)
from streamtimelens.memory.hierarchical_event import (  # noqa: E402
    BOUNDARY_COSINE,
    EMBEDDING_PRECISION,
    HEM_SCHEMA_VERSION,
    LEVEL_CAPACITIES,
    MAX_FINE_EVENT_DURATION_S,
    HierarchicalEventMemory,
    event_memory_sha256,
    load_event_memory,
    write_event_snapshot,
)
from streamtimelens.protocol.snapshot import SnapshotReader, SnapshotWriter, directory_bytes  # noqa: E402
from streamtimelens.protocol.types import Budget, VideoMeta  # noqa: E402
from streamtimelens.retrieval.event_candidates import (  # noqa: E402
    EVENT_CANDIDATE_MARGIN_S,
    EVENT_EXPAND_NEIGHBORS,
    EVENT_MERGE_GAP_S,
    EVENT_TOP_K,
)


IMMUTABLE_CONFIG = {
    "schema_version": 1,
    "stage": "hem01_effect_free_t0_j1",
    "method": "HEM-01",
    "event_schema": HEM_SCHEMA_VERSION,
    "levels": len(LEVEL_CAPACITIES),
    "level_capacities": list(LEVEL_CAPACITIES),
    "boundary_cosine": BOUNDARY_COSINE,
    "max_fine_event_duration_s": MAX_FINE_EVENT_DURATION_S,
    "embedding_precision": EMBEDDING_PRECISION,
    "promotion": "oldest_adjacent_pair",
    "terminal_compaction": "oldest_adjacent_pair",
    "history_layout": "disjoint_recent_fine_old_coarse",
    "primary_budget_bytes": 1024 * 1024,
    "secondary_budget_bytes": 8 * 1024 * 1024,
    "synthetic_embedding_dim": 512,
    "synthetic_token_count": 512,
    "require_query_independent_ingest": True,
    "require_single_pass": True,
    "require_support_conservation": True,
    "require_snapshot_unchanged_on_read": True,
    "require_default_manifest_bit_exact": True,
    "forbid_effect_data": True,
    "formal_data_allowed": False,
    "reader_top_k": EVENT_TOP_K,
    "reader_merge_gap_s": EVENT_MERGE_GAP_S,
    "reader_expand_neighbors": EVENT_EXPAND_NEIGHBORS,
    "reader_candidate_margin_s": EVENT_CANDIDATE_MARGIN_S,
    "require_reader_snapshot_only": True,
    "require_decoder_seek_count_zero": True,
}


def _validate_config(config: dict) -> None:
    if config != IMMUTABLE_CONFIG:
        changed = {
            key: (config.get(key), expected)
            for key, expected in IMMUTABLE_CONFIG.items()
            if config.get(key) != expected
        }
        raise ValueError(
            f"HEM-01 immutable mechanism config changed: {changed}; "
            f"field_delta={sorted(set(config) ^ set(IMMUTABLE_CONFIG))}"
        )


def _file_fingerprint(reader: SnapshotReader) -> tuple[tuple[str, int, str], ...]:
    rows = []
    for name in sorted(reader.manifest.allowed_files):
        path = reader.path(name)
        rows.append((name, path.stat().st_size, hashlib.sha256(path.read_bytes()).hexdigest()))
    return tuple(rows)


def _synthetic_audit(config: dict, source_revision: str) -> dict:
    dimension = int(config["synthetic_embedding_dim"])
    count = int(config["synthetic_token_count"])
    memory = HierarchicalEventMemory()
    reasons = {}
    for index in range(count):
        vector = np.zeros(dimension, dtype=np.float32)
        phase = (index // 3) % 4
        vector[phase] = 1.0
        vector[(phase + 1) % dimension] = 0.05 * (index % 3)
        vector /= np.linalg.norm(vector)
        reason = memory.observe(
            timestamp_s=float(index * 2), frame_index=index, embedding=vector,
        )
        reasons[reason] = reasons.get(reason, 0) + 1
    events = memory.events()
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        writer = SnapshotWriter(root, source_revision=source_revision)
        manifest = write_event_snapshot(
            memory, writer, name="snapshot", t_q=float(count * 2),
            meta=VideoMeta("hem01-synthetic", float(count * 2), 0.5, count),
            budget=Budget(int(config["primary_budget_bytes"]), 0),
            config={"method": "HEM-01", "contract": HEM_SCHEMA_VERSION},
            source_revision=source_revision,
        )
        reader = SnapshotReader(root / "snapshot")
        before = _file_fingerprint(reader)
        restored = load_event_memory(reader)
        after = _file_fingerprint(SnapshotReader(reader.root))
        parameters = set(inspect.signature(HierarchicalEventMemory.observe).parameters)
        audit = {
            "schema_version": 1,
            "effect_data_read": False,
            "input_token_count": count,
            "input_embedding_dim": dimension,
            "event_count": len(events),
            "level_counts": list(memory.level_counts),
            "level_capacities": list(LEVEL_CAPACITIES),
            "support_count": sum(row.support_count for row in events),
            "oldest_scale": events[0].scale,
            "newest_scale": events[-1].scale,
            "observe_parameters": sorted(parameters),
            "query_free_api": not parameters & {"query", "query_id", "gt", "gt_span"},
            "reason_counts": reasons,
            "event_memory_sha256": event_memory_sha256(events),
            "roundtrip_sha256": event_memory_sha256(restored),
            "snapshot_state_bytes": manifest.state_bytes,
            "snapshot_budget_bytes": manifest.budget_bytes,
            "snapshot_filesystem_bytes": directory_bytes(reader.root),
            "snapshot_unchanged": before == after,
            "event_preview": [row.to_record() for row in (*events[:3], *events[-3:])],
        }
        checks = {
            "support_conserved": audit["support_count"] == count,
            "capacities_respected": all(
                used <= cap for used, cap in zip(memory.level_counts, LEVEL_CAPACITIES)
            ),
            "old_history_coarsened": audit["oldest_scale"] > 0,
            "recent_history_fine": audit["newest_scale"] == 0,
            "query_free_api": audit["query_free_api"],
            "roundtrip_bit_stable": audit["event_memory_sha256"] == audit["roundtrip_sha256"],
            "snapshot_within_budget": (
                manifest.state_bytes == audit["snapshot_filesystem_bytes"]
                and manifest.state_bytes <= manifest.budget_bytes
            ),
            "snapshot_unchanged": audit["snapshot_unchanged"],
        }
        if not all(checks.values()):
            raise RuntimeError(f"HEM-01 synthetic audit failed: {checks}")
        return {**audit, "checks": checks}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path,
        default=PACKAGE_ROOT / "configs/exploration/hem01_mechanism_gate.yaml",
    )
    parser.add_argument(
        "--output", type=Path,
        default=PACKAGE_ROOT.parent / "results/hem01/T0_J1",
    )
    args = parser.parse_args()
    metadata = git_metadata()
    if metadata["is_dirty"]:
        raise RuntimeError("HEM-01 T0/J1 requires a clean Git commit")
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    _validate_config(config)
    command = [
        sys.executable, "-m", "pytest", "-q",
        "tests/test_hierarchical_event_memory.py", "tests/test_protocol.py",
        "tests/test_hem_reader_and_ingest.py", "tests/test_visual_route.py",
    ]
    completed = subprocess.run(
        command, cwd=PACKAGE_ROOT, text=True, capture_output=True, check=False,
    )
    audit, audit_error = None, None
    try:
        audit = _synthetic_audit(config, metadata["commit"])
    except Exception as exc:
        audit_error = f"{type(exc).__name__}: {exc}"
    final_metadata = git_metadata()
    if final_metadata["is_dirty"] or final_metadata["commit"] != metadata["commit"]:
        raise RuntimeError("Git state changed during HEM-01 T0/J1")
    output = versioned_result_path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    if audit is not None:
        (output / "event_audit.json").write_text(
            json.dumps(audit, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    passed = completed.returncode == 0 and audit_error is None
    report = {
        "schema_version": 1,
        "stage": "hem01_effect_free_t0_j1",
        "effect_data_read": False,
        "passed": passed,
        "test_returncode": completed.returncode,
        "test_stdout": completed.stdout,
        "test_stderr": completed.stderr,
        "synthetic_audit_error": audit_error,
        "assertions": {
            "automated_tests_passed": completed.returncode == 0,
            **(audit["checks"] if audit is not None else {}),
        },
    }
    report_path = output / "mechanism_gate.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    write_provenance(
        output,
        configuration={
            **config,
            "config_source_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
            "test_command": command,
        },
        config_path=args.config,
        command=[sys.executable, str(Path(__file__).resolve()), "--config", str(args.config)],
        extra_metadata={
            "effect_data_read": False,
            "mechanism_gate_sha256": hashlib.sha256(report_path.read_bytes()).hexdigest(),
        },
    )
    write_artifact_manifest(output)
    print(json.dumps({"passed": passed, "output": str(output)}, sort_keys=True))
    return 0 if passed else (completed.returncode or 2)


if __name__ == "__main__":
    raise SystemExit(main())
