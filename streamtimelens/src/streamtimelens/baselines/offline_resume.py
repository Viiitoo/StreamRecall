"""Validation helpers for provenance-safe Offline TimeLens recovery."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping


def sha256_file(path: Path) -> str:
    """Hash one file without loading it into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_model(path: Path) -> str:
    """Hash a model file or a complete, path-sensitive model directory."""
    path = path.resolve()
    if path.is_file():
        return sha256_file(path)
    if not path.is_dir():
        raise FileNotFoundError(f"offline model does not exist: {path}")
    files = sorted(row for row in path.rglob("*") if row.is_file())
    if not files:
        raise ValueError(f"offline model directory is empty: {path}")
    digest = hashlib.sha256()
    for row in files:
        relative = row.relative_to(path).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(row.stat().st_size.to_bytes(16, "big"))
        with row.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line:
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"JSONL row must be an object: {path}:{line_number}")
        rows.append(row)
    return rows


def shard_for_query(query_id: str, num_shards: int) -> int:
    return int(hashlib.sha256(query_id.encode("utf-8")).hexdigest(), 16) % num_shards


def queries_for_shard(
    rows: Iterable[Mapping[str, Any]], num_shards: int, shard_index: int,
) -> list[Mapping[str, Any]]:
    return [
        row for row in rows
        if shard_for_query(str(row["query_id"]), num_shards) == shard_index
    ]


def validate_prediction_rows(
    predictions: Iterable[Mapping[str, Any]],
    queries: Iterable[Mapping[str, Any]],
    *,
    num_shards: int,
    shard_index: int,
) -> set[str]:
    """Reject duplicate, unknown, misidentified, and out-of-shard predictions."""
    query_rows = list(queries)
    expected = {str(row["query_id"]): str(row["video_id"]) for row in query_rows}
    if len(expected) != len(query_rows):
        raise ValueError("offline query manifest contains duplicate query IDs")
    completed: set[str] = set()
    for row in predictions:
        query_id = str(row.get("query_id", ""))
        if not query_id or query_id in completed:
            raise ValueError(f"offline predictions contain duplicate/empty query ID: {query_id!r}")
        if query_id not in expected:
            raise ValueError(f"offline prediction does not belong to query manifest: {query_id}")
        if shard_for_query(query_id, num_shards) != shard_index:
            raise ValueError(f"offline prediction does not belong to shard {shard_index}: {query_id}")
        if str(row.get("video_id", "")) != expected[query_id]:
            raise ValueError(f"offline prediction video identity mismatch: {query_id}")
        completed.add(query_id)
    return completed


def validate_resume_source(
    source_root: Path,
    *,
    model_sha256: str,
    query_manifest_sha256: str,
    queries: Iterable[Mapping[str, Any]],
    num_shards: int,
    shard_index: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Validate and describe a completed prefix imported from another revision."""
    source_root = source_root.resolve()
    config_path = source_root / "config.resolved.json"
    provenance_path = source_root / "provenance.json"
    predictions_path = source_root / "predictions.jsonl"
    for path in (config_path, provenance_path, predictions_path):
        if not path.is_file():
            raise ValueError(f"offline resume source is incomplete; missing {path.name}: {source_root}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    required = {
        "model_sha256": model_sha256,
        "query_manifest_sha256": query_manifest_sha256,
        "num_shards": num_shards,
        "shard_index": shard_index,
        "protocol": "offline_upper_bound_not_streaming",
    }
    mismatches = {
        key: {"expected": value, "actual": config.get(key)}
        for key, value in required.items() if config.get(key) != value
    }
    if mismatches:
        raise ValueError(f"offline resume source provenance mismatch: {mismatches}")
    source_commit = provenance.get("code", {}).get("commit")
    if not isinstance(source_commit, str) or len(source_commit) != 40:
        raise ValueError("offline resume source has no complete source commit")
    predictions = load_jsonl(predictions_path)
    validate_prediction_rows(
        predictions, queries, num_shards=num_shards, shard_index=shard_index,
    )
    metadata = {
        "source_directory": str(source_root),
        "source_commit": source_commit,
        "predictions_file": str(predictions_path),
        "predictions_sha256": sha256_file(predictions_path),
        "imported_predictions": len(predictions),
    }
    return predictions, metadata
