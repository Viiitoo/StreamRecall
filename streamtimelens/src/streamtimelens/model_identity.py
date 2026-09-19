"""Verify local model assets against the preregistered content manifest."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_registered_model(
    registry_path: Path,
    *,
    model_key: str,
    model_root: Path,
    revision: str,
    content_sha256: str,
    verify_files: bool,
) -> dict[str, Any]:
    """Match identity metadata and optionally rehash every registered file."""
    payload = json.loads(registry_path.read_text(encoding="utf-8"))
    try:
        registered = dict(payload["models"][model_key])
    except (KeyError, TypeError) as exc:
        raise ValueError(f"model identity registry has no {model_key!r} entry") from exc
    if (
        registered.get("revision") != revision
        or registered.get("content_sha256") != content_sha256
        or not model_root.is_dir()
    ):
        raise ValueError(f"registered {model_key} identity does not match Hybrid V3 config")
    if verify_files:
        observed = []
        for expected in registered.get("files", []):
            relative = str(expected["path"])
            path = (model_root / relative).resolve()
            try:
                path.relative_to(model_root.resolve())
            except ValueError as exc:
                raise ValueError("model registry path escapes the model root") from exc
            row = {
                "bytes": path.stat().st_size,
                "path": relative,
                "sha256": sha256_file(path),
            }
            if row != expected:
                raise ValueError(f"registered {model_key} file changed: {relative}")
            observed.append(row)
        canonical = json.dumps(
            observed, sort_keys=True, separators=(",", ":"),
            ensure_ascii=False, allow_nan=False,
        ).encode("utf-8")
        if hashlib.sha256(canonical).hexdigest() != content_sha256:
            raise ValueError(f"registered {model_key} content hash disagrees with its file table")
    return {
        "model_key": model_key,
        "path": str(model_root.resolve()),
        "revision": revision,
        "content_sha256": content_sha256,
        "file_count": int(registered.get("file_count", 0)),
        "total_bytes": int(registered.get("total_bytes", 0)),
        "files_verified": bool(verify_files),
        "registry": str(registry_path.resolve()),
        "registry_sha256": sha256_file(registry_path),
    }
