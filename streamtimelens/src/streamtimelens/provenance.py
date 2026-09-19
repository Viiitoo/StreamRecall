"""Run provenance that remains explicit when Git or GPU metadata is absent."""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from .config import ResolvedConfig, write_resolved_config


WORK_ROOT = Path(__file__).resolve().parents[3]
PROJECT_ROOT = WORK_ROOT.parent
REFERENCE_ROOTS = {
    "timelens": WORK_ROOT / "third_party" / "TimeLens",
    "vst": PROJECT_ROOT / "ref" / "vst",
    "oasis": PROJECT_ROOT / "ref" / "oasis",
    "onvtg": PROJECT_ROOT / "ref" / "onvtg",
    "streamchat": PROJECT_ROOT / "ref" / "streamchat",
    "revisionllm": PROJECT_ROOT / "ref" / "revisionllm",
}


def _git(path: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(path), *args], text=True, stderr=subprocess.DEVNULL).strip()


def git_metadata(path: Path) -> dict[str, Any]:
    """Return honest unavailable metadata instead of inventing a revision."""
    try:
        revision = _git(path, "rev-parse", "HEAD")
        status = _git(path, "status", "--porcelain=v1")
        return {"status": "available", "commit": revision, "dirty": bool(status), "changed_files": status.splitlines()}
    except (OSError, subprocess.CalledProcessError):
        return {"status": "unavailable", "commit": None, "dirty": None, "changed_files": []}


def runtime_metadata() -> dict[str, Any]:
    result: dict[str, Any] = {"python": sys.version.split()[0], "platform": platform.platform(), "torch": "unavailable",
                              "transformers": "unavailable", "cuda": "unavailable", "gpus": []}
    try:
        import torch
        result["torch"] = torch.__version__
        result["cuda"] = torch.version.cuda or "unavailable"
        if torch.cuda.is_available():
            result["gpus"] = [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())]
    except ImportError:
        pass
    try:
        import transformers
        result["transformers"] = transformers.__version__
    except ImportError:
        pass
    return result


def collect_provenance(
    config: ResolvedConfig, *, model_revisions: Mapping[str, str] | None = None,
    data_revisions: Mapping[str, str] | None = None, command: Sequence[str] | None = None,
    started_at_utc: str | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "started_at_utc": started_at_utc or datetime.now(timezone.utc).isoformat(),
        "ended_at_utc": None,
        "command": list(command) if command is not None else list(sys.argv),
        "resolved_config_sha256": config.sha256,
        "code": {"main": git_metadata(WORK_ROOT), "references": {name: git_metadata(path) for name, path in REFERENCE_ROOTS.items()}},
        "models": dict(model_revisions or {}),
        "data": dict(data_revisions or {}),
        "runtime": runtime_metadata(),
    }


def write_run_provenance(
    output_dir: Path | str, config: ResolvedConfig, *, model_revisions: Mapping[str, str] | None = None,
    data_revisions: Mapping[str, str] | None = None, command: Sequence[str] | None = None,
) -> Path:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    write_resolved_config(config, destination / "config.resolved.yaml")
    metadata = collect_provenance(config, model_revisions=model_revisions, data_revisions=data_revisions, command=command)
    metadata["ended_at_utc"] = datetime.now(timezone.utc).isoformat()
    path = destination / "provenance.json"
    path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    revision = metadata["code"]["main"]["commit"] or "unavailable"
    (destination / "git_revision.txt").write_text(revision + "\n", encoding="utf-8")
    return path
