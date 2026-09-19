#!/usr/bin/env python3
"""Version and annotate legacy output already present below ``results``."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from baas.provenance import RESULTS_ROOT, code_version, write_provenance


RESULT_SUFFIXES = {".json", ".jsonl", ".log", ".csv", ".txt"}


def _is_version_directory(path: Path) -> bool:
    return len(path.name) == 40 and all(character in "0123456789abcdef" for character in path.name)


def _legacy_top_level_dirs() -> list[Path]:
    return [
        path for path in RESULTS_ROOT.iterdir()
        if path.is_dir() and not _is_version_directory(path) and any(item.is_file() for item in path.rglob("*"))
    ]


def _output_dirs(version_root: Path) -> list[Path]:
    candidates = {
        path.parent
        for path in version_root.rglob("*")
        if path.is_file() and path.suffix in RESULT_SUFFIXES
    }
    # A run directory can contain multiple output files.  Keep its shallowest
    # directory, not one provenance record for every individual artifact.
    return sorted(
        (path for path in candidates if not any(parent in candidates for parent in path.parents)),
        key=lambda path: str(path),
    )


def _configuration_for(directory: Path) -> dict[str, Any]:
    smoke_files = sorted(directory.glob("p1_smoke_sample_*.json"))
    if smoke_files:
        samples = []
        for path in smoke_files:
            record = json.loads(path.read_text(encoding="utf-8"))
            samples.append({
                "output": path.name,
                "sample_index": record.get("sample_index"),
                "sampling": record.get("sampling"),
            })
        return {"kind": "p1_official_smoke", "runs": samples}
    if (directory / "charades-timelens.jsonl").is_file():
        return {
            "kind": "p2_official_timelens_evaluation",
            "dataset": "charades-timelens",
            "split": "test",
            "model": "TimeLens-7B",
            "sampling": {"min_tokens": 64, "total_tokens": 14336, "fps": 2},
            "metrics_file": "charades-timelens.log",
            "prediction_file": "charades-timelens.jsonl",
        }
    return {"kind": "legacy_result", "unrecoverable_configuration": "unknown"}


def main() -> int:
    version = code_version()
    version_root = RESULTS_ROOT / version
    migrated: list[tuple[Path, Path]] = []
    for source in _legacy_top_level_dirs():
        destination = version_root / source.name
        if destination.exists():
            raise RuntimeError(f"refusing to merge legacy output into {destination}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source), str(destination))
        migrated.append((source, destination))

    historical_code = {
        "commit": version,
        "short_commit": version[:12],
        "describe": None,
        "branch": None,
        "is_dirty": None,
        "status_porcelain": None,
        "revision_inferred": True,
    }
    for source, destination in migrated:
        for output_dir in _output_dirs(destination):
            write_provenance(
                output_dir,
                configuration=_configuration_for(output_dir),
                config_filename="config.resolved.json",
                code=historical_code,
                command=[str(Path(__file__).resolve())],
                extra_metadata={"backfilled": True, "legacy_original_path": str(source)},
            )
            print(f"backfilled {output_dir.relative_to(RESULTS_ROOT)}")

    # Empty legacy paths do not represent results and should not look like
    # non-versioned result directories after migration.
    for path in sorted(RESULTS_ROOT.iterdir(), key=lambda item: item.name):
        if path.is_dir() and not _is_version_directory(path):
            try:
                path.rmdir()
            except OSError:
                pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
