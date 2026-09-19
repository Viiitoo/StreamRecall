"""Versioned result paths and reproducibility metadata for experiment outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = PROJECT_ROOT / "results"


def _git(*args: str) -> str:
    """Run git for this checkout and return stripped stdout."""
    return subprocess.check_output(
        ["git", "-c", f"safe.directory={PROJECT_ROOT}", "-C", str(PROJECT_ROOT), *args],
        text=True,
        stderr=subprocess.DEVNULL,
    ).strip()


def git_metadata() -> dict[str, Any]:
    """Return the immutable commit identity plus the working-tree state."""
    revision = _git("rev-parse", "HEAD")
    status = _git("status", "--porcelain=v1")
    try:
        branch = _git("branch", "--show-current")
    except subprocess.CalledProcessError:  # pragma: no cover - detached HEAD
        branch = None
    try:
        describe = _git("describe", "--always", "--long", "--dirty")
    except subprocess.CalledProcessError:  # pragma: no cover - shallow unusual checkout
        describe = revision
    return {
        "commit": revision,
        "short_commit": revision[:12],
        "describe": describe,
        "branch": branch or None,
        "is_dirty": bool(status),
        "status_porcelain": status.splitlines(),
    }


def code_version() -> str:
    """Directory name for results produced from the current Git commit."""
    return git_metadata()["commit"]


def versioned_result_path(path: Path | str) -> Path:
    """Insert the current commit immediately below this project's ``results``.

    Paths outside this checkout's ``results`` tree are rejected: experiment,
    evaluation, and smoke artifacts must all follow the same versioned layout.
    Paths that are already versioned are left untouched, which makes the
    function safe for nested chunk runs.
    """
    output = Path(path)
    absolute = output if output.is_absolute() else (Path.cwd() / output)
    absolute = absolute.resolve(strict=False)
    results_root = RESULTS_ROOT.resolve()
    try:
        relative = absolute.relative_to(results_root)
    except ValueError as exc:
        raise ValueError(f"result path must be below {results_root}: {absolute}") from exc
    version = code_version()
    if relative.parts[:1] == (version,):
        return absolute
    return results_root / version / relative


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_artifact_manifest(
    output_dir: Path | str,
    *,
    filename: str = "artifact_manifest.json",
) -> Path:
    """Hash every completed result artifact except the manifest itself.

    The checksum index cannot recursively include its own digest.  All other
    regular files below ``output_dir`` are listed with a stable relative path,
    byte count, and SHA-256.  Symlinks are rejected so the manifest always
    describes self-contained result bytes.
    """
    destination = Path(output_dir).resolve()
    if not destination.is_dir() or Path(filename).name != filename:
        raise ValueError("artifact manifest needs an existing result directory and safe filename")
    manifest_path = destination / filename
    rows = []
    for path in sorted(destination.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"result artifact cannot be a symlink: {path}")
        if not path.is_file() or path == manifest_path:
            continue
        rows.append({
            "path": str(path.relative_to(destination)),
            "size_bytes": path.stat().st_size,
            "sha256": _sha256(path),
        })
    payload = {
        "schema_version": 1,
        "hash": "sha256",
        "self_excluded": filename,
        "file_count": len(rows),
        "total_bytes": sum(int(row["size_bytes"]) for row in rows),
        "files": rows,
    }
    manifest_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest_path


def _write_configuration_snapshot(path: Path, configuration: Mapping[str, Any]) -> None:
    if path.suffix.lower() in {".yaml", ".yml"}:
        text = yaml.safe_dump(dict(configuration), allow_unicode=True, sort_keys=True)
    else:
        text = json.dumps(configuration, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n"
    path.write_text(text, encoding="utf-8")


def write_provenance(
    output_dir: Path | str,
    *,
    configuration: Mapping[str, Any] | None = None,
    config_path: Path | str | None = None,
    config_filename: str = "config.resolved.yaml",
    command: Sequence[str] | None = None,
    code: Mapping[str, Any] | None = None,
    extra_metadata: Mapping[str, Any] | None = None,
) -> Path:
    """Freeze config and execution metadata next to a result directory."""
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    source_path: Path | None = None
    if config_path is not None:
        source_path = Path(config_path).resolve()
        snapshot_path = destination / config_filename
        if configuration is None:
            configuration = yaml.safe_load(source_path.read_text(encoding="utf-8"))
        # Always serialize the resolved configuration.  Copying the source YAML
        # would lose CLI/runtime overrides while claiming that the snapshot was
        # final, which defeats the provenance contract.
        _write_configuration_snapshot(snapshot_path, configuration)
    elif configuration is not None:
        snapshot_path = destination / config_filename
        _write_configuration_snapshot(snapshot_path, configuration)
    else:
        snapshot_path = None

    metadata: dict[str, Any] = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "code": dict(code) if code is not None else git_metadata(),
        "command": list(command) if command is not None else list(sys.argv),
        "runtime": {
            "hostname": socket.gethostname(),
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "working_directory": str(Path.cwd()),
        },
        "configuration": configuration,
    }
    if extra_metadata:
        metadata.update(extra_metadata)
    if snapshot_path is not None:
        metadata["configuration_file"] = snapshot_path.name
        metadata["configuration_sha256"] = _sha256(snapshot_path)
    if source_path is not None:
        metadata["configuration_source"] = str(source_path)
        metadata["configuration_source_sha256"] = _sha256(source_path)
    provenance_path = destination / "provenance.json"
    provenance_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    commit = metadata["code"].get("commit")
    if commit:
        (destination / "git_revision.txt").write_text(f"{commit}\n", encoding="utf-8")
    return provenance_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--print-code-version", action="store_true")
    parser.add_argument("--output-dir")
    parser.add_argument("--config")
    parser.add_argument("--config-filename", default="config.resolved.yaml")
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        metavar="DOTTED_KEY=YAML_VALUE",
        help="Add a resolved configuration value; may be repeated.",
    )
    parser.add_argument(
        "--override-string",
        action="append",
        default=[],
        metavar="DOTTED_KEY=VALUE",
        help="Add a resolved string value without YAML type coercion.",
    )
    parser.add_argument("command", nargs=argparse.REMAINDER)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.print_code_version:
        print(code_version())
        return 0
    if not args.output_dir:
        raise SystemExit("--output-dir is required unless --print-code-version is used")
    configuration = None
    if args.config:
        configuration = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    elif args.override or args.override_string:
        configuration = {}
    for item, parse_yaml in [*((item, True) for item in args.override), *((item, False) for item in args.override_string)]:
        if "=" not in item:
            raise SystemExit(f"invalid --override {item!r}; expected KEY=VALUE")
        dotted_key, raw_value = item.split("=", 1)
        if not dotted_key or any(not part for part in dotted_key.split(".")):
            raise SystemExit(f"invalid override key: {dotted_key!r}")
        target = configuration
        for part in dotted_key.split(".")[:-1]:
            existing = target.setdefault(part, {})
            if not isinstance(existing, dict):
                raise SystemExit(f"override parent is not a mapping: {dotted_key!r}")
            target = existing
        target[dotted_key.split(".")[-1]] = yaml.safe_load(raw_value) if parse_yaml else raw_value
    command = args.command or None
    if command and command[0] == "--":
        command = command[1:]
    write_provenance(
        args.output_dir,
        configuration=configuration,
        config_path=args.config,
        config_filename=args.config_filename,
        command=command,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
