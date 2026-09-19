"""Deterministic method/budget/dataset matrix expansion and resumable execution."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Sequence


CommandRunner = Callable[[Sequence[str], Dict[str, str]], int]


@dataclass(frozen=True)
class MatrixVideo:
    video_id: str
    path: str


@dataclass(frozen=True)
class MatrixQuery:
    query_id: str
    video_id: str
    query: str


@dataclass(frozen=True)
class MatrixDataset:
    name: str
    videos: tuple[MatrixVideo, ...]
    queries: tuple[MatrixQuery, ...]


@dataclass(frozen=True)
class MatrixSpec:
    methods: tuple[str, ...]
    budgets: tuple[str, ...]
    datasets: tuple[MatrixDataset, ...]
    rhos: tuple[float, ...]
    gpu_ids: tuple[str, ...]
    ingest_command: tuple[str, ...]
    query_command: tuple[str, ...]
    protocol_config: str = ""
    model_config: str = ""
    run_id: str = "run-001"
    query_grid: tuple[dict[str, Any], ...] = ({},)
    query_mode: str = "per_query"

    def __post_init__(self) -> None:
        if not self.methods or not self.budgets or not self.datasets or not self.gpu_ids:
            raise ValueError("matrix axes and GPU IDs must not be empty")
        if not self.rhos or any(not 0 < value <= 1 for value in self.rhos):
            raise ValueError("matrix rho values must be in (0,1]")
        if tuple(sorted(set(self.rhos))) != self.rhos:
            raise ValueError("matrix rho values must be unique and sorted")
        if not self.ingest_command or not self.query_command or not self.run_id:
            raise ValueError("matrix commands and run ID are required")
        if not self.query_grid or any(not isinstance(row, dict) for row in self.query_grid):
            raise ValueError("matrix query grid must contain mappings")
        if len({_canonical_hash(row) for row in self.query_grid}) != len(self.query_grid):
            raise ValueError("matrix query grid contains duplicate variants")
        if self.query_mode not in ("per_query", "per_video_grid"):
            raise ValueError("matrix query mode is invalid")


@dataclass(frozen=True)
class MatrixJob:
    method: str
    budget: str
    dataset: str
    video: MatrixVideo
    queries: tuple[MatrixQuery, ...]
    rhos: tuple[float, ...]
    gpu_id: str
    config_hash: str
    job_hash: str
    job_dir: Path
    ingest_command: tuple[str, ...]
    query_command: tuple[str, ...]
    protocol_config: str
    model_config: str
    query_grid: tuple[dict[str, Any], ...]
    query_mode: str


def _canonical_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _file_identity(value: str) -> dict[str, str]:
    path = Path(value)
    if not path.is_file():
        return {"value": value, "sha256": "unavailable"}
    return {"value": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def expand_matrix(spec: MatrixSpec, output_root: Path) -> list[MatrixJob]:
    jobs = []
    for method in spec.methods:
        for budget in spec.budgets:
            config_hash = _canonical_hash({
                "method": _file_identity(method), "budget": _file_identity(budget),
                "rhos": spec.rhos,
                "protocol": _file_identity(spec.protocol_config),
                "models": _file_identity(spec.model_config),
            })
            for dataset in spec.datasets:
                video_ids = [video.video_id for video in dataset.videos]
                if len(video_ids) != len(set(video_ids)):
                    raise ValueError(f"dataset {dataset.name} contains duplicate video IDs")
                for video in dataset.videos:
                    queries = tuple(query for query in dataset.queries if query.video_id == video.video_id)
                    shard = int(hashlib.sha256(video.video_id.encode("utf-8")).hexdigest(), 16)
                    gpu_id = spec.gpu_ids[shard % len(spec.gpu_ids)]
                    descriptor = {
                        "method": method, "budget": budget, "dataset": dataset.name,
                        "video": asdict(video), "queries": [asdict(query) for query in queries],
                        "rhos": spec.rhos, "config_hash": config_hash,
                        "ingest_command": spec.ingest_command, "query_command": spec.query_command,
                        "protocol_config": spec.protocol_config,
                        "model_config": spec.model_config,
                        "query_grid": spec.query_grid,
                        "query_mode": spec.query_mode,
                    }
                    job_hash = _canonical_hash(descriptor)
                    method_name = Path(method).stem
                    job_dir = (
                        output_root / method_name / dataset.name / config_hash / spec.run_id / video.video_id
                    )
                    jobs.append(MatrixJob(
                        method, budget, dataset.name, video, queries, spec.rhos, gpu_id,
                        config_hash, job_hash, job_dir, spec.ingest_command, spec.query_command,
                        spec.protocol_config, spec.model_config, spec.query_grid, spec.query_mode,
                    ))
    return sorted(jobs, key=lambda job: (
        job.gpu_id, job.method, job.budget, job.dataset, job.video.video_id,
    ))


def _format_command(template: Sequence[str], values: dict[str, str]) -> tuple[str, ...]:
    try:
        return tuple(part.format_map(values) for part in template)
    except KeyError as exc:
        raise ValueError(f"unknown matrix command placeholder: {exc.args[0]}") from exc


def _read_marker(path: Path, expected_hash: str) -> bool:
    if not path.exists():
        return False
    row = json.loads(path.read_text(encoding="utf-8"))
    if row.get("job_hash") != expected_hash:
        raise ValueError(f"completion marker hash conflict: {path}")
    return row.get("status") == "complete"


def _write_marker(path: Path, job_hash: str, kind: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps({
        "format_version": 1, "job_hash": job_hash, "kind": kind, "status": "complete",
    }, sort_keys=True) + "\n"
    handle, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            stream.write(payload)
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _subprocess_runner(command: Sequence[str], env: dict[str, str]) -> int:
    return subprocess.run(list(command), env=env, check=False).returncode


def execute_matrix(
    jobs: Iterable[MatrixJob], *, runner: CommandRunner = _subprocess_runner,
) -> dict[str, int]:
    job_list = list(jobs)
    shards: dict[str, list[MatrixJob]] = {}
    for job in job_list:
        shards.setdefault(job.gpu_id, []).append(job)
    counters = {"ingest_executed": 0, "ingest_resumed": 0, "query_executed": 0, "query_resumed": 0}

    def execute_shard(gpu_id: str, shard_jobs: list[MatrixJob]) -> dict[str, int]:
        local = {key: 0 for key in counters}
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = gpu_id
        for job in shard_jobs:
            job.job_dir.mkdir(parents=True, exist_ok=True)
            config_path = job.job_dir / "config.resolved.json"
            config_payload = {
                "job_hash": job.job_hash, "config_hash": job.config_hash,
                "method": job.method, "budget": job.budget, "dataset": job.dataset,
                "video": asdict(job.video), "rhos": list(job.rhos), "gpu_id": job.gpu_id,
                "query_grid": list(job.query_grid),
                "query_mode": job.query_mode,
            }
            if config_path.exists():
                existing = json.loads(config_path.read_text(encoding="utf-8"))
                if existing != config_payload:
                    raise ValueError(f"matrix job directory config conflict: {job.job_dir}")
            else:
                config_path.write_text(
                    json.dumps(config_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8",
                )
            resolved_config_path = job.job_dir / "query.resolved.yaml"
            budget_values: dict[str, Any] = {}
            model_values: dict[str, Any] = {}
            if Path(job.budget).is_file():
                import yaml

                budget_values = yaml.safe_load(Path(job.budget).read_text(encoding="utf-8"))
            if job.model_config and Path(job.model_config).is_file():
                import yaml

                model_values = yaml.safe_load(Path(job.model_config).read_text(encoding="utf-8"))
            if job.protocol_config and job.model_config:
                from streamtimelens.config import resolve_config, write_resolved_config

                resolved = resolve_config(
                    protocol_path=job.protocol_config, budget_path=job.budget,
                    method_path=job.method, model_path=job.model_config,
                )
                if resolved_config_path.exists():
                    from streamtimelens.config import read_resolved_config

                    if read_resolved_config(resolved_config_path).sha256 != resolved.sha256:
                        raise ValueError(f"matrix resolved config conflict: {job.job_dir}")
                else:
                    write_resolved_config(resolved, resolved_config_path)
            ingest_marker = job.job_dir / ".ingest.complete.json"
            values = {
                "method": job.method, "budget": job.budget, "dataset": job.dataset,
                "method_name": "full" if Path(job.method).stem == "streamtimelens" else Path(job.method).stem,
                "budget_bytes": str(budget_values.get("memory_bytes", "")),
                "writer_calls": str(budget_values.get("writer_calls_per_minute", "")),
                "writer_model": str(model_values.get("writer_model", "")),
                "text_embedder": str(model_values.get("text_embedder", "")),
                "clip_model": str(model_values.get("clip_model", "")),
                "resolved_config": str(resolved_config_path),
                "video_id": job.video.video_id, "video_path": job.video.path,
                "job_dir": str(job.job_dir), "rhos": ",".join(f"{rho:g}" for rho in job.rhos),
            }
            if _read_marker(ingest_marker, job.job_hash):
                local["ingest_resumed"] += 1
            else:
                if runner(_format_command(job.ingest_command, values), env) != 0:
                    raise RuntimeError(f"matrix ingest failed: {job.video.video_id}")
                _write_marker(ingest_marker, job.job_hash, "ingest")
                local["ingest_executed"] += 1
            if job.query_mode == "per_video_grid":
                queries_path = job.job_dir / "queries.input.jsonl"
                queries_path.write_text(
                    "".join(json.dumps(asdict(query), sort_keys=True) + "\n" for query in job.queries),
                    encoding="utf-8",
                )
                grid_path = job.job_dir / "query_grid.resolved.json"
                grid_path.write_text(
                    json.dumps(list(job.query_grid), indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                query_hash = _canonical_hash({
                    "job_hash": job.job_hash, "queries": [asdict(query) for query in job.queries],
                    "query_grid": job.query_grid, "rhos": job.rhos,
                })
                marker = job.job_dir / ".query_batch.complete.json"
                batch_values = {
                    **values, "queries": str(queries_path), "query_grid": str(grid_path),
                    "snapshot_root": str(job.job_dir / "snapshots" / job.video.video_id),
                    "predictions_dir": str(job.job_dir / "predictions"),
                }
                if _read_marker(marker, query_hash):
                    local["query_resumed"] += 1
                else:
                    if runner(_format_command(job.query_command, batch_values), env) != 0:
                        raise RuntimeError(f"matrix batch query failed: {job.video.video_id}")
                    _write_marker(marker, query_hash, "query_batch")
                    local["query_executed"] += 1
                continue
            for query in job.queries:
                for rho in job.rhos:
                    for variant_index, query_config in enumerate(job.query_grid):
                        variant_hash = _canonical_hash(query_config)
                        variant = f"qg_{variant_index:02d}_{variant_hash[:8]}"
                        query_hash = _canonical_hash({
                            "job_hash": job.job_hash, "query": asdict(query), "rho": rho,
                            "query_config": query_config,
                        })
                        suffix = "" if len(job.query_grid) == 1 and not query_config else f".{variant}"
                        marker = job.job_dir / "queries" / (
                            f"{query.query_id}.rho_{rho:.2f}{suffix}.complete.json"
                        )
                        query_values = {
                            **values, "query_id": query.query_id, "query": query.query,
                            "rho": f"{rho:.2f}", "query_variant": variant,
                            "snapshot": str(job.job_dir / "snapshots" / job.video.video_id / f"rho_{rho:.2f}"),
                            **{key: str(value) for key, value in query_config.items()},
                        }
                        if _read_marker(marker, query_hash):
                            local["query_resumed"] += 1
                        else:
                            if runner(_format_command(job.query_command, query_values), env) != 0:
                                raise RuntimeError(
                                    f"matrix query failed: {query.query_id}@{rho:.2f}/{variant}"
                                )
                            _write_marker(marker, query_hash, "query")
                            local["query_executed"] += 1
        return local

    with ThreadPoolExecutor(max_workers=max(1, len(shards))) as pool:
        futures = [pool.submit(execute_shard, gpu_id, values) for gpu_id, values in sorted(shards.items())]
        for future in futures:
            local = future.result()
            for key, value in local.items():
                counters[key] += value
    return counters
