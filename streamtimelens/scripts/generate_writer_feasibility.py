#!/usr/bin/env python3
"""Generate resumable raw outputs for one fixed writer checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from functools import partial
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT / "src"))
sys.path.insert(0, str(PACKAGE_ROOT.parent / "src"))

from baas.provenance import versioned_result_path, write_provenance
from streamtimelens.evaluation.writer_generation import (
    WriterChunk, decord_frame_loader, generate_writer_record,
)
from streamtimelens.refiner.model_service import TimeLensModelService


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _rows(path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunks", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--output", type=Path, required=True, help="Path below work/results")
    parser.add_argument(
        "--video-root", type=Path,
        help="Optional runtime root containing <video_id>.mp4 (for container path remapping)",
    )
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    chunks = [WriterChunk.from_mapping(row) for row in _rows(args.chunks)]
    if len({chunk.chunk_id for chunk in chunks}) != len(chunks):
        parser.error("writer chunk input contains duplicate chunk IDs")
    output = versioned_result_path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    records_path = output / "writer_outputs.jsonl"
    existing_rows = _rows(records_path) if args.resume and records_path.is_file() else []
    if records_path.exists() and not args.resume:
        parser.error(f"records already exist; pass --resume or choose another output: {records_path}")
    existing: dict[str, dict[str, object]] = {}
    expected_ids = {chunk.chunk_id for chunk in chunks}
    for row in existing_rows:
        chunk_id = str(row.get("chunk_id", ""))
        if row.get("model") != args.model_id or chunk_id not in expected_ids or chunk_id in existing:
            parser.error("resume records contain a model/chunk mismatch or duplicate")
        existing[chunk_id] = row

    service = TimeLensModelService.get(args.model, device_map=args.device_map)
    configuration = {
        "chunks": str(args.chunks.resolve()), "chunks_sha256": _sha256(args.chunks),
        "chunk_count": len(chunks), "model": str(args.model.resolve()),
        "model_id": args.model_id, "model_revision": args.model_revision,
        "video_root": str(args.video_root.resolve()) if args.video_root else None,
        "device_map": args.device_map, "resume": args.resume,
    }
    write_provenance(
        output, configuration=configuration, config_filename="config.resolved.json",
        extra_metadata={"model_hashes": service.hashes},
    )
    with records_path.open("a", encoding="utf-8") as stream:
        for ordinal, chunk in enumerate(chunks, start=1):
            if chunk.chunk_id in existing:
                continue
            row = generate_writer_record(
                chunk, model_id=args.model_id, model_revision=args.model_revision,
                service=service,
                frame_loader=partial(decord_frame_loader, video_root=args.video_root),
            )
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            stream.flush()
            print(json.dumps({
                "model": args.model_id, "completed": ordinal, "total": len(chunks),
                "chunk_id": chunk.chunk_id,
            }, sort_keys=True), flush=True)
    final_rows = _rows(records_path)
    if len(final_rows) != len(chunks):
        raise RuntimeError(f"writer generation is incomplete: {len(final_rows)}/{len(chunks)}")
    final_rows.sort(key=lambda row: str(row["chunk_id"]))
    records_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in final_rows),
        encoding="utf-8",
    )
    print(json.dumps({
        "model": args.model_id, "records": len(final_rows), "output": str(output),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
