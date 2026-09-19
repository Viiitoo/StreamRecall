"""Deterministic pre-formal regression for snapshot and query contracts."""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from streamtimelens.config import BudgetConfig, ProtocolConfig
from streamtimelens.memory.cards import EvidenceCard
from streamtimelens.protocol.snapshot import SnapshotReader, SnapshotWriter
from streamtimelens.protocol.types import Budget, VideoMeta
from streamtimelens.retrieval.embedder import CardTextEmbedder
from streamtimelens.retrieval.query_pipeline import answer_card_snapshot


GOLDEN_RATIOS = (.25, .5, .75, 1.0)
GOLDEN_QUERIES = (
    ("golden-door", "person opens door"),
    ("golden-walk", "person walks away"),
    ("golden-cup", "person moves cup"),
)
_PIXEL = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
)


class _GoldenEmbeddingBackend:
    def encode(self, texts: Sequence[str]) -> Any:
        vectors = []
        for text in texts:
            normalized = text.lower()
            if "door" in normalized:
                vectors.append([1, 0, 0, 0])
            elif "walk" in normalized:
                vectors.append([0, 1, 0, 0])
            elif "cup" in normalized:
                vectors.append([0, 0, 1, 0])
            else:
                vectors.append([0, 0, 0, 1])
        return np.asarray(vectors, dtype=np.float32)


class _GoldenRefiner:
    last_call_stats = {"generated_tokens": 6, "backend": "golden"}

    def generate(self, messages: Sequence[dict[str, Any]], videos: Sequence[Any], max_new_tokens: int) -> str:
        del videos, max_new_tokens
        prompt = str(messages).lower()
        if "described by: person opens door\\n" in prompt:
            return "The event occurs from 2.2 - 3.8 seconds."
        if "described by: person walks away\\n" in prompt:
            return "The event occurs from 4.7 - 5.8 seconds."
        return "The event occurs from 7.0 - 8.8 seconds."


def _cards(embedder: CardTextEmbedder) -> list[EvidenceCard]:
    definitions = (
        ("wait", .5, 1.5, "person waits", (1,)),
        ("door", 2.0, 4.0, "person opens door", (2, 4)),
        ("walk", 4.5, 6.0, "person walks away", (5, 6)),
        ("cup", 6.5, 9.0, "person moves cup", (7, 9)),
    )
    cards = []
    for card_id, start, end, summary, indices in definitions:
        refs = [f"{index:09d}.jpg" for index in indices]
        cards.append(EvidenceCard(
            card_id, 0, start, end, summary, normalized_text=summary,
            raw_ref_ids=refs, raw_ref_status={ref: "available" for ref in refs},
            boundary_cache={
                "left_frame_ids": refs[:1], "right_frame_ids": refs[-1:],
                "internal_frame_ids": [],
            },
        ))
    embedder.embed_cards(cards)
    return cards


def _canonical_prediction(payload: dict[str, Any]) -> dict[str, Any]:
    attempts = payload["diagnostics"]["refinement_attempts"]
    return {
        "rho_q": payload["rho_q"],
        "query_id": payload["query_id"],
        "status": payload["status"],
        "span": payload["span"],
        "retrieved_cards": payload["retrieved_cards"],
        "candidates": payload["candidates"],
        "raw_answers": payload["raw_answers"],
        "attempts": [{
            "status": attempt["status"],
            "parsed_span": (
                list(attempt["parsed_span"]) if attempt["parsed_span"] is not None else None
            ),
            "parse_status": attempt["parse_status"],
            "frame_refs": list(attempt["frame_refs"]),
        } for attempt in attempts],
    }


def _resource_shape(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "resource_fields": sorted(payload["resource"]),
        "query_embedding_fields": sorted(payload["resource"]["query_embedding"]),
        "refiner_resource_fields": [
            sorted(resource) for resource in payload["resource"]["refiner"]
        ],
    }


def run_formal_golden(output_root: Path) -> dict[str, Any]:
    """Run one video, four arrival ratios, and three queries without external models."""
    output_root.mkdir(parents=True, exist_ok=True)
    embedder = CardTextEmbedder(
        "deterministic-golden-text", revision="v1", backend=_GoldenEmbeddingBackend(),
    )
    cards = _cards(embedder)
    meta = VideoMeta("golden-video", 10.0, 1.0, 10, 1, 1)
    budget = Budget(256 * 1024, 4, refine_calls_per_query=1, max_frames_per_refine=4)
    snapshots = output_root / "snapshots"
    writer = SnapshotWriter(snapshots, source_revision="formal-golden-v1")
    hashes = {}
    predictions = []
    resource_shape = None
    for ratio in GOLDEN_RATIOS:
        t_q = ratio * meta.duration_s
        retained = [card for card in cards if card.t_end <= t_q]
        refs = sorted({ref for card in retained for ref in card.raw_ref_ids})
        name = f"rho_{ratio:.2f}"
        writer.write(
            name=name, t_q=t_q, meta=meta, budget=budget,
            cards=[card.serializable() for card in retained],
            raw_frames=[(ref, _PIXEL) for ref in refs],
            raw_metadata={
                ref: {"blob": ref, "frame_index": int(ref[:9]),
                      "timestamp_s": float(int(ref[:9])), "novelty": 0.0}
                for ref in refs
            },
            writer_calls=len(retained), config={"case": "formal-golden-v1"},
            embedder=vars(embedder.metadata),
        )
        snapshot = SnapshotReader(snapshots / name)
        hashes[name] = (snapshots / name / "manifest.sha256").read_text(encoding="ascii").strip()
        for query_id, query in GOLDEN_QUERIES:
            prediction = answer_card_snapshot(
                query_id=query_id, query=query, snapshot=snapshot, embedder=embedder,
                protocol=ProtocolConfig(not_found_threshold=0.0),
                budget=BudgetConfig(
                    budget.memory_bytes, budget.writer_calls_per_minute,
                    refine_calls_per_query=1, max_frames_per_refine=4,
                ),
                refiner_service=_GoldenRefiner(),
            ).to_dict()
            current_shape = _resource_shape(prediction)
            if resource_shape is None:
                resource_shape = current_shape
            elif resource_shape != current_shape:
                raise AssertionError("golden query resources do not have a stable schema")
            predictions.append(_canonical_prediction(prediction))
    return {
        "schema_version": 1,
        "case": "one_video_four_rho_three_queries",
        "snapshot_hashes": hashes,
        "predictions": predictions,
        "resource_shape": resource_shape,
        "offline_timelens_entrypoint": "scripts/run_offline_timelens.py",
    }


def assert_golden_matches(actual: dict[str, Any], expected_path: Path) -> None:
    expected = json.loads(expected_path.read_text(encoding="utf-8"))
    if actual != expected:
        raise AssertionError(
            "formal golden regression changed; inspect the actual artifact and update "
            "the fixture only for an intentional model/schema version change"
        )
