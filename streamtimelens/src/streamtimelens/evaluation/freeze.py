"""Gate-checked development configuration freeze."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping

from streamtimelens.config import ResolvedConfig
from streamtimelens.refiner.prompts import SPARSE_LOCAL_TEMPLATE_SHA256, SPARSE_LOCAL_VERSION
from streamtimelens.retrieval.embedder import EMBEDDING_TEMPLATE_SHA256, EMBEDDING_TEMPLATE_VERSION
from streamtimelens.writer.prompts import WRITER_PROMPT_VERSION, build_writer_prompt


_VISUAL_PARAMETER_KEYS = {
    "writer", "clip_model", "clip_revision", "clip_sha256",
    "jpeg_short_edge", "jpeg_quality", "embedding_precision",
    "anchor_fraction", "top_k", "expand_neighbors", "merge_gap_s",
    "coarse_margin_s", "timelens_enabled",
}


def _mapping_sha256(value: Mapping[str, Any]) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def freeze_visual_configuration(
    resolved_configs: Mapping[str, ResolvedConfig],
    selection: Mapping[str, Any],
    protocol_audit: Mapping[str, Any],
    refiner_gate: Mapping[str, Any],
    visual_parameters: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Freeze the visual route without reviving the structured-writer gates."""
    if not bool(protocol_audit.get("passed")):
        raise ValueError("cannot freeze before the visual protocol audit passes")
    if not bool(selection.get("passed", True)):
        raise ValueError("cannot freeze a failed visual dev selection")
    selected = tuple(map(str, selection.get("selected_config_ids", ())))
    if not selected or any(config_id not in resolved_configs for config_id in selected):
        raise ValueError("visual freeze selection does not match supplied resolved configs")
    decision = str(refiner_gate.get("decision", ""))
    if decision not in ("enable", "disable"):
        raise ValueError("visual refiner decision must be enable or disable")
    if decision == "enable" and not bool(refiner_gate.get("passed")):
        raise ValueError("cannot enable TimeLens before the visual refiner gate passes")

    configs = {}
    for config_id in sorted(selected):
        parameters = visual_parameters.get(config_id)
        if not isinstance(parameters, Mapping):
            raise ValueError(f"visual parameters are missing for {config_id}")
        missing = _VISUAL_PARAMETER_KEYS - set(parameters)
        if missing:
            raise ValueError(
                f"visual parameters for {config_id} omit: {', '.join(sorted(missing))}"
            )
        if parameters.get("writer") != "none":
            raise ValueError("frozen visual ingestion must record writer=none")
        if bool(parameters.get("timelens_enabled")) != (decision == "enable"):
            raise ValueError("visual parameters disagree with the refiner decision")
        configs[config_id] = {
            "sha256": resolved_configs[config_id].sha256,
            "resolved": resolved_configs[config_id].canonical_dict(),
            "visual": dict(parameters),
            "visual_sha256": _mapping_sha256(parameters),
        }
    return {
        "schema_version": 1,
        "route": "semantic_visual_cache_posthoc_localization",
        "status": "frozen",
        "selected_writer": None,
        "refiner_decision": decision,
        "prompt_hashes": {
            "refiner_version": SPARSE_LOCAL_VERSION,
            "refiner_template_sha256": SPARSE_LOCAL_TEMPLATE_SHA256,
        },
        "configs": configs,
        "sources": {
            "protocol_audit_sha256": _mapping_sha256(protocol_audit),
            "selection_sha256": _mapping_sha256(selection),
            "refiner_gate_sha256": _mapping_sha256(refiner_gate),
        },
        "post_freeze_policy": (
            "No parameter or model changes after formal-test access; only explicit bug fixes "
            "are allowed and every affected method must be rerun."
        ),
    }


def freeze_dev_configuration(
    resolved_configs: Mapping[str, ResolvedConfig],
    selection: Mapping[str, Any],
    oracle_metrics: Mapping[str, Any],
    writer_metrics: Mapping[str, Any],
) -> dict[str, Any]:
    gate = oracle_metrics.get("p0_gate")
    if not isinstance(gate, Mapping) or not bool(gate.get("passed")):
        raise ValueError("cannot freeze before the Oracle P0 gate passes")
    if writer_metrics.get("decision") != "go" or not writer_metrics.get("selected_writer"):
        raise ValueError("cannot freeze before writer feasibility selects a checkpoint")
    selected = tuple(map(str, selection.get("selected_config_ids", ())))
    if not selected or any(config_id not in resolved_configs for config_id in selected):
        raise ValueError("freeze selection does not match supplied resolved configs")
    writer_prompt = build_writer_prompt(segment=(0, 1), sampled_timestamps=(0, 1))
    configs = {
        config_id: {
            "sha256": resolved_configs[config_id].sha256,
            "resolved": resolved_configs[config_id].canonical_dict(),
        }
        for config_id in sorted(selected)
    }
    sources = {
        "selection_sha256": hashlib.sha256(json.dumps(
            selection, sort_keys=True, separators=(",", ":"), allow_nan=False,
        ).encode("utf-8")).hexdigest(),
        "oracle_metrics_sha256": hashlib.sha256(json.dumps(
            oracle_metrics, sort_keys=True, separators=(",", ":"), allow_nan=False,
        ).encode("utf-8")).hexdigest(),
        "writer_metrics_sha256": hashlib.sha256(json.dumps(
            writer_metrics, sort_keys=True, separators=(",", ":"), allow_nan=False,
        ).encode("utf-8")).hexdigest(),
    }
    return {
        "schema_version": 1,
        "status": "frozen",
        "selected_writer": writer_metrics["selected_writer"],
        "prompt_hashes": {
            "writer_version": WRITER_PROMPT_VERSION,
            "writer_template_sha256": writer_prompt.template_sha256,
            "refiner_version": SPARSE_LOCAL_VERSION,
            "refiner_template_sha256": SPARSE_LOCAL_TEMPLATE_SHA256,
            "embedding_template_version": EMBEDDING_TEMPLATE_VERSION,
            "embedding_template_sha256": EMBEDDING_TEMPLATE_SHA256,
        },
        "configs": configs,
        "sources": sources,
        "post_freeze_policy": (
            "Only explicit bug fixes are allowed; rerun every affected method after a fix."
        ),
    }
