"""Configuration guard for the periodic-trigger Evidence-fixed ablation."""

from __future__ import annotations

from typing import Any, Mapping


def validate_evidence_fixed_pair(
    full: Mapping[str, Any], fixed: Mapping[str, Any],
) -> None:
    """Require the named method and trigger to be the only config differences."""
    if full.get("method") != "streamtimelens" or fixed.get("method") != "evidence-fixed":
        raise ValueError("evidence-fixed configs have unexpected method identifiers")
    if full.get("trigger") == fixed.get("trigger") or fixed.get("trigger") != "periodic":
        raise ValueError("evidence-fixed must use the periodic trigger")
    comparable_full = {key: value for key, value in full.items() if key not in {"method", "trigger"}}
    comparable_fixed = {key: value for key, value in fixed.items() if key not in {"method", "trigger"}}
    if comparable_full != comparable_fixed:
        keys = sorted(set(comparable_full) | set(comparable_fixed))
        changed = [key for key in keys if comparable_full.get(key) != comparable_fixed.get(key)]
        raise ValueError("evidence-fixed changes non-trigger fields: " + ", ".join(changed))
