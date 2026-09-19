"""Frozen three-row semantic diagnostic for the SnAG pooled adaptation."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Mapping, Sequence


REQUIRED_G2_ROWS = (
    "full-store-dense-grid-head",
    "pooled-state-frozen-dense-grid-head",
    "pooled-state-trained-physical-time-head",
)


def _json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def assemble_snag_g2_rows(
    full_grid_run: Path | str,
    pooled_frozen_run: Path | str,
    pooled_trained_run: Path | str,
) -> list[dict[str, object]]:
    """Assemble G2 rows only from mutually compatible completed run artifacts.

    The full-grid checkpoint must be reused byte-for-byte by the frozen pooled
    run.  Both pooled runs must consume the same snapshots and all three runs
    must use the same query cohort.  This makes the three-row diagnostic an
    actual intervention on state/head semantics instead of a hand-authored
    collection of unrelated finite metrics.
    """
    roots = tuple(Path(value).expanduser().resolve(strict=True) for value in (
        full_grid_run, pooled_frozen_run, pooled_trained_run,
    ))
    records = []
    for root in roots:
        metrics = _json(root / "metrics.json")
        protocol = _json(root / "protocol_audit.json")
        provenance = _json(root / "provenance.json")
        checkpoint = root / "snag_physical_reader.pth"
        configuration = provenance.get("configuration")
        if not isinstance(configuration, Mapping):
            raise ValueError(f"missing resolved configuration: {root}")
        overall = metrics.get("overall")
        if not isinstance(overall, Mapping):
            raise ValueError(f"missing overall metrics: {root}")
        records.append({
            "root": root,
            "metrics": dict(overall),
            "protocol_passed": bool(protocol.get("passed")),
            "provenance": provenance,
            "configuration": configuration,
            "checkpoint_sha256": _sha256(checkpoint),
            "snapshot_manifest_sha256": str(provenance.get("snapshot_index_sha256", "")),
            "query_manifest_sha256": str(provenance.get("query_manifest_sha256", "")),
        })
    full, frozen, trained = records
    if full["configuration"].get("method") != "snag-adapt-input-full":
        raise ValueError("full-grid G2 row is not a full-store run")
    if any(row["configuration"].get("method") != "snag-adapt-pooled-B" for row in (frozen, trained)):
        raise ValueError("pooled G2 rows are not pooled-state runs")
    if len({row["query_manifest_sha256"] for row in records}) != 1:
        raise ValueError("G2 runs do not use the same query cohort")
    if frozen["snapshot_manifest_sha256"] != trained["snapshot_manifest_sha256"]:
        raise ValueError("G2 pooled runs do not use the same snapshot manifest")
    if full["checkpoint_sha256"] != frozen["checkpoint_sha256"]:
        raise ValueError("G2 frozen pooled run did not reuse the dense-grid checkpoint")
    if not str(frozen["provenance"].get("reused_checkpoint", "")):
        raise ValueError("G2 frozen pooled run lacks checkpoint-reuse provenance")
    names = REQUIRED_G2_ROWS
    return [{
        "name": name,
        "completed": True,
        "protocol_passed": row["protocol_passed"],
        "metrics": row["metrics"],
        "checkpoint_sha256": row["checkpoint_sha256"],
        "snapshot_manifest_sha256": row["snapshot_manifest_sha256"],
        "query_manifest_sha256": row["query_manifest_sha256"],
        "source_run": str(row["root"]),
    } for name, row in zip(names, records)]


def audit_snag_g2_diagnostics(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    """Require all diagnostic identities and finite metrics without ranking them."""
    indexed = {str(row.get("name")): row for row in rows}
    if len(indexed) != len(rows) or set(indexed) != set(REQUIRED_G2_ROWS):
        raise ValueError("SnAG G2 diagnostic must contain exactly the three frozen rows")
    output = []
    for name in REQUIRED_G2_ROWS:
        row = indexed[name]
        metrics = row.get("metrics")
        finite = isinstance(metrics, Mapping) and all(
            key in metrics and math.isfinite(float(metrics[key]))
            for key in ("miou", "r1_iou_0.3", "r5_iou_0.3")
        )
        checks = {
            "completed": bool(row.get("completed")),
            "protocol_passed": bool(row.get("protocol_passed")),
            "finite_metrics": finite,
            "checkpoint_sha256_recorded": len(str(row.get("checkpoint_sha256", ""))) == 64,
            "snapshot_manifest_sha256_recorded": len(
                str(row.get("snapshot_manifest_sha256", ""))
            ) == 64,
        }
        output.append({"name": name, "checks": checks, "passed": all(checks.values())})
    return {
        "schema_version": 1,
        "gate": "SnAG-G2-three-row-input-semantics",
        "passed": all(row["passed"] for row in output),
        "rows": output,
        "comparison_rule": "diagnostic-only; no row is required to outperform another",
    }
