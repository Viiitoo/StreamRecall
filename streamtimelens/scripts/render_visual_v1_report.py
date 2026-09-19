#!/usr/bin/env python3
"""Validate and render the complete frozen visual v1 formal report."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT / "src"))
sys.path.insert(0, str(PACKAGE_ROOT.parent / "src"))

from baas.provenance import versioned_result_path, write_provenance


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _percent(value: float) -> str:
    return f"{float(value):.2f}"


def _v5_table(rows: list[dict], selected_ids: set[str]) -> list[str]:
    lines = [
        "| Dataset | Budget | rho | Cohort | mIoU | R@1@0.5 | R@1@0.7 | "
        "Abs. mIoU 95% CI | Paired delta 95% CI |",
        "| --- | ---: | ---: | --- | ---: | ---: | ---: | --- | --- |",
    ]
    for row in sorted(
        (item for item in rows if item["config_id"] in selected_ids),
        key=lambda item: (
            item["dataset"], item["budget_bytes"], item["cohort"], item["rho_q"],
        ),
    ):
        absolute = row["bootstrap_miou"]
        paired = row["paired_bootstrap_vs_budget_reference"]
        lines.append(
            f"| {row['dataset']} | {int(row['budget_bytes']) // 1024} KiB | "
            f"{row['rho_q']:.2f} | {row['cohort']} | {_percent(row['miou'])} | "
            f"{_percent(row['recall_at_05'])} | {_percent(row['recall_at_07'])} | "
            f"[{100 * absolute['ci_low']:.2f}, {100 * absolute['ci_high']:.2f}] | "
            f"[{100 * paired['ci_low']:.2f}, {100 * paired['ci_high']:.2f}] |"
        )
    return lines


def _resource_table(rows: list[dict], selected_ids: set[str]) -> list[str]:
    lines = [
        "| Dataset | Budget | rho | Snapshot bytes | Ingest GPU s | Query GPU s | "
        "Mean latency s | Throughput x realtime |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    natural = [
        row for row in rows
        if row["config_id"] in selected_ids and row["cohort"] == "natural"
    ]
    for row in sorted(natural, key=lambda item: (
        item["dataset"], item["budget_bytes"], item["rho_q"],
    )):
        lines.append(
            f"| {row['dataset']} | {int(row['budget_bytes']) // 1024} KiB | "
            f"{row['rho_q']:.2f} | {row['snapshot_bytes']:.0f} | "
            f"{row['ingest_gpu_s']:.2f} | {row['query_gpu_s']:.2f} | "
            f"{row['query_latency_s']:.4f} | {row['realtime_throughput']:.2f} |"
        )
    return lines


def _lag_table(rows: list[dict], selected_ids: set[str]) -> list[str]:
    lines = [
        "| Dataset | Budget | Cohort | Lag bin | n | mIoU | R@1@0.5 |",
        "| --- | ---: | --- | --- | ---: | ---: | ---: |",
    ]
    for row in sorted(
        (item for item in rows if item["config_id"] in selected_ids),
        key=lambda item: (
            item["dataset"], item["budget_bytes"], item["cohort"], item["lag_bin"],
        ),
    ):
        lines.append(
            f"| {row['dataset']} | {int(row['budget_bytes']) // 1024} KiB | "
            f"{row['cohort']} | {row['lag_bin']} | {row['count']} | "
            f"{_percent(row['miou'])} | {_percent(row['recall_at_05'])} |"
        )
    return lines


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--formal-summaries", type=Path, required=True)
    parser.add_argument("--lag-summaries", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--long-summary", type=Path, required=True)
    parser.add_argument("--offline-summary", type=Path, required=True)
    parser.add_argument("--negative-selection", type=Path, required=True)
    parser.add_argument("--refiner-gate", type=Path, required=True)
    parser.add_argument("--audit", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = [
        args.formal_summaries, args.lag_summaries, args.selection, args.long_summary,
        args.offline_summary, args.negative_selection, args.refiner_gate, *args.audit,
    ]
    audits = [json.loads(path.read_text(encoding="utf-8")) for path in args.audit]
    if not all(row.get("passed") for row in audits):
        raise ValueError("final visual report requires every protocol audit to pass")
    formal = _jsonl(args.formal_summaries)
    lag = _jsonl(args.lag_summaries)
    if len(formal) != 154 or {float(row["rho_q"]) for row in formal} != {.25, .5, .75, 1.0}:
        raise ValueError("final visual report requires the canonical 154-row V5 summary")
    if any(
        row["bootstrap_miou"].get("resamples") != 10000
        or row["paired_bootstrap_vs_budget_reference"].get("resamples") != 10000
        for row in formal
    ):
        raise ValueError("final visual report requires 10,000-resample bootstrap evidence")
    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    selected = selection.get("selected", [])
    selected_ids = {str(row["config_id"]) for row in selected}
    if len(selected_ids) != 2 or {int(row["budget_bytes"]) for row in selected} != {262144, 1048576}:
        raise ValueError("final visual report requires one selected V6 config per budget")
    long_report = json.loads(args.long_summary.read_text(encoding="utf-8"))
    if set(long_report.get("configs", {})) != selected_ids:
        raise ValueError("V6 long-video report does not match the frozen V5 selection")
    offline = json.loads(args.offline_summary.read_text(encoding="utf-8"))
    if set(offline) != {"charades", "qvhighlights"} or any(
        row.get("protocol") != "offline_upper_bound_not_streaming" for row in offline.values()
    ):
        raise ValueError("offline upper bound is incomplete or mislabeled")
    negative = json.loads(args.negative_selection.read_text(encoding="utf-8"))
    refiner = json.loads(args.refiner_gate.read_text(encoding="utf-8"))
    if negative.get("semantic_better_budget_points") or refiner.get("decision") != "disable":
        raise ValueError("frozen v1 negative decisions do not match their source records")

    lines = [
        "# StreamTimeLens frozen visual v1 final report", "",
        "Protocol status: all supplied visual audits passed; Offline TimeLens is reported "
        "only as `offline_upper_bound_not_streaming`.", "",
        "## Frozen V5 accuracy", "",
        *_v5_table(formal, selected_ids), "",
        "The machine-readable V5 artifact contains all 154 rows (11 configs × 2 datasets × "
        "four natural plus three fixed cohorts).", "",
        "## Lag bins", "", *_lag_table(lag, selected_ids), "",
        "## Resources", "", *_resource_table(formal, selected_ids), "",
        "## Offline TimeLens upper bound", "",
        "| Dataset | mIoU | R@1@0.5 | R@1@0.7 | GPU s | Mean query latency s |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for dataset, row in sorted(offline.items()):
        lines.append(
            f"| {dataset} | {_percent(row['miou'])} | {_percent(row['recall_at_05'])} | "
            f"{_percent(row['recall_at_07'])} | {row['gpu_s']:.2f} | "
            f"{row['query_latency_s']:.4f} |"
        )
    lines.extend(["", "## ActivityNet V6 long-video diagnostics", ""])
    for config_id, report in sorted(long_report["configs"].items()):
        lines.extend([f"### {config_id}", "", "| Slice | n | mIoU | Candidate R@5 | "
                      "Earliest anchor | Amortized GPU s/query | Snapshot bytes |",
                      "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"])
        slices = {**report["length_bins"], **{
            f"fixed C0.25 @ rho={rho}": value
            for rho, value in report["fixed_0.25_by_rho"].items()
        }, "natural": report["natural"]}
        for label, row in slices.items():
            lines.append(
                f"| {label} | {row['count']} | {100 * row['miou']:.2f} | "
                f"{100 * row['candidate_recall_at_5']:.2f} | "
                f"{100 * row['earliest_anchor_retention_rate']:.2f} | "
                f"{row['amortized_gpu_s_per_query']:.4f} | {row['snapshot_bytes']:.0f} |"
            )
        lines.append("")
    lines.extend([
        "## Frozen negative results", "",
        f"- Semantic cache had {len(negative.get('semantic_better_budget_points', []))} "
        "budget points with stable benefit over Uniform; v1 therefore uses Uniform raw cache.",
        f"- The v1 local TimeLens refiner decision is `{refiner['decision']}` because "
        f"`{refiner.get('reason', 'the frozen gate failed')}`.",
        "- The structured writer/event-forest route remains a negative auxiliary result and is "
        "not mixed into the protocol-compliant visual table.", "",
        "## Provenance", "",
    ])
    for path in paths:
        lines.append(f"- `{path.resolve()}` — SHA-256 `{_sha256(path)}`")
    output_root = versioned_result_path(args.output)
    output_root.mkdir(parents=True, exist_ok=True)
    hashes = {str(path.resolve()): _sha256(path) for path in paths}
    write_provenance(
        output_root,
        configuration={
            "inputs": hashes, "selected_config_ids": sorted(selected_ids),
            "v5_summary_rows": len(formal), "bootstrap_resamples": 10000,
            "offline_protocol": "offline_upper_bound_not_streaming",
        },
        config_filename="config.resolved.json",
    )
    (output_root / "report.json").write_text(json.dumps({
        "selected": selected, "formal_summaries": formal, "lag_summaries": lag,
        "offline": offline, "activitynet": long_report,
        "negative_results": {"selection": negative, "refiner_gate": refiner},
        "input_sha256": hashes,
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_root / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"output_root": str(output_root), "formal_rows": len(formal)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
