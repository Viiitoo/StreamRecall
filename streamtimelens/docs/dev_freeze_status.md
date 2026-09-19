# Development freeze status

Status on 2026-08-30: **A0 complete; A1=`fix`; A2=`stop_writer`; the original
writer freeze is closed; visual V1/V2 are complete and `frozen_visual_v1.yaml`
is frozen with Uniform coarse readout and TimeLens disabled**.

The independent Charades-STA `derived-dev-v1` manifest is fixed at
`artifacts/dev/input_manifest.json`: 100 videos, 219 queries, 100 writer chunks,
and zero video/query-ID overlap with TimeLens-Bench Charades test. TimeLens-7B,
base Qwen2.5-VL-7B, CLIP ViT-B/32, and MiniLM are pinned to local snapshots with
recursive hashes in `artifacts/dev/model_hashes.json`. All four pass offline
structural/loading checks. TimeLens-Bench test remains explicitly excluded from
development selection.

The deterministic pre-formal golden regression is frozen in
`tests/fixtures/formal_golden_v1.json`; run `scripts/run_golden_regression.py`
before any formal matrix. This validates protocol/query contracts only and does
not waive the independent-development gates below.

The Oracle and writer gates have run on the fixed independent dev inputs. Their
results and route decision are recorded in `docs/s10_p0_gate_decision.md`.
`configs/matrices/dev_small_grid.template.yaml` is now the approved
Uniform/Semantic visual-cache V1/V2 matrix and may run on independent dev. The
old `scripts/freeze_dev_config.py` and `configs/frozen_v1.yaml` remain
prohibited. The visual run audited 400 method/budget/video runs and 1,600
snapshots. Candidate Recall@5 peaked at 0.5018 under 1 MiB, so boundary
refinement was not feasible; paired bootstrap found no Semantic advantage.
The distinct `configs/frozen_visual_v1.yaml` consequently contains only Uniform
Pareto configurations and records the refiner decision as `disable`.

S11 remains intentionally inactive: its documented precondition (the full method
beats the strongest same-budget baseline at a medium budget on independent dev)
has not been established.
