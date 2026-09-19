# S5 writer feasibility status

Status on 2026-08-30: **completed with `decision=stop_writer`; no writer is
authorized for A3**.

The workspace contains pinned TimeLens-7B and base
Qwen2.5-VL-7B-Instruct checkpoints plus 100 fixed, independent Charades-STA
writer chunks. Their paths, revisions, recursive hashes, and data leakage report
are recorded in `artifacts/dev/input_manifest.json`. F5.7 completed on all 100
paired chunks; the final TimeLens-Bench test split remains prohibited for
writer selection. The decision is in `docs/s10_p0_gate_decision.md` and
`results/c744661d6ae6efbef8e8ee7acd3d8d9587b632b8/s10_writer_feasibility/`.

The executable report path is implemented by
`scripts/run_writer_feasibility.py`. It requires paired rows for identical
chunks with `model`, `chunk_id`, `segment`, `sampled_timestamps`, `raw_output`,
`gt_spans`, and `gpu_s`; it rejects duplicates, mismatched chunk sets, or a
count other than 100. It reports direct JSON validity, schema usability,
fallback rate, event count, GT coverage, endpoint error, and GPU seconds, then
writes `metrics.json`, `writer_feasibility_report.md`, resolved config,
provenance, and the Git revision below `results/<full-commit>/...`.

Generate paired outputs from both pinned checkpoints on
`artifacts/dev/writer_chunks.jsonl`. Run one process per GPU/checkpoint; both
commands are resumable and write below the current commit SHA:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=streamtimelens/src:src \
python streamtimelens/scripts/generate_writer_feasibility.py \
  --chunks artifacts/dev/writer_chunks.jsonl \
  --model models/TimeLens-7B --model-id timelens-7b \
  --model-revision 1740e2694669a4d03549454ed4af1cd740769f4f \
  --output results/s10_writer_timelens --resume

CUDA_VISIBLE_DEVICES=1 PYTHONPATH=streamtimelens/src:src \
python streamtimelens/scripts/generate_writer_feasibility.py \
  --chunks artifacts/dev/writer_chunks.jsonl \
  --model models/Qwen2.5-VL-7B-Instruct --model-id qwen2.5-vl-7b \
  --model-revision cc594898137f460bfe9f0759e9844b3ce807cfb5 \
  --output results/s10_writer_qwen --resume
```

Then aggregate the two generated JSONL files:

```bash
PYTHONPATH=streamtimelens/src:src \
python streamtimelens/scripts/run_writer_feasibility.py \
  --records results/<SHA>/s10_writer_timelens/writer_outputs.jsonl \
            results/<SHA>/s10_writer_qwen/writer_outputs.jsonl \
  --expected-chunks 100 \
  --output results/s5_writer_feasibility
```

When `--semantic-oracle-coverage` is omitted, the aggregator deterministically
computes the query-time oracle upper bound from the same fixed chunks: for each
GT span it takes the maximum IoU over all spans whose two endpoints are retained
`sampled_timestamps`. The source and value are frozen in resolved config and
provenance; the optional CLI value exists only for a separately versioned,
equivalent reservoir-oracle measurement.
