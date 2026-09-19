# StreamTimeLens

The opt-in JQ-01 baseline-preserving joint quality ranker and its staged
development commands are documented in [docs/jq01.md](docs/jq01.md).

Reference implementation of post-hoc query streaming video temporal grounding.
The current primary route is a
query-blind semantic visual cache (`semantic_reservoir`) followed by post-hoc
frame retrieval and temporal localization. Structured writers remain available
for reproducibility and negative comparison, but are no longer the mainline.

The package deliberately keeps ingestion and query code separate: the query
command accepts a snapshot only and the `SnapshotReader` rejects every path
outside that snapshot directory.

Quick smoke run (uses a synthetic JSON frame stream, so no model or video is
needed):

```bash
CODE_REVISION=$(git -C .. rev-parse HEAD)
PYTHONPATH=src python scripts/build_snapshots.py --video-id demo \
  --frames tests/fixtures/frames.jsonl --duration 12 --arrival-ratios .5,1 \
  --output ../results/streamtimelens-smoke/snapshots --budget 1m
PYTHONPATH=src python scripts/answer_snapshots.py \
  --snapshot "../results/${CODE_REVISION}/streamtimelens-smoke/snapshots/demo/rho_0.50" \
  --query 'person opens door'
```

`build_snapshots.py` can also decode an MP4 sequentially with `--video`; it
requires OpenCV.  The default writer is deterministic `unknown_event` cards,
which makes protocol smokes independent of model weights. S5 also provides the
production `TimeLensEvidenceWriter`: it shares the frozen process-wide TimeLens
service, uses a separate query-independent structured prompt, validates/repairs
JSON locally without a model retry, embeds cards, and allocates boundary frames.

Prediction and metric runs must use `streamtimelens.provenance.prepare_result_directory`,
which delegates to the workspace's versioned result/provenance implementation.

Developer checks use Ruff 0.1.15. On hosts where `files.pythonhosted.org` is
slow, install the same wheel from the verified Huawei Cloud mirror:

```bash
python3 -m pip install --no-cache-dir \
  --index-url https://repo.huaweicloud.com/repository/pypi/simple \
  'ruff==0.1.15'
ruff check streamtimelens/src streamtimelens/tests streamtimelens/scripts
```

S2 adds a process-wide frozen TimeLens service, sparse global-timestamp input,
versioned local-grounding prompts, and the Oracle three-condition runner. Run the
deterministic checkpoint smoke in the inference container with:

```bash
PYTHONPATH=streamtimelens/src:third_party/TimeLens \
python streamtimelens/scripts/s2_sparse_smoke.py \
  --model models/TimeLens-7B --samples artifacts/p1_charades_20.json \
  --video-root third_party/TimeLens/data/TimeLens-Bench/videos/charades \
  --output results/s2_sparse_smoke/sample0
```

`run_oracle_refiner.py` freezes the `GT±{2,5,10}s`, 16/32-frame,
uniform/boundary-heavy matrix. Pass `--model` and `--video-root` for its built-in
Decord/TimeLens three-condition runner; `--runner module:factory` supports another
frame source, and `--predictions` aggregates an existing JSONL without re-running
the model. TimeLens-Bench test annotations must not be used to execute the Oracle
development gate.

S3 adds two query-blind raw-frame baselines over the same snapshot protocol:

- `uniform_raw` keeps deterministic temporal slots when duration metadata is
  known (and a seeded online reservoir when it is not).
- `semantic_reservoir` combines temporal coverage anchors with CLIP novelty.

Both use frozen CLIP ViT-B/32 at 0.5 FPS, content-addressed 224-short-edge JPEG
storage, fp16 or int8 persisted embeddings, the same frame-candidate retrieval,
and the S2 sparse local TimeLens refiner. Ingest never receives query text and
query execution never receives the original video path. Example:

```bash
PYTHONPATH=streamtimelens/src:src python streamtimelens/scripts/build_snapshots.py \
  --video video.mp4 --method uniform_raw --clip-model /models/clip-vit-base-patch32 \
  --budget 1m --output results/s3_uniform

PYTHONPATH=streamtimelens/src:src:third_party/TimeLens \
python streamtimelens/scripts/answer_snapshots.py \
  --snapshot results/<git-sha>/s3_uniform/<video-id>/rho_0.50 \
  --query 'person opens a door' --retrieval frame \
  --clip-model /models/clip-vit-base-patch32 --timelens-model models/TimeLens-7B
```

The `--pixel-only` Uniform option is an explicit storage ablation and cannot run
CLIP frame-to-query retrieval. It is never enabled for the primary baseline.
`--retrieval auto` routes such snapshots to the dependency-free lexical path;
an explicit `--retrieval frame` request is rejected before loading CLIP.

Historical S5 structured-writer snapshot example (not the current mainline):

```bash
PYTHONPATH=streamtimelens/src:src:third_party/TimeLens \
python streamtimelens/scripts/build_snapshots.py \
  --video video.mp4 --method full --writer timelens \
  --writer-model models/TimeLens-7B --writer-revision frozen-local \
  --card-embedding minilm --text-embedder /models/all-MiniLM-L6-v2 \
  --budget 1m --output results/s5_structured
```

Every writer call records raw output, parse class, token/resource statistics and
boundary admission in the external ingest trace. Query-visible cards contain
prompt/template/model/embedding provenance and explicit raw-ref status. The
fixed 100-chunk feasibility comparison consumes paired raw outputs and writes a
versioned report with:

```bash
PYTHONPATH=streamtimelens/src:src \
python streamtimelens/scripts/run_writer_feasibility.py \
  --records artifacts/timelens100k_dev_writer_outputs.jsonl \
  --expected-chunks 100 --output results/s5_writer_feasibility
```

S7–S9 complete the hybrid card/CLIP retrieval path, hierarchy-aware MMR,
neighbor expansion, budgeted local refinement, confidence/NOT_FOUND handling,
same-budget VST/OASIS/Evidence-fixed baselines, official VTG metrics, diagnostics,
resource accounting, paired video bootstrap, and resumable matrix execution. A
card snapshot is queried only with its matching resolved configuration:

```bash
PYTHONPATH=streamtimelens/src:src:third_party/TimeLens \
python streamtimelens/scripts/answer_snapshots.py \
  --snapshot results/<git-sha>/<run>/snapshots/<video-id>/rho_0.75 \
  --query-id query-001 --query 'person opens the door' --retrieval hybrid \
  --resolved-config results/<git-sha>/<run>/config.resolved.yaml \
  --timelens-model models/TimeLens-7B

PYTHONPATH=streamtimelens/src:src \
python streamtimelens/scripts/run_matrix.py \
  --matrix streamtimelens/configs/matrices/dev_small_grid.template.yaml \
  --output results/dev_small_grid
```

S10's engineering and reporting tools are runnable independently. The smoke is
model-free and must not be used for model or threshold selection. The golden
regression freezes one video, four arrival ratios and three queries, including
snapshot hashes, retrieved IDs, parsed spans and resource-field schemas:

```bash
PYTHONPATH=streamtimelens/src:src \
python streamtimelens/scripts/run_engineering_smoke.py \
  --videos 20 --output results/s10_engineering_smoke

PYTHONPATH=streamtimelens/src:src \
python streamtimelens/scripts/run_golden_regression.py \
  --output results/s10_formal_golden
```

The visual route runs the Uniform/Semantic dev grid without a writer. It will
freeze to `configs/frozen_visual_v1.yaml` only after the visual protocol audit,
retrieval dev selection, and an explicit TimeLens-refiner enable/disable
decision. Formal tables and the long-video extension then use:

```bash
PYTHONPATH=streamtimelens/src:src \
python streamtimelens/scripts/audit_visual_cache.py \
  --matrix-root results/<git-sha>/visual_v1_smoke --output results/visual_v1_audit

PYTHONPATH=streamtimelens/src:src \
python streamtimelens/scripts/summarize_visual_dev.py \
  --matrix-root results/<git-sha>/visual_v2_grid \
  --queries artifacts/dev/dev_queries.jsonl --output results/visual_v2_summary

PYTHONPATH=streamtimelens/src:src \
python streamtimelens/scripts/select_visual_dev.py \
  --runs results/<git-sha>/visual_v2_summary/visual_dev_summaries.jsonl \
  --output results/visual_v2_selection
```

Frame-query prediction JSON records the ranked frame IDs, timestamps and CLIP
scores, all candidate envelopes, the selected readout, and its final selection
or fallback reason. `coarse_visual` and `timelens_local` are explicit readouts;
the latter requires a model and always falls back to the coarse span on invalid
output.

Hybrid V3 is an experimental readout that reuses an immutable Frozen Visual V2
snapshot. CLIP retrieves a diverse set of at most six temporal clusters, using
an explicit merge-gap tolerance so nominal 0.5 FPS samples are not split by
frame-rate rounding. The boundary-anchored allocator reserves the earliest and
latest visible snapshot frames within the 16-frame budget. One greedy TimeLens
call returns a versioned strict `REFINE Cxx` line, `KEEP Cxx`, or
`NOT_FOUND`; `KEEP` preserves the
selected candidate when sparse evidence cannot support tighter boundaries.
Invalid output, exceptions, and (by default) model `NOT_FOUND` fall back to the
paired CLIP coarse prediction without changing the snapshot. The evaluation
reports coarse, selected/fallback, refined/fallback, and candidate-oracle stages.
The frozen V2 configuration is not modified. A reverse-label configuration is
provided for the fixed candidate-order diagnostic without changing frame input.

The adaptive V3 prompt and acceptance gate use only the number of query-visible
input frames: with at most seven frames the selected candidate is preserved;
with denser evidence the model is instructed to refine inside that candidate.
This protects sparse boundaries without keying policy to an arrival ratio.
Dense refinement emits evidence time without a candidate label; the parser maps
it to the highest-score containing candidate, removing the label-choice channel.
Refined boundaries replace that candidate only when their duration is at least
80% of the candidate duration; stronger shrinkage preserves the candidate.

Run model inference in the pinned `vlm-timelens:0.1` image. The
`hybrid-v3` optional dependency group records its Python-side versions;
FlashAttention 2.7.4.post1 must be installed with `--no-build-isolation`, as in
the TimeLens model card.

```bash
PYTHONPATH=streamtimelens/src:src:third_party/TimeLens \
python streamtimelens/scripts/answer_hybrid_v3.py \
  --snapshot results/<git-sha>/<frozen-run>/snapshots/<video-id>/rho_0.75 \
  --query-id query-001 --query 'person opens the door' \
  --config streamtimelens/configs/exploration/clip_timelens_hybrid_v3.yaml \
  --output results/hybrid_v3/smoke

PYTHONPATH=streamtimelens/src:src:third_party/TimeLens \
python streamtimelens/scripts/run_hybrid_v3_matrix.py \
  --snapshot-root results/<git-sha>/<frozen-run>/snapshots \
  --queries artifacts/dev/queries.input.jsonl \
  --config streamtimelens/configs/exploration/clip_timelens_hybrid_v3.yaml \
  --output results/hybrid_v3/independent-dev
```

The matrix command is resumable and shardable. It accepts a GT-free query
manifest and writes predictions, protocol audit, cost summary, full resolved
configuration, and provenance below `results/<git-sha>/hybrid_v3/...`. Run
`evaluate_hybrid_v3.py` separately with annotations to apply the past-only
filter and produce paired CLIP/Hybrid VTG metrics.

```bash
PYTHONPATH=streamtimelens/src:src \
python streamtimelens/scripts/render_formal_report.py \
  --summaries artifacts/formal_summaries.jsonl \
  --methods uniform_raw,semantic_reservoir \
  --output results/formal_report

PYTHONPATH=streamtimelens/src:src \
python streamtimelens/scripts/prepare_long_video_extension.py \
  --configs artifacts/frozen_long_configs.jsonl \
  --observations artifacts/activitynet_memory_aging.jsonl \
  --output results/activitynet_long
```

The executable frozen visual V5/V6 path keeps formal ground truth out of every
ingest and query subprocess. It first materializes GT-free manifests plus a
hash-locked evaluation-only arrival plan, then creates one exact matrix per
frozen byte budget:

```bash
PYTHONPATH=streamtimelens/src:src \
python streamtimelens/scripts/prepare_formal_visual.py \
  --dataset charades \
  --annotations third_party/TimeLens/data/TimeLens-Bench/charades-timelens.json \
  --video-root third_party/TimeLens/data/TimeLens-Bench/videos/charades \
  --frozen streamtimelens/configs/frozen_visual_v1.yaml \
  --output results/visual_v5_inputs_charades

PYTHONPATH=streamtimelens/src:src \
python streamtimelens/scripts/summarize_visual_formal.py \
  --matrix-root results/<git-sha>/visual_v5_charades_256k \
  --matrix-root results/<git-sha>/visual_v5_charades_1m \
  --annotations results/<git-sha>/visual_v5_inputs_charades/annotations.evaluation_only.jsonl \
  --arrival-plan results/<git-sha>/visual_v5_inputs_charades/arrival_plan.jsonl \
  --audit results/<git-sha>/visual_v5_charades_audit/visual_protocol_audit.json \
  --frozen streamtimelens/configs/frozen_visual_v1.yaml \
  --output results/visual_v5_charades_summary
```

After both V5 datasets finish, `select_visual_long_configs.py` chooses at most
one frozen winner per budget. Pass those IDs to `prepare_formal_visual.py
--config-ids`, run ActivityNet, and render length/aging/amortization tables with
`summarize_visual_long.py`. Offline TimeLens supports four stable
`--num-shards/--shard-index` shards, same-revision `--resume`, and explicit
cross-revision `--resume-from`. Cross-revision imports require matching model
content hash, query-manifest hash, shard count/index, and source commit; the
source predictions hash is frozen into provenance. It remains labeled as a
non-streaming upper bound.

The structured route still has no `frozen_v1.yaml`: A1 returned `fix` and A2
returned `stop_writer`. The independent visual V1/V2 run selected Uniform after
Semantic showed no paired-bootstrap advantage; candidate Recall@5 peaked at
0.5018, so V3 disabled TimeLens. `configs/frozen_visual_v1.yaml` therefore
freezes coarse Uniform Pareto configurations. TimeLens-Bench test was not read
during selection; S11 remains inactive. See `docs/frozen_visual_v1_decision.md`.

The real A1 runner supports stable query-level `--num-shards/--shard-index`
partitioning and `--resume`. A2 raw generations are produced with
`scripts/generate_writer_feasibility.py`; run one checkpoint per GPU, then pass
both generated JSONL files to `scripts/run_writer_feasibility.py --records`.
