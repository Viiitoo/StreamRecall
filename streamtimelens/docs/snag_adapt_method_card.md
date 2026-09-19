# SnAG-adapt method card

## Identity and scope

The upstream reference is `fmu2/snag_release` at commit
`44dd90eea9a65b64f7088974eae352c1e26ef6e3`, accompanying the CVPR 2024
SnAG paper. Its original task is offline long-form video temporal grounding
from complete pre-extracted video and text features.

The project method is named **SnAG-adapt-pooled-B**. It is an architecture
adaptation for strict post-hoc-query streaming VTG, not an official SnAG
checkpoint reproduction. Official parity results, when available, are kept in
the external-reference track and never mixed with strict results.

## Protocol differences

Upstream evaluation can read the complete feature file after the query is
known and does not impose a serialized memory budget. Here the writer sees
only sequential video frames and is query/GT blind. At each late-query time it
freezes an immutable snapshot; answering may read only that snapshot, cannot
open the video or replay features, and is charged for every serialized
query-visible byte. Main metrics include only events ending by query arrival.

## Preserved mechanisms

- Query-independent visual encoding followed by late text fusion.
- A temporal transformer backbone, multi-level temporal features, point
  classification, and left/right boundary-offset prediction.
- Ranked temporal proposals and temporal NMS.

## Explicit adaptations

- Replaces upstream pre-extracted features with a one-pass, 2 fps normalized
  CLIP ViT-B/32 stream.
- Replaces the dense, regular full-video grid with a 1 MiB near/far snapshot:
  recent tokens are retained and older adjacent tokens are merged by
  count-weighted means.
- Adds absolute physical-time center, support width, aggregation count/level,
  and asymmetric uncertainty metadata for every retained token.
- Replaces the frozen official dense-grid head with a reader trained on the
  same frozen pooled-state semantics. Boundary offsets are normalized by the
  token support width and decoded in seconds.
- Removes all query-time access to source video, external feature stores, and
  mutable writer state. Adds immutable hashing, actual-byte accounting,
  no-future checks, and independent-query repeat checks.

## Implementation and frozen components

Implementation lives under `streamtimelens/baselines/snag_*.py`; the pinned
upstream loader is under `baselines/vendor/`. The 1 MiB development recipe is
`streamtimelens/configs/snag/pooled_1m_v1.yaml`. Formal runs must use a copied
strict configuration with `formal_frozen: true`, an absolute checkpoint path,
and its SHA-256. CLIP model ID, revision, local content hash, writer policy,
reader architecture, training split seed, and arrival ratios are recorded in
every result's resolved configuration and provenance.

The G2 input-semantics diagnostic uses one independent long-video cohort and
three explicitly adapted rows: a reader trained and evaluated on full-store
dense-grid snapshots, that exact byte-identical frozen reader evaluated on
pooled snapshots, and a physical-time reader trained and evaluated on those
same pooled snapshots. The first two rows isolate compression at fixed reader
weights; the third measures adaptation to physical-time pooling. None of these
CLIP-based rows is called an official SnAG head. Official-model identity is
established separately by the pinned TACoS G0 parity run.

## Comparison boundary

SnAG-adapt may be compared end-to-end with other systems in the strict system
track when the same dataset, arrivals, eligible cohort, byte budget, and metric
implementation are used. It cannot isolate memory-policy effects when the
backbone/reader differs; controlled writer comparisons need a shared reader.
Its results describe this project's adaptation only. They must not be called
official SnAG reproduction results, attributed to the upstream authors, or
used to claim superiority over published offline/query-known methods.
