# HEM-01 development contract

HEM-01 is the active research cycle after JQ-01 and LF-01 closed. The user
explicitly approved its start on 2026-09-11. It is a query-independent change
to ingestion and snapshot representation, so no effect run is allowed before
the new writer/reader contract passes effect-free T0/J1.

The implementation is an independent StreamTimeLens rewrite. OnVTG is used
only as behavioral evidence for separate temporal scales; no source is copied,
and its ephemeral GPU FIFO is not treated as a persistent snapshot.

## Event state v1

Each event stores `start_s`, `end_s`, `support_count`, first/last frame index,
scale, stable event ID, and an fp16 normalized visual embedding. The public
`observe(timestamp_s, frame_index, embedding)` API contains no query, query ID,
GT, annotation, source, or video path.

The only frozen v1 structure is:

- three levels with capacities `64/32/16`;
- adjacent fine tokens merge while cosine is at least `0.85` and total fine
  duration is at most 8 seconds; otherwise a query-blind visual/duration
  boundary closes the event;
- overflow promotes the oldest adjacent pair to the next scale; overflow at
  the last scale compacts its oldest adjacent pair;
- events across levels form one chronological, non-overlapping partition rather
  than duplicated frame plus parent state; support is conserved exactly;
- recent history remains at scale 0 while older history becomes progressively
  coarser. Input timestamps and frame indices must be strictly increasing.

The event list is serialized as `event_memory.json` through `SnapshotWriter`.
It is included in the manifest allow-list, SHA checks and actual filesystem byte
budget. No Python/GPU cache, ingest trace, pixel, original-video path, or
unlisted sidecar is query-visible. `SnapshotReader.read_event_memory()` and the
HEM validator reject malformed, future, overlapping, reordered, or identity-
mismatched events.

Adding the optional event field must not change Frozen V2's default snapshot
bytes: when event memory is disabled, the manifest omits the new key entirely.

## T0/J1

The effect-free gate uses 512 synthetic 512-dimensional visual tokens. It must
prove deterministic IDs/bytes, capacity bounds, support conservation, old-
coarse/recent-fine layout, query-free serialized state, 1 MiB byte compliance,
read-only snapshot access, and unchanged default snapshot serialization. It
also runs the snapshot/protocol regressions. No development annotation or
D-lock identity is read.

```bash
python3 scripts/run_hem01_mechanism_gate.py
```

Artifacts are written to
`work/results/<full-git-sha>/hem01/T0_J1/` with resolved config, provenance,
event schema/trace, snapshot audit and recursive SHA manifest.

After T0/J1, HEM-01 still needs a separately committed budgeted-ingestion and
query-reader effect contract before generating new development snapshots.
D-lock remains unprepared and unconsumed.

## V2-original development r1 contract

HEM r1 uses the same frozen CLIP ViT-B/32 and 0.5 FPS visual sampling as Frozen
V2. Each MP4 is decoded once in forward order at a 2 FPS packet cadence; decoder
seek count must remain zero. One shared frozen model may be reused across videos,
but its sampling clock is reset at each video boundary. Arrival snapshots remain
at rho `0.25/0.50/0.75/1.00`, with 1 MiB as the first development budget.

The fixed query reader scores every persisted event embedding against the CLIP
text embedding, takes Top-8 with deterministic ties, groups hits separated by at
most four seconds, and includes one stored adjacent event on each side. It uses
zero candidate/final margin and selects the highest-scoring group. It has no
trainable parameter, GT input, original-video access, threshold or safety gate.

r1 reuses exactly the 552 stream-visible consumed development observations for
paired comparison with Frozen V2. The effect output reports final mIoU/R@0.7,
Top-5 candidate oracle/recall, video bootstrap, rho and good-baseline slices,
event-level coverage/support, bytes and query latency. Freezing requires:

- video-equal final mIoU 95% CI lower bound above zero and standard R@0.7 no
  regression;
- every rho slice at least `-0.02` and good-baseline delta at least `-0.01`;
- Top-5 HEM candidate recall@0.5 and candidate-oracle mIoU no worse than the
  corresponding Frozen V2 final-span recall/mIoU;
- 100% support accounting, zero seek, immutable snapshots, complete paired
  coverage/provenance, and readout p95 at most 50 ms excluding shared text encode.

This is development evidence, not an independent claim. Failure cannot be
repaired by tuning Top-K, merge gap, margins or boundary cosine in place; a new
structural revision requires a new contract and clean commit.

## V2-original development r1 result

The complete paired run was executed from clean commit
`8e09f53de0a5f5fa5e7ef3316a61a3b59fdd2a90`. Its immutable output is:

`work/results/8e09f53de0a5f5fa5e7ef3316a61a3b59fdd2a90/hem01/v2_original_dev/v2-original-r1/`

All protocol and operational gates passed: 100 videos were decoded once with
zero seeks; all 400 snapshots conserved support and remained byte-identical
across readout; maximum snapshot state was 20,273 bytes under the 1 MiB budget;
readout p95 was 1.68 ms; and all 2,410 entries in `artifact_manifest.json` were
independently rehashed successfully. The full regression suite passed 252 tests.

The effect gates failed. Standard mIoU changed from 0.392039 to 0.377429
(`-0.014610`), standard R@0.7 changed by `-0.001812`, and video-equal paired
mIoU changed by `-0.024610` with 95% CI `[-0.043679, -0.006797]`.
Good-baseline delta was `-0.044845`. The rho deltas were `+0.073002`,
`+0.021990`, `-0.025689`, and `-0.056472` at 0.25, 0.50, 0.75, and 1.00.

The benchmark also failed to exercise the proposed hierarchy: no level-1 or
level-2 event appeared in any real snapshot, the mean query-visible event count
was 3.85, and 549 of 552 observations produced only one candidate. Thus r1
primarily measured flat event segmentation and an over-wide single-candidate
readout, not long-history hierarchical retention. This is a structural
benchmark/method mismatch rather than a byte, decoder, latency, mutation, or
provenance failure.

The recorded decision is `hem01_r1_not_frozen`. HEM-01 is closed for the current
V2-original short-video development objective. The failed run is retained, and
its capacities, Top-K, merge gap, neighbor expansion, margin, and boundary
cosine will not be tuned post hoc. D-lock remains unprepared and unconsumed.
