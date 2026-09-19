# S10 P0 gate decision

Status on 2026-08-30: **A1 fix; A2 stop_writer; structured-writer stages not activated; visual-cache V0 activated**.

## Frozen inputs

- independent dev: Charades-STA `derived-dev-v1`, 100 videos and 219 queries;
- writer set: 100 fixed paired chunks;
- input manifest SHA-256:
  `361ec3fed62c08825f7f5e6980d7e93b6b7d3d860df58a90047d30e410864f12`;
- raw inference commit: `5436608fb53e405a7d72acc05a2da45e1ae8741b`;
- Oracle final aggregation commit: `8a6b0dc4c4935ae4656101d656ef2a558bdf0ab0`.

No QVHighlights-TimeLens or ActivityNet-TimeLens formal-test asset was
downloaded or read.

## A1 Oracle local refiner

The complete matrix contains 2,628 examples and 7,884 unique predictions (all
three modes, with no missing or duplicate pair). The final result is
`results/8a6b0dc4c4935ae4656101d656ef2a558bdf0ab0/s10_oracle_dev/`.

Decision: **fix**.

- dense mIoU: 0.4255;
- sparse mIoU: 0.3504;
- sparse-vs-dense drop: 0.0751, above the 0.05 gate;
- strict signed-bias tolerance: 0.1613 seconds;
- sparse signed start/end bias: 1.2177 / -0.9323 seconds.

The condition breakdown rules out a simple missing-output failure. Uniform
sampling satisfies the mIoU-drop criterion in all six settings, while
boundary-heavy sampling loses 0.0720--0.2254 mIoU. Only the 10-second,
16-frame uniform condition passes both aggregate criteria. Shorter margins
retain systematic endpoint bias in both dense and sparse modes, so a future
visual-index pivot must revalidate prompt/timestamp calibration and should not
claim the current sparse adapter passed P0.

## A2 writer feasibility

The paired 100-chunk comparison is at
`results/c744661d6ae6efbef8e8ee7acd3d8d9587b632b8/s10_writer_feasibility/`.
The query-time semantic-reservoir oracle upper bound is 0.8991 mean IoU.

Decision: **stop_writer**.

| checkpoint | schema usable | fallback | GT coverage IoU | GPU seconds |
| --- | ---: | ---: | ---: | ---: |
| Qwen2.5-VL-7B | 0.000 | 1.000 | 0.0000 | 811.455 |
| TimeLens-7B | 0.120 | 0.880 | 0.0248 | 805.953 |

The preserved raw outputs show that 99/100 Qwen and 85/100 TimeLens calls hit
the fixed 256-token generation cap. Qwen also wrapped all 100 answers in
Markdown fences. These are measured feasibility outcomes, not manually repaired
records. Both writers trail the visual oracle by far more than the 0.05 stop
threshold.

## Consequence

The original structured-writer protocol required both A1 and A2 to return `go`
before its A3. That condition is false, so the original A3/A4 remain closed and
`configs/frozen_v1.yaml` must not be generated.

The 2026-08-30 research review separately approved the visual-cache route:

- stop the structured heavy-writer mainline and preserve these results as a
  negative comparison;
- promote the already implemented query-blind `semantic_reservoir` and
  `uniform_raw` methods to the new V1/V2 protocol and retrieval gates;
- treat A1=`fix` as a restriction on the TimeLens local-refinement claim, not as
  a blocker for coarse visual temporal localization;
- freeze the new route only through `frozen_visual_v1.yaml`, whose prerequisites
  are visual protocol audit, visual dev selection, and an explicit refiner
  enable/disable decision;
- keep formal test assets unread until that new freeze is committed, and keep
  S11 inactive.

## Visual-route outcome

V1/V2 subsequently completed on the independent `derived-dev-v1` split. All
400 runs and 1,600 snapshots passed protocol audit. Candidate Recall@5 peaked
at 0.5018 at 1 MiB, below the 0.70 refiner-feasibility gate, and paired
bootstrap did not establish a Semantic-over-Uniform improvement. V3 therefore
records `disable` without loading TimeLens on retrieved candidates. The separate
`configs/frozen_visual_v1.yaml` freezes the coarse Uniform Pareto set; the old
writer freeze remains closed.
