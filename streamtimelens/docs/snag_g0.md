# SnAG G0 upstream parity

The G0 implementation is pinned to upstream revision
`44dd90eea9a65b64f7088974eae352c1e26ef6e3`. The loader rejects any other
checkout revision, loads the real `PtTransformer` with strict `model_ema`
state-dict matching, and records SHA-256 digests for the option and checkpoint.

Run the checkpoint-backed snapshot round-trip layer with:

```bash
PYTHONPATH=streamtimelens/src:src python3 streamtimelens/scripts/run_snag_g0_parity.py \
  --upstream-root ../ref/snag_release \
  --opt /path/to/experiment/opt.yaml \
  --checkpoint /path/to/experiment/models/last.pth \
  --video-feature /path/to/video.npy \
  --text-feature /path/to/query.npy \
  --expected-official-metrics /path/to/published_metrics.json \
  --actual-official-metrics /path/to/evaluator_metrics.json \
  --video-id VIDEO_ID --duration DURATION_SECONDS --fps SOURCE_FPS
```

This stage verifies the real model's FPN shapes, masks, logits, offsets,
physical-time decode, compiled upstream Gaussian soft-NMS, voting, and Top-K
predictions before and after the features pass sequentially through a float32
full-store snapshot. Its result remains `classification=not-a-result`: G0
passes only after the same pinned checkpoint is evaluated on the complete
official dataset and its protocol metrics match the published reference
within the frozen tolerance. Synthetic checkpoints cannot satisfy that gate.
