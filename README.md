# StreamRecall

A streaming video agent that watches once, stores a byte-bounded visual memory, and answers later questions with a time interval and cited frames.

[Watch the demo](#demo) · [How it works](#method) · [Run locally](#quick-start) · [Evaluation](#evaluation) · [Citation](#citation)

## Demo

[![Play the 24-second StreamRecall demo. The clip plays, a question arrives, and the model returns a time interval with three cited frames.](docs/assets/streamrecall-demo.gif)][play]

**[Play the full-size video (MP4, 24 s)][play]** · [Download MP4](https://github.com/Viiitoo/StreamRecall/raw/refs/heads/main/docs/assets/streamrecall-demo.mp4) · [Recorded API output](docs/assets/meadow-run.json)

The demo follows a single Hybrid V3 run on a 24-second excerpt of *Big Buck Bunny*. Clip times start at zero. The recording plays the input at 2× speed.

| In the demo | What happens |
| --- | --- |
| 0:01.5–0:13.5 | The clip plays. The writer retains 12 frames in a 249,637-byte snapshot. |
| 0:13.5–0:17 | After playback ends, the query arrives. “When does the rabbit lean down to smell the white flowers?” |
| 0:17–0:24 | The model returns **8–24 s** and cites frames at **10, 16 and 22 s**. The predicted interval extends past the flower scene. |

The footage is a silent excerpt with playback and recall overlays. *Big Buck Bunny* © 2008 Blender Foundation / www.bigbuckbunny.org. [Film credits](https://peach.blender.org/about/) · [CC BY 3.0](https://creativecommons.org/licenses/by/3.0/)

[play]: https://cdn.jsdelivr.net/gh/Viiitoo/StreamRecall@main/docs/assets/streamrecall-demo.mp4

## Method

![A single pass over the stream produces an immutable snapshot. A later query searches the snapshot for a time interval, evidence and an audit.](docs/assets/recall-flow.svg)

StreamRecall separates ingestion from recall by construction. The writer stores
sparse visual evidence and metadata under a real byte budget. At query time,
CLIP retrieves diverse temporal candidates, a bounded allocator selects no more
than the allowed evidence frames, and TimeLens optionally refines the selected
interval. Invalid model output or insufficient evidence falls back safely—or
returns an explicit `NOT_FOUND`.

## Protocol

| Capability | Conventional video QA | StreamRecall strict mode |
|---|:---:|:---:|
| Question known during encoding | Often | **No** |
| Replay source video at query time | Often | **Forbidden** |
| Access future frames | Possible | **Forbidden** |
| Memory limit | Token/config estimate | **Serialized bytes** |
| Evidence authorization | Optional | **Answer-bound references** |
| Failure behavior | Free-form answer | **Explicit abstention** |
| Reproducibility | Run-dependent | **Hash + config + provenance** |

The API never returns internal snapshot paths. Evidence endpoints serve only
frame references cited by the persisted answer, while snapshot readers reject
path traversal, hash mismatches, undeclared files, and budget violations.

## Interactive workspace

The Next.js workspace shows memory timelines, temporal candidates, evidence cards and an agent trace. FastAPI streams turn events and persists sessions in SQLite. Strict uploads can discard the app-owned source once the snapshot is ready.

The default local backend simulates recall so the interface runs without a GPU. The recorded demo above uses the CLIP + TimeLens backend. [API and workspace guide](apps/README.md)

## Quick start

Requirements: Python 3.8+, Node.js 20.9+, and npm.

```bash
git clone https://github.com/Viiitoo/StreamRecall.git
cd StreamRecall
make setup
make demo
```

After both services report ready, open the local UI at `http://127.0.0.1:3000`.
The interactive API documentation is served locally at
`http://127.0.0.1:8000/docs`. These are local development addresses—not a
hosted demo. If StreamRecall is running on a remote machine, use SSH port
forwarding or your own deployment URL.

<details>
<summary><b>Run the real CLIP + TimeLens backend</b></summary>

Initialize the separately licensed TimeLens dependency, then point StreamRecall
at a directory of audited snapshots:

```bash
git submodule update --init third_party/TimeLens
export STREAMRECALL_BACKEND=hybrid-v3
export STREAMRECALL_SNAPSHOT_ROOT=/absolute/path/to/snapshots
export STREAMRECALL_CLIP_DEVICE=cuda:0
export STREAMRECALL_DEVICE_MAP=cuda:0
make demo
```

Models load lazily on the first query. Snapshot registration accepts only paths
relative to `STREAMRECALL_SNAPSHOT_ROOT` and validates the manifest, allow-list,
content hash, and byte budget before creating a session. See
[`apps/README.md`](apps/README.md) for upload, registration, and container examples.

</details>

## Evaluation

StreamRecall includes a reproducible evaluation stack for delayed-query
streaming video temporal grounding:

- frozen method, model, budget, and protocol configurations;
- official temporal IoU / recall metrics and group-wise diagnostics;
- paired video bootstrap and resource accounting;
- immutable result directories keyed by Git commit;
- resolved config snapshots, hashes, commands, environment, and provenance;
- deterministic golden regressions and protocol audits.

| Engineering gate | Status |
|---|:---:|
| Python API + core regression suite | ✅ |
| Next.js production build | ✅ |
| Ruff static checks | ✅ |
| Snapshot integrity and actual-byte budget | ✅ |
| No-future-frame and snapshot-only audit | ✅ |
| Formal cross-dataset benchmark | 🚧 In progress |

We intentionally do **not** report development results as test-set performance
or claim state of the art before the formal cross-dataset benchmark is complete.

### Run verification

```bash
git submodule update --init third_party/TimeLens
make test
```

The browser suite lives in `apps/web/tests` and can be run with
`npm run test:e2e` after installing a Playwright browser.

## Repository map

```text
StreamRecall/
├── apps/
│   ├── api/                 FastAPI, workers, persistence, evidence API
│   └── web/                 Next.js visual-memory workspace
├── streamtimelens/          Memory, retrieval, grounding, evaluation engine
├── src/baas/                Shared benchmark and provenance utilities
├── configs/                 Reproducible baseline configurations
├── scripts/                 Experiment, audit, and reproduction entry points
├── artifacts/               Small public manifests only
└── third_party/TimeLens/     Optional upstream dependency (Git submodule)
```

Model checkpoints, datasets, uploads, runtime databases, logs, and generated
results are intentionally excluded from Git.

## Scope and roadmap

StreamRecall is a research prototype—not a production surveillance, incident
response, or safety-decision system.

- [x] Immutable, byte-bounded visual snapshots
- [x] Snapshot-only query API with evidence authorization
- [x] CLIP retrieval + optional TimeLens local refinement
- [x] Interactive product demo and strict upload path
- [x] Reproducible evaluation and protocol audit stack
- [ ] Complete formal evaluation across all frozen datasets
- [x] Publish a small redistributable end-to-end video example
- [ ] Package a hardware-portable real-model deployment profile

## License

Code authored for this project is released under the [MIT License](LICENSE).

The optional `third_party/TimeLens` submodule, model weights, and related assets
are governed by the upstream TimeLens license, which includes academic-use and
geographic restrictions. Those terms are **not** replaced by this repository's
MIT license. Datasets and models remain subject to their own upstream terms.

## Citation

If StreamRecall is useful in your research or engineering work, please cite the
repository:

```bibtex
@software{streamrecall2026,
  title  = {StreamRecall: Auditable Memory-Bounded Recall for Streaming Video},
  author = {Viiitoo},
  year   = {2026},
  url    = {https://github.com/Viiitoo/StreamRecall}
}
```

## Acknowledgements

StreamRecall builds on ideas and components from
[TimeLens](https://github.com/TencentARC/TimeLens),
[CLIP](https://github.com/openai/CLIP), and temporal video grounding research
including SnAG. Third-party implementations remain attributed to their original
authors and are not represented as official StreamRecall results.
