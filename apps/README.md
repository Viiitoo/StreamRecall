# StreamRecall product prototype

This directory contains the runnable StreamRecall product:

- `api/`: FastAPI session, conversation, recall, evidence and SSE APIs;
- `web/`: Next.js visual-memory workspace;
- `run-local.sh`: starts both services for local development.

The default backend is deliberately deterministic and model-free. It
exercises the complete product contract—bounded snapshot state, temporal
candidates, evidence references, explicit abstention and audit output—without
occupying a GPU. A production-shaped adapter also connects the same API contract
to the frozen StreamTimeLens Hybrid V3 implementation for real CLIP + TimeLens
inference.

## Run

From the repository root, install the isolated backend environment and locked
frontend dependencies, then start both services:

```bash
make setup
make demo
```

After both services report ready, open the local UI at `http://127.0.0.1:3000`.
The API documentation is served locally at `http://127.0.0.1:8000/docs`.
These addresses are only reachable from the machine running StreamRecall;
remote setups require port forwarding or a deployed URL.

The browser uses same-origin `/api` and `/healthz` URLs. Next.js proxies those
requests to `http://127.0.0.1:8000`, so the UI also works when Codex, SSH or a
development environment remaps the frontend to another localhost port. Set
`STREAMRECALL_API_ORIGIN` on the Next.js process only when FastAPI lives at a
different server-side origin.

## Verify

```bash
make test
cd apps/web && npm run test:e2e
```

On older Linux hosts unsupported by the current Playwright browser bundle, set
`PLAYWRIGHT_CHROMIUM_EXECUTABLE` to an existing Chromium executable before
running the E2E test.

## Product contract demonstrated today

- one immutable 1 MiB-budget session snapshot;
- actual state usage surfaced in the UI;
- conversation and turn state persisted in SQLite;
- non-blocking turn submission with recoverable, live SSE Agent events;
- `FOUND` and `NOT_FOUND` are separate outcomes;
- answer spans, alternatives, evidence frames and costs are structured;
- evidence endpoints reject frame references not cited by the answer;
- Agent execution is available through an SSE event stream;
- audit output reports snapshot-only access, future-frame isolation and model
  call/frame limits.

The demo labels its TimeLens stage as simulated in `/healthz`; it must not be
used to claim real model latency or accuracy.

Turn creation returns HTTP 202 immediately with a persisted `running` turn. A
bounded worker pool executes recall independently of the request lifecycle, and
the web client renders each persisted Agent event as it arrives. The event
endpoint honors both `?after=<event-id>` and the standard `Last-Event-ID` header,
so EventSource reconnects resume without duplicating trace steps. Worker errors
become a terminal `failed` turn and a sanitized `turn.failed` event rather than
leaking model internals.

## Use the real Hybrid V3 backend

The API can switch to the existing frozen CLIP + TimeLens implementation
without changing its REST contract. Point the service at a directory containing
audited StreamTimeLens snapshots and select the backend before launching:

```bash
export STREAMRECALL_BACKEND=hybrid-v3
export STREAMRECALL_SNAPSHOT_ROOT=/absolute/path/to/snapshot-root
export STREAMRECALL_CLIP_DEVICE=cuda:0
export STREAMRECALL_DEVICE_MAP=cuda:0
./run-local.sh
```

Register one snapshot using a path relative to `STREAMRECALL_SNAPSHOT_ROOT`:

```bash
curl -X POST http://127.0.0.1:8000/api/v1/sessions/from-snapshot \
  -H 'Content-Type: application/json' \
  -d '{
    "name": "Charades strict snapshot",
    "snapshot_relative_path": "video-id/rho_1.00",
    "suggested_queries": ["person opens a door"]
  }'
```

Registration constructs a `SnapshotReader`, verifies the manifest hash, checks
the complete file allow-list and byte budget, and stores only the validated
snapshot under an opaque session ID. Absolute paths and parent traversal are
rejected. The internal snapshot path is never returned by the API. Evidence
images are read back through `SnapshotReader.frame_path`, and only frame refs
cited by the persisted answer can be requested.

The Hybrid backend loads CLIP and TimeLens lazily on the first query, verifies
their registered identities, preserves the frozen Hybrid V3 configuration and
maps the native `HybridQueryOutput` into the same answer/candidate/audit schema
used by the model-free demo. Real inference should run in the project's pinned
GPU environment described by `streamtimelens/README.md`.

For a local GPU-backed API, build the small API layer on top of the pinned model
image and bind the repository at runtime:

```bash
docker build -f apps/api/Dockerfile.hybrid -t streamrecall-hybrid-api apps/api
docker run --rm --gpus 'device=0' -p 127.0.0.1:8000:8000 \
  -v "$PWD:/workspace" -v "$PWD:$PWD" streamrecall-hybrid-api
```

The second bind preserves absolute snapshot paths already persisted by the host
ingest worker. This lets the containerized Hybrid API read existing snapshots
without rewriting trusted session metadata.

A product-path smoke query on the uploaded ActivityNet sample
`v_arfBwR8qgPw` completed against its 45 KiB, three-frame `rho_1.00` snapshot:
one model call returned grounded evidence at 0.0, 2.0 and 4.0 seconds in about
1.11 seconds. The resulting 0.0–5.32 second interval is deliberately coarser
than the 2.54–5.32 second reference span, making the accuracy tradeoff of an
extremely sparse snapshot visible rather than hiding it.

## Upload and ingest

The **New stream** action uploads one MP4 to an opaque generated path and starts
a background ingest job. The worker invokes the frozen Uniform visual-memory
configuration at 2 decode FPS and 0.5 CLIP FPS, materializing snapshots at
arrival ratios 0.25, 0.5, 0.75 and 1.0 in a single decode pass. The completed
`rho_1.00` snapshot is validated and registered automatically.

Strict upload sessions set `retain_source=false` and delete the app-owned MP4
copy only after a valid final snapshot has been produced. A failed ingest keeps
its input for diagnosis. Uploads are streamed in 1 MiB chunks, limited to 512
MiB by default, checked for an MP4 container header and never stored under the
client-provided filename. Override the limit with
`STREAMRECALL_MAX_UPLOAD_BYTES`.

An actual CLIP ingest smoke confirmed that this worker produces snapshot config
hash `6b5f256ec8e34514d6321e1b17220e4e061de7ac7653878ac62e19877ddf8a47`,
which exactly matches the frozen hash required by Hybrid V3.
