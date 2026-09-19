"use client";

import {
  ArrowUp,
  Brain,
  Check,
  CheckCircle,
  CirclesThree,
  ClockCounterClockwise,
  Database,
  Eye,
  FilmStrip,
  Fingerprint,
  Gauge,
  Info,
  LockKey,
  MagnifyingGlass,
  Plus,
  Play,
  Pause,
  ShieldCheck,
  Sparkle,
  UploadSimple,
  WarningCircle,
  Waveform,
  X,
} from "@phosphor-icons/react";
import { FormEvent, useEffect, useMemo, useRef, useState } from "react";

// Same-origin requests survive Codex/SSH/dev-server port remapping. Next proxies
// them to FastAPI without exposing the backend origin to the browser.
const API = process.env.NEXT_PUBLIC_API_URL ?? "";

type TimelineFrame = { frame_ref: string; timestamp_s: number; importance: number };
type Session = {
  id: string;
  name: string;
  mode: string;
  source_type: string;
  status: string;
  duration_s: number;
  observed_until_s: number;
  memory_budget_bytes: number;
  state_bytes: number;
  retained_frame_count: number;
  snapshot_id: string;
  snapshot_sha256: string;
  suggested_queries: string[];
  timeline: TimelineFrame[];
};
type Evidence = { frame_ref: string; timestamp_s: number; role: string };
type Answer = {
  status: string;
  message: string;
  resolved_query: string;
  span: { start_s: number; end_s: number } | null;
  confidence: number;
  evidence: Evidence[];
  alternatives: Array<{ start_s: number; end_s: number; score: number; candidate_id: string }>;
  reason: string;
  limitations: string[];
  audit: Record<string, boolean | number>;
  cost: { search_ms: number; refine_ms: number; total_ms: number; generated_tokens: number };
};
type AgentEvent = { event_id: number; type: string; payload: Record<string, unknown> };
type ChatTurn = {
  id: string;
  question: string;
  status: "running" | "completed" | "failed";
  answer?: Answer;
  events: AgentEvent[];
};
type UploadJob = { id: string; status: string; progress: number; stage: string; error?: { detail?: string } };
type Health = { status: string; backend: string; models: { clip: string; timelens: string } };
type LocalSource = { sessionId?: string; url: string; name: string };

function timecode(value: number) {
  const minute = Math.floor(value / 60);
  const second = Math.floor(value % 60);
  return `${minute.toString().padStart(2, "0")}:${second.toString().padStart(2, "0")}`;
}

function bytes(value: number) {
  return value >= 1048576 ? `${(value / 1048576).toFixed(1)} MiB` : `${Math.round(value / 1024)} KiB`;
}

function Timeline({ session, answer }: { session: Session; answer?: Answer }) {
  const ticks = [0, 0.25, 0.5, 0.75, 1];
  return (
    <div className="timeline-shell">
      <div className="timeline-labels">
        <span>MEMORY MAP</span>
        <span>{session.retained_frame_count} moments retained</span>
      </div>
      <div className="timeline-track">
        <div className="observed" style={{ width: `${(session.observed_until_s / session.duration_s) * 100}%` }} />
        {session.timeline.map((frame) => (
          <button
            className="memory-mark"
            key={frame.frame_ref}
            title={`${timecode(frame.timestamp_s)} · retained evidence`}
            style={{
              left: `${(frame.timestamp_s / session.duration_s) * 100}%`,
              height: `${14 + frame.importance * 25}px`,
              opacity: 0.5 + frame.importance * 0.5,
            }}
          />
        ))}
        {answer?.alternatives.map((candidate) => (
          <span
            className="candidate-range"
            key={candidate.candidate_id}
            style={{
              left: `${(candidate.start_s / session.duration_s) * 100}%`,
              width: `${((candidate.end_s - candidate.start_s) / session.duration_s) * 100}%`,
            }}
          />
        ))}
        {answer?.span && (
          <span
            className="answer-range"
            style={{
              left: `${(answer.span.start_s / session.duration_s) * 100}%`,
              width: `${Math.max(1.8, ((answer.span.end_s - answer.span.start_s) / session.duration_s) * 100)}%`,
            }}
          />
        )}
      </div>
      <div className="timeline-ticks">
        {ticks.map((tick) => <span key={tick}>{timecode(session.duration_s * tick)}</span>)}
      </div>
      <div className="timeline-legend">
        <span><i className="legend-dot retained" /> Retained memory</span>
        <span><i className="legend-dot candidate" /> Alternative</span>
        <span><i className="legend-dot final" /> Grounded answer</span>
      </div>
    </div>
  );
}

function EvidenceCard({ item, turnId }: { item: Evidence; turnId: string }) {
  const url = `${API}/api/v1/turns/${turnId}/evidence/${item.frame_ref}`;
  return (
    <article className="evidence-card">
      {/* The URL is an audited API resource rather than a Next image asset. */}
      {/* eslint-disable-next-line @next/next/no-img-element */}
      <img src={url} alt={`Evidence at ${timecode(item.timestamp_s)}`} />
      <div className="evidence-meta">
        <span>{timecode(item.timestamp_s)}</span>
        <span>{item.role}</span>
      </div>
    </article>
  );
}

function Trace({ answer }: { answer: Answer }) {
  const items = answer.status === "NOT_FOUND" ? [
    { icon: MagnifyingGlass, label: "Memory search", note: "No reliable candidate", value: `${answer.cost.search_ms} ms` },
    { icon: ShieldCheck, label: "Safe abstention", note: "Source video was not replayed", value: "passed" },
  ] : [
    { icon: MagnifyingGlass, label: "Semantic retrieval", note: "6 temporal clusters compared", value: `${answer.cost.search_ms} ms` },
    { icon: CirclesThree, label: "Evidence allocation", note: "Boundary-anchored sampling", value: "16 frames" },
    { icon: Brain, label: "Temporal verification", note: "TimeLens-7B · 1 model call", value: `${answer.cost.refine_ms} ms` },
    { icon: CheckCircle, label: "Answer grounded", note: answer.reason.replaceAll("_", " "), value: `${answer.cost.total_ms} ms` },
  ];
  return (
    <div className="trace-list">
      {items.map(({ icon: Icon, label, note, value }, index) => (
        <div className="trace-item" key={label}>
          <div className="trace-rail">
            <span><Icon size={15} weight="bold" /></span>
            {index < items.length - 1 && <i />}
          </div>
          <div><strong>{label}</strong><small>{note}</small></div>
          <em>{value}</em>
        </div>
      ))}
    </div>
  );
}

const eventCopy: Record<string, string> = {
  "turn.started": "Turn accepted",
  "intent.resolved": "Intent resolved",
  "memory.search.started": "Searching immutable memory",
  "memory.search.completed": "Candidate moments retrieved",
  "candidate.inspection.completed": "Evidence frames allocated",
  "model.refinement.started": "Temporal verification started",
  "model.refinement.completed": "Temporal verification complete",
  "answer.completed": "Grounded answer ready",
  "turn.failed": "Recall worker failed",
};

function LiveTrace({ events, failed }: { events: AgentEvent[]; failed: boolean }) {
  return (
    <div className="live-trace" aria-live="polite">
      <div className="live-trace-heading">
        <span className={failed ? "failed" : "active"} />
        <strong>{failed ? "Agent stopped" : "Agent working"}</strong>
        <small>{events.length} events persisted</small>
      </div>
      {events.map((item, index) => (
        <div className="live-event" key={item.event_id}>
          <i>{index < events.length - 1 ? <Check size={11} weight="bold" /> : <Waveform size={11} />}</i>
          <span>{eventCopy[item.type] ?? item.type.replaceAll(".", " ")}</span>
        </div>
      ))}
    </div>
  );
}

export default function Home() {
  const [sessions, setSessions] = useState<Session[]>([]);
  const [session, setSession] = useState<Session | null>(null);
  const [conversationId, setConversationId] = useState<string | null>(null);
  const [query, setQuery] = useState("");
  const [turns, setTurns] = useState<ChatTurn[]>([]);
  const [working, setWorking] = useState(false);
  const [error, setError] = useState("");
  const [uploadOpen, setUploadOpen] = useState(false);
  const [uploadName, setUploadName] = useState("");
  const [uploadFile, setUploadFile] = useState<File | null>(null);
  const [uploadJob, setUploadJob] = useState<UploadJob | null>(null);
  const [uploadSubmitting, setUploadSubmitting] = useState(false);
  const [uploadError, setUploadError] = useState("");
  const [health, setHealth] = useState<Health | null>(null);
  const [playing, setPlaying] = useState(false);
  const [playbackTime, setPlaybackTime] = useState(0);
  const [localSource, setLocalSource] = useState<LocalSource | null>(null);
  const localSourceUrl = useRef<string | null>(null);
  const latest = turns.at(-1);
  const uploadBusy = uploadSubmitting || Boolean(uploadJob && !["failed", "completed"].includes(uploadJob.status));

  async function refreshSessions(preferredId?: string) {
    return fetch(`${API}/api/v1/sessions`)
      .then((response) => {
        if (!response.ok) throw new Error("API unavailable");
        return response.json();
      })
      .then((rows: Session[]) => {
        setSessions(rows);
        setSession(
          rows.find((row) => row.id === preferredId)
          ?? rows.find((row) => row.source_type === "demo")
          ?? rows[0],
        );
      })
  }

  useEffect(() => {
    refreshSessions().catch(() => setError("无法连接 StreamRecall API，请确认后端已在 8000 端口启动。"));
    fetch(`${API}/healthz`).then((response) => response.json()).then(setHealth).catch(() => undefined);
    // The API origin is fixed for the lifetime of this page.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => () => {
    if (localSourceUrl.current) URL.revokeObjectURL(localSourceUrl.current);
  }, []);

  useEffect(() => {
    setPlaying(false);
    setPlaybackTime(session?.duration_s ?? 0);
  }, [session?.id, session?.duration_s]);

  useEffect(() => {
    if (!playing || !session || (session.source_type !== "demo" && !session.timeline.length)) return;
    // The demo is a 24-second compressed visualization of the full memory
    // timeline, not access to a stored source video.
    const secondsPerTick = session.duration_s / 240;
    const timer = window.setInterval(() => {
      setPlaybackTime((current) => Math.min(session.duration_s, current + secondsPerTick));
    }, 100);
    return () => window.clearInterval(timer);
  }, [playing, session]);

  useEffect(() => {
    if (session && playing && playbackTime >= session.duration_s) setPlaying(false);
  }, [playbackTime, playing, session]);

  const usage = useMemo(() => session ? (session.state_bytes / session.memory_budget_bytes) * 100 : 0, [session]);
  const queryAvailable = session?.source_type === "demo" || health?.backend === "hybrid-v3";
  const replayAvailable = session?.source_type === "demo" || Boolean(session?.timeline.length);
  const replayProgress = session ? Math.min(1, playbackTime / session.duration_s) : 1;
  const replayFrame = useMemo(() => {
    if (!session || session.source_type === "demo") return null;
    return session.timeline.reduce<TimelineFrame | null>(
      (selected, frame) => frame.timestamp_s <= playbackTime ? frame : selected,
      session.timeline[0] ?? null,
    );
  }, [playbackTime, session]);
  const hasLocalSource = localSource?.sessionId === session?.id;

  function stageLocalSource(file: File | null, sessionId?: string, updateUpload = true) {
    if (updateUpload) setUploadFile(file);
    if (!file) return;
    if (localSourceUrl.current) URL.revokeObjectURL(localSourceUrl.current);
    const url = URL.createObjectURL(file);
    localSourceUrl.current = url;
    setLocalSource({ sessionId, url, name: file.name });
  }

  function toggleReplay() {
    if (!session || !replayAvailable) return;
    if (playing) {
      setPlaying(false);
      return;
    }
    if (playbackTime >= session.duration_s) setPlaybackTime(0);
    setPlaying(true);
  }

  function focusReplay(start: number) {
    if (!replayAvailable) return;
    setPlaybackTime(start);
    setPlaying(true);
  }

  async function ensureConversation(active: Session) {
    if (conversationId) return conversationId;
    const response = await fetch(`${API}/api/v1/sessions/${active.id}/conversations`, { method: "POST" });
    if (!response.ok) throw new Error("Failed to create conversation");
    const value = await response.json();
    setConversationId(value.id);
    return value.id as string;
  }

  async function ask(event?: FormEvent, suggested?: string) {
    event?.preventDefault();
    const message = (suggested ?? query).trim();
    if (!session || !message || working) return;
    if (!queryAvailable) {
      setError("当前服务仍是 Demo backend；真实快照需要 Hybrid V3。输入内容已保留。 ");
      return;
    }
    setWorking(true);
    setError("");
    try {
      const conversation = await ensureConversation(session);
      const response = await fetch(`${API}/api/v1/conversations/${conversation}/turns`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message }),
      });
      if (!response.ok) {
        const detail = await response.json().catch(() => ({}));
        throw new Error(detail.detail ?? "Query failed");
      }
      const turn = await response.json();
      setTurns((current) => [...current, {
        id: turn.id, question: message, status: "running", events: [],
      }]);
      setQuery("");
      await new Promise<void>((resolve, reject) => {
        const source = new EventSource(`${API}/api/v1/turns/${turn.id}/events`);
        source.addEventListener("trace", (raw) => {
          const received = JSON.parse((raw as MessageEvent).data) as AgentEvent;
          setTurns((current) => current.map((item) => {
            if (item.id !== turn.id || item.events.some((seen) => seen.event_id === received.event_id)) return item;
            if (received.type === "answer.completed") {
              return {
                ...item,
                status: "completed",
                answer: received.payload.answer as Answer,
                events: [...item.events, received],
              };
            }
            if (received.type === "turn.failed") {
              return { ...item, status: "failed", events: [...item.events, received] };
            }
            return { ...item, events: [...item.events, received] };
          }));
          if (received.type === "answer.completed" || received.type === "turn.failed") {
            source.close();
            if (received.type === "turn.failed") reject(new Error("Recall worker failed"));
            else resolve();
          }
        });
        source.onerror = () => {
          // EventSource automatically reconnects and sends Last-Event-ID. Keep the
          // persisted turn visible instead of discarding progress on a network gap.
          setError("事件流暂时中断，正在从上一个 Agent 事件恢复…");
        };
      });
      setError("");
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "查询执行失败，请检查 API 状态后重试。");
    } finally {
      setWorking(false);
    }
  }

  async function upload(event?: FormEvent) {
    event?.preventDefault();
    if (!uploadFile || !uploadName.trim() || uploadBusy) return;
    setUploadSubmitting(true);
    setUploadError("");
    setUploadJob(null);
    const form = new FormData();
    form.append("name", uploadName.trim());
    form.append("file", uploadFile);
    form.append("retain_source", "false");
    try {
      const response = await fetch(`${API}/api/v1/sessions/upload`, { method: "POST", body: form });
      if (!response.ok) {
        const detail = await response.json().catch(() => ({}));
        throw new Error(detail.detail ?? "Upload failed");
      }
      const created = await response.json();
      const sessionId = created.session.id as string;
      const jobId = created.job.id as string;
      setUploadJob(created.job);
      const poll = window.setInterval(async () => {
        try {
          const jobResponse = await fetch(`${API}/api/v1/jobs/${jobId}`);
          if (!jobResponse.ok) throw new Error("Job status unavailable");
          const job: UploadJob = await jobResponse.json();
          setUploadJob(job);
          if (job.status === "completed") {
            window.clearInterval(poll);
            setLocalSource((current) => current ? { ...current, sessionId } : current);
            await refreshSessions(sessionId);
            setConversationId(null);
            setTurns([]);
            setUploadOpen(false);
            setUploadJob(null);
            setUploadName("");
            setUploadFile(null);
          } else if (job.status === "failed") {
            window.clearInterval(poll);
            setUploadError(job.error?.detail ?? "Ingest failed");
          }
        } catch {
          window.clearInterval(poll);
          setUploadError("Lost connection while tracking ingest progress.");
        }
      }, 1200);
    } catch (cause) {
      setUploadError(cause instanceof Error ? cause.message : "Upload failed");
    } finally {
      setUploadSubmitting(false);
    }
  }

  if (!session) {
    return <main className="loading"><Waveform size={28} /><span>{error || "Loading visual memory…"}</span></main>;
  }

  return (
    <main className="app-shell">
      <header className="topbar">
        <div className="brand">
          <span className="brand-mark"><Waveform size={22} weight="bold" /></span>
          <div><strong>StreamRecall</strong><small>VISUAL MEMORY AGENT</small></div>
        </div>
        <div className="top-status">
          <span className="live-dot" /> SYSTEM ONLINE
          <i />
          <span><LockKey size={15} weight="bold" /> STRICT MEMORY</span>
          <button aria-label="System information"><Info size={18} /></button>
        </div>
      </header>

      <section className="workspace-heading">
        <div>
          <p className="eyebrow"><Fingerprint size={15} /> SESSION / {session.id.toUpperCase()}</p>
          <h1>{session.name}</h1>
        </div>
        <div className="heading-actions">
          <button className="new-session" onClick={() => setUploadOpen(true)}><Plus size={14} weight="bold" /> New stream</button>
          {sessions.length > 1 && (
            <label className="session-switcher">
              <span>ACTIVE MEMORY</span>
              <select
                value={session.id}
                onChange={(event) => {
                  const selected = sessions.find((row) => row.id === event.target.value);
                  if (!selected) return;
                  setSession(selected);
                  setConversationId(null);
                  setTurns([]);
                  setError("");
                }}
              >
                {sessions.map((row) => <option key={row.id} value={row.id}>{row.name}</option>)}
              </select>
            </label>
          )}
          <div className="snapshot-chip">
            <ShieldCheck size={20} weight="fill" />
            <div><strong>Snapshot verified</strong><small>{session.snapshot_sha256.slice(0, 12)}…</small></div>
          </div>
        </div>
      </section>

      <div className="workspace-grid">
        <section className="visual-column">
          <article className="panel video-panel">
            <div className={`video-scene ${playing ? "previewing" : ""}`}>
              {hasLocalSource && localSource ? (
                <video
                  className="local-source-video"
                  src={localSource.url}
                  controls
                  playsInline
                  preload="metadata"
                  aria-label={`Local source preview: ${localSource.name}`}
                />
              ) : <>
              {session.source_type === "demo" ? <>
                <div className="room-window" />
                <div className="room-cabinet"><i /><i /><i /></div>
                <div className="room-table"><span style={{
                  opacity: replayProgress < .32 ? 0 : 1,
                  transform: `translate(${Math.min(0, (replayProgress - .5) * 470)}px, ${Math.min(0, (replayProgress - .5) * 230)}px)`,
                }} /><i /></div>
                <div className="person" style={{ transform: `translateX(${Math.sin(replayProgress * Math.PI) * 34}px)` }}><i /><span /></div>
              </> : replayFrame ? <>
                {/* Retained snapshot frame; the deleted source video is never fetched. */}
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img
                  className="snapshot-replay-frame"
                  src={`${API}/api/v1/sessions/${session.id}/frames/${replayFrame.frame_ref}`}
                  alt={`Retained frame at ${timecode(replayFrame.timestamp_s)}`}
                />
                <div className="snapshot-replay-badge">RETAINED FRAME · {timecode(replayFrame.timestamp_s)}</div>
              </> : null}
              <div className="camera-overlay top-left"><span className="live-dot" /> {session.source_type === "demo" ? "OBSERVED" : "SNAPSHOT REPLAY"}</div>
              <div className="camera-overlay top-right">{session.source_type === "demo" ? "CAM 01 · EDGE" : `${session.retained_frame_count} FRAMES · STRICT`}</div>
              <button
                className="play-button"
                aria-label={replayAvailable ? (playing ? "Pause memory replay" : "Play memory replay") : "Snapshot replay unavailable"}
                onClick={toggleReplay}
                disabled={!replayAvailable}
                title={replayAvailable ? "24-second compressed memory replay" : "No retained frames are available"}
              >
                {replayAvailable ? (playing ? <Pause size={23} weight="fill" /> : <Play size={23} weight="fill" />) : <LockKey size={21} weight="bold" />}
              </button>
              <div className="video-time"><strong>{timecode(playbackTime)}</strong><span>/ {timecode(session.duration_s)}</span></div>
              <div className="video-progress"><i style={{ width: `${replayProgress * 100}%` }} /></div>
              </>}
            </div>
            <div className="panel-footer">
              <div><FilmStrip size={17} /><span>{hasLocalSource ? "Local source preview" : playing ? "Memory replay · 24 s" : session.source_type === "demo" ? "Stream complete" : `${session.retained_frame_count} retained frames`}</span></div>
              <div><ClockCounterClockwise size={17} /><span>Single-pass ingest</span></div>
              {session.source_type === "demo" ? (
                <button>Demo source <span>⌄</span></button>
              ) : (
                <label className="local-source-picker">
                  <input
                    type="file"
                    accept="video/mp4,.mp4"
                    onChange={(event) => stageLocalSource(event.target.files?.[0] ?? null, session.id, false)}
                  />
                  {hasLocalSource ? "Replace local source" : "Attach local source"}
                </label>
              )}
            </div>
          </article>

          <article className="panel memory-panel">
            <div className="panel-title-row">
              <div>
                <p className="eyebrow"><Database size={15} /> IMMUTABLE SNAPSHOT</p>
                <h2>What the agent remembers</h2>
              </div>
              <div className="budget-stat">
                <span>{bytes(session.state_bytes)}</span>
                <small>of {bytes(session.memory_budget_bytes)}</small>
              </div>
            </div>
            <div className="budget-bar"><i style={{ width: `${usage}%` }} /></div>
            <Timeline session={session} answer={latest?.answer} />
            <div className="metric-grid">
              <div><Eye size={17} /><span>Observed</span><strong>{timecode(session.observed_until_s)}</strong></div>
              <div><Database size={17} /><span>Retained</span><strong>{session.retained_frame_count} frames</strong></div>
              <div><Gauge size={17} /><span>Budget used</span><strong>{usage.toFixed(1)}%</strong></div>
              <div><LockKey size={17} /><span>Original video</span><strong>Isolated</strong></div>
            </div>
          </article>
        </section>

        <section className="panel agent-panel">
          <div className="agent-header">
            <div className="agent-orb"><Sparkle size={21} weight="fill" /></div>
            <div><h2>Recall Agent</h2><p>Ask what happened before the question arrived.</p></div>
            <span className={`agent-ready ${working ? "working" : ""}`}><i /> {working ? "WORKING" : "READY"}</span>
          </div>

          <div className="conversation">
            {turns.length === 0 && (
              <div className="welcome">
                <div className="welcome-icon"><Brain size={28} /></div>
                <h3>I remember fragments, not the recording.</h3>
                <p>I can search the bounded visual snapshot, compare moments, and ground an answer in retained evidence.</p>
                <div className="suggestions">
                  {session.suggested_queries.map((item) => (
                    <button key={item} onClick={() => ask(undefined, item)}><span>{item}</span><ArrowUp size={14} /></button>
                  ))}
                </div>
              </div>
            )}

            {turns.map((turn) => (
              <div className="turn" key={turn.id}>
                <div className="user-message"><span>{turn.question}</span></div>
                {!turn.answer ? (
                  <div className={`answer-card pending ${turn.status === "failed" ? "not-found" : ""}`}>
                    <LiveTrace events={turn.events} failed={turn.status === "failed"} />
                  </div>
                ) : <div className={`answer-card ${turn.answer.status === "NOT_FOUND" ? "not-found" : ""}`}>
                  <div className="answer-status">
                    {turn.answer.status === "NOT_FOUND" ? <WarningCircle size={18} /> : <CheckCircle size={18} weight="fill" />}
                    <span>{turn.answer.status.replaceAll("_", " ")}</span>
                    <em>{Math.round(turn.answer.confidence * 100)}% confidence</em>
                  </div>
                  <p>{turn.answer.message}</p>
                  {turn.answer.span && (
                    <div className="grounded-time">
                      <ClockCounterClockwise size={18} />
                      <strong>{timecode(turn.answer.span.start_s)}</strong>
                      <span>→</span>
                      <strong>{timecode(turn.answer.span.end_s)}</strong>
                      <button
                        onClick={() => focusReplay(turn.answer!.span!.start_s)}
                        disabled={!replayAvailable}
                        title={replayAvailable ? "Replay from this grounded moment" : "Original video is unavailable in strict mode"}
                      ><Play size={12} weight="fill" /> Focus</button>
                    </div>
                  )}
                  {turn.answer.evidence.length > 0 && (
                    <div className="evidence-grid">
                      {turn.answer.evidence.map((item) => <EvidenceCard key={item.frame_ref} item={item} turnId={turn.id} />)}
                    </div>
                  )}
                  <details className="trace-details" open={turn === latest}>
                    <summary><span><Waveform size={16} /> Agent trace</span><small>{turn.answer.cost.total_ms} ms total</small></summary>
                    <Trace answer={turn.answer} />
                  </details>
                  <div className="answer-audit"><ShieldCheck size={15} /><span>Snapshot only</span><i /><span>No future frames</span><i /><span>{turn.answer.audit.model_calls} model call</span></div>
                </div>}
              </div>
            ))}
          </div>

          <div className="composer-wrap">
            {error && <div className="error-banner"><WarningCircle size={16} />{error}</div>}
            <form className="composer" onSubmit={ask}>
              <textarea
                value={query}
                onChange={(event) => setQuery(event.target.value)}
                onKeyDown={(event) => {
                  if (event.key === "Enter" && !event.shiftKey) {
                    event.preventDefault();
                    ask();
                  }
                }}
                placeholder={queryAvailable ? "Ask about a past event…" : "Enable Hybrid V3 to query this snapshot"}
                rows={2}
              />
              <div className="composer-footer">
                <span><LockKey size={14} /> Search limited to {bytes(session.state_bytes)} snapshot</span>
                <button disabled={!query.trim() || working} aria-label="Send query"><ArrowUp size={18} weight="bold" /></button>
              </div>
            </form>
            <p className="disclaimer"><ShieldCheck size={13} /> Raw video is unavailable to the query agent in strict mode.</p>
          </div>
        </section>
      </div>

      {uploadOpen && (
        <div className="modal-backdrop" role="presentation" onMouseDown={() => !uploadBusy && setUploadOpen(false)}>
          <section className="upload-modal" role="dialog" aria-modal="true" aria-labelledby="upload-title" onMouseDown={(event) => event.stopPropagation()}>
            <button className="modal-close" onClick={() => setUploadOpen(false)} disabled={uploadBusy} aria-label="Close"><X size={18} /></button>
            <div className="upload-icon"><UploadSimple size={25} /></div>
            <p className="eyebrow"><FilmStrip size={14} /> NEW VISUAL MEMORY</p>
            <h2 id="upload-title">Ingest an MP4 stream</h2>
            <p className="modal-copy">The video is decoded once into a 1 MiB immutable snapshot. In strict mode, the uploaded source is deleted after successful ingest.</p>
            <form onSubmit={upload}>
              <label>
                <span>SESSION NAME</span>
                <input value={uploadName} onChange={(event) => setUploadName(event.target.value)} placeholder="Kitchen · evening stream" disabled={uploadBusy} />
              </label>
              <label className={`file-drop ${uploadFile ? "selected" : ""}`}>
                <input type="file" accept="video/mp4,.mp4" onChange={(event) => stageLocalSource(event.target.files?.[0] ?? null)} disabled={uploadBusy} />
                <UploadSimple size={20} />
                <strong>{uploadFile?.name ?? "Choose an MP4 file"}</strong>
                <small>{uploadFile ? `${(uploadFile.size / 1048576).toFixed(1)} MiB` : "Up to 512 MiB"}</small>
              </label>
              {uploadJob && (
                <div className="ingest-progress">
                  <div><span>{uploadJob.stage.replaceAll("_", " ")}</span><em>{Math.round(uploadJob.progress * 100)}%</em></div>
                  <i><span style={{ width: `${Math.max(4, uploadJob.progress * 100)}%` }} /></i>
                  <small>Single-pass CLIP ingest may take several minutes.</small>
                </div>
              )}
              {uploadSubmitting && !uploadJob && (
                <div className="ingest-progress" aria-live="polite">
                  <div><span>uploading source</span><em>UPLOADING</em></div>
                  <i><span className="upload-indeterminate" /></i>
                  <small>Transferring the MP4 to the isolated ingest worker…</small>
                </div>
              )}
              {uploadError && <div className="error-banner"><WarningCircle size={16} />{uploadError}</div>}
              <div className="strict-notice"><ShieldCheck size={18} /><div><strong>Strict memory mode</strong><small>{health?.backend === "hybrid-v3" ? "Query workers receive the snapshot, never this MP4." : "A real snapshot will be built. Restart with Hybrid V3 to query it."}</small></div></div>
              <button
                type="button"
                className="upload-submit"
                onClick={() => void upload()}
                disabled={!uploadFile || !uploadName.trim() || uploadBusy}
              >
                {uploadSubmitting ? "Uploading source…" : uploadBusy ? "Building visual memory…" : uploadJob?.status === "failed" ? "Retry ingest" : "Start single-pass ingest"}<ArrowUp size={15} />
              </button>
            </form>
          </section>
        </div>
      )}
    </main>
  );
}
