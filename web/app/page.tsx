"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

// ── Static metadata ──────────────────────────────────────────────────────────
type AgentKey = "orchestrator" | "researcher" | "analyst" | "writer" | "critic";

const AGENT_ORDER: AgentKey[] = [
  "orchestrator",
  "researcher",
  "analyst",
  "writer",
  "critic",
];

const AGENT_META: Record<AgentKey, { emoji: string; name: string; role: string }> = {
  orchestrator: { emoji: "🧭", name: "Orchestrator", role: "Plans & synthesises" },
  researcher: { emoji: "🔍", name: "Researcher", role: "Gathers web + docs" },
  analyst: { emoji: "📊", name: "Analyst", role: "Evaluates the evidence" },
  writer: { emoji: "✍️", name: "Writer", role: "Drafts the report" },
  critic: { emoji: "🧐", name: "Critic", role: "Scores & critiques" },
};

const STEPS = [
  { key: "plan", label: "Plan" },
  { key: "research", label: "Research" },
  { key: "analyze", label: "Analyze" },
  { key: "write", label: "Write" },
  { key: "review", label: "Review" },
  { key: "finalize", label: "Finalize" },
];
const STEP_ORDER = STEPS.map((s) => s.key);

const EXAMPLES = [
  "Impact of AI on healthcare diagnostics",
  "The future of solid-state batteries",
  "Economic effects of remote work",
];

// Shown when a run ends before completing — almost always the serverless
// time limit on Vercel's free (Hobby) tier.
const TIMEOUT_MSG =
  "The run didn't finish in time — Vercel's free (Hobby) tier kills any " +
  "request after 60 seconds, and the full pipeline can exceed that with the " +
  "70B model or multiple revision rounds. Fixes: use llama-3.1-8b-instant " +
  "with 1 revision round (the safe default), keep the topic focused, or " +
  "deploy on Vercel Pro and raise maxDuration for longer runs.";

type AgentState = { text: string; status: "idle" | "streaming" | "done" };
const blankAgents = (): Record<AgentKey, AgentState> =>
  AGENT_ORDER.reduce(
    (acc, k) => ({ ...acc, [k]: { text: "", status: "idle" } }),
    {} as Record<AgentKey, AgentState>
  );

interface ServerInfo {
  models: string[];
  webSearch: boolean;
  serverKey: boolean;
}

export default function Page() {
  // ── Inputs ──────────────────────────────────────────────────────────────
  const [topic, setTopic] = useState("");
  const [model, setModel] = useState("llama-3.1-8b-instant");
  const [style, setStyle] = useState<"precise" | "balanced" | "creative">("balanced");
  const [maxIterations, setMaxIterations] = useState(1);
  const [qualityThreshold, setQualityThreshold] = useState(7);
  const [useWebSearch, setUseWebSearch] = useState(true);
  const [context, setContext] = useState("");
  const [userKey, setUserKey] = useState("");
  const [showSettings, setShowSettings] = useState(false);

  // ── Run state ─────────────────────────────────────────────────────────────
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<{ message: string; daily?: boolean } | null>(null);
  const [agents, setAgents] = useState(blankAgents);
  const [doneSteps, setDoneSteps] = useState<Set<string>>(new Set());
  const [currentStep, setCurrentStep] = useState<string | null>(null);
  const [statusLine, setStatusLine] = useState<string>("");
  const [score, setScore] = useState<{ value: number; iteration: number } | null>(null);
  const [citations, setCitations] = useState<string[]>([]);
  const [report, setReport] = useState<string | null>(null);
  const [openAgent, setOpenAgent] = useState<AgentKey | null>(null);
  const [elapsed, setElapsed] = useState(0);
  const [serverInfo, setServerInfo] = useState<ServerInfo | null>(null);

  const abortRef = useRef<AbortController | null>(null);
  const startRef = useRef<number>(0);
  const userStoppedRef = useRef(false);

  // Discover server capabilities (default model, web search, key availability).
  useEffect(() => {
    fetch("/api/research")
      .then((r) => r.json())
      .then((d: ServerInfo) => {
        setServerInfo(d);
        if (d.models?.length) setModel((m) => (d.models.includes(m) ? m : d.models[0]));
      })
      .catch(() => {});
  }, []);

  // Live elapsed timer.
  useEffect(() => {
    if (!running) return;
    const id = setInterval(() => setElapsed((Date.now() - startRef.current) / 1000), 100);
    return () => clearInterval(id);
  }, [running]);

  const keyMissing = !!serverInfo && !serverInfo.serverKey && !userKey.trim();

  // ── Run the pipeline (read the NDJSON stream) ───────────────────────────--
  const run = useCallback(async () => {
    const t = topic.trim();
    if (!t || running) return;

    // Reset
    setError(null);
    setReport(null);
    setCitations([]);
    setScore(null);
    setAgents(blankAgents());
    setDoneSteps(new Set());
    setCurrentStep(null);
    setStatusLine("Connecting…");
    setOpenAgent(null);
    setElapsed(0);
    startRef.current = Date.now();
    setRunning(true);

    const ctrl = new AbortController();
    abortRef.current = ctrl;
    userStoppedRef.current = false;

    let gotFinal = false;
    let gotError = false;
    let loopDone = false;
    let stalled = false;
    let lastActivity = Date.now();
    const STALL_MS = 45000; // no events for this long => function likely timed out

    const advanceTo = (step: string) => {
      const idx = STEP_ORDER.indexOf(step);
      if (idx === -1) return; // init / revise — not a stepper node
      setDoneSteps((prev) => {
        const next = new Set(prev);
        for (let i = 0; i < idx; i++) next.add(STEP_ORDER[i]);
        return next;
      });
      setCurrentStep(step);
    };

    const handle = (e: any) => {
      lastActivity = Date.now();
      switch (e.type) {
        case "step":
          setStatusLine(e.message);
          if (e.step === "revise") advanceTo("write");
          else advanceTo(e.step);
          break;
        case "agent_start": {
          const k = e.agent as AgentKey;
          setOpenAgent(k);
          setAgents((prev) => ({ ...prev, [k]: { text: "", status: "streaming" } }));
          break;
        }
        case "token": {
          const k = e.agent as AgentKey;
          setAgents((prev) => ({
            ...prev,
            [k]: { text: prev[k].text + e.text, status: "streaming" },
          }));
          break;
        }
        case "agent_done": {
          const k = e.agent as AgentKey;
          setAgents((prev) => ({ ...prev, [k]: { ...prev[k], status: "done" } }));
          break;
        }
        case "citations":
          setCitations(e.urls || []);
          break;
        case "score":
          setScore({ value: e.score, iteration: e.iteration });
          break;
        case "final":
          gotFinal = true;
          setReport(e.markdown);
          setCitations(e.citations || []);
          setDoneSteps(new Set(STEP_ORDER));
          setCurrentStep(null);
          setStatusLine(`Done in ${(e.durationMs / 1000).toFixed(1)}s`);
          setOpenAgent(null);
          break;
        case "error":
          gotError = true;
          setError({ message: e.message, daily: e.daily });
          break;
      }
    };

    // Watchdog: if the stream goes silent for too long, the serverless function
    // was almost certainly killed at the 60s limit — abort and surface why,
    // instead of leaving the UI frozen mid-step.
    const watchdog = window.setInterval(() => {
      if (!gotFinal && !gotError && Date.now() - lastActivity > STALL_MS) {
        stalled = true;
        ctrl.abort();
      }
    }, 3000);

    try {
      const res = await fetch("/api/research", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        signal: ctrl.signal,
        body: JSON.stringify({
          topic: t,
          model,
          style,
          maxIterations,
          qualityThreshold,
          useWebSearch,
          context: context.trim() || undefined,
          apiKey: userKey.trim() || undefined,
        }),
      });

      if (!res.ok || !res.body) {
        const data = await res.json().catch(() => ({}));
        throw new Error(data.error || `Request failed (${res.status})`);
      }

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop() ?? "";
        for (const line of lines) {
          const s = line.trim();
          if (!s) continue;
          try {
            handle(JSON.parse(s));
          } catch {
            /* partial line — shouldn't happen after split, ignore */
          }
        }
      }
      loopDone = true; // stream closed normally
    } catch (err: any) {
      if (err?.name === "AbortError") {
        // Watchdog abort => timed out. User-initiated Stop => stay silent.
        if (stalled && !userStoppedRef.current) setError({ message: TIMEOUT_MSG });
      } else {
        setError({ message: err?.message || "Something went wrong." });
      }
    } finally {
      window.clearInterval(watchdog);
      // Stream closed cleanly but no final report arrived => the function was
      // cut off (e.g. 60s limit) without a chance to emit an error event.
      if (loopDone && !gotFinal && !gotError && !userStoppedRef.current) {
        setError({ message: TIMEOUT_MSG });
      }
      // Stop any agent left mid-stream (error/timeout/stop interrupted it)
      // so its card doesn't spin forever.
      setAgents((prev) => {
        let changed = false;
        const next = { ...prev };
        for (const k of AGENT_ORDER) {
          if (next[k].status === "streaming") {
            next[k] = { ...next[k], status: "done" };
            changed = true;
          }
        }
        return changed ? next : prev;
      });
      setRunning(false);
      abortRef.current = null;
    }
  }, [
    topic,
    running,
    model,
    style,
    maxIterations,
    qualityThreshold,
    useWebSearch,
    context,
    userKey,
  ]);

  const stop = () => {
    userStoppedRef.current = true;
    abortRef.current?.abort();
  };

  const downloadMd = () => {
    if (!report) return;
    const safe =
      topic
        .slice(0, 40)
        .replace(/[^a-z0-9 -]/gi, "_")
        .trim()
        .replace(/\s+/g, "_")
        .toLowerCase() || "report";
    const blob = new Blob([report], { type: "text/markdown;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `report_${safe}.md`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const onTopicKey = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) run();
  };

  const anyActivity = running || report || agents.orchestrator.status !== "idle";

  return (
    <div className="wrap">
      {/* Header */}
      <header className="header">
        <div className="brand">
          <div className="logo">🔬</div>
          <div>
            <h1>Multi-Agent Research System</h1>
            <p>Five specialised AI agents research, analyse, write &amp; critique — live.</p>
          </div>
        </div>
        <span className={`badge ${serverInfo?.serverKey || userKey ? "green" : ""}`}>
          Groq{serverInfo?.webSearch ? " · Tavily search" : ""}
        </span>
      </header>

      {/* Input card */}
      <div className="card" style={{ marginTop: 18 }}>
        <label className="field" htmlFor="topic">
          Research topic
        </label>
        <div className="topic-row">
          <textarea
            id="topic"
            rows={1}
            placeholder="e.g. Impact of AI on healthcare diagnostics"
            value={topic}
            onChange={(e) => setTopic(e.target.value)}
            onKeyDown={onTopicKey}
            disabled={running}
          />
          {running ? (
            <button className="btn-ghost" onClick={stop} style={{ minWidth: 110 }}>
              ■ Stop
            </button>
          ) : (
            <button className="btn-primary" onClick={run} disabled={!topic.trim() || keyMissing}>
              Run research
            </button>
          )}
        </div>

        {!running && !anyActivity && (
          <div className="examples">
            {EXAMPLES.map((ex) => (
              <span key={ex} className="chip" onClick={() => setTopic(ex)}>
                {ex}
              </span>
            ))}
          </div>
        )}

        {keyMissing && (
          <p className="hint" style={{ color: "var(--amber)" }}>
            No Groq key configured on the server — add one in Settings below, or set{" "}
            <code>GROQ_API_KEY</code> on Vercel. Free key:{" "}
            <a href="https://console.groq.com/keys" target="_blank" rel="noreferrer">
              console.groq.com/keys
            </a>
          </p>
        )}

        <button className="settings-toggle" onClick={() => setShowSettings((s) => !s)}>
          {showSettings ? "▾ Hide settings" : "▸ Settings (model, style, web search, key)"}
        </button>

        {showSettings && (
          <div className="grid">
            <div>
              <label className="field">Model</label>
              <select value={model} onChange={(e) => setModel(e.target.value)} disabled={running}>
                {(serverInfo?.models ?? ["llama-3.1-8b-instant", "llama-3.3-70b-versatile"]).map(
                  (m) => (
                    <option key={m} value={m}>
                      {m}
                      {m.includes("8b") ? "  (fast, 500k/day)" : "  (higher quality)"}
                    </option>
                  )
                )}
              </select>
              {(model.includes("70b") || maxIterations > 1) && (
                <p className="hint" style={{ color: "var(--amber)" }}>
                  70B and/or multiple rounds are slower — on Vercel&apos;s free tier
                  (60s limit) long runs can get cut off. 8B + 1 round is the safe
                  default.
                </p>
              )}
            </div>

            <div>
              <label className="field">Response style</label>
              <select
                value={style}
                onChange={(e) => setStyle(e.target.value as any)}
                disabled={running}
              >
                <option value="precise">Precise</option>
                <option value="balanced">Balanced</option>
                <option value="creative">Creative</option>
              </select>
            </div>

            <div>
              <label className="field">
                Max revision rounds <span className="range-val">{maxIterations}</span>
              </label>
              <input
                type="range"
                min={1}
                max={3}
                value={maxIterations}
                onChange={(e) => setMaxIterations(Number(e.target.value))}
                disabled={running}
              />
              <p className="hint">More rounds = higher quality but slower (and more tokens).</p>
            </div>

            <div>
              <label className="field">
                Quality bar (Critic score to pass){" "}
                <span className="range-val">{qualityThreshold}/10</span>
              </label>
              <input
                type="range"
                min={1}
                max={10}
                value={qualityThreshold}
                onChange={(e) => setQualityThreshold(Number(e.target.value))}
                disabled={running}
              />
            </div>

            <div>
              <label className="field">Web search</label>
              <label className="switch">
                <input
                  type="checkbox"
                  checked={useWebSearch}
                  onChange={(e) => setUseWebSearch(e.target.checked)}
                  disabled={running}
                />
                <span>
                  {serverInfo && !serverInfo.webSearch
                    ? "Needs a TAVILY_API_KEY on the server"
                    : "Use real web search (Tavily)"}
                </span>
              </label>
            </div>

            <div>
              <label className="field">Your Groq API key (optional)</label>
              <input
                type="password"
                placeholder={serverInfo?.serverKey ? "Using server key" : "gsk_..."}
                value={userKey}
                onChange={(e) => setUserKey(e.target.value)}
                disabled={running}
                autoComplete="off"
              />
              <p className="hint">Stays in your browser; sent only to run your request.</p>
            </div>

            <div style={{ gridColumn: "1 / -1" }}>
              <label className="field">Reference notes (optional — grounds the research)</label>
              <textarea
                rows={3}
                placeholder="Paste any background text, notes, or excerpts you want the agents to use…"
                value={context}
                onChange={(e) => setContext(e.target.value)}
                disabled={running}
              />
            </div>
          </div>
        )}
      </div>

      {/* Error */}
      {error && (
        <div className={`error ${error.daily ? "daily" : ""}`} style={{ marginTop: 18 }}>
          {error.message}
        </div>
      )}

      {/* Progress */}
      {anyActivity && (
        <div className="card" style={{ marginTop: 18 }}>
          <div
            style={{
              display: "flex",
              justifyContent: "space-between",
              alignItems: "center",
              marginBottom: 14,
            }}
          >
            <p className="section-title" style={{ margin: 0 }}>
              {report ? "Pipeline complete" : "Agents at work"}
            </p>
            <span className="timer">{elapsed.toFixed(1)}s</span>
          </div>

          <div className="stepper">
            {STEPS.map((s) => {
              const cls = doneSteps.has(s.key)
                ? "done"
                : currentStep === s.key
                  ? "active"
                  : "";
              return (
                <span key={s.key} className={`step ${cls}`}>
                  <span className="dot" />
                  {s.label}
                </span>
              );
            })}
          </div>

          {statusLine && (
            <p className="hint" style={{ marginTop: 12, marginBottom: 4 }}>
              {statusLine}
            </p>
          )}

          {/* Agent stream cards */}
          <div style={{ marginTop: 16 }}>
            {AGENT_ORDER.map((k) => {
              const a = agents[k];
              if (a.status === "idle" && !(k === "critic" && score)) return null;
              const meta = AGENT_META[k];
              const open = openAgent === k;
              return (
                <div className="agent" key={k}>
                  <div
                    className="agent-head"
                    onClick={() => setOpenAgent(open ? null : k)}
                  >
                    <span className="agent-emoji">{meta.emoji}</span>
                    <div>
                      <div className="agent-name">{meta.name}</div>
                    </div>
                    <div className="agent-status">
                      {k === "critic" && score && (
                        <span className={`score-pill ${score.value >= qualityThreshold ? "good" : ""}`}>
                          {score.value}/10
                        </span>
                      )}
                      {a.status === "streaming" && <span className="spinner" />}
                      {a.status === "done" && <span style={{ color: "var(--green)" }}>✓</span>}
                      {a.status === "idle" && <span>{meta.role}</span>}
                      <span className={`caret ${open ? "open" : ""}`}>›</span>
                    </div>
                  </div>
                  {open && (
                    <AgentBody text={a.text} streaming={a.status === "streaming"} role={meta.role} />
                  )}
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* Final report */}
      {report && (
        <div className="card print-area" style={{ marginTop: 18 }}>
          <div className="report-actions">
            <button className="btn-ghost" onClick={downloadMd}>
              ⬇ Download Markdown
            </button>
            <button className="btn-ghost" onClick={() => window.print()}>
              🖨 Print / Save as PDF
            </button>
            {citations.length > 0 && (
              <span className="badge" style={{ alignSelf: "center" }}>
                {citations.length} source{citations.length === 1 ? "" : "s"}
              </span>
            )}
          </div>
          <div className="report">
            <ReactMarkdown remarkPlugins={[remarkGfm]}>{report}</ReactMarkdown>
          </div>
        </div>
      )}

      <p className="footer">
        Same five-agent pipeline as the Streamlit app — Orchestrator · Researcher · Analyst ·
        Writer · Critic. Running on Groq + Vercel.
      </p>
    </div>
  );
}

// Auto-scrolling agent output panel.
function AgentBody({
  text,
  streaming,
  role,
}: {
  text: string;
  streaming: boolean;
  role: string;
}) {
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (streaming && ref.current) ref.current.scrollTop = ref.current.scrollHeight;
  }, [text, streaming]);
  return (
    <div className="agent-body" ref={ref}>
      {text || (streaming ? "…" : <span style={{ color: "var(--faint)" }}>{role}</span>)}
    </div>
  );
}
