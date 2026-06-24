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

// Shown when a single step stalls with no response (a real hang, not pacing).
const TIMEOUT_MSG =
  "A step stopped responding and was cancelled. This is usually a transient " +
  "Groq hiccup — try again. If it persists, switch to llama-3.1-8b-instant " +
  "and keep the topic focused.";

// ── Client-side rate pacing ──────────────────────────────────────────────────
// Groq's free tier caps llama-3.1-8b-instant at 6,000 tokens/minute. We run one
// agent per request and throttle between requests so the rolling 60-second
// token total stays under budget — no 429s, no need to pay, and (because the
// waits happen here, not inside a serverless function) no 60s timeout.
const PACE_BUDGET = 5200; // tokens per rolling 60s window (margin under 6,000)
const PACE_WINDOW = 60000;
const PACE_EST = 2400; // assumed tokens for the next step, used for the pre-check
const STALL_MS = 45000; // a single step silent this long => cancel it

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

interface RunState {
  plan: string;
  findings: string;
  citations: string[];
  analysis: string;
  draft: string;
  critique: string;
  iteration: number;
  score: number;
}
const emptyRunState = (): RunState => ({
  plan: "",
  findings: "",
  citations: [],
  analysis: "",
  draft: "",
  critique: "",
  iteration: 0,
  score: 0,
});

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
  const tokenLogRef = useRef<{ t: number; tokens: number }[]>([]);

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

  // ── Run the pipeline: one agent per request, paced on the client ──────────--
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
    setStatusLine("Starting…");
    setOpenAgent(null);
    setElapsed(0);
    startRef.current = Date.now();
    tokenLogRef.current = [];
    userStoppedRef.current = false;
    setRunning(true);

    const ctrl = new AbortController();
    abortRef.current = ctrl;

    let lastActivity = Date.now();
    let stalled = false;
    let handledError = false;

    const cfgBase = {
      topic: t,
      model,
      style,
      useWebSearch,
      context: context.trim() || undefined,
      apiKey: userKey.trim() || undefined,
    };

    const advanceTo = (step: string) => {
      const idx = STEP_ORDER.indexOf(step);
      if (idx === -1) return;
      setDoneSteps((prev) => {
        const next = new Set(prev);
        for (let i = 0; i < idx; i++) next.add(STEP_ORDER[i]);
        return next;
      });
      setCurrentStep(step);
    };

    // Watchdog: cancel the run only if the *active* step goes silent. Pacing
    // waits refresh lastActivity, so intentional pauses don't trip it.
    const watchdog = window.setInterval(() => {
      if (Date.now() - lastActivity > STALL_MS) {
        stalled = true;
        ctrl.abort();
      }
    }, 3000);

    // Throttle before a step so the rolling 60s token total stays under budget.
    const paceFor = async () => {
      while (true) {
        if (userStoppedRef.current) throw new DOMException("aborted", "AbortError");
        const now = Date.now();
        tokenLogRef.current = tokenLogRef.current.filter((e) => now - e.t < PACE_WINDOW);
        const sum = tokenLogRef.current.reduce((a, e) => a + e.tokens, 0);
        if (tokenLogRef.current.length === 0 || sum + PACE_EST <= PACE_BUDGET) break;
        const oldest = tokenLogRef.current[0];
        const waitMs = Math.min(PACE_WINDOW - (now - oldest.t) + 800, 20000);
        for (let waited = 0; waited < waitMs; waited += 1000) {
          if (userStoppedRef.current) throw new DOMException("aborted", "AbortError");
          lastActivity = Date.now(); // pacing is intentional — keep the watchdog happy
          setStatusLine(
            `Pacing to stay within the free Groq rate limit… ${Math.ceil(
              (waitMs - waited) / 1000
            )}s`
          );
          await sleep(1000);
        }
      }
    };

    // Run one step: POST, stream its events into the UI, return the new state.
    const runOneStep = async (
      step: string,
      state: RunState
    ): Promise<{ state: RunState; markdown?: string }> => {
      const res = await fetch("/api/research", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        signal: ctrl.signal,
        body: JSON.stringify({ ...cfgBase, step, state }),
      });
      if (!res.ok || !res.body) {
        const data = await res.json().catch(() => ({}));
        throw new Error(data.error || `Request failed (${res.status})`);
      }
      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      let result: { state: RunState; markdown?: string } | null = null;

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop() ?? "";
        for (const line of lines) {
          const sLine = line.trim();
          if (!sLine) continue;
          let e: any;
          try {
            e = JSON.parse(sLine);
          } catch {
            continue;
          }
          lastActivity = Date.now();
          switch (e.type) {
            case "step":
              setStatusLine(e.message);
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
            case "result":
              result = { state: e.state as RunState, markdown: e.markdown };
              if (typeof e.tokensUsed === "number" && e.tokensUsed > 0) {
                tokenLogRef.current.push({ t: Date.now(), tokens: e.tokensUsed });
              }
              break;
            case "error":
              handledError = true;
              setError({ message: e.message, daily: e.daily });
              throw new Error(e.message);
          }
        }
      }
      if (!result) throw new Error("Step ended without a result.");
      return result;
    };

    try {
      let state = emptyRunState();

      advanceTo("plan");
      state = (await runOneStep("plan", state)).state;

      await paceFor();
      advanceTo("research");
      state = (await runOneStep("research", state)).state;

      await paceFor();
      advanceTo("analyze");
      state = (await runOneStep("analyze", state)).state;

      await paceFor();
      advanceTo("write");
      state = (await runOneStep("write", state)).state;

      // Critique → revise loop (client controls the rounds).
      while (true) {
        await paceFor();
        advanceTo("review");
        state = (await runOneStep("review", state)).state;
        const passed = state.score >= qualityThreshold;
        if (state.iteration >= maxIterations || passed) break;
        await paceFor();
        advanceTo("write");
        state = (await runOneStep("revise", state)).state;
      }

      await paceFor();
      advanceTo("finalize");
      const fin = await runOneStep("finalize", state);
      state = fin.state;

      if (fin.markdown) setReport(fin.markdown);
      setCitations(state.citations || []);
      setDoneSteps(new Set(STEP_ORDER));
      setCurrentStep(null);
      setOpenAgent(null);
      setStatusLine(`Done in ${((Date.now() - startRef.current) / 1000).toFixed(1)}s`);
    } catch (err: any) {
      if (err?.name === "AbortError") {
        // Watchdog stall => show timeout. User pressed Stop => stay silent.
        if (stalled && !userStoppedRef.current) setError({ message: TIMEOUT_MSG });
      } else if (!handledError) {
        setError({ message: err?.message || "Something went wrong." });
      }
    } finally {
      window.clearInterval(watchdog);
      // Stop any agent left mid-stream so its card doesn't spin forever.
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
