/**
 * The multi-agent research pipeline, ported faithfully from the Python
 * LangGraph workflow (workflow.py + agents/*.py):
 *
 *   Orchestrator (plan) → Researcher → Analyst → Writer → Critic
 *        → [score < bar & rounds left] → revise → Critic
 *        → Orchestrator (finalize)
 *
 * Agent system prompts and base temperatures match the Python agents exactly.
 * Inter-agent text is clipped to the same char budgets to bound token usage.
 */

import { streamChat, ChatMessage, DailyLimitError } from "./groq";
import { tavilySearch, formatResults, SearchResult } from "./search";

// ── Event protocol emitted to the client (NDJSON) ────────────────────────────

export type PipelineEvent =
  | { type: "step"; step: string; message: string }
  | { type: "agent_start"; agent: AgentKey; label: string }
  | { type: "token"; agent: AgentKey; text: string }
  | { type: "agent_done"; agent: AgentKey }
  | { type: "citations"; urls: string[] }
  | { type: "score"; score: number; iteration: number }
  | {
      type: "final";
      report: string;
      markdown: string;
      citations: string[];
      durationMs: number;
    }
  | { type: "error"; message: string; daily?: boolean };

export type Emit = (e: PipelineEvent) => void | Promise<void>;

export type AgentKey =
  | "orchestrator"
  | "researcher"
  | "analyst"
  | "writer"
  | "critic";

export interface RunConfig {
  topic: string;
  apiKey: string;
  model: string;
  /** "precise" | "balanced" | "creative" — shifts every agent's temperature */
  style: "precise" | "balanced" | "creative";
  maxIterations: number; // max critique→revise rounds
  qualityThreshold: number; // critic score (1-10) needed to skip revision
  useWebSearch: boolean;
  tavilyKey?: string;
  /** Optional pasted reference notes (lightweight RAG substitute). */
  context?: string;
}

// ── System prompts (verbatim from agents/*.py) ───────────────────────────────

const PROMPTS: Record<AgentKey, string> = {
  orchestrator: `You are the Research Director overseeing a multi-agent research team.
Your agents: Researcher, Analyst, Writer, Critic.

Given a research topic, you:
1. Break it into clear sub-tasks for each agent
2. Synthesize outputs from all agents
3. Make final decisions on conflicting information
4. Ensure the final output meets quality standards
5. Provide a concise task brief for each agent

Be directive and clear. Delegate effectively. Summarize what each agent should focus on.`,

  researcher: `You are an expert Research Specialist. Your job is to:
1. Gather comprehensive information on the given topic
2. Search the web for current and relevant data
3. Read and analyze documents from the knowledge base
4. Extract key facts, statistics, and insights
5. Identify gaps in existing knowledge

Always cite your sources. Be thorough but concise.
Return structured findings with clear sections: Background, Key Findings, Data Points, Sources.`,

  analyst: `You are a rigorous Data & Logic Analyst. Your job is to:
1. Analyze research findings critically
2. Identify patterns, trends, and correlations
3. Run calculations or data analysis when needed
4. Evaluate the strength of evidence
5. Spot contradictions, biases, or weak arguments
6. Quantify findings where possible

Be critical and objective. Always explain your reasoning.
Return structured analysis: Key Patterns, Evidence Quality, Gaps, Quantitative Insights, Conclusions.`,

  writer: `You are an expert Research Writer. Your job is to:
1. Synthesize research findings and analysis into clear, coherent reports
2. Write with precision, clarity, and appropriate depth
3. Structure content with proper headings and flow
4. Highlight the most important insights prominently
5. Make complex information accessible without oversimplifying
6. Include executive summaries for long reports

Write in a professional but readable style.
Always structure your output as: Executive Summary, Introduction, Main Findings, Analysis, Conclusions, Recommendations.`,

  critic: `You are a rigorous Quality Reviewer and Devil's Advocate. Your job is to:
1. Review the draft report for accuracy and completeness
2. Challenge assumptions and weak arguments
3. Identify missing information or perspectives
4. Check logical consistency
5. Suggest specific improvements
6. Rate the overall quality (1-10) with justification

Be constructively critical. Do not simply praise.
Return your response in this exact format:
- Strengths: [list strengths]
- Critical Issues: [list issues]
- Missing Elements: [list gaps]
- Specific Suggestions: [list improvements]
- SCORE:[number] (e.g. SCORE:7)

The SCORE line must appear exactly as shown at the end of your response.`,
};

// Base temperatures per agent (from each agent's __init__ in agents/*.py)
const BASE_TEMP: Record<AgentKey, number> = {
  orchestrator: 0.3,
  researcher: 0.2,
  analyst: 0.1,
  writer: 0.5,
  critic: 0.2,
};

const STYLE_OFFSET = { precise: -0.1, balanced: 0.0, creative: 0.2 } as const;

const LABELS: Record<AgentKey, string> = {
  orchestrator: "Orchestrator",
  researcher: "Researcher",
  analyst: "Analyst",
  writer: "Writer",
  critic: "Critic",
};

// ── Helpers (ported from workflow.py) ────────────────────────────────────────

/** Truncate inter-agent text to bound token usage (~4 chars/token). */
function clip(text: string, maxChars: number): string {
  if (text && text.length > maxChars) {
    return text.slice(0, maxChars) + "\n\n[...truncated for length...]";
  }
  return text || "";
}

/** Pull unique, clean URLs from text. */
function extractUrls(text: string): string[] {
  const raw = text.match(/https?:\/\/[^\s)\]>"']+/g) ?? [];
  const seen = new Set<string>();
  const out: string[] = [];
  for (const u of raw) {
    const cleaned = u.replace(/[.,;:)\]>"']+$/, "");
    if (!seen.has(cleaned)) {
      seen.add(cleaned);
      out.push(cleaned);
    }
  }
  return out;
}

function tempFor(agent: AgentKey, style: RunConfig["style"]): number {
  const t = BASE_TEMP[agent] + STYLE_OFFSET[style];
  return Math.max(0, Math.min(1, t));
}

// ── Core: run one agent turn, streaming its tokens to the client ─────────────

async function runAgent(
  agent: AgentKey,
  task: string,
  cfg: RunConfig,
  emit: Emit,
  opts: { extraContext?: string; maxTokens?: number } = {}
): Promise<string> {
  await emit({ type: "agent_start", agent, label: LABELS[agent] });

  let userContent = task;
  if (opts.extraContext) {
    userContent = `${opts.extraContext}\n\nTask: ${task}`;
  }

  const messages: ChatMessage[] = [
    { role: "system", content: PROMPTS[agent] },
    { role: "user", content: userContent },
  ];

  let out = "";
  for await (const delta of streamChat({
    apiKey: cfg.apiKey,
    model: cfg.model,
    temperature: tempFor(agent, cfg.style),
    messages,
    maxTokens: opts.maxTokens ?? 2048,
  })) {
    out += delta;
    await emit({ type: "token", agent, text: delta });
  }

  await emit({ type: "agent_done", agent });
  return out.trim();
}

// ── The pipeline ─────────────────────────────────────────────────────────────

export async function runResearch(cfg: RunConfig, emit: Emit): Promise<void> {
  const start = Date.now();

  // Build the optional reference-documents block once (lightweight RAG).
  const refBlock = cfg.context?.trim()
    ? `<reference_documents>\n${clip(cfg.context.trim(), 6000)}\n</reference_documents>`
    : "";

  try {
    // ── 1. Plan ────────────────────────────────────────────────────────────
    await emit({
      step: "plan",
      type: "step",
      message: "Orchestrator: Planning research strategy...",
    });
    const planTask = `Research topic: ${cfg.topic}

Create a specific task brief for each agent:
- Researcher: What to research and gather
- Analyst: What to analyze and evaluate
- Writer: How to structure and present the report
- Critic: What quality standards to enforce

Return as a structured plan.`;
    const plan = await runAgent("orchestrator", planTask, cfg, emit, {
      extraContext: refBlock || undefined,
      maxTokens: 700,
    });

    // ── 2. Research (optional real web search via Tavily) ───────────────────
    await emit({
      step: "research",
      type: "step",
      message: "Researcher: Gathering information from the web...",
    });

    let webResults: SearchResult[] = [];
    let webBlock = "";
    if (cfg.useWebSearch && cfg.tavilyKey) {
      webResults = await tavilySearch(cfg.topic, cfg.tavilyKey, 6);
      if (webResults.length) {
        webBlock = `<web_search_results>\n${formatResults(webResults)}\n</web_search_results>`;
      }
    }

    const researchContext = [refBlock, webBlock].filter(Boolean).join("\n\n");
    const researchTask = `Topic: ${cfg.topic}
Plan context: ${clip(plan, 500)}

Research this topic thoroughly. ${
      webBlock
        ? "Use the web search results provided above and analyze any reference documents."
        : "Analyze available documents and draw on your knowledge."
    } Extract key facts, statistics, and insights, and cite sources with their URLs.`;

    const findings = await runAgent("researcher", researchTask, cfg, emit, {
      extraContext: researchContext || undefined,
      maxTokens: 1400,
    });

    // Citations = URLs from search results + any URLs the model surfaced.
    const citations = Array.from(
      new Set([...webResults.map((r) => r.url).filter(Boolean), ...extractUrls(findings)])
    );
    await emit({ type: "citations", urls: citations });

    // ── 3. Analyze ──────────────────────────────────────────────────────────
    await emit({
      step: "analyze",
      type: "step",
      message: "Analyst: Identifying patterns and evaluating evidence...",
    });
    const analysisTask = `Topic: ${cfg.topic}
Plan context: ${clip(plan, 300)}

Research Findings:
${clip(findings, 4000)}

Analyze these findings critically. Identify patterns, evaluate evidence quality, and derive insights.`;
    const analysis = await runAgent("analyst", analysisTask, cfg, emit, {
      maxTokens: 1400,
    });

    // ── 4. Write ──────────────────────────────────────────────────────────--
    await emit({
      step: "write",
      type: "step",
      message: "Writer: Drafting the research report...",
    });
    const writeTask = `Topic: ${cfg.topic}
Plan context: ${clip(plan, 300)}

Research Findings:
${clip(findings, 4000)}

Analysis:
${clip(analysis, 3000)}

Write a comprehensive, well-structured research report. Include all key insights.`;
    let draft = await runAgent("writer", writeTask, cfg, emit, { maxTokens: 1800 });

    // ── 5. Critique → revise loop ────────────────────────────────────────────
    let iteration = 0;
    let critique = "";
    while (true) {
      await emit({
        step: "review",
        type: "step",
        message: "Critic: Reviewing and scoring the draft...",
      });
      const critiqueTask = `Topic: ${cfg.topic}

Draft Report:
${clip(draft, 5000)}

Review this report critically. Evaluate quality, completeness, and accuracy.`;
      critique = await runAgent("critic", critiqueTask, cfg, emit, {
        maxTokens: 900,
      });
      iteration += 1;

      const m = critique.match(/SCORE:\s*(\d+)/i);
      const score = m ? parseInt(m[1], 10) : NaN;
      await emit({
        type: "score",
        score: Number.isNaN(score) ? 0 : score,
        iteration,
      });

      const passed = !Number.isNaN(score) && score >= cfg.qualityThreshold;
      if (iteration >= cfg.maxIterations || passed) break;

      // Revise
      await emit({
        step: "revise",
        type: "step",
        message: "Writer: Revising report based on critique...",
      });
      const reviseTask = `Topic: ${cfg.topic}

Original Draft:
${clip(draft, 5000)}

Critique received:
${clip(critique, 2000)}

Rewrite the report addressing all critique points. Significantly improve the identified weaknesses.`;
      draft = await runAgent("writer", reviseTask, cfg, emit, { maxTokens: 1800 });
    }

    // ── 6. Finalize ──────────────────────────────────────────────────────────
    await emit({
      step: "finalize",
      type: "step",
      message: "Orchestrator: Synthesizing final report...",
    });
    const finalizeTask = `Topic: ${cfg.topic}

Draft Report:
${clip(draft, 6000)}

Critique:
${clip(critique, 2000)}

Synthesize the draft and critique into a final, polished research report. Incorporate all feedback, resolve any gaps, and add an executive summary.`;
    const final = await runAgent("orchestrator", finalizeTask, cfg, emit, {
      maxTokens: 2200,
    });

    // ── Build the final markdown (matches workflow.py finalize_node) ─────────
    let refs = "";
    if (citations.length) {
      refs =
        "\n\n## References\n\n" +
        citations.map((u, i) => `${i + 1}. ${u}`).join("\n");
    }
    const markdown = `# Research Report: ${cfg.topic}\n\n${final}${refs}\n\n---\n*Generated by the Multi-Agent Research System*`;

    await emit({
      type: "final",
      report: final,
      markdown,
      citations,
      durationMs: Date.now() - start,
    });
  } catch (err) {
    if (err instanceof DailyLimitError) {
      await emit({ type: "error", message: err.message, daily: true });
    } else {
      await emit({
        type: "error",
        message: err instanceof Error ? err.message : String(err),
      });
    }
  }
}
