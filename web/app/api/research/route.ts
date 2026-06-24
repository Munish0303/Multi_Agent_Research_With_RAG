/**
 * POST /api/research
 *
 * Runs ONE agent step of the pipeline and streams it back as newline-delimited
 * JSON (NDJSON). The browser calls this once per step — plan → research →
 * analyze → write → review → (revise → review)* → finalize — carrying the full
 * `state` between calls and pacing the calls to stay under Groq's per-minute
 * token limit. Keeping one agent per request means each request finishes well
 * inside the serverless time limit, and the pacing waits happen on the client
 * (not inside a function), so the free tier never trips the 60s cap.
 */

import { NextRequest } from "next/server";
import {
  runStep,
  RunConfig,
  PipelineEvent,
  PipelineState,
  StepName,
  emptyState,
} from "@/lib/pipeline";
import { AVAILABLE_MODELS, DEFAULT_MODEL, DailyLimitError } from "@/lib/groq";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";
export const maxDuration = 60;

const STEPS: StepName[] = [
  "plan",
  "research",
  "analyze",
  "write",
  "review",
  "revise",
  "finalize",
];

interface Body {
  step?: string;
  state?: unknown;
  topic?: string;
  model?: string;
  style?: string;
  useWebSearch?: boolean;
  context?: string;
  apiKey?: string; // optional visitor-supplied key (takes precedence)
}

/** Coerce a client-supplied state blob into a safe PipelineState. */
function sanitizeState(raw: unknown): PipelineState {
  const s = emptyState();
  if (raw && typeof raw === "object") {
    const r = raw as Record<string, unknown>;
    const str = (v: unknown) => (typeof v === "string" ? v.slice(0, 40000) : "");
    s.plan = str(r.plan);
    s.findings = str(r.findings);
    s.analysis = str(r.analysis);
    s.draft = str(r.draft);
    s.critique = str(r.critique);
    if (Array.isArray(r.citations)) {
      s.citations = r.citations.filter((x): x is string => typeof x === "string").slice(0, 50);
    }
    if (Number.isFinite(r.iteration)) s.iteration = Math.max(0, Math.floor(r.iteration as number));
    if (Number.isFinite(r.score)) s.score = Math.max(0, Math.floor(r.score as number));
  }
  return s;
}

export async function POST(req: NextRequest) {
  let body: Body;
  try {
    body = (await req.json()) as Body;
  } catch {
    return Response.json({ error: "Invalid JSON body." }, { status: 400 });
  }

  const step = body.step as StepName;
  if (!STEPS.includes(step)) {
    return Response.json(
      { error: `Invalid step. Expected one of: ${STEPS.join(", ")}` },
      { status: 400 }
    );
  }

  const topic = (body.topic ?? "").trim();
  if (!topic) {
    return Response.json({ error: "A research topic is required." }, { status: 400 });
  }
  if (topic.length > 500) {
    return Response.json({ error: "Topic is too long (max 500 characters)." }, { status: 400 });
  }

  // Key precedence: visitor-supplied key → server env var.
  const apiKey = (body.apiKey?.trim() || process.env.GROQ_API_KEY || "").trim();
  if (!apiKey) {
    return Response.json(
      {
        error:
          "No Groq API key available. Set GROQ_API_KEY in the deployment " +
          "environment, or enter your own key in the UI. Get a free key at " +
          "https://console.groq.com/keys",
      },
      { status: 400 }
    );
  }

  const model =
    body.model && (AVAILABLE_MODELS as readonly string[]).includes(body.model)
      ? body.model
      : process.env.GROQ_MODEL || DEFAULT_MODEL;

  const style =
    body.style === "precise" || body.style === "creative" ? body.style : "balanced";

  const useWebSearch = body.useWebSearch !== false;
  const tavilyKey = process.env.TAVILY_API_KEY?.trim() || undefined;

  const cfg: RunConfig = {
    topic,
    apiKey,
    model,
    style,
    // Loop control lives on the client now; these are unused by runStep.
    maxIterations: 1,
    qualityThreshold: 7,
    useWebSearch,
    tavilyKey,
    context: typeof body.context === "string" ? body.context.slice(0, 12000) : undefined,
  };

  const state = sanitizeState(body.state);

  const encoder = new TextEncoder();
  const stream = new ReadableStream<Uint8Array>({
    async start(controller) {
      const emit = (e: PipelineEvent) => {
        try {
          controller.enqueue(encoder.encode(JSON.stringify(e) + "\n"));
        } catch {
          // controller already closed (client disconnected) — ignore
        }
      };

      // On the first step, tell the client whether real web search is active.
      if (step === "plan") {
        emit({
          type: "step",
          step: "init",
          message:
            useWebSearch && tavilyKey
              ? "Web search: Tavily enabled"
              : useWebSearch
                ? "Web search: no key configured — using model knowledge only"
                : "Web search: disabled",
        });
      }

      try {
        const result = await runStep(step, cfg, state, emit);
        emit({
          type: "result",
          step,
          tokensUsed: result.tokensUsed,
          state: result.state,
          report: result.report,
          markdown: result.markdown,
        });
      } catch (err) {
        if (err instanceof DailyLimitError) {
          emit({ type: "error", message: err.message, daily: true });
        } else {
          emit({
            type: "error",
            message: err instanceof Error ? err.message : String(err),
          });
        }
      } finally {
        controller.close();
      }
    },
  });

  return new Response(stream, {
    headers: {
      "Content-Type": "application/x-ndjson; charset=utf-8",
      "Cache-Control": "no-cache, no-transform",
      Connection: "keep-alive",
      "X-Accel-Buffering": "no",
    },
  });
}

// A tiny GET so visiting the route in a browser doesn't 405-confuse people,
// and so the UI can discover server capabilities on load.
export async function GET() {
  return Response.json({
    ok: true,
    usage: "POST { step, state, topic, model?, style?, useWebSearch?, context?, apiKey? }",
    steps: STEPS,
    models: AVAILABLE_MODELS,
    webSearch: Boolean(process.env.TAVILY_API_KEY),
    serverKey: Boolean(process.env.GROQ_API_KEY),
  });
}
