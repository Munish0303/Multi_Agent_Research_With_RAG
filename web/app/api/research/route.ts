/**
 * POST /api/research
 *
 * Streams the multi-agent research pipeline back to the browser as
 * newline-delimited JSON (NDJSON). Each line is one PipelineEvent.
 *
 * Runs on the Node.js serverless runtime so the response can stream for the
 * full duration of a run (see maxDuration / vercel.json).
 */

import { NextRequest } from "next/server";
import { runResearch, RunConfig, PipelineEvent } from "@/lib/pipeline";
import { AVAILABLE_MODELS, DEFAULT_MODEL } from "@/lib/groq";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";
export const maxDuration = 60;

interface Body {
  topic?: string;
  model?: string;
  style?: string;
  maxIterations?: number;
  qualityThreshold?: number;
  useWebSearch?: boolean;
  context?: string;
  apiKey?: string; // optional visitor-supplied key (takes precedence)
}

const clampInt = (v: unknown, lo: number, hi: number, dflt: number): number => {
  const n = Math.floor(Number(v));
  if (Number.isNaN(n)) return dflt;
  return Math.max(lo, Math.min(hi, n));
};

export async function POST(req: NextRequest) {
  let body: Body;
  try {
    body = (await req.json()) as Body;
  } catch {
    return Response.json({ error: "Invalid JSON body." }, { status: 400 });
  }

  const topic = (body.topic ?? "").trim();
  if (!topic) {
    return Response.json({ error: "A research topic is required." }, { status: 400 });
  }
  if (topic.length > 500) {
    return Response.json(
      { error: "Topic is too long (max 500 characters)." },
      { status: 400 }
    );
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
    body.style === "precise" || body.style === "creative"
      ? body.style
      : "balanced";

  const useWebSearch = body.useWebSearch !== false;
  const tavilyKey = process.env.TAVILY_API_KEY?.trim() || undefined;

  const cfg: RunConfig = {
    topic,
    apiKey,
    model,
    style,
    maxIterations: clampInt(body.maxIterations, 1, 3, 1),
    qualityThreshold: clampInt(body.qualityThreshold, 1, 10, 7),
    useWebSearch,
    tavilyKey,
    context: typeof body.context === "string" ? body.context.slice(0, 12000) : undefined,
  };

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

      // Tell the client up front whether real web search is active.
      emit({
        type: "step",
        step: "init",
        message: useWebSearch && tavilyKey
          ? "Web search: Tavily enabled"
          : useWebSearch
            ? "Web search: no key configured — using model knowledge only"
            : "Web search: disabled",
      });

      await runResearch(cfg, emit);
      controller.close();
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

// A tiny GET so visiting the route in a browser doesn't 405-confuse people.
export async function GET() {
  return Response.json({
    ok: true,
    usage: "POST { topic, model?, style?, maxIterations?, qualityThreshold?, useWebSearch?, context?, apiKey? }",
    models: AVAILABLE_MODELS,
    webSearch: Boolean(process.env.TAVILY_API_KEY),
    serverKey: Boolean(process.env.GROQ_API_KEY),
  });
}
