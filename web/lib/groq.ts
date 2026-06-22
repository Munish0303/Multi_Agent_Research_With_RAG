/**
 * Minimal streaming client for Groq's OpenAI-compatible chat API.
 *
 * Mirrors the Python side (config/providers.py + agents/base.py):
 *   - same models (llama-3.1-8b-instant, llama-3.3-70b-versatile)
 *   - per-minute rate limits (429) are retried with backoff
 *   - per-day token limits fail fast with a clear, actionable message
 */

const GROQ_URL = "https://api.groq.com/openai/v1/chat/completions";

export const AVAILABLE_MODELS = [
  "llama-3.1-8b-instant", // 500k tokens/DAY — best for repeated demo use
  "llama-3.3-70b-versatile", // 100k tokens/day — higher quality, fewer runs
] as const;

export const DEFAULT_MODEL = "llama-3.1-8b-instant";

export class DailyLimitError extends Error {
  constructor() {
    super(
      "The free-tier daily token limit for this model has been reached. " +
        "Switch to a different model (e.g. llama-3.1-8b-instant), wait for the " +
        "daily reset, or upgrade your Groq plan."
    );
    this.name = "DailyLimitError";
  }
}

export interface ChatMessage {
  role: "system" | "user" | "assistant";
  content: string;
}

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

function isDailyLimit(msg: string): boolean {
  const low = msg.toLowerCase();
  return (
    low.includes("per day") ||
    low.includes("tpd") ||
    /try again in\s+\d+\s*m/.test(low)
  );
}

/**
 * Stream a chat completion from Groq, yielding text deltas as they arrive.
 * Retries per-minute rate limits; throws DailyLimitError on per-day limits.
 */
export async function* streamChat(opts: {
  apiKey: string;
  model: string;
  temperature: number;
  messages: ChatMessage[];
  maxTokens?: number;
  maxRetries?: number;
}): AsyncGenerator<string, void, unknown> {
  const { apiKey, model, temperature, messages, maxTokens = 2048 } = opts;
  const maxRetries = opts.maxRetries ?? 3;

  for (let attempt = 0; attempt < maxRetries; attempt++) {
    let res: Response;
    try {
      res = await fetch(GROQ_URL, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${apiKey}`,
        },
        body: JSON.stringify({
          model,
          temperature,
          max_tokens: maxTokens,
          stream: true,
          messages,
        }),
      });
    } catch (e) {
      // Network blip — retry with backoff
      if (attempt < maxRetries - 1) {
        await sleep(Math.min((2 ** attempt) * 1000, 10000));
        continue;
      }
      throw e;
    }

    if (res.status === 429 || (!res.ok && res.status >= 500)) {
      const body = await res.text();
      if (res.status === 429 && isDailyLimit(body)) {
        throw new DailyLimitError();
      }
      if (attempt < maxRetries - 1) {
        // Honour the provider's suggested retry delay if present, but cap it:
        // inside a 60s serverless function a long silent sleep would blow the
        // whole run, so fail fast (surfacing an error) rather than hang.
        const m = body.match(/try again in\s+(\d+\.?\d*)\s*s/i);
        const headerWait = Number(res.headers.get("retry-after")) * 1000;
        const wait = m
          ? (parseFloat(m[1]) + 1) * 1000
          : headerWait || (2 ** attempt) * 2000;
        await sleep(Math.min(wait, 12000));
        continue;
      }
      throw new Error(`Groq request failed (${res.status}): ${body.slice(0, 300)}`);
    }

    if (!res.ok || !res.body) {
      const body = await res.text().catch(() => "");
      throw new Error(`Groq request failed (${res.status}): ${body.slice(0, 300)}`);
    }

    // ── Parse the SSE stream ────────────────────────────────────────────────
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      const lines = buffer.split("\n");
      buffer = lines.pop() ?? ""; // keep the trailing partial line

      for (const line of lines) {
        const trimmed = line.trim();
        if (!trimmed.startsWith("data:")) continue;
        const data = trimmed.slice(5).trim();
        if (data === "[DONE]") return;
        try {
          const json = JSON.parse(data);
          const delta = json?.choices?.[0]?.delta?.content;
          if (delta) yield delta as string;
        } catch {
          // partial/keepalive line — ignore
        }
      }
    }
    return; // stream finished cleanly
  }
}

/** Non-streaming convenience wrapper: collect the full completion text. */
export async function chat(opts: {
  apiKey: string;
  model: string;
  temperature: number;
  messages: ChatMessage[];
  maxTokens?: number;
}): Promise<string> {
  let out = "";
  for await (const chunk of streamChat(opts)) out += chunk;
  return out;
}
