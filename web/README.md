# Multi-Agent Research — Web (Vercel)

An interactive, browser-native front-end for the Multi-Agent Research System,
built to deploy on **Vercel**. It runs the **same five-agent pipeline** as the
Streamlit app — Orchestrator → Researcher → Analyst → Writer → Critic, with a
critique→revise loop — but reimplemented in TypeScript so it fits Vercel's
serverless model, calling **Groq** directly and **streaming each agent's output
live** to the browser.

The Python/Streamlit app in the repo root is **unchanged** — this is a separate,
self-contained app in `web/`.

> **Why a rewrite instead of deploying the Python app?** Vercel's serverless
> functions can't host the Python stack (ChromaDB + LangGraph + ONNX embeddings
> exceed the function size/time limits). Groq is fast enough that the whole
> pipeline streams to completion well within a serverless function's duration.

---

## What you get

- **Live agent streaming** — watch each agent think and write, token by token,
  in expandable panels, with a step tracker and the Critic's 1–10 score.
- **Same tunables as Streamlit** — model, response style (Precise/Balanced/
  Creative), max revision rounds, quality bar, web-search toggle.
- **Optional grounding** — paste reference notes to ground the research
  (a lightweight stand-in for the Streamlit app's document RAG).
- **Real citations** — when a Tavily key is configured, the Researcher does
  real web search and the report gets a References section.
- **Bring-your-own-key** — visitors can paste their own Groq key in the UI, so a
  public demo doesn't burn the deployer's quota.
- **Export** — download the report as Markdown, or Print → Save as PDF.

---

## Deploy to Vercel (≈ 2 minutes)

### ⚠️ The one setting that matters: **Root Directory = `web`**

This Next.js app lives in the `web/` subdirectory. The repo root is a Python
project, so if you import the repo with the default settings Vercel will detect
**Python** and try to build `requirements.txt` — that is not this app and it
will not work. You must point Vercel at `web/`.

### Steps

1. Push this repo to GitHub (the `web/` directory must be committed).
2. On <https://vercel.com/new>, **Import** the repository.
3. In the import screen, expand **Root Directory**, click **Edit**, and select
   **`web`**. Vercel will now detect **Next.js** automatically.
   - Already created the project at the root? Go to **Project → Settings →
     General → Root Directory**, set it to `web`, **Save**, then redeploy.
4. Add **Environment Variables** (Settings → Environment Variables):

   | Name             | Required | Value                                                            |
   | ---------------- | -------- | ---------------------------------------------------------------- |
   | `GROQ_API_KEY`   | Yes\*    | Free key from <https://console.groq.com/keys>                    |
   | `GROQ_MODEL`     | No       | `llama-3.1-8b-instant` (default) or `llama-3.3-70b-versatile`    |
   | `TAVILY_API_KEY` | No       | Enables real web search + citations (<https://app.tavily.com>)   |

   \* Not strictly required if every visitor will paste their own key in the UI,
   but recommended so the app works out of the box.
5. **Deploy.** You get a permanent public URL.

> **Function duration:** the streaming endpoint is capped at 60s
> (`vercel.json` → `maxDuration`, the Hobby-tier max). Defaults (8B model, 1
> revision round) finish comfortably inside that. On Pro you can raise it for
> the 70B model or more revision rounds.

---

## Run locally

```bash
cd web
cp .env.example .env.local      # add your GROQ_API_KEY (and optional TAVILY_API_KEY)
npm install
npm run dev                     # http://localhost:3000
```

No key in `.env.local`? You can still run it — just paste a Groq key into the
**Settings → Your Groq API key** field in the UI.

---

## How it maps to the Python code

| Python (`agents/`, `workflow.py`)        | Here (`lib/`)                              |
| ---------------------------------------- | ------------------------------------------ |
| `workflow.py` LangGraph state machine    | `lib/pipeline.ts` (`runResearch`)          |
| Agent system prompts + temperatures      | `lib/pipeline.ts` (`PROMPTS`, `BASE_TEMP`) |
| `config/providers.py` (Groq chat)        | `lib/groq.ts` (`streamChat`)               |
| `tools/toolkit.py` `web_search` (Tavily) | `lib/search.ts` (`tavilySearch`)           |
| Streamlit live progress UI               | `app/page.tsx` (NDJSON stream reader)      |
| FastAPI endpoint                          | `app/api/research/route.ts`                |

System prompts, per-agent base temperatures, the response-style temperature
offset, the inter-agent text-clipping budgets, the `SCORE:N/10` parsing, the
critique→revise loop, and the daily-rate-limit handling are all ported to match
the Python behaviour.

### Differences from the Python app (by necessity on serverless)

- **Embeddings/ChromaDB RAG** is replaced by an optional "reference notes"
  textarea (no persistent vector store on serverless). For full PDF/document
  RAG, use the Streamlit app.
- **Web search** uses Tavily only (the keyless DuckDuckGo path isn't reliable
  from serverless). Without a `TAVILY_API_KEY`, the Researcher relies on the
  model's own knowledge.
- **Python code execution** by the Analyst is omitted (no subprocess sandbox on
  serverless); the Analyst reasons analytically instead.
