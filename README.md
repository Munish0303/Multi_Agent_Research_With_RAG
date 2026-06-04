# Multi-Agent Research System

An autonomous research system where five specialised AI agents collaborate to **research, analyse, write, and critique** structured reports. It runs the same codebase two ways:

- **Locally** — 100% offline on [Ollama](https://ollama.com), no API key, no data leaves your machine
- **In the cloud** — on [Groq](https://groq.com)'s hosted models, deployable to Streamlit Cloud for free so anyone can use it from a browser

The active backend is chosen automatically: set a `GROQ_API_KEY` and it uses Groq; leave it blank and it uses Ollama.

---

## Demo

```bash
streamlit run app.py          # open http://localhost:8501
```

Type a topic → watch each agent work in real time → download the finished report as **PDF or Markdown**. Upload your own PDFs/notes to ground the research in your documents (RAG).

---

## Architecture

```
User Query
    │
    ▼
┌─────────────┐
│ Orchestrator│  Plans the research strategy
└──────┬──────┘
       ▼
┌─────────────┐
│  Researcher │  Web search (Tavily / Brave / DuckDuckGo) + RAG retrieval
└──────┬──────┘
       ▼
┌─────────────┐
│   Analyst   │  Pattern analysis, evidence evaluation, Python calculations
└──────┬──────┘
       ▼
┌─────────────┐
│    Writer   │  Structured report (Executive Summary → Recommendations)
└──────┬──────┘
       ▼
┌─────────────┐
│    Critic   │  Quality review + numeric score (SCORE:N/10)
└──────┬──────┘
       │ score below bar & rounds remaining?
       ▼
┌─────────────┐
│   Revise    │  Writer rewrites addressing the critique  ──► back to Critic
└──────┬──────┘
       ▼
┌─────────────┐
│  Finalize   │  Orchestrator synthesises the final report → saves PDF + MD
└─────────────┘
```

The pipeline is a **LangGraph state machine**: each agent is a node, and the critique → revise loop is a conditional edge driven by the Critic's score.

### Tech Stack

| Layer | Local mode | Cloud mode |
|---|---|---|
| LLM inference | Ollama (`llama3.2` / any local model) | Groq (`llama-3.1-8b-instant`, `llama-3.3-70b`) |
| Embeddings (RAG) | Ollama `nomic-embed-text` (768-dim) | `fastembed` ONNX `bge-small-en` (384-dim, no PyTorch) |
| Orchestration | LangGraph state machine | ← same |
| Vector store | ChromaDB | ← same |
| Web search | DuckDuckGo · Tavily · Brave | ← same |
| Web UI | Streamlit | ← same |
| REST API | FastAPI | ← same |
| PDF export | fpdf2 | ← same |
| Memory | Short-term buffer + long-term ChromaDB per agent | ← same |

---

## Setup (local)

### 1. Install Ollama and pull models

```bash
# macOS / Linux
curl -fsSL https://ollama.ai/install.sh | sh
# Windows — download from https://ollama.com/download

ollama pull llama3.2          # main LLM (~2 GB)
ollama pull nomic-embed-text  # embedding model (~274 MB)
```

### 2. Clone and install

```bash
git clone https://github.com/Munish0303/Multi_Agent_Research_RAG.git
cd Multi_Agent_Research_RAG
pip install -r requirements.txt
```

### 3. Configure

```bash
cp .env.example .env
# Leave GROQ_API_KEY blank to run locally on Ollama.
# Optionally add a TAVILY_API_KEY for better web search.
```

---

## Usage

### Web UI (recommended)

```bash
ollama serve          # keep running in a separate terminal
streamlit run app.py  # open http://localhost:8501
```

- **New Research** — live per-step progress tracker, PDF + MD download, source citations
- **History** — every past run saved (SQLite) with re-download links
- **Upload Documents** — ingest PDFs / TXT / MD for RAG-grounded research
- **Sidebar** — switch models, toggle web search & RAG, and tune the agents:
  - *Response style* (Precise / Balanced / Creative)
  - *Max revision rounds* and *Quality bar* (Critic score to pass)
  - *Document chunks per agent* (RAG depth)

### CLI

```bash
python main.py research "Impact of AI on healthcare"   # basic run
python main.py research "Quantum computing" --ingest   # use uploaded docs
python main.py research "Topic" --no-rag               # web-only, skip RAG
python main.py status                                  # system status
```

### REST API

```bash
uvicorn api:app --reload        # docs at http://localhost:8000/docs
```

```bash
curl -X POST http://localhost:8000/research \
  -H "Content-Type: application/json" \
  -d '{"topic": "Future of renewable energy"}'

curl http://localhost:8000/research/{job_id}            # poll for result
curl -X POST http://localhost:8000/ingest -F "file=@paper.pdf"
```

---

## Deploy to the cloud (free)

Make it accessible from any browser with **Groq + Streamlit Community Cloud** — no GPU, no Ollama, no cost. Full walkthrough in [`DEPLOYMENT.md`](./DEPLOYMENT.md). Short version:

1. Get a free Groq key at <https://console.groq.com/keys>
2. Push the repo to GitHub
3. On <https://share.streamlit.io>, create an app pointing at `app.py`
4. In **Advanced settings → Secrets**, add:
   ```toml
   GROQ_API_KEY = "gsk_your_key"
   GROQ_MODEL = "llama-3.1-8b-instant"
   ```
5. Deploy → you get a permanent public URL

The repo pins **Python 3.11** (`runtime.txt`) and uses **fastembed** instead of PyTorch so it builds and runs within the free tier's limits.

---

## Output

Each run produces two files in `./outputs/`:

| File | Description |
|---|---|
| `report_<topic>.md` | Full Markdown report with a References section |
| `report_<topic>.pdf` | Formatted PDF — cover page, headings, page numbers |

A real generated report is included in [`example_output/`](./example_output/).

---

## Project Structure

```
Multi_Agent_Research_RAG/
├── app.py               # Streamlit web UI (live progress, history, uploads)
├── api.py               # FastAPI REST server
├── main.py              # CLI entry point (Typer)
├── workflow.py          # LangGraph pipeline + run configuration
│
├── agents/
│   ├── base.py          # BaseAgent — LLM calls, retry/rate-limit, tool parsing
│   ├── orchestrator.py  # Planning + final synthesis
│   ├── researcher.py    # Web search + RAG retrieval
│   ├── analyst.py       # Critical analysis + Python execution
│   ├── writer.py        # Report drafting + revision
│   └── critic.py        # Quality scoring (SCORE:N/10)
│
├── config/
│   ├── settings.py      # Env config + writable-storage handling
│   └── providers.py     # LLM/embedding factory (Ollama ↔ Groq)
│
├── rag/pipeline.py      # Document ingestion (robust PDF) + retrieval
├── memory/manager.py    # ShortTermMemory + LongTermMemory (ChromaDB)
├── tools/toolkit.py     # web_search, read_file, execute_python, save_output
│
├── utils/
│   ├── pdf_export.py    # Markdown → PDF (Unicode-safe)
│   ├── history.py       # SQLite run history
│   └── job_store.py     # Process-persistent job state for Streamlit threads
│
├── runtime.txt          # Pins Python 3.11 for Streamlit Cloud
├── .streamlit/          # Theme + secrets template
├── DEPLOYMENT.md        # Cloud deployment guide
├── .env.example
└── requirements.txt
```

---

## Web Search

The Researcher auto-selects the best available provider:

| Provider | Key required | Free tier | Notes |
|---|---|---|---|
| Tavily | Yes | 1,000/month | Best quality, recommended |
| Brave Search | Yes | 2,000/month | Good alternative |
| DuckDuckGo | No | Unlimited\* | Default, rate-limited (retries with backoff) |

Set a key in `.env` (or Streamlit secrets) to upgrade:
```
TAVILY_API_KEY=tvly-...
```

---

## Key Engineering Decisions

- **Provider abstraction** (`config/providers.py`) — one factory returns Ollama or Groq for chat, and Ollama-embeddings or fastembed for RAG, chosen by the presence of `GROQ_API_KEY`. Same code, two deployment modes; vectors from different embedding dims are kept in separate collections to avoid clashes.
- **LangGraph state machine** — agents are nodes; the critique→revise loop is a conditional edge, not hand-rolled recursion.
- **Thread-safe Streamlit** — the long-running pipeline runs in a background thread that writes to a process-level `_JOB` dict (not `st.session_state`, which can't be written from worker threads), so the UI updates live without `ScriptRunContext` errors.
- **Rate-limit handling** — per-minute 429s are retried with backoff using the provider's suggested delay; per-day limits fail fast with a clear, actionable message.
- **Token budgeting** — inter-agent text is clipped to keep a full run comfortably within hosted free-tier daily token limits.
- **Robust ingestion** — malformed PDFs fall back to a lenient page-by-page reader; uploads are verified (`fsync` + size check) so truncated writes can't silently corrupt the index.
- **Tool-call robustness** — handles native tool calls, JSON-narrated tool calls, schema-dict arguments, and Groq's `tool_use_failed` by retrying without tools.
- **Unicode-safe PDF export** — smart quotes, em-dashes, arrows, and symbols are transliterated so the built-in PDF font never crashes on real LLM output.
