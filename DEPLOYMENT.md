# Deployment Guide

This app runs in **two interchangeable modes** with zero code changes:

| Mode | LLM | Embeddings | When |
|---|---|---|---|
| **Local** | Ollama (`llama3.2`) | Ollama (`nomic-embed-text`) | Development on your machine |
| **Cloud** | Groq (hosted Llama) | sentence-transformers (in-process) | Deployed for others to use |

The provider is chosen automatically: **if `GROQ_API_KEY` is set, it uses Groq; otherwise Ollama.**

---

## Deploy to Streamlit Community Cloud (free, recommended)

This gives you a public URL anyone (e.g. a recruiter) can open and use — no install, always on.

### 1. Get a free Groq API key
- Sign up at **https://console.groq.com/keys**
- Create an API key (starts with `gsk_...`)
- Free tier is generous and very fast — runs the same Llama models used locally.

### 2. Push the project to GitHub
```bash
git init
git add .
git commit -m "Multi-Agent Research System"
git remote add origin https://github.com/<you>/multi-agent-research.git
git push -u origin main
```
> `.gitignore` already excludes `.env`, the vector DB, and outputs.

### 3. Create the app on Streamlit Cloud
1. Go to **https://share.streamlit.io** and sign in with GitHub.
2. **New app** → pick your repo → set **Main file path** to `app.py`.
3. Click **Advanced settings → Secrets** and paste:
   ```toml
   GROQ_API_KEY = "gsk_your_real_key_here"
   GROQ_MODEL = "llama-3.1-8b-instant"
   ```
4. Click **Deploy**.

First boot takes a few minutes (it installs deps and downloads the ~80 MB
embedding model). After that you get a permanent URL like
`https://your-app.streamlit.app` to share.

### 4. Verify
- Sidebar should show **✅ Groq (cloud)**.
- Enter a topic → watch the agents run → download the PDF.

---

## Notes & gotchas

- **No Ollama in the cloud.** That's expected — Groq replaces it. The sidebar
  model dropdown automatically shows Groq models when deployed.
- **RAG starts empty in the cloud.** The vector DB isn't committed to git.
  Visitors can upload documents in the *Upload Documents* tab to try RAG.
- **Embedding dimensions differ** (cloud MiniLM = 384-dim, local nomic = 768-dim).
  The code keeps them in separate ChromaDB collections (`_hf` suffix in cloud),
  so switching modes never corrupts your local index.
- **Costs:** Streamlit Community Cloud is free; Groq's free tier covers demo use.
  No credit card needed for either.

---

## Alternative: quick live demo with a tunnel (no deploy)

If you just need a URL for a scheduled call and your machine will be on:

```bash
# Terminal 1 — your normal local stack
ollama serve
streamlit run app.py

# Terminal 2 — expose it publicly (install ngrok first)
ngrok http 8501
```

ngrok prints a public `https://...ngrok.app` URL that tunnels to your laptop.
Works only while your machine and Ollama are running — good for a live demo,
not for always-on access.

---

## Alternative: full cloud with Ollama (GPU host)

If you specifically want Ollama (not Groq) in the cloud, run both services on a
GPU VM (RunPod, Fly.io, Railway, a VPS, etc.) via Docker, and point
`OLLAMA_BASE_URL` at it. This keeps the 100%-open-source stack but costs money
(~$30+/month or hourly GPU rental). The Groq path above is simpler and free.
