"""
Streamlit Web UI for the Multi-Agent Research System.
Run with:  streamlit run app.py
"""
import streamlit as st
import threading
import time
import os
from datetime import datetime

# ── Bridge Streamlit Cloud secrets → environment variables ────────────────────
# Streamlit stores deployment secrets in st.secrets, but config.settings reads
# os.getenv(). Copy them across BEFORE any config.* module is imported so the
# Groq key (and any other secret) is visible to the rest of the app.
try:
    for _k, _v in st.secrets.items():
        if isinstance(_v, str):
            os.environ.setdefault(_k, _v)
except Exception:
    pass  # no secrets file locally — that's fine, we fall back to .env / Ollama

# ── Persistent job store ──────────────────────────────────────────────────────
# Imported from a separate module so Python's import cache keeps it alive
# across Streamlit reruns (app.py is re-exec'd on every interaction; a
# module-level dict defined HERE would be reset each time).
from utils.job_store import _JOB

# ── Page config (must be the very first Streamlit call) ───────────────────────
st.set_page_config(
    page_title="Multi-Agent Research System",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .stTabs [data-baseweb="tab-list"] { gap: 6px; }
    .stTabs [data-baseweb="tab"]      { padding: 6px 18px; }
    hr { border-color: #e0e0e0; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ── Cached helpers (always run in the main Streamlit thread) ──────────────────

@st.cache_data(ttl=300)
def _available_models() -> list:
    """Chat models for the active provider (Groq list, or discovered Ollama models)."""
    from config.providers import active_provider, available_models
    if active_provider() == "groq":
        return available_models()

    # Ollama: query the local daemon and keep only chat-capable models
    try:
        import requests as _req
        r = _req.get("http://localhost:11434/api/tags", timeout=3)
        if r.status_code == 200:
            models = r.json().get("models", [])
            seen, result = set(), []
            for m in models:
                name = m["name"]
                fam  = m.get("details", {}).get("family", "").lower()
                if "embed" in name.lower() or fam == "nomic-bert":
                    continue
                if name not in seen:
                    seen.add(name)
                    result.append(name)
            return result or ["llama3.1"]
    except Exception:
        pass
    return ["llama3.1"]


@st.cache_data(ttl=10)
def _backend_ok() -> bool:
    """True if the active LLM backend is reachable."""
    from config.providers import active_provider
    if active_provider() == "groq":
        from config.settings import GROQ_API_KEY
        return bool(GROQ_API_KEY)
    try:
        import requests as _req
        return _req.get("http://localhost:11434/api/tags", timeout=2).status_code == 200
    except Exception:
        return False


def _doc_count() -> int:
    """
    Return the number of indexed document chunks.
    Checks session state first (set immediately after ingest) so the sidebar
    updates in the same rerun rather than waiting for a ChromaDB round-trip.
    Falls back to a direct ChromaDB query that does NOT create embeddings
    (avoids model-download latency that caused the old RAGPipeline() call to
    silently return 0 on cloud).
    """
    # Fast path: ingestion sets this in the same session
    if st.session_state.get("_docs_count") is not None:
        return st.session_state["_docs_count"]

    # Slow path: query ChromaDB directly (no embeddings needed for a count)
    try:
        import chromadb
        from chromadb.config import Settings as _CS
        from config.settings import CHROMA_PERSIST_DIR
        from config.providers import collection_suffix

        client = chromadb.PersistentClient(
            path=CHROMA_PERSIST_DIR,
            settings=_CS(anonymized_telemetry=False),
        )
        col_name = "research_docs" + collection_suffix()
        try:
            col = client.get_collection(col_name)
            count = col.count()
            st.session_state["_docs_count"] = count   # cache in session state
            return count
        except Exception:
            return 0
    except Exception:
        return 0


def _paths_for(topic: str) -> tuple:
    from config.settings import OUTPUT_DIR
    safe = "".join(c if c.isalnum() or c in " -" else "_" for c in topic[:30])
    base = f"report_{safe.strip().replace(' ', '_').lower()}"
    return (
        os.path.join(OUTPUT_DIR, f"{base}.md"),
        os.path.join(OUTPUT_DIR, f"{base}.pdf"),
    )


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🔬 Research System")
    st.divider()

    from config.providers import provider_label, active_provider

    st.markdown("**System Status**")
    _is_groq = active_provider() == "groq"
    if _backend_ok():
        st.success(f"✅ {provider_label()}")
    elif _is_groq:
        st.error("❌ GROQ_API_KEY missing — add it to your secrets")
    else:
        st.error("❌ Ollama offline — run `ollama serve`")

    n_docs = _doc_count()
    st.metric("Indexed documents", n_docs)
    st.divider()

    st.markdown("**Settings**")
    model_options = _available_models()
    sel_model   = st.selectbox("LLM model", model_options)
    rag_enabled = st.toggle(
        "Use uploaded docs (RAG)",
        value=(n_docs > 0),
        help="Uses documents you upload in the 'Upload Documents' tab to "
             "ground research. Automatically off when no documents are indexed.",
    )
    if rag_enabled and n_docs == 0:
        st.caption("⚠️ No documents indexed yet — upload some in the Documents tab.")

    # ── Agent tuning ───────────────────────────────────────────────────────
    with st.expander("⚙️ Agent Settings", expanded=False):
        style = st.select_slider(
            "Response style",
            options=["Precise", "Balanced", "Creative"],
            value="Balanced",
            help="Shifts every agent's temperature. Precise = factual & "
                 "repeatable, Creative = more varied prose.",
        )
        # Map style → temperature offset (preserves each agent's relative temp)
        _style_offset = {"Precise": -0.1, "Balanced": 0.0, "Creative": 0.25}
        temp_offset = _style_offset[style]

        max_iterations = st.slider(
            "Max revision rounds", min_value=0, max_value=3, value=1,
            help="How many times the Critic can send the report back to the "
                 "Writer for improvement. Lower = fewer tokens used.",
        )

        quality_threshold = st.slider(
            "Quality bar (Critic score to pass)", min_value=5, max_value=10, value=7,
            help="The report is revised until the Critic scores it at least "
                 "this high (or max rounds is reached).",
        )

        rag_k = st.slider(
            "Document chunks per agent (RAG)", min_value=2, max_value=8, value=4,
            disabled=(not rag_enabled),
            help="How many chunks each agent retrieves from your uploaded "
                 "documents.",
        )

        web_search_on = st.toggle(
            "Enable web search",
            value=True,
            help="Let the Researcher search the web. Turn off for "
                 "document-only (RAG) research.",
        )

    # Assemble the run configuration passed to the pipeline
    agent_config = {
        "temperature_offset": temp_offset,
        "rag_k":              rag_k,
        "use_web_search":     web_search_on,
        "max_iterations":     max_iterations,
        "quality_threshold":  quality_threshold,
    }

    st.divider()
    st.caption("LangGraph · Ollama · ChromaDB · Streamlit")


# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_new, tab_hist, tab_docs = st.tabs(
    ["🔍  New Research", "📚  History", "📄  Upload Documents"]
)

STEP_ICONS = {
    "plan":     "🎯",
    "research": "🔍",
    "analyze":  "📊",
    "write":    "✍️",
    "review":   "🔎",
    "revise":   "🔄",
    "finalize": "✅",
}

PIPELINE = [
    ("plan",     "Plan"),
    ("research", "Research"),
    ("analyze",  "Analyze"),
    ("write",    "Write"),
    ("review",   "Review"),
    ("revise",   "Revise"),
    ("finalize", "Finalize"),
]


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — NEW RESEARCH
# ══════════════════════════════════════════════════════════════════════════════
with tab_new:
    st.header("Start New Research")

    topic_in = st.text_input(
        "Research topic",
        placeholder="e.g. US Stock Market",
        disabled=_JOB["running"],
        label_visibility="collapsed",
    )

    col_btn, col_status = st.columns([1, 3])
    with col_btn:
        go = st.button(
            "🚀  Start Research",
            type="primary",
            disabled=(_JOB["running"] or not topic_in.strip()),
            use_container_width=True,
        )
    with col_status:
        if _JOB["running"]:
            elapsed = int(time.time() - (_JOB["start_ts"] or time.time()))
            st.info(f"⏳ Researching **{_JOB['topic']}** — {elapsed}s elapsed")
        elif _JOB["result"]:
            st.success("✅ Research complete — report below")

    # ── Launch worker ──────────────────────────────────────────────────────────
    if go and topic_in.strip():
        _JOB["running"]      = True
        _JOB["active_step"]  = ""
        _JOB["progress"]     = []
        _JOB["result"]       = None
        _JOB["citations"]    = []
        _JOB["error"]        = None
        _JOB["topic"]        = topic_in.strip()
        _JOB["start_ts"]     = time.time()

        _snap_topic  = topic_in.strip()
        _snap_model  = sel_model
        _snap_rag    = rag_enabled
        _snap_config = dict(agent_config)   # snapshot for the thread

        def _worker():
            # ── Never touches st.session_state — writes only to _JOB ────────
            try:
                from workflow import run_research
                from rag.pipeline import RAGPipeline

                rag = RAGPipeline() if _snap_rag else None

                def _cb(step: str, msg: str) -> None:
                    _JOB["active_step"] = step
                    _JOB["progress"].append({"step": step, "message": msg})

                result = run_research(
                    _snap_topic,
                    rag=rag,
                    progress_callback=_cb,
                    model=_snap_model,
                    config=_snap_config,
                )

                _JOB["result"]    = result.get("report", "")
                _JOB["citations"] = result.get("citations", [])

                # Persist to history
                md_p, pdf_p = _paths_for(_snap_topic)
                try:
                    from utils.history import save_run
                    save_run(
                        topic           = _snap_topic,
                        started_at      = datetime.fromtimestamp(
                                              _JOB["start_ts"]
                                          ).isoformat(timespec="seconds"),
                        duration        = time.time() - _JOB["start_ts"],
                        citations_count = len(_JOB["citations"]),
                        md_path         = md_p  if os.path.exists(md_p)  else None,
                        pdf_path        = pdf_p if os.path.exists(pdf_p) else None,
                        status          = "complete",
                        report_preview  = (_JOB["result"] or "")[:400],
                    )
                except Exception:
                    pass

            except Exception as exc:
                import traceback
                _JOB["error"] = f"{exc}\n\n{traceback.format_exc()}"

            finally:
                _JOB["running"] = False

        threading.Thread(target=_worker, daemon=True).start()
        st.rerun()

    # ── Pipeline tracker + live progress ──────────────────────────────────────
    is_running        = _JOB["running"]
    active            = _JOB["active_step"]
    progress_snapshot = list(_JOB["progress"])   # thread-safe snapshot
    seen_steps        = {item["step"] for item in progress_snapshot}

    if is_running or progress_snapshot:
        st.subheader("Agent Progress")

        # ── Step pipeline bar ──────────────────────────────────────────────
        cols = st.columns(len(PIPELINE))
        for col, (key, label) in zip(cols, PIPELINE):
            with col:
                if key == active and is_running:
                    # Currently running — blue highlight
                    st.markdown(
                        f"<div style='text-align:center;padding:6px 2px;"
                        f"background:#dbeafe;border-radius:8px;"
                        f"font-weight:600;font-size:0.8rem;'>⏳ {label}</div>",
                        unsafe_allow_html=True,
                    )
                elif key in seen_steps:
                    # Completed — green
                    st.markdown(
                        f"<div style='text-align:center;padding:6px 2px;"
                        f"background:#dcfce7;border-radius:8px;"
                        f"font-size:0.8rem;'>✅ {label}</div>",
                        unsafe_allow_html=True,
                    )
                else:
                    # Not started — grey
                    st.markdown(
                        f"<div style='text-align:center;padding:6px 2px;"
                        f"background:#f3f4f6;border-radius:8px;"
                        f"color:#9ca3af;font-size:0.8rem;'>○ {label}</div>",
                        unsafe_allow_html=True,
                    )

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Detailed step log ──────────────────────────────────────────────
        for item in progress_snapshot:
            icon = STEP_ICONS.get(item["step"], "⚙️")
            st.success(f"{icon} **{item['step'].capitalize()}** — {item['message']}")

        # ── Current step indicator ─────────────────────────────────────────
        if is_running:
            if not active:
                st.info("⚙️ **Initializing** — loading agents and pipeline…")
            else:
                icon  = STEP_ICONS.get(active, "⚙️")
                label = dict(PIPELINE).get(active, active.capitalize())
                st.info(f"{icon} **{label}** is running — waiting for LLM…")

    # Auto-refresh every 2 s while the worker is alive
    if _JOB["running"]:
        time.sleep(2)
        st.rerun()

    # ── Error display ──────────────────────────────────────────────────────────
    if _JOB["error"]:
        st.error("Research failed")
        with st.expander("Error details"):
            st.code(_JOB["error"])

    # ── Results ───────────────────────────────────────────────────────────────
    if _JOB["result"]:
        st.divider()
        st.subheader("📑 Research Report")

        md_path, pdf_path = _paths_for(_JOB["topic"])
        dl1, dl2, _ = st.columns([1, 1, 3])

        with dl1:
            if os.path.exists(pdf_path):
                with open(pdf_path, "rb") as _f:
                    st.download_button(
                        "📥 Download PDF", _f.read(),
                        file_name=os.path.basename(pdf_path),
                        mime="application/pdf",
                        use_container_width=True,
                    )
        with dl2:
            if os.path.exists(md_path):
                with open(md_path, "r", encoding="utf-8") as _f:
                    st.download_button(
                        "📥 Download MD", _f.read(),
                        file_name=os.path.basename(md_path),
                        mime="text/markdown",
                        use_container_width=True,
                    )

        cites = _JOB["citations"]
        if cites:
            with st.expander(f"🔗 Sources & Citations ({len(cites)})"):
                for i, url in enumerate(cites, 1):
                    st.markdown(f"{i}. [{url}]({url})")

        with st.expander("📄 Full Report", expanded=True):
            st.markdown(_JOB["result"])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — HISTORY
# ══════════════════════════════════════════════════════════════════════════════
with tab_hist:
    st.header("Research History")

    if st.button("🔄 Refresh", key="hist_refresh"):
        st.rerun()

    try:
        from utils.history import get_all_runs, delete_run
        runs = get_all_runs()
    except Exception as _e:
        st.warning(f"Could not load history: {_e}")
        runs = []

    if not runs:
        st.info("No research runs yet. Start a new research to see history here.")
    else:
        for run in runs:
            c_topic, c_pdf, c_md, c_del = st.columns([4, 1, 1, 1])

            with c_topic:
                dt  = (run.get("started_at") or "")[:16].replace("T", " ")
                dur = run.get("duration_seconds") or 0
                cit = run.get("citations_count", 0)
                st.markdown(f"**{run['topic']}**")
                st.caption(f"🕒 {dt}  ·  ⏱ {dur:.0f}s  ·  🔗 {cit} sources")

            with c_pdf:
                pdf = run.get("pdf_path")
                if pdf and os.path.exists(pdf):
                    with open(pdf, "rb") as _f:
                        st.download_button(
                            "📥 PDF", _f.read(),
                            file_name=os.path.basename(pdf),
                            mime="application/pdf",
                            key=f"dl_pdf_{run['id']}",
                            use_container_width=True,
                        )

            with c_md:
                md = run.get("md_path")
                if md and os.path.exists(md):
                    with open(md, "r", encoding="utf-8") as _f:
                        st.download_button(
                            "📥 MD", _f.read(),
                            file_name=os.path.basename(md),
                            mime="text/markdown",
                            key=f"dl_md_{run['id']}",
                            use_container_width=True,
                        )

            with c_del:
                if st.button("🗑️", key=f"del_{run['id']}", help="Remove record"):
                    delete_run(run["id"])
                    st.rerun()

            if run.get("report_preview"):
                with st.expander("Preview"):
                    st.markdown(run["report_preview"] + " …")

            st.divider()


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — UPLOAD DOCUMENTS
# ══════════════════════════════════════════════════════════════════════════════
with tab_docs:
    st.header("Upload Documents for RAG")
    st.markdown(
        "Upload PDFs, plain-text, or markdown files. "
        "Agents will draw on them during every future run."
    )

    uploads = st.file_uploader(
        "Choose files",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if uploads and st.button("📤 Ingest into Knowledge Base", type="primary"):
        import traceback
        results = []          # list of (status, filename, detail)
        verified_count = None

        with st.spinner("Ingesting… (first run downloads the embedding model)"):
            try:
                from rag.pipeline import RAGPipeline
                from config.settings import UPLOAD_DIR

                _rag_inst = RAGPipeline()

                for uf in uploads:
                    dest = os.path.join(UPLOAD_DIR, uf.name)
                    try:
                        # getvalue() returns the complete uploaded bytes
                        data = uf.getvalue()
                        with open(dest, "wb") as _f:
                            _f.write(data)
                            _f.flush()
                            os.fsync(_f.fileno())   # ensure bytes hit disk before read
                        # Sanity check: did the full file land on disk?
                        if os.path.getsize(dest) != len(data):
                            raise IOError(
                                f"Incomplete write: {os.path.getsize(dest)} of "
                                f"{len(data)} bytes saved."
                            )
                        chunks = _rag_inst.ingest_file(dest)
                        results.append(("ok", uf.name, f"{chunks} chunks indexed"))
                    except Exception as _e:
                        results.append((
                            "err", uf.name,
                            f"{type(_e).__name__}: {_e}\n\n{traceback.format_exc()}"
                        ))

                # Ground truth: ask ChromaDB how many chunks are ACTUALLY stored
                try:
                    verified_count = _rag_inst.collection_count()
                except Exception:
                    verified_count = None

            except Exception as _e:
                # RAGPipeline() itself failed (e.g. embedding model init)
                results.append((
                    "fatal", "Knowledge base initialisation",
                    f"{type(_e).__name__}: {_e}\n\n{traceback.format_exc()}"
                ))

        # Persist results in session state so they SURVIVE the rerun
        st.session_state["_ingest_results"] = results
        if verified_count is not None:
            st.session_state["_docs_count"] = verified_count
        st.rerun()

    # ── Render last ingest results (these survive st.rerun) ────────────────
    _results = st.session_state.get("_ingest_results")
    if _results:
        any_ok = False
        for status, name, detail in _results:
            if status == "ok":
                any_ok = True
                st.success(f"✅ **{name}** — {detail}")
            else:
                st.error(f"❌ **{name}** failed")
                with st.expander(f"Why did '{name}' fail? (full error)"):
                    st.code(detail)

        _cnt = st.session_state.get("_docs_count")
        if _cnt is not None:
            if _cnt > 0:
                st.info(f"📚 Knowledge base now holds **{_cnt}** indexed chunks "
                        "(verified directly from the vector store).")
            elif any_ok:
                st.warning(
                    "Files reported success but the vector store is still empty — "
                    "the embedding step likely failed silently. See the debug panel."
                )

    # ── Debug panel — exposes the actual backend state ────────────────────
    with st.expander("🔧 Debug: vector store state"):
        try:
            from config.providers import active_provider, collection_suffix
            from config.settings import CHROMA_PERSIST_DIR
            st.write({
                "provider":       active_provider(),
                "persist_dir":    os.path.abspath(CHROMA_PERSIST_DIR),
                "persist_writable": os.access(CHROMA_PERSIST_DIR, os.W_OK)
                                    if os.path.exists(CHROMA_PERSIST_DIR) else "dir missing",
                "collection":     "research_docs" + collection_suffix(),
                "live_count":     _doc_count(),
            })
        except Exception as _e:
            st.write(f"debug error: {_e}")

    st.divider()
    st.subheader("Files in Knowledge Base")
    from config.settings import UPLOAD_DIR
    try:
        _files = sorted(os.listdir(UPLOAD_DIR))
        _files = [f for f in _files if not f.startswith(".")]
        if _files:
            for _fn in _files:
                _fp   = os.path.join(UPLOAD_DIR, _fn)
                _size = os.path.getsize(_fp)
                st.markdown(f"📎 `{_fn}` &nbsp; {_size // 1024} KB")
        else:
            st.info("No files uploaded yet.")
    except Exception:
        st.info("Upload directory not accessible.")
