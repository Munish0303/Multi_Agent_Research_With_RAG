from dotenv import load_dotenv
import os
import warnings

load_dotenv()

# Suppress noisy deprecation warnings from LangChain / LangGraph internals
warnings.filterwarnings("ignore", category=DeprecationWarning, module="langgraph")
warnings.filterwarnings("ignore", message=".*allowed_objects.*", category=UserWarning)

# Suppress ChromaDB telemetry noise
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./data/vectordb")
UPLOAD_DIR = os.getenv("UPLOAD_DIR", "./data/uploads")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "./outputs")

# ── Cloud LLM provider (for deployment) ───────────────────────────────────────
# If GROQ_API_KEY is set, the app uses Groq's hosted Llama models + in-process
# sentence-transformers embeddings (no Ollama needed). Leave blank to run
# everything locally on Ollama. This lets the SAME codebase run locally and
# deploy to Streamlit Cloud / any web host unchanged.
# Get a free key at: https://console.groq.com/keys
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")  # 500k tokens/day free tier
# Embedding model used when running on a cloud provider (CPU-friendly, ~80 MB)
HF_EMBED_MODEL = os.getenv("HF_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# Optional search API keys — set whichever you have (Tavily recommended)
# Leave blank to use DuckDuckGo (free, no key, but rate-limited)
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")
BRAVE_API_KEY = os.getenv("BRAVE_API_KEY", "")

# ── Ensure writable storage (Streamlit Cloud repo mount can be read-only) ─────
import tempfile


def _ensure_writable(path: str, fallback_name: str) -> str:
    """Return *path* if we can create+write it, else a temp-dir fallback.
    Both the RAG writer and the count reader import this value, so they always
    agree on the same location."""
    try:
        os.makedirs(path, exist_ok=True)
        testfile = os.path.join(path, ".write_test")
        with open(testfile, "w") as _f:
            _f.write("ok")
        os.remove(testfile)
        return path
    except Exception:
        alt = os.path.join(tempfile.gettempdir(), fallback_name)
        os.makedirs(alt, exist_ok=True)
        return alt


CHROMA_PERSIST_DIR = _ensure_writable(CHROMA_PERSIST_DIR, "marag_vectordb")
UPLOAD_DIR         = _ensure_writable(UPLOAD_DIR, "marag_uploads")
OUTPUT_DIR         = _ensure_writable(OUTPUT_DIR, "marag_outputs")

# FastEmbed downloads its ONNX model to this cache on first use — make sure
# it lands somewhere writable on cloud hosts.
os.environ.setdefault("FASTEMBED_CACHE_PATH",
                      os.path.join(tempfile.gettempdir(), "fastembed_cache"))
os.makedirs(os.environ["FASTEMBED_CACHE_PATH"], exist_ok=True)
