"""
LLM / embedding provider factory.

Lets the SAME codebase run two ways with zero code changes:

  • Locally  → Ollama (chat + embeddings), 100% offline, no API key
  • Deployed → Groq (chat) + fastembed (embeddings), runs on any
               cloud host (Streamlit Cloud, etc.) with just a free GROQ_API_KEY

fastembed is used in cloud mode because it is ONNX-based (no PyTorch),
downloads only ~80 MB, and fits comfortably on Streamlit Cloud's free tier.
sentence-transformers / HuggingFace are NOT used — they pull in PyTorch (~1 GB)
which exceeds the free tier's memory.

The active provider is chosen automatically: if GROQ_API_KEY is present we use
Groq, otherwise Ollama. All imports are lazy so neither stack needs the other's
dependencies installed.
"""
from typing import Optional

from config.settings import (
    OLLAMA_MODEL, OLLAMA_BASE_URL, OLLAMA_EMBED_MODEL,
    GROQ_API_KEY, GROQ_MODEL,
)

# Cache the embedding object — loading a sentence-transformers model is slow
# and we never want to do it more than once per process.
_embeddings_cache = None


def active_provider() -> str:
    """Return 'groq' when a Groq key is configured, else 'ollama'."""
    return "groq" if GROQ_API_KEY else "ollama"


def provider_label() -> str:
    """Human-readable label for the UI."""
    return "Groq (cloud)" if active_provider() == "groq" else "Ollama (local)"


def default_model() -> str:
    """Default chat model name for the active provider."""
    return GROQ_MODEL if active_provider() == "groq" else OLLAMA_MODEL


def available_models() -> list:
    """List of selectable chat models for the active provider."""
    if active_provider() == "groq":
        return [
            "llama-3.1-8b-instant",      # 500k tokens/DAY — best for repeated demo use
            "llama-3.3-70b-versatile",   # 100k tokens/day — higher quality, fewer runs
        ]
    # Ollama: discovered dynamically by the caller (see app.py)
    return [OLLAMA_MODEL]


def get_chat_model(temperature: float = 0.3, model: Optional[str] = None):
    """Return a LangChain chat model for the active provider."""
    if active_provider() == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(
            model=model or GROQ_MODEL,
            api_key=GROQ_API_KEY,
            temperature=temperature,
        )
    from langchain_ollama import ChatOllama
    return ChatOllama(
        model=model or OLLAMA_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=temperature,
    )


def get_embeddings():
    """Return a LangChain embeddings object for the active provider (cached)."""
    global _embeddings_cache
    if _embeddings_cache is not None:
        return _embeddings_cache

    if active_provider() == "groq":
        # Groq has no embedding endpoint.
        # Use fastembed: ONNX-based, no PyTorch, ~80 MB download.
        # langchain-community ships FastEmbedEmbeddings out of the box.
        import os
        from langchain_community.embeddings import FastEmbedEmbeddings
        _embeddings_cache = FastEmbedEmbeddings(
            model_name="BAAI/bge-small-en-v1.5",   # 384-dim, fast, accurate
            cache_dir=os.environ.get("FASTEMBED_CACHE_PATH"),
        )
    else:
        from langchain_ollama import OllamaEmbeddings
        _embeddings_cache = OllamaEmbeddings(
            model=OLLAMA_EMBED_MODEL,
            base_url=OLLAMA_BASE_URL,
        )
    return _embeddings_cache


def collection_suffix() -> str:
    """
    Suffix appended to ChromaDB collection names so vectors from different
    embedding models (768-dim nomic vs 384-dim MiniLM) never collide in the
    same collection. Local Ollama keeps the original unsuffixed names so
    existing indexed documents continue to work.
    """
    return "_hf" if active_provider() == "groq" else ""
