from typing import List, Optional
from langchain_core.tools import tool
from duckduckgo_search import DDGS
import subprocess
import sys
import os
import time
import random


# ---------------------------------------------------------------------------
# Web search — provider priority: Tavily > Brave > DuckDuckGo (with retry)
# ---------------------------------------------------------------------------

def _search_tavily(query: str, max_results: int) -> str:
    """Search using Tavily (free tier: 1,000 searches/month). Set TAVILY_API_KEY."""
    try:
        from tavily import TavilyClient
        client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])
        response = client.search(query, max_results=max_results)
        results = response.get("results", [])
        if not results:
            return "No results found."
        formatted = []
        for i, r in enumerate(results, 1):
            formatted.append(f"[{i}] {r.get('title', '')}\n{r.get('url', '')}\n{r.get('content', '')}\n")
        return "\n".join(formatted)
    except ImportError:
        return "UNAVAILABLE:tavily-python not installed. Run: pip install tavily-python"
    except Exception as e:
        return f"Tavily search failed: {e}"


def _search_brave(query: str, max_results: int) -> str:
    """Search using Brave Search API (free tier: 2,000 queries/month). Set BRAVE_API_KEY."""
    try:
        import requests as req
        api_key = os.environ["BRAVE_API_KEY"]
        headers = {
            "Accept": "application/json",
            "Accept-Encoding": "gzip",
            "X-Subscription-Token": api_key,
        }
        params = {"q": query, "count": min(max_results, 20)}
        resp = req.get(
            "https://api.search.brave.com/res/v1/web/search",
            headers=headers,
            params=params,
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        results = data.get("web", {}).get("results", [])
        if not results:
            return "No results found."
        formatted = []
        for i, r in enumerate(results, 1):
            formatted.append(f"[{i}] {r.get('title', '')}\n{r.get('url', '')}\n{r.get('description', '')}\n")
        return "\n".join(formatted)
    except ImportError:
        return "UNAVAILABLE:requests not installed."
    except Exception as e:
        return f"Brave search failed: {e}"


def _search_duckduckgo(query: str, max_results: int) -> str:
    """Search using DuckDuckGo with retry + exponential backoff (no API key needed)."""
    max_retries = 3
    base_delay = 2.0

    for attempt in range(max_retries):
        try:
            # Small polite delay to reduce rate-limit risk
            time.sleep(base_delay + random.uniform(0, 1))
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=max_results))
            if not results:
                return "No results found."
            formatted = []
            for i, r in enumerate(results, 1):
                formatted.append(f"[{i}] {r['title']}\n{r['href']}\n{r['body']}\n")
            return "\n".join(formatted)
        except Exception as e:
            if attempt < max_retries - 1:
                wait = base_delay * (2 ** attempt) + random.uniform(0, 1)
                time.sleep(wait)
            else:
                return f"DuckDuckGo search failed after {max_retries} attempts: {e}"

    return "Search failed."


@tool
def web_search(query: str, max_results: int = 5) -> str:
    """Search the web for current information. Returns relevant snippets with sources.

    Provider auto-selected by available API keys:
      - TAVILY_API_KEY set  → Tavily  (recommended, 1,000 free searches/month)
      - BRAVE_API_KEY set   → Brave   (2,000 free searches/month)
      - Neither set         → DuckDuckGo with retry (free, no key needed)
    """
    max_results = min(max_results, 10)

    if os.environ.get("TAVILY_API_KEY"):
        result = _search_tavily(query, max_results)
        if not result.startswith("UNAVAILABLE:") and not result.startswith("Tavily search failed"):
            return result

    if os.environ.get("BRAVE_API_KEY"):
        result = _search_brave(query, max_results)
        if not result.startswith("Brave search failed"):
            return result

    return _search_duckduckgo(query, max_results)


# ---------------------------------------------------------------------------
# File tools
# ---------------------------------------------------------------------------

@tool
def read_file(file_path: str) -> str:
    """Read a text or markdown file and return its content."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        if len(content) > 8000:
            content = content[:8000] + "\n... [truncated]"
        return content
    except Exception as e:
        return f"Error reading file: {e}"


@tool
def list_files(directory: str = "./data/uploads") -> str:
    """List all files in a directory."""
    try:
        files = os.listdir(directory)
        if not files:
            return f"No files found in {directory}"
        return "\n".join([f"- {f}" for f in files])
    except Exception as e:
        return f"Error listing directory: {e}"


# ---------------------------------------------------------------------------
# Code execution (sandboxed)
# ---------------------------------------------------------------------------

_BLOCKED_IMPORTS = {"os", "sys", "subprocess", "shutil", "pathlib", "socket", "requests", "urllib", "http"}


@tool
def execute_python(code: str) -> str:
    """Execute Python code for data analysis (math, statistics, string processing). No file system or network access."""
    for mod in _BLOCKED_IMPORTS:
        if f"import {mod}" in code or f"from {mod}" in code:
            return f"Blocked: '{mod}' is not allowed. Only use math/statistics/data analysis libraries."
    try:
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=15,
            env={"PATH": os.environ.get("PATH", "")},
        )
        output = result.stdout
        if result.stderr:
            output += f"\nSTDERR: {result.stderr}"
        return output if output else "Code executed with no output."
    except subprocess.TimeoutExpired:
        return "Execution timed out (15s limit)."
    except Exception as e:
        return f"Execution error: {e}"


@tool
def save_output(filename: str, content: str, directory: str = "./outputs") -> str:
    """Save content to a file in the outputs directory."""
    try:
        os.makedirs(directory, exist_ok=True)
        path = os.path.join(directory, filename)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return f"Saved to {path}"
    except Exception as e:
        return f"Save failed: {e}"


ALL_TOOLS = [web_search, read_file, list_files, execute_python, save_output]
TOOL_MAP = {t.name: t for t in ALL_TOOLS}
