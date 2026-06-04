from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from langchain_core.tools import BaseTool
from langchain_core.messages import SystemMessage, HumanMessage
import json
import re
import time

from memory.manager import AgentMemory
from rag.pipeline import RAGPipeline
from config.providers import get_chat_model


class DailyLimitError(RuntimeError):
    """Raised when the provider's per-DAY token quota is exhausted — retrying
    won't help, so we surface a clear message instead of hanging."""


def _invoke_with_retry(llm, messages, max_retries: int = 4):
    """
    Invoke the LLM, retrying on per-minute rate limits (429) with backoff.
    A per-DAY limit (or any wait longer than ~2 min) fails fast with a clear
    DailyLimitError rather than retrying pointlessly.
    """
    for attempt in range(max_retries):
        try:
            return llm.invoke(messages)
        except Exception as exc:
            msg = str(exc)
            low = msg.lower()
            is_rate_limit = (
                "429" in msg
                or "rate_limit_exceeded" in low
                or "rate limit" in low
            )
            if not is_rate_limit:
                raise

            # Daily quota exhausted → no point waiting (resets hours later)
            if "per day" in low or "tpd" in low or re.search(r"try again in\s+\d+\s*m", low):
                raise DailyLimitError(
                    "The free-tier daily token limit for this model has been "
                    "reached. Switch to a different model in the sidebar "
                    "(e.g. llama-3.1-8b-instant), wait for the daily reset, or "
                    "upgrade your Groq plan."
                ) from exc

            if attempt < max_retries - 1:
                # Per-minute limit: wait the suggested delay and retry
                m = re.search(r"try again in\s+(\d+\.?\d*)\s*s", msg, re.IGNORECASE)
                wait = float(m.group(1)) + 2 if m else (2 ** attempt) * 10
                wait = min(wait, 60)
                time.sleep(wait)
                continue
            raise


def _parse_text_tool_calls(content: str, tool_names: set) -> list:
    """
    Fallback: some local models narrate tool calls as JSON text instead of
    using native tool_calls. Scan the response for JSON objects whose "name"
    matches a known tool and extract the arguments.
    """
    calls = []
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", content):
        pos = match.start()
        try:
            obj, _ = decoder.raw_decode(content, pos)
            if isinstance(obj, dict) and obj.get("name") in tool_names:
                # Support "parameters", "arguments", or "args" as the key
                args = obj.get("parameters") or obj.get("arguments") or obj.get("args") or {}
                if isinstance(args, dict):
                    calls.append({"name": obj["name"], "args": args})
        except (json.JSONDecodeError, ValueError):
            continue
    return calls


class BaseAgent(ABC):
    def __init__(
        self,
        name: str,
        role: str,
        tools: List[BaseTool] = None,
        rag: Optional[RAGPipeline] = None,
        temperature: float = 0.3,
        model: Optional[str] = None,
        temperature_offset: float = 0.0,
        rag_k: int = 4,
    ):
        self.name = name
        self.role = role
        self.tools = tools or []
        self.rag = rag
        self.rag_k = rag_k
        self.memory = AgentMemory(agent_id=name.lower().replace(" ", "_"))

        # A global "response style" offset shifts every agent's tuned base
        # temperature up or down while preserving their relative ordering
        # (Analyst stays the most precise, Writer the most creative).
        effective_temp = max(0.0, min(1.0, temperature + temperature_offset))

        # Provider chosen automatically: Groq if GROQ_API_KEY is set, else Ollama
        self.llm = get_chat_model(temperature=effective_temp, model=model)

        if self.tools:
            self.llm_with_tools = self.llm.bind_tools(self.tools)
        else:
            self.llm_with_tools = self.llm

    @abstractmethod
    def system_prompt(self) -> str:
        pass

    def _build_context(self, task: str) -> str:
        # Truncate to avoid exceeding the embedding model's context limit.
        # nomic-embed-text supports ~8192 tokens; 500 chars is a safe retrieval query.
        retrieval_query = task[:500]
        context_parts = []

        if self.rag and self.rag.collection_count() > 0:
            docs = self.rag.retrieve(retrieval_query, k=self.rag_k)
            if docs:
                rag_context = "\n\n".join([d.page_content for d in docs])
                context_parts.append(f"<retrieved_documents>\n{rag_context}\n</retrieved_documents>")

        # Only call the embedding model if there are actually memories stored.
        # Calling similarity_search on an empty collection still hits the Ollama
        # embedding API and can hang if the model is slow to respond.
        if self.memory.has_memories():
            memories = self.memory.recall_relevant(retrieval_query)
            if memories:
                context_parts.append(f"<past_memories>\n{memories}\n</past_memories>")

        return "\n\n".join(context_parts)

    @staticmethod
    def _sanitize_args(args: dict) -> dict:
        """Remove args where the model emitted a JSON-schema dict instead of a
        real value (e.g. max_results = {"type": "integer"} instead of 5).
        The tool will fall back to its own parameter defaults for missing keys."""
        return {
            k: v for k, v in args.items()
            if not (isinstance(v, dict) and "type" in v)
        }

    def _execute_tools(self, tool_calls: list) -> list:
        """Run a list of tool call dicts [{name, args}] and return result strings."""
        tool_map = {t.name: t for t in self.tools}
        results = []
        for tc in tool_calls:
            tool = tool_map.get(tc["name"])
            if tool:
                clean_args = self._sanitize_args(tc["args"])
                try:
                    result = tool.invoke(clean_args)
                except Exception as exc:
                    result = f"Tool error ({tc['name']}): {exc}"
                results.append(f"[{tc['name']}]: {result}")
        return results

    def run(self, task: str, context: str = "") -> str:
        self.memory.short_term.add("human", task)

        extra_context = self._build_context(task)
        full_input = task
        if extra_context:
            full_input = f"{extra_context}\n\nTask: {task}"
        if context:
            full_input = f"{context}\n\n{full_input}"

        messages = [
            SystemMessage(content=self.system_prompt()),
            *self.memory.short_term.get_messages()[:-1],
            HumanMessage(content=full_input),
        ]

        # Primary LLM call — with rate-limit retry AND tool_use_failed fallback.
        try:
            response = _invoke_with_retry(self.llm_with_tools, messages)
        except Exception as _tool_err:
            _msg = str(_tool_err).lower()
            if "tool_use_failed" in _msg or ("400" in _msg and "tool" in _msg):
                response = _invoke_with_retry(self.llm, messages)
            else:
                raise

        output = response.content

        # --- Primary path: native tool_calls from the model ---
        tool_results = []
        if hasattr(response, "tool_calls") and response.tool_calls:
            native_calls = [{"name": tc["name"], "args": tc["args"]} for tc in response.tool_calls]
            tool_results = self._execute_tools(native_calls)

        # --- Fallback: model narrated the tool call as JSON text ---
        if not tool_results and self.tools:
            tool_names = {t.name for t in self.tools}
            text_calls = _parse_text_tool_calls(output, tool_names)
            if text_calls:
                tool_results = self._execute_tools(text_calls)

        # If any tools ran, send results back for a final synthesis
        if tool_results:
            tool_output = "\n".join(tool_results)
            follow_up = messages + [
                response,
                HumanMessage(
                    content=f"Tool results:\n{tool_output}\n\nNow provide your final answer based on these results."
                ),
            ]
            response = _invoke_with_retry(self.llm, follow_up)
            output = response.content

        self.memory.short_term.add("ai", output)
        return output
