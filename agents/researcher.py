from agents.base import BaseAgent
from tools.toolkit import web_search, read_file, list_files
from rag.pipeline import RAGPipeline
from typing import Optional


class ResearcherAgent(BaseAgent):
    def __init__(
        self,
        rag: Optional[RAGPipeline] = None,
        model: Optional[str] = None,
        temperature_offset: float = 0.0,
        rag_k: int = 4,
        use_web_search: bool = True,
    ):
        # Optionally drop the web_search tool (RAG / offline-only mode)
        tools = [web_search, read_file, list_files] if use_web_search \
            else [read_file, list_files]
        super().__init__(
            name="Researcher",
            role="Research Specialist",
            tools=tools,
            rag=rag,
            temperature=0.2,
            model=model,
            temperature_offset=temperature_offset,
            rag_k=rag_k,
        )

    def system_prompt(self) -> str:
        return """You are an expert Research Specialist. Your job is to:
1. Gather comprehensive information on the given topic
2. Search the web for current and relevant data
3. Read and analyze documents from the knowledge base
4. Extract key facts, statistics, and insights
5. Identify gaps in existing knowledge

Always cite your sources. Be thorough but concise.
Return structured findings with clear sections: Background, Key Findings, Data Points, Sources."""
