from agents.base import BaseAgent
from tools.toolkit import save_output
from rag.pipeline import RAGPipeline
from typing import Optional


class WriterAgent(BaseAgent):
    def __init__(
        self,
        rag: Optional[RAGPipeline] = None,
        model: Optional[str] = None,
        temperature_offset: float = 0.0,
        rag_k: int = 4,
    ):
        super().__init__(
            name="Writer",
            role="Research Writer",
            tools=[save_output],
            rag=rag,
            temperature=0.5,
            model=model,
            temperature_offset=temperature_offset,
            rag_k=rag_k,
        )

    def system_prompt(self) -> str:
        return """You are an expert Research Writer. Your job is to:
1. Synthesize research findings and analysis into clear, coherent reports
2. Write with precision, clarity, and appropriate depth
3. Structure content with proper headings and flow
4. Highlight the most important insights prominently
5. Make complex information accessible without oversimplifying
6. Include executive summaries for long reports

Write in a professional but readable style.
Always structure your output as: Executive Summary, Introduction, Main Findings, Analysis, Conclusions, Recommendations."""
