from agents.base import BaseAgent
from tools.toolkit import execute_python
from rag.pipeline import RAGPipeline
from typing import Optional


class AnalystAgent(BaseAgent):
    def __init__(
        self,
        rag: Optional[RAGPipeline] = None,
        model: Optional[str] = None,
        temperature_offset: float = 0.0,
        rag_k: int = 4,
    ):
        super().__init__(
            name="Analyst",
            role="Data & Logic Analyst",
            tools=[execute_python],
            rag=rag,
            temperature=0.1,
            model=model,
            temperature_offset=temperature_offset,
            rag_k=rag_k,
        )

    def system_prompt(self) -> str:
        return """You are a rigorous Data & Logic Analyst. Your job is to:
1. Analyze research findings critically
2. Identify patterns, trends, and correlations
3. Run calculations or data analysis using Python when needed
4. Evaluate the strength of evidence
5. Spot contradictions, biases, or weak arguments
6. Quantify findings where possible

Be critical and objective. Always explain your reasoning.
Return structured analysis: Key Patterns, Evidence Quality, Gaps, Quantitative Insights, Conclusions."""
