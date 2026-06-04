from agents.base import BaseAgent
from rag.pipeline import RAGPipeline
from typing import Optional


class CriticAgent(BaseAgent):
    def __init__(
        self,
        rag: Optional[RAGPipeline] = None,
        model: Optional[str] = None,
        temperature_offset: float = 0.0,
        rag_k: int = 4,
    ):
        super().__init__(
            name="Critic",
            role="Quality Reviewer",
            tools=[],
            rag=rag,
            temperature=0.2,
            model=model,
            temperature_offset=temperature_offset,
            rag_k=rag_k,
        )

    def system_prompt(self) -> str:
        return """You are a rigorous Quality Reviewer and Devil's Advocate. Your job is to:
1. Review the draft report for accuracy and completeness
2. Challenge assumptions and weak arguments
3. Identify missing information or perspectives
4. Check logical consistency
5. Suggest specific improvements
6. Rate the overall quality (1-10) with justification

Be constructively critical. Do not simply praise.
Return your response in this exact format:
- Strengths: [list strengths]
- Critical Issues: [list issues]
- Missing Elements: [list gaps]
- Specific Suggestions: [list improvements]
- SCORE:[number] (e.g. SCORE:7)

The SCORE line must appear exactly as shown at the end of your response."""
