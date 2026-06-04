from agents.base import BaseAgent
from rag.pipeline import RAGPipeline
from typing import Optional


class OrchestratorAgent(BaseAgent):
    def __init__(
        self,
        rag: Optional[RAGPipeline] = None,
        model: Optional[str] = None,
        temperature_offset: float = 0.0,
        rag_k: int = 4,
    ):
        super().__init__(
            name="Orchestrator",
            role="Research Director",
            tools=[],
            rag=rag,
            temperature=0.3,
            model=model,
            temperature_offset=temperature_offset,
            rag_k=rag_k,
        )

    def system_prompt(self) -> str:
        return """You are the Research Director overseeing a multi-agent research team.
Your agents: Researcher, Analyst, Writer, Critic.

Given a research topic, you:
1. Break it into clear sub-tasks for each agent
2. Synthesize outputs from all agents
3. Make final decisions on conflicting information
4. Ensure the final output meets quality standards
5. Provide a concise task brief for each agent

Be directive and clear. Delegate effectively. Summarize what each agent should focus on."""

    def plan(self, topic: str) -> dict:
        """Generate a task plan for all agents."""
        prompt = f"""Research topic: {topic}

Create a specific task brief for each agent:
- Researcher: What to research and gather
- Analyst: What to analyze and evaluate  
- Writer: How to structure and present the report
- Critic: What quality standards to enforce

Return as a structured plan."""

        plan_text = self.run(prompt)

        return {
            "topic": topic,
            "plan": plan_text,
            "researcher_task": f"Research this topic comprehensively: {topic}",
            "analyst_task": f"Analyze the research findings for: {topic}",
            "writer_task": f"Write a comprehensive report on: {topic}",
            "critic_task": f"Review and critique the draft report on: {topic}",
        }
