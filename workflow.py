from typing import TypedDict, Optional, List, Callable
from langgraph.graph import StateGraph, END
from rich.console import Console
from rich.panel import Panel
import re
import time
import os

from agents.researcher import ResearcherAgent
from agents.analyst import AnalystAgent
from agents.writer import WriterAgent
from agents.critic import CriticAgent
from agents.orchestrator import OrchestratorAgent
from rag.pipeline import RAGPipeline
from config.settings import OUTPUT_DIR, OLLAMA_MODEL
from utils.pdf_export import export_pdf

console = Console()


class ResearchState(TypedDict):
    topic: str
    research_plan: str
    research_findings: str
    analysis: str
    draft_report: str
    critique: str
    final_report: str
    citations: List[str]
    iteration: int
    status: str


# ---------------------------------------------------------------------------
# Tunable run configuration
# ---------------------------------------------------------------------------

DEFAULT_CONFIG = {
    "temperature_offset": 0.0,   # shift all agent temperatures (-0.1 .. +0.3)
    "rag_k":              4,     # how many document chunks each agent retrieves
    "use_web_search":     True,  # let the Researcher hit the web
    "max_iterations":     1,     # max critique→revise rounds (1 keeps token use low)
    "quality_threshold":  7,     # critic score (1-10) needed to skip revision
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_urls(text: str) -> List[str]:
    """Pull unique, clean URLs from web-search result text."""
    raw = re.findall(r'https?://[^\s\)\]>"\']+', text)
    cleaned = [u.rstrip(r'.,;:)>]"\'') for u in raw]
    seen: set = set()
    unique: List[str] = []
    for u in cleaned:
        if u not in seen:
            seen.add(u)
            unique.append(u)
    return unique


def _clip(text: str, max_chars: int) -> str:
    """Truncate inter-agent text to bound token usage (≈4 chars/token).
    Keeps the pipeline well under hosted free-tier daily token limits."""
    if text and len(text) > max_chars:
        return text[:max_chars] + "\n\n[...truncated for length...]"
    return text or ""


# ---------------------------------------------------------------------------
# Workflow builder
# ---------------------------------------------------------------------------

def build_workflow(
    rag: Optional[RAGPipeline] = None,
    progress_callback: Optional[Callable[[str, str], None]] = None,
    model: Optional[str] = None,
    config: Optional[dict] = None,
):
    _model = model or OLLAMA_MODEL
    cfg = {**DEFAULT_CONFIG, **(config or {})}

    temp_off = cfg["temperature_offset"]
    rag_k    = cfg["rag_k"]
    use_web  = cfg["use_web_search"]

    def _progress(step: str, msg: str) -> None:
        console.print(Panel(f"[bold]{msg}[/bold]"))
        if progress_callback:
            progress_callback(step, msg)

    orchestrator = OrchestratorAgent(rag=rag, model=_model,
                                     temperature_offset=temp_off, rag_k=rag_k)
    researcher   = ResearcherAgent(rag=rag, model=_model,
                                   temperature_offset=temp_off, rag_k=rag_k,
                                   use_web_search=use_web)
    analyst      = AnalystAgent(rag=rag, model=_model,
                                temperature_offset=temp_off, rag_k=rag_k)
    writer       = WriterAgent(rag=rag, model=_model,
                               temperature_offset=temp_off, rag_k=rag_k)
    critic       = CriticAgent(rag=rag, model=_model,
                               temperature_offset=temp_off, rag_k=rag_k)

    # ── Nodes ────────────────────────────────────────────────────────────────

    def plan_node(state: ResearchState) -> dict:
        _progress("plan", "Orchestrator: Planning research strategy...")
        plan_data = orchestrator.plan(state["topic"])
        console.print(f"[dim]{plan_data['plan'][:300]}...[/dim]")
        return {"research_plan": plan_data["plan"], "status": "planned"}

    def research_node(state: ResearchState) -> dict:
        _progress("research", "Researcher: Gathering information from the web...")
        task = (
            f"Topic: {state['topic']}\n"
            f"Plan context: {state['research_plan'][:500]}\n\n"
            "Research this topic thoroughly. Search the web and analyze available documents."
        )
        findings = researcher.run(task)
        console.print(f"[dim]{findings[:300]}...[/dim]")

        urls = _extract_urls(findings)
        _progress("research", f"Researcher: Done — found {len(urls)} sources")
        return {"research_findings": findings, "citations": urls, "status": "researched"}

    def analysis_node(state: ResearchState) -> dict:
        _progress("analyze", "Analyst: Identifying patterns and evaluating evidence...")
        task = (
            f"Topic: {state['topic']}\n"
            f"Plan context: {state['research_plan'][:300]}\n\n"
            f"Research Findings:\n{_clip(state['research_findings'], 4000)}\n\n"
            "Analyze these findings critically. Identify patterns, evaluate evidence quality, "
            "and derive insights."
        )
        analysis = analyst.run(task)
        console.print(f"[dim]{analysis[:300]}...[/dim]")
        return {"analysis": analysis, "status": "analyzed"}

    def write_node(state: ResearchState) -> dict:
        _progress("write", "Writer: Drafting the research report...")
        task = (
            f"Topic: {state['topic']}\n"
            f"Plan context: {state['research_plan'][:300]}\n\n"
            f"Research Findings:\n{_clip(state['research_findings'], 4000)}\n\n"
            f"Analysis:\n{_clip(state['analysis'], 3000)}\n\n"
            "Write a comprehensive, well-structured research report. Include all key insights."
        )
        draft = writer.run(task)
        console.print(f"[dim]{draft[:300]}...[/dim]")
        return {"draft_report": draft, "status": "drafted"}

    def critique_node(state: ResearchState) -> dict:
        _progress("review", "Critic: Reviewing and scoring the draft...")
        task = (
            f"Topic: {state['topic']}\n\n"
            f"Draft Report:\n{_clip(state['draft_report'], 5000)}\n\n"
            "Review this report critically. Evaluate quality, completeness, and accuracy."
        )
        critique = critic.run(task)
        console.print(f"[dim]{critique[:300]}...[/dim]")

        score_match = re.search(r"SCORE:(\d+)", critique, re.IGNORECASE)
        score = score_match.group(1) if score_match else "?"
        _progress("review", f"Critic: Review complete — Score: {score}/10")

        return {
            "critique": critique,
            "iteration": state["iteration"] + 1,
            "status": "critiqued",
        }

    def should_revise(state: ResearchState) -> str:
        if state["iteration"] >= cfg["max_iterations"]:
            return "finalize"
        match = re.search(r"SCORE:(\d+)", state.get("critique", ""), re.IGNORECASE)
        if match and int(match.group(1)) >= cfg["quality_threshold"]:
            return "finalize"
        return "revise"

    def revise_node(state: ResearchState) -> dict:
        _progress("revise", "Writer: Revising report based on critique...")
        task = (
            f"Topic: {state['topic']}\n\n"
            f"Original Draft:\n{_clip(state['draft_report'], 5000)}\n\n"
            f"Critique received:\n{_clip(state['critique'], 2000)}\n\n"
            "Rewrite the report addressing all critique points. "
            "Significantly improve the identified weaknesses."
        )
        revised = writer.run(task)
        return {"draft_report": revised, "status": "revised"}

    def finalize_node(state: ResearchState) -> dict:
        _progress("finalize", "Orchestrator: Synthesizing final report...")
        task = (
            f"Topic: {state['topic']}\n\n"
            f"Draft Report:\n{_clip(state['draft_report'], 6000)}\n\n"
            f"Critique:\n{_clip(state['critique'], 2000)}\n\n"
            "Synthesize the draft and critique into a final, polished research report. "
            "Incorporate all feedback, resolve any gaps, and add an executive summary."
        )
        final = orchestrator.run(task)

        # ── File paths ─────────────────────────────────────────────────────
        safe_topic = "".join(
            c if c.isalnum() or c in " -" else "_"
            for c in state["topic"][:30]
        )
        base_name = f"report_{safe_topic.strip().replace(' ', '_').lower()}"
        md_path  = os.path.join(OUTPUT_DIR, f"{base_name}.md")
        pdf_path = os.path.join(OUTPUT_DIR, f"{base_name}.pdf")

        # ── References section ─────────────────────────────────────────────
        refs_section = ""
        if state.get("citations"):
            refs_section = "\n\n## References\n\n"
            for i, url in enumerate(state["citations"], 1):
                refs_section += f"{i}. {url}\n"

        md_content = (
            f"# Research Report: {state['topic']}\n\n"
            f"{final}"
            f"{refs_section}\n\n"
            "---\n*Generated by Multi-Agent Research System*"
        )

        # ── Save markdown ──────────────────────────────────────────────────
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(md_content)

        # ── Save PDF ───────────────────────────────────────────────────────
        try:
            export_pdf(md_content, pdf_path, state["topic"])
            console.print(f"\n[bold green]Reports saved:[/bold green]")
            console.print(f"  [cyan]Markdown:[/cyan] {md_path}")
            console.print(f"  [cyan]PDF:     [/cyan] {pdf_path}")
        except Exception as e:
            console.print(f"\n[bold green]Report saved (MD):[/bold green] {md_path}")
            console.print(f"[yellow]PDF export failed: {e}[/yellow]")

        _progress("finalize", f"Done — saved to {os.path.basename(md_path)}")
        return {"final_report": final, "status": "complete"}

    # ── Graph assembly ────────────────────────────────────────────────────────
    workflow = StateGraph(ResearchState)
    workflow.add_node("plan",     plan_node)
    workflow.add_node("research", research_node)
    workflow.add_node("analyze",  analysis_node)
    workflow.add_node("write",    write_node)
    workflow.add_node("review",   critique_node)
    workflow.add_node("revise",   revise_node)
    workflow.add_node("finalize", finalize_node)

    workflow.set_entry_point("plan")
    workflow.add_edge("plan",     "research")
    workflow.add_edge("research", "analyze")
    workflow.add_edge("analyze",  "write")
    workflow.add_edge("write",    "review")
    workflow.add_conditional_edges(
        "review",
        should_revise,
        {"revise": "revise", "finalize": "finalize"},
    )
    workflow.add_edge("revise",   "review")
    workflow.add_edge("finalize", END)

    return workflow.compile()


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_research(
    topic: str,
    rag: Optional[RAGPipeline] = None,
    progress_callback: Optional[Callable[[str, str], None]] = None,
    model: Optional[str] = None,
    config: Optional[dict] = None,
) -> dict:
    """
    Run the full multi-agent research pipeline.

    config (optional) overrides any of DEFAULT_CONFIG:
        temperature_offset, rag_k, use_web_search,
        max_iterations, quality_threshold

    Returns a dict with keys:
        report          – final report text
        citations       – list of source URLs found during research
        duration_seconds – wall-clock time
        topic           – the original topic
    """
    graph = build_workflow(rag=rag, progress_callback=progress_callback,
                           model=model, config=config)

    initial_state: ResearchState = {
        "topic": topic,
        "research_plan": "",
        "research_findings": "",
        "analysis": "",
        "draft_report": "",
        "critique": "",
        "final_report": "",
        "citations": [],
        "iteration": 0,
        "status": "starting",
    }

    console.print(
        f"\n[bold white]Starting research on:[/bold white] "
        f"[bold yellow]{topic}[/bold yellow]\n"
    )
    start = time.time()
    final_state = graph.invoke(initial_state)
    elapsed = time.time() - start

    console.print(f"\n[bold green]Research complete in {elapsed:.1f}s[/bold green]")

    return {
        "report":           final_state.get("final_report", ""),
        "citations":        final_state.get("citations", []),
        "duration_seconds": elapsed,
        "topic":            topic,
    }
