#!/usr/bin/env python3
"""
Multi-Agent Research System
CLI entry point
"""
import typer
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt, Confirm
from rich import print as rprint

from workflow import run_research
from rag.pipeline import RAGPipeline
from config.settings import UPLOAD_DIR, OUTPUT_DIR

app = typer.Typer(help="Multi-Agent Research System powered by local LLMs")
console = Console()


def print_banner():
    console.print(Panel.fit(
        "[bold cyan]Multi-Agent Research System[/bold cyan]\n"
        "[dim]Researcher → Analyst → Writer → Critic[/dim]\n"
        "[dim]Powered by Ollama + LangGraph + ChromaDB[/dim]",
        border_style="cyan"
    ))


@app.command()
def research(
    topic: str = typer.Argument(None, help="Research topic"),
    ingest: bool = typer.Option(False, "--ingest", "-i", help="Ingest documents from uploads/ first"),
    no_rag: bool = typer.Option(False, "--no-rag", help="Skip RAG, use web search only"),
):
    """Run a full multi-agent research workflow on a topic."""
    print_banner()

    rag = None if no_rag else RAGPipeline()

    if ingest and rag:
        console.print(f"\n[cyan]Ingesting documents from {UPLOAD_DIR}...[/cyan]")
        count = rag.ingest_directory()
        console.print(f"[green]✓ Ingested {count} chunks[/green]\n")

    if not topic:
        topic = Prompt.ask("[bold]Enter your research topic[/bold]")

    if not topic.strip():
        console.print("[red]No topic provided.[/red]")
        raise typer.Exit(1)

    result = run_research(topic, rag=rag)
    report = result.get("report", "") if isinstance(result, dict) else result
    citations = result.get("citations", []) if isinstance(result, dict) else []

    console.print("\n" + "="*60)
    console.print(Panel(report[:2000] + ("..." if len(report) > 2000 else ""), title="[bold]Final Report Preview[/bold]", border_style="green"))
    if citations:
        console.print(f"\n[bold]Sources found:[/bold] {len(citations)} URLs")
    console.print(f"\n[dim]Full report saved to {OUTPUT_DIR}/[/dim]")


@app.command()
def ingest(
    path: str = typer.Argument(None, help="File or directory to ingest"),
):
    """Ingest documents into the vector database."""
    print_banner()
    rag = RAGPipeline()

    target = path or UPLOAD_DIR
    if os.path.isfile(target):
        count = rag.ingest_file(target)
        console.print(f"[green]✓ Ingested {count} chunks from {target}[/green]")
    elif os.path.isdir(target):
        count = rag.ingest_directory(target)
        console.print(f"[green]✓ Ingested {count} total chunks from {target}[/green]")
    else:
        console.print(f"[red]Path not found: {target}[/red]")


@app.command()
def status():
    """Show system status and document count."""
    print_banner()
    rag = RAGPipeline()
    doc_count = rag.collection_count()

    console.print(f"\n[bold]System Status[/bold]")
    console.print(f"  Documents in vector DB: [cyan]{doc_count}[/cyan]")
    console.print(f"  Upload directory:       [cyan]{UPLOAD_DIR}[/cyan]")
    console.print(f"  Output directory:       [cyan]{OUTPUT_DIR}[/cyan]")

    uploads = os.listdir(UPLOAD_DIR)
    if uploads:
        console.print(f"\n  Files in uploads/:")
        for f in uploads:
            console.print(f"    - {f}")
    else:
        console.print(f"\n  [dim]No files in uploads/ yet. Drop PDFs or .txt files there.[/dim]")

    outputs = os.listdir(OUTPUT_DIR)
    if outputs:
        console.print(f"\n  Generated reports:")
        for f in outputs:
            console.print(f"    - {f}")


if __name__ == "__main__":
    app()
