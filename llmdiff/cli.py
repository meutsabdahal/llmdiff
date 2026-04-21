from __future__ import annotations
import json
import asyncio
from pathlib import Path
from typing import Optional
import typer
from rich.console import Console
from rich.panel import Panel

from llmdiff.config import ModelConfig, SideConfig, TestCase, RunConfig
from llmdiff.runner import run_all


app = typer.Typer(
    name="llmdiff",
    help="git diff for LLM prompts",
    add_completion=False,
)
console = Console()


def _load_prompt(path: Path) -> str:
    if not path.exists():
        typer.echo(f"Error: prompt file not found: {path}", err=True)
        raise typer.Exit(1)
    return path.read_text().strip()


def _load_cases(path: Path) -> list[TestCase]:
    if not path.exists():
        typer.echo(f"Error: inputs file not found: {path}", err=True)
        raise typer.Exit(1)
    raw = json.loads(path.read_text())
    return [TestCase(**c) for c in raw]


@app.command()
def main(
    prompt_a: Path = typer.Option(..., "--prompt-a", help="System prompt A"),
    prompt_b: Path = typer.Option(..., "--prompt-b", help="System prompt B"),
    inputs: Path = typer.Option(..., "--inputs", help="Test cases JSON"),
    model: str = typer.Option("gpt-4o-mini", "--model"),
    model_a: Optional[str] = typer.Option(None, "--model-a"),
    model_b: Optional[str] = typer.Option(None, "--model-b"),
    provider: str = typer.Option("ollama", "--provider"),
    base_url: str = typer.Option("http://localhost:11434", "--base-url"),
    temperature: Optional[float] = typer.Option(None, "--temperature"),
    param_a: Optional[str] = typer.Option(
        None, "--param-a", help="e.g. temperature=0.0"
    ),
    param_b: Optional[str] = typer.Option(None, "--param-b"),
    concurrency: int = typer.Option(5, "--concurrency"),
    no_semantic: bool = typer.Option(False, "--no-semantic"),
    filter_changed: bool = typer.Option(False, "--filter"),
    threshold: Optional[float] = typer.Option(None, "--threshold"),
    output_format: str = typer.Option("inline", "--format"),
    output: Optional[Path] = typer.Option(None, "--output"),
):
    """Compare two LLM prompt configurations across a set of test cases."""

    def _parse_param(s: Optional[str]) -> dict:
        if not s:
            return {}
        key, _, val = s.partition("=")
        return {key.strip(): float(val) if "." in val else val.strip()}

    params_a = _parse_param(param_a)
    params_b = _parse_param(param_b)

    temp_a = params_a.get("temperature", temperature)
    temp_b = params_b.get("temperature", temperature)

    if provider != "ollama":
        typer.echo(
            "Warning: only Ollama is implemented in this build; continuing with Ollama API.",
            err=True,
        )

    cfg_a = ModelConfig(model=model_a or model, base_url=base_url, temperature=temp_a)
    cfg_b = ModelConfig(model=model_b or model, base_url=base_url, temperature=temp_b)

    system_a = _load_prompt(prompt_a)
    system_b = _load_prompt(prompt_b)
    cases = _load_cases(inputs)

    run_cfg = RunConfig(
        side_a=SideConfig(prompt=system_a, model_cfg=cfg_a),
        side_b=SideConfig(prompt=system_b, model_cfg=cfg_b),
        cases=cases,
        concurrency=concurrency,
        semantic=not no_semantic,
        output_format=output_format,
        filter_changed=filter_changed,
        threshold=threshold,
    )

    results = asyncio.run(run_all(run_cfg))
    for case, resp_a, resp_b in results:
        console.print(
            Panel(
                f"[bold]A:[/bold] {resp_a}\n\n[bold]B:[/bold] {resp_b}",
                title=f"Case: {case.id}",
            )
        )
