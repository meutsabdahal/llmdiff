from __future__ import annotations
import json
import asyncio
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

from llmdiff.config import ModelConfig, SideConfig, TestCase, RunConfig
from llmdiff.runner import run_all
from llmdiff.differ import compute_diff
from llmdiff.metrics import semantic_similarity, compute_summary
from llmdiff.renderers.terminal import render_case_inline, render_summary
from llmdiff.renderers.json_ import render_json

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
    try:
        raw = json.loads(path.read_text())
    except json.JSONDecodeError as e:
        typer.echo(f"Error: invalid JSON in {path}: {e}", err=True)
        raise typer.Exit(1)
    return [TestCase(**c) for c in raw]


@app.command()
def main(
    prompt_a: Path = typer.Option(..., "--prompt-a", help="System prompt file A"),
    prompt_b: Path = typer.Option(..., "--prompt-b", help="System prompt file B"),
    inputs: Path = typer.Option(..., "--inputs", help="Test cases JSON file"),
    model: str = typer.Option("llama3.2", "--model", help="Ollama model name"),
    base_url: str = typer.Option(
        "http://localhost:11434", "--base-url", help="Ollama base URL"
    ),
    temperature: Optional[float] = typer.Option(None, "--temperature"),
    concurrency: int = typer.Option(3, "--concurrency", help="Parallel case limit"),
    no_semantic: bool = typer.Option(False, "--no-semantic", help="Skip embedding similarity"),
    filter_changed: bool = typer.Option(False, "--filter", help="Only show changed cases"),
    threshold: Optional[float] = typer.Option(
        None, "--threshold",
        help="Flag as changed if similarity drops below this value"
    ),
    output_format: str = typer.Option("inline", "--format", help="inline | side-by-side | json"),
    output: Optional[Path] = typer.Option(None, "--output", help="Save JSON report to file"),
):
    """
    Compare two system prompts across a set of test cases using a local Ollama model.

    Example:\n
        llmdiff --prompt-a v1.txt --prompt-b v2.txt --inputs cases.json --model llama3.2
    """
    model_cfg = ModelConfig(
        model=model,
        base_url=base_url,
        temperature=temperature,
    )

    run_cfg = RunConfig(
        side_a=SideConfig(prompt=_load_prompt(prompt_a), model_cfg=model_cfg),
        side_b=SideConfig(prompt=_load_prompt(prompt_b), model_cfg=model_cfg),
        cases=_load_cases(inputs),
        concurrency=concurrency,
        semantic=not no_semantic,
        output_format=output_format,
        filter_changed=filter_changed or (threshold is not None),
        threshold=threshold,
    )

    asyncio.run(_run(run_cfg, output_path=output))


async def _run(cfg: RunConfig, output_path: Optional[Path] = None):
    label_a = f"prompt-a / {cfg.side_a.model_cfg.model}"
    label_b = f"prompt-b / {cfg.side_b.model_cfg.model}"

    # Check Ollama is reachable before starting
    try:
        import httpx
        async with httpx.AsyncClient() as client:
            r = await client.get(
                f"{cfg.side_a.model_cfg.base_url}/api/tags", timeout=5.0
            )
            r.raise_for_status()
    except Exception:
        console.print(
            f"[red]Error:[/red] Cannot reach Ollama at "
            f"{cfg.side_a.model_cfg.base_url}. Is it running?"
        )
        raise typer.Exit(1)

    with console.status(f"[dim]Running {len(cfg.cases)} cases...[/dim]"):
        raw_results = await run_all(cfg)

    results = []
    for case, resp_a, resp_b in raw_results:
        sim = None
        if cfg.semantic:
            loop = asyncio.get_event_loop()
            sim = await loop.run_in_executor(
                None, semantic_similarity, resp_a, resp_b
            )
        result = compute_diff(
            case_id=case.id,
            response_a=resp_a,
            response_b=resp_b,
            similarity=sim,
            threshold=cfg.threshold,
        )
        results.append(result)

    display = [r for r in results if r.changed] if cfg.filter_changed else results
    summary = compute_summary(results)

    if cfg.output_format == "json":
        out = render_json(results, summary)
        if output_path:
            output_path.write_text(out)
            console.print(f"[dim]Saved to {output_path}[/dim]")
        else:
            print(out)
        return

    for result in display:
        render_case_inline(result, label_a=label_a, label_b=label_b)

    render_summary(summary)

    if output_path:
        output_path.write_text(render_json(results, summary))
        console.print(f"[dim]JSON report saved to {output_path}[/dim]")