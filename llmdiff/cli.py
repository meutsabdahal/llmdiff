from __future__ import annotations
import json
import asyncio
from pathlib import Path
from typing import Optional

import httpx
import typer
from rich.console import Console
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
)

from llmdiff.config import ModelConfig, SideConfig, TestCase, RunConfig
from llmdiff.runner import run_case, check_models_available
from llmdiff.differ import compute_diff
from llmdiff.metrics import semantic_similarity, compute_summary
from llmdiff.renderers.terminal import render_case_inline, render_summary
from llmdiff.renderers.json_ import render_json
from llmdiff.renderers.html import render_html

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
    # --- model flags ---
    model: str = typer.Option(
        "llama3.2",
        "--model",
        help="Model for both sides (ignored if --model-a / --model-b are set)",
    ),
    model_a: Optional[str] = typer.Option(
        None, "--model-a", help="Model for side A (overrides --model)"
    ),
    model_b: Optional[str] = typer.Option(
        None, "--model-b", help="Model for side B (overrides --model)"
    ),
    base_url: str = typer.Option(
        "http://localhost:11434", "--base-url", help="Ollama base URL"
    ),
    temperature: Optional[float] = typer.Option(None, "--temperature"),
    concurrency: int = typer.Option(3, "--concurrency"),
    no_semantic: bool = typer.Option(False, "--no-semantic"),
    filter_changed: bool = typer.Option(False, "--filter"),
    threshold: Optional[float] = typer.Option(None, "--threshold"),
    output_format: str = typer.Option("inline", "--format"),
    output: Optional[Path] = typer.Option(None, "--output"),
):
    """
    Compare two LLM prompt configurations across a set of test cases.

    Compare two prompts on the same model:\n
        llmdiff --prompt-a v1.txt --prompt-b v2.txt --inputs cases.json --model llama3.2\n

    Compare two models on the same prompt:\n
        llmdiff --prompt-a prompt.txt --prompt-b prompt.txt --model-a llama3.2 --model-b mistral --inputs cases.json
    """
    resolved_model_a = model_a or model
    resolved_model_b = model_b or model

    if resolved_model_a == resolved_model_b and prompt_a == prompt_b:
        console.print(
            "[yellow]Warning:[/yellow] Both sides are identical "
            "(same prompt file, same model). Results will show no diff."
        )

    model_cfg_a = ModelConfig(
        model=resolved_model_a,
        base_url=base_url,
        temperature=temperature,
    )
    model_cfg_b = ModelConfig(
        model=resolved_model_b,
        base_url=base_url,
        temperature=temperature,
    )

    run_cfg = RunConfig(
        side_a=SideConfig(prompt=_load_prompt(prompt_a), model_cfg=model_cfg_a),
        side_b=SideConfig(prompt=_load_prompt(prompt_b), model_cfg=model_cfg_b),
        cases=_load_cases(inputs),
        concurrency=concurrency,
        semantic=not no_semantic,
        output_format=output_format,
        filter_changed=filter_changed or (threshold is not None),
        threshold=threshold,
    )
    asyncio.run(_run(run_cfg, output_path=output))


async def _run(cfg: RunConfig, output_path: Optional[Path] = None):
    # Build labels that are informative for both use cases:
    # - same model, different prompts: show "prompt-a / llama3.2" vs "prompt-b / llama3.2"
    # - different models, same prompt: show "prompt-a / llama3.2" vs "prompt-b / mistral"
    label_a = f"prompt-a  [{cfg.side_a.model_cfg.model}]"
    label_b = f"prompt-b  [{cfg.side_b.model_cfg.model}]"

    # Collect unique models (could be 1 or 2)
    models_needed = list(
        {
            cfg.side_a.model_cfg.model,
            cfg.side_b.model_cfg.model,
        }
    )

    try:
        async with httpx.AsyncClient() as client:
            await check_models_available(
                client, cfg.side_a.model_cfg.base_url, models_needed
            )
    except RuntimeError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)

    results = []
    semaphore = asyncio.Semaphore(cfg.concurrency)
    async with httpx.AsyncClient() as client:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
            transient=True,
        ) as progress:
            task = progress.add_task(
                f"Running {len(cfg.cases)} cases...", total=len(cfg.cases)
            )

            async def run_and_track(case: TestCase):
                result = await run_one(cfg, case, client, semaphore)
                progress.advance(task, 1)
                progress.update(task, description=f"Done: {case.id}")
                return result

            try:
                results = await asyncio.gather(
                    *[run_and_track(case) for case in cfg.cases]
                )
            except RuntimeError as e:
                console.print(f"[red]Error:[/red] {e}")
                raise typer.Exit(1)

    display = [r for r in results if r.changed] if cfg.filter_changed else results
    summary = compute_summary(results)

    if cfg.output_format == "json":
        out = render_json(results, summary)
        if output_path:
            if output_path.suffix.lower() == ".html":
                output_path.write_text(render_html(results, summary))
            else:
                output_path.write_text(out)
            console.print(f"[dim]Report saved to {output_path}[/dim]")
        else:
            print(out)
        return

    for result in display:
        render_case_inline(result, label_a=label_a, label_b=label_b)

    render_summary(summary)

    if output_path:
        if output_path.suffix.lower() == ".html":
            output_path.write_text(render_html(results, summary))
        else:
            output_path.write_text(render_json(results, summary))
        console.print(f"[dim]Report saved to {output_path}[/dim]")


async def run_one(
    cfg: RunConfig,
    case: TestCase,
    client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
):
    """Run one case, then compute diff and semantic similarity if enabled."""
    resp_a, resp_b = await run_case(client, semaphore, cfg, case)

    sim = None
    if cfg.semantic:
        loop = asyncio.get_running_loop()
        sim = await loop.run_in_executor(None, semantic_similarity, resp_a, resp_b)

    return compute_diff(
        case_id=case.id,
        response_a=resp_a,
        response_b=resp_b,
        similarity=sim,
        threshold=cfg.threshold,
    )
