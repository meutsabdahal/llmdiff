from __future__ import annotations
import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Optional

import httpx
import typer
from pydantic import ValidationError
from rich.console import Console
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
)

from llmdiff.config import ModelConfig, SideConfig, TestCase, RunConfig, OutputFormat
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


def _load_local_env() -> None:
    """Loads .env variables if they are not already set in the shell."""
    candidates = [
        Path.cwd() / ".env",
        Path(__file__).resolve().parents[1] / ".env",
    ]
    env_path = next((p for p in candidates if p.exists()), None)
    if env_path is None:
        return

    for raw_line in env_path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def _configure_model_logging() -> None:
    """Reduces noisy transformer/hub warnings while keeping errors visible."""
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

    for logger_name in ("transformers", "huggingface_hub", "sentence_transformers"):
        logging.getLogger(logger_name).setLevel(logging.ERROR)

    try:
        from transformers.utils import logging as transformers_logging

        transformers_logging.set_verbosity_error()
        transformers_logging.disable_progress_bar()
    except Exception:
        # transformers is optional until semantic scoring is used
        pass


def _bootstrap_runtime_env() -> None:
    _load_local_env()
    _configure_model_logging()


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

    if not isinstance(raw, list):
        typer.echo(
            f"Error: {path} must contain a JSON array of test cases.",
            err=True,
        )
        raise typer.Exit(1)

    cases: list[TestCase] = []
    for i, case_raw in enumerate(raw):
        if not isinstance(case_raw, dict):
            typer.echo(
                f"Error: test case at index {i} must be a JSON object.",
                err=True,
            )
            raise typer.Exit(1)

        try:
            cases.append(TestCase(**case_raw))
        except ValidationError as e:
            first = e.errors()[0]
            loc = ".".join(str(p) for p in first.get("loc", ()))
            msg = first.get("msg", "invalid value")
            case_id = case_raw.get("id")
            case_hint = f" (id={case_id})" if isinstance(case_id, str) else ""
            typer.echo(
                f"Error: invalid test case at index {i}{case_hint}: {loc}: {msg}",
                err=True,
            )
            raise typer.Exit(1)

    if not cases:
        typer.echo(
            f"Error: {path} must contain at least one test case.",
            err=True,
        )
        raise typer.Exit(1)

    return cases


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
    temperature_a: Optional[float] = typer.Option(None, "--temperature-a"),
    temperature_b: Optional[float] = typer.Option(None, "--temperature-b"),
    temperature: Optional[float] = typer.Option(
        None,
        "--temperature",
        help="Temperature for both sides (overridden by --temperature-a / --temperature-b)",
    ),
    concurrency: int = typer.Option(
        3,
        "--concurrency",
        min=1,
        help="Maximum number of test cases to run concurrently (must be >= 1)",
    ),
    no_semantic: bool = typer.Option(False, "--no-semantic"),
    filter_changed: bool = typer.Option(False, "--filter"),
    threshold: Optional[float] = typer.Option(
        None,
        "--threshold",
        min=0.0,
        max=1.0,
        help="Mark results as changed when similarity is below this value (0.0-1.0)",
    ),
    output_format: OutputFormat = typer.Option(
        OutputFormat.INLINE,
        "--format",
        case_sensitive=False,
        help="Output format: inline, json, or html",
    ),
    output: Optional[Path] = typer.Option(None, "--output"),
):
    """
    Compare two LLM prompt configurations across a set of test cases.

    Compare two prompts on the same model:\n
        llmdiff --prompt-a v1.txt --prompt-b v2.txt --inputs cases.json --model llama3.2\n

    Compare two models on the same prompt:\n
        llmdiff --prompt-a prompt.txt --prompt-b prompt.txt --model-a llama3.2 --model-b mistral --inputs cases.json
    """
    _bootstrap_runtime_env()

    if output is not None and output_format == OutputFormat.INLINE:
        typer.echo("Error: --output requires --format json or --format html.", err=True)
        raise typer.Exit(1)

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
        temperature=temperature_a if temperature_a is not None else temperature,
    )
    model_cfg_b = ModelConfig(
        model=resolved_model_b,
        base_url=base_url,
        temperature=temperature_b if temperature_b is not None else temperature,
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

    if cfg.output_format == OutputFormat.JSON:
        out = render_json(results, summary)
        if output_path:
            output_path.write_text(out)
            console.print(f"[dim]Report saved to {output_path}[/dim]")
        else:
            print(out)
        return

    if cfg.output_format == OutputFormat.HTML:
        out = render_html(results, summary)
        if output_path:
            output_path.write_text(out)
            console.print(f"[dim]Report saved to {output_path}[/dim]")
        else:
            print(out)
        return

    for result in display:
        render_case_inline(result, label_a=label_a, label_b=label_b)

    render_summary(summary)


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
