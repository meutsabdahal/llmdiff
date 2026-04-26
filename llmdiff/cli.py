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
from llmdiff.runner import run_case, check_models_available, configure_request_policy
from llmdiff.differ import compute_diff
from llmdiff.metrics import semantic_similarity, semantic_similarities, compute_summary
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


def _collect_policy_failures(
    results: list,
    summary,
    fail_on_changed: bool,
    fail_if_avg_below: Optional[float],
    fail_if_any_below_threshold: Optional[float],
) -> list[str]:
    failures: list[str] = []

    if fail_on_changed and summary.changed > 0:
        failures.append(
            "--fail-on-changed triggered: "
            f"{summary.changed}/{summary.total} cases are marked changed."
        )

    if fail_if_avg_below is not None:
        if summary.avg_similarity is None:
            failures.append(
                "--fail-if-avg-below could not be evaluated because "
                "semantic similarity scores are unavailable."
            )
        elif summary.avg_similarity < fail_if_avg_below:
            failures.append(
                "--fail-if-avg-below triggered: "
                f"avg similarity {summary.avg_similarity:.4f} < {fail_if_avg_below:.4f}."
            )

    if fail_if_any_below_threshold is not None:
        scored = [r for r in results if r.similarity is not None]
        if not scored:
            failures.append(
                "--fail-if-any-below-threshold could not be evaluated because "
                "semantic similarity scores are unavailable."
            )
        else:
            failing = [
                r
                for r in scored
                if r.similarity is not None
                and r.similarity < fail_if_any_below_threshold
            ]
            if failing:
                worst = min(failing, key=lambda r: r.similarity)
                failures.append(
                    "--fail-if-any-below-threshold triggered: "
                    f"{len(failing)} case(s) below {fail_if_any_below_threshold:.4f}; "
                    f"worst={worst.case_id} ({worst.similarity:.4f})."
                )

    return failures


def _write_output_report(output_path: Path, content: str) -> None:
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        console.print(
            "[red]Error:[/red] Failed to create output directory "
            f"'{output_path.parent}': {e}"
        )
        raise typer.Exit(1)

    try:
        output_path.write_text(content, encoding="utf-8")
    except OSError as e:
        console.print(
            "[red]Error:[/red] Failed to write report to " f"'{output_path}': {e}"
        )
        raise typer.Exit(1)


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
    max_tokens: int = typer.Option(
        1024,
        "--max-tokens",
        min=1,
        help="Max generation tokens for both sides (overridden by --max-tokens-a / --max-tokens-b)",
    ),
    max_tokens_a: Optional[int] = typer.Option(
        None,
        "--max-tokens-a",
        min=1,
        help="Max generation tokens for side A",
    ),
    max_tokens_b: Optional[int] = typer.Option(
        None,
        "--max-tokens-b",
        min=1,
        help="Max generation tokens for side B",
    ),
    concurrency: int = typer.Option(
        3,
        "--concurrency",
        min=1,
        help="Maximum number of test cases to run concurrently (must be >= 1)",
    ),
    request_timeout: float = typer.Option(
        120.0,
        "--request-timeout",
        min=0.1,
        help="Per-request timeout in seconds for each Ollama /api/chat call.",
    ),
    retry_attempts: int = typer.Option(
        2,
        "--retry-attempts",
        min=0,
        help="Retry count for transient Ollama request failures.",
    ),
    retry_backoff_base: float = typer.Option(
        0.5,
        "--retry-backoff-base",
        min=0.0,
        help="Base seconds for exponential retry backoff (capped internally).",
    ),
    no_semantic: bool = typer.Option(False, "--no-semantic"),
    semantic_batch_size: int = typer.Option(
        24,
        "--semantic-batch-size",
        min=1,
        help="Number of response pairs to score per embedding batch",
    ),
    filter_changed: bool = typer.Option(False, "--filter"),
    threshold: Optional[float] = typer.Option(
        None,
        "--threshold",
        min=0.0,
        max=1.0,
        help="Mark results as changed when similarity is below this value (0.0-1.0)",
    ),
    fail_on_changed: bool = typer.Option(
        False,
        "--fail-on-changed",
        help="Exit with code 1 when at least one case is marked changed.",
    ),
    fail_if_avg_below: Optional[float] = typer.Option(
        None,
        "--fail-if-avg-below",
        min=0.0,
        max=1.0,
        help="Exit with code 1 when run-level avg similarity is below this value.",
    ),
    fail_if_any_below_threshold: Optional[float] = typer.Option(
        None,
        "--fail-if-any-below-threshold",
        min=0.0,
        max=1.0,
        help="Exit with code 1 when any case similarity is below this value.",
    ),
    max_lines: int = typer.Option(
        40,
        "--max-lines",
        min=0,
        help="Max response lines per side in inline output (0 = no limit)",
    ),
    max_diff_lines: int = typer.Option(
        120,
        "--max-diff-lines",
        min=0,
        help="Max diff lines per case in inline output (0 = no limit)",
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

    if no_semantic and (
        fail_if_avg_below is not None or fail_if_any_below_threshold is not None
    ):
        typer.echo(
            "Error: --fail-if-avg-below and --fail-if-any-below-threshold "
            "require semantic scoring (remove --no-semantic).",
            err=True,
        )
        raise typer.Exit(1)

    resolved_model_a = model_a or model
    resolved_model_b = model_b or model

    if resolved_model_a == resolved_model_b and prompt_a == prompt_b:
        console.print(
            "[yellow]Warning:[/yellow] Both sides are identical "
            "(same prompt file, same model). Results will show no diff."
        )

    configure_request_policy(
        request_timeout=request_timeout,
        max_retries=retry_attempts,
        retry_backoff_base=retry_backoff_base,
    )

    model_cfg_a = ModelConfig(
        model=resolved_model_a,
        base_url=base_url,
        temperature=temperature_a if temperature_a is not None else temperature,
        max_tokens=max_tokens_a if max_tokens_a is not None else max_tokens,
    )
    model_cfg_b = ModelConfig(
        model=resolved_model_b,
        base_url=base_url,
        temperature=temperature_b if temperature_b is not None else temperature,
        max_tokens=max_tokens_b if max_tokens_b is not None else max_tokens,
    )

    run_cfg = RunConfig(
        side_a=SideConfig(prompt=_load_prompt(prompt_a), model_cfg=model_cfg_a),
        side_b=SideConfig(prompt=_load_prompt(prompt_b), model_cfg=model_cfg_b),
        cases=_load_cases(inputs),
        concurrency=concurrency,
        semantic=not no_semantic,
        semantic_batch_size=semantic_batch_size,
        output_format=output_format,
        max_response_lines=max_lines,
        max_diff_lines=max_diff_lines,
        filter_changed=filter_changed or (threshold is not None),
        threshold=threshold,
    )
    asyncio.run(
        _run(
            run_cfg,
            output_path=output,
            fail_on_changed=fail_on_changed,
            fail_if_avg_below=fail_if_avg_below,
            fail_if_any_below_threshold=fail_if_any_below_threshold,
        )
    )


async def _run(
    cfg: RunConfig,
    output_path: Optional[Path] = None,
    fail_on_changed: bool = False,
    fail_if_avg_below: Optional[float] = None,
    fail_if_any_below_threshold: Optional[float] = None,
):
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
    total_steps = len(cfg.cases) + (1 if cfg.semantic else 0)

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
                f"Running {len(cfg.cases)} cases...", total=total_steps
            )

            try:
                if cfg.semantic:

                    async def run_case_and_track(case: TestCase):
                        resp_a, resp_b = await run_case(client, semaphore, cfg, case)
                        progress.advance(task, 1)
                        progress.update(task, description=f"Done: {case.id}")
                        return case, resp_a, resp_b

                    responses = await asyncio.gather(
                        *[run_case_and_track(case) for case in cfg.cases]
                    )

                    progress.update(task, description="Scoring semantic similarity...")
                    pairs = [(resp_a, resp_b) for _, resp_a, resp_b in responses]
                    loop = asyncio.get_running_loop()
                    similarities = await loop.run_in_executor(
                        None,
                        semantic_similarities,
                        pairs,
                        cfg.semantic_batch_size,
                    )
                    if len(similarities) != len(responses):
                        raise RuntimeError(
                            "Semantic scoring returned an unexpected number of scores."
                        )

                    progress.advance(task, 1)
                    progress.update(task, description="Semantic scoring complete")

                    for (case, resp_a, resp_b), sim in zip(responses, similarities):
                        results.append(
                            compute_diff(
                                case_id=case.id,
                                response_a=resp_a,
                                response_b=resp_b,
                                similarity=sim,
                                threshold=cfg.threshold,
                            )
                        )
                else:

                    async def run_and_track(case: TestCase):
                        result = await run_one(cfg, case, client, semaphore)
                        progress.advance(task, 1)
                        progress.update(task, description=f"Done: {case.id}")
                        return result

                    results = await asyncio.gather(
                        *[run_and_track(case) for case in cfg.cases]
                    )
            except RuntimeError as e:
                console.print(f"[red]Error:[/red] {e}")
                raise typer.Exit(1)

    display = [r for r in results if r.changed] if cfg.filter_changed else results
    summary = compute_summary(results)
    policy_failures = _collect_policy_failures(
        results=results,
        summary=summary,
        fail_on_changed=fail_on_changed,
        fail_if_avg_below=fail_if_avg_below,
        fail_if_any_below_threshold=fail_if_any_below_threshold,
    )

    if cfg.output_format == OutputFormat.JSON:
        out = render_json(results, summary)
        if output_path:
            _write_output_report(output_path, out)
            console.print(f"[dim]Report saved to {output_path}[/dim]")
        else:
            print(out)
    elif cfg.output_format == OutputFormat.HTML:
        out = render_html(results, summary)
        if output_path:
            _write_output_report(output_path, out)
            console.print(f"[dim]Report saved to {output_path}[/dim]")
        else:
            print(out)
    else:
        for result in display:
            render_case_inline(
                result,
                label_a=label_a,
                label_b=label_b,
                max_response_lines=cfg.max_response_lines,
                max_diff_lines=cfg.max_diff_lines,
            )

        render_summary(summary)

    if policy_failures:
        for failure in policy_failures:
            console.print(f"[red]Failure policy:[/red] {failure}")
        raise typer.Exit(1)


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
