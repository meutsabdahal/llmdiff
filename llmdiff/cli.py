from __future__ import annotations
import asyncio
import json
import logging
import os
import re
import math
from pathlib import Path
from typing import Optional

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
from llmdiff.runner import (
    run_diffs,
    configure_request_policy,
    MAX_RETRY_ATTEMPTS,
    MAX_RETRY_BACKOFF_SECONDS,
)
from llmdiff.metrics import compute_summary
from llmdiff.renderers.terminal import render_case_inline, render_summary
from llmdiff.renderers.json_ import render_json
from llmdiff.renderers.html import render_html

app = typer.Typer(
    name="llmdiff",
    help="git diff for LLM prompts",
    add_completion=False,
)
console = Console()
_ENV_KEY_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _parse_env_assignment(raw_line: str) -> tuple[str, str] | None:
    line = raw_line.strip()
    if not line or line.startswith("#"):
        return None

    if line.startswith("export "):
        line = line[len("export ") :].strip()

    if "=" not in line:
        raise ValueError("expected KEY=VALUE assignment")

    key, raw_value = line.split("=", 1)
    key = key.strip()
    if not key:
        raise ValueError("missing environment variable name before '='")
    if not _ENV_KEY_PATTERN.match(key):
        raise ValueError(f"invalid environment variable name '{key}'")

    value = raw_value.strip()
    if not value:
        return key, ""

    if value[0] in {'"', "'"}:
        quote = value[0]
        escaped = False
        closing_index: int | None = None

        for i, ch in enumerate(value[1:], start=1):
            if ch == quote and not escaped:
                closing_index = i
                break
            escaped = ch == "\\" and not escaped

        if closing_index is None:
            raise ValueError("unterminated quoted value")

        parsed_value = value[1:closing_index]
        trailing = value[closing_index + 1 :].strip()
        if trailing and not trailing.startswith("#"):
            raise ValueError("unexpected characters after quoted value")

        return key, parsed_value

    # Unquoted values support inline comments after at least one whitespace char.
    parsed_value = re.split(r"\s+#", value, maxsplit=1)[0].rstrip()
    return key, parsed_value


def _load_local_env() -> None:
    """Loads .env variables if they are not already set in the shell."""
    candidates = [
        Path.cwd() / ".env",
        Path(__file__).resolve().parents[1] / ".env",
    ]
    env_path = next((p for p in candidates if p.exists()), None)
    if env_path is None:
        return

    if not env_path.is_file():
        typer.echo(f"Error: .env path is not a file: {env_path}", err=True)
        raise typer.Exit(1)

    try:
        lines = env_path.read_text(encoding="utf-8").splitlines()
    except UnicodeDecodeError:
        typer.echo(f"Error: .env file is not valid UTF-8: {env_path}", err=True)
        raise typer.Exit(1)
    except OSError as e:
        typer.echo(f"Error: failed to read .env file {env_path}: {e}", err=True)
        raise typer.Exit(1)

    for line_no, raw_line in enumerate(lines, start=1):
        try:
            parsed = _parse_env_assignment(raw_line)
        except ValueError as e:
            typer.echo(
                f"Error: invalid .env line {line_no} in {env_path}: {e}",
                err=True,
            )
            raise typer.Exit(1)

        if parsed is None:
            continue

        key, value = parsed
        if key not in os.environ:
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
    if not path.is_file():
        typer.echo(f"Error: prompt path is not a file: {path}", err=True)
        raise typer.Exit(1)

    try:
        prompt = path.read_text(encoding="utf-8").strip()
    except UnicodeDecodeError:
        typer.echo(f"Error: prompt file is not valid UTF-8: {path}", err=True)
        raise typer.Exit(1)
    except OSError as e:
        typer.echo(f"Error: failed to read prompt file {path}: {e}", err=True)
        raise typer.Exit(1)

    if not prompt:
        typer.echo(f"Error: prompt file is empty: {path}", err=True)
        raise typer.Exit(1)

    return prompt


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


def _iter_case_chunks(cases: list[TestCase], chunk_size: int):
    for i in range(0, len(cases), chunk_size):
        yield cases[i : i + chunk_size]


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
    base_url_a: Optional[str] = typer.Option(
        None,
        "--base-url-a",
        help="Ollama base URL for side A (overrides --base-url)",
    ),
    base_url_b: Optional[str] = typer.Option(
        None,
        "--base-url-b",
        help="Ollama base URL for side B (overrides --base-url)",
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
        max=MAX_RETRY_ATTEMPTS,
        help=(
            "Retry count for transient Ollama request failures "
            f"(0-{MAX_RETRY_ATTEMPTS})."
        ),
    ),
    retry_backoff_base: float = typer.Option(
        0.5,
        "--retry-backoff-base",
        min=0.0,
        help=(
            "Base seconds for exponential retry backoff "
            f"(capped at {MAX_RETRY_BACKOFF_SECONDS:.1f}s)."
        ),
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
    resolved_base_url_a = base_url_a or base_url
    resolved_base_url_b = base_url_b or base_url

    if resolved_model_a == resolved_model_b and prompt_a == prompt_b:
        console.print(
            "[yellow]Warning:[/yellow] Both sides are identical "
            "(same prompt file, same model). Results will show no diff."
        )

    try:
        configure_request_policy(
            request_timeout=request_timeout,
            max_retries=retry_attempts,
            retry_backoff_base=retry_backoff_base,
        )
    except ValueError as e:
        typer.echo(f"Error: invalid request policy: {e}", err=True)
        raise typer.Exit(1)

    model_cfg_a = ModelConfig(
        model=resolved_model_a,
        base_url=resolved_base_url_a,
        temperature=temperature_a if temperature_a is not None else temperature,
        max_tokens=max_tokens_a if max_tokens_a is not None else max_tokens,
    )
    model_cfg_b = ModelConfig(
        model=resolved_model_b,
        base_url=resolved_base_url_b,
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
    semantic_chunks = (
        math.ceil(len(cfg.cases) / cfg.semantic_batch_size) if cfg.semantic else 0
    )
    total_steps = len(cfg.cases) + semantic_chunks

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

        def on_case_completed(case: TestCase) -> None:
            progress.advance(task, 1)
            progress.update(task, description=f"Done: {case.id}")

        def on_semantic_scoring_start() -> None:
            progress.update(task, description="Scoring semantic similarity...")

        def on_semantic_scoring_complete() -> None:
            progress.advance(task, 1)
            progress.update(task, description="Semantic scoring complete")

        try:
            if cfg.semantic:
                results = []
                for chunk_cases in _iter_case_chunks(
                    cfg.cases, cfg.semantic_batch_size
                ):
                    chunk_cfg = cfg.model_copy(update={"cases": chunk_cases})
                    chunk_results = await run_diffs(
                        chunk_cfg,
                        on_case_completed=on_case_completed,
                        on_semantic_scoring_start=on_semantic_scoring_start,
                        on_semantic_scoring_complete=on_semantic_scoring_complete,
                    )
                    results.extend(chunk_results)
            else:
                results = await run_diffs(
                    cfg,
                    on_case_completed=on_case_completed,
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
