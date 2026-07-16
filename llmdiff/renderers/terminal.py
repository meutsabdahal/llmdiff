from __future__ import annotations

from rich import box
from rich.console import Console
from rich.table import Table
from rich.text import Text

from llmdiff.differ import DiffResult
from llmdiff.metrics import Summary

console = Console()


def _similarity_color(score: float | None) -> str:
    if score is None:
        return "dim"
    if score >= 0.90:
        return "green"
    if score >= 0.70:
        return "yellow"
    return "red"


def _changed_badge(changed: bool) -> Text:
    if changed:
        return Text("CHANGED", style="bold red")
    return Text("unchanged", style="dim green")


def _truncate_lines(lines: list[str], max_lines: int) -> tuple[list[str], int]:
    if max_lines == 0:
        return lines, 0
    if len(lines) <= max_lines:
        return lines, 0
    return lines[:max_lines], len(lines) - max_lines


def _print_case_header(result: DiffResult) -> None:
    sim = result.similarity
    sim_str = f"{sim:.2f}" if sim is not None else "n/a"
    sim_color = _similarity_color(sim)

    header = Text()
    header.append(f" Case: {result.case_id}  ", style="bold")
    header.append(f"Similarity: {sim_str}  ", style=sim_color)
    header.append(_changed_badge(result.changed))

    console.rule(header)


def _print_diff_section(result: DiffResult, max_diff_lines: int) -> None:
    if not result.unified_diff:
        return

    console.print(" [dim]Diff[/dim]")
    console.print()
    diff_lines = [
        line
        for line in result.unified_diff
        if not (line.startswith("+++") or line.startswith("---"))
    ]
    diff_lines, diff_hidden = _truncate_lines(diff_lines, max_diff_lines)

    for line in diff_lines:
        if line.startswith("@@"):
            console.print(f"  [dim]{line}[/dim]", highlight=False)
        elif line.startswith("+"):
            console.print(f"  [green]{line}[/green]", highlight=False)
        elif line.startswith("-"):
            console.print(f"  [red]{line}[/red]", highlight=False)
        else:
            console.print(f"  {line}", highlight=False)
    if diff_hidden:
        console.print(
            "  [dim]... "
            f"({diff_hidden} diff lines hidden; use --max-diff-lines 0 to show all)[/dim]",
            highlight=False,
        )
    console.print()


def _latency_str(timing) -> str:
    if timing is None:
        return "n/a"
    text = f"{timing.latency_s:.2f}s"
    if timing.cached:
        text += " (cached)"
    return text


def _timing_line(timing_a, timing_b) -> str | None:
    if timing_a is None and timing_b is None:
        return None

    parts = [f"Latency: A {_latency_str(timing_a)} / B {_latency_str(timing_b)}"]

    def _rate_str(timing) -> str:
        if timing is None or timing.tokens_per_s is None:
            return "n/a"
        return f"{timing.tokens_per_s:.1f} tok/s"

    rate_a = _rate_str(timing_a)
    rate_b = _rate_str(timing_b)
    if rate_a != "n/a" or rate_b != "n/a":
        parts.append(f"Throughput: A {rate_a} / B {rate_b}")

    return "  │  ".join(parts)


def _print_case_metrics(result: DiffResult) -> None:
    # Stability metrics (only in stability mode, --runs > 1)
    st = result.stability
    if st is not None:
        if st.beyond_noise:
            verdict = "[bold red]beyond sampling noise[/bold red]"
        else:
            verdict = "[green]within sampling noise[/green]"
        console.print(
            f" [dim]Stability ({st.runs} runs): "
            f"{st.similarity_mean:.2f} ± {st.similarity_std:.2f}  │  "
            f"95% CI {st.ci95_low:.2f}–{st.ci95_high:.2f}  │  "
            f"self-similarity A {st.self_similarity_a:.2f} / "
            f"B {st.self_similarity_b:.2f}[/dim]  │  " + verdict
        )

    # Metrics footer
    sim = result.similarity
    sc = result.structural_changes
    pct = sc["length_pct"]
    pct_str = f"+{pct:.0f}%" if pct >= 0 else f"{pct:.0f}%"
    struct = []
    if sc["lists_changed"]:
        struct.append("lists changed")
    if sc["code_blocks_changed"]:
        struct.append("code blocks changed")
    struct_str = ", ".join(struct) if struct else "same"

    console.print(
        f" [dim]Δ Length: {pct_str}  │  "
        f"Semantic distance: {f'{1-sim:.2f}' if sim is not None else 'n/a'}  │  "
        f"Structure: {struct_str}[/dim]"
    )
    timing_line = _timing_line(result.timing_a, result.timing_b)
    if timing_line is not None:
        console.print(f" [dim]{timing_line}[/dim]")
    console.print()


def render_case_inline(
    result: DiffResult,
    label_a: str = "A",
    label_b: str = "B",
    max_response_lines: int = 40,
    max_diff_lines: int = 120,
):
    _print_case_header(result)

    for label, response, length in (
        (label_a, result.response_a, result.length_a),
        (label_b, result.response_b, result.length_b),
    ):
        console.print(
            f" [bold]{label}[/bold]  [dim]{length} words[/dim]",
            highlight=False,
        )
        console.print()
        lines, hidden = _truncate_lines(response.splitlines(), max_response_lines)
        for line in lines:
            console.print(f"  {line}", highlight=False)
        if hidden:
            console.print(
                f"  [dim]... ({hidden} lines hidden; use --max-lines 0 to show all)[/dim]",
                highlight=False,
            )
        console.print()

    _print_diff_section(result, max_diff_lines)
    _print_case_metrics(result)


def _response_cell(response: str, max_lines: int) -> Text:
    lines, hidden = _truncate_lines(response.splitlines(), max_lines)
    cell = Text("\n".join(lines))
    if hidden:
        cell.append(
            f"\n... ({hidden} lines hidden; use --max-lines 0 to show all)",
            style="dim",
        )
    return cell


def render_case_side_by_side(
    result: DiffResult,
    label_a: str = "A",
    label_b: str = "B",
    max_response_lines: int = 40,
    max_diff_lines: int = 120,
):
    """Render a case with the A/B responses in two equal columns.

    Mirrors the HTML report's response grid: labelled columns separated by
    a vertical rule, with each response wrapped to its column width.
    """
    _print_case_header(result)

    table = Table(
        box=box.MINIMAL,
        expand=True,
        padding=(0, 1),
        header_style="",
        border_style="dim",
    )
    for label, length in ((label_a, result.length_a), (label_b, result.length_b)):
        table.add_column(
            Text.assemble((label, "bold"), (f"  {length} words", "dim")),
            ratio=1,
            overflow="fold",
        )
    table.add_row(
        _response_cell(result.response_a, max_response_lines),
        _response_cell(result.response_b, max_response_lines),
    )
    console.print(table)
    console.print()

    _print_diff_section(result, max_diff_lines)
    _print_case_metrics(result)


def render_summary(summary: Summary):
    console.rule()
    console.print()
    console.print(f" [bold]Summary[/bold] — {summary.total} test cases")
    console.rule(style="dim")

    pct = int(summary.changed / summary.total * 100) if summary.total else 0
    console.print(f" Changed:     [bold red]{summary.changed}[/bold red]  ({pct}%)")
    console.print(f" Unchanged:   [green]{summary.unchanged}[/green]")

    if summary.avg_similarity is not None:
        console.print(f" Avg similarity:   [bold]{summary.avg_similarity:.2f}[/bold]")
    if summary.most_diverged:
        cid, score = summary.most_diverged
        console.print(f" Most diverged:    [red]{cid}[/red]  ({score:.2f})")
    if summary.least_changed:
        cid, score = summary.least_changed
        console.print(f" Least changed:    [green]{cid}[/green]  ({score:.2f})")
    if summary.beyond_noise is not None:
        console.print(
            f" Beyond noise:     [bold red]{summary.beyond_noise}[/bold red]"
            "  (changes larger than sampling variance)"
        )
    if summary.avg_latency_a is not None or summary.avg_latency_b is not None:
        lat_a = (
            f"{summary.avg_latency_a:.2f}s"
            if summary.avg_latency_a is not None
            else "n/a"
        )
        lat_b = (
            f"{summary.avg_latency_b:.2f}s"
            if summary.avg_latency_b is not None
            else "n/a"
        )
        console.print(f" Avg latency:      A {lat_a} / B {lat_b}")
    if (
        summary.avg_tokens_per_s_a is not None
        or summary.avg_tokens_per_s_b is not None
    ):
        rate_a = (
            f"{summary.avg_tokens_per_s_a:.1f} tok/s"
            if summary.avg_tokens_per_s_a is not None
            else "n/a"
        )
        rate_b = (
            f"{summary.avg_tokens_per_s_b:.1f} tok/s"
            if summary.avg_tokens_per_s_b is not None
            else "n/a"
        )
        console.print(f" Avg throughput:   A {rate_a} / B {rate_b}")

    console.rule(style="dim")
    console.print()
