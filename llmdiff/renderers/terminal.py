from __future__ import annotations
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.table import Table
from rich.rule import Rule
from rich import box

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


def render_case_inline(
    result: DiffResult,
    label_a: str = "A",
    label_b: str = "B",
):
    sim = result.similarity
    sim_str = f"{sim:.2f}" if sim is not None else "n/a"
    sim_color = _similarity_color(sim)

    header = Text()
    header.append(f" Case: {result.case_id}  ", style="bold")
    header.append(f"Similarity: {sim_str}  ", style=sim_color)
    header.append(_changed_badge(result.changed))

    console.rule(header)

    # Version A
    console.print(
        f" [bold]{label_a}[/bold]  [dim]{result.length_a} words[/dim]",
        highlight=False,
    )
    console.print()
    for line in result.response_a.splitlines():
        console.print(f"  {line}", highlight=False)
    console.print()

    # Version B
    console.print(
        f" [bold]{label_b}[/bold]  [dim]{result.length_b} words[/dim]",
        highlight=False,
    )
    console.print()
    for line in result.response_b.splitlines():
        console.print(f"  {line}", highlight=False)
    console.print()

    # Diff
    if result.unified_diff:
        console.print(" [dim]Diff[/dim]")
        console.print()
        for line in result.unified_diff:
            if line.startswith("+++") or line.startswith("---"):
                continue
            if line.startswith("@@"):
                console.print(f"  [dim]{line}[/dim]", highlight=False)
            elif line.startswith("+"):
                console.print(f"  [green]{line}[/green]", highlight=False)
            elif line.startswith("-"):
                console.print(f"  [red]{line}[/red]", highlight=False)
            else:
                console.print(f"  {line}", highlight=False)
        console.print()

    # Metrics footer
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
    console.print()


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

    console.rule(style="dim")
    console.print()
