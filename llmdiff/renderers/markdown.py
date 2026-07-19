from __future__ import annotations

import re

from llmdiff.differ import DiffResult, diff_display_rows, format_length_pct
from llmdiff.metrics import Summary

_BACKTICK_RUN_RE = re.compile(r"`+")


def _inline_code(text: str) -> str:
    """Inline code span whose delimiter is longer than any backtick run inside."""
    longest = max((len(m) for m in _BACKTICK_RUN_RE.findall(text)), default=0)
    delim = "`" * (longest + 1)
    # A space keeps a leading/trailing backtick from merging with the delimiter.
    pad = " " if text.startswith("`") or text.endswith("`") else ""
    return f"{delim}{pad}{text}{pad}{delim}"


def _fenced_block(text: str, lang: str = "") -> str:
    """Fenced code block whose fence is longer than any backtick run inside."""
    longest = max((len(m) for m in _BACKTICK_RUN_RE.findall(text)), default=0)
    fence = "`" * max(3, longest + 1)
    return f"{fence}{lang}\n{text}\n{fence}"


def _summary_section(summary: Summary) -> list[str]:
    headers = ["Total", "Changed", "Unchanged", "Avg similarity"]
    values = [
        str(summary.total),
        str(summary.changed),
        str(summary.unchanged),
        (
            f"{summary.avg_similarity:.2f}"
            if summary.avg_similarity is not None
            else "n/a"
        ),
    ]
    if summary.beyond_noise is not None:
        headers.append("Beyond noise")
        values.append(str(summary.beyond_noise))

    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + " --- |" * len(headers),
        "| " + " | ".join(values) + " |",
        "",
    ]

    extremes = []
    if summary.most_diverged:
        cid, score = summary.most_diverged
        extremes.append(f"Most diverged: {_inline_code(cid)} ({score:.2f})")
    if summary.least_changed:
        cid, score = summary.least_changed
        extremes.append(f"Least changed: {_inline_code(cid)} ({score:.2f})")
    if extremes:
        lines += [" · ".join(extremes), ""]

    perf = []
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
        perf.append(f"Avg latency: A {lat_a} / B {lat_b}")
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
        perf.append(f"Avg throughput: A {rate_a} / B {rate_b}")
    if perf:
        lines += [" · ".join(perf), ""]

    return lines


def _case_section(result: DiffResult) -> list[str]:
    badge = "🔴 changed" if result.changed else "🟢 unchanged"
    lines = [f"### {_inline_code(result.case_id)} — {badge}", ""]

    sim = result.similarity
    pct_str = format_length_pct(result.structural_changes["length_pct"])
    metrics = [
        f"Similarity: **{f'{sim:.2f}' if sim is not None else 'n/a'}**",
        f"Length: {result.length_a} → {result.length_b} words ({pct_str})",
    ]
    structure = []
    if result.structural_changes.get("lists_changed"):
        structure.append("lists changed")
    if result.structural_changes.get("code_blocks_changed"):
        structure.append("code blocks changed")
    if structure:
        metrics.append("Structure: " + ", ".join(structure))
    if result.timing_a is not None or result.timing_b is not None:

        def _latency_str(timing) -> str:
            if timing is None:
                return "n/a"
            text = f"{timing.latency_s:.2f}s"
            if timing.cached:
                text += " (cached)"
            return text

        metrics.append(
            f"Latency: A {_latency_str(result.timing_a)}"
            f" / B {_latency_str(result.timing_b)}"
        )
    lines += [" · ".join(metrics), ""]

    st = result.stability
    if st is not None:
        verdict = (
            "**beyond sampling noise**"
            if st.beyond_noise
            else "within sampling noise"
        )
        lines += [
            f"Stability ({st.runs} runs): "
            f"{st.similarity_mean:.2f} ± {st.similarity_std:.2f}"
            f" · 95% CI {st.ci95_low:.2f}–{st.ci95_high:.2f}"
            f" · self-similarity A {st.self_similarity_a:.2f}"
            f" / B {st.self_similarity_b:.2f}"
            f" · {verdict}",
            "",
        ]

    diff_lines = diff_display_rows(result.unified_diff)
    if diff_lines:
        # Blank lines around the fence are required for GitHub to render
        # markdown inside <details>.
        lines += [
            "<details>",
            "<summary>Diff</summary>",
            "",
            _fenced_block("\n".join(diff_lines), "diff"),
            "",
            "</details>",
            "",
        ]

    return lines


def render_markdown(results: list[DiffResult], summary: Summary) -> str:
    """GitHub-flavored Markdown report for Actions job summaries and PR comments."""
    lines = ["## llmdiff report", ""]
    lines += _summary_section(summary)
    for result in results:
        lines += _case_section(result)
    return "\n".join(lines).rstrip() + "\n"
