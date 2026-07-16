from dataclasses import replace

from rich.console import Console

import llmdiff.renderers.terminal as terminal
from llmdiff.differ import DiffResult


def _make_result() -> DiffResult:
    response_a = "\n".join(f"A line {i}" for i in range(1, 9))
    response_b = "\n".join(f"B line {i}" for i in range(1, 9))
    unified_diff = [
        "--- version-a",
        "+++ version-b",
        "@@ -1,8 +1,8 @@",
        "-A line 1",
        "+B line 1",
        "-A line 2",
        "+B line 2",
        "-A line 3",
        "+B line 3",
        "-A line 4",
        "+B line 4",
    ]

    return DiffResult(
        case_id="case-1",
        response_a=response_a,
        response_b=response_b,
        unified_diff=unified_diff,
        changed=True,
        similarity=0.5,
        length_a=8,
        length_b=8,
        structural_changes={
            "lists_changed": False,
            "code_blocks_changed": False,
            "length_pct": 0.0,
            "word_count_a": 8,
            "word_count_b": 8,
        },
    )


def test_render_case_inline_truncates_long_sections(monkeypatch):
    fake_console = Console(record=True, width=160)
    monkeypatch.setattr(terminal, "console", fake_console)

    terminal.render_case_inline(
        _make_result(),
        max_response_lines=3,
        max_diff_lines=4,
    )

    out = fake_console.export_text()
    assert "lines hidden; use --max-lines 0 to show all" in out
    assert "diff lines hidden; use --max-diff-lines 0 to show all" in out


def test_render_case_inline_no_truncation_when_disabled(monkeypatch):
    fake_console = Console(record=True, width=160)
    monkeypatch.setattr(terminal, "console", fake_console)

    terminal.render_case_inline(
        _make_result(),
        max_response_lines=0,
        max_diff_lines=0,
    )

    out = fake_console.export_text()
    assert "lines hidden; use --max-lines 0 to show all" not in out
    assert "diff lines hidden; use --max-diff-lines 0 to show all" not in out


def test_render_case_side_by_side_puts_responses_in_columns(monkeypatch):
    fake_console = Console(record=True, width=120)
    monkeypatch.setattr(terminal, "console", fake_console)

    terminal.render_case_side_by_side(
        _make_result(),
        label_a="prompt-a",
        label_b="prompt-b",
    )

    out = fake_console.export_text()
    lines = out.splitlines()

    # Column headers share one line, as do the paired response lines.
    header_lines = [ln for ln in lines if "prompt-a" in ln and "prompt-b" in ln]
    assert header_lines, out
    assert "8 words" in header_lines[0]
    paired = [ln for ln in lines if "A line 1" in ln and "B line 1" in ln]
    assert paired, out

    # Diff and metrics footer still follow the columns.
    assert "Diff" in out
    assert "Semantic distance" in out


def test_render_case_side_by_side_truncates_each_side(monkeypatch):
    fake_console = Console(record=True, width=120)
    monkeypatch.setattr(terminal, "console", fake_console)

    terminal.render_case_side_by_side(
        _make_result(),
        max_response_lines=3,
        max_diff_lines=4,
    )

    out = fake_console.export_text()
    assert out.count("lines hidden; use --max-lines 0 to show all") >= 1
    assert "A line 4" not in out
    assert "B line 4" not in out
    assert "diff lines hidden; use --max-diff-lines 0 to show all" in out


def test_render_case_side_by_side_wraps_long_lines(monkeypatch):
    fake_console = Console(record=True, width=60)
    monkeypatch.setattr(terminal, "console", fake_console)

    long_line = "alpha " * 40
    result = replace(_make_result(), response_a=long_line, response_b="short")

    terminal.render_case_side_by_side(result, max_response_lines=0)

    out = fake_console.export_text()
    # The long response wraps within its column instead of being cropped.
    assert out.count("alpha") == 40
