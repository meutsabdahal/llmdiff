from __future__ import annotations

import difflib
import re
from dataclasses import dataclass

from llmdiff.config import ChangedWhen, DiffMode
from llmdiff.metrics import SideTiming, StabilityStats


@dataclass
class DiffResult:
    case_id: str
    response_a: str
    response_b: str
    unified_diff: list[str]  # unified-diff-shaped rows in the configured DiffMode
    changed: bool
    similarity: float | None  # None if --no-semantic; mean over runs in stability mode
    length_a: int  # word count
    length_b: int
    structural_changes: dict  # keys: lists, code_blocks, length_pct
    stability: StabilityStats | None = None  # set when runs > 1
    timing_a: SideTiming | None = None  # None for baseline side A / untimed cache hits
    timing_b: SideTiming | None = None


_ORDERED_LIST_MARKER_RE = re.compile(r"^\d+[.)](?:\s|$)")


def _count_structural(text: str) -> dict:
    lines = text.splitlines()
    return {
        "list_items": sum(
            1
            for l in lines
            if l.strip().startswith(("-", "*", "+"))
            or _ORDERED_LIST_MARKER_RE.match(l.strip())
        ),
        "code_blocks": text.count("```"),
        "word_count": len(text.split()),
    }


def _structural_diff(a: str, b: str) -> dict:
    sa = _count_structural(a)
    sb = _count_structural(b)
    length_pct = 0.0
    if sa["word_count"] > 0:
        length_pct = (sb["word_count"] - sa["word_count"]) / sa["word_count"] * 100

    return {
        "lists_changed": sa["list_items"] != sb["list_items"],
        "code_blocks_changed": sa["code_blocks"] != sb["code_blocks"],
        "length_pct": round(length_pct, 1),
        "word_count_a": sa["word_count"],
        "word_count_b": sb["word_count"],
    }


# Sentence boundary: end punctuation followed by whitespace, or a blank
# line (paragraph break). Single line breaks are NOT boundaries — models
# reflow prose freely, and a wrapped sentence must stay one unit.
_SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?])\s+|\n\s*\n+")
_WHITESPACE_RUN_RE = re.compile(r"\s+")

# Context units around each change: difflib's default 3 for line/sentence
# hunks; more for tokens, which are far smaller units.
_UNIT_CONTEXT = 3
_TOKEN_CONTEXT = 8

_DIFF_HEADER_A = "--- version-a"
_DIFF_HEADER_B = "+++ version-b"


def _split_units(text: str, diff_mode: DiffMode) -> list[str]:
    # splitlines() without keepends: trailing newlines on diff rows made
    # every renderer emit a blank line after each one.
    if diff_mode == DiffMode.TOKEN:
        return text.split()
    if diff_mode == DiffMode.SENTENCE:
        # Collapse internal whitespace (including reflowed line breaks) so a
        # rewrapped but unedited sentence compares and displays as one unit.
        units = (" ".join(s.split()) for s in _SENTENCE_BOUNDARY_RE.split(text))
        return [unit for unit in units if unit]
    return text.splitlines()


def _comparison_key(unit: str, ignore_whitespace: bool, ignore_case: bool) -> str:
    if ignore_whitespace:
        unit = _WHITESPACE_RUN_RE.sub(" ", unit).strip()
    if ignore_case:
        unit = unit.casefold()
    return unit


def _format_range(start: int, stop: int) -> str:
    """Unified-diff range text, matching difflib's formatting exactly."""
    beginning = start + 1
    length = stop - start
    if length == 1:
        return str(beginning)
    if not length:
        beginning -= 1
    return f"{beginning},{length}"


def _unified_from_opcodes(
    a_units: list[str],
    b_units: list[str],
    a_keys: list[str],
    b_keys: list[str],
    join_runs: bool,
    context: int,
) -> list[str]:
    """Unified-diff-shaped output comparing keys but displaying originals.

    With join_runs (token mode), each equal/deleted/inserted run becomes one
    output line instead of one line per unit. Matches difflib.unified_diff
    line-for-line when keys equal units and join_runs is False.
    """
    matcher = difflib.SequenceMatcher(a=a_keys, b=b_keys, autojunk=False)
    out: list[str] = []

    for group in matcher.get_grouped_opcodes(context):
        if not out:
            out.append(_DIFF_HEADER_A)
            out.append(_DIFF_HEADER_B)
        first, last = group[0], group[-1]
        out.append(
            f"@@ -{_format_range(first[1], last[2])} "
            f"+{_format_range(first[3], last[4])} @@"
        )
        for tag, i1, i2, j1, j2 in group:
            if tag == "equal":
                emitted = a_units[i1:i2]
                if join_runs:
                    emitted = [" ".join(emitted)] if emitted else []
                out.extend(f" {unit}" for unit in emitted)
                continue
            if tag in ("replace", "delete"):
                removed = a_units[i1:i2]
                if join_runs:
                    removed = [" ".join(removed)] if removed else []
                out.extend(f"-{unit}" for unit in removed)
            if tag in ("replace", "insert"):
                added = b_units[j1:j2]
                if join_runs:
                    added = [" ".join(added)] if added else []
                out.extend(f"+{unit}" for unit in added)

    return out


def diff_display_rows(unified_diff: list[str]) -> list[str]:
    """The diff rows without the two file-header rows.

    Headers are identified by position, not prefix: content lines starting
    with "--"/"++" (a removed markdown rule, "++i;") render as rows starting
    "---"/"+++" and must survive header stripping.
    """
    if unified_diff[:2] == [_DIFF_HEADER_A, _DIFF_HEADER_B]:
        return unified_diff[2:]
    return unified_diff


def compute_diff(
    case_id: str,
    response_a: str,
    response_b: str,
    similarity: float | None,
    threshold: float | None,
    changed_when: ChangedWhen = ChangedWhen.ANY,
    stability: StabilityStats | None = None,
    timing_a: SideTiming | None = None,
    timing_b: SideTiming | None = None,
    diff_mode: DiffMode = DiffMode.LINE,
    ignore_whitespace: bool = False,
    ignore_case: bool = False,
) -> DiffResult:
    """Compute the textual diff and change status for one case.

    diff_mode selects the diff unit (lines, tokens, or sentences). The
    ignore toggles normalize units for *comparison* only — the diff still
    displays the original text, but units differing only in whitespace or
    case no longer count as changes.

    In stability mode, response_a/response_b are the first sample of each
    side (shown as the representative diff) and similarity is the mean
    cross-side similarity over all runs.
    """
    a_units = _split_units(response_a, diff_mode)
    b_units = _split_units(response_b, diff_mode)
    a_keys = [_comparison_key(u, ignore_whitespace, ignore_case) for u in a_units]
    b_keys = [_comparison_key(u, ignore_whitespace, ignore_case) for u in b_units]

    unified = _unified_from_opcodes(
        a_units,
        b_units,
        a_keys,
        b_keys,
        join_runs=diff_mode == DiffMode.TOKEN,
        context=_TOKEN_CONTEXT if diff_mode == DiffMode.TOKEN else _UNIT_CONTEXT,
    )

    # Compare the keys, not the rendered rows: a content line starting with
    # "--"/"++" renders as a "---..."/"+++..." row that a prefix check would
    # mistake for a file header.
    has_line_diff = a_keys != b_keys
    below_threshold = (
        threshold is not None and similarity is not None and similarity < threshold
    )

    if changed_when == ChangedWhen.LINES:
        changed = has_line_diff
    elif changed_when == ChangedWhen.SEMANTIC:
        changed = below_threshold
    else:
        changed = has_line_diff or below_threshold

    structural = _structural_diff(response_a, response_b)

    return DiffResult(
        case_id=case_id,
        response_a=response_a,
        response_b=response_b,
        unified_diff=unified,
        changed=changed,
        similarity=similarity,
        length_a=structural["word_count_a"],
        length_b=structural["word_count_b"],
        structural_changes=structural,
        stability=stability,
        timing_a=timing_a,
        timing_b=timing_b,
    )
