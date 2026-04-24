from __future__ import annotations
import difflib
import re
from dataclasses import dataclass


@dataclass
class DiffResult:
    case_id: str
    response_a: str
    response_b: str
    unified_diff: list[str]  # output of difflib.unified_diff
    changed: bool
    similarity: float | None  # None if --no-semantic
    length_a: int  # word count
    length_b: int
    structural_changes: dict  # keys: lists, code_blocks, length_pct


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


def compute_diff(
    case_id: str,
    response_a: str,
    response_b: str,
    similarity: float | None,
    threshold: float | None,
) -> DiffResult:
    a_lines = response_a.splitlines(keepends=True)
    b_lines = response_b.splitlines(keepends=True)

    unified = list(
        difflib.unified_diff(
            a_lines,
            b_lines,
            fromfile="version-a",
            tofile="version-b",
            lineterm="",
        )
    )

    # "changed" means: any line-level diff exists, or similarity is below threshold
    has_line_diff = any(
        l.startswith(("+", "-")) and not l.startswith(("+++", "---")) for l in unified
    )
    below_threshold = (
        threshold is not None and similarity is not None and similarity < threshold
    )
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
    )
