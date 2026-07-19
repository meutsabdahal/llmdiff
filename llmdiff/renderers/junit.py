from __future__ import annotations

import re
from xml.etree import ElementTree as ET

from llmdiff.differ import DiffResult, diff_display_rows
from llmdiff.metrics import Summary

# XML 1.0 forbids most C0 control characters even when escaped; strip them so
# a response containing stray escape bytes cannot produce an unparseable
# report that breaks the CI test-results ingester.
_XML_ILLEGAL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


def _xml_safe(text: str) -> str:
    return _XML_ILLEGAL_RE.sub("", text)


def _failure_message(result: DiffResult) -> str:
    sim = result.similarity
    sim_str = f"{sim:.4f}" if sim is not None else "n/a"
    pct = result.structural_changes["length_pct"]
    return f"Responses diverged (similarity {sim_str}, length {pct:+.0f}%)"


def _failure_details(result: DiffResult) -> str:
    lines = [
        f"similarity: {f'{result.similarity:.4f}' if result.similarity is not None else 'n/a'}",
        f"length: A {result.length_a} words, B {result.length_b} words "
        f"({result.structural_changes['length_pct']:+.0f}%)",
    ]

    st = result.stability
    if st is not None:
        verdict = "beyond sampling noise" if st.beyond_noise else "within sampling noise"
        lines.append(
            f"stability ({st.runs} runs): {st.similarity_mean:.2f} ± "
            f"{st.similarity_std:.2f}, 95% CI {st.ci95_low:.2f}-{st.ci95_high:.2f} "
            f"({verdict})"
        )

    diff_lines = diff_display_rows(result.unified_diff)
    if diff_lines:
        lines.append("")
        lines.append("diff:")
        lines.extend(diff_lines)

    return "\n".join(lines)


def _case_duration_s(result: DiffResult) -> float | None:
    """Wall duration of one case: the two sides run in parallel, so the case
    takes as long as its slower side."""
    latencies = [
        t.latency_s for t in (result.timing_a, result.timing_b) if t is not None
    ]
    return max(latencies) if latencies else None


def render_junit(results: list[DiffResult], summary: Summary) -> str:
    """JUnit XML report: one <testcase> per case, changed cases as failures.

    Consumable by the test-report tabs of GitHub Actions, GitLab, Jenkins,
    CircleCI, and similar CI systems.
    """
    durations = [d for r in results if (d := _case_duration_s(r)) is not None]
    # Totals count the rendered cases, not the full run: with --filter the
    # document holds only the changed cases and must stay self-consistent.
    totals = {
        "name": "llmdiff",
        "tests": str(len(results)),
        "failures": str(sum(1 for r in results if r.changed)),
        "errors": "0",
        "skipped": "0",
    }
    if durations:
        totals["time"] = f"{sum(durations):.3f}"
    testsuites = ET.Element("testsuites", totals)
    suite = ET.SubElement(testsuites, "testsuite", totals)

    for result in results:
        attrs = {"name": _xml_safe(result.case_id), "classname": "llmdiff"}
        duration = _case_duration_s(result)
        if duration is not None:
            attrs["time"] = f"{duration:.3f}"
        case_el = ET.SubElement(suite, "testcase", attrs)
        if result.changed:
            failure = ET.SubElement(
                case_el,
                "failure",
                {
                    "message": _xml_safe(_failure_message(result)),
                    "type": "PromptBehaviorChanged",
                },
            )
            failure.text = _xml_safe(_failure_details(result))

    ET.indent(testsuites)
    return ET.tostring(testsuites, encoding="unicode", xml_declaration=True)
