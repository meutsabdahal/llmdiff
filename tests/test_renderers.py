import json

from llmdiff.differ import DiffResult
from llmdiff.metrics import StabilityStats, Summary
from llmdiff.renderers.html import render_html
from llmdiff.renderers.json_ import render_json
from llmdiff.renderers.markdown import render_markdown


def _sample_result() -> DiffResult:
    return DiffResult(
        case_id="case-1",
        response_a="A",
        response_b="B",
        unified_diff=["-A", "+B"],
        changed=True,
        similarity=0.42,
        length_a=1,
        length_b=1,
        structural_changes={
            "lists_changed": False,
            "code_blocks_changed": False,
            "length_pct": 0.0,
            "word_count_a": 1,
            "word_count_b": 1,
        },
    )


def _sample_summary() -> Summary:
    return Summary(
        total=1,
        changed=1,
        unchanged=0,
        avg_similarity=0.42,
        most_diverged=("case-1", 0.42),
        least_changed=("case-1", 0.42),
    )


def test_render_json_schema_has_expected_fields():
    payload = json.loads(render_json([_sample_result()], _sample_summary()))

    assert set(payload.keys()) == {"summary", "cases"}
    assert set(payload["summary"].keys()) == {
        "total",
        "changed_count",
        "unchanged_count",
        "avg_similarity",
        "most_diverged",
        "least_changed",
        "beyond_noise_count",
    }

    assert len(payload["cases"]) == 1
    case = payload["cases"][0]
    assert set(case.keys()) == {
        "id",
        "changed",
        "similarity",
        "response_a",
        "response_b",
        "length_a",
        "length_b",
        "length_pct",
        "diff",
        "stability",
    }
    assert case["id"] == "case-1"
    assert case["response_a"] == "A"
    assert case["response_b"] == "B"


def test_render_html_contains_embedded_cases_and_summary_blocks():
    html = render_html([_sample_result()], _sample_summary())

    assert "const cases = " in html
    assert "const summary = " in html
    assert "case-1" in html
    assert "llmdiff report" in html


def test_render_markdown_contains_summary_table_and_case_section():
    md = render_markdown([_sample_result()], _sample_summary())

    assert md.startswith("## llmdiff report\n")
    assert "| Total | Changed | Unchanged | Avg similarity |" in md
    assert "| 1 | 1 | 0 | 0.42 |" in md
    assert "Most diverged: `case-1` (0.42)" in md
    assert "### `case-1` — 🔴 changed" in md
    assert "Similarity: **0.42**" in md
    assert "<details>\n<summary>Diff</summary>\n\n```diff\n-A\n+B\n```" in md


def test_render_markdown_omits_diff_details_for_identical_responses():
    result = _sample_result()
    result.unified_diff = []
    result.changed = False

    md = render_markdown([result], _sample_summary())

    assert "### `case-1` — 🟢 unchanged" in md
    assert "<details>" not in md


def test_render_markdown_extends_fence_when_content_contains_backticks():
    result = _sample_result()
    result.unified_diff = ["-A", "+```python"]

    md = render_markdown([result], _sample_summary())

    assert "````diff\n-A\n+```python\n````" in md


def test_render_markdown_widens_code_span_for_backticks_in_case_id():
    result = _sample_result()
    result.case_id = "case`1"

    md = render_markdown([result], _sample_summary())

    assert "### ``case`1`` — 🔴 changed" in md


def test_render_markdown_includes_stability_line_and_beyond_noise_column():
    result = _sample_result()
    result.stability = StabilityStats(
        runs=5,
        similarity_mean=0.42,
        similarity_std=0.03,
        ci95_low=0.39,
        ci95_high=0.45,
        self_similarity_a=0.95,
        self_similarity_b=0.94,
        beyond_noise=True,
    )
    summary = _sample_summary()
    summary.beyond_noise = 1

    md = render_markdown([result], summary)

    assert "| Total | Changed | Unchanged | Avg similarity | Beyond noise |" in md
    assert "| 1 | 1 | 0 | 0.42 | 1 |" in md
    assert (
        "Stability (5 runs): 0.42 ± 0.03 · 95% CI 0.39–0.45"
        " · self-similarity A 0.95 / B 0.94 · **beyond sampling noise**"
    ) in md


def test_render_html_escapes_script_sensitive_characters_in_embedded_json():
    result = _sample_result()
    result.case_id = "<case&1>"
    result.response_a = "</script>"
    result.response_b = "<img src=x onerror=1>"

    html = render_html([result], _sample_summary())

    assert "\\u003ccase\\u00261\\u003e" in html
    assert "\\u003c/script\\u003e" in html
    assert "\\u003cimg src=x onerror=1\\u003e" in html
    assert html.count("</script>") == 1
