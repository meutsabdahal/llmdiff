import json

from llmdiff.differ import DiffResult
from llmdiff.metrics import Summary
from llmdiff.renderers.html import render_html
from llmdiff.renderers.json_ import render_json


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
    }

    assert len(payload["cases"]) == 1
    case = payload["cases"][0]
    assert set(case.keys()) == {
        "id",
        "changed",
        "similarity",
        "length_a",
        "length_b",
        "length_pct",
        "diff",
    }
    assert case["id"] == "case-1"


def test_render_html_contains_embedded_cases_and_summary_blocks():
    html = render_html([_sample_result()], _sample_summary())

    assert "const cases = " in html
    assert "const summary = " in html
    assert "case-1" in html
    assert "llmdiff report" in html


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
