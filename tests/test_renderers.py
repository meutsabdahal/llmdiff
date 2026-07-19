import json
from xml.etree import ElementTree

from llmdiff.differ import DiffResult
from llmdiff.metrics import SideTiming, StabilityStats, Summary
from llmdiff.renderers.html import render_html
from llmdiff.renderers.json_ import render_json
from llmdiff.renderers.junit import render_junit
from llmdiff.renderers.markdown import render_markdown
from llmdiff.renderers.sarif import render_sarif


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
        "avg_latency_s_a",
        "avg_latency_s_b",
        "avg_tokens_per_s_a",
        "avg_tokens_per_s_b",
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
        "timing",
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


def test_render_markdown_keeps_diff_rows_that_look_like_headers():
    result = _sample_result()
    result.unified_diff = [
        "--- version-a",
        "+++ version-b",
        "@@ -1,3 +1,3 @@",
        "----",
        "+++counter;",
    ]

    md = render_markdown([result], _sample_summary())

    assert "----" in md
    assert "+++counter;" in md
    assert "version-a" not in md
    assert "version-b" not in md


def test_render_junit_keeps_diff_rows_that_look_like_headers():
    result = _sample_result()
    result.unified_diff = [
        "--- version-a",
        "+++ version-b",
        "@@ -1,3 +1,3 @@",
        "----",
        "+++counter;",
    ]

    xml = render_junit([result], _sample_summary())

    assert "----" in xml
    assert "+++counter;" in xml
    assert "version-a" not in xml
    assert "version-b" not in xml


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


def _unchanged_result() -> DiffResult:
    return DiffResult(
        case_id="case-same",
        response_a="A",
        response_b="A",
        unified_diff=[],
        changed=False,
        similarity=0.99,
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


def _two_case_summary() -> Summary:
    return Summary(
        total=2,
        changed=1,
        unchanged=1,
        avg_similarity=0.7,
        most_diverged=("case-1", 0.42),
        least_changed=("case-same", 0.99),
    )


def test_render_junit_marks_changed_cases_as_failures():
    xml = render_junit([_sample_result(), _unchanged_result()], _two_case_summary())

    root = ElementTree.fromstring(xml)
    assert root.tag == "testsuites"
    assert root.get("tests") == "2"
    assert root.get("failures") == "1"

    testcases = root.findall("./testsuite/testcase")
    assert [tc.get("name") for tc in testcases] == ["case-1", "case-same"]

    failures = testcases[0].findall("failure")
    assert len(failures) == 1
    assert "similarity 0.4200" in failures[0].get("message")
    assert "-A" in failures[0].text and "+B" in failures[0].text

    assert testcases[1].findall("failure") == []


def test_render_junit_totals_follow_rendered_results():
    # With --filter the renderer receives only the changed cases; the suite
    # totals must describe the document, not the full run.
    xml = render_junit([_sample_result()], _two_case_summary())

    root = ElementTree.fromstring(xml)
    assert root.get("tests") == "1"
    assert root.get("failures") == "1"


def test_render_junit_strips_xml_illegal_control_characters():
    result = _sample_result()
    result.response_a = "bad\x08byte"
    result.unified_diff = ["-bad\x08byte", "+ok"]

    xml = render_junit([result], _sample_summary())

    root = ElementTree.fromstring(xml)  # raises if the XML is invalid
    failure = root.find("./testsuite/testcase/failure")
    assert "\x08" not in failure.text
    assert "-badbyte" in failure.text


def test_render_sarif_reports_only_changed_cases_with_locations():
    sarif = json.loads(
        render_sarif(
            [_sample_result(), _unchanged_result()],
            _two_case_summary(),
            inputs_uri="tests/cases.json",
            case_lines={"case-1": 12},
        )
    )

    assert sarif["version"] == "2.1.0"
    run = sarif["runs"][0]
    assert run["tool"]["driver"]["name"] == "llmdiff"
    assert run["tool"]["driver"]["rules"][0]["id"] == "prompt-behavior-changed"

    results = run["results"]
    assert len(results) == 1
    entry = results[0]
    assert entry["ruleId"] == "prompt-behavior-changed"
    assert entry["level"] == "warning"
    assert "case-1" in entry["message"]["text"]
    assert entry["partialFingerprints"] == {"llmdiffCaseId": "case-1"}

    location = entry["locations"][0]["physicalLocation"]
    assert location["artifactLocation"]["uri"] == "tests/cases.json"
    assert location["region"]["startLine"] == 12


def test_render_sarif_omits_locations_without_inputs_uri():
    sarif = json.loads(render_sarif([_sample_result()], _sample_summary()))

    entry = sarif["runs"][0]["results"][0]
    assert "locations" not in entry


def _timed_result() -> DiffResult:
    result = _sample_result()
    result.timing_a = SideTiming(latency_s=1.234, tokens=40, tokens_per_s=32.4)
    result.timing_b = SideTiming(
        latency_s=0.876, tokens=20, tokens_per_s=22.8, cached=True
    )
    return result


def test_render_json_includes_timing_per_case_and_summary_averages():
    summary = _sample_summary()
    summary.avg_latency_a = 1.234
    summary.avg_latency_b = 0.876
    summary.avg_tokens_per_s_a = 32.4
    summary.avg_tokens_per_s_b = 22.8

    payload = json.loads(render_json([_timed_result()], summary))

    timing = payload["cases"][0]["timing"]
    assert timing["a"] == {
        "latency_s": 1.234,
        "tokens": 40,
        "tokens_per_s": 32.4,
        "cached": False,
    }
    assert timing["b"]["cached"] is True
    assert payload["summary"]["avg_latency_s_a"] == 1.234
    assert payload["summary"]["avg_tokens_per_s_b"] == 22.8


def test_render_json_timing_is_none_when_untimed():
    payload = json.loads(render_json([_sample_result()], _sample_summary()))

    assert payload["cases"][0]["timing"] == {"a": None, "b": None}
    assert payload["summary"]["avg_latency_s_a"] is None


def test_render_junit_sets_time_attributes_from_latency():
    xml = render_junit([_timed_result()], _sample_summary())

    root = ElementTree.fromstring(xml)
    # Sides run in parallel, so the case takes as long as its slower side.
    assert root.get("time") == "1.234"
    testcase = root.find("./testsuite/testcase")
    assert testcase.get("time") == "1.234"


def test_render_junit_omits_time_attributes_when_untimed():
    xml = render_junit([_sample_result()], _sample_summary())

    root = ElementTree.fromstring(xml)
    assert root.get("time") is None
    assert root.find("./testsuite/testcase").get("time") is None


def test_render_markdown_includes_latency_line_and_summary_averages():
    summary = _sample_summary()
    summary.avg_latency_a = 1.2
    summary.avg_latency_b = 0.9
    summary.avg_tokens_per_s_a = 32.4
    summary.avg_tokens_per_s_b = 22.8

    md = render_markdown([_timed_result()], summary)

    assert "Latency: A 1.23s / B 0.88s (cached)" in md
    assert "Avg latency: A 1.20s / B 0.90s" in md
    assert "Avg throughput: A 32.4 tok/s / B 22.8 tok/s" in md


def test_render_html_embeds_timing_payload():
    html = render_html([_timed_result()], _sample_summary())

    assert '"timing"' in html
    assert '"latency_s": 1.234' in html
