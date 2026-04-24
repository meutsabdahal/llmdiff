from llmdiff.differ import DiffResult
from llmdiff.metrics import Summary
from llmdiff.renderers.html import render_html


def test_render_html_preserves_zero_avg_similarity():
    result = DiffResult(
        case_id="case-1",
        response_a="A",
        response_b="B",
        unified_diff=["-A", "+B"],
        changed=True,
        similarity=0.0,
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

    summary = Summary(
        total=1,
        changed=1,
        unchanged=0,
        avg_similarity=0.0,
        most_diverged=("case-1", 0.0),
        least_changed=("case-1", 0.0),
    )

    html = render_html([result], summary)

    assert '"avg_similarity": 0.0' in html


def test_render_html_escapes_script_breakout_payloads():
    payload = "</script><script>alert(1)</script>&<b>x</b>"

    result = DiffResult(
        case_id="case-1",
        response_a=payload,
        response_b=payload,
        unified_diff=[],
        changed=False,
        similarity=1.0,
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

    summary = Summary(
        total=1,
        changed=0,
        unchanged=1,
        avg_similarity=1.0,
        most_diverged=("case-1", 1.0),
        least_changed=("case-1", 1.0),
    )

    html = render_html([result], summary)

    assert payload not in html
    assert (
        "\\u003c/script\\u003e\\u003cscript\\u003ealert(1)"
        "\\u003c/script\\u003e\\u0026\\u003cb\\u003ex\\u003c/b\\u003e" in html
    )
    assert html.count("</script>") == 1
