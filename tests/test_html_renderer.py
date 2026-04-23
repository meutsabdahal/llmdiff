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
