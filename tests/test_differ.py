import difflib

from llmdiff.config import ChangedWhen, DiffMode
from llmdiff.differ import _structural_diff, compute_diff, diff_display_rows


def test_identical_responses_not_changed():
    result = compute_diff(
        case_id="test",
        response_a="Hello world",
        response_b="Hello world",
        similarity=1.0,
        threshold=None,
    )
    assert not result.changed
    assert result.unified_diff == []


def test_different_responses_changed():
    result = compute_diff(
        case_id="test",
        response_a="Hello world",
        response_b="Goodbye world",
        similarity=0.5,
        threshold=None,
    )
    assert result.changed
    assert len(result.unified_diff) > 0


def test_threshold_triggers_changed():
    # Even if diff is minor, low similarity should flag as changed
    result = compute_diff(
        case_id="test",
        response_a="A",
        response_b="A",  # identical text
        similarity=0.2,  # but low similarity (hypothetical)
        threshold=0.5,
    )
    assert result.changed


def test_length_pct_calculation():
    sc = _structural_diff("one two three", "one two three four five")
    assert sc["length_pct"] > 0  # B is longer


def test_length_pct_is_none_when_side_a_is_empty():
    # Growth from an empty response has no percentage; 0.0 here would
    # display as "+0%", i.e. no length change.
    sc = _structural_diff("", "some words here")
    assert sc["length_pct"] is None


def test_length_pct_is_zero_when_both_sides_empty():
    sc = _structural_diff("", "")
    assert sc["length_pct"] == 0.0


def test_structural_list_detection():
    a = "Here are options:\n- Option one\n- Option two"
    b = "Here are options:\n1. Option one\n2. Option two"
    sc = _structural_diff(a, b)
    # Both have list items, counts may differ by marker type
    assert "lists_changed" in sc


def test_structural_multi_digit_ordered_list_detection():
    a = "9. Option one\n10. Option two\n11) Option three"
    b = "- Option one\n- Option two\n- Option three"
    sc = _structural_diff(a, b)

    # Same number of list items despite different marker styles.
    assert not sc["lists_changed"]


def test_changed_when_semantic_ignores_line_diff_above_threshold():
    result = compute_diff(
        case_id="test",
        response_a="The capital is Kathmandu.",
        response_b="Kathmandu is the capital.",
        similarity=0.95,
        threshold=0.8,
        changed_when=ChangedWhen.SEMANTIC,
    )

    assert result.unified_diff  # wording differs
    assert not result.changed  # but semantically equivalent


def test_changed_when_semantic_flags_below_threshold():
    result = compute_diff(
        case_id="test",
        response_a="A",
        response_b="A",
        similarity=0.2,
        threshold=0.8,
        changed_when=ChangedWhen.SEMANTIC,
    )

    assert result.changed


def test_changed_when_lines_ignores_threshold():
    result = compute_diff(
        case_id="test",
        response_a="A",
        response_b="A",
        similarity=0.2,
        threshold=0.8,
        changed_when=ChangedWhen.LINES,
    )

    assert not result.changed


def test_diff_lines_have_no_trailing_newlines():
    result = compute_diff(
        case_id="test",
        response_a="line one\nline two",
        response_b="line one\nline changed",
        similarity=None,
        threshold=None,
    )

    assert result.unified_diff
    assert all(not line.endswith("\n") for line in result.unified_diff)
    assert "-line two" in result.unified_diff
    assert "+line changed" in result.unified_diff


def test_no_semantic_result():
    result = compute_diff(
        case_id="test",
        response_a="foo",
        response_b="bar",
        similarity=None,
        threshold=None,
    )
    assert result.similarity is None


# --- diff modes ---


def _diff(a, b, **kwargs):
    return compute_diff(
        case_id="test",
        response_a=a,
        response_b=b,
        similarity=None,
        threshold=None,
        **kwargs,
    )


def test_line_mode_matches_difflib_unified_output():
    result = _diff("one\ntwo\nthree", "one\nTWO\nthree")

    expected = list(
        difflib.unified_diff(
            ["one", "two", "three"],
            ["one", "TWO", "three"],
            fromfile="version-a",
            tofile="version-b",
            lineterm="",
        )
    )
    assert result.unified_diff == expected


def test_token_mode_ignores_reflowed_lines():
    prose = "The quick brown fox jumps over the lazy dog"
    reflowed = "The quick brown\nfox jumps over\nthe lazy dog"

    line_result = _diff(prose, reflowed)
    token_result = _diff(prose, reflowed, diff_mode=DiffMode.TOKEN)

    assert line_result.changed
    assert not token_result.changed
    assert token_result.unified_diff == []


def test_token_mode_joins_changed_runs_into_compact_lines():
    result = _diff(
        "The quick brown fox jumps over the lazy dog",
        "The quick red fox leaps over the lazy dog",
        diff_mode=DiffMode.TOKEN,
    )

    assert result.changed
    body = [
        l
        for l in result.unified_diff
        if not l.startswith(("+++", "---", "@@"))
    ]
    assert "-brown" in body
    assert "+red" in body
    assert "-jumps" in body
    assert "+leaps" in body
    # Context runs are joined, not one token per line.
    assert " The quick" in body


def test_sentence_mode_isolates_reworded_sentence():
    a = "Hello there. The sky is blue. Goodbye now."
    b = "Hello there. The sky is bright green. Goodbye now."

    result = _diff(a, b, diff_mode=DiffMode.SENTENCE)

    assert result.changed
    removed = [
        l
        for l in result.unified_diff
        if l.startswith("-") and not l.startswith("---")
    ]
    added = [
        l
        for l in result.unified_diff
        if l.startswith("+") and not l.startswith("+++")
    ]
    assert removed == ["-The sky is blue."]
    assert added == ["+The sky is bright green."]


def test_sentence_mode_treats_reflowed_sentence_as_unchanged():
    a = "Thanks for reaching out! The refund takes 5 business days."
    b = "Thanks for reaching\nout! The refund takes 5\nbusiness days."

    result = _diff(a, b, diff_mode=DiffMode.SENTENCE)

    assert not result.changed
    assert result.unified_diff == []


def test_sentence_mode_splits_unpunctuated_blocks_on_blank_lines():
    result = _diff(
        "Intro heading\n\n- alpha\n- beta",
        "Intro heading\n\n- alpha\n- gamma",
        diff_mode=DiffMode.SENTENCE,
    )

    assert result.changed
    # Single newlines are not boundaries, so the list block is one unit.
    assert "-- alpha - beta" in result.unified_diff
    assert "+- alpha - gamma" in result.unified_diff
    assert " Intro heading" in result.unified_diff


def test_removed_markdown_rule_line_marks_changed():
    # A removed "---" line renders as a "----" row, which a prefix-based
    # header check would mistake for the "--- version-a" file header.
    result = _diff("Intro.\n---\nDetails.", "Intro.\nDetails.")

    assert result.changed
    assert "----" in result.unified_diff


def test_added_line_starting_with_plus_plus_marks_changed():
    result = _diff("keep", "keep\n++counter;")

    assert result.changed
    assert "+++counter;" in result.unified_diff


def test_token_mode_removed_dash_dash_token_marks_changed():
    result = _diff(
        "wait -- stop now", "wait stop now", diff_mode=DiffMode.TOKEN
    )

    assert result.changed
    assert "---" in result.unified_diff


def test_diff_display_rows_strips_headers_by_position_not_prefix():
    result = _diff("Intro.\n---\nDetails.", "Intro.\nDetails.")

    rows = diff_display_rows(result.unified_diff)

    assert "--- version-a" not in rows
    assert "+++ version-b" not in rows
    assert "----" in rows


def test_ignore_case_marks_case_only_change_unchanged():
    result = _diff("Hello World", "hello world", ignore_case=True)

    assert not result.changed
    assert result.unified_diff == []


def test_ignore_whitespace_marks_spacing_only_change_unchanged():
    result = _diff("a  b\tc", " a b c ", ignore_whitespace=True)

    assert not result.changed
    assert result.unified_diff == []


def test_ignore_toggles_display_original_text():
    # The case-only line is treated as equal but must display as written
    # in side A; the real change is still reported.
    result = _diff(
        "Hello World\nsame line\nold ending",
        "hello world\nsame line\nnew ending",
        ignore_case=True,
    )

    assert result.changed
    assert " Hello World" in result.unified_diff
    assert "-old ending" in result.unified_diff
    assert "+new ending" in result.unified_diff


def test_ignore_toggles_apply_to_token_mode():
    result = _diff(
        "The Quick Fox",
        "the quick fox",
        diff_mode=DiffMode.TOKEN,
        ignore_case=True,
    )

    assert not result.changed
    assert result.unified_diff == []
