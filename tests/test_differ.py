import pytest
from llmdiff.differ import compute_diff, _structural_diff


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


def test_no_semantic_result():
    result = compute_diff(
        case_id="test",
        response_a="foo",
        response_b="bar",
        similarity=None,
        threshold=None,
    )
    assert result.similarity is None
