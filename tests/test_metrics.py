import pytest
from llmdiff.metrics import compute_summary


class FakeResult:
    def __init__(self, case_id, changed, similarity):
        self.case_id = case_id
        self.changed = changed
        self.similarity = similarity


def test_summary_counts():
    results = [
        FakeResult("a", True, 0.4),
        FakeResult("b", False, 0.95),
        FakeResult("c", True, 0.6),
    ]
    s = compute_summary(results)
    assert s.total == 3
    assert s.changed == 2
    assert s.unchanged == 1


def test_summary_avg_similarity():
    results = [
        FakeResult("a", True, 0.4),
        FakeResult("b", False, 1.0),
    ]
    s = compute_summary(results)
    assert abs(s.avg_similarity - 0.7) < 0.01


def test_summary_most_diverged():
    results = [
        FakeResult("a", True, 0.3),
        FakeResult("b", True, 0.8),
    ]
    s = compute_summary(results)
    assert s.most_diverged[0] == "a"


def test_summary_no_similarity():
    results = [FakeResult("a", True, None)]
    s = compute_summary(results)
    assert s.avg_similarity is None
