import pytest
import llmdiff.metrics as metrics
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


def test_semantic_similarities_batches_and_scores(monkeypatch):
    class FakeModel:
        def __init__(self):
            self.calls = []

        def encode(self, texts, normalize_embeddings=True):
            self.calls.append(list(texts))
            mapping = {
                "same_a": [1.0, 0.0, 0.0],
                "same_b": [1.0, 0.0, 0.0],
                "orth_a": [1.0, 0.0, 0.0],
                "orth_b": [0.0, 1.0, 0.0],
            }
            return [mapping[t] for t in texts]

    fake_model = FakeModel()
    monkeypatch.setattr(metrics, "_get_model", lambda: fake_model)

    scores = metrics.semantic_similarities(
        [("same_a", "same_b"), ("orth_a", "orth_b"), ("same_a", "same_b")],
        batch_size=2,
    )

    assert scores == [1.0, 0.0, 1.0]
    assert len(fake_model.calls) == 2
    assert len(fake_model.calls[0]) == 4
    assert len(fake_model.calls[1]) == 2


def test_semantic_similarity_single_pair_wrapper(monkeypatch):
    called = {}

    def fake_semantic_similarities(pairs, batch_size=24):
        called["pairs"] = pairs
        called["batch_size"] = batch_size
        return [0.55]

    monkeypatch.setattr(metrics, "semantic_similarities", fake_semantic_similarities)

    score = metrics.semantic_similarity("a", "b")

    assert score == 0.55
    assert called["pairs"] == [("a", "b")]
    assert called["batch_size"] == 1


def test_semantic_similarities_accepts_pair_iterables(monkeypatch):
    class FakeModel:
        def __init__(self):
            self.calls = []

        def encode(self, texts, normalize_embeddings=True):
            self.calls.append(list(texts))
            mapping = {
                "same_a": [1.0, 0.0, 0.0],
                "same_b": [1.0, 0.0, 0.0],
                "orth_a": [1.0, 0.0, 0.0],
                "orth_b": [0.0, 1.0, 0.0],
            }
            return [mapping[t] for t in texts]

    fake_model = FakeModel()
    monkeypatch.setattr(metrics, "_get_model", lambda: fake_model)

    pairs = ((a, b) for a, b in [("same_a", "same_b"), ("orth_a", "orth_b")])
    scores = metrics.semantic_similarities(pairs, batch_size=1)

    assert scores == [1.0, 0.0]
    assert fake_model.calls == [["same_a", "same_b"], ["orth_a", "orth_b"]]


def test_semantic_similarities_skips_model_load_for_empty_iterable(monkeypatch):
    monkeypatch.setattr(
        metrics,
        "_get_model",
        lambda: (_ for _ in ()).throw(AssertionError("model load should be skipped")),
    )

    scores = metrics.semantic_similarities(iter(()), batch_size=4)

    assert scores == []
