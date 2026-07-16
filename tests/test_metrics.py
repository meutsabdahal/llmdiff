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


def test_model_loading_notice_goes_to_stderr_not_stdout(monkeypatch, capsys):
    """The notice must not pollute stdout: piped --format json output
    (llmdiff ... --format json | jq) has to stay parseable."""
    import sys
    import types

    class FakeSentenceTransformer:
        def __init__(self, *_args, **_kwargs):
            pass

    fake_module = types.ModuleType("sentence_transformers")
    fake_module.SentenceTransformer = FakeSentenceTransformer  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_module)
    monkeypatch.setattr(metrics, "_model", None)

    metrics._get_model()

    captured = capsys.readouterr()
    assert "Loading embedding model" not in captured.out
    assert "Loading embedding model" in captured.err


def test_aggregate_timings_means_over_present_samples():
    timings = [
        metrics.SideTiming(latency_s=1.0, tokens=40, tokens_per_s=20.0),
        metrics.SideTiming(latency_s=3.0, tokens=60, tokens_per_s=30.0, cached=True),
        None,
    ]

    agg = metrics.aggregate_timings(timings)

    assert agg.latency_s == 2.0
    assert agg.tokens == 50
    assert agg.tokens_per_s == 25.0
    assert agg.cached is True  # any replayed sample marks the aggregate


def test_aggregate_timings_returns_none_when_nothing_timed():
    assert metrics.aggregate_timings([None, None]) is None


def test_summary_averages_latency_and_throughput_per_side():
    results = [
        FakeResult("a", True, 0.4),
        FakeResult("b", False, 0.9),
    ]
    results[0].timing_a = metrics.SideTiming(latency_s=1.0, tokens_per_s=10.0)
    results[0].timing_b = metrics.SideTiming(latency_s=2.0, tokens_per_s=40.0)
    results[1].timing_a = metrics.SideTiming(latency_s=3.0)  # no throughput data
    results[1].timing_b = None  # e.g. served from a pre-timing cache entry

    s = compute_summary(results)

    assert s.avg_latency_a == 2.0
    assert s.avg_latency_b == 2.0
    assert s.avg_tokens_per_s_a == 10.0
    assert s.avg_tokens_per_s_b == 40.0


def test_summary_timing_fields_none_without_timing_data():
    s = compute_summary([FakeResult("a", True, 0.4)])

    assert s.avg_latency_a is None
    assert s.avg_latency_b is None
    assert s.avg_tokens_per_s_a is None
    assert s.avg_tokens_per_s_b is None
