from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from threading import Lock

_MISSING_SEMANTIC_DEPS_MSG = (
    "Semantic scoring dependencies are not installed. "
    "Install with 'uv sync --all-extras' (source checkout) or "
    "'pip install \"llmdiff-cli[semantic]\"' (package install), or run with --no-semantic."
)

# Fully qualified name plus a pinned revision so a compromised or
# force-pushed upstream Hub repo cannot silently change the weights we load.
_EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
_EMBEDDING_MODEL_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"

_model = None
_model_lock = Lock()


def _get_model():
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                from rich.console import Console

                try:
                    from sentence_transformers import SentenceTransformer
                except Exception:
                    raise RuntimeError(_MISSING_SEMANTIC_DEPS_MSG) from None

                # stderr, not stdout: piped --format json/html output must
                # stay parseable, and this notice would corrupt it.
                Console(stderr=True).print(
                    "[dim]Loading embedding model (first run only)...[/dim]"
                )
                _model = SentenceTransformer(
                    _EMBEDDING_MODEL_NAME,
                    revision=_EMBEDDING_MODEL_REVISION,
                )
    return _model


def _cosine_from_normalized(a, b) -> float:
    if len(a) != len(b):
        raise RuntimeError("Embedding vectors have mismatched dimensions.")
    score = sum(float(x) * float(y) for x, y in zip(a, b))
    # clamp to [0, 1] — floating point can produce tiny negatives
    return max(0.0, min(1.0, float(score)))


def _append_similarity_scores_from_pair_batch(
    model, pair_batch, scores: list[float]
) -> None:
    texts: list[str] = []
    for a, b in pair_batch:
        texts.extend((a, b))

    embeddings = model.encode(texts, normalize_embeddings=True)
    if len(embeddings) != len(texts):
        raise RuntimeError("Embedding model returned an unexpected number of vectors.")

    for i in range(0, len(embeddings), 2):
        scores.append(_cosine_from_normalized(embeddings[i], embeddings[i + 1]))


def semantic_similarities(
    pairs: Iterable[tuple[str, str]],
    batch_size: int = 24,
) -> list[float]:
    """Returns cosine similarity [0, 1] for each (a, b) pair."""
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")

    model = None
    scores: list[float] = []
    pair_batch: list[tuple[str, str]] = []

    for pair in pairs:
        if model is None:
            model = _get_model()
        pair_batch.append(pair)
        if len(pair_batch) == batch_size:
            _append_similarity_scores_from_pair_batch(model, pair_batch, scores)
            pair_batch.clear()

    if pair_batch:
        if model is None:
            model = _get_model()
        _append_similarity_scores_from_pair_batch(model, pair_batch, scores)

    return scores


# Two-sided 95% critical values of Student's t-distribution, keyed by degrees
# of freedom. Covers df 1..24, i.e. up to MAX_STABILITY_RUNS samples; the
# normal approximation is the defensive fallback beyond that.
_T_CRITICAL_95 = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
    11: 2.201,
    12: 2.179,
    13: 2.160,
    14: 2.145,
    15: 2.131,
    16: 2.120,
    17: 2.110,
    18: 2.101,
    19: 2.093,
    20: 2.086,
    21: 2.080,
    22: 2.074,
    23: 2.069,
    24: 2.064,
}
_Z_CRITICAL_95 = 1.960


@dataclass
class StabilityStats:
    """Per-case variance metrics from a stability-mode run (N samples/side).

    beyond_noise is True when even the 95% CI upper bound of the cross-side
    similarity stays below the lower of the two self-consistency scores: the
    prompts' outputs differ more than either prompt differs from itself, so
    the change cannot be explained by sampling noise alone.
    """

    runs: int
    similarity_mean: float
    similarity_std: float
    ci95_low: float
    ci95_high: float
    self_similarity_a: float
    self_similarity_b: float
    beyond_noise: bool


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values)


def compute_stability_stats(
    cross_similarities: Sequence[float],
    self_similarities_a: Sequence[float],
    self_similarities_b: Sequence[float],
) -> StabilityStats:
    """Aggregate similarity scores from N repeated runs of one case.

    cross_similarities holds sim(A_i, B_i) for each run i; the self lists
    hold pairwise similarities within one side's samples (its noise floor).
    """
    runs = len(cross_similarities)
    if runs < 2:
        raise ValueError("stability stats require at least 2 runs")
    if not self_similarities_a or not self_similarities_b:
        raise ValueError("stability stats require self-similarity scores")

    mean = _mean(cross_similarities)
    variance = sum((s - mean) ** 2 for s in cross_similarities) / (runs - 1)
    std = math.sqrt(variance)

    t_critical = _T_CRITICAL_95.get(runs - 1, _Z_CRITICAL_95)
    half_width = t_critical * std / math.sqrt(runs)
    ci95_low = max(0.0, mean - half_width)
    ci95_high = min(1.0, mean + half_width)

    self_a = _mean(self_similarities_a)
    self_b = _mean(self_similarities_b)
    noise_floor = min(self_a, self_b)

    return StabilityStats(
        runs=runs,
        similarity_mean=mean,
        similarity_std=std,
        ci95_low=ci95_low,
        ci95_high=ci95_high,
        self_similarity_a=self_a,
        self_similarity_b=self_b,
        beyond_noise=ci95_high < noise_floor,
    )


@dataclass
class SideTiming:
    """Request timing for one side of a case.

    latency_s is client wall-clock time for the successful request (retries
    excluded). tokens and tokens_per_s come from Ollama's eval_count /
    eval_duration when the response includes them. cached marks values
    replayed from the response cache — they were measured when the response
    was originally fetched, not during this run. In stability mode the values
    are means over the run's samples.
    """

    latency_s: float
    tokens: int | None = None
    tokens_per_s: float | None = None
    cached: bool = False


def aggregate_timings(timings: Sequence[SideTiming | None]) -> SideTiming | None:
    """Mean timing over stability-mode samples; None if nothing was timed."""
    present = [t for t in timings if t is not None]
    if not present:
        return None

    token_counts = [t.tokens for t in present if t.tokens is not None]
    rates = [t.tokens_per_s for t in present if t.tokens_per_s is not None]
    return SideTiming(
        latency_s=_mean([t.latency_s for t in present]),
        tokens=round(_mean(token_counts)) if token_counts else None,
        tokens_per_s=_mean(rates) if rates else None,
        cached=any(t.cached for t in present),
    )


@dataclass
class Summary:
    total: int
    changed: int
    unchanged: int
    avg_similarity: float | None
    most_diverged: tuple[str, float] | None  # (case_id, similarity)
    least_changed: tuple[str, float] | None
    beyond_noise: int | None = None  # stability mode only
    avg_latency_a: float | None = None  # seconds, mean over timed cases
    avg_latency_b: float | None = None
    avg_tokens_per_s_a: float | None = None
    avg_tokens_per_s_b: float | None = None


def compute_summary(results) -> Summary:
    changed = [r for r in results if r.changed]
    unchanged = [r for r in results if not r.changed]

    sims = [(r.case_id, r.similarity) for r in results if r.similarity is not None]
    avg_sim = sum(s for _, s in sims) / len(sims) if sims else None

    most_diverged = min(sims, key=lambda x: x[1]) if sims else None
    least_changed = max(sims, key=lambda x: x[1]) if sims else None

    stability = [
        s for r in results if (s := getattr(r, "stability", None)) is not None
    ]
    beyond_noise = (
        sum(1 for s in stability if s.beyond_noise) if stability else None
    )

    def _avg_latency(attr: str) -> float | None:
        latencies = [
            t.latency_s
            for r in results
            if (t := getattr(r, attr, None)) is not None
        ]
        return _mean(latencies) if latencies else None

    def _avg_rate(attr: str) -> float | None:
        rates = [
            t.tokens_per_s
            for r in results
            if (t := getattr(r, attr, None)) is not None
            and t.tokens_per_s is not None
        ]
        return _mean(rates) if rates else None

    return Summary(
        total=len(results),
        changed=len(changed),
        unchanged=len(unchanged),
        avg_similarity=avg_sim,
        most_diverged=most_diverged,
        least_changed=least_changed,
        beyond_noise=beyond_noise,
        avg_latency_a=_avg_latency("timing_a"),
        avg_latency_b=_avg_latency("timing_b"),
        avg_tokens_per_s_a=_avg_rate("timing_a"),
        avg_tokens_per_s_b=_avg_rate("timing_b"),
    )
