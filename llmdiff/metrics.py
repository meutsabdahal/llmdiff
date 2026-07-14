from __future__ import annotations
from collections.abc import Iterable
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

                Console().print(
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


@dataclass
class Summary:
    total: int
    changed: int
    unchanged: int
    avg_similarity: float | None
    most_diverged: tuple[str, float] | None  # (case_id, similarity)
    least_changed: tuple[str, float] | None


def compute_summary(results) -> Summary:
    from llmdiff.differ import DiffResult

    changed = [r for r in results if r.changed]
    unchanged = [r for r in results if not r.changed]

    sims = [(r.case_id, r.similarity) for r in results if r.similarity is not None]
    avg_sim = sum(s for _, s in sims) / len(sims) if sims else None

    most_diverged = min(sims, key=lambda x: x[1]) if sims else None
    least_changed = max(sims, key=lambda x: x[1]) if sims else None

    return Summary(
        total=len(results),
        changed=len(changed),
        unchanged=len(unchanged),
        avg_similarity=avg_sim,
        most_diverged=most_diverged,
        least_changed=least_changed,
    )
