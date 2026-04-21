from __future__ import annotations
from dataclasses import dataclass
from functools import lru_cache

_model = None


def _get_model():
    global _model
    if _model is None:
        from rich.console import Console
        from sentence_transformers import SentenceTransformer

        Console().print("[dim]Loading embedding model (first run only)...[/dim]")
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


def semantic_similarity(a: str, b: str) -> float:
    """Returns cosine similarity [0, 1] between two strings."""
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np

    model = _get_model()
    embeddings = model.encode([a, b], normalize_embeddings=True)
    score = float(cosine_similarity([embeddings[0]], [embeddings[1]])[0][0])
    # clamp to [0, 1] — floating point can produce tiny negatives
    return max(0.0, min(1.0, score))


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
