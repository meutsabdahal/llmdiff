from __future__ import annotations

import json

from llmdiff.differ import DiffResult
from llmdiff.metrics import StabilityStats, Summary


def stability_payload(stats: StabilityStats | None) -> dict | None:
    """JSON-safe stability dict, shared by the JSON and HTML renderers."""
    if stats is None:
        return None
    return {
        "runs": stats.runs,
        "similarity_mean": round(stats.similarity_mean, 4),
        "similarity_std": round(stats.similarity_std, 4),
        "ci95": [round(stats.ci95_low, 4), round(stats.ci95_high, 4)],
        "self_similarity_a": round(stats.self_similarity_a, 4),
        "self_similarity_b": round(stats.self_similarity_b, 4),
        "beyond_noise": stats.beyond_noise,
    }


def render_json(results: list[DiffResult], summary: Summary) -> str:
    return json.dumps(
        {
            "summary": {
                "total": summary.total,
                "changed_count": summary.changed,
                "unchanged_count": summary.unchanged,
                "avg_similarity": (
                    round(summary.avg_similarity, 4)
                    if summary.avg_similarity is not None
                    else None
                ),
                "most_diverged": summary.most_diverged,
                "least_changed": summary.least_changed,
                "beyond_noise_count": summary.beyond_noise,
            },
            "cases": [
                {
                    "id": r.case_id,
                    "changed": r.changed,
                    "similarity": (
                        round(r.similarity, 4) if r.similarity is not None else None
                    ),
                    "response_a": r.response_a,
                    "response_b": r.response_b,
                    "length_a": r.length_a,
                    "length_b": r.length_b,
                    "length_pct": r.structural_changes["length_pct"],
                    "diff": r.unified_diff,
                    "stability": stability_payload(r.stability),
                }
                for r in results
            ],
        },
        indent=2,
    )
