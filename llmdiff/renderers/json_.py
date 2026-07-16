from __future__ import annotations

import json

from llmdiff.differ import DiffResult
from llmdiff.metrics import SideTiming, StabilityStats, Summary


def timing_payload(timing: SideTiming | None) -> dict | None:
    """JSON-safe timing dict, shared by the JSON and HTML renderers."""
    if timing is None:
        return None
    return {
        "latency_s": round(timing.latency_s, 3),
        "tokens": timing.tokens,
        "tokens_per_s": (
            round(timing.tokens_per_s, 1)
            if timing.tokens_per_s is not None
            else None
        ),
        "cached": timing.cached,
    }


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
                "avg_latency_s_a": (
                    round(summary.avg_latency_a, 3)
                    if summary.avg_latency_a is not None
                    else None
                ),
                "avg_latency_s_b": (
                    round(summary.avg_latency_b, 3)
                    if summary.avg_latency_b is not None
                    else None
                ),
                "avg_tokens_per_s_a": (
                    round(summary.avg_tokens_per_s_a, 1)
                    if summary.avg_tokens_per_s_a is not None
                    else None
                ),
                "avg_tokens_per_s_b": (
                    round(summary.avg_tokens_per_s_b, 1)
                    if summary.avg_tokens_per_s_b is not None
                    else None
                ),
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
                    "timing": {
                        "a": timing_payload(r.timing_a),
                        "b": timing_payload(r.timing_b),
                    },
                }
                for r in results
            ],
        },
        indent=2,
    )
