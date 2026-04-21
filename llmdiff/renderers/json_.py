from __future__ import annotations
import json
from llmdiff.differ import DiffResult
from llmdiff.metrics import Summary


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
            },
            "cases": [
                {
                    "id": r.case_id,
                    "changed": r.changed,
                    "similarity": (
                        round(r.similarity, 4) if r.similarity is not None else None
                    ),
                    "length_a": r.length_a,
                    "length_b": r.length_b,
                    "length_pct": r.structural_changes["length_pct"],
                    "diff": r.unified_diff,
                }
                for r in results
            ],
        },
        indent=2,
    )
