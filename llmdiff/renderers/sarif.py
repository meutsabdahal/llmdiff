from __future__ import annotations

import json
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _package_version

from llmdiff.differ import DiffResult
from llmdiff.metrics import Summary

_RULE_ID = "prompt-behavior-changed"
_PROJECT_URI = "https://github.com/meutsabdahal/llmdiff"


def _tool_version() -> str:
    try:
        return _package_version("llmdiff-cli")
    except PackageNotFoundError:
        return "0.0.0"


def _result_message(result: DiffResult) -> str:
    sim = result.similarity
    sim_str = f"{sim:.4f}" if sim is not None else "n/a"
    pct = result.structural_changes["length_pct"]
    message = (
        f"Case '{result.case_id}': responses diverged "
        f"(similarity {sim_str}, length {pct:+.0f}%)."
    )

    st = result.stability
    if st is not None:
        verdict = "beyond" if st.beyond_noise else "within"
        message += (
            f" Stability over {st.runs} runs: {st.similarity_mean:.2f} ± "
            f"{st.similarity_std:.2f}, {verdict} sampling noise."
        )
    return message


def render_sarif(
    results: list[DiffResult],
    summary: Summary,
    inputs_uri: str | None = None,
    case_lines: dict[str, int] | None = None,
) -> str:
    """SARIF 2.1.0 report: one warning per changed case.

    Uploadable to GitHub code scanning (`github/codeql-action/upload-sarif`)
    and other SARIF consumers. When `inputs_uri` is given, each result points
    at the test-cases file, at the line where the case is defined when
    `case_lines` provides it.
    """
    sarif_results = []
    for result in results:
        if not result.changed:
            continue

        entry: dict = {
            "ruleId": _RULE_ID,
            "ruleIndex": 0,
            "level": "warning",
            "message": {"text": _result_message(result)},
            # Stable identity for the finding so re-runs update rather than
            # duplicate alerts in consumers that track fingerprints.
            "partialFingerprints": {"llmdiffCaseId": result.case_id},
        }
        if inputs_uri is not None:
            entry["locations"] = [
                {
                    "physicalLocation": {
                        "artifactLocation": {"uri": inputs_uri},
                        "region": {
                            "startLine": (case_lines or {}).get(result.case_id, 1)
                        },
                    },
                    "logicalLocations": [
                        {"name": result.case_id, "kind": "member"}
                    ],
                }
            ]
        sarif_results.append(entry)

    document = {
        "$schema": "https://json.schemastore.org/sarif-2.1.0.json",
        "version": "2.1.0",
        "runs": [
            {
                "tool": {
                    "driver": {
                        "name": "llmdiff",
                        "informationUri": _PROJECT_URI,
                        "version": _tool_version(),
                        "rules": [
                            {
                                "id": _RULE_ID,
                                "name": "PromptBehaviorChanged",
                                "shortDescription": {
                                    "text": (
                                        "Prompt or model change altered the "
                                        "response for a test case"
                                    )
                                },
                                "fullDescription": {
                                    "text": (
                                        "llmdiff compared the side A and side B "
                                        "responses for this test case and found "
                                        "they differ: a line-level diff, or "
                                        "semantic similarity below the "
                                        "configured threshold."
                                    )
                                },
                                "helpUri": _PROJECT_URI,
                                "defaultConfiguration": {"level": "warning"},
                            }
                        ],
                    }
                },
                "results": sarif_results,
            }
        ],
    }
    return json.dumps(document, indent=2)
