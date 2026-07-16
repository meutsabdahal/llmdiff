from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from pydantic import ValidationError

from llmdiff.config import ModelConfig, SideConfig, TestCase

# Bump when the file layout changes; loading refuses files from a newer
# schema so older llmdiff versions never misread them.
BASELINE_VERSION = 1


class BaselineError(Exception):
    """A baseline file cannot be read or does not match the current run."""


def case_input_hash(case: TestCase) -> str:
    """Hash of everything sent to the model for a case except the prompt.

    Stored per case in the baseline so a compare run can detect that a test
    case's input changed since the snapshot — the saved response would then
    answer a different question and the diff would be meaningless.
    """
    canonical = json.dumps(
        case.messages(),
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass
class Baseline:
    prompt: str
    model_cfg: ModelConfig
    created_at: str
    responses: dict[str, str]  # case id -> saved response
    input_hashes: dict[str, str]  # case id -> input hash at snapshot time

    def side_config(self) -> SideConfig:
        return SideConfig(prompt=self.prompt, model_cfg=self.model_cfg)


def build_baseline_document(
    side: SideConfig,
    responses: list[tuple[TestCase, str]],
) -> dict:
    return {
        "version": BASELINE_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "prompt": side.prompt,
        "model_cfg": side.model_cfg.model_dump(),
        "cases": [
            {
                "id": case.id,
                "input_hash": case_input_hash(case),
                "response": response,
            }
            for case, response in responses
        ],
    }


def render_baseline(document: dict) -> str:
    return json.dumps(document, indent=2, ensure_ascii=False)


def _require_str(raw: dict, key: str, where: str) -> str:
    value = raw.get(key)
    if not isinstance(value, str):
        raise BaselineError(f"{where} is missing a string '{key}' field")
    return value


def load_baseline(path: Path) -> Baseline:
    if not path.is_file():
        raise BaselineError(f"baseline file not found: {path}")

    try:
        raw_text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        raise BaselineError(f"baseline file is not valid UTF-8: {path}") from None
    except OSError as e:
        raise BaselineError(f"failed to read baseline file {path}: {e}") from None

    try:
        raw = json.loads(raw_text)
    except json.JSONDecodeError as e:
        raise BaselineError(f"invalid JSON in baseline file {path}: {e}") from None

    if not isinstance(raw, dict):
        raise BaselineError(f"baseline file {path} must contain a JSON object")

    version = raw.get("version")
    if not isinstance(version, int):
        raise BaselineError(
            f"baseline file {path} has no integer 'version' field; "
            "was it created with --save-baseline?"
        )
    if version > BASELINE_VERSION:
        raise BaselineError(
            f"baseline file {path} uses schema version {version}, which is "
            "newer than this llmdiff supports; upgrade llmdiff or re-create "
            "the baseline"
        )

    prompt = _require_str(raw, "prompt", f"baseline file {path}")
    if not prompt.strip():
        raise BaselineError(f"baseline file {path} has an empty prompt")

    model_cfg_raw = raw.get("model_cfg")
    if not isinstance(model_cfg_raw, dict):
        raise BaselineError(f"baseline file {path} is missing 'model_cfg'")
    try:
        model_cfg = ModelConfig(**model_cfg_raw)
    except (ValidationError, TypeError) as e:
        raise BaselineError(
            f"baseline file {path} has an invalid 'model_cfg': {e}"
        ) from None

    cases_raw = raw.get("cases")
    if not isinstance(cases_raw, list) or not cases_raw:
        raise BaselineError(
            f"baseline file {path} must contain a non-empty 'cases' array"
        )

    responses: dict[str, str] = {}
    input_hashes: dict[str, str] = {}
    for i, case_raw in enumerate(cases_raw):
        if not isinstance(case_raw, dict):
            raise BaselineError(
                f"baseline case at index {i} in {path} must be a JSON object"
            )
        where = f"baseline case at index {i} in {path}"
        case_id = _require_str(case_raw, "id", where)
        if case_id in responses:
            raise BaselineError(f"duplicate case id '{case_id}' in baseline {path}")
        responses[case_id] = _require_str(case_raw, "response", where)
        input_hashes[case_id] = _require_str(case_raw, "input_hash", where)

    created_at = raw.get("created_at")

    return Baseline(
        prompt=prompt,
        model_cfg=model_cfg,
        created_at=created_at if isinstance(created_at, str) else "",
        responses=responses,
        input_hashes=input_hashes,
    )


def validate_baseline_cases(baseline: Baseline, cases: list[TestCase]) -> None:
    """Raises BaselineError unless every current case has a usable snapshot."""
    missing = [c.id for c in cases if c.id not in baseline.responses]
    changed = [
        c.id
        for c in cases
        if c.id in baseline.input_hashes
        and baseline.input_hashes[c.id] != case_input_hash(c)
    ]

    if not missing and not changed:
        return

    problems = []
    if missing:
        problems.append(f"not in the baseline: {', '.join(missing)}")
    if changed:
        problems.append(f"input changed since the baseline: {', '.join(changed)}")
    raise BaselineError(
        "baseline does not match the current test cases "
        f"({'; '.join(problems)}). Re-create it with --save-baseline."
    )
