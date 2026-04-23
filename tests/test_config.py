import pytest
from pydantic import ValidationError

from llmdiff.config import (
    ModelConfig,
    SideConfig,
    TestCase as PromptCase,
    RunConfig,
    OutputFormat,
)


def _base_config() -> dict:
    return {
        "side_a": SideConfig(
            prompt="Prompt A",
            model_cfg=ModelConfig(model="llama3.2"),
        ),
        "side_b": SideConfig(
            prompt="Prompt B",
            model_cfg=ModelConfig(model="llama3.2"),
        ),
        "cases": [PromptCase(id="case-1", user="Hello")],
    }


def test_concurrency_must_be_positive():
    cfg = _base_config()
    cfg["concurrency"] = 0

    with pytest.raises(ValidationError):
        RunConfig(**cfg)


def test_threshold_must_be_within_zero_to_one():
    cfg = _base_config()

    cfg["threshold"] = -0.1
    with pytest.raises(ValidationError):
        RunConfig(**cfg)

    cfg["threshold"] = 1.1
    with pytest.raises(ValidationError):
        RunConfig(**cfg)


def test_threshold_accepts_bounds_and_output_format_enum():
    cfg = _base_config()
    cfg["threshold"] = 0.0
    cfg["output_format"] = OutputFormat.JSON
    rc = RunConfig(**cfg)

    assert rc.threshold == 0.0
    assert rc.output_format == OutputFormat.JSON

    cfg["threshold"] = 1.0
    rc = RunConfig(**cfg)
    assert rc.threshold == 1.0
