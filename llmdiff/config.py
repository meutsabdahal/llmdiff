from __future__ import annotations
from pydantic import BaseModel, field_validator
from typing import Literal


class ModelConfig(BaseModel):
    model: str
    base_url: str = "http://localhost:11434"
    temperature: float | None = None
    max_tokens: int = 1024

    @field_validator("temperature")
    @classmethod
    def temperature_range(cls, v):
        if v is not None and not (0.0 <= v <= 2.0):
            raise ValueError("temperature must be between 0.0 and 2.0")
        return v


class SideConfig(BaseModel):
    prompt: str  # system prompt text, already loaded from file
    model_cfg: ModelConfig


class TestCase(BaseModel):
    id: str
    user: str
    context: list[dict] | None = None  # prior conversation turns


class RunConfig(BaseModel):
    side_a: SideConfig
    side_b: SideConfig
    cases: list[TestCase]
    concurrency: int = 3  # conservative default for local models
    semantic: bool = True
    output_format: Literal["inline", "side-by-side", "json"] = "inline"
    filter_changed: bool = False
    threshold: float | None = None
