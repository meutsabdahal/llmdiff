from __future__ import annotations
from enum import Enum
from pydantic import BaseModel, field_validator
from typing import Literal


class OutputFormat(str, Enum):
    INLINE = "inline"
    JSON = "json"
    HTML = "html"


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str

    @field_validator("content")
    @classmethod
    def content_must_not_be_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("content must not be empty")
        return v


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

    @field_validator("max_tokens")
    @classmethod
    def max_tokens_must_be_positive(cls, v: int) -> int:
        if v < 1:
            raise ValueError("max_tokens must be at least 1")
        return v


class SideConfig(BaseModel):
    prompt: str  # system prompt text, already loaded from file
    model_cfg: ModelConfig


class TestCase(BaseModel):
    id: str
    user: str
    context: list[ChatMessage] | None = None  # prior conversation turns

    @field_validator("id", "user")
    @classmethod
    def required_text_fields(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("must not be empty")
        return v


class RunConfig(BaseModel):
    side_a: SideConfig
    side_b: SideConfig
    cases: list[TestCase]
    concurrency: int = 3  # conservative default for local models
    semantic: bool = True
    semantic_batch_size: int = 24
    output_format: OutputFormat = OutputFormat.INLINE
    max_response_lines: int = 40
    max_diff_lines: int = 120
    filter_changed: bool = False
    threshold: float | None = None

    @field_validator("cases")
    @classmethod
    def cases_must_not_be_empty(cls, v: list[TestCase]) -> list[TestCase]:
        if not v:
            raise ValueError("cases must contain at least one test case")
        return v

    @field_validator("concurrency")
    @classmethod
    def concurrency_must_be_positive(cls, v: int) -> int:
        if v < 1:
            raise ValueError("concurrency must be at least 1")
        return v

    @field_validator("semantic_batch_size")
    @classmethod
    def semantic_batch_size_must_be_positive(cls, v: int) -> int:
        if v < 1:
            raise ValueError("semantic_batch_size must be at least 1")
        return v

    @field_validator("max_response_lines", "max_diff_lines")
    @classmethod
    def display_limits_must_be_non_negative(cls, v: int) -> int:
        if v < 0:
            raise ValueError("display line limits must be 0 or greater")
        return v

    @field_validator("threshold")
    @classmethod
    def threshold_range(cls, v: float | None) -> float | None:
        if v is not None and not (0.0 <= v <= 1.0):
            raise ValueError("threshold must be between 0.0 and 1.0")
        return v
