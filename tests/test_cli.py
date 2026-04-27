import pytest
import typer
from typer.testing import CliRunner
import json

import llmdiff.cli as cli
from llmdiff.config import (
    ModelConfig,
    OutputFormat,
    RunConfig,
    SideConfig,
    TestCase as PromptCase,
)
from llmdiff.differ import DiffResult


runner = CliRunner()


def _mk_diff(case_id: str, changed: bool) -> DiffResult:
    return DiffResult(
        case_id=case_id,
        response_a="A",
        response_b="B" if changed else "A",
        unified_diff=["-A", "+B"] if changed else [],
        changed=changed,
        similarity=0.8,
        length_a=1,
        length_b=1,
        structural_changes={
            "lists_changed": False,
            "code_blocks_changed": False,
            "length_pct": 0.0,
            "word_count_a": 1,
            "word_count_b": 1,
        },
    )


def test_cli_rejects_output_for_inline_format():
    result = runner.invoke(
        cli.app,
        [
            "--prompt-a",
            "missing-a.txt",
            "--prompt-b",
            "missing-b.txt",
            "--inputs",
            "missing-cases.json",
            "--output",
            "report.json",
        ],
    )

    assert result.exit_code == 1
    assert "--output requires --format json or --format html." in result.output


def test_load_cases_requires_json_array(tmp_path, capsys):
    path = tmp_path / "cases.json"
    path.write_text('{"id": "x", "user": "hi"}')

    with pytest.raises(typer.Exit):
        cli._load_cases(path)

    captured = capsys.readouterr()
    assert "must contain a JSON array of test cases" in captured.err


def test_load_cases_reports_context_validation_details(tmp_path, capsys):
    path = tmp_path / "cases.json"
    path.write_text(
        '[{"id":"case-1","user":"hello","context":[{"role":"invalid","content":"x"}]}]'
    )

    with pytest.raises(typer.Exit):
        cli._load_cases(path)

    captured = capsys.readouterr()
    assert "invalid test case at index 0" in captured.err
    assert "context.0.role" in captured.err


def test_load_cases_accepts_valid_context_messages(tmp_path):
    path = tmp_path / "cases.json"
    path.write_text(
        '[{"id":"case-1","user":"hello","context":[{"role":"user","content":"x"}]}]'
    )

    cases = cli._load_cases(path)

    assert len(cases) == 1
    assert cases[0].id == "case-1"
    assert cases[0].context is not None
    assert cases[0].context[0].role == "user"


def test_cli_supports_asymmetric_base_urls(tmp_path, monkeypatch):
    prompt_a = tmp_path / "prompt-a.txt"
    prompt_b = tmp_path / "prompt-b.txt"
    inputs = tmp_path / "cases.json"
    prompt_a.write_text("prompt a")
    prompt_b.write_text("prompt b")
    inputs.write_text(json.dumps([{"id": "case-1", "user": "hello"}]))

    captured = {}

    async def fake_run(cfg, **_kwargs):
        captured["cfg"] = cfg

    monkeypatch.setattr(cli, "_run", fake_run)

    result = runner.invoke(
        cli.app,
        [
            "--prompt-a",
            str(prompt_a),
            "--prompt-b",
            str(prompt_b),
            "--inputs",
            str(inputs),
            "--base-url",
            "http://default:11434",
            "--base-url-a",
            "http://side-a:11434",
            "--base-url-b",
            "http://side-b:11434",
            "--no-semantic",
        ],
    )

    assert result.exit_code == 0
    cfg = captured["cfg"]
    assert cfg.side_a.model_cfg.base_url == "http://side-a:11434"
    assert cfg.side_b.model_cfg.base_url == "http://side-b:11434"


def test_cli_base_url_a_falls_back_to_base_url(tmp_path, monkeypatch):
    prompt_a = tmp_path / "prompt-a.txt"
    prompt_b = tmp_path / "prompt-b.txt"
    inputs = tmp_path / "cases.json"
    prompt_a.write_text("prompt a")
    prompt_b.write_text("prompt b")
    inputs.write_text(json.dumps([{"id": "case-1", "user": "hello"}]))

    captured = {}

    async def fake_run(cfg, **_kwargs):
        captured["cfg"] = cfg

    monkeypatch.setattr(cli, "_run", fake_run)

    result = runner.invoke(
        cli.app,
        [
            "--prompt-a",
            str(prompt_a),
            "--prompt-b",
            str(prompt_b),
            "--inputs",
            str(inputs),
            "--base-url",
            "http://default:11434",
            "--base-url-a",
            "http://side-a:11434",
            "--no-semantic",
        ],
    )

    assert result.exit_code == 0
    cfg = captured["cfg"]
    assert cfg.side_a.model_cfg.base_url == "http://side-a:11434"
    assert cfg.side_b.model_cfg.base_url == "http://default:11434"


@pytest.mark.asyncio
async def test_run_filters_unchanged_cases(monkeypatch):
    side_a = SideConfig(prompt="Prompt A", model_cfg=ModelConfig(model="llama3.2"))
    side_b = SideConfig(prompt="Prompt B", model_cfg=ModelConfig(model="llama3.2"))
    cfg = RunConfig(
        side_a=side_a,
        side_b=side_b,
        cases=[
            PromptCase(id="changed", user="hello"),
            PromptCase(id="unchanged", user="hello"),
        ],
        semantic=False,
        output_format=OutputFormat.INLINE,
        filter_changed=True,
    )

    async def fake_check_models_available(*_args, **_kwargs):
        return None

    async def fake_run_one(_cfg, case, _client, _semaphore):
        if case.id == "changed":
            return _mk_diff("changed", changed=True)
        return _mk_diff("unchanged", changed=False)

    rendered_case_ids = []
    rendered_summary = []

    def fake_render_case_inline(result, label_a, label_b, **_kwargs):
        rendered_case_ids.append(result.case_id)

    def fake_render_summary(summary):
        rendered_summary.append(summary)

    monkeypatch.setattr(cli, "check_models_available", fake_check_models_available)
    monkeypatch.setattr(cli, "run_one", fake_run_one)
    monkeypatch.setattr(cli, "render_case_inline", fake_render_case_inline)
    monkeypatch.setattr(cli, "render_summary", fake_render_summary)

    await cli._run(cfg)

    assert rendered_case_ids == ["changed"]
    assert len(rendered_summary) == 1
    assert rendered_summary[0].total == 2
    assert rendered_summary[0].changed == 1


@pytest.mark.asyncio
async def test_run_uses_batched_semantic_scoring(monkeypatch):
    side_a = SideConfig(prompt="Prompt A", model_cfg=ModelConfig(model="llama3.2"))
    side_b = SideConfig(prompt="Prompt B", model_cfg=ModelConfig(model="llama3.2"))
    cfg = RunConfig(
        side_a=side_a,
        side_b=side_b,
        cases=[
            PromptCase(id="same", user="hello"),
            PromptCase(id="diff", user="hello"),
        ],
        semantic=True,
        semantic_batch_size=2,
        output_format=OutputFormat.INLINE,
        filter_changed=False,
    )

    async def fake_check_models_available(*_args, **_kwargs):
        return None

    async def fake_run_case(_client, _semaphore, _cfg, case):
        if case.id == "same":
            return "same output", "same output"
        return "left output", "right output"

    semantic_calls = {}

    def fake_semantic_similarities(pairs, batch_size):
        semantic_calls["pairs"] = pairs
        semantic_calls["batch_size"] = batch_size
        return [0.99, 0.22]

    rendered_case_ids = []
    rendered_summary = []

    def fake_render_case_inline(result, label_a, label_b, **_kwargs):
        rendered_case_ids.append(result.case_id)

    def fake_render_summary(summary):
        rendered_summary.append(summary)

    monkeypatch.setattr(cli, "check_models_available", fake_check_models_available)
    monkeypatch.setattr(cli, "run_case", fake_run_case)
    monkeypatch.setattr(cli, "semantic_similarities", fake_semantic_similarities)
    monkeypatch.setattr(cli, "render_case_inline", fake_render_case_inline)
    monkeypatch.setattr(cli, "render_summary", fake_render_summary)

    await cli._run(cfg)

    assert semantic_calls["batch_size"] == 2
    assert semantic_calls["pairs"] == [
        ("same output", "same output"),
        ("left output", "right output"),
    ]
    assert rendered_case_ids == ["same", "diff"]
    assert len(rendered_summary) == 1
    assert rendered_summary[0].total == 2
    assert rendered_summary[0].changed == 1


@pytest.mark.asyncio
async def test_run_checks_models_for_each_endpoint(monkeypatch):
    side_a = SideConfig(
        prompt="Prompt A",
        model_cfg=ModelConfig(model="llama3.2", base_url="http://a:11434"),
    )
    side_b = SideConfig(
        prompt="Prompt B",
        model_cfg=ModelConfig(model="mistral", base_url="http://b:11434"),
    )
    cfg = RunConfig(
        side_a=side_a,
        side_b=side_b,
        cases=[PromptCase(id="case-1", user="hello")],
        semantic=False,
        output_format=OutputFormat.INLINE,
    )

    calls = []

    async def fake_check_models_available(_client, endpoint, models):
        calls.append((endpoint, tuple(models)))

    async def fake_run_one(_cfg, _case, _client, _semaphore):
        return _mk_diff("case-1", changed=False)

    monkeypatch.setattr(cli, "check_models_available", fake_check_models_available)
    monkeypatch.setattr(cli, "run_one", fake_run_one)
    monkeypatch.setattr(cli, "render_case_inline", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(cli, "render_summary", lambda *_args, **_kwargs: None)

    await cli._run(cfg)

    assert sorted(calls) == [
        ("http://a:11434", ("llama3.2",)),
        ("http://b:11434", ("mistral",)),
    ]
