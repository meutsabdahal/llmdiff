import pytest
import typer
from typer.testing import CliRunner

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

    def fake_render_case_inline(result, label_a, label_b):
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
