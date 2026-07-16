import json
import os
from xml.etree import ElementTree

import pytest
import typer
from typer.testing import CliRunner

import llmdiff.cli as cli
from llmdiff.config import (
    ModelConfig,
    OutputFormat,
    RunConfig,
    SideConfig,
)
from llmdiff.config import (
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


def test_cli_version_flag_prints_version_and_exits():
    result = runner.invoke(cli.app, ["--version"])

    assert result.exit_code == 0
    assert result.output.startswith("llmdiff ")


def test_cli_missing_inputs_suggests_init():
    result = runner.invoke(cli.app, ["--prompt-a", "a.txt", "--prompt-b", "b.txt"])

    assert result.exit_code == 2
    assert "missing required option: --inputs" in result.output
    assert "llmdiff init" in result.output


def test_init_scaffolds_example_files(tmp_path):
    result = runner.invoke(cli.app, ["init", str(tmp_path)])

    assert result.exit_code == 0
    prompt_a = tmp_path / "prompts" / "v1.txt"
    prompt_b = tmp_path / "prompts" / "v2.txt"
    cases_path = tmp_path / "cases.json"
    assert prompt_a.is_file()
    assert prompt_b.is_file()
    assert cases_path.is_file()
    # The scaffold must pass the same validation the compare command applies.
    assert cli._load_prompt(prompt_a) != cli._load_prompt(prompt_b)
    cases = cli._load_cases(cases_path)
    assert [case.id for case in cases] == [
        "basic-greeting",
        "refusal-boundary",
        "multi-turn",
    ]
    # The scaffold demonstrates tagging for selective execution.
    assert cases[0].tags == ["smoke"]
    assert "llmdiff --prompt-a" in result.output


def test_init_skips_existing_files_without_force(tmp_path):
    cases_path = tmp_path / "cases.json"
    cases_path.write_text("[]", encoding="utf-8")

    result = runner.invoke(cli.app, ["init", str(tmp_path)])

    assert result.exit_code == 0
    assert "skipped" in result.output
    assert cases_path.read_text(encoding="utf-8") == "[]"
    assert (tmp_path / "prompts" / "v1.txt").is_file()


def test_init_force_overwrites_existing_files(tmp_path):
    cases_path = tmp_path / "cases.json"
    cases_path.write_text("[]", encoding="utf-8")

    result = runner.invoke(cli.app, ["init", str(tmp_path), "--force"])

    assert result.exit_code == 0
    assert len(cli._load_cases(cases_path)) == 3


def test_init_rejects_non_directory_target(tmp_path):
    target = tmp_path / "cases.json"
    target.write_text("[]", encoding="utf-8")

    result = runner.invoke(cli.app, ["init", str(target)])

    assert result.exit_code == 1
    assert "not a directory" in result.output


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
    assert "--output requires a non-inline --format" in result.output


def test_cli_changed_when_semantic_requires_threshold():
    result = runner.invoke(
        cli.app,
        [
            "--prompt-a",
            "missing-a.txt",
            "--prompt-b",
            "missing-b.txt",
            "--inputs",
            "missing-cases.json",
            "--changed-when",
            "semantic",
        ],
    )

    assert result.exit_code == 1
    assert "--changed-when semantic requires --threshold" in result.output


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


def test_load_cases_rejects_non_utf8_file(tmp_path, capsys):
    path = tmp_path / "cases.json"
    path.write_bytes(b"\xff\xfe\x00bad")

    with pytest.raises(typer.Exit):
        cli._load_cases(path)

    captured = capsys.readouterr()
    assert "not valid UTF-8" in captured.err


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


def test_parse_env_assignment_unescapes_quoted_values():
    assert cli._parse_env_assignment('KEY="a\\"b"') == ("KEY", 'a"b')
    assert cli._parse_env_assignment("KEY='a\\'b'") == ("KEY", "a'b")
    assert cli._parse_env_assignment('KEY="a\\\\b"') == ("KEY", "a\\b")


def test_load_local_env_only_imports_allowlisted_keys(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "HF_TOKEN=from-env-file\nHF_ENDPOINT=http://attacker.example\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HF_ENDPOINT", raising=False)

    cli._load_local_env()

    assert os.environ["HF_TOKEN"] == "from-env-file"
    assert "HF_ENDPOINT" not in os.environ


def test_load_local_env_does_not_override_shell_values(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text("HF_TOKEN=from-env-file\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HF_TOKEN", "from-shell")

    cli._load_local_env()

    assert os.environ["HF_TOKEN"] == "from-shell"


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


@pytest.mark.parametrize(
    ("extra_args", "expected_use_cache"),
    [([], True), (["--no-cache"], False)],
)
def test_cli_no_cache_flag_controls_caching(
    tmp_path, monkeypatch, extra_args, expected_use_cache
):
    prompt_a = tmp_path / "prompt-a.txt"
    prompt_b = tmp_path / "prompt-b.txt"
    inputs = tmp_path / "cases.json"
    prompt_a.write_text("prompt a")
    prompt_b.write_text("prompt b")
    inputs.write_text(json.dumps([{"id": "case-1", "user": "hello"}]))

    captured = {}

    async def fake_run(cfg, **kwargs):
        captured["use_cache"] = kwargs["use_cache"]

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
            "--no-semantic",
            *extra_args,
        ],
    )

    assert result.exit_code == 0
    assert captured["use_cache"] is expected_use_cache


def _write_inputs_and_prompt(tmp_path):
    prompt = tmp_path / "prompt.txt"
    inputs = tmp_path / "cases.json"
    prompt.write_text("a prompt")
    inputs.write_text(json.dumps([{"id": "case-1", "user": "hello"}]))
    return prompt, inputs


def test_cli_requires_prompts_without_baseline_flags(tmp_path):
    _, inputs = _write_inputs_and_prompt(tmp_path)

    result = runner.invoke(cli.app, ["--inputs", str(inputs)])

    assert result.exit_code == 1
    assert "--prompt-a and --prompt-b are required" in result.output


def test_cli_save_baseline_runs_snapshot(tmp_path, monkeypatch):
    prompt, inputs = _write_inputs_and_prompt(tmp_path)
    captured = {}

    async def fake_run_snapshot(side, cases, **kwargs):
        captured["side"] = side
        captured["cases"] = cases
        captured["kwargs"] = kwargs

    monkeypatch.setattr(cli, "_run_snapshot", fake_run_snapshot)

    result = runner.invoke(
        cli.app,
        [
            "--prompt-a",
            str(prompt),
            "--inputs",
            str(inputs),
            "--save-baseline",
            str(tmp_path / "baseline.json"),
            "--model",
            "mistral",
        ],
    )

    assert result.exit_code == 0
    assert captured["side"].prompt == "a prompt"
    assert captured["side"].model_cfg.model == "mistral"
    assert [c.id for c in captured["cases"]] == ["case-1"]
    assert captured["kwargs"]["output_path"] == tmp_path / "baseline.json"


@pytest.mark.parametrize(
    ("extra_args", "expected_error"),
    [
        (["--prompt-b", "b.txt"], "remove --prompt-b"),
        (["--runs", "3"], "remove --runs"),
        (["--format", "json"], "remove --output / --format"),
        (["--fail-on-changed"], "failure policies do not apply"),
    ],
)
def test_cli_save_baseline_rejects_incompatible_flags(
    tmp_path, extra_args, expected_error
):
    prompt, inputs = _write_inputs_and_prompt(tmp_path)

    result = runner.invoke(
        cli.app,
        [
            "--prompt-a",
            str(prompt),
            "--inputs",
            str(inputs),
            "--save-baseline",
            str(tmp_path / "baseline.json"),
            *extra_args,
        ],
    )

    assert result.exit_code == 1
    assert expected_error in result.output


@pytest.mark.parametrize(
    ("extra_args", "expected_error"),
    [
        (["--prompt-a", "a.txt"], "remove --prompt-a"),
        (["--runs", "3"], "cannot be combined with --baseline"),
        ([], "--baseline requires --prompt-b"),
    ],
)
def test_cli_baseline_rejects_incompatible_flags(tmp_path, extra_args, expected_error):
    prompt, inputs = _write_inputs_and_prompt(tmp_path)
    args = ["--inputs", str(inputs), "--baseline", str(tmp_path / "baseline.json")]
    if expected_error != "--baseline requires --prompt-b":
        args += ["--prompt-b", str(prompt)]

    result = runner.invoke(cli.app, args + extra_args)

    assert result.exit_code == 1
    assert expected_error in result.output


def test_cli_save_baseline_and_baseline_are_mutually_exclusive(tmp_path):
    _, inputs = _write_inputs_and_prompt(tmp_path)

    result = runner.invoke(
        cli.app,
        [
            "--inputs",
            str(inputs),
            "--save-baseline",
            str(tmp_path / "b1.json"),
            "--baseline",
            str(tmp_path / "b2.json"),
        ],
    )

    assert result.exit_code == 1
    assert "mutually exclusive" in result.output


def test_cli_baseline_compare_serves_side_a_from_file(tmp_path, monkeypatch):
    from llmdiff.baseline import build_baseline_document, render_baseline
    from llmdiff.config import ModelConfig as MC
    from llmdiff.config import SideConfig as SC

    prompt, inputs = _write_inputs_and_prompt(tmp_path)
    baseline_side = SC(prompt="saved prompt", model_cfg=MC(model="saved-model"))
    baseline_path = tmp_path / "baseline.json"
    baseline_path.write_text(
        render_baseline(
            build_baseline_document(
                baseline_side,
                [(PromptCase(id="case-1", user="hello"), "saved answer")],
            )
        ),
        encoding="utf-8",
    )

    captured = {}

    async def fake_run_diffs(cfg, **kwargs):
        captured["cfg"] = cfg
        captured["baseline_responses"] = kwargs.get("baseline_responses")
        return [_mk_diff("case-1", changed=False)]

    monkeypatch.setattr(cli, "run_diffs", fake_run_diffs)
    monkeypatch.setattr(cli, "render_case_inline", lambda *_a, **_k: None)
    monkeypatch.setattr(cli, "render_summary", lambda *_a, **_k: None)

    result = runner.invoke(
        cli.app,
        [
            "--prompt-b",
            str(prompt),
            "--inputs",
            str(inputs),
            "--baseline",
            str(baseline_path),
            "--no-semantic",
        ],
    )

    assert result.exit_code == 0
    assert captured["cfg"].side_a.prompt == "saved prompt"
    assert captured["cfg"].side_a.model_cfg.model == "saved-model"
    assert captured["baseline_responses"] == {"case-1": "saved answer"}


def test_cli_baseline_compare_rejects_stale_baseline(tmp_path):
    from llmdiff.baseline import build_baseline_document, render_baseline
    from llmdiff.config import ModelConfig as MC
    from llmdiff.config import SideConfig as SC

    prompt, inputs = _write_inputs_and_prompt(tmp_path)
    baseline_path = tmp_path / "baseline.json"
    baseline_path.write_text(
        render_baseline(
            build_baseline_document(
                SC(prompt="saved prompt", model_cfg=MC(model="m")),
                [(PromptCase(id="case-1", user="an older question"), "answer")],
            )
        ),
        encoding="utf-8",
    )

    result = runner.invoke(
        cli.app,
        [
            "--prompt-b",
            str(prompt),
            "--inputs",
            str(inputs),
            "--baseline",
            str(baseline_path),
            "--no-semantic",
        ],
    )

    assert result.exit_code == 1
    assert "input changed since the baseline: case-1" in result.output


def test_cli_runs_flag_enables_stability_mode(tmp_path, monkeypatch):
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
            "--runs",
            "5",
        ],
    )

    assert result.exit_code == 0
    assert captured["cfg"].runs == 5


def test_cli_runs_requires_semantic_scoring():
    result = runner.invoke(
        cli.app,
        [
            "--prompt-a",
            "missing-a.txt",
            "--prompt-b",
            "missing-b.txt",
            "--inputs",
            "missing-cases.json",
            "--runs",
            "3",
            "--no-semantic",
        ],
    )

    assert result.exit_code == 1
    assert "--runs requires semantic scoring" in result.output


def test_cli_seed_applies_to_both_sides(tmp_path, monkeypatch):
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
            "--seed",
            "7",
            "--no-semantic",
        ],
    )

    assert result.exit_code == 0
    cfg = captured["cfg"]
    assert cfg.side_a.model_cfg.seed == 7
    assert cfg.side_b.model_cfg.seed == 7


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


def _write_tagged_inputs(tmp_path):
    prompt_a = tmp_path / "prompt-a.txt"
    prompt_b = tmp_path / "prompt-b.txt"
    inputs = tmp_path / "cases.json"
    prompt_a.write_text("prompt a")
    prompt_b.write_text("prompt b")
    inputs.write_text(
        json.dumps(
            [
                {"id": "greet", "user": "hi", "tags": ["smoke"]},
                {"id": "refuse", "user": "no", "tags": ["safety", "slow"]},
                {"id": "untagged", "user": "hey"},
            ]
        )
    )
    return prompt_a, prompt_b, inputs


def _invoke_with_tags(tmp_path, monkeypatch, extra_args):
    prompt_a, prompt_b, inputs = _write_tagged_inputs(tmp_path)
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
            "--no-semantic",
            *extra_args,
        ],
    )
    return result, captured


def test_load_cases_accepts_and_strips_tags(tmp_path):
    path = tmp_path / "cases.json"
    path.write_text('[{"id":"case-1","user":"hello","tags":[" smoke "]}]')

    cases = cli._load_cases(path)

    assert cases[0].tags == ["smoke"]


def test_load_cases_rejects_blank_tags(tmp_path, capsys):
    path = tmp_path / "cases.json"
    path.write_text('[{"id":"case-1","user":"hello","tags":["  "]}]')

    with pytest.raises(typer.Exit):
        cli._load_cases(path)

    captured = capsys.readouterr()
    assert "tags must not be empty or whitespace" in captured.err


def test_cli_tag_selects_matching_cases(tmp_path, monkeypatch):
    result, captured = _invoke_with_tags(tmp_path, monkeypatch, ["--tag", "smoke"])

    assert result.exit_code == 0
    assert [c.id for c in captured["cfg"].cases] == ["greet"]
    assert "running 1 of 3 cases" in result.output


def test_cli_multiple_tags_match_any(tmp_path, monkeypatch):
    result, captured = _invoke_with_tags(
        tmp_path, monkeypatch, ["--tag", "smoke", "--tag", "safety"]
    )

    assert result.exit_code == 0
    assert [c.id for c in captured["cfg"].cases] == ["greet", "refuse"]


def test_cli_exclude_tag_drops_cases(tmp_path, monkeypatch):
    result, captured = _invoke_with_tags(
        tmp_path, monkeypatch, ["--exclude-tag", "slow"]
    )

    assert result.exit_code == 0
    assert [c.id for c in captured["cfg"].cases] == ["greet", "untagged"]


def test_cli_tag_and_exclude_tag_combine(tmp_path, monkeypatch):
    result, captured = _invoke_with_tags(
        tmp_path,
        monkeypatch,
        ["--tag", "smoke", "--tag", "safety", "--exclude-tag", "slow"],
    )

    assert result.exit_code == 0
    assert [c.id for c in captured["cfg"].cases] == ["greet"]


def test_cli_tag_with_no_matches_errors_with_available_tags(tmp_path, monkeypatch):
    result, _captured = _invoke_with_tags(tmp_path, monkeypatch, ["--tag", "nope"])

    assert result.exit_code == 1
    assert "no test cases match the tag filter" in result.output
    assert "available tags: safety, slow, smoke" in result.output


def test_cli_unknown_tag_warns_but_runs(tmp_path, monkeypatch):
    result, captured = _invoke_with_tags(
        tmp_path, monkeypatch, ["--tag", "smoke", "--tag", "typo"]
    )

    assert result.exit_code == 0
    assert "tag(s) not present in any case: typo" in result.output
    assert [c.id for c in captured["cfg"].cases] == ["greet"]


def test_cli_junit_output_writes_parseable_xml(tmp_path, monkeypatch):
    prompt_a = tmp_path / "prompt-a.txt"
    prompt_b = tmp_path / "prompt-b.txt"
    inputs = tmp_path / "cases.json"
    prompt_a.write_text("prompt a")
    prompt_b.write_text("prompt b")
    inputs.write_text(
        json.dumps([{"id": "case-1", "user": "x"}, {"id": "case-2", "user": "y"}])
    )
    report = tmp_path / "report.xml"

    async def fake_run_diffs(_cfg, **_kwargs):
        return [_mk_diff("case-1", changed=True), _mk_diff("case-2", changed=False)]

    monkeypatch.setattr(cli, "run_diffs", fake_run_diffs)

    result = runner.invoke(
        cli.app,
        [
            "--prompt-a",
            str(prompt_a),
            "--prompt-b",
            str(prompt_b),
            "--inputs",
            str(inputs),
            "--no-semantic",
            "--format",
            "junit",
            "--output",
            str(report),
        ],
    )

    assert result.exit_code == 0
    root = ElementTree.fromstring(report.read_text(encoding="utf-8"))
    assert root.get("tests") == "2"
    assert root.get("failures") == "1"
    assert root.find("./testsuite/testcase[@name='case-1']/failure") is not None
    assert root.find("./testsuite/testcase[@name='case-2']/failure") is None


def test_cli_sarif_output_points_at_inputs_file(tmp_path, monkeypatch):
    prompt_a = tmp_path / "prompt-a.txt"
    prompt_b = tmp_path / "prompt-b.txt"
    inputs = tmp_path / "cases.json"
    prompt_a.write_text("prompt a")
    prompt_b.write_text("prompt b")
    inputs.write_text(json.dumps([{"id": "case-1", "user": "hello"}], indent=2))
    report = tmp_path / "report.sarif"

    async def fake_run_diffs(_cfg, **_kwargs):
        return [_mk_diff("case-1", changed=True)]

    monkeypatch.setattr(cli, "run_diffs", fake_run_diffs)
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(
        cli.app,
        [
            "--prompt-a",
            "prompt-a.txt",
            "--prompt-b",
            "prompt-b.txt",
            "--inputs",
            "cases.json",
            "--no-semantic",
            "--format",
            "sarif",
            "--output",
            "report.sarif",
        ],
    )

    assert result.exit_code == 0
    sarif = json.loads(report.read_text(encoding="utf-8"))
    entry = sarif["runs"][0]["results"][0]
    assert "case-1" in entry["message"]["text"]
    location = entry["locations"][0]["physicalLocation"]
    assert location["artifactLocation"]["uri"] == "cases.json"
    # indent=2 puts the "id" key of the first case on line 3.
    assert location["region"]["startLine"] == 3


def test_cli_side_by_side_flag_sets_run_config(tmp_path, monkeypatch):
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
            "--side-by-side",
            "--no-semantic",
        ],
    )

    assert result.exit_code == 0
    assert captured["cfg"].side_by_side is True


def test_cli_side_by_side_requires_inline_format():
    result = runner.invoke(
        cli.app,
        [
            "--prompt-a",
            "missing-a.txt",
            "--prompt-b",
            "missing-b.txt",
            "--inputs",
            "missing-cases.json",
            "--side-by-side",
            "--format",
            "json",
        ],
    )

    assert result.exit_code == 1
    assert "--side-by-side only applies to the inline format" in result.output


@pytest.mark.asyncio
async def test_run_side_by_side_uses_columnar_renderer(monkeypatch):
    side_a = SideConfig(prompt="Prompt A", model_cfg=ModelConfig(model="llama3.2"))
    side_b = SideConfig(prompt="Prompt B", model_cfg=ModelConfig(model="llama3.2"))
    cfg = RunConfig(
        side_a=side_a,
        side_b=side_b,
        cases=[PromptCase(id="case-1", user="hello")],
        semantic=False,
        output_format=OutputFormat.INLINE,
        side_by_side=True,
    )

    async def fake_run_diffs(_cfg, **_kwargs):
        return [_mk_diff("case-1", changed=True)]

    rendered = {"inline": [], "side_by_side": []}
    monkeypatch.setattr(cli, "run_diffs", fake_run_diffs)
    monkeypatch.setattr(
        cli,
        "render_case_inline",
        lambda result, **_kwargs: rendered["inline"].append(result.case_id),
    )
    monkeypatch.setattr(
        cli,
        "render_case_side_by_side",
        lambda result, **_kwargs: rendered["side_by_side"].append(result.case_id),
    )
    monkeypatch.setattr(cli, "render_summary", lambda *_args, **_kwargs: None)

    await cli._run(cfg)

    assert rendered["side_by_side"] == ["case-1"]
    assert rendered["inline"] == []


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

    async def fake_run_diffs(_cfg, **_kwargs):
        return [
            _mk_diff("changed", changed=True),
            _mk_diff("unchanged", changed=False),
        ]

    rendered_case_ids = []
    rendered_summary = []

    def fake_render_case_inline(result, label_a, label_b, **_kwargs):
        rendered_case_ids.append(result.case_id)

    def fake_render_summary(summary):
        rendered_summary.append(summary)

    monkeypatch.setattr(cli, "run_diffs", fake_run_diffs)
    monkeypatch.setattr(cli, "render_case_inline", fake_render_case_inline)
    monkeypatch.setattr(cli, "render_summary", fake_render_summary)

    await cli._run(cfg)

    assert rendered_case_ids == ["changed"]
    assert len(rendered_summary) == 1
    assert rendered_summary[0].total == 2
    assert rendered_summary[0].changed == 1


@pytest.mark.asyncio
async def test_run_renders_results_from_runner_orchestration(monkeypatch):
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

    async def fake_run_diffs(_cfg, **_kwargs):
        return [
            _mk_diff("same", changed=False),
            _mk_diff("diff", changed=True),
        ]

    rendered_case_ids = []
    rendered_summary = []

    def fake_render_case_inline(result, label_a, label_b, **_kwargs):
        rendered_case_ids.append(result.case_id)

    def fake_render_summary(summary):
        rendered_summary.append(summary)

    monkeypatch.setattr(cli, "run_diffs", fake_run_diffs)
    monkeypatch.setattr(cli, "render_case_inline", fake_render_case_inline)
    monkeypatch.setattr(cli, "render_summary", fake_render_summary)

    await cli._run(cfg)

    assert rendered_case_ids == ["same", "diff"]
    assert len(rendered_summary) == 1
    assert rendered_summary[0].total == 2
    assert rendered_summary[0].changed == 1


@pytest.mark.asyncio
async def test_run_reports_runner_errors(monkeypatch):
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

    async def fake_run_diffs(_cfg, **_kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(cli, "run_diffs", fake_run_diffs)
    monkeypatch.setattr(cli, "render_case_inline", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(cli, "render_summary", lambda *_args, **_kwargs: None)

    with pytest.raises(typer.Exit):
        await cli._run(cfg)


@pytest.mark.asyncio
async def test_run_semantic_processes_cases_in_chunks(monkeypatch):
    side_a = SideConfig(prompt="Prompt A", model_cfg=ModelConfig(model="llama3.2"))
    side_b = SideConfig(prompt="Prompt B", model_cfg=ModelConfig(model="llama3.2"))
    cfg = RunConfig(
        side_a=side_a,
        side_b=side_b,
        cases=[
            PromptCase(id="case-1", user="hello"),
            PromptCase(id="case-2", user="hello"),
            PromptCase(id="case-3", user="hello"),
            PromptCase(id="case-4", user="hello"),
            PromptCase(id="case-5", user="hello"),
        ],
        semantic=True,
        semantic_batch_size=2,
        output_format=OutputFormat.INLINE,
    )

    chunk_calls = []
    check_models_flags = []

    async def fake_run_diffs(chunk_cfg, **_kwargs):
        chunk_calls.append([case.id for case in chunk_cfg.cases])
        check_models_flags.append(_kwargs.get("check_models"))
        return [_mk_diff(case.id, changed=False) for case in chunk_cfg.cases]

    monkeypatch.setattr(cli, "run_diffs", fake_run_diffs)
    monkeypatch.setattr(cli, "render_case_inline", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(cli, "render_summary", lambda *_args, **_kwargs: None)

    await cli._run(cfg)

    assert chunk_calls == [
        ["case-1", "case-2"],
        ["case-3", "case-4"],
        ["case-5"],
    ]
    # Only the first chunk pays for the model availability preflight.
    assert check_models_flags == [True, False, False]
