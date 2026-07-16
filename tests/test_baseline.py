import json

import httpx
import pytest

import llmdiff.runner as runner_module
from llmdiff.baseline import (
    BaselineError,
    build_baseline_document,
    case_input_hash,
    load_baseline,
    render_baseline,
    validate_baseline_cases,
)
from llmdiff.cache import ResponseCache
from llmdiff.config import (
    ChatMessage,
    ModelConfig,
    RunConfig,
    SideConfig,
    TestCase,
)
from llmdiff.runner import run_baseline_snapshot, run_diffs


def _case(case_id: str = "case-1", user: str = "hello", context=None) -> TestCase:
    return TestCase(id=case_id, user=user, context=context)


def _side(prompt: str = "Prompt", **model_overrides) -> SideConfig:
    model_kwargs = {"model": "llama3.2", **model_overrides}
    return SideConfig(prompt=prompt, model_cfg=ModelConfig(**model_kwargs))


def _write_baseline(path, side=None, responses=None) -> None:
    side = side or _side()
    responses = responses or [(_case(), "baseline answer")]
    path.write_text(
        render_baseline(build_baseline_document(side, responses)),
        encoding="utf-8",
    )


# --- case_input_hash ---


def test_case_input_hash_is_stable_and_ignores_id():
    assert case_input_hash(_case()) == case_input_hash(_case(case_id="other-id"))


def test_case_input_hash_changes_with_user_and_context():
    base = case_input_hash(_case())

    assert case_input_hash(_case(user="different")) != base
    assert (
        case_input_hash(
            _case(context=[ChatMessage(role="assistant", content="turn")])
        )
        != base
    )


# --- build / render / load round trip ---


def test_baseline_round_trip(tmp_path):
    side = _side(prompt="You are terse.", temperature=0.5, seed=7)
    cases = [
        _case("c1", "hello"),
        _case("c2", "bye", context=[ChatMessage(role="user", content="hi")]),
    ]
    path = tmp_path / "baseline.json"
    _write_baseline(path, side, [(cases[0], "answer 1"), (cases[1], "answer 2")])

    loaded = load_baseline(path)

    assert loaded.prompt == "You are terse."
    assert loaded.model_cfg == side.model_cfg
    assert loaded.responses == {"c1": "answer 1", "c2": "answer 2"}
    assert loaded.input_hashes == {
        "c1": case_input_hash(cases[0]),
        "c2": case_input_hash(cases[1]),
    }
    assert loaded.created_at
    assert loaded.side_config() == side


def test_load_baseline_rejects_missing_file(tmp_path):
    with pytest.raises(BaselineError, match="not found"):
        load_baseline(tmp_path / "nope.json")


def test_load_baseline_rejects_invalid_json(tmp_path):
    path = tmp_path / "baseline.json"
    path.write_text("not-json", encoding="utf-8")

    with pytest.raises(BaselineError, match="invalid JSON"):
        load_baseline(path)


def test_load_baseline_rejects_non_baseline_document(tmp_path):
    path = tmp_path / "baseline.json"
    path.write_text(json.dumps({"summary": {}, "cases": []}), encoding="utf-8")

    with pytest.raises(BaselineError, match="version"):
        load_baseline(path)


def test_load_baseline_rejects_newer_schema(tmp_path):
    path = tmp_path / "baseline.json"
    _write_baseline(path)
    doc = json.loads(path.read_text(encoding="utf-8"))
    doc["version"] = 99
    path.write_text(json.dumps(doc), encoding="utf-8")

    with pytest.raises(BaselineError, match="newer"):
        load_baseline(path)


def test_load_baseline_rejects_duplicate_case_ids(tmp_path):
    path = tmp_path / "baseline.json"
    _write_baseline(
        path, responses=[(_case("c1"), "one"), (_case("c1", "other"), "two")]
    )

    with pytest.raises(BaselineError, match="duplicate case id 'c1'"):
        load_baseline(path)


def test_load_baseline_rejects_non_string_response(tmp_path):
    path = tmp_path / "baseline.json"
    _write_baseline(path)
    doc = json.loads(path.read_text(encoding="utf-8"))
    doc["cases"][0]["response"] = 42
    path.write_text(json.dumps(doc), encoding="utf-8")

    with pytest.raises(BaselineError, match="'response'"):
        load_baseline(path)


# --- validate_baseline_cases ---


def test_validate_baseline_cases_accepts_matching_cases(tmp_path):
    path = tmp_path / "baseline.json"
    _write_baseline(path)

    validate_baseline_cases(load_baseline(path), [_case()])


def test_validate_baseline_cases_reports_missing_and_changed(tmp_path):
    path = tmp_path / "baseline.json"
    _write_baseline(
        path, responses=[(_case("c1"), "one"), (_case("c2", "bye"), "two")]
    )
    current = [
        _case("c1", "hello"),  # unchanged
        _case("c2", "bye but edited"),  # input changed since snapshot
        _case("c3", "new case"),  # not snapshotted
    ]

    with pytest.raises(BaselineError) as exc_info:
        validate_baseline_cases(load_baseline(path), current)

    message = str(exc_info.value)
    assert "not in the baseline: c3" in message
    assert "input changed since the baseline: c2" in message
    assert "--save-baseline" in message


# --- run_baseline_snapshot ---


class CountingClient:
    def __init__(self):
        self.post_calls = 0
        self.get_calls = 0

    async def post(self, url, json=None, timeout=None):
        self.post_calls += 1
        request = httpx.Request("POST", url)
        return httpx.Response(
            200,
            request=request,
            json={"message": {"content": f"response {self.post_calls}"}},
        )

    async def get(self, url, timeout=None):
        self.get_calls += 1
        request = httpx.Request("GET", url)
        return httpx.Response(
            200,
            request=request,
            json={"models": [{"name": "llama3.2:latest"}]},
        )

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return None


@pytest.mark.asyncio
async def test_run_baseline_snapshot_returns_response_per_case(monkeypatch):
    client = CountingClient()
    monkeypatch.setattr(runner_module.httpx, "AsyncClient", lambda: client)
    cases = [_case("c1", "hello"), _case("c2", "bye")]

    completed = []
    responses = await run_baseline_snapshot(
        _side(),
        cases,
        on_case_completed=lambda case: completed.append(case.id),
    )

    assert [case.id for case, _ in responses] == ["c1", "c2"]
    assert all(response for _, response in responses)
    assert client.post_calls == 2
    assert client.get_calls == 1  # one model preflight
    assert sorted(completed) == ["c1", "c2"]


@pytest.mark.asyncio
async def test_run_baseline_snapshot_uses_cache(tmp_path, monkeypatch):
    cache = ResponseCache(cache_dir=tmp_path / "cache")
    cases = [_case("c1", "hello")]

    client = CountingClient()
    monkeypatch.setattr(runner_module.httpx, "AsyncClient", lambda: client)
    first = await run_baseline_snapshot(_side(), cases, cache=cache)
    assert client.post_calls == 1

    client = CountingClient()
    monkeypatch.setattr(runner_module.httpx, "AsyncClient", lambda: client)
    second = await run_baseline_snapshot(_side(), cases, cache=cache)
    assert client.post_calls == 0
    assert client.get_calls == 0  # fully cached: no preflight either
    assert [r for _, r in second] == [r for _, r in first]


# --- run_diffs with baseline_responses ---


def _cfg(**overrides) -> RunConfig:
    defaults = dict(
        side_a=_side("Baseline prompt"),
        side_b=_side("New prompt"),
        cases=[_case("c1", "hello")],
        semantic=False,
    )
    defaults.update(overrides)
    return RunConfig(**defaults)


@pytest.mark.asyncio
async def test_run_diffs_serves_side_a_from_baseline(monkeypatch):
    client = CountingClient()
    monkeypatch.setattr(runner_module.httpx, "AsyncClient", lambda: client)

    checked_endpoints = []

    async def fake_check_models_available(_client, endpoint, models):
        checked_endpoints.append((endpoint, tuple(models)))

    monkeypatch.setattr(
        runner_module, "check_models_available", fake_check_models_available
    )

    cfg = _cfg(
        side_a=_side("Baseline prompt", model="saved-model"),
        side_b=_side("New prompt", base_url="http://b:11434"),
    )
    results = await run_diffs(
        cfg, baseline_responses={"c1": "saved baseline answer"}
    )

    (result,) = results
    assert result.response_a == "saved baseline answer"
    assert result.response_b == "response 1"
    assert client.post_calls == 1  # only side B queried
    # Preflight covers side B's endpoint only; the baseline model is never
    # required to exist.
    assert checked_endpoints == [("http://b:11434", ("llama3.2",))]


@pytest.mark.asyncio
async def test_run_diffs_errors_when_case_missing_from_baseline(monkeypatch):
    client = CountingClient()
    monkeypatch.setattr(runner_module.httpx, "AsyncClient", lambda: client)

    async def fake_check_models_available(*_args, **_kwargs):
        return None

    monkeypatch.setattr(
        runner_module, "check_models_available", fake_check_models_available
    )

    with pytest.raises(RuntimeError, match="missing from the baseline"):
        await run_diffs(_cfg(), baseline_responses={"other-case": "answer"})


@pytest.mark.asyncio
async def test_run_diffs_rejects_baseline_with_stability_mode():
    cfg = _cfg(runs=3, semantic=True)

    with pytest.raises(RuntimeError, match="cannot be combined"):
        await run_diffs(cfg, baseline_responses={"c1": "answer"})
