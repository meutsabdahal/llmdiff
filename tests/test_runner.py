import pytest
import httpx

import llmdiff.runner as runner
from llmdiff.config import ModelConfig, RunConfig, SideConfig, TestCase
from llmdiff.runner import _call_ollama, check_models_available


class FakeAsyncClient:
    def __init__(
        self,
        *,
        post_response: httpx.Response | None = None,
        get_response: httpx.Response | None = None,
        post_exc: Exception | None = None,
        get_exc: Exception | None = None,
    ):
        self.post_response = post_response
        self.get_response = get_response
        self.post_exc = post_exc
        self.get_exc = get_exc

    async def post(self, *_args, **_kwargs):
        if self.post_exc is not None:
            raise self.post_exc
        assert self.post_response is not None
        return self.post_response

    async def get(self, *_args, **_kwargs):
        if self.get_exc is not None:
            raise self.get_exc
        assert self.get_response is not None
        return self.get_response


def _response(
    method: str,
    url: str,
    status_code: int,
    *,
    json_body: dict | list | None = None,
    text_body: str = "",
) -> httpx.Response:
    request = httpx.Request(method, url)
    if json_body is not None:
        return httpx.Response(status_code, request=request, json=json_body)
    return httpx.Response(status_code, request=request, text=text_body)


def _side() -> SideConfig:
    return SideConfig(prompt="Prompt", model_cfg=ModelConfig(model="llama3.2"))


@pytest.mark.asyncio
async def test_call_ollama_handles_timeout():
    request = httpx.Request("POST", "http://localhost:11434/api/chat")
    client = FakeAsyncClient(post_exc=httpx.ReadTimeout("timed out", request=request))

    with pytest.raises(RuntimeError, match="Timed out"):
        await _call_ollama(client, _side(), [])


@pytest.mark.asyncio
async def test_call_ollama_handles_404_model_not_found():
    client = FakeAsyncClient(
        post_response=_response(
            "POST",
            "http://localhost:11434/api/chat",
            404,
            json_body={"error": "model not found"},
        )
    )

    with pytest.raises(RuntimeError, match="Pull it first"):
        await _call_ollama(client, _side(), [])


@pytest.mark.asyncio
async def test_call_ollama_handles_non_json_response_body():
    client = FakeAsyncClient(
        post_response=_response(
            "POST",
            "http://localhost:11434/api/chat",
            200,
            text_body="not-json",
        )
    )

    with pytest.raises(RuntimeError, match="non-JSON"):
        await _call_ollama(client, _side(), [])


@pytest.mark.asyncio
async def test_call_ollama_handles_missing_message_content():
    client = FakeAsyncClient(
        post_response=_response(
            "POST",
            "http://localhost:11434/api/chat",
            200,
            json_body={"message": {}},
        )
    )

    with pytest.raises(RuntimeError, match="message.content"):
        await _call_ollama(client, _side(), [])


@pytest.mark.asyncio
async def test_check_models_available_handles_http_status_error():
    client = FakeAsyncClient(
        get_response=_response(
            "GET",
            "http://localhost:11434/api/tags",
            500,
            json_body={"error": "internal error"},
        )
    )

    with pytest.raises(RuntimeError, match="status 500"):
        await check_models_available(client, "http://localhost:11434", ["llama3.2"])


@pytest.mark.asyncio
async def test_check_models_available_handles_non_json_response():
    client = FakeAsyncClient(
        get_response=_response(
            "GET",
            "http://localhost:11434/api/tags",
            200,
            text_body="oops",
        )
    )

    with pytest.raises(RuntimeError, match="non-JSON"):
        await check_models_available(client, "http://localhost:11434", ["llama3.2"])


@pytest.mark.asyncio
async def test_check_models_available_handles_invalid_shape():
    client = FakeAsyncClient(
        get_response=_response(
            "GET",
            "http://localhost:11434/api/tags",
            200,
            json_body={"models": "not-a-list"},
        )
    )

    with pytest.raises(RuntimeError, match="models' array"):
        await check_models_available(client, "http://localhost:11434", ["llama3.2"])


@pytest.mark.asyncio
async def test_check_models_available_reports_endpoint_for_missing_models():
    client = FakeAsyncClient(
        get_response=_response(
            "GET",
            "http://localhost:11434/api/tags",
            200,
            json_body={"models": [{"name": "llama3.2:latest"}]},
        )
    )

    with pytest.raises(
        RuntimeError,
        match=r"Model\(s\) not found in Ollama at http://localhost:11434",
    ):
        await check_models_available(client, "http://localhost:11434", ["mistral"])


@pytest.mark.asyncio
async def test_run_diffs_checks_models_for_each_endpoint(monkeypatch):
    cfg = RunConfig(
        side_a=SideConfig(
            prompt="Prompt A",
            model_cfg=ModelConfig(model="llama3.2", base_url="http://a:11434"),
        ),
        side_b=SideConfig(
            prompt="Prompt B",
            model_cfg=ModelConfig(model="mistral", base_url="http://b:11434"),
        ),
        cases=[TestCase(id="case-1", user="hello")],
        semantic=False,
    )

    calls = []

    async def fake_check_models_available(_client, endpoint, models):
        calls.append((endpoint, tuple(models)))

    async def fake_run_case(_client, _semaphore, _cfg, _case):
        return "same", "same"

    monkeypatch.setattr(runner, "check_models_available", fake_check_models_available)
    monkeypatch.setattr(runner, "run_case", fake_run_case)

    results = await runner.run_diffs(cfg)

    assert sorted(calls) == [
        ("http://a:11434", ("llama3.2",)),
        ("http://b:11434", ("mistral",)),
    ]
    assert len(results) == 1
    assert results[0].case_id == "case-1"
    assert not results[0].changed


@pytest.mark.asyncio
async def test_run_diffs_uses_batched_semantic_scoring(monkeypatch):
    cfg = RunConfig(
        side_a=SideConfig(prompt="Prompt A", model_cfg=ModelConfig(model="llama3.2")),
        side_b=SideConfig(prompt="Prompt B", model_cfg=ModelConfig(model="llama3.2")),
        cases=[
            TestCase(id="same", user="hello"),
            TestCase(id="diff", user="hello"),
        ],
        semantic=True,
        semantic_batch_size=2,
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

    monkeypatch.setattr(runner, "check_models_available", fake_check_models_available)
    monkeypatch.setattr(runner, "run_case", fake_run_case)
    monkeypatch.setattr(runner, "semantic_similarities", fake_semantic_similarities)

    results = await runner.run_diffs(cfg)

    assert semantic_calls["batch_size"] == 2
    assert semantic_calls["pairs"] == [
        ("same output", "same output"),
        ("left output", "right output"),
    ]
    assert [r.case_id for r in results] == ["same", "diff"]
    assert results[0].similarity == pytest.approx(0.99)
    assert results[1].similarity == pytest.approx(0.22)
    assert not results[0].changed
    assert results[1].changed


@pytest.mark.asyncio
async def test_run_diffs_invokes_progress_callbacks(monkeypatch):
    cfg = RunConfig(
        side_a=SideConfig(prompt="Prompt A", model_cfg=ModelConfig(model="llama3.2")),
        side_b=SideConfig(prompt="Prompt B", model_cfg=ModelConfig(model="llama3.2")),
        cases=[TestCase(id="case-1", user="hello")],
        semantic=True,
    )

    async def fake_check_models_available(*_args, **_kwargs):
        return None

    async def fake_run_case(_client, _semaphore, _cfg, _case):
        return "same output", "same output"

    def fake_semantic_similarities(_pairs, _batch_size):
        return [1.0]

    callbacks = {
        "cases": [],
        "semantic_started": 0,
        "semantic_completed": 0,
    }

    def on_case_completed(case: TestCase) -> None:
        callbacks["cases"].append(case.id)

    def on_semantic_scoring_start() -> None:
        callbacks["semantic_started"] += 1

    def on_semantic_scoring_complete() -> None:
        callbacks["semantic_completed"] += 1

    monkeypatch.setattr(runner, "check_models_available", fake_check_models_available)
    monkeypatch.setattr(runner, "run_case", fake_run_case)
    monkeypatch.setattr(runner, "semantic_similarities", fake_semantic_similarities)

    await runner.run_diffs(
        cfg,
        on_case_completed=on_case_completed,
        on_semantic_scoring_start=on_semantic_scoring_start,
        on_semantic_scoring_complete=on_semantic_scoring_complete,
    )

    assert callbacks["cases"] == ["case-1"]
    assert callbacks["semantic_started"] == 1
    assert callbacks["semantic_completed"] == 1
