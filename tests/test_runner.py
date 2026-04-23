import pytest
import httpx

from llmdiff.config import ModelConfig, SideConfig
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
