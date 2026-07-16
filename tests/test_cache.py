import asyncio
import json

import httpx
import pytest

import llmdiff.runner as runner_module
from llmdiff.cache import ResponseCache, default_cache_dir
from llmdiff.config import ChatMessage, ModelConfig, RunConfig, SideConfig, TestCase
from llmdiff.metrics import SideTiming
from llmdiff.runner import run_case, run_diffs


def _side(**model_overrides) -> SideConfig:
    prompt = model_overrides.pop("prompt", "Prompt")
    model_kwargs = {"model": "llama3.2", **model_overrides}
    return SideConfig(prompt=prompt, model_cfg=ModelConfig(**model_kwargs))


_MESSAGES = [{"role": "user", "content": "hello"}]


def test_default_cache_dir_prefers_xdg_cache_home(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    assert default_cache_dir() == tmp_path / "xdg" / "llmdiff"

    monkeypatch.delenv("XDG_CACHE_HOME")
    assert default_cache_dir().parts[-2:] == (".cache", "llmdiff")


def test_cache_get_returns_none_for_missing_entry(tmp_path):
    cache = ResponseCache(cache_dir=tmp_path / "cache")
    assert cache.get(_side(), _MESSAGES) is None


def test_cache_round_trip(tmp_path):
    cache = ResponseCache(cache_dir=tmp_path / "cache")
    cache.set(_side(), _MESSAGES, "cached response")

    assert cache.get(_side(), _MESSAGES) == "cached response"


def test_cache_key_covers_prompt_model_parameters_and_case(tmp_path):
    cache = ResponseCache(cache_dir=tmp_path / "cache")
    cache.set(_side(), _MESSAGES, "baseline")

    variants = [
        (_side(prompt="Other prompt"), _MESSAGES),
        (_side(model="mistral"), _MESSAGES),
        (_side(base_url="http://other:11434"), _MESSAGES),
        (_side(temperature=0.7), _MESSAGES),
        (_side(max_tokens=2048), _MESSAGES),
        (_side(seed=7), _MESSAGES),
        (_side(), [{"role": "user", "content": "different input"}]),
        (
            _side(),
            [{"role": "assistant", "content": "prior turn"}, *_MESSAGES],
        ),
    ]

    for side, messages in variants:
        assert cache.get(side, messages) is None

    assert cache.get(_side(), _MESSAGES) == "baseline"


def test_cache_treats_corrupt_entries_as_misses(tmp_path):
    cache = ResponseCache(cache_dir=tmp_path / "cache")
    cache.set(_side(), _MESSAGES, "good")
    (entry,) = list((tmp_path / "cache").glob("*.json"))

    entry.write_text("not-json", encoding="utf-8")
    assert cache.get(_side(), _MESSAGES) is None

    entry.write_text(json.dumps({"response": 42}), encoding="utf-8")
    assert cache.get(_side(), _MESSAGES) is None


def test_cache_set_survives_unwritable_directory(tmp_path):
    blocker = tmp_path / "blocker"
    blocker.write_text("a file where the cache dir should go", encoding="utf-8")
    cache = ResponseCache(cache_dir=blocker / "cache")

    cache.set(_side(), _MESSAGES, "response")

    assert cache.get(_side(), _MESSAGES) is None


class CountingClient:
    def __init__(self):
        self.post_calls = 0

    async def post(self, url, json=None, timeout=None):
        self.post_calls += 1
        request = httpx.Request("POST", url)
        return httpx.Response(
            200,
            request=request,
            json={"message": {"content": f"response for {json['model']}"}},
        )


def _cfg() -> RunConfig:
    return RunConfig(
        side_a=SideConfig(prompt="Prompt A", model_cfg=ModelConfig(model="llama3.2")),
        side_b=SideConfig(prompt="Prompt B", model_cfg=ModelConfig(model="mistral")),
        cases=[
            TestCase(
                id="case-1",
                user="hello",
                context=[ChatMessage(role="assistant", content="earlier turn")],
            )
        ],
        semantic=False,
    )


@pytest.mark.asyncio
async def test_run_case_populates_and_reuses_cache(tmp_path):
    cache = ResponseCache(cache_dir=tmp_path / "cache")
    cfg = _cfg()
    semaphore = asyncio.Semaphore(1)

    client = CountingClient()
    first = await run_case(client, semaphore, cfg, cfg.cases[0], cache=cache)
    assert client.post_calls == 2

    client = CountingClient()
    second = await run_case(client, semaphore, cfg, cfg.cases[0], cache=cache)
    assert client.post_calls == 0
    # Same responses; timing is replayed from the cache and marked as such.
    assert second[:2] == first[:2]
    assert second[2] is not None and second[2].cached
    assert second[3] is not None and second[3].cached
    assert not first[2].cached and not first[3].cached


@pytest.mark.asyncio
async def test_run_case_without_cache_always_queries(tmp_path):
    cfg = _cfg()
    semaphore = asyncio.Semaphore(1)

    client = CountingClient()
    await run_case(client, semaphore, cfg, cfg.cases[0])
    await run_case(client, semaphore, cfg, cfg.cases[0])

    assert client.post_calls == 4


@pytest.mark.asyncio
async def test_run_diffs_skips_model_check_when_fully_cached(tmp_path, monkeypatch):
    cache = ResponseCache(cache_dir=tmp_path / "cache")
    cfg = _cfg()
    messages = runner_module._case_messages(cfg.cases[0])
    cache.set(cfg.side_a, messages, "answer a")
    cache.set(cfg.side_b, messages, "answer b")

    check_calls = []

    async def fake_check_models_available(_client, endpoint, models):
        check_calls.append(endpoint)

    monkeypatch.setattr(
        runner_module, "check_models_available", fake_check_models_available
    )

    results = await run_diffs(cfg, cache=cache)

    assert check_calls == []
    assert len(results) == 1
    assert results[0].response_a == "answer a"
    assert results[0].response_b == "answer b"


@pytest.mark.asyncio
async def test_run_diffs_checks_models_when_cache_incomplete(tmp_path, monkeypatch):
    cache = ResponseCache(cache_dir=tmp_path / "cache")
    cfg = _cfg()
    messages = runner_module._case_messages(cfg.cases[0])
    cache.set(cfg.side_a, messages, "answer a")

    check_calls = []

    async def fake_check_models_available(_client, endpoint, models):
        check_calls.append(endpoint)

    async def fake_run_case(_client, _semaphore, _cfg, _case, cache=None, baseline_responses=None):
        return "answer a", "answer b", None, None

    monkeypatch.setattr(
        runner_module, "check_models_available", fake_check_models_available
    )
    monkeypatch.setattr(runner_module, "run_case", fake_run_case)

    await run_diffs(cfg, cache=cache)

    assert len(check_calls) == 1


def test_cache_round_trips_timing(tmp_path):
    cache = ResponseCache(cache_dir=tmp_path)
    timing = SideTiming(latency_s=1.5, tokens=42, tokens_per_s=28.0)

    cache.set(_side(), _MESSAGES, "resp", timing=timing)
    hit = cache.get_with_timing(_side(), _MESSAGES)

    assert hit is not None
    response, replayed = hit
    assert response == "resp"
    assert replayed.latency_s == 1.5
    assert replayed.tokens == 42
    assert replayed.tokens_per_s == 28.0
    assert replayed.cached is True


def test_cache_entries_without_timing_return_none_timing(tmp_path):
    cache = ResponseCache(cache_dir=tmp_path)

    cache.set(_side(), _MESSAGES, "resp")
    hit = cache.get_with_timing(_side(), _MESSAGES)

    assert hit == ("resp", None)
