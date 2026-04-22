from __future__ import annotations
import asyncio
import httpx
from llmdiff.config import RunConfig, SideConfig, TestCase


async def _call_ollama(
    client: httpx.AsyncClient,
    side: SideConfig,
    messages: list[dict],
) -> str:
    payload = {
        "model": side.model_cfg.model,
        "stream": False,
        "messages": [
            {"role": "system", "content": side.prompt},
            *messages,
        ],
        "options": {},
    }
    if side.model_cfg.temperature is not None:
        payload["options"]["temperature"] = side.model_cfg.temperature

    try:
        resp = await client.post(
            f"{side.model_cfg.base_url}/api/chat",
            json=payload,
            timeout=120.0,
        )
        resp.raise_for_status()
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            raise RuntimeError(
                f"Model '{side.model_cfg.model}' not found in Ollama.\n"
                f"Pull it first:  ollama pull {side.model_cfg.model}"
            ) from None
        raise
    except httpx.ConnectError:
        raise RuntimeError(
            f"Cannot connect to Ollama at {side.model_cfg.base_url}.\n"
            f"Is it running?  ollama serve"
        ) from None

    return resp.json()["message"]["content"]


async def run_case(
    client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
    cfg: RunConfig,
    case: TestCase,
) -> tuple[str, str]:
    """Run both sides for a single test case. Returns (response_a, response_b)."""
    messages = (case.context or []) + [{"role": "user", "content": case.user}]
    async with semaphore:
        resp_a, resp_b = await asyncio.gather(
            _call_ollama(client, cfg.side_a, messages),
            _call_ollama(client, cfg.side_b, messages),
        )
    return resp_a, resp_b


async def run_all(cfg: RunConfig) -> list[tuple[TestCase, str, str]]:
    """
    Run all test cases concurrently (up to cfg.concurrency at a time).
    Returns list of (case, response_a, response_b).
    """
    semaphore = asyncio.Semaphore(cfg.concurrency)

    async with httpx.AsyncClient() as client:
        tasks = [run_case(client, semaphore, cfg, case) for case in cfg.cases]
        pairs = await asyncio.gather(*tasks)

    return [(case, resp_a, resp_b) for case, (resp_a, resp_b) in zip(cfg.cases, pairs)]
