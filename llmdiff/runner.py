from __future__ import annotations
import asyncio
import httpx
from llmdiff.config import RunConfig, SideConfig, TestCase


def _response_detail(resp: httpx.Response) -> str:
    try:
        payload = resp.json()
    except ValueError:
        text = resp.text.strip()
        return text[:200] if text else "no response body"

    if isinstance(payload, dict):
        err = payload.get("error") or payload.get("message")
        if isinstance(err, str) and err.strip():
            return err.strip()

    return str(payload)[:200]


async def _call_ollama(
    client: httpx.AsyncClient,
    side: SideConfig,
    messages: list[dict[str, str]],
) -> str:
    payload = {
        "model": side.model_cfg.model,
        "stream": False,
        "messages": [
            {"role": "system", "content": side.prompt},
            *messages,
        ],
        "options": {
            "num_predict": side.model_cfg.max_tokens,
        },
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
        detail = _response_detail(e.response)
        if e.response.status_code == 404:
            raise RuntimeError(
                f"Model '{side.model_cfg.model}' not found in Ollama.\n"
                f"Pull it first:  ollama pull {side.model_cfg.model}\n"
                f"Details: {detail}"
            ) from None
        raise RuntimeError(
            f"Ollama request failed with status {e.response.status_code} "
            f"for model '{side.model_cfg.model}': {detail}"
        ) from None
    except httpx.TimeoutException:
        raise RuntimeError(
            f"Timed out waiting for model '{side.model_cfg.model}' at "
            f"{side.model_cfg.base_url}."
        ) from None
    except httpx.ConnectError:
        raise RuntimeError(
            f"Cannot connect to Ollama at {side.model_cfg.base_url}.\n"
            f"Is it running?  ollama serve"
        ) from None
    except httpx.RequestError as e:
        raise RuntimeError(
            f"Request to Ollama failed at {side.model_cfg.base_url}: {e}"
        ) from None

    try:
        body = resp.json()
    except ValueError:
        raise RuntimeError(
            "Ollama returned a non-JSON response from /api/chat."
        ) from None

    try:
        content = body["message"]["content"]
    except (KeyError, TypeError):
        raise RuntimeError(
            "Ollama response is missing expected field 'message.content'."
        ) from None

    if not isinstance(content, str):
        raise RuntimeError("Ollama response field 'message.content' must be text.")

    return content


async def check_models_available(
    client: httpx.AsyncClient,
    base_url: str,
    models: list[str],
) -> None:
    """Raises RuntimeError if any requested model is not pulled in Ollama."""
    try:
        resp = await client.get(f"{base_url}/api/tags", timeout=5.0)
        resp.raise_for_status()
    except httpx.TimeoutException:
        raise RuntimeError(
            f"Timed out while checking available models at {base_url}."
        ) from None
    except httpx.ConnectError:
        raise RuntimeError(
            f"Cannot connect to Ollama at {base_url}.\n" f"Start it with:  ollama serve"
        ) from None
    except httpx.HTTPStatusError as e:
        detail = _response_detail(e.response)
        raise RuntimeError(
            f"Failed to query Ollama models (status {e.response.status_code}): {detail}"
        ) from None
    except httpx.RequestError as e:
        raise RuntimeError(
            f"Failed to query Ollama models at {base_url}: {e}"
        ) from None

    try:
        body = resp.json()
    except ValueError:
        raise RuntimeError("Ollama /api/tags returned a non-JSON response.") from None

    models_payload = body.get("models")
    if not isinstance(models_payload, list):
        raise RuntimeError(
            "Unexpected Ollama /api/tags response: expected a 'models' array."
        )

    pulled = set()
    # also keep full names like "llama3.1:8b"
    pulled_full = set()
    for model_data in models_payload:
        if not isinstance(model_data, dict):
            continue
        name = model_data.get("name")
        if not isinstance(name, str):
            continue
        pulled.add(name.split(":")[0])
        pulled_full.add(name)

    available = pulled | pulled_full

    missing = [
        m for m in models if m not in available and m.split(":")[0] not in pulled
    ]
    if missing:
        missing_str = "\n".join(f"  ollama pull {m}" for m in missing)
        raise RuntimeError(f"Model(s) not found in Ollama:\n{missing_str}")


async def run_case(
    client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
    cfg: RunConfig,
    case: TestCase,
) -> tuple[str, str]:
    """Run both sides for a single test case. Returns (response_a, response_b)."""
    messages = [m.model_dump() for m in (case.context or [])]
    messages.append({"role": "user", "content": case.user})
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
