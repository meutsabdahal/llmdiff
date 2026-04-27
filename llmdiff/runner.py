from __future__ import annotations
import asyncio
import httpx
from llmdiff.config import RunConfig, SideConfig, TestCase


_DEFAULT_REQUEST_TIMEOUT_SECONDS = 120.0
_DEFAULT_MAX_RETRIES = 2
_DEFAULT_RETRY_BACKOFF_BASE_SECONDS = 0.5
MAX_RETRY_ATTEMPTS = 8
MAX_RETRY_BACKOFF_SECONDS = 8.0
_RETRYABLE_STATUS_CODES = {408, 409, 425, 429, 500, 502, 503, 504}


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


def _validate_request_policy(
    *,
    request_timeout: float,
    max_retries: int,
    retry_backoff_base: float,
) -> None:
    if request_timeout <= 0:
        raise ValueError("request_timeout must be greater than 0")
    if max_retries < 0:
        raise ValueError("max_retries must be 0 or greater")
    if max_retries > MAX_RETRY_ATTEMPTS:
        raise ValueError(f"max_retries must be {MAX_RETRY_ATTEMPTS} or less")
    if retry_backoff_base < 0:
        raise ValueError("retry_backoff_base must be 0 or greater")


def configure_request_policy(
    *,
    request_timeout: float,
    max_retries: int,
    retry_backoff_base: float,
) -> None:
    _validate_request_policy(
        request_timeout=request_timeout,
        max_retries=max_retries,
        retry_backoff_base=retry_backoff_base,
    )

    global _DEFAULT_REQUEST_TIMEOUT_SECONDS
    global _DEFAULT_MAX_RETRIES
    global _DEFAULT_RETRY_BACKOFF_BASE_SECONDS

    _DEFAULT_REQUEST_TIMEOUT_SECONDS = request_timeout
    _DEFAULT_MAX_RETRIES = max_retries
    _DEFAULT_RETRY_BACKOFF_BASE_SECONDS = retry_backoff_base


def _retry_delay_seconds(retry_attempt: int, backoff_base: float) -> float:
    if backoff_base <= 0:
        return 0.0

    delay = backoff_base * (2 ** (retry_attempt - 1))
    return min(delay, MAX_RETRY_BACKOFF_SECONDS)


async def _sleep_before_retry(retry_attempt: int, backoff_base: float) -> None:
    delay = _retry_delay_seconds(retry_attempt, backoff_base)
    if delay > 0:
        await asyncio.sleep(delay)


async def _call_ollama(
    client: httpx.AsyncClient,
    side: SideConfig,
    messages: list[dict[str, str]],
    request_timeout: float | None = None,
    max_retries: int | None = None,
    retry_backoff_base: float | None = None,
) -> str:
    request_timeout = (
        _DEFAULT_REQUEST_TIMEOUT_SECONDS if request_timeout is None else request_timeout
    )
    max_retries = _DEFAULT_MAX_RETRIES if max_retries is None else max_retries
    retry_backoff_base = (
        _DEFAULT_RETRY_BACKOFF_BASE_SECONDS
        if retry_backoff_base is None
        else retry_backoff_base
    )
    _validate_request_policy(
        request_timeout=request_timeout,
        max_retries=max_retries,
        retry_backoff_base=retry_backoff_base,
    )
    total_attempts = max_retries + 1

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

    for attempt in range(1, total_attempts + 1):
        try:
            resp = await client.post(
                f"{side.model_cfg.base_url}/api/chat",
                json=payload,
                timeout=request_timeout,
            )
            resp.raise_for_status()
        except httpx.HTTPStatusError as e:
            detail = _response_detail(e.response)
            status_code = e.response.status_code

            if status_code == 404:
                raise RuntimeError(
                    f"Model '{side.model_cfg.model}' not found in Ollama.\n"
                    f"Pull it first:  ollama pull {side.model_cfg.model}\n"
                    f"Details: {detail}"
                ) from None

            if status_code in _RETRYABLE_STATUS_CODES and attempt < total_attempts:
                await _sleep_before_retry(attempt, retry_backoff_base)
                continue

            attempt_suffix = (
                f" after {total_attempts} attempt(s)"
                if total_attempts > 1 and status_code in _RETRYABLE_STATUS_CODES
                else ""
            )
            raise RuntimeError(
                f"Ollama request failed with status {status_code}{attempt_suffix} "
                f"for model '{side.model_cfg.model}': {detail}"
            ) from None
        except httpx.TimeoutException:
            if attempt < total_attempts:
                await _sleep_before_retry(attempt, retry_backoff_base)
                continue

            raise RuntimeError(
                f"Timed out waiting for model '{side.model_cfg.model}' at "
                f"{side.model_cfg.base_url} after {total_attempts} attempt(s)."
            ) from None
        except httpx.ConnectError:
            if attempt < total_attempts:
                await _sleep_before_retry(attempt, retry_backoff_base)
                continue

            raise RuntimeError(
                f"Cannot connect to Ollama at {side.model_cfg.base_url} after "
                f"{total_attempts} attempt(s).\n"
                f"Is it running?  ollama serve"
            ) from None
        except httpx.TransportError as e:
            if attempt < total_attempts:
                await _sleep_before_retry(attempt, retry_backoff_base)
                continue

            raise RuntimeError(
                f"Request to Ollama failed at {side.model_cfg.base_url} after "
                f"{total_attempts} attempt(s): {e}"
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

    raise RuntimeError(
        f"Ollama request failed for model '{side.model_cfg.model}' after "
        f"{total_attempts} attempt(s)."
    )


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
