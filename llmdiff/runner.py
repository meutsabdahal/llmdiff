from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from itertools import combinations

import httpx

from llmdiff.cache import ResponseCache
from llmdiff.config import RunConfig, SideConfig, TestCase
from llmdiff.differ import DiffResult, compute_diff
from llmdiff.metrics import (
    SideTiming,
    aggregate_timings,
    compute_stability_stats,
    semantic_similarities,
)

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
) -> tuple[str, SideTiming]:
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

    options: dict[str, float | int] = {"num_predict": side.model_cfg.max_tokens}
    if side.model_cfg.temperature is not None:
        options["temperature"] = side.model_cfg.temperature
    if side.model_cfg.seed is not None:
        options["seed"] = side.model_cfg.seed

    payload = {
        "model": side.model_cfg.model,
        "stream": False,
        "messages": [
            {"role": "system", "content": side.prompt},
            *messages,
        ],
        "options": options,
    }

    for attempt in range(1, total_attempts + 1):
        try:
            # Timed per attempt so retries report the successful request's
            # latency, not the sum of failed attempts and backoff sleeps.
            request_started = time.perf_counter()
            resp = await client.post(
                f"{side.model_cfg.base_url}/api/chat",
                json=payload,
                timeout=request_timeout,
            )
            latency_s = time.perf_counter() - request_started
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

        # Throughput from Ollama's own generation counters when present;
        # eval_duration is nanoseconds.
        eval_count = body.get("eval_count")
        eval_duration = body.get("eval_duration")
        tokens = (
            eval_count
            if isinstance(eval_count, int) and not isinstance(eval_count, bool)
            else None
        )
        tokens_per_s = None
        if (
            tokens is not None
            and isinstance(eval_duration, int)
            and eval_duration > 0
        ):
            tokens_per_s = tokens / (eval_duration / 1e9)

        return content, SideTiming(
            latency_s=latency_s, tokens=tokens, tokens_per_s=tokens_per_s
        )

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

    def _is_missing(requested: str) -> bool:
        # A tagged request must match the exact pulled tag; matching on the
        # base name alone would pass preflight for e.g. "llama3.1:70b" when
        # only "llama3.1:8b" is pulled, deferring the failure to a 404 later.
        if ":" in requested:
            return requested not in pulled_full
        return requested not in pulled

    missing = [m for m in models if _is_missing(m)]
    if missing:
        missing_str = "\n".join(f"  ollama pull {m}" for m in missing)
        raise RuntimeError(
            f"Model(s) not found in Ollama at {base_url}:\n{missing_str}"
        )


def _models_needed_by_endpoint(
    sides: tuple[SideConfig, ...],
) -> dict[str, list[str]]:
    models_needed_by_endpoint: dict[str, set[str]] = {}

    for side in sides:
        base_url = side.model_cfg.base_url
        if base_url not in models_needed_by_endpoint:
            models_needed_by_endpoint[base_url] = set()
        models_needed_by_endpoint[base_url].add(side.model_cfg.model)

    return {
        endpoint: sorted(models)
        for endpoint, models in models_needed_by_endpoint.items()
    }


async def ensure_models_available(
    client: httpx.AsyncClient,
    sides: tuple[SideConfig, ...],
) -> None:
    """Raises RuntimeError if any given side's model is missing."""
    for endpoint, models_needed in _models_needed_by_endpoint(sides).items():
        await check_models_available(client, endpoint, models_needed)


def _case_messages(case: TestCase) -> list[dict[str, str]]:
    return case.messages()


def _side_for_sample(side: SideConfig, sample: int) -> SideConfig:
    """Returns the side config used for one stability-mode sample.

    When a seed is configured, sample i runs with seed + i: repeated samples
    stay reproducible without collapsing into N identical generations.
    """
    if sample == 0 or side.model_cfg.seed is None:
        return side

    model_cfg = side.model_cfg.model_copy(
        update={"seed": side.model_cfg.seed + sample}
    )
    return side.model_copy(update={"model_cfg": model_cfg})


def _all_responses_cached(
    cfg: RunConfig,
    cache: ResponseCache | None,
    sides: tuple[SideConfig, ...],
) -> bool:
    if cache is None:
        return False

    for case in cfg.cases:
        messages = _case_messages(case)
        for side in sides:
            for sample in range(cfg.runs):
                side_variant = _side_for_sample(side, sample)
                if cache.get(side_variant, messages, sample=sample) is None:
                    return False

    return True


async def run_case(
    client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
    cfg: RunConfig,
    case: TestCase,
    cache: ResponseCache | None = None,
    baseline_responses: dict[str, str] | None = None,
) -> tuple[str, str, SideTiming | None, SideTiming | None]:
    """Run both sides for a single test case.

    Returns (response_a, response_b, timing_a, timing_b). Timing is None for
    a baseline side A and for cache entries written before timing existed.

    With baseline_responses, side A is the saved snapshot response and only
    side B is queried.
    """
    messages = _case_messages(case)

    async def side_response(side: SideConfig) -> tuple[str, SideTiming | None]:
        if cache is not None:
            cached = cache.get_with_timing(side, messages)
            if cached is not None:
                return cached

        response, timing = await _call_ollama(client, side, messages)
        if cache is not None:
            cache.set(side, messages, response, timing=timing)
        return response, timing

    if baseline_responses is not None:
        if case.id not in baseline_responses:
            raise RuntimeError(
                f"Case '{case.id}' is missing from the baseline; "
                "re-create it with --save-baseline."
            )
        async with semaphore:
            resp_b, timing_b = await side_response(cfg.side_b)
        return baseline_responses[case.id], resp_b, None, timing_b

    async with semaphore:
        (resp_a, timing_a), (resp_b, timing_b) = await asyncio.gather(
            side_response(cfg.side_a),
            side_response(cfg.side_b),
        )
    return resp_a, resp_b, timing_a, timing_b


async def run_baseline_snapshot(
    side: SideConfig,
    cases: list[TestCase],
    concurrency: int = 3,
    cache: ResponseCache | None = None,
    check_models: bool = True,
    on_case_completed: Callable[[TestCase], None] | None = None,
) -> list[tuple[TestCase, str]]:
    """Run a single side over all cases and return (case, response) pairs.

    Backs --save-baseline: no diffing, no second side, just the responses
    the baseline document stores.
    """
    semaphore = asyncio.Semaphore(concurrency)

    async with httpx.AsyncClient() as client:
        if check_models:
            all_cached = cache is not None and all(
                cache.get(side, _case_messages(case)) is not None for case in cases
            )
            if not all_cached:
                await ensure_models_available(client, (side,))

        async def run_one(case: TestCase) -> tuple[TestCase, str]:
            messages = _case_messages(case)
            if cache is not None:
                cached = cache.get(side, messages)
                if cached is not None:
                    if on_case_completed is not None:
                        on_case_completed(case)
                    return case, cached

            async with semaphore:
                response, timing = await _call_ollama(client, side, messages)
            if cache is not None:
                cache.set(side, messages, response, timing=timing)
            if on_case_completed is not None:
                on_case_completed(case)
            return case, response

        return list(await asyncio.gather(*[run_one(case) for case in cases]))


async def run_case_samples(
    client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
    cfg: RunConfig,
    case: TestCase,
    cache: ResponseCache | None = None,
) -> tuple[list[str], list[str], list[SideTiming | None], list[SideTiming | None]]:
    """Run both sides cfg.runs times for one case (stability mode).

    Samples run sequentially with the two sides in parallel, so one case
    holds one semaphore slot and never exceeds the two concurrent requests
    a single-run case would issue.
    """
    messages = _case_messages(case)

    async def side_response(
        side: SideConfig, sample: int
    ) -> tuple[str, SideTiming | None]:
        side_variant = _side_for_sample(side, sample)
        if cache is not None:
            cached = cache.get_with_timing(side_variant, messages, sample=sample)
            if cached is not None:
                return cached

        response, timing = await _call_ollama(client, side_variant, messages)
        if cache is not None:
            cache.set(side_variant, messages, response, sample=sample, timing=timing)
        return response, timing

    samples_a: list[str] = []
    samples_b: list[str] = []
    timings_a: list[SideTiming | None] = []
    timings_b: list[SideTiming | None] = []
    async with semaphore:
        for sample in range(cfg.runs):
            (resp_a, timing_a), (resp_b, timing_b) = await asyncio.gather(
                side_response(cfg.side_a, sample),
                side_response(cfg.side_b, sample),
            )
            samples_a.append(resp_a)
            samples_b.append(resp_b)
            timings_a.append(timing_a)
            timings_b.append(timing_b)

    return samples_a, samples_b, timings_a, timings_b


async def _run_case_responses(
    client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
    cfg: RunConfig,
    on_case_completed: Callable[[TestCase], None] | None = None,
    cache: ResponseCache | None = None,
    baseline_responses: dict[str, str] | None = None,
) -> list[tuple[TestCase, str, str, SideTiming | None, SideTiming | None]]:
    async def run_case_and_track(
        case: TestCase,
    ) -> tuple[TestCase, str, str, SideTiming | None, SideTiming | None]:
        resp_a, resp_b, timing_a, timing_b = await run_case(
            client,
            semaphore,
            cfg,
            case,
            cache=cache,
            baseline_responses=baseline_responses,
        )
        if on_case_completed is not None:
            on_case_completed(case)
        return case, resp_a, resp_b, timing_a, timing_b

    return await asyncio.gather(*[run_case_and_track(case) for case in cfg.cases])


def _stability_pairs(
    samples_a: list[str],
    samples_b: list[str],
) -> list[tuple[str, str]]:
    """Score pairs for one case, in a fixed layout consumed positionally:

    runs cross pairs (A_i, B_i), then C(runs, 2) self pairs within A, then
    C(runs, 2) self pairs within B.
    """
    pairs = list(zip(samples_a, samples_b))
    pairs.extend(combinations(samples_a, 2))
    pairs.extend(combinations(samples_b, 2))
    return pairs


def _pairs_per_case(runs: int) -> int:
    self_pairs = runs * (runs - 1) // 2
    return runs + 2 * self_pairs


async def _run_stability_diffs(
    cfg: RunConfig,
    on_case_completed: Callable[[TestCase], None] | None = None,
    on_semantic_scoring_start: Callable[[], None] | None = None,
    on_semantic_scoring_complete: Callable[[], None] | None = None,
    check_models: bool = True,
    cache: ResponseCache | None = None,
) -> list[DiffResult]:
    """Stability-mode variant of run_diffs: cfg.runs samples per case/side."""
    if not cfg.semantic:
        raise RuntimeError(
            "Stability mode (runs > 1) requires semantic scoring; "
            "remove --no-semantic."
        )

    semaphore = asyncio.Semaphore(cfg.concurrency)
    sides = (cfg.side_a, cfg.side_b)

    async with httpx.AsyncClient() as client:
        if check_models and not _all_responses_cached(cfg, cache, sides):
            await ensure_models_available(client, sides)

        async def run_case_and_track(
            case: TestCase,
        ) -> tuple[
            TestCase,
            list[str],
            list[str],
            list[SideTiming | None],
            list[SideTiming | None],
        ]:
            samples_a, samples_b, timings_a, timings_b = await run_case_samples(
                client, semaphore, cfg, case, cache=cache
            )
            if on_case_completed is not None:
                on_case_completed(case)
            return case, samples_a, samples_b, timings_a, timings_b

        responses = await asyncio.gather(
            *[run_case_and_track(case) for case in cfg.cases]
        )

    if on_semantic_scoring_start is not None:
        on_semantic_scoring_start()

    pairs: list[tuple[str, str]] = []
    for _, samples_a, samples_b, _, _ in responses:
        pairs.extend(_stability_pairs(samples_a, samples_b))

    loop = asyncio.get_running_loop()
    scores = await loop.run_in_executor(
        None,
        semantic_similarities,
        pairs,
        cfg.semantic_batch_size,
    )

    if len(scores) != len(pairs):
        raise RuntimeError("Semantic scoring returned an unexpected number of scores.")

    if on_semantic_scoring_complete is not None:
        on_semantic_scoring_complete()

    per_case = _pairs_per_case(cfg.runs)
    self_pairs = cfg.runs * (cfg.runs - 1) // 2
    results = []
    for index, (case, samples_a, samples_b, timings_a, timings_b) in enumerate(
        responses
    ):
        base = index * per_case
        cross = scores[base : base + cfg.runs]
        self_a = scores[base + cfg.runs : base + cfg.runs + self_pairs]
        self_b = scores[base + cfg.runs + self_pairs : base + per_case]
        stats = compute_stability_stats(cross, self_a, self_b)

        results.append(
            compute_diff(
                case_id=case.id,
                response_a=samples_a[0],
                response_b=samples_b[0],
                similarity=stats.similarity_mean,
                threshold=cfg.threshold,
                changed_when=cfg.changed_when,
                stability=stats,
                timing_a=aggregate_timings(timings_a),
                timing_b=aggregate_timings(timings_b),
                diff_mode=cfg.diff_mode,
                ignore_whitespace=cfg.ignore_whitespace,
                ignore_case=cfg.ignore_case,
            )
        )

    return results


async def run_diffs(
    cfg: RunConfig,
    on_case_completed: Callable[[TestCase], None] | None = None,
    on_semantic_scoring_start: Callable[[], None] | None = None,
    on_semantic_scoring_complete: Callable[[], None] | None = None,
    check_models: bool = True,
    cache: ResponseCache | None = None,
    baseline_responses: dict[str, str] | None = None,
) -> list[DiffResult]:
    """
    Execute a full llmdiff run and return computed diffs.

    Steps:
    1. Validate required models are available for each configured endpoint
       (skipped when check_models is False, e.g. for follow-up chunks of a
       run that already validated them, or when every response is already
       cached — a fully cached run must not require a reachable Ollama).
    2. Run all prompt cases concurrently, reusing cached responses when a
       cache is provided.
    3. Optionally compute semantic similarity scores in batches.
    4. Compute line-level diffs and change status for each case.

    When cfg.runs > 1 (stability mode), each case is sampled cfg.runs times
    per side and results carry StabilityStats separating sampling noise from
    real prompt changes; this mode requires semantic scoring.

    With baseline_responses (case id -> saved response), side A is served
    from the baseline and never queried, so only side B's model must be
    available.
    """
    if cfg.runs > 1:
        if baseline_responses is not None:
            raise RuntimeError(
                "Stability mode (runs > 1) cannot be combined with a "
                "baseline: a baseline stores a single response per case."
            )
        return await _run_stability_diffs(
            cfg,
            on_case_completed=on_case_completed,
            on_semantic_scoring_start=on_semantic_scoring_start,
            on_semantic_scoring_complete=on_semantic_scoring_complete,
            check_models=check_models,
            cache=cache,
        )

    semaphore = asyncio.Semaphore(cfg.concurrency)
    sides = (
        (cfg.side_b,) if baseline_responses is not None else (cfg.side_a, cfg.side_b)
    )

    async with httpx.AsyncClient() as client:
        if check_models and not _all_responses_cached(cfg, cache, sides):
            await ensure_models_available(client, sides)
        responses = await _run_case_responses(
            client,
            semaphore,
            cfg,
            on_case_completed=on_case_completed,
            cache=cache,
            baseline_responses=baseline_responses,
        )

    similarities: list[float | None]
    if cfg.semantic:
        if on_semantic_scoring_start is not None:
            on_semantic_scoring_start()

        pairs = [(resp_a, resp_b) for _, resp_a, resp_b, _, _ in responses]
        loop = asyncio.get_running_loop()
        scores = await loop.run_in_executor(
            None,
            semantic_similarities,
            pairs,
            cfg.semantic_batch_size,
        )

        if len(scores) != len(responses):
            raise RuntimeError(
                "Semantic scoring returned an unexpected number of scores."
            )

        similarities = list(scores)

        if on_semantic_scoring_complete is not None:
            on_semantic_scoring_complete()
    else:
        similarities = [None] * len(responses)

    return [
        compute_diff(
            case_id=case.id,
            response_a=resp_a,
            response_b=resp_b,
            similarity=similarity,
            threshold=cfg.threshold,
            changed_when=cfg.changed_when,
            timing_a=timing_a,
            timing_b=timing_b,
            diff_mode=cfg.diff_mode,
            ignore_whitespace=cfg.ignore_whitespace,
            ignore_case=cfg.ignore_case,
        )
        for (case, resp_a, resp_b, timing_a, timing_b), similarity in zip(
            responses, similarities
        )
    ]
