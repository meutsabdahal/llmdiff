import asyncio
import json
import math

import httpx
import pytest
from rich.console import Console

import llmdiff.renderers.terminal as terminal
import llmdiff.runner as runner_module
from llmdiff.cache import ResponseCache
from llmdiff.config import ModelConfig, RunConfig, SideConfig, TestCase
from llmdiff.differ import compute_diff
from llmdiff.metrics import StabilityStats, compute_stability_stats, compute_summary
from llmdiff.renderers.json_ import render_json
from llmdiff.runner import (
    _side_for_sample,
    _stability_pairs,
    run_case_samples,
    run_diffs,
)


def _cfg(runs: int = 3, **overrides) -> RunConfig:
    defaults = dict(
        side_a=SideConfig(prompt="Prompt A", model_cfg=ModelConfig(model="llama3.2")),
        side_b=SideConfig(prompt="Prompt B", model_cfg=ModelConfig(model="llama3.2")),
        cases=[TestCase(id="case-1", user="hello")],
        runs=runs,
        semantic=True,
    )
    defaults.update(overrides)
    return RunConfig(**defaults)


def _stats(**overrides) -> StabilityStats:
    defaults = dict(
        runs=3,
        similarity_mean=0.8,
        similarity_std=0.05,
        ci95_low=0.68,
        ci95_high=0.92,
        self_similarity_a=0.95,
        self_similarity_b=0.96,
        beyond_noise=True,
    )
    defaults.update(overrides)
    return StabilityStats(**defaults)


# --- compute_stability_stats ---


def test_stability_stats_mean_std_and_t_interval():
    cross = [0.7, 0.8, 0.9]
    stats = compute_stability_stats(cross, [0.95], [0.96])

    assert stats.runs == 3
    assert stats.similarity_mean == pytest.approx(0.8)
    assert stats.similarity_std == pytest.approx(0.1)
    # df=2 -> t=4.303; half-width = 4.303 * 0.1 / sqrt(3)
    half = 4.303 * 0.1 / math.sqrt(3)
    assert stats.ci95_low == pytest.approx(max(0.0, 0.8 - half))
    assert stats.ci95_high == pytest.approx(min(1.0, 0.8 + half))
    assert stats.self_similarity_a == pytest.approx(0.95)
    assert stats.self_similarity_b == pytest.approx(0.96)


def test_stability_stats_ci_is_clamped_to_unit_interval():
    stats = compute_stability_stats([0.1, 0.9], [0.9], [0.9])

    assert stats.ci95_low == 0.0
    assert stats.ci95_high == 1.0


def test_stability_stats_beyond_noise_when_ci_below_noise_floor():
    # Tight cross similarity far below highly self-consistent sides.
    stats = compute_stability_stats([0.50, 0.51, 0.52, 0.49], [0.97, 0.98], [0.96])
    assert stats.beyond_noise is True

    # Sides are as noisy as the cross comparison: not distinguishable.
    stats = compute_stability_stats([0.50, 0.51, 0.52, 0.49], [0.5, 0.52], [0.51])
    assert stats.beyond_noise is False


def test_stability_stats_rejects_insufficient_samples():
    with pytest.raises(ValueError, match="at least 2 runs"):
        compute_stability_stats([0.8], [0.9], [0.9])

    with pytest.raises(ValueError, match="self-similarity"):
        compute_stability_stats([0.8, 0.9], [], [0.9])


# --- pair layout and per-sample side configs ---


def test_stability_pairs_layout():
    pairs = _stability_pairs(["a1", "a2", "a3"], ["b1", "b2", "b3"])

    assert pairs == [
        ("a1", "b1"),
        ("a2", "b2"),
        ("a3", "b3"),
        ("a1", "a2"),
        ("a1", "a3"),
        ("a2", "a3"),
        ("b1", "b2"),
        ("b1", "b3"),
        ("b2", "b3"),
    ]


def test_side_for_sample_offsets_seed_per_run():
    side = SideConfig(prompt="P", model_cfg=ModelConfig(model="llama3.2", seed=100))

    assert _side_for_sample(side, 0) is side
    assert _side_for_sample(side, 1).model_cfg.seed == 101
    assert _side_for_sample(side, 4).model_cfg.seed == 104
    # The original config is never mutated.
    assert side.model_cfg.seed == 100


def test_side_for_sample_without_seed_is_unchanged():
    side = SideConfig(prompt="P", model_cfg=ModelConfig(model="llama3.2"))
    assert _side_for_sample(side, 3) is side


# --- run_case_samples ---


class RecordingClient:
    def __init__(self):
        self.payloads = []

    async def post(self, url, json=None, timeout=None):
        self.payloads.append(json)
        request = httpx.Request("POST", url)
        return httpx.Response(
            200,
            request=request,
            json={
                "message": {
                    "content": f"sample {len(self.payloads)} of {json['model']}"
                }
            },
        )


@pytest.mark.asyncio
async def test_run_case_samples_returns_n_samples_per_side():
    cfg = _cfg(runs=3)
    client = RecordingClient()

    samples_a, samples_b, timings_a, timings_b = await run_case_samples(
        client, asyncio.Semaphore(1), cfg, cfg.cases[0]
    )

    assert len(samples_a) == 3
    assert len(samples_b) == 3
    assert len(timings_a) == 3
    assert len(timings_b) == 3
    assert len(client.payloads) == 6


@pytest.mark.asyncio
async def test_run_case_samples_sends_offset_seeds():
    side = SideConfig(
        prompt="P", model_cfg=ModelConfig(model="llama3.2", seed=100)
    )
    cfg = _cfg(runs=3, side_a=side, side_b=side.model_copy(deep=True))
    client = RecordingClient()

    await run_case_samples(client, asyncio.Semaphore(1), cfg, cfg.cases[0])

    seeds = sorted(p["options"]["seed"] for p in client.payloads)
    assert seeds == [100, 100, 101, 101, 102, 102]


@pytest.mark.asyncio
async def test_run_case_samples_caches_each_sample_separately(tmp_path):
    cache = ResponseCache(cache_dir=tmp_path / "cache")
    cfg = _cfg(runs=3)

    client = RecordingClient()
    first = await run_case_samples(
        client, asyncio.Semaphore(1), cfg, cfg.cases[0], cache=cache
    )
    assert len(client.payloads) == 6
    assert len(list((tmp_path / "cache").glob("*.json"))) == 6

    client = RecordingClient()
    second = await run_case_samples(
        client, asyncio.Semaphore(1), cfg, cfg.cases[0], cache=cache
    )
    assert client.payloads == []
    # Same samples; timing is replayed from the cache and marked as such.
    assert second[:2] == first[:2]
    assert all(t is not None and t.cached for t in second[2] + second[3])


# --- run_diffs stability path ---


@pytest.mark.asyncio
async def test_run_diffs_stability_mode_end_to_end(monkeypatch):
    cfg = _cfg(runs=2, threshold=0.9)

    async def fake_check_models_available(*_args, **_kwargs):
        return None

    scored_pairs = []

    def fake_semantic_similarities(pairs, _batch_size):
        scored_pairs.extend(pairs)
        # layout per case: 2 cross, 1 self-A, 1 self-B; the cross scores are
        # identical so the tiny-sample t-interval stays narrow and the case
        # lands beyond the noise floor.
        return [0.65, 0.65, 0.98, 0.97]

    async def fake_run_case_samples(_client, _semaphore, _cfg, _case, cache=None):
        return ["a1", "a2"], ["b1", "b2"], [None, None], [None, None]

    monkeypatch.setattr(
        runner_module, "check_models_available", fake_check_models_available
    )
    monkeypatch.setattr(
        runner_module, "semantic_similarities", fake_semantic_similarities
    )
    monkeypatch.setattr(runner_module, "run_case_samples", fake_run_case_samples)

    results = await run_diffs(cfg)

    assert scored_pairs == [
        ("a1", "b1"),
        ("a2", "b2"),
        ("a1", "a2"),
        ("b1", "b2"),
    ]
    (result,) = results
    assert result.similarity == pytest.approx(0.65)
    assert result.response_a == "a1"
    assert result.response_b == "b1"
    assert result.changed  # mean 0.65 < threshold 0.9
    st = result.stability
    assert st is not None
    assert st.runs == 2
    assert st.similarity_mean == pytest.approx(0.65)
    assert st.self_similarity_a == pytest.approx(0.98)
    assert st.self_similarity_b == pytest.approx(0.97)
    assert st.beyond_noise is True


@pytest.mark.asyncio
async def test_run_diffs_stability_mode_requires_semantic():
    cfg = _cfg(runs=2, semantic=False)

    with pytest.raises(RuntimeError, match="requires semantic scoring"):
        await run_diffs(cfg)


@pytest.mark.asyncio
async def test_run_diffs_stability_invokes_progress_callbacks(monkeypatch):
    cfg = _cfg(runs=2)

    async def fake_check_models_available(*_args, **_kwargs):
        return None

    async def fake_run_case_samples(_client, _semaphore, _cfg, _case, cache=None):
        return ["a", "a"], ["a", "a"], [None, None], [None, None]

    def fake_semantic_similarities(_pairs, _batch_size):
        return [1.0, 1.0, 1.0, 1.0]

    events = {"cases": [], "started": 0, "completed": 0}

    monkeypatch.setattr(
        runner_module, "check_models_available", fake_check_models_available
    )
    monkeypatch.setattr(
        runner_module, "semantic_similarities", fake_semantic_similarities
    )
    monkeypatch.setattr(runner_module, "run_case_samples", fake_run_case_samples)

    await run_diffs(
        cfg,
        on_case_completed=lambda case: events["cases"].append(case.id),
        on_semantic_scoring_start=lambda: events.__setitem__(
            "started", events["started"] + 1
        ),
        on_semantic_scoring_complete=lambda: events.__setitem__(
            "completed", events["completed"] + 1
        ),
    )

    assert events == {"cases": ["case-1"], "started": 1, "completed": 1}


@pytest.mark.asyncio
async def test_run_diffs_stability_skips_model_check_when_fully_cached(
    tmp_path, monkeypatch
):
    cache = ResponseCache(cache_dir=tmp_path / "cache")
    cfg = _cfg(runs=2)
    messages = runner_module._case_messages(cfg.cases[0])
    for side in (cfg.side_a, cfg.side_b):
        for sample in range(cfg.runs):
            cache.set(side, messages, f"sample {sample}", sample=sample)

    check_calls = []

    async def fake_check_models_available(_client, endpoint, _models):
        check_calls.append(endpoint)

    def fake_semantic_similarities(pairs, _batch_size):
        return [1.0] * len(pairs)

    monkeypatch.setattr(
        runner_module, "check_models_available", fake_check_models_available
    )
    monkeypatch.setattr(
        runner_module, "semantic_similarities", fake_semantic_similarities
    )

    results = await run_diffs(cfg, cache=cache)

    assert check_calls == []
    assert results[0].stability is not None


# --- summary and renderers ---


def _diff_with_stability(case_id: str, stats: StabilityStats):
    return compute_diff(
        case_id=case_id,
        response_a="a",
        response_b="b",
        similarity=stats.similarity_mean,
        threshold=None,
        stability=stats,
    )


def test_compute_summary_counts_beyond_noise_cases():
    results = [
        _diff_with_stability("c1", _stats(beyond_noise=True)),
        _diff_with_stability("c2", _stats(beyond_noise=False)),
        _diff_with_stability("c3", _stats(beyond_noise=True)),
    ]

    assert compute_summary(results).beyond_noise == 2


def test_compute_summary_beyond_noise_is_none_without_stability():
    result = compute_diff(
        case_id="c1",
        response_a="a",
        response_b="a",
        similarity=1.0,
        threshold=None,
    )

    assert compute_summary([result]).beyond_noise is None


def test_render_json_includes_stability():
    results = [_diff_with_stability("c1", _stats())]
    payload = json.loads(render_json(results, compute_summary(results)))

    assert payload["summary"]["beyond_noise_count"] == 1
    stability = payload["cases"][0]["stability"]
    assert stability == {
        "runs": 3,
        "similarity_mean": 0.8,
        "similarity_std": 0.05,
        "ci95": [0.68, 0.92],
        "self_similarity_a": 0.95,
        "self_similarity_b": 0.96,
        "beyond_noise": True,
    }


def test_render_json_stability_is_null_without_stability_mode():
    result = compute_diff(
        case_id="c1",
        response_a="a",
        response_b="a",
        similarity=1.0,
        threshold=None,
    )
    payload = json.loads(render_json([result], compute_summary([result])))

    assert payload["cases"][0]["stability"] is None
    assert payload["summary"]["beyond_noise_count"] is None


def test_terminal_renderer_shows_stability_line(monkeypatch):
    fake_console = Console(record=True, width=200)
    monkeypatch.setattr(terminal, "console", fake_console)

    terminal.render_case_inline(_diff_with_stability("c1", _stats()))
    out = fake_console.export_text()

    assert "Stability (3 runs): 0.80 ± 0.05" in out
    assert "95% CI 0.68–0.92" in out
    assert "self-similarity A 0.95 / B 0.96" in out
    assert "beyond sampling noise" in out


def test_terminal_summary_shows_beyond_noise_count(monkeypatch):
    fake_console = Console(record=True, width=200)
    monkeypatch.setattr(terminal, "console", fake_console)

    results = [_diff_with_stability("c1", _stats(beyond_noise=True))]
    terminal.render_summary(compute_summary(results))
    out = fake_console.export_text()

    assert "Beyond noise:" in out
