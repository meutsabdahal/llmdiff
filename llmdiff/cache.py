from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path

from llmdiff.config import SideConfig
from llmdiff.metrics import SideTiming

# Bump when the key layout or entry format changes so stale entries from
# older llmdiff versions are never returned.
_CACHE_SCHEMA_VERSION = 1


def default_cache_dir() -> Path:
    xdg_cache_home = os.environ.get("XDG_CACHE_HOME", "").strip()
    base = Path(xdg_cache_home) if xdg_cache_home else Path.home() / ".cache"
    return base / "llmdiff"


class ResponseCache:
    """File-backed cache of model responses.

    Each entry is a JSON file named by the SHA-256 of the full request
    identity: system prompt, model, endpoint, sampling parameters, and case
    messages. The endpoint is part of the key because two Ollama servers can
    serve different weights under the same model name (the --base-url-a /
    --base-url-b comparison case).

    Cache failures never fail a run: unreadable or corrupt entries behave as
    misses, and failed writes are skipped.
    """

    def __init__(self, cache_dir: Path | None = None):
        self.cache_dir = cache_dir if cache_dir is not None else default_cache_dir()

    @staticmethod
    def _key(
        side: SideConfig,
        messages: list[dict[str, str]],
        sample: int = 0,
    ) -> str:
        identity = {
            "version": _CACHE_SCHEMA_VERSION,
            "prompt": side.prompt,
            "model": side.model_cfg.model,
            "base_url": side.model_cfg.base_url,
            "temperature": side.model_cfg.temperature,
            "max_tokens": side.model_cfg.max_tokens,
            "seed": side.model_cfg.seed,
            "messages": messages,
        }
        # Stability mode stores each repeated sample under its own key so a
        # cached re-run reproduces N distinct samples instead of one response
        # repeated N times. Sample 0 keeps the single-run key, so a plain run
        # and the first stability sample share a cache entry.
        if sample > 0:
            identity["sample"] = sample
        canonical = json.dumps(
            identity,
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def _entry_path(
        self,
        side: SideConfig,
        messages: list[dict[str, str]],
        sample: int = 0,
    ) -> Path:
        return self.cache_dir / f"{self._key(side, messages, sample)}.json"

    def get(
        self,
        side: SideConfig,
        messages: list[dict[str, str]],
        sample: int = 0,
    ) -> str | None:
        hit = self.get_with_timing(side, messages, sample)
        return hit[0] if hit is not None else None

    def get_with_timing(
        self,
        side: SideConfig,
        messages: list[dict[str, str]],
        sample: int = 0,
    ) -> tuple[str, SideTiming | None] | None:
        """Cache hit as (response, timing); timing is None for entries
        written before timing was recorded. Replayed timing is marked cached.
        """
        try:
            raw = self._entry_path(side, messages, sample).read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            return None

        try:
            entry = json.loads(raw)
        except json.JSONDecodeError:
            return None

        if not isinstance(entry, dict):
            return None

        response = entry.get("response")
        if not isinstance(response, str):
            return None

        return response, self._timing_from_entry(entry)

    @staticmethod
    def _timing_from_entry(entry: dict) -> SideTiming | None:
        timing = entry.get("timing")
        if not isinstance(timing, dict):
            return None

        latency = timing.get("latency_s")
        if not isinstance(latency, (int, float)) or latency < 0:
            return None

        tokens = timing.get("tokens")
        tokens_per_s = timing.get("tokens_per_s")
        return SideTiming(
            latency_s=float(latency),
            tokens=tokens if isinstance(tokens, int) else None,
            tokens_per_s=(
                float(tokens_per_s)
                if isinstance(tokens_per_s, (int, float))
                else None
            ),
            cached=True,
        )

    def set(
        self,
        side: SideConfig,
        messages: list[dict[str, str]],
        response: str,
        sample: int = 0,
        timing: SideTiming | None = None,
    ) -> None:
        entry: dict = {
            "model": side.model_cfg.model,
            "response": response,
        }
        if timing is not None:
            entry["timing"] = {
                "latency_s": timing.latency_s,
                "tokens": timing.tokens,
                "tokens_per_s": timing.tokens_per_s,
            }
        path = self._entry_path(side, messages, sample)

        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            # Write-then-rename keeps concurrent readers (and a second llmdiff
            # process writing the same key) from ever seeing a partial entry.
            fd, tmp_name = tempfile.mkstemp(dir=self.cache_dir, suffix=".tmp")
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    json.dump(entry, f, ensure_ascii=False)
                os.replace(tmp_name, path)
            except OSError:
                Path(tmp_name).unlink(missing_ok=True)
        except OSError:
            return
