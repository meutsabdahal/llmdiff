from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path

from llmdiff.config import SideConfig

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
    def _key(side: SideConfig, messages: list[dict[str, str]]) -> str:
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
        canonical = json.dumps(
            identity,
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def _entry_path(self, side: SideConfig, messages: list[dict[str, str]]) -> Path:
        return self.cache_dir / f"{self._key(side, messages)}.json"

    def get(self, side: SideConfig, messages: list[dict[str, str]]) -> str | None:
        try:
            raw = self._entry_path(side, messages).read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            return None

        try:
            entry = json.loads(raw)
        except json.JSONDecodeError:
            return None

        if not isinstance(entry, dict):
            return None

        response = entry.get("response")
        return response if isinstance(response, str) else None

    def set(
        self,
        side: SideConfig,
        messages: list[dict[str, str]],
        response: str,
    ) -> None:
        entry = {
            "model": side.model_cfg.model,
            "response": response,
        }
        path = self._entry_path(side, messages)

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
