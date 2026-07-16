from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

from llmdiff.config import ChangedWhen

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover - exercised only on Python 3.10
    import tomli as tomllib

DEFAULT_CONFIG_FILENAME = "llmdiff.toml"

_RATIO_KEYS = ("threshold", "fail_if_avg_below", "fail_if_any_below_threshold")
_KNOWN_KEYS = frozenset(
    (*_RATIO_KEYS, "changed_when", "fail_on_changed")
)


class PolicyConfigError(Exception):
    """A regression-policy config file is missing, unreadable, or invalid."""


@dataclass
class RegressionPolicy:
    """Thresholds and failure rules loaded from a project config file.

    Every field is optional: None means "not set here", so CLI flags and
    built-in defaults can fill the gap. source records which file the policy
    came from (None when no config file was found).
    """

    threshold: float | None = None
    changed_when: ChangedWhen | None = None
    fail_on_changed: bool | None = None
    fail_if_avg_below: float | None = None
    fail_if_any_below_threshold: float | None = None
    source: Path | None = None


def _parse_policy_table(table: object, path: Path) -> RegressionPolicy:
    if not isinstance(table, dict):
        raise PolicyConfigError(f"[policy] in {path} must be a TOML table.")

    unknown = sorted(set(table) - _KNOWN_KEYS)
    if unknown:
        raise PolicyConfigError(
            f"unknown key(s) in [policy] of {path}: {', '.join(unknown)}. "
            f"Valid keys: {', '.join(sorted(_KNOWN_KEYS))}."
        )

    policy = RegressionPolicy(source=path)

    for key in _RATIO_KEYS:
        if key not in table:
            continue
        value = table[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise PolicyConfigError(
                f"[policy] {key} in {path} must be a number, "
                f"got {value!r}."
            )
        if not (0.0 <= value <= 1.0):
            raise PolicyConfigError(
                f"[policy] {key} in {path} must be between 0.0 and 1.0, "
                f"got {value}."
            )
        setattr(policy, key, float(value))

    if "fail_on_changed" in table:
        value = table["fail_on_changed"]
        if not isinstance(value, bool):
            raise PolicyConfigError(
                f"[policy] fail_on_changed in {path} must be true or false, "
                f"got {value!r}."
            )
        policy.fail_on_changed = value

    if "changed_when" in table:
        value = table["changed_when"]
        valid = ", ".join(member.value for member in ChangedWhen)
        if not isinstance(value, str) or value not in {
            member.value for member in ChangedWhen
        }:
            raise PolicyConfigError(
                f"[policy] changed_when in {path} must be one of: {valid}; "
                f"got {value!r}."
            )
        policy.changed_when = ChangedWhen(value)

    return policy


def load_regression_policy(config_path: Path | None) -> RegressionPolicy:
    """Load the regression policy from a config file.

    With an explicit path, the file must exist. Otherwise looks for
    llmdiff.toml in the working directory and returns an empty policy when
    absent. Raises PolicyConfigError for unreadable or invalid files.
    """
    if config_path is not None:
        if not config_path.is_file():
            raise PolicyConfigError(f"config file not found: {config_path}")
        path = config_path
    else:
        path = Path(DEFAULT_CONFIG_FILENAME)
        if not path.is_file():
            return RegressionPolicy()

    try:
        raw = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        raise PolicyConfigError(f"config file is not valid UTF-8: {path}")
    except OSError as e:
        raise PolicyConfigError(f"failed to read config file {path}: {e}")

    try:
        data = tomllib.loads(raw)
    except tomllib.TOMLDecodeError as e:
        raise PolicyConfigError(f"invalid TOML in {path}: {e}")

    if "policy" not in data:
        return RegressionPolicy(source=path)

    return _parse_policy_table(data["policy"], path)
