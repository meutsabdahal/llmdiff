import pytest

from llmdiff.config import ChangedWhen
from llmdiff.policy import (
    PolicyConfigError,
    RegressionPolicy,
    load_regression_policy,
)


def test_returns_empty_policy_when_no_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    policy = load_regression_policy(None)

    assert policy == RegressionPolicy()
    assert policy.source is None


def test_discovers_llmdiff_toml_in_working_directory(tmp_path, monkeypatch):
    (tmp_path / "llmdiff.toml").write_text(
        '[policy]\nthreshold = 0.75\nchanged_when = "semantic"\n',
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    policy = load_regression_policy(None)

    assert policy.threshold == 0.75
    assert policy.changed_when == ChangedWhen.SEMANTIC
    assert policy.source is not None


def test_parses_all_policy_keys(tmp_path):
    path = tmp_path / "custom.toml"
    path.write_text(
        "[policy]\n"
        "threshold = 0.7\n"
        'changed_when = "lines"\n'
        "fail_on_changed = true\n"
        "fail_if_avg_below = 0.8\n"
        "fail_if_any_below_threshold = 0.6\n",
        encoding="utf-8",
    )

    policy = load_regression_policy(path)

    assert policy.threshold == 0.7
    assert policy.changed_when == ChangedWhen.LINES
    assert policy.fail_on_changed is True
    assert policy.fail_if_avg_below == 0.8
    assert policy.fail_if_any_below_threshold == 0.6
    assert policy.source == path


def test_explicit_config_path_must_exist(tmp_path):
    with pytest.raises(PolicyConfigError, match="config file not found"):
        load_regression_policy(tmp_path / "missing.toml")


def test_file_without_policy_section_is_empty_policy(tmp_path):
    path = tmp_path / "llmdiff.toml"
    path.write_text("# future sections may live here\n", encoding="utf-8")

    policy = load_regression_policy(path)

    assert policy.threshold is None
    assert policy.fail_on_changed is None
    assert policy.source == path


def test_invalid_toml_is_reported(tmp_path):
    path = tmp_path / "llmdiff.toml"
    path.write_text("[policy\nthreshold = 0.7\n", encoding="utf-8")

    with pytest.raises(PolicyConfigError, match="invalid TOML"):
        load_regression_policy(path)


def test_unknown_policy_key_is_rejected(tmp_path):
    path = tmp_path / "llmdiff.toml"
    path.write_text("[policy]\nfail_on_change = true\n", encoding="utf-8")

    with pytest.raises(PolicyConfigError, match="fail_on_change") as excinfo:
        load_regression_policy(path)

    # The error teaches the valid vocabulary.
    assert "fail_on_changed" in str(excinfo.value)


@pytest.mark.parametrize(
    "line",
    [
        'threshold = "high"',
        "threshold = true",
        "threshold = 1.5",
        "fail_if_avg_below = -0.1",
        "fail_on_changed = 1",
        'changed_when = "sometimes"',
        "changed_when = 3",
    ],
)
def test_invalid_policy_values_are_rejected(tmp_path, line):
    path = tmp_path / "llmdiff.toml"
    path.write_text(f"[policy]\n{line}\n", encoding="utf-8")

    with pytest.raises(PolicyConfigError):
        load_regression_policy(path)


def test_policy_must_be_a_table(tmp_path):
    path = tmp_path / "llmdiff.toml"
    path.write_text('policy = "strict"\n', encoding="utf-8")

    with pytest.raises(PolicyConfigError, match="must be a TOML table"):
        load_regression_policy(path)
