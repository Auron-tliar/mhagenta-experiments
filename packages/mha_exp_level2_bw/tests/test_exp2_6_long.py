"""Check long CLI budgets and operational-failure handling without Docker."""

import json

import pytest

from mha_exp_level2_bw.exp2_6_direct import batch, runner


@pytest.mark.parametrize("valid", [True, False])
def test_four_hour_batch_budget_and_failure_boundary(tmp_path, monkeypatch, valid):
    """Four-hour runs keep learning enabled and stop after an invalid attempt."""
    calls = []
    monkeypatch.setattr(runner, "source_identity", lambda: {"fixture": "fixed"})

    def prepare(*args, **kwargs):
        profile = kwargs["profile_override"]
        assert profile == runner.Profile(60, 5000, 600, 32, 100, 100, 14400)
        assert args[5] == "cuda" and kwargs["frozen_policy"] is None
        args[2].mkdir()
        return {"run": args[3]}

    def execute(config, directory):
        calls.append(config["run"])
        return {"operationally_valid": valid, "production": {"accomplished": 0}}

    monkeypatch.setattr(runner, "prepare", prepare)
    monkeypatch.setattr(runner, "execute", execute)
    output = tmp_path / "batch"
    if valid:
        batch.run_batch([0, 1], output, duration_seconds=14400)
        assert calls == [0, 1]  # Unsuccessful goals are retained and do not block.
    else:
        with pytest.raises(RuntimeError, match="failed validation"):
            batch.run_batch([0, 1], output, duration_seconds=14400)
        assert calls == [0]
    saved = json.loads((output / "batch.json").read_text())
    assert saved["status"] == ("completed" if valid else "blocked-operational-failure")


def test_reject_more_than_four_hours_before_output(tmp_path):
    """The longer CLI allowance remains bounded before preparation begins."""
    output = tmp_path / "batch"
    with pytest.raises(ValueError, match="budget"):
        batch.run_batch([0, 1], output, duration_seconds=14401)
    assert not output.exists()
