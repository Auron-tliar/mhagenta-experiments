"""Checks for diamond outcomes and run-level achievement aggregation."""

import json
from pathlib import Path

import pytest

from mha_exp_level2_cr.exp2_1.reporting import process_execution_metrics
from mha_exp_level2_cr.exp2_1.treatment import treatment_for_run


def _write_run(root: Path, run: int, *, diamond: bool) -> None:
    """Write a complete run with repeated wood events and an optional diamond."""
    agent_id, env_id = f"exp_agent2_1_{run}", f"exp_env2_1_{run}"
    agent_out, env_out = root / agent_id / "out", root / env_id / "out"
    agent_out.mkdir(parents=True)
    env_out.mkdir(parents=True)
    highest = "collect_diamond" if diamond else "collect_wood"
    termination = "target_achieved" if diamond else "action_budget_exhausted"
    state = {
        **treatment_for_run(run), "phase": "complete", "episode_id": 0,
        "terminal_reason": termination, "target_achieved": diamond,
        "observations": 2, "action_requests": 2, "action_statuses": 2,
        "last_decision": {"action": "do", "reason": "collect"},
        "decision_trace": [{"request_id": i, "episode_id": 0, "reason": "collect"}
                           for i in range(2)],
    }
    states = {"llreasoner_0": state, "perceptor_0": {"requests": 2, "observations": 2},
              "actuator_0": {"requests": 2, "statuses": 2}}
    for module, value in states.items():
        (agent_out / f"{agent_id}.{module}.json").write_text(json.dumps(value))
    trace = [{
        "event_id": i, "request_id": i, "episode_id": 0, "event_kind": "native_action",
        "action": "do", "elapsed_seconds": i + 1, "player_pos": [0, 0],
        "inventory": {"health": 9, "food": 9, "drink": 9, "energy": 9},
        "environment_done": False, "newly_achieved": ["collect_wood"] + (
            ["collect_diamond"] if diamond and i == 1 else ["collect_drink"]
        ),
    } for i in range(2)]
    (env_out / f"{env_id}.json").write_text(json.dumps({
        "execution_trace": trace, "native_actions": 2, "illegal_actions": 0,
        "target_achieved": diamond, "highest_diamond_path_achievement": highest,
    }))
    (root / f"{agent_id}.log").write_text("[INFO] Agent finished\n")
    (root / f"{env_id}.log").write_text("[INFO] Environment closed\n")


def test_diamond_statistics_count_runs_not_repeated_achievement_events(tmp_path: Path) -> None:
    """Partial progress is a failed goal; missing runs do not inflate the denominator."""
    _write_run(tmp_path, 0, diamond=True)
    _write_run(tmp_path, 1, diamond=False)
    report = process_execution_metrics(tmp_path, expected_executions=[
        {"execution_id": f"run-{run}", "run_id": run, "factors": treatment_for_run(run)}
        for run in range(3)
    ])
    analysis = report["analyses"]["controller_runs"]
    summary = analysis["summary"]
    assert summary["runs"] == 2
    assert summary["successes"] == 1 and summary["failures"] == 1
    assert summary["success_rate"] == 0.5
    assert summary["achievement_attainment"]["collect_wood"] == {
        "runs": 2, "rate": 1.0, "total_events": 4,
    }
    assert summary["achievement_attainment"]["collect_drink"]["runs"] == 2
    assert summary["achievement_attainment"]["collect_diamond"]["runs"] == 1
    assert summary["achievement_attainment"]["collect_drink"]["total_events"] == 3
    assert summary["achievement_events_per_run"]["median"] == 4
    assert summary["distinct_achievements_per_run"]["mean"] == 2.5
    assert summary["highest_diamond_path_achievement"]["collect_wood"]["runs"] == 1
    assert summary["highest_diamond_path_achievement"]["collect_diamond"]["runs"] == 1
    assert summary["highest_diamond_path_achievement"]["collect_stone"]["runs"] == 0
    assert summary["termination_counts"] == {"target_achieved": 1, "action_budget_exhausted": 1}
    assert analysis["eligibility"]["excluded"][0]["execution_id"] == "run-2"
    assert report["executions"][1]["certificate_passed"] is True
    assert report["executions"][1]["task_outcome"]["status"] == "failure"
    assert all(row["censoring"]["status"] == "observed"
               for row in report["analyses"]["episodes"]["rows"])
    assert report["executions"][0]["termination"]["reason"] == "target_achieved"


def test_missing_runs_do_not_become_zero_success_observations(tmp_path: Path) -> None:
    """An entirely missing batch has no observed success rate."""
    report = process_execution_metrics(tmp_path, expected_executions=[
        {"execution_id": "run-0", "run_id": 0, "factors": treatment_for_run(0)},
    ])
    summary = report["analyses"]["controller_runs"]["summary"]
    assert summary["runs"] == 0 and summary["success_rate"] is None


def test_negative_run_id_is_rejected() -> None:
    """The uniform treatment retains the nonnegative seed-index contract."""
    with pytest.raises(ValueError, match="nonnegative"):
        treatment_for_run(-1)
