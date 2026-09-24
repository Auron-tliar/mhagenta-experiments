"""Cross-experiment contract checks for additive Level 2 CR reports."""

from __future__ import annotations

import hashlib
import json
from decimal import Decimal
from importlib import import_module
from pathlib import Path

import pytest
from mha_exp_level2_cr.exp2_7.evaluate import _usage_rows

PROCESSOR_MODULES = (
    "mha_exp_level2_cr.exp2_1.reporting",
    "mha_exp_level2_cr.exp2_2.reporting",
    "mha_exp_level2_cr.exp2_3.reporting",
    "mha_exp_level2_cr.exp2_4.reporting",
    "mha_exp_level2_cr.exp2_7.evaluate",
)


@pytest.mark.parametrize("module_name", PROCESSOR_MODULES)
def test_processor_accounts_for_expected_missing_execution(
    tmp_path: Path,
    module_name: str,
) -> None:
    """Every adapter must retain an expected cell even when no run exists."""

    processor = import_module(module_name).process_execution_metrics
    factors = (
        {"profile": "standard", "observation_format": "symbolic", "value_condition": "with"}
        if module_name.endswith("exp2_7.evaluate")
        else {}
    )
    expected = [{"execution_id": "run-7", "run_id": 7, "factors": factors}]

    report = processor(tmp_path, expected_executions=expected)
    first_json = (tmp_path / "execution-metrics.json").read_text(encoding="utf-8")
    repeated = processor(tmp_path, expected_executions=expected)

    assert report == repeated
    assert first_json == (tmp_path / "execution-metrics.json").read_text(
        encoding="utf-8"
    )
    assert report["execution_accounting"] == {
        "expected": 1,
        "present": 0,
        "readable": 0,
        "operationally_valid": 0,
    }
    assert report["executions"][0]["task_outcome"]["status"] == "unobserved"
    assert "run_missing" in (
        report["executions"][0]["readability_reasons"]
        + report["executions"][0]["operational_reasons"]
    )
    for analysis in report["analyses"].values():
        eligibility = analysis["eligibility"]
        if "eligible_execution_ids" not in eligibility:
            continue
        assert eligibility["eligible_execution_ids"] == []
        reasons = ["run_missing"]
        if module_name.endswith("exp2_4.reporting"):
            reasons.append("required_provenance_missing")
        assert eligibility["excluded"] == [{
            "execution_id": "run-7", "reasons": reasons,
        }]
    if module_name.endswith("exp2_3.reporting"):
        outcome = report["analyses"]["execution_outcomes"]["rows"][0]
        for field in (
            "minimum_needs",
            "actions_at_or_below_intervention",
            "cumulative_reward",
            "achievement_counts",
            "environment_counters",
        ):
            assert outcome[field] is None
        assert report["analyses"]["stage_activity"]["rows"] == []
    assert json.loads(first_json) == report


@pytest.mark.parametrize("module_name", PROCESSOR_MODULES)
def test_ad_hoc_empty_directory_does_not_claim_completeness(
    tmp_path: Path,
    module_name: str,
) -> None:
    """Discovery mode must distinguish an empty cohort from a complete one."""

    report = import_module(module_name).process_execution_metrics(tmp_path)

    assert report["expected_executions"] is None
    assert report["completeness_assessable"] is False
    assert report["execution_accounting"]["expected"] is None
    assert report["executions"] == []


def test_cumulative_usage_deltas_do_not_cross_failed_calls() -> None:
    """A failed call makes the following successful delta unattributable."""

    events = [
        {"module_id": "llreasoner_0", "kind": "llm_call", "outcome": "success",
         "input_tokens": 10, "output_tokens": 2, "estimated_cost_usd": "0.10"},
        {"module_id": "llreasoner_0", "kind": "llm_call", "outcome": "success",
         "input_tokens": 15, "output_tokens": 5, "estimated_cost_usd": "0.20"},
        {"module_id": "llreasoner_0", "kind": "llm_call", "outcome": "failure"},
        {"module_id": "llreasoner_0", "kind": "llm_call", "outcome": "success",
         "input_tokens": 20, "output_tokens": 8, "estimated_cost_usd": "0.30"},
        {"module_id": "llreasoner_0", "kind": "llm_call", "outcome": "success",
         "input_tokens": 25, "output_tokens": 10, "estimated_cost_usd": "0.40"},
    ]

    calls, totals = _usage_rows(
        events,
        {"llreasoner_0": {"calls": 5, "failures": 1,
                           "estimated_cost_usd": Decimal("0.40")}},
        "run-1",
        1,
    )
    rows = [row for row in calls if row["module_id"] == "llreasoner_0"]

    assert rows[0]["per_call_usage"]["input_tokens"] == 10
    assert rows[1]["per_call_usage"]["input_tokens"] == 5
    assert rows[2]["per_call_usage"] is None
    assert rows[3]["per_call_usage"] is None
    assert rows[4]["per_call_usage"]["input_tokens"] == 5
    assert next(row for row in totals if row["module_id"] == "llreasoner_0")[
        "cumulative_input_tokens"
    ] == 25


def test_exp2_7_joins_supported_matched_context_and_validates_identity(
    tmp_path: Path,
) -> None:
    """Evaluation cells must agree with their expected treatment identity."""

    context = {
        "schema_version": "2-7-cr-matched-context-v9",
        "profile": {"name": "standard"},
        "model_policy_version": "models-v1",
        "goal_chain_policy_version": "goals-v1",
        "knowledge_admission_policy_version": "knowledge-v1",
        "control_treatment": "control-v1",
        "designs": [["symbolic", "with"]],
        "runs": [{"run": 7}],
    }
    (tmp_path / "matched-context-standard.json").write_text(
        json.dumps(context), encoding="utf-8"
    )
    run_root = tmp_path / "standard-symbolic-with" / "run-007"
    run_root.mkdir(parents=True)
    evaluation = {
        "schema_version": "2-7-cr-evaluation-v10",
        "run": 7,
        "profile": {"name": "standard"},
        "observation_format": "symbolic",
        "value_condition": "with",
        "architecture_reasons": [],
        "scientific_outcomes": {"primary_goal_achieved": True, "alive": True},
    }
    (run_root / "evaluation.json").write_text(
        json.dumps(evaluation), encoding="utf-8"
    )
    expected = [{
        "execution_id": "profile-standard/run-7/observation-symbolic/value-with",
        "run_id": 7,
        "factors": {
            "profile": "standard",
            "observation_format": "symbolic",
            "value_condition": "with",
        },
    }]

    report = import_module(
        "mha_exp_level2_cr.exp2_7.evaluate"
    ).process_execution_metrics(tmp_path, expected_executions=expected)

    assert report["executions"][0]["operationally_valid"] is True
    assert report["executions"][0]["metric_availability"]["matched_context"] == "existing"
    assert report["executions"][0]["metric_availability"]["usage"] == {
        "status": "unavailable",
        "reason": "incomplete_final_module_usage",
    }
    assert report["analyses"]["module_usage"]["eligibility"][
        "eligible_execution_ids"
    ] == []
    assert report["analyses"]["matched_contexts"]["eligibility"] == {
        "eligible_profile_ids": ["standard"], "excluded": [],
    }
    assert report["analyses"]["matched_contexts"]["summary"] == {
        "raw_descriptive_count": 1,
        "supported_raw_descriptive_count": 1,
    }


def test_exp2_3_reports_full_domain_trace_without_inventing_events(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The canonical trace supplies all full-domain analyses and censoring."""

    agent_out = tmp_path / "exp_agent2_3_3" / "out"
    agent_out.mkdir(parents=True)
    trace = [
        {
            "revision": 1,
            "elapsed_seconds": 1.0,
            "belief": {
                "achievements": [],
                "needs": {"health": 9, "food": 9, "drink": 9, "energy": 9},
                "known_terrain": 63,
                "reachable_cells": 20,
                "unknown_frontier": 18,
            },
            "stage": {
                "derived": "collect-table-wood",
                "support": {"supported": True},
            },
            "intentions": [{
                "kind": "selected",
                "id": "intention-1",
                "name": "collect-table-wood",
            }],
            "planning": {
                "intention_id": "intention-1",
                "stage": "collect-table-wood",
                "classification": "accepted",
                "elapsed_seconds": 0.2,
                "plan_length": 1,
                "validation_status": "VALID",
                "accepted": True,
                "rejection": None,
                "plan_id": "plan-1",
                "actions": [{"name": "do-tree"}],
            },
            "monitoring": {},
            "decision": {
                "kind": "planned-action",
                "action_id": "action-1",
                "intention_id": "intention-1",
                "plan_id": "plan-1",
            },
        },
        {
            "revision": 2,
            "elapsed_seconds": 2.0,
            "belief": {
                "achievements": ["collect_wood"],
                "needs": {"health": 9, "food": 9, "drink": 9, "energy": 9},
                "known_terrain": 63,
                "reachable_cells": 20,
                "unknown_frontier": 18,
            },
            "stage": {
                "derived": "collect-table-wood",
                "support": {"supported": False},
            },
            "intentions": [{
                "kind": "retained",
                "id": "intention-1",
                "name": "collect-table-wood",
            }],
            "planning": {
                "intention_id": "intention-1",
                "stage": "collect-table-wood",
                "classification": "missing-knowledge",
                "accepted": False,
                "rejection": "missing-target",
            },
            "monitoring": {
                "action_result": {"action_id": "action-1", "reward": 1.0},
                "plan_goal": {
                    "plan_id": "plan-1",
                    "status": "plan-goal-complete",
                },
            },
            "decision": {
                "kind": "explore-action",
                "action_id": "action-2",
                "intention_id": "intention-1",
                "reason": "missing-knowledge",
            },
        },
        {
            "revision": 3,
            "elapsed_seconds": 3.0,
            "belief": {
                "achievements": ["collect_wood"],
                "needs": {"health": 3, "food": 4, "drink": 4, "energy": 3},
                "known_terrain": 70,
                "reachable_cells": 25,
                "unknown_frontier": 16,
            },
            "stage": {
                "derived": "collect-table-wood",
                "support": {"supported": True},
            },
            "intentions": [
                {
                    "kind": "preempted",
                    "id": "intention-1",
                    "name": "collect-table-wood",
                },
                {
                    "kind": "selected",
                    "id": "intention-2",
                    "name": "restore-food",
                },
            ],
            "planning": None,
            "monitoring": {
                "action_result": {"action_id": "action-2", "reward": 0.0},
            },
            "decision": {
                "kind": "recovery-action",
                "action_id": "action-3",
                "intention_id": "intention-2",
            },
        },
        {
            "revision": 4,
            "elapsed_seconds": 4.0,
            "belief": {
                "achievements": ["collect_wood", "place_table"],
                "needs": {"health": 3, "food": 5, "drink": 5, "energy": 4},
                "known_terrain": 70,
                "reachable_cells": 25,
                "unknown_frontier": 16,
            },
            "stage": {
                "derived": "place-table",
                "support": {"supported": True},
            },
            "intentions": [
                {
                    "kind": "completed",
                    "id": "intention-2",
                    "name": "restore-food",
                },
                {
                    "kind": "selected",
                    "id": "intention-3",
                    "name": "place-table",
                },
            ],
            "planning": {
                "intention_id": "intention-3",
                "stage": "place-table",
                "classification": "accepted",
                "elapsed_seconds": 0.3,
                "plan_length": 1,
                "validation_status": "VALID",
                "accepted": True,
                "rejection": None,
                "plan_id": "plan-2",
                "actions": [{"name": "place-table"}],
            },
            "monitoring": {
                "action_result": {"action_id": "action-3", "reward": 0.5},
            },
            "decision": {
                "kind": "planned-action",
                "action_id": "action-4",
                "intention_id": "intention-3",
                "plan_id": "plan-2",
            },
        },
        {
            "revision": 5,
            "elapsed_seconds": 6.0,
            "belief": {
                "achievements": ["collect_wood", "place_table"],
                "needs": {"health": 4, "food": 5, "drink": 5, "energy": 4},
                "known_terrain": 70,
                "reachable_cells": 25,
                "unknown_frontier": 16,
            },
            "stage": {
                "derived": "place-table",
                "support": {"supported": True},
            },
            "intentions": [],
            "planning": None,
            "monitoring": {
                "action_result": {"action_id": "action-4", "reward": 2.0},
            },
            "decision": {
                "kind": "scientific-terminal",
                "reason": "action_budget_exhausted",
            },
        },
    ]
    state = {
        "run": {
            "phase": "scientific-terminal",
            "terminal_reason": "action_budget_exhausted",
            "agent_steps": 4,
            "elapsed_seconds": 6.0,
            "highest_milestone": "place_table",
            "treatment": {"initial_support": "wood-supported"},
            "trace": trace,
        }
    }
    (agent_out / "exp_agent2_3_3.hlreasoner_0.json").write_text(
        json.dumps(state), encoding="utf-8"
    )
    for module in ("perceptor_0", "actuator_0", "llreasoner_0", "knowledge_0"):
        (agent_out / f"exp_agent2_3_3.{module}.json").write_text(
            json.dumps({"failure": None}), encoding="utf-8"
        )
    environment_out = tmp_path / "exp_env2_3_3" / "out"
    environment_out.mkdir(parents=True)
    (environment_out / "exp_env2_3_3.environment.json").write_text(
        json.dumps({
            "observation_requests": 5,
            "native_actions": 4,
            "illegal_actions": 0,
            "post_terminal_observations": 0,
            "close_requests": 1,
            "closed": True,
            "treatment": {"initial_support": "wood-supported"},
            "achievement_counts": {
                "collect_wood": 1,
                "place_table": 1,
                "collect_diamond": 0,
            },
        }),
        encoding="utf-8",
    )
    (tmp_path / "exp_agent2_3_3.log").write_text(
        "[INFO] retained fixture", encoding="utf-8"
    )
    runner = import_module("mha_exp_level2_cr.exp2_3.runner")
    monkeypatch.setattr(runner, "_result_errors", lambda *_args: [])

    report = import_module(
        "mha_exp_level2_cr.exp2_3.reporting"
    ).process_execution_metrics(
        tmp_path,
        expected_executions=[{
            "execution_id": "run-3", "run_id": 3, "factors": {},
        }],
    )

    outcome = report["analyses"]["execution_outcomes"]["rows"][0]
    assert outcome["scientifically_eligible"] is True
    assert outcome["highest_milestone"] == "place_table"
    assert outcome["minimum_needs"] == {
        "health": 3,
        "food": 4,
        "drink": 4,
        "energy": 3,
    }
    assert outcome["actions_at_or_below_intervention"] == {
        "health": 2,
        "food": 1,
        "drink": 1,
        "energy": 1,
    }
    assert outcome["cumulative_reward"] == 3.5
    assert outcome["achievement_counts"] == {
        "collect_diamond": 0,
        "collect_wood": 1,
        "place_table": 1,
    }
    assert outcome["environment_counters"] == {
        "observation_requests": 5,
        "native_actions": 4,
        "illegal_actions": 0,
        "post_terminal_observations": 0,
        "close_requests": 1,
        "closed": True,
    }
    milestone_rows = report["analyses"]["milestone_outcomes"]["rows"]
    assert len(milestone_rows) == 10
    assert milestone_rows[0]["attained"] is True
    assert milestone_rows[0]["first_action_count"] == 1
    assert milestone_rows[-1]["censoring"]["status"] == "right_censored"
    planning_rows = report["analyses"]["planning_attempts"]["rows"]
    assert [
        (
            row["intention_id"],
            row["attempt_for_intention"],
            row["is_replan"],
        )
        for row in planning_rows
    ] == [
        ("intention-1", 1, False),
        ("intention-1", 2, True),
        ("intention-3", 1, False),
    ]
    assert report["analyses"]["planning_attempts"]["summary"][
        "replanning_attempts"
    ] == 1
    assert report["analyses"]["plan_chunks"]["rows"][0]["outcome"] == (
        "plan-goal-complete"
    )
    recovery = report["analyses"]["recovery_episodes"]["rows"][0]
    assert recovery["need"] == "food"
    assert recovery["completed"] is True
    exploration = report["analyses"]["exploration_episodes"]["rows"][0]
    assert exploration["newly_known_cells"] == 7
    aggregate = report["analyses"]["initial_support_aggregates"]["rows"][0]
    assert aggregate["initial_support"] == "wood-supported"
    assert aggregate["planning_accepted"] == 2
    assert aggregate["replanning_attempts"] == 1
    assert aggregate["minimum_needs"]["food"]["min"] == 4.0
    assert aggregate["actions_at_or_below_intervention"]["health"][
        "median"
    ] == 2.0
    assert aggregate["recovery_by_need"]["food"] == {
        "episodes": 1,
        "completed": 1,
        "preempted": 0,
        "right_censored": 0,
        "actions": 1,
        "completion_rate": 1.0,
    }
    assert aggregate["environment_counter_totals"]["native_actions"] == 4
    assert aggregate["cumulative_reward"]["median"] == 3.5
    assert aggregate["achievement_successes"]["collect_diamond"] == 0
    assert aggregate["achievement_rates"]["place_table"] == 1.0
    assert "actions" not in aggregate["milestones"]["collect_wood"]
    assert aggregate["milestones"]["collect_wood"][
        "actions_among_attained"
    ]["median"] == 1.0
    assert aggregate["milestones"]["collect_diamond"]["right_censored"] == 1

    stage_rows = report["analyses"]["stage_activity"]["rows"]
    assert sum(row["actions"] for row in stage_rows) == outcome["steps"]
    assert stage_rows == [
        {
            "execution_id": "run-3",
            "run_id": 3,
            "stage": "collect-table-wood",
            "revisions": 3,
            "actions": 3,
            "planned_actions": 1,
            "recovery_actions": 1,
            "exploration_actions": 1,
            "planning_attempts": 2,
            "accepted_plans": 1,
            "planner_elapsed_seconds": 0.2,
            "inter_revision_wall_seconds": 3.0,
            "source_ref": {
                "state": "exp_agent2_3_3/out/exp_agent2_3_3.hlreasoner_0.json",
                "trace_index": 0,
            },
            "terminal_source_index": 2,
        },
        {
            "execution_id": "run-3",
            "run_id": 3,
            "stage": "place-table",
            "revisions": 2,
            "actions": 1,
            "planned_actions": 1,
            "recovery_actions": 0,
            "exploration_actions": 0,
            "planning_attempts": 1,
            "accepted_plans": 1,
            "planner_elapsed_seconds": 0.3,
            "inter_revision_wall_seconds": 2.0,
            "source_ref": {
                "state": "exp_agent2_3_3/out/exp_agent2_3_3.hlreasoner_0.json",
                "trace_index": 3,
            },
            "terminal_source_index": 4,
        },
    ]


def test_exp2_3_recovery_summary_covers_outcomes_and_empty_needs() -> None:
    reporting = import_module("mha_exp_level2_cr.exp2_3.reporting")
    rows = [
        {
            "need": "food",
            "completed": True,
            "actions": 2,
            "censoring": {"status": "observed", "reason": None},
        },
        {
            "need": "drink",
            "completed": False,
            "actions": 1,
            "censoring": {
                "status": "observed",
                "reason": "recovery_preempted",
            },
        },
        {
            "need": "energy",
            "completed": False,
            "actions": 3,
            "censoring": {
                "status": "right_censored",
                "reason": "execution_ended_during_recovery",
            },
        },
    ]

    summary = reporting._recovery_summary(rows)

    assert summary == {
        "food": {
            "episodes": 1,
            "completed": 1,
            "preempted": 0,
            "right_censored": 0,
            "actions": 2,
            "completion_rate": 1.0,
        },
        "drink": {
            "episodes": 1,
            "completed": 0,
            "preempted": 1,
            "right_censored": 0,
            "actions": 1,
            "completion_rate": 0.0,
        },
        "energy": {
            "episodes": 1,
            "completed": 0,
            "preempted": 0,
            "right_censored": 1,
            "actions": 3,
            "completion_rate": 0.0,
        },
    }
    assert reporting._recovery_summary([]) == {
        need: {
            "episodes": 0,
            "completed": 0,
            "preempted": 0,
            "right_censored": 0,
            "actions": 0,
            "completion_rate": None,
        }
        for need in ("food", "drink", "energy")
    }


def test_exp2_3_execution_statistics_preserve_unavailable_values() -> None:
    reporting = import_module("mha_exp_level2_cr.exp2_3.reporting")
    trace = [{
        "belief": {"needs": {"health": 9}},
        "monitoring": {"action_result": {"action_id": "action-1"}},
        "decision": {"kind": "planned-action"},
    }]

    statistics = reporting._execution_statistics(trace, None)

    assert statistics["minimum_needs"] == {
        "health": 9,
        "food": None,
        "drink": None,
        "energy": None,
    }
    assert statistics["actions_at_or_below_intervention"] == {
        "health": 0,
        "food": None,
        "drink": None,
        "energy": None,
    }
    assert statistics["cumulative_reward"] is None
    assert statistics["achievement_counts"] is None
    assert all(
        value is None
        for value in statistics["environment_counters"].values()
    )


@pytest.mark.parametrize(
    "field",
    (
        "protocol_version",
        "treatment_id",
        "manifest_digest",
        "task_id",
        "seed",
        "initial_support",
        "action_budget",
        "episode_length",
    ),
)
def test_exp2_3_certifies_every_frozen_treatment_field(field: str) -> None:
    reporting = import_module("mha_exp_level2_cr.exp2_3.reporting")
    treatment = import_module(
        "mha_exp_level2_cr.exp2_3.treatment"
    ).treatment_for_run(3)
    changed = {**treatment, field: None}

    assert reporting._treatment_identity_errors(
        {"factors": treatment},
        changed,
        {"treatment": changed},
    ) == ["treatment_identity_mismatch"]


def test_exp2_3_certifies_environment_treatment_copy() -> None:
    reporting = import_module("mha_exp_level2_cr.exp2_3.reporting")
    treatment = import_module(
        "mha_exp_level2_cr.exp2_3.treatment"
    ).treatment_for_run(3)

    assert reporting._treatment_identity_errors(
        {"factors": treatment},
        treatment,
        {"treatment": treatment},
    ) == []
    assert reporting._treatment_identity_errors(
        {"factors": treatment},
        treatment,
        {"treatment": {**treatment, "seed": 1004}},
    ) == ["environment_treatment_mismatch"]


def test_exp2_3_exploration_episodes_split_on_intention_change() -> None:
    reporting = import_module("mha_exp_level2_cr.exp2_3.reporting")
    trace = [
        {
            "belief": {"known_terrain": 1},
            "stage": {"support": {"supported": False}},
            "decision": {
                "kind": "explore-action",
                "reason": "missing-knowledge",
                "intention_id": "intention-1",
            },
        },
        {
            "belief": {"known_terrain": 2},
            "stage": {"support": {"supported": False}},
            "decision": {
                "kind": "explore-action",
                "reason": "missing-knowledge",
                "intention_id": "intention-2",
            },
        },
        {
            "belief": {"known_terrain": 3},
            "stage": {"support": {"supported": True}},
            "decision": {
                "kind": "planned-action",
                "intention_id": "intention-2",
            },
        },
    ]

    rows = reporting._exploration_rows(
        {"execution_id": "run-0", "run_id": 0},
        trace,
        "state.json",
    )

    assert [row["intention_id"] for row in rows] == [
        "intention-1",
        "intention-2",
    ]
    assert [row["actions"] for row in rows] == [1, 1]
    assert [row["support_revealed"] for row in rows] == [False, True]


def test_exp2_3_non_mapping_stage_marks_only_execution_unreadable(
    tmp_path: Path,
) -> None:
    agent_out = tmp_path / "exp_agent2_3_0" / "out"
    agent_out.mkdir(parents=True)
    (agent_out / "exp_agent2_3_0.hlreasoner_0.json").write_text(
        json.dumps({
            "run": {
                "treatment": {},
                "trace": [{
                    "belief": {},
                    "stage": [],
                    "intentions": [],
                    "planning": None,
                    "monitoring": {},
                    "decision": None,
                }],
            },
        }),
        encoding="utf-8",
    )

    report = import_module(
        "mha_exp_level2_cr.exp2_3.reporting"
    ).process_execution_metrics(
        tmp_path,
        expected_executions=[{
            "execution_id": "run-0",
            "run_id": 0,
            "factors": {},
        }],
    )

    execution = report["executions"][0]
    assert execution["readable"] is False
    assert execution["readability_reasons"] == ["invalid_trace_row"]
    assert report["analyses"]["stage_activity"]["rows"] == []


@pytest.mark.parametrize(
    "malformed_field",
    ("row_elapsed", "run_elapsed", "final_needs", "map_coverage"),
)
def test_exp2_3_malformed_reporting_values_do_not_abort_batch(
    tmp_path: Path,
    malformed_field: str,
) -> None:
    """A malformed execution must not prevent later executions being reported."""

    def state() -> dict[str, object]:
        return {
            "run": {
                "phase": "scientific-terminal",
                "terminal_reason": "action_budget_exhausted",
                "agent_steps": 0,
                "elapsed_seconds": 1.0,
                "highest_milestone": None,
                "treatment": {},
                "trace": [{
                    "revision": 1,
                    "elapsed_seconds": 1.0,
                    "belief": {
                        "achievements": [],
                        "needs": {
                            "health": 9,
                            "food": 9,
                            "drink": 9,
                            "energy": 9,
                        },
                        "known_terrain": 1,
                        "reachable_cells": 1,
                        "unknown_frontier": 1,
                    },
                    "stage": {},
                    "intentions": [],
                    "planning": None,
                    "monitoring": {},
                    "decision": {
                        "kind": "scientific-terminal",
                        "reason": "action_budget_exhausted",
                    },
                }],
            },
        }

    malformed = state()
    run = malformed["run"]
    assert isinstance(run, dict)
    trace = run["trace"]
    assert isinstance(trace, list)
    row = trace[0]
    assert isinstance(row, dict)
    belief = row["belief"]
    assert isinstance(belief, dict)
    if malformed_field == "row_elapsed":
        row["elapsed_seconds"] = "bad"
    elif malformed_field == "run_elapsed":
        run["elapsed_seconds"] = "bad"
    elif malformed_field == "final_needs":
        belief["needs"] = []
    else:
        belief["known_terrain"] = "bad"

    for run_id, value in ((0, malformed), (1, state())):
        out = tmp_path / f"exp_agent2_3_{run_id}" / "out"
        out.mkdir(parents=True)
        (out / f"exp_agent2_3_{run_id}.hlreasoner_0.json").write_text(
            json.dumps(value),
            encoding="utf-8",
        )

    report = import_module(
        "mha_exp_level2_cr.exp2_3.reporting"
    ).process_execution_metrics(
        tmp_path,
        expected_executions=[
            {"execution_id": f"run-{run_id}", "run_id": run_id, "factors": {}}
            for run_id in (0, 1)
        ],
    )

    assert [row["readable"] for row in report["executions"]] == [False, True]
    outcomes = report["analyses"]["execution_outcomes"]["rows"]
    assert outcomes[0]["scientifically_eligible"] is False
    assert outcomes[0]["wall_seconds"] is None
    assert outcomes[0]["final_needs"] is None
    assert outcomes[0]["map_coverage"] is None
    assert outcomes[1]["wall_seconds"] == 1.0
    assert {
        row["execution_id"]
        for row in report["analyses"]["milestone_outcomes"]["rows"]
    } == {"run-1"}


def test_exp2_2_joins_only_matching_playback_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Playback evidence is comparable only for the persisted checkpoint."""

    out = tmp_path / "exp_agent2_2_5" / "out"
    playback_root = out / "policy_evaluation"
    playback_root.mkdir(parents=True)
    low = {
        "action_histogram": [1, 1],
        "completed_episodes": 1,
        "total_episode_length": 2,
        "episodes_started": 1,
        "target_successes": 1,
        "deaths": 0,
        "truncations": 0,
        "transitions_emitted": 2,
        "contract_errors": 0,
    }
    learner = {
        "training_updates": 1,
        "target_syncs": 1,
        "models_published": 1,
        "model_artifact": "policy.pt",
        "contract_errors": 0,
    }
    (out / "exp_agent2_2_5.llreasoner_0.json").write_text(
        json.dumps(low), encoding="utf-8"
    )
    (out / "exp_agent2_2_5.learner_0.json").write_text(
        json.dumps(learner), encoding="utf-8"
    )
    checkpoint = out / "policy.pt"
    checkpoint.write_bytes(b"checkpoint")
    playback = {
        "schema_version": "2-2-cr-playback-v1",
        "protocol": {
            "checkpoint_sha256": hashlib.sha256(b"checkpoint").hexdigest(),
            "checkpoint_metadata": {},
            "target_achievement": "collect_diamond",
            "seed": 0,
            "requested_episodes": 1,
            "max_steps": 10,
            "mode": "gif",
            "fps": 2.0,
            "symbolic": False,
            "crafter_length": 10_000,
        },
        "episodes": [{
            "episode": 1, "success": True, "death": False, "steps": 2,
            "window_closed": False, "gif": None,
        }],
    }
    (playback_root / "playback-results.json").write_text(
        json.dumps(playback), encoding="utf-8"
    )
    runner = import_module("mha_exp_level2_cr.exp2_2.runner")
    monkeypatch.setattr(
        runner,
        "check_results_detailed",
        lambda *_args, **_kwargs: (True, [], {"sha256": "checkpoint"}),
    )

    report = import_module(
        "mha_exp_level2_cr.exp2_2.reporting"
    ).process_execution_metrics(
        tmp_path,
        expected_executions=[{
            "execution_id": "run-5", "run_id": 5, "factors": {},
        }],
    )

    assert report["executions"][0]["metric_availability"]["playback"] == "existing"
    assert report["analyses"]["playback_episodes"]["rows"][0][
        "protocol_compatible"
    ] is True
    protocol = report["analyses"]["playback_protocols"]["rows"][0]
    assert protocol["compatible"] is True
    assert report["analyses"]["playback_episodes"]["rows"][0][
        "protocol_id"
    ] == protocol["protocol_id"]


def test_usage_rows_distinguish_missing_state_from_observed_zero() -> None:
    calls, totals = _usage_rows([], {}, "run-1", 1)
    missing = next(row for row in totals if row["module_id"] == "llreasoner_0")
    assert calls == []
    assert missing["final_state_available"] is False
    assert missing["calls"] is None
    assert missing["estimated_cost_usd"] is None
    assert missing["cost_reconciled"] is None
    assert missing["reconciliation_unavailable_reason"] == "final_module_state_missing"

    _, totals = _usage_rows([], {
        "llreasoner_0": {
            "calls": 0, "failures": 0, "budget_exhausted": False,
            "estimated_cost_usd": "0",
        },
    }, "run-1", 1)
    observed = next(row for row in totals if row["module_id"] == "llreasoner_0")
    assert observed["final_state_valid"] is True
    assert observed["calls"] == 0 and observed["failures"] == 0
    assert observed["estimated_cost_usd"] == "0"
    assert observed["budget_exhausted"] is False
    assert observed["cost_reconciled"] is None
    assert observed["reconciliation_unavailable_reason"] == (
        "successful_event_snapshot_missing"
    )


def test_playback_protocol_distinguishes_malformed_and_incompatible(
    tmp_path: Path,
) -> None:
    loader = import_module(
        "mha_exp_level2_cr.exp2_2.reporting"
    )._playback_protocol
    path = tmp_path / "playback-results.json"
    path.write_text("{", encoding="utf-8")
    protocol, episodes, status = loader(
        path, checkpoint_sha256="expected", target_achievement="target",
    )
    assert protocol is None and episodes == [] and status == "invalid_json"

    path.write_text(json.dumps({
        "schema_version": "2-2-cr-playback-v1",
        "protocol": {
            "checkpoint_sha256": "different", "checkpoint_metadata": {},
            "target_achievement": "target", "seed": 0,
            "requested_episodes": 1, "max_steps": 10, "mode": "gif",
            "fps": 2.0, "crafter_length": 10_000, "symbolic": False,
        },
        "episodes": [{
            "episode": 1, "success": False, "death": False, "steps": 10,
            "window_closed": False, "gif": "episode.gif",
        }],
    }), encoding="utf-8")
    protocol, episodes, status = loader(
        path, checkpoint_sha256="expected", target_achievement="target",
    )
    assert protocol is not None and protocol["compatible"] is False
    assert len(episodes) == 1 and status == "incompatible_protocol"


def test_usage_rows_reject_invalid_cumulative_and_final_fields() -> None:
    events = [{
        "module_id": "llreasoner_0", "kind": "llm_call", "outcome": "success",
        "input_tokens": "invalid", "output_tokens": 1,
        "estimated_cost_usd": "0.1",
    }]
    calls, totals = _usage_rows(events, {
        "llreasoner_0": {
            "calls": "1", "failures": -1, "budget_exhausted": 0,
            "estimated_cost_usd": "invalid",
        },
    }, "run-1", 1)
    call = next(row for row in calls if row["module_id"] == "llreasoner_0")
    total = next(row for row in totals if row["module_id"] == "llreasoner_0")
    assert call["cumulative_input_tokens"] is None
    assert call["per_call_usage"] is None
    assert total["final_state_valid"] is False
    assert total["calls"] is None and total["failures"] is None
    assert total["budget_exhausted"] is None
    assert total["estimated_cost_usd"] is None
    assert total["cost_reconciled"] is None


def test_exp2_7_does_not_normalize_unsupported_evaluation(tmp_path: Path) -> None:
    run_root = tmp_path / "standard-symbolic-with" / "run-007"
    run_root.mkdir(parents=True)
    (run_root / "evaluation.json").write_text(json.dumps({
        "schema_version": "unsupported",
        "run": 7,
        "profile": {"name": "standard"},
        "observation_format": "symbolic",
        "value_condition": "with",
        "scientific_outcomes": {"primary_goal_achieved": True},
    }), encoding="utf-8")
    expected = [{
        "execution_id": "profile-standard/run-7/observation-symbolic/value-with",
        "run_id": 7,
        "factors": {
            "profile": "standard", "observation_format": "symbolic",
            "value_condition": "with",
        },
    }]

    report = import_module(
        "mha_exp_level2_cr.exp2_7.evaluate"
    ).process_execution_metrics(tmp_path, expected_executions=expected)

    assert report["executions"][0]["operational_reasons"] == ["unsupported_schema"]
    assert report["analyses"]["module_usage"]["rows"] == []
    assert report["analyses"]["goal_behavior_transitions"]["rows"] == []


def test_exp2_1_emits_censored_milestones_and_stall_windows(tmp_path: Path) -> None:
    agent_out = tmp_path / "exp_agent2_1_0" / "out"
    env_out = tmp_path / "exp_env2_1_0" / "out"
    agent_out.mkdir(parents=True)
    env_out.mkdir(parents=True)
    inventory = {"health": 9, "food": 9, "drink": 9, "energy": 9}
    decisions = [{
        "request_id": index, "episode_id": 0, "reason": "wait",
    } for index in range(6)]
    trace = [{
        "event_id": index, "episode_id": 0, "event_kind": "native_action",
        "request_id": index, "action": "noop", "elapsed_seconds": float(index + 1),
        "player_pos": [0, 0], "inventory": inventory,
        "newly_achieved": ["collect_wood"] if index == 0 else [],
        "environment_done": False,
    } for index in range(6)]
    (agent_out / "exp_agent2_1_0.llreasoner_0.json").write_text(
        json.dumps({"decision_trace": decisions}), encoding="utf-8"
    )
    (env_out / "exp_env2_1_0.json").write_text(json.dumps({
        "execution_trace": trace,
        "highest_diamond_path_achievement": "collect_wood",
    }), encoding="utf-8")

    report = import_module(
        "mha_exp_level2_cr.exp2_1.reporting"
    ).process_execution_metrics(
        tmp_path,
        expected_executions=[{"execution_id": "run-0", "run_id": 0, "factors": {}}],
    )

    attained = next(row for row in report["analyses"]["milestones"]["rows"]
                    if row["milestone"] == "collect_wood")
    open_milestone = next(
        row for row in report["analyses"]["milestones"]["rows"]
        if row["milestone"] == "collect_diamond"
    )
    stall = report["analyses"]["stall_windows"]["rows"][0]
    assert attained["first_attainment_action"] == 1
    assert attained["censoring"]["status"] == "observed"
    assert open_milestone["censoring"] == {
        "status": "right_censored", "reason": "external_execution_end",
    }
    assert stall["actions_observed"] == 6
    assert stall["censoring"]["status"] == "right_censored"


def test_exp2_1_rejects_malformed_decision_rows(tmp_path: Path) -> None:
    agent_out = tmp_path / "exp_agent2_1_0" / "out"
    env_out = tmp_path / "exp_env2_1_0" / "out"
    agent_out.mkdir(parents=True)
    env_out.mkdir(parents=True)
    (agent_out / "exp_agent2_1_0.llreasoner_0.json").write_text(
        json.dumps({"decision_trace": ["invalid"]}), encoding="utf-8"
    )
    (env_out / "exp_env2_1_0.json").write_text(
        json.dumps({"execution_trace": []}), encoding="utf-8"
    )

    report = import_module(
        "mha_exp_level2_cr.exp2_1.reporting"
    ).process_execution_metrics(
        tmp_path,
        expected_executions=[{
            "execution_id": "run-0", "run_id": 0, "factors": {},
        }],
    )

    assert report["executions"][0]["readability_reasons"] == [
        "invalid_trace_row"
    ]
    assert report["analyses"]["native_actions"]["rows"] == []


def test_exp2_2_rejects_malformed_training_counters(tmp_path: Path) -> None:
    out = tmp_path / "exp_agent2_2_0" / "out"
    out.mkdir(parents=True)
    (out / "exp_agent2_2_0.llreasoner_0.json").write_text(json.dumps({
        "action_histogram": [1, "invalid"],
    }), encoding="utf-8")
    (out / "exp_agent2_2_0.learner_0.json").write_text(
        json.dumps({}), encoding="utf-8"
    )

    report = import_module(
        "mha_exp_level2_cr.exp2_2.reporting"
    ).process_execution_metrics(
        tmp_path,
        expected_executions=[{
            "execution_id": "run-0", "run_id": 0, "factors": {},
        }],
    )

    assert report["executions"][0]["readability_reasons"] == [
        "invalid_counter"
    ]
    assert report["analyses"]["training_runs"]["rows"] == []


def test_exp2_5_rejects_malformed_policy_count_values() -> None:
    from mha_exp_common.utils import module_name

    runtime = import_module("mha_exp_level2_cr.exp2_5.runtime")
    runner = import_module("mha_exp_level2_cr.exp2_5.runner")
    environment = import_module("mha_exp_level2_cr.exp2_5.environment")
    states = runtime.initial_states()
    states["llreasoner"]["policy_action_counts"] = {"policy": "invalid"}
    passed, errors = runner.check_results(
        {module_name(name, 0): value for name, value in states.items()},
        environment.initial_state(),
    )
    assert not passed and "policy counts disagree" in errors


@pytest.mark.parametrize(
    "field",
    (
        "protocol_version",
        "treatment_id",
        "manifest_digest",
        "task_id",
        "seed",
        "stratum",
        "primary_intention",
        "no_mobs",
        "daylight_effects",
        "duration",
        "total_action_cap",
    ),
)
def test_exp2_4_certifies_every_frozen_treatment_field(field: str) -> None:
    reporting = import_module("mha_exp_level2_cr.exp2_4.reporting")
    treatment = import_module(
        "mha_exp_level2_cr.exp2_4.treatment"
    ).treatment_for_run(0)

    assert reporting._treatment_identity_errors(
        {"factors": treatment}, {**treatment, field: "changed"}
    ) == ["treatment_identity_mismatch"]


def _write_exp2_4_provenance(root: Path) -> None:
    payload = {
        "protocol_version": "2-4-cr-execution-provenance-v1",
        "workspace_git_commit": "1" * 40,
        "workspace_git_dirty": False,
        "mhagenta_git_commit": "2" * 40,
        "mhagenta_git_dirty": False,
        "mhagenta_version": "1.4.12",
        "experiment_source_sha256": "3" * 64,
        "crafter_source_sha256": "4" * 64,
        "mhagenta_source_sha256": "5" * 64,
        "uv_lock_sha256": "6" * 64,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    payload["provenance_id"] = hashlib.sha256(canonical).hexdigest()
    (root / "execution-provenance.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )


def _exp2_4_atomic(
    action_id: int,
    movement: str,
    source: tuple[int, int],
    destination: tuple[int, int],
) -> dict[str, object]:
    return {
        "action_id": f"action-{action_id}",
        "action": 1,
        "movement_kind": movement,
        "source_cell": list(source),
        "destination_cell": list(destination),
        "dispatch_revision": action_id,
        "legal": True,
        "confirmation_revision": action_id + 1,
    }


def _exp2_4_activity(
    index: int,
    activity: str,
    status: str,
    atomic: list[dict[str, object]],
    *,
    start: int,
    end: int | None,
    desired: tuple[str, str] = ("inventory_at_least", "wood"),
    interruption: dict[str, object] | None = None,
) -> dict[str, object]:
    return {
        "goal_id": f"activity-{index}",
        "stage": "table",
        "plan_revision": index,
        "activity": activity,
        "desired": {
            "predicate": desired[0],
            "arguments": [desired[1], 1] if desired[0] == "inventory_at_least" else [desired[1]],
            "extras": None,
        },
        "based_on_revision": index,
        "known_cell_count_start": start,
        "known_cell_count_end": end,
        "atomic": atomic,
        "completion_revision": None if end is None else index + 1,
        "final_value": None if end is None else status == "succeeded",
        "status": status,
        "failure_reason": "need_interruption" if interruption else "" if status == "succeeded" else "test_failure",
        "interruption": interruption,
    }


def _write_exp2_4_run(
    root: Path,
    run_id: int,
    activities: list[dict[str, object]],
    *,
    native_actions: int,
    diamond: int,
    status: str,
    achievement_counts: dict[str, int],
) -> None:
    agent_id = f"exp_agent2_4_{run_id}"
    out = root / agent_id / "out"
    env_out = root / f"exp_env2_4_{run_id}" / "out"
    out.mkdir(parents=True)
    env_out.mkdir(parents=True)
    high = {
        "hierarchy": {
            "status": status,
            "terminal_reason": "diamond_obtained" if diamond else "total_action_bound",
            "intention": {"name": "obtain_diamond", "target_value": 1},
            "activities": activities,
            "highest_milestone": "diamond" if diamond else "start",
            "current_stage": "diamond" if diamond else "table",
        },
        "abstract_state": {
            "inventory": {
                "diamond": diamond,
                "health": 9,
                "food": 8,
                "drink": 7,
                "energy": 6,
            },
            "known_cell_count": max(
                (int(row["known_cell_count_end"] or row["known_cell_count_start"]) for row in activities),
                default=0,
            ),
        },
        "need_interruptions": sum(row["status"] == "interrupted" for row in activities),
        "recoveries": 1 if any(row["status"] == "interrupted" for row in activities) else 0,
        "failure": None,
        "active_seconds": 1.0,
    }
    for module in (
        "perceptor_0", "actuator_0", "llreasoner_0", "knowledge_0", "goalgraph_0",
    ):
        (out / f"{agent_id}.{module}.json").write_text(
            json.dumps({"failure": None, "active_seconds": 1.0}), encoding="utf-8"
        )
    (out / f"{agent_id}.hlreasoner_0.json").write_text(
        json.dumps(high), encoding="utf-8"
    )
    (env_out / f"exp_env2_4_{run_id}.json").write_text(
        json.dumps({
            "native_actions": native_actions,
            "terminal": True,
            "dead": False,
            "achievement_counts": achievement_counts,
        }),
        encoding="utf-8",
    )
    (root / f"{agent_id}.log").write_text("[INFO] complete\n", encoding="utf-8")
    (root / f"exp_env2_4_{run_id}.log").write_text(
        "[INFO] complete\n", encoding="utf-8"
    )


def test_exp2_4_reports_corrected_metrics_and_eligible_run_cohort(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_exp2_4_provenance(tmp_path)
    activities = [
        _exp2_4_activity(
            1, "place_table", "succeeded",
            [
                _exp2_4_atomic(1, "turn", (0, 0), (0, 1)),
                _exp2_4_atomic(2, "none", (0, 0), (0, 0)),
            ],
            start=10, end=11,
        ),
        _exp2_4_activity(
            2, "get_diamond", "succeeded",
            [_exp2_4_atomic(3, "walk", (0, 0), (1, 0))],
            start=11, end=11, desired=("inventory_at_least", "diamond"),
        ),
        _exp2_4_activity(
            3, "get_wood", "interrupted",
            [_exp2_4_atomic(4, "none", (1, 0), (1, 0))],
            start=11, end=11,
            interruption={"need": "drink"},
        ),
        _exp2_4_activity(
            4, "explore", "failed",
            [_exp2_4_atomic(5, "walk", (1, 0), (2, 0))],
            start=11, end=13, desired=("reachable_target_kind", "water"),
        ),
        _exp2_4_activity(
            5, "explore", "active",
            [_exp2_4_atomic(6, "none", (2, 0), (2, 0))],
            start=13, end=None, desired=("placement_opportunity", "furnace"),
        ),
    ]
    _write_exp2_4_run(
        tmp_path, 0, activities, native_actions=6, diamond=1, status="succeeded",
        achievement_counts={"collect_wood": 1, "collect_diamond": 1},
    )
    failed = [_exp2_4_activity(
        1, "explore", "failed", [_exp2_4_atomic(1, "walk", (0, 0), (1, 0))],
        start=5, end=6, desired=("reachable_target_kind", "tree"),
    )]
    _write_exp2_4_run(
        tmp_path, 1, failed, native_actions=1, diamond=0, status="blocked",
        achievement_counts={"collect_wood": 0, "collect_diamond": 0},
    )
    _write_exp2_4_run(
        tmp_path, 2, failed, native_actions=2, diamond=1, status="succeeded",
        achievement_counts={"collect_wood": 1, "collect_diamond": 1},
    )
    runner = import_module("mha_exp_level2_cr.exp2_4.runner")
    monkeypatch.setattr(runner, "check_results", lambda *args, **kwargs: True)
    reporting = import_module("mha_exp_level2_cr.exp2_4.reporting")
    report = reporting.process_execution_metrics(
        tmp_path,
        expected_executions=[
            {"execution_id": f"run-{run}", "run_id": run, "factors": {}}
            for run in range(3)
        ],
    )

    runs = report["analyses"]["hierarchy_runs"]["rows"]
    run = runs[0]
    assert run["atomic_actions_per_completed_activity"] == {
        "numerator": 3, "denominator": 2, "value": 1.5,
    }
    assert [
        run["succeeded_activity_actions"],
        run["interrupted_activity_actions"],
        run["failed_activity_actions"],
        run["open_activity_actions"],
    ] == [3, 1, 1, 1]
    assert run["attributed_atomic_actions"] == run["native_actions"] == 6
    assert runs[1]["atomic_actions_per_completed_activity"]["value"] is None
    assert run["exploration_by_purpose"] == {
        "placement:furnace": {
            "activities": 1,
            "actions": 1,
            "known_cells_added": 0,
            "known_cells_added_missing": 1,
        },
        "target:water": {
            "activities": 1,
            "actions": 1,
            "known_cells_added": 2,
            "known_cells_added_missing": 0,
        },
    }
    assert run["exploration_known_cells_added"] == 2
    assert run["exploration_known_cells_added_missing"] == 1
    assert run["known_cells_added_per_explore_action"] is None
    assert run["interruptions_by_need"] == {"drink": 1}
    assert (run["unique_visited_cells"], run["walk_actions"], run["turn_actions"], run["nonmovement_actions"]) == (3, 2, 1, 3)
    milestones = report["analyses"]["milestones"]["rows"][:6]
    assert next(row for row in milestones if row["milestone"] == "table")["native_action_ordinal"] == 2
    assert next(row for row in milestones if row["milestone"] == "diamond")["native_action_ordinal"] == 3
    censored = next(row for row in milestones if row["milestone"] == "wood_pickaxe")
    assert censored["censoring"] == {
        "status": "right_censored", "action": 6, "reason": "diamond_obtained",
    }
    summary = report["analyses"]["hierarchy_runs"]["summary"]
    assert summary["diamond_success"]["successes"] == 1
    assert summary["diamond_success"]["denominator"] == 2
    assert summary["diamond_success"]["wilson_95_interval"] == list(
        reporting.wilson_interval(1, 2)
    )
    assert summary["achievements"]["collect_diamond"]["denominator"] == 2
    assert report["executions"][2]["operational_reasons"] == [
        "atomic_action_accounting_mismatch"
    ]
    assert report["analyses"]["activities"]["unit"] == "activity"
    assert report["analyses"]["activities"]["nesting"] == "activities_within_execution"
    assert summary["diamond_success"]["denominator"] != len(
        report["analyses"]["activities"]["rows"]
    )


def test_exp2_4_malformed_execution_does_not_abort_later_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_exp2_4_provenance(tmp_path)
    malformed = [_exp2_4_activity(
        1,
        "explore",
        "failed",
        [_exp2_4_atomic(1, "walk", (0, 0), (1, 0))],
        start=5,
        end=6,
        desired=("reachable_target_kind", "tree"),
    )]
    malformed[0]["atomic"][0]["source_cell"] = ["bad", 0]
    valid = [_exp2_4_activity(
        1,
        "explore",
        "failed",
        [_exp2_4_atomic(1, "walk", (0, 0), (1, 0))],
        start=5,
        end=6,
        desired=("reachable_target_kind", "tree"),
    )]
    for run_id, activities in enumerate((malformed, valid)):
        _write_exp2_4_run(
            tmp_path,
            run_id,
            activities,
            native_actions=1,
            diamond=0,
            status="blocked",
            achievement_counts={"collect_wood": 0, "collect_diamond": 0},
        )
    runner = import_module("mha_exp_level2_cr.exp2_4.runner")
    monkeypatch.setattr(runner, "check_results", lambda *args, **kwargs: True)
    report = import_module(
        "mha_exp_level2_cr.exp2_4.reporting"
    ).process_execution_metrics(
        tmp_path,
        expected_executions=[
            {"execution_id": f"run-{run}", "run_id": run, "factors": {}}
            for run in range(2)
        ],
    )

    assert report["executions"][0]["readability_reasons"] == [
        "invalid_trace_row"
    ]
    assert report["executions"][1]["operationally_valid"] is True
    assert report["analyses"]["hierarchy_runs"]["rows"][0]["execution_id"] == "run-1"
    assert (tmp_path / "execution-metrics.json").is_file()


@pytest.mark.parametrize(
    ("content", "reason"),
    [(None, "required_provenance_missing"), ("not-json", "required_provenance_malformed")],
)
def test_exp2_4_provenance_is_required_for_operational_validity(
    tmp_path: Path, content: str | None, reason: str
) -> None:
    if content is not None:
        (tmp_path / "execution-provenance.json").write_text(content, encoding="utf-8")
    report = import_module(
        "mha_exp_level2_cr.exp2_4.reporting"
    ).process_execution_metrics(
        tmp_path,
        expected_executions=[{"execution_id": "run-0", "run_id": 0, "factors": {}}],
    )
    assert reason in report["executions"][0]["operational_reasons"]
    assert report["execution_provenance"]["status"] == "invalid"
