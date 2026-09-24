"""Cross-experiment contract checks for additive Level 2 BW reports."""

from __future__ import annotations

import json
from importlib import import_module
from pathlib import Path

import pytest

PROCESSOR_MODULES = (
    "mha_exp_level2_bw.exp2_1.reporting",
    "mha_exp_level2_bw.exp2_2.reporting",
    "mha_exp_level2_bw.exp2_3.reporting",
    "mha_exp_level2_bw.exp2_4.reporting",
    "mha_exp_level2_bw.exp2_5.reporting",
    "mha_exp_level2_bw.exp2_7.evaluate",
)


def _exp2_2_summary(
    run: int,
    *,
    training_successes: int = 1,
    ordinary_episodes: int = 1,
) -> dict[str, object]:
    protocol = {
        "scheduling_mode": "synchronous",
        "warmup_transitions": 1,
        "total_training_transitions": 10,
        "training_updates": 5,
        "target_sync_steps": 2,
        "evaluation_seeds": [100],
        "behavior_window_transitions": 10,
        "optimization_window_updates": 5,
    }
    return {
        "run": run,
        "protocol": protocol,
        "checker_passed": True,
        "protocol_complete": True,
        "counts": {
            "training_transitions": 10, "learner_updates": 5,
            "target_syncs": 2, "training_seconds": 2.0,
            "optimization_seconds": 1.0, "transition_throughput": 5.0,
            "update_throughput": 5.0,
        },
        "models": {"published": 1},
        "training": {
            "successes": training_successes,
            "truncations": ordinary_episodes - training_successes,
            "budget_cutoffs": 0,
            "ordinary_episodes": ordinary_episodes,
            "success_rate": training_successes / ordinary_episodes,
            "behavior_windows": [{
                "transition_start": 1, "transition_end": 10,
                "completed_episodes": ordinary_episodes,
                "successes": training_successes,
            }],
        },
        "optimization": {"windows": [{
            "update_start": 1, "update_end": 5,
            "mean_loss": 0.5, "final_loss": 0.25,
        }]},
        "evaluation": {
            "cases": [{"requested_seed": 100, "applied_seed": 100,
                       "success": True, "steps": 3}],
            "successes": 1, "success_rate": 1.0,
            "mean_successful_episode_length": 3.0,
        },
        "isolation": {"evaluation_replay_insertion_attempts": 0},
        "checkpoint": {"sha256": f"checkpoint-{run}"},
        "goal_graph": {},
        "environment": {},
    }


@pytest.mark.parametrize("module_name", PROCESSOR_MODULES)
def test_processor_accounts_for_expected_missing_execution(
    tmp_path: Path,
    module_name: str,
) -> None:
    """Every adapter must retain an expected cell even when no run exists."""

    processor = import_module(module_name).process_execution_metrics
    expected = [{"execution_id": "run-7", "run_id": 7, "factors": {}}]

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
    assert report["executions"][0]["execution_id"] == "run-7"
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
        assert eligibility["excluded"] == [{
            "execution_id": "run-7", "reasons": ["run_missing"],
        }]
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


def test_exp2_7_reconciles_per_request_and_module_usage(tmp_path: Path) -> None:
    """Per-request usage is compared with, not added to, cumulative state."""

    out = tmp_path / "with" / "exp_agent2_7_1" / "out"
    responses = out / "llm_responses"
    responses.mkdir(parents=True)
    result = {
        "schema_version": "2-7-bw-result-v1",
        "run": 1,
        "value_condition": "with",
        "architecture_valid": True,
        "matched_inclusion": True,
        "strict_bw_success": True,
        "scientific_outcomes": {
            "attempted_actions": 1,
            "illegal_or_invalid_actions": 0,
        },
    }
    (out / "result.json").write_text(json.dumps(result), encoding="utf-8")
    call = {
        "success": True,
        "latency_seconds": 0.5,
        "schema_retry": False,
        "raw_response": {
            "usage": {
                "input_tokens": 10,
                "output_tokens": 2,
                "total_tokens": 12,
            },
            "proxy_accounting": {"last_request_cost_usd": "0.01"},
        },
    }
    (responses / "llreasoner_0.jsonl").write_text(
        json.dumps(call) + "\n", encoding="utf-8"
    )
    state = {
        "response_records": 1,
        "repair_count": 0,
        "usage": {
            "input_tokens": 10,
            "output_tokens": 2,
            "total_tokens": 12,
            "estimated_cost_usd": "0.01",
        },
    }
    (out / "exp_agent2_7_1.llreasoner_0.json").write_text(
        json.dumps(state), encoding="utf-8"
    )

    report = import_module(
        "mha_exp_level2_bw.exp2_7.evaluate"
    ).process_execution_metrics(
        tmp_path,
        expected_executions=[{
            "execution_id": "run-1/value-with",
            "run_id": 1,
            "factors": {"value_condition": "with"},
        }],
    )

    usage = report["analyses"]["module_usage"]["rows"][0]
    assert usage["per_request_usage_reconciled"] is True
    assert usage["per_request_usage_sum"] == {
        "input_tokens": 10,
        "output_tokens": 2,
        "total_tokens": 12,
        "estimated_cost_usd": "0.01",
    }
    assert usage["budget_exhausted"] is None
    assert usage["usage_field_unavailable_reasons"] == {
        "budget_exhausted": "missing_or_invalid",
    }
    missing_modules = report["analyses"]["module_usage"]["rows"][1:]
    assert missing_modules
    assert all(row["usage_status"] == "unavailable" for row in missing_modules)
    assert "p95" not in usage["call_latency_seconds"]


@pytest.mark.parametrize(
    "content,reason", [("{", "invalid_json"), ("[]", "invalid_root_type")],
)
def test_exp2_2_preserves_present_unreadable_summary(
    tmp_path: Path,
    content: str,
    reason: str,
) -> None:
    run_root = tmp_path / "exp_agent2_2_4"
    run_root.mkdir()
    (run_root / "run-summary.json").write_text(content, encoding="utf-8")

    report = import_module(
        "mha_exp_level2_bw.exp2_2.reporting"
    ).process_execution_metrics(
        tmp_path,
        expected_executions=[{"execution_id": "run-4", "run_id": 4, "factors": {}}],
    )

    execution = report["executions"][0]
    assert execution["present"] is True and execution["readable"] is False
    assert execution["readability_reasons"] == [reason]


def test_exp2_2_rejects_embedded_run_mismatch_without_rows(tmp_path: Path) -> None:
    run_root = tmp_path / "exp_agent2_2_4"
    run_root.mkdir()
    (run_root / "run-summary.json").write_text(
        json.dumps(_exp2_2_summary(9)), encoding="utf-8"
    )

    report = import_module(
        "mha_exp_level2_bw.exp2_2.reporting"
    ).process_execution_metrics(
        tmp_path,
        expected_executions=[{"execution_id": "run-4", "run_id": 4, "factors": {}}],
    )

    assert report["executions"][0]["operational_reasons"] == ["identity_mismatch"]
    assert report["analyses"]["training_episodes"]["rows"] == []


def test_exp2_2_preserves_canonical_run_evidence_and_denominators(
    tmp_path: Path,
) -> None:
    for run, successes, episodes in ((0, 1, 1), (1, 0, 9)):
        run_root = tmp_path / f"exp_agent2_2_{run}"
        run_root.mkdir()
        (run_root / "run-summary.json").write_text(
            json.dumps(_exp2_2_summary(
                run, training_successes=successes, ordinary_episodes=episodes,
            )),
            encoding="utf-8",
        )
    expected = [
        {"execution_id": f"run-{run}", "run_id": run, "factors": {}}
        for run in (0, 1)
    ]

    report = import_module(
        "mha_exp_level2_bw.exp2_2.reporting"
    ).process_execution_metrics(tmp_path, expected_executions=expected)

    training = report["analyses"]["training_episodes"]
    assert training["summary"]["per_execution_success_rate"]["median"] == 0.5
    assert training["summary"]["pooled_descriptive"] == {
        "successes": 1, "ordinary_episodes": 10, "success_rate": 0.1,
    }
    run_row = report["analyses"]["run_learning"]["rows"][0]
    assert run_row["training_seconds"] == 2.0
    assert run_row["transition_throughput"] == 5.0
    assert run_row["isolation"]["evaluation_replay_insertion_attempts"] == 0
    assert run_row["checkpoint"]["sha256"] == "checkpoint-0"


def test_exp2_2_declares_case_level_eligibility(tmp_path: Path) -> None:
    summary = _exp2_2_summary(0)
    case = summary["evaluation"]["cases"][0]
    case.update({
        "analysis_eligible": False,
        "eligibility_reasons": ["held_out_case_mismatch"],
    })
    run_root = tmp_path / "exp_agent2_2_0"
    run_root.mkdir()
    (run_root / "run-summary.json").write_text(
        json.dumps(summary), encoding="utf-8"
    )

    report = import_module(
        "mha_exp_level2_bw.exp2_2.reporting"
    ).process_execution_metrics(
        tmp_path,
        expected_executions=[{
            "execution_id": "run-0", "run_id": 0, "factors": {},
        }],
    )

    analysis = report["analyses"]["evaluation_cases"]
    assert analysis["eligibility"]["eligible_case_ids"] == []
    assert analysis["eligibility"]["excluded_cases"] == [{
        "case_id": "run-0/case-0", "reasons": ["held_out_case_mismatch"],
    }]
    assert analysis["summary"]["count"] == 0


def test_exp2_7_architecture_is_independent_from_operational_validity(
    tmp_path: Path,
) -> None:
    out = tmp_path / "with" / "exp_agent2_7_2" / "out"
    out.mkdir(parents=True)
    (out / "result.json").write_text(json.dumps({
        "schema_version": "2-7-bw-result-v1",
        "run": 2,
        "value_condition": "with",
        "architecture_valid": False,
        "matched_inclusion": True,
        "strict_bw_success": False,
        "scientific_outcomes": {
            "attempted_actions": 1, "illegal_or_invalid_actions": 0,
        },
    }), encoding="utf-8")

    report = import_module(
        "mha_exp_level2_bw.exp2_7.evaluate"
    ).process_execution_metrics(
        tmp_path,
        expected_executions=[{
            "execution_id": "run-2/value-with", "run_id": 2,
            "factors": {"value_condition": "with"},
        }],
    )

    execution = report["executions"][0]
    assert execution["operationally_valid"] is True
    assert execution["identity"]["architecture_valid"] is False
    assert report["analyses"]["task_outcomes"]["eligibility"] == {
        "eligible_execution_ids": ["run-2/value-with"],
        "excluded": [],
    }


def test_exp2_7_malformed_call_row_is_unreadable_and_not_normalized(
    tmp_path: Path,
) -> None:
    out = tmp_path / "with" / "exp_agent2_7_2" / "out"
    responses = out / "llm_responses"
    responses.mkdir(parents=True)
    (out / "result.json").write_text(json.dumps({
        "schema_version": "2-7-bw-result-v1", "run": 2,
        "value_condition": "with", "architecture_valid": True,
        "matched_inclusion": True, "strict_bw_success": False,
        "scientific_outcomes": {},
    }), encoding="utf-8")
    (responses / "llreasoner_0.jsonl").write_text("{", encoding="utf-8")

    report = import_module(
        "mha_exp_level2_bw.exp2_7.evaluate"
    ).process_execution_metrics(
        tmp_path,
        expected_executions=[{
            "execution_id": "run-2/value-with", "run_id": 2,
            "factors": {"value_condition": "with"},
        }],
    )

    execution = report["executions"][0]
    assert execution["readable"] is False
    assert execution["readability_reasons"] == ["invalid_trace_row"]
    assert report["analyses"]["llm_calls"]["rows"] == []


def test_exp2_1_rejects_malformed_nested_rows(tmp_path: Path) -> None:
    out = tmp_path / "exp_agent2_1_0" / "out"
    out.mkdir(parents=True)
    (out / "exp_agent2_1_0.llreasoner_0.json").write_text(json.dumps({
        "episode_results": ["invalid"], "decision_trace": [],
    }), encoding="utf-8")

    report = import_module(
        "mha_exp_level2_bw.exp2_1.reporting"
    ).process_execution_metrics(
        tmp_path,
        expected_executions=[{
            "execution_id": "run-0", "run_id": 0, "factors": {},
        }],
    )

    assert report["executions"][0]["readability_reasons"] == [
        "invalid_trace_row"
    ]
    assert report["analyses"]["episodes"]["rows"] == []


def test_exp2_3_present_malformed_state_is_not_reported_missing(tmp_path: Path) -> None:
    out = tmp_path / "exp_agent2_3_3" / "out"
    out.mkdir(parents=True)
    (out / "exp_agent2_3_3.hlreasoner_0.json").write_text("{", encoding="utf-8")

    report = import_module(
        "mha_exp_level2_bw.exp2_3.reporting"
    ).process_execution_metrics(
        tmp_path,
        expected_executions=[{"execution_id": "run-3", "run_id": 3, "factors": {}}],
    )

    execution = report["executions"][0]
    assert execution["present"] is True and execution["readable"] is False
    assert execution["readability_reasons"] == ["invalid_json"]


def test_exp2_4_reports_transfer_and_atomic_hierarchy_facts(
    tmp_path: Path,
) -> None:
    out = tmp_path / "exp_agent2_4_1" / "out"
    out.mkdir(parents=True)
    high = {
        "retained_hierarchy": {
            "hierarchy_id": "hierarchy-1", "status": "completed",
            "intention": {"top": "B0", "bottom": "B1"},
            "plan": {
                "engine": "enhsp", "transfers": [{
                    "block": "B0", "source": "left", "destination": "right",
                    "source_support": "table", "destination_support": "B1",
                }],
            },
            "execution": [{
                "goal_id": "compound-1", "step_index": 0, "status": "completed",
                "dispatch_observation_seq": 1, "completion_observation_seq": 3,
                "completion_belief_observation_seq": 3,
                "atomic_rows": [{"action": "pick-up", "legal": True}],
            }],
        },
        "goal_completions": 1, "completed_transfers": 1,
        "compound_failures": 0, "failure": None,
    }
    (out / "exp_agent2_4_1.hlreasoner_0.json").write_text(
        json.dumps(high), encoding="utf-8"
    )

    report = import_module(
        "mha_exp_level2_bw.exp2_4.reporting"
    ).process_execution_metrics(
        tmp_path,
        expected_executions=[{"execution_id": "run-1", "run_id": 1, "factors": {}}],
    )

    transfer = report["analyses"]["transfers"]["rows"][0]
    assert transfer["plan_source"] == "fallback"
    assert transfer["transfer_direction"]["destination"] == "right"
    assert report["analyses"]["atomic_actions"]["rows"][0][
        "transfer_id"
    ] == "compound-1"


def test_exp2_4_rejects_partially_malformed_nested_hierarchy(
    tmp_path: Path,
) -> None:
    out = tmp_path / "exp_agent2_4_1" / "out"
    out.mkdir(parents=True)
    (out / "exp_agent2_4_1.hlreasoner_0.json").write_text(json.dumps({
        "retained_hierarchy": {
            "plan": {"engine": "lpg", "transfers": [{}]},
            "execution": [{"atomic_rows": [{"legal": True}, "bad-row"]}],
        },
    }), encoding="utf-8")

    report = import_module(
        "mha_exp_level2_bw.exp2_4.reporting"
    ).process_execution_metrics(
        tmp_path,
        expected_executions=[{
            "execution_id": "run-1", "run_id": 1, "factors": {},
        }],
    )

    execution = report["executions"][0]
    assert execution["readable"] is False
    assert execution["readability_reasons"] == ["invalid_trace_row"]
    assert report["analyses"]["transfers"]["rows"] == []
    assert report["analyses"]["atomic_actions"]["rows"] == []


def test_exp2_1_missing_logs_make_historical_checker_unavailable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent_out = tmp_path / "exp_agent2_1_0" / "out"
    environment_out = tmp_path / "exp_env2_1_0" / "out"
    agent_out.mkdir(parents=True)
    environment_out.mkdir(parents=True)
    for module_id, state in {
        "perceptor_0": {},
        "actuator_0": {},
        "llreasoner_0": {
            "episode_results": [], "decision_trace": [],
            "open_episode": None, "illegal_actions": 0,
        },
    }.items():
        (agent_out / f"exp_agent2_1_0.{module_id}.json").write_text(
            json.dumps(state), encoding="utf-8"
        )
    (environment_out / "exp_env2_1_0.json").write_text(
        "{}", encoding="utf-8"
    )
    runner = import_module("mha_exp_level2_bw.exp2_1.runner")
    monkeypatch.setattr(
        runner,
        "check_results",
        lambda *_args, **_kwargs: pytest.fail("checker must not be called"),
    )

    report = import_module(
        "mha_exp_level2_bw.exp2_1.reporting"
    ).process_execution_metrics(
        tmp_path,
        expected_executions=[{
            "execution_id": "run-0", "run_id": 0, "factors": {},
        }],
    )

    execution = report["executions"][0]
    assert execution["certificate_status"] == "unavailable"
    assert execution["operationally_valid"] is False
    assert execution["operational_reasons"] == ["required_log_missing"]
    assert report["analyses"]["controller_runs"]["eligibility"][
        "eligible_execution_ids"
    ] == []


def test_exp2_5_checker_requires_complete_runtime_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent_out = tmp_path / "exp_agent2_5_0" / "out"
    environment_out = tmp_path / "exp_env2_5_0" / "out"
    agent_out.mkdir(parents=True)
    environment_out.mkdir(parents=True)
    required = {
        "perceptor_0", "actuator_0", "llreasoner_0", "knowledge_0",
        "goalgraph_0", "hlreasoner_0",
    }
    for module_id in required:
        state = (
            {"goal_runs": [], "goal_completion_count": 0, "failure": None}
            if module_id == "hlreasoner_0" else {}
        )
        (agent_out / f"exp_agent2_5_0.{module_id}.json").write_text(
            json.dumps(state), encoding="utf-8"
        )
    (environment_out / "exp_env2_5_0.json").write_text(
        "{}", encoding="utf-8"
    )
    policy = import_module("mha_exp_level2_bw.exp2_5.policy")
    monkeypatch.setattr(
        policy, "artifact_paths", lambda: (tmp_path / "manifest", tmp_path / "model")
    )
    monkeypatch.setattr(
        policy,
        "validate_manifest",
        lambda *_args: {
            "checkpoint_sha256": "artifact", "architecture": "frozen",
            "training": {}, "certification": {},
        },
    )
    runner = import_module("mha_exp_level2_bw.exp2_5.runner")
    monkeypatch.setattr(
        runner,
        "check_results",
        lambda *_args, **_kwargs: pytest.fail("checker must not be called"),
    )

    report = import_module(
        "mha_exp_level2_bw.exp2_5.reporting"
    ).process_execution_metrics(
        tmp_path,
        expected_executions=[{
            "execution_id": "run-0", "run_id": 0, "factors": {},
        }],
    )

    execution = report["executions"][0]
    assert execution["certificate_status"] == "unavailable"
    assert execution["operational_reasons"] == ["required_log_missing"]


@pytest.mark.parametrize(
    ("artifact_mode", "reason"),
    [
        ("missing", "preparation_artifact_unavailable"),
        ("mismatched", "artifact_identity_mismatch"),
    ],
)
def test_exp2_5_artifact_join_controls_analysis_eligibility(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    artifact_mode: str,
    reason: str,
) -> None:
    agent_out = tmp_path / "exp_agent2_5_0" / "out"
    environment_out = tmp_path / "exp_env2_5_0" / "out"
    agent_out.mkdir(parents=True)
    environment_out.mkdir(parents=True)
    states = {
        "perceptor_0": {}, "actuator_0": {}, "knowledge_0": {},
        "goalgraph_0": {},
        "llreasoner_0": {
            "policy_id": "retained-policy",
            "checkpoint_sha256": "retained-checkpoint",
        },
        "hlreasoner_0": {
            "goal_runs": [], "goal_completion_count": 0, "failure": None,
        },
    }
    for module_id, state in states.items():
        (agent_out / f"exp_agent2_5_0.{module_id}.json").write_text(
            json.dumps(state), encoding="utf-8"
        )
    (environment_out / "exp_env2_5_0.json").write_text(
        "{}", encoding="utf-8"
    )
    for runtime_id in ("exp_agent2_5_0", "exp_env2_5_0"):
        (tmp_path / f"{runtime_id}.log").write_text(
            "[now][INFO]::[fixture]::started\n", encoding="utf-8"
        )
    policy = import_module("mha_exp_level2_bw.exp2_5.policy")
    monkeypatch.setattr(
        policy,
        "artifact_paths",
        lambda: (tmp_path / "manifest", tmp_path / "model"),
    )
    if artifact_mode == "missing":
        monkeypatch.setattr(
            policy,
            "validate_manifest",
            lambda *_args: (_ for _ in ()).throw(FileNotFoundError()),
        )
    else:
        monkeypatch.setattr(
            policy,
            "validate_manifest",
            lambda *_args: {
                "checkpoint_sha256": "current-checkpoint",
                "architecture": "current-policy",
                "training": {}, "certification": {},
            },
        )
        runner = import_module("mha_exp_level2_bw.exp2_5.runner")
        monkeypatch.setattr(
            runner, "check_results", lambda *_args, **_kwargs: False
        )

    report = import_module(
        "mha_exp_level2_bw.exp2_5.reporting"
    ).process_execution_metrics(
        tmp_path,
        expected_executions=[{
            "execution_id": "run-0", "run_id": 0, "factors": {},
        }],
    )

    execution = report["executions"][0]
    assert execution["operationally_valid"] is True
    assert execution["identity"]["policy_artifact"] == {
        "policy_id": "retained-policy",
        "checkpoint_sha256": "retained-checkpoint",
    }
    assert execution["metric_availability"]["artifact_identity"] == {
        "status": "unavailable", "reason": reason,
    }
    assert report["analyses"]["policy_runs"]["eligibility"] == {
        "eligible_execution_ids": [],
        "excluded": [{"execution_id": "run-0", "reasons": [reason]}],
    }
