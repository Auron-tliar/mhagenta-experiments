from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from mha_exp_level2_bw.exp2_7.llm import MODEL_POLICY, perform_api_preflight
from mha_exp_level2_bw.exp2_7.runner import (
    LifecycleConfig,
    _bind_paid_run_evidence,
    _build_agent_modules,
    _treatment_manifest,
)


def test_lifecycle_is_one_cutoff_plus_one_bounded_grace() -> None:
    lifecycle = LifecycleConfig()
    assert lifecycle.agent_exec_duration == (
        lifecycle.behavior_cutoff + lifecycle.drain_grace
    )
    assert lifecycle.environment_exec_duration == (
        lifecycle.agent_exec_duration + lifecycle.environment_finalization_grace
    )
    assert lifecycle.behavior_cutoff == 600.0
    assert lifecycle.request_timeout == 60.0
    assert lifecycle.request_timeout < lifecycle.module_term_timeout <= lifecycle.drain_grace
    assert lifecycle.request_timeout <= lifecycle.drain_grace


def test_treatment_freezes_exact_models_and_complete_state_schema() -> None:
    treatment = _treatment_manifest((0, 1), LifecycleConfig())
    assert treatment["models"] == {
        "policy_version": "2-7-bw-nano-role-reasoning-v4",
        "fast": {
            "name": "fast",
            "model": "gpt-5.4-nano-2026-03-17",
            "reasoning_effort": "low",
        },
        "deliberative": {
            "name": "deliberative",
            "model": "gpt-5.4-nano-2026-03-17",
            "reasoning_effort": "high",
        },
        "fallback": None,
    }
    assert len(treatment["module_state_schema_sha256"]) == 64


def test_topology_remains_nine_modules_with_idle_conditional_learners() -> None:
    modules, schemas = _build_agent_modules(
        primary_goals=[
            {
                "goal_id": "primary_0",
                "predicate": "On",
                "arguments": ["B0", "B1"],
                "status": "pending",
                "primary": True,
                "order": 0,
            }
        ],
        value_condition="balanced",
        exchange_name="test",
        lifecycle=LifecycleConfig(),
        max_budget_usd="4.00",
    )
    assert set(modules) == {
        "perceptor",
        "actuator",
        "ll_reasoner",
        "knowledge",
        "hl_reasoner",
        "goal_graph",
        "memory",
        "learners",
    }
    assert len(modules["learners"]) == 2
    assert set(schemas) == {
        "perceptor_0",
        "actuator_0",
        "llreasoner_0",
        "knowledge_0",
        "hlreasoner_0",
        "goalgraph_0",
        "memory_0",
        "learner_0",
        "learner_1",
    }
    for learner in modules["learners"]:
        assert learner.initial_state["initial_call_pending"] is False
        assert learner.initial_state["involvement_status"] == "not_requested"
    cognitive = [modules[role] for role in ("ll_reasoner", "knowledge", "hl_reasoner", "goal_graph", "memory")]
    for module in [*cognitive, *modules["learners"]]:
        expected_effort = "high" if module.module_id in {"knowledge_0", "hlreasoner_0", "memory_0"} else "low"
        assert module.initial_state["selected_model"] == "gpt-5.4-nano-2026-03-17"
        assert module.initial_state["reasoning_effort"] == expected_effort
        assert module.init_kwargs["profile"]["model"] == "gpt-5.4-nano-2026-03-17"
        assert module.init_kwargs["profile"]["reasoning_effort"] == expected_effort
        assert module.init_kwargs["request_timeout"] == 60.0


def test_exact_model_preflight_has_no_candidate_or_fallback_path() -> None:
    calls: list[dict[str, Any]] = []

    class Responses:
        def parse(self, **kwargs: Any) -> Any:
            calls.append(kwargs)
            return SimpleNamespace(
                model=kwargs["model"],
                output_parsed=kwargs["text_format"](ok=True),
            )

    client = SimpleNamespace(responses=Responses(), close=lambda: None)
    result = perform_api_preflight("secret", client_factory=lambda _: client)
    assert result["success"] is True
    assert [item["model"] for item in calls] == [
        MODEL_POLICY.fast.model,
        MODEL_POLICY.deliberative.model,
    ]
    assert calls[0]["reasoning"] == {"effort": "low"}
    assert calls[1]["reasoning"] == {"effort": "high"}


def test_cleanup_evidence_controls_final_matched_inclusion(tmp_path: Path) -> None:
    result_path = tmp_path / "result.json"
    paid_path = tmp_path / "paid.json"
    result_path.write_text(
        json.dumps(
            {
                "architecture_valid": True,
                "cohort_comparable": True,
                "matched_inclusion": True,
                "comparability_checks": {"exact_models": True},
                "evidence": {},
            }
        ),
        encoding="utf-8",
    )
    paid_path.write_text(
        json.dumps(
            {
                "schema_version": "configured-paid-run-v1",
                "cleanup_success": False,
                "failures": [{"stage": "post_run_images"}],
            }
        ),
        encoding="utf-8",
    )
    _bind_paid_run_evidence(result_path=result_path, evidence_path=paid_path)
    result = json.loads(result_path.read_text(encoding="utf-8"))
    assert result["comparability_checks"]["runtime_cleanup_valid"] is False
    assert result["cohort_comparable"] is False
    assert result["matched_inclusion"] is False


def test_recovered_schema_response_is_diagnostic_after_cleanup(tmp_path: Path) -> None:
    """Cleanup binding must not silently turn a recovered call into cohort exclusion."""
    result_path, paid_path = tmp_path / "result.json", tmp_path / "paid.json"
    result_path.write_text(json.dumps({"execution_valid": True, "architecture_valid": True,
        "comparability_checks": {"exact_models": True, "lifecycle_complete": True, "no_schema_repairs": False},
        "evidence": {}}))
    paid_path.write_text(json.dumps({"cleanup_success": True, "failures": []}))
    _bind_paid_run_evidence(result_path=result_path, evidence_path=paid_path)
    result = json.loads(result_path.read_text())
    assert result["matched_inclusion"] is True
    assert result["comparability_checks"]["no_schema_repairs"] is False
