"""Compact retained-state results for the current 2-5-CR treatment."""

from __future__ import annotations

import json
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from mha_exp_common.batch import normalize_runs
from mha_exp_common.names import HLREASONER, LLREASONER
from mha_exp_common.utils import agent_name, env_name, gather_states, module_name


def process_results(
    runs: int | tuple[int, int] | Sequence[int], exp_path: str | os.PathLike[str],
) -> tuple[bool, dict[str, Any]]:
    """Recheck saved action evidence and report survival and crafting progress."""

    from .runner import EXPERIMENT_ID, check_results, preflight_artifacts
    from .result_validation import time_limit_evidence
    path = Path(exp_path).resolve()
    artifacts = preflight_artifacts()
    results = []
    for run in normalize_runs(runs)[0]:
        agent_id, environment_id = agent_name(run, EXPERIMENT_ID), env_name(run, EXPERIMENT_ID)
        saved = {}
        for runtime_id in (agent_id, environment_id):
            if (path / runtime_id / "out").is_dir():
                saved.update(gather_states(path / runtime_id, True, no_warnings=True))
        agent = saved.get(agent_id, {})
        environment = saved.get(environment_id, {}).get(environment_id, {})
        passed, errors = check_results(agent, environment, run_root=path,
                                       runtime_ids=(agent_id, environment_id), artifact_evidence=artifacts)
        valid, execution_errors = check_results(agent, environment, run_root=path,
                                               runtime_ids=(agent_id, environment_id),
                                               artifact_evidence=artifacts, require_objective=False)
        ll, hl = agent.get(module_name(LLREASONER, 0), {}), agent.get(module_name(HLREASONER, 0), {})
        timeout = time_limit_evidence(agent, environment, path, (agent_id, environment_id)) if valid else None
        raw_execution_errors = execution_errors
        if timeout is not None:
            _, raw_execution_errors = check_results(agent, environment, run_root=path,
                runtime_ids=(agent_id, environment_id), artifact_evidence=artifacts,
                require_objective=False, allow_time_limit=False)
        results.append({
            "run": run, "passed": passed, "errors": errors,
            "environment_seed": environment.get("environment_seed"),
            "enable_eat_cow": hl.get("enable_eat_cow", True),
            "execution_valid": valid, "execution_errors": execution_errors,
            "native_actions": environment.get("native_action_count"),
            "terminal_reason": "time_limit" if timeout is not None else hl.get("terminal_reason"),
            "survived": hl.get("survived"),
            "raw_terminal_reason": hl.get("terminal_reason"), "raw_survived": hl.get("survived"),
            "alive_final": bool(environment.get("inventory", {}).get("health", 0) > 0
                                and environment.get("terminal") is False),
            "time_limit_evidence": timeout, "raw_execution_errors": raw_execution_errors,
            "inventory": environment.get("inventory"), "achievements": environment.get("achievement_counts"),
            "policy_actions": ll.get("policy_action_counts"),
            "primitive_actions": len(hl.get("primitive_decisions", [])),
            "completed_activities": sum(item["status"] == "succeeded" for item in ll.get("activities", []) if item["kind"] == "activity"),
        })
    report = {"experiment": "2-5-CR", "passed": all(row["passed"] for row in results), "runs": results}
    path.mkdir(parents=True, exist_ok=True)
    (path / "summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    return report["passed"], report
