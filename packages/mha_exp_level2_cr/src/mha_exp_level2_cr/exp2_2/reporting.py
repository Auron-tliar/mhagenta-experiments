"""Execution reporting for behavior-based 2-2-CR learning."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from mha_exp_common.execution_metrics import (
    analysis_eligibility,
    eligible_analysis_rows,
    execution_envelope,
    new_report,
    read_json_object,
    write_execution_metrics,
)
from mha_exp_common.metrics import empirical_action_diversity, summarize_numbers

from .treatment import runtime_identity


def _training_shape_error(
    low: Mapping[str, Any],
    learner: Mapping[str, Any],
) -> str | None:
    """Validate persisted counters and the action histogram before aggregation."""

    histogram = low.get("action_histogram")
    if not isinstance(histogram, list):
        return "invalid_root_type"
    if not all(type(value) is int and value >= 0 for value in histogram):
        return "invalid_counter"
    for state, keys in (
        (low, (
            "episodes_started", "completed_episodes", "target_successes",
            "deaths", "truncations", "total_episode_length",
            "transitions_emitted", "training_closed_at_transition",
        )),
        (learner, ("training_updates", "target_syncs", "models_published")),
    ):
        for key in keys:
            value = state.get(key)
            if value is not None and (type(value) is not int or value < 0):
                return "invalid_counter"
    elapsed = low.get("training_closed_at_elapsed_seconds")
    if elapsed is not None and (
        not isinstance(elapsed, (int, float)) or isinstance(elapsed, bool)
        or elapsed < 0
    ):
        return "invalid_counter"
    return None


def _playback_protocol(
    path: Path,
    *,
    checkpoint_sha256: str | None,
    target_achievement: str,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]], str]:
    """Validate one optional playback document and preserve its provenance."""

    document, reason = read_json_object(path)
    if document is None:
        return None, [], reason or "invalid_json"
    protocol = document.get("protocol")
    episodes = document.get("episodes")
    if not isinstance(protocol, Mapping) or not isinstance(episodes, list) or not all(
        isinstance(row, Mapping) for row in episodes
    ):
        return None, [], "invalid_root_type"
    required = {
        "checkpoint_sha256", "checkpoint_metadata", "target_achievement", "seed",
        "requested_episodes", "max_steps", "mode", "fps", "crafter_length",
        "symbolic",
    }
    valid_types = (
        set(protocol) == required
        and isinstance(protocol.get("checkpoint_sha256"), str)
        and isinstance(protocol.get("checkpoint_metadata"), Mapping)
        and isinstance(protocol.get("target_achievement"), str)
        and type(protocol.get("seed")) is int
        and type(protocol.get("requested_episodes")) is int
        and protocol["requested_episodes"] > 0
        and type(protocol.get("max_steps")) is int
        and protocol["max_steps"] > 0
        and protocol.get("mode") in {"gif", "human"}
        and isinstance(protocol.get("fps"), (int, float))
        and not isinstance(protocol.get("fps"), bool)
        and protocol["fps"] > 0
        and type(protocol.get("crafter_length")) is int
        and type(protocol.get("symbolic")) is bool
    )
    rows_valid = all(
        type(row.get("episode")) is int
        and type(row.get("success")) is bool
        and type(row.get("death")) is bool
        and type(row.get("steps")) is int
        and row["steps"] >= 0
        and type(row.get("window_closed")) is bool
        and (row.get("gif") is None or isinstance(row.get("gif"), str))
        for row in episodes
    )
    identity_valid = (
        checkpoint_sha256 is not None
        and protocol.get("checkpoint_sha256") == checkpoint_sha256
    )
    compatible = (
        document.get("schema_version") == "2-2-cr-playback-v1"
        and valid_types
        and rows_valid
        and len(episodes) <= protocol.get("requested_episodes", -1)
        and protocol.get("symbolic") is False
        and protocol.get("crafter_length") == 10_000
        and protocol.get("target_achievement") == target_achievement
        and identity_valid
    )
    serialized = json.dumps(protocol, sort_keys=True, separators=(",", ":"))
    protocol_id = f"playback-{hashlib.sha256(serialized.encode()).hexdigest()}"
    protocol_row = {
        "protocol_id": protocol_id,
        "schema_version": document.get("schema_version"),
        "compatible": compatible,
        "protocol": dict(protocol),
        "source_ref": path.name,
    }
    normalized = [
        {"protocol_id": protocol_id, **dict(row)} for row in episodes
    ] if valid_types and rows_valid else []
    return protocol_row, normalized, "compatible" if compatible else "incompatible_protocol"


def _states(out: Path) -> dict[str, dict[str, Any]]:
    values: dict[str, dict[str, Any]] = {}
    for path in out.glob("*.json"):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(value, dict):
            values[path.stem.rsplit(".", 1)[-1]] = value
    return values


def _runtime_ids(root: Path, run: int) -> tuple[str, str]:
    """Prefer isolated CR IDs while retaining historical pre-isolation reports."""
    agent_id, environment_id, _ = runtime_identity(run)
    legacy = (f"exp_agent2_2_{run}", f"exp_env2_2_{run}")
    if (root / agent_id).is_dir() or not (root / legacy[0]).is_dir():
        return agent_id, environment_id
    return legacy


def process_execution_metrics(
    root: Path,
    *,
    expected_executions: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build fixed-workload training and frozen-evaluation metrics."""

    root = Path(root).resolve()
    discovered = sorted({
        int(path.name.rsplit("_", 1)[-1])
        for pattern in ("exp_agent2_2_cr_*", "exp_agent2_2_*")
        for path in root.glob(pattern)
        if path.name.rsplit("_", 1)[-1].isdigit()
    })
    specs = [dict(item) for item in expected_executions or ()]
    if expected_executions is None:
        specs = [{"execution_id": f"run-{run}", "run_id": run,
                  "factors": {}, "expected": False} for run in discovered]
    report = new_report("2-2-cr", None if expected_executions is None else specs)
    training_rows: list[dict[str, Any]] = []
    evaluation_rows: list[dict[str, Any]] = []
    playback_rows: list[dict[str, Any]] = []
    playback_protocol_rows: list[dict[str, Any]] = []
    for spec in specs:
        run_id = int(spec["run_id"])
        agent_id, environment_id = _runtime_ids(root, run_id)
        out = root / agent_id / "out"
        states = _states(out) if out.is_dir() else {}
        low = states.get("llreasoner_0")
        learner = states.get("learner_0")
        if not isinstance(low, Mapping) or not isinstance(learner, Mapping):
            reason = "required_state_missing" if out.is_dir() else "run_missing"
            report["executions"].append(execution_envelope(
                spec, present=out.is_dir(), readable=False, operationally_valid=False,
                readability_reasons=(reason,), operational_reasons=(reason,),
            ))
            continue
        shape_error = _training_shape_error(low, learner)
        if shape_error is not None:
            report["executions"].append(execution_envelope(
                spec, present=True, readable=False, operationally_valid=False,
                readability_reasons=(shape_error,),
                operational_reasons=(shape_error,),
            ))
            continue
        raw_histogram = low.get("action_histogram")
        histogram = {str(index): count for index, count in enumerate(raw_histogram)
                     if count > 0}
        completed = int(low.get("completed_episodes", 0))
        total_length = int(low.get("total_episode_length", 0))
        training_rows.append({
            "execution_id": spec["execution_id"], "run_id": run_id,
            "episodes_started": low.get("episodes_started"), "completed_episodes": completed,
            "target_successes": low.get("target_successes"), "deaths": low.get("deaths"),
            "truncations": low.get("truncations"),
            "mean_completed_episode_length": None if completed == 0 else total_length / completed,
            "transitions": low.get("transitions_emitted"),
            "updates": learner.get("training_updates"), "target_syncs": learner.get("target_syncs"),
            "models_published": learner.get("models_published"),
            "device": learner.get("device"),
            "torch_version": learner.get("torch_version"),
            "cuda_runtime": learner.get("cuda_runtime"),
            "cuda_available": learner.get("cuda_available"),
            "cuda_device_name": learner.get("cuda_device_name"),
            "training_closed_at_transition": low.get("training_closed_at_transition"),
            "training_closed_at_elapsed_seconds": low.get("training_closed_at_elapsed_seconds"),
            "action_histogram": histogram, "total_actions": sum(histogram.values()),
            "distinct_actions": len(histogram),
            "empirical_action_diversity_bits": empirical_action_diversity(histogram),
            "phase_timestamps": low.get("phase_timestamps"),
            "training_episodes": low.get("training_episodes", []),
            "reward_components": states.get("knowledge_0", {}).get("reward_components", {}),
            "episode_rewards": states.get("knowledge_0", {}).get("episode_rewards", []),
            "milestone_episodes": {
                name: sum(any(event.get("achievement") == name for event in episode.get("achievement_progression", []))
                          for episode in low.get("training_episodes", []))
                for name in ("collect_wood", "place_table", "make_wood_pickaxe", "collect_stone",
                             "make_stone_pickaxe", "collect_coal", "collect_iron", "place_furnace",
                             "make_iron_pickaxe", "collect_diamond")
            },
            "optimization_windows": learner.get("optimization_windows"),
            "learner_loss": learner.get("optimization_windows"),
            "td_error": learner.get("optimization_windows"),
            "gradient_statistics": {"status": "unavailable", "reason": "gradients_not_persisted"},
        })
        for index, case in enumerate(low.get("evaluation_cases", [])):
            if isinstance(case, Mapping):
                evaluation_rows.append({
                    "execution_id": spec["execution_id"],
                    "run_id": run_id,
                    "evaluation_case_id": index,
                    **dict(case),
                })
        from .modules import POLICY_FILENAME
        from .policy import TARGET_ACHIEVEMENT

        model_artifact = learner.get("model_artifact")
        checkpoint_name = (
            model_artifact
            if isinstance(model_artifact, str) and model_artifact
            else POLICY_FILENAME
        )
        checkpoint = out / checkpoint_name
        checkpoint_sha256 = (
            hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            if checkpoint.is_file()
            else None
        )
        playback_files = sorted((out / "policy_evaluation").glob("playback-results.json"))
        playback_status = "playback_missing"
        if playback_files:
            protocol_row, episode_rows, playback_status = _playback_protocol(
                playback_files[0], checkpoint_sha256=checkpoint_sha256,
                target_achievement=TARGET_ACHIEVEMENT.value,
            )
            if protocol_row is not None:
                protocol_row.update({
                    "execution_id": spec["execution_id"], "run_id": run_id,
                    "source_ref": str(playback_files[0].relative_to(root)).replace("\\", "/"),
                })
                playback_protocol_rows.append(protocol_row)
                playback_rows.extend({
                    "execution_id": spec["execution_id"], "run_id": run_id,
                    "protocol_compatible": protocol_row["compatible"], **row,
                } for row in episode_rows)
        logs = {}
        for runtime_id in (agent_id, environment_id):
            path = root / f"{runtime_id}.log"
            if path.is_file():
                logs[runtime_id] = path.read_text(encoding="utf-8", errors="replace").splitlines()
        env_states = _states(root / environment_id / "out")
        environment = next(iter(env_states.values()), {})
        from .runner import check_results_detailed
        required_checker_states = {
            "perceptor_0", "actuator_0", "llreasoner_0", "knowledge_0",
            "memory_0", "learner_0",
        }
        required_log_ids = (agent_id, environment_id)
        checker_available = (
            required_checker_states <= states.keys()
            and bool(env_states)
            and checkpoint.is_file()
            and all(runtime_id in logs and logs[runtime_id]
                    for runtime_id in required_log_ids)
        )
        if checker_available:
            passed, _, checkpoint_info = check_results_detailed(
                states, logs, model_path=checkpoint, environment_state=environment,
                required_log_ids=required_log_ids,
            )
            certificate_status = "passed" if passed else "failed"
        else:
            checkpoint_info = {
                "path": str(checkpoint),
                "sha256": checkpoint_sha256,
            }
            certificate_status = "unavailable"
        operational_reasons = []
        if not required_checker_states <= states.keys() or not env_states:
            operational_reasons.append("required_state_missing")
        if not all(runtime_id in logs and logs[runtime_id]
                   for runtime_id in required_log_ids):
            operational_reasons.append("required_log_missing")
        if not all(type(state.get("contract_errors", 0)) is int
                   and state.get("contract_errors", 0) == 0
                   for state in states.values()):
            operational_reasons.append("contract_errors")
        operational = not operational_reasons
        evaluation_cases = [
            row for row in evaluation_rows
            if row["execution_id"] == spec["execution_id"]
        ]
        evaluation_success = any(row.get("success") is True for row in evaluation_cases)
        task_success = evaluation_success and low.get("phase") == "complete" and certificate_status == "passed"
        report["executions"].append(execution_envelope(
            spec, present=True, readable=True, operationally_valid=operational,
            operational_reasons=operational_reasons,
            certificate_status=certificate_status,
            task_status="success" if task_success else "failure",
            task_reason=None if task_success else ("incomplete_or_invalid_evaluation" if evaluation_success
                                                   else "no_frozen_evaluation_success"),
            termination_reason=(
                "evaluation_complete" if low.get("phase") == "complete"
                else "time_limit"
            ),
            identity={"checkpoint": checkpoint_info,
                      "treatment": low.get("treatment")},
            metric_availability={"training": "existing", "action_diversity": "derived",
                                 "learner_loss": "existing",
                                 "frozen_evaluation": "existing",
                                 "certificate": (
                                     "existing" if checker_available
                                     else {"status": "unavailable", "reason": "checker_input_missing"}
                                 ),
                                 "playback": (
                                     "existing" if playback_status == "compatible"
                                     else {"status": "unavailable", "reason": playback_status}
                                 )},
            source_refs=(str(next(out.glob("*.llreasoner_0.json")).relative_to(root)).replace("\\", "/"),),
        ))
    training_eligibility = analysis_eligibility(
        report["executions"], required_metrics=("training",),
    )
    playback_eligibility = analysis_eligibility(
        report["executions"], required_metrics=("playback",),
    )
    eligible_training = eligible_analysis_rows(
        training_rows, training_eligibility
    )
    eligible_evaluation = eligible_analysis_rows(
        evaluation_rows, training_eligibility
    )
    eligible_playback = eligible_analysis_rows(
        playback_rows, playback_eligibility
    )
    eligible_protocols = eligible_analysis_rows(
        playback_protocol_rows, playback_eligibility
    )
    report["analyses"] = {
        "training_runs": {"unit": "execution", "nesting": "none",
                          "eligibility": training_eligibility, "rows": training_rows,
                          "summary": {"completed_episodes": summarize_numbers(
                              [row["completed_episodes"] for row in eligible_training], include_iqr=True),
                                      "target_successes": summarize_numbers(
                              [row["target_successes"] for row in eligible_training], include_iqr=True)}},
        "frozen_evaluation": {
            "unit": "evaluation_episode",
            "nesting": "episodes_within_execution",
            "eligibility": training_eligibility,
            "rows": evaluation_rows,
            "summary": {
                "count": len(eligible_evaluation),
                "raw_descriptive_count": len(evaluation_rows),
                "successes": sum(
                    row.get("success") is True for row in eligible_evaluation
                ),
                "returns": summarize_numbers(
                    [row.get("return") for row in eligible_evaluation],
                    include_iqr=True,
                ),
                "lengths": summarize_numbers(
                    [row.get("length") for row in eligible_evaluation],
                    include_iqr=True,
                ),
            },
        },
        "playback_episodes": {"unit": "playback_episode", "nesting": "episodes_within_protocol_within_execution",
                              "eligibility": playback_eligibility, "rows": playback_rows,
                              "summary": {"count": len(eligible_playback),
                                          "raw_descriptive_count": len(playback_rows)}},
        "playback_protocols": {
            "unit": "playback_protocol",
            "nesting": "one_protocol_within_execution",
            "eligibility": playback_eligibility,
            "rows": playback_protocol_rows,
            "summary": {
                "count": len(eligible_protocols),
                "raw_descriptive_count": len(playback_protocol_rows),
                "compatible": sum(row["compatible"] for row in eligible_protocols),
            },
        },
    }
    write_execution_metrics(root, report)
    return report
