"""Common JSON envelope helpers for additive Level 2 execution reports."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "level-2-execution-metrics-v1"
OUTPUT_FILENAME = "execution-metrics.json"


def read_json_object(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    """Read a JSON object and return a stable readability reason on failure."""

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except OSError:
        return None, "required_file_missing"
    except json.JSONDecodeError:
        return None, "invalid_json"
    if not isinstance(value, dict):
        return None, "invalid_root_type"
    return value, None


def nullable_certificate_status(value: Any) -> str:
    """Map a retained strict Boolean certificate without inventing an outcome."""

    if value is True:
        return "passed"
    if value is False:
        return "failed"
    return "unavailable"


def analysis_eligibility(
    executions: Sequence[Mapping[str, Any]],
    *,
    required_metrics: Sequence[str] = (),
) -> dict[str, Any]:
    """Classify executions with valid operation and required metric evidence."""

    eligible: list[str] = []
    excluded: list[dict[str, Any]] = []
    for execution in executions:
        execution_id = str(execution["execution_id"])
        reasons = list(dict.fromkeys([
            *execution.get("readability_reasons", ()),
            *execution.get("operational_reasons", ()),
        ]))
        if (
            execution.get("readable") is True
            and execution.get("operationally_valid") is True
        ):
            availability = execution.get("metric_availability", {})
            for metric in required_metrics:
                value = availability.get(metric) if isinstance(
                    availability, Mapping
                ) else None
                status = (
                    value.get("status") if isinstance(value, Mapping)
                    else value
                )
                if status in {"existing", "derived", "instrumented"}:
                    continue
                reason = (
                    value.get("reason") if isinstance(value, Mapping)
                    else None
                )
                reasons.append(str(reason or f"{metric}_unavailable"))
            if not reasons:
                eligible.append(execution_id)
                continue
        excluded.append({
            "execution_id": execution_id,
            "reasons": list(dict.fromkeys(reasons)) or [
                "operationally_invalid"
            ],
        })
    return {"eligible_execution_ids": eligible, "excluded": excluded}


def eligible_analysis_rows(
    rows: Sequence[Mapping[str, Any]],
    eligibility: Mapping[str, Any],
) -> list[Mapping[str, Any]]:
    """Return rows belonging to the analysis's eligible execution set."""

    eligible_ids = set(eligibility.get("eligible_execution_ids", ()))
    return [row for row in rows if row.get("execution_id") in eligible_ids]


def execution_spec(
    run_id: int,
    *,
    execution_id: str | None = None,
    factors: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build one expected execution-cell identity."""

    return {
        "execution_id": execution_id or f"run-{run_id}",
        "run_id": run_id,
        "factors": dict(factors or {}),
    }


def treatment_identity_reasons(
    spec: Mapping[str, Any],
    actual: Any,
    *,
    keys: Sequence[str] = ("protocol_version", "treatment_id"),
) -> list[str]:
    """Return stable reasons when retained treatment identity differs."""

    expected = spec.get("factors", {})
    compared = [key for key in keys if isinstance(expected, Mapping) and key in expected]
    if not compared:
        return []
    if not isinstance(actual, Mapping):
        return ["treatment_identity_missing"]
    if any(actual.get(key) != expected.get(key) for key in compared):
        return ["treatment_identity_mismatch"]
    return []


def new_report(
    experiment_id: str,
    expected_executions: Sequence[Mapping[str, Any]] | None,
) -> dict[str, Any]:
    """Create an empty deterministic execution-metrics document."""

    expected = (
        None
        if expected_executions is None
        else [dict(item) for item in expected_executions]
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": experiment_id,
        "expected_executions": expected,
        "completeness_assessable": expected is not None,
        "execution_accounting": {
            "expected": None if expected is None else len(expected),
            "present": 0,
            "readable": 0,
            "operationally_valid": 0,
        },
        "executions": [],
        "analyses": {},
        "preparation_artifacts": [],
    }


def execution_envelope(
    spec: Mapping[str, Any],
    *,
    present: bool,
    readable: bool,
    operationally_valid: bool,
    readability_reasons: Sequence[str] = (),
    operational_reasons: Sequence[str] = (),
    certificate_status: str = "unavailable",
    task_status: str = "unobserved",
    task_reason: str | None = None,
    termination_reason: str = "unknown",
    identity: Mapping[str, Any] | None = None,
    metric_availability: Mapping[str, Any] | None = None,
    source_refs: Sequence[str] = (),
) -> dict[str, Any]:
    """Build one execution envelope without conflating classification axes."""

    if certificate_status not in {
        "passed", "failed", "not_evaluated", "unavailable",
    }:
        raise ValueError("Invalid certificate status.")
    certificate_passed = (
        True if certificate_status == "passed"
        else False if certificate_status == "failed"
        else None
    )
    return {
        "execution_id": spec["execution_id"],
        "run_id": spec["run_id"],
        "expected": True if spec.get("expected", True) else None,
        "present": present,
        "readable": readable,
        "readability_reasons": list(readability_reasons),
        "operationally_valid": operationally_valid,
        "operational_reasons": list(operational_reasons),
        "certificate_status": certificate_status,
        "certificate_passed": certificate_passed,
        "task_outcome": {"status": task_status, "reason": task_reason},
        "termination": {"reason": termination_reason},
        "identity": {**dict(spec.get("factors", {})), **dict(identity or {})},
        "metric_availability": dict(metric_availability or {}),
        "source_refs": list(source_refs),
    }


def finalize_report(report: dict[str, Any]) -> dict[str, Any]:
    """Recompute execution-accounting counts after envelopes are populated."""

    executions = report["executions"]
    report["execution_accounting"].update({
        "present": sum(item["present"] for item in executions),
        "readable": sum(item["readable"] for item in executions),
        "operationally_valid": sum(
            item["operationally_valid"] for item in executions
        ),
    })
    return report


def write_execution_metrics(root: Path, report: dict[str, Any]) -> Path:
    """Finalize and write an execution report using stable JSON formatting."""

    root.mkdir(parents=True, exist_ok=True)
    target = root / OUTPUT_FILENAME
    target.write_text(
        json.dumps(finalize_report(report), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return target
