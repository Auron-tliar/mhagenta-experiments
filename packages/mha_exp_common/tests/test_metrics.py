from __future__ import annotations

import math
from pathlib import Path

import pytest
from mha_exp_common.execution_metrics import (
    analysis_eligibility,
    eligible_analysis_rows,
    execution_envelope,
    execution_spec,
    finalize_report,
    new_report,
    nullable_certificate_status,
    read_json_object,
)
from mha_exp_common.metrics import (
    empirical_action_diversity,
    summarize_numbers,
    wilson_interval,
)


def test_summarize_numbers_tracks_missing_and_small_samples() -> None:
    assert summarize_numbers([]) == {
        "count": 0, "missing": 0, "median": None, "min": None, "max": None,
    }
    assert summarize_numbers([None, None]) == {
        "count": 0, "missing": 2, "median": None, "min": None, "max": None,
    }
    assert summarize_numbers([1, None, 4, 2]) == {
        "count": 3, "missing": 1, "median": 2.0, "min": 1.0, "max": 4.0,
    }
    assert summarize_numbers([1, 2, 3, 4])["median"] == 2.5


def test_summarize_numbers_gates_iqr_and_p95() -> None:
    small = summarize_numbers([1, 2, 3], include_iqr=True, p95_min_count=4)
    assert "iqr" not in small
    assert "p95" not in small

    summary = summarize_numbers(
        [1, 2, 3, 4], include_iqr=True, p95_min_count=4,
    )
    assert summary["q1"] == pytest.approx(1.75)
    assert summary["q3"] == pytest.approx(3.25)
    assert summary["iqr"] == pytest.approx(1.5)
    assert summary["p95"] == pytest.approx(3.85)


@pytest.mark.parametrize("values", [[math.inf], [math.nan], [True], ["1"]])
def test_summarize_numbers_rejects_invalid_values(values: list[object]) -> None:
    with pytest.raises((TypeError, ValueError)):
        summarize_numbers(values)  # type: ignore[arg-type]


def test_wilson_interval_covers_edge_and_mixed_proportions() -> None:
    assert wilson_interval(0, 0) is None
    zero = wilson_interval(0, 10)
    all_success = wilson_interval(10, 10)
    mixed = wilson_interval(5, 10)
    assert zero is not None and zero[0] == pytest.approx(0.0)
    assert all_success is not None and all_success[1] == pytest.approx(1.0)
    assert mixed is not None and mixed[0] < 0.5 < mixed[1]


@pytest.mark.parametrize("successes,total", [(-1, 1), (2, 1), (0, -1)])
def test_wilson_interval_rejects_invalid_counts(successes: int, total: int) -> None:
    with pytest.raises(ValueError):
        wilson_interval(successes, total)


def test_wilson_interval_rejects_boolean_counts() -> None:
    with pytest.raises(TypeError):
        wilson_interval(True, 1)


def test_empirical_action_diversity_uses_observed_counts() -> None:
    assert empirical_action_diversity({}) is None
    assert empirical_action_diversity({"left": 4}) == pytest.approx(0.0)
    assert empirical_action_diversity({"left": 2, "right": 2}) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        empirical_action_diversity({"left": -1})


def test_execution_envelopes_keep_classifications_independent() -> None:
    spec = execution_spec(2, factors={"condition": "balanced"})
    report = new_report("2-7-bw", [spec])
    report["executions"].append(execution_envelope(
        spec,
        present=True,
        readable=True,
        operationally_valid=True,
        certificate_status="failed",
        task_status="success",
        termination_reason="task_completed",
    ))
    finalize_report(report)
    assert report["execution_accounting"] == {
        "expected": 1, "present": 1, "readable": 1,
        "operationally_valid": 1,
    }
    assert report["executions"][0]["certificate_passed"] is False
    assert report["executions"][0]["task_outcome"]["status"] == "success"


def test_json_object_loading_preserves_readability_reasons(tmp_path: Path) -> None:
    missing = tmp_path / "missing.json"
    assert read_json_object(missing) == (None, "required_file_missing")
    missing.write_text("{", encoding="utf-8")
    assert read_json_object(missing) == (None, "invalid_json")
    missing.write_text("[]", encoding="utf-8")
    assert read_json_object(missing) == (None, "invalid_root_type")
    missing.write_text('{"value": 1}', encoding="utf-8")
    assert read_json_object(missing) == ({"value": 1}, None)


def test_nullable_certificate_mapping_does_not_invent_failure() -> None:
    assert nullable_certificate_status(True) == "passed"
    assert nullable_certificate_status(False) == "failed"
    assert nullable_certificate_status(None) == "unavailable"
    assert nullable_certificate_status(0) == "unavailable"


def test_analysis_eligibility_separates_raw_rows_from_denominator() -> None:
    executions = [
        execution_envelope(
            execution_spec(0), present=True, readable=True,
            operationally_valid=True,
        ),
        execution_envelope(
            execution_spec(1), present=True, readable=True,
            operationally_valid=False,
            operational_reasons=("fatal_runtime_error",),
        ),
        execution_envelope(
            execution_spec(2), present=False, readable=False,
            operationally_valid=False,
            readability_reasons=("run_missing",),
            operational_reasons=("run_missing",),
        ),
    ]

    eligibility = analysis_eligibility(executions)

    assert eligibility == {
        "eligible_execution_ids": ["run-0"],
        "excluded": [
            {"execution_id": "run-1", "reasons": ["fatal_runtime_error"]},
            {"execution_id": "run-2", "reasons": ["run_missing"]},
        ],
    }
    assert eligible_analysis_rows([
        {"execution_id": "run-0", "value": 1},
        {"execution_id": "run-1", "value": 99},
    ], eligibility) == [{"execution_id": "run-0", "value": 1}]


def test_analysis_eligibility_requires_named_metric_evidence() -> None:
    executions = [{
        "execution_id": "run-0",
        "readable": True,
        "operationally_valid": True,
        "readability_reasons": [],
        "operational_reasons": [],
        "metric_availability": {
            "playback": {
                "status": "unavailable", "reason": "playback_missing",
            },
        },
    }]

    eligibility = analysis_eligibility(
        executions, required_metrics=("playback",),
    )

    assert eligibility == {
        "eligible_execution_ids": [],
        "excluded": [{
            "execution_id": "run-0", "reasons": ["playback_missing"],
        }],
    }
