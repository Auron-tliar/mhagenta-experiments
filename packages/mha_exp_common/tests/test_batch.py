"""Focused tests for shared experiment batch behavior."""

from pathlib import Path
from types import SimpleNamespace

import pytest
from mha_exp_common import batch
from mha_exp_common.openai_budget import (
    TemporaryCredentialError,
    TemporaryCredentialSecurityError,
)
from mha_exp_common.runtime_secret import RuntimeSecretCleanupError


@pytest.mark.parametrize("removed", [True, False])
def test_exact_cleanup_interrupt_requires_verified_absence(
    monkeypatch: pytest.MonkeyPatch,
    removed: bool,
) -> None:
    target = "exp_agent"
    present = True
    interrupt = KeyboardInterrupt("docker interrupted")

    def run(command: list[str], **kwargs: object) -> SimpleNamespace:
        nonlocal present
        if "--format" in command:
            return SimpleNamespace(
                returncode=0,
                stdout=f"{target}\n" if present else "",
            )
        if command[:4] == ["docker", "container", "rm", "--force"]:
            if removed:
                present = False
            raise interrupt
        raise AssertionError(command)

    monkeypatch.setattr(batch.subprocess, "run", run)

    if removed:
        with pytest.raises(KeyboardInterrupt) as caught:
            batch.cleanup_run_containers(target, "absent-environment", phase="post_run")
        assert caught.value is interrupt
    else:
        with pytest.raises(batch.DockerCleanupError) as caught:
            batch.cleanup_run_containers(target, "absent-environment", phase="post_run")
        assert caught.value.__cause__ is interrupt
        assert caught.value.evidence["verified_absent"] == ["absent-environment"]


def test_process_only_preserves_existing_results(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    marker = tmp_path / "saved-result.json"
    marker.write_text("saved", encoding="utf-8")
    monkeypatch.setattr(
        batch,
        "cleanup_exp_docker",
        lambda **kwargs: pytest.fail("Docker cleanup must not run"),
    )
    monkeypatch.setattr(
        batch.shutil,
        "rmtree",
        lambda *args, **kwargs: pytest.fail("Existing output must not be deleted"),
    )

    available = batch.run_batch(
        experiment_id="test",
        title="TEST",
        runs=1,
        exp_path=tmp_path,
        runner=lambda *args: pytest.fail("Execution callback must not run"),
        process_only=True,
    )

    assert available is True
    assert marker.read_text(encoding="utf-8") == "saved"


@pytest.mark.parametrize("create_directory", [False, True])
def test_process_only_reports_missing_results_without_creating_output(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    create_directory: bool,
) -> None:
    output = tmp_path / "output"
    if create_directory:
        output.mkdir()

    available = batch.run_batch(
        experiment_id="test",
        title="TEST",
        runs=1,
        exp_path=output,
        runner=lambda *args: pytest.fail("Execution callback must not run"),
        process_only=True,
    )

    assert available is False
    assert output.exists() is create_directory
    assert "No existing results found" in capsys.readouterr().out


def test_normal_batch_behavior_is_unchanged(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    marker = tmp_path / "old-result.json"
    marker.write_text("old", encoding="utf-8")
    cleanup_calls: list[str] = []
    runner_calls: list[int] = []
    monkeypatch.setattr(
        batch,
        "cleanup_exp_docker",
        lambda prefix: cleanup_calls.append(prefix),
    )

    result = batch.run_batch(
        experiment_id="test",
        title="TEST",
        runs=2,
        exp_path=tmp_path,
        runner=lambda run, path, version: runner_calls.append(run) or True,
    )

    assert result is True
    assert not marker.exists()
    assert cleanup_calls == ["exp_", "exp_"]
    assert runner_calls == [0, 1]


def test_cleanup_can_be_delegated_to_runner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        batch,
        "cleanup_exp_docker",
        lambda **kwargs: pytest.fail("Shared cleanup must be disabled"),
    )
    runner_calls: list[int] = []

    result = batch.run_batch(
        experiment_id="test",
        title="TEST",
        runs=2,
        exp_path=tmp_path,
        runner=lambda run, path, version: runner_calls.append(run) or True,
        cleanup_before_run=False,
    )

    assert result is True
    assert runner_calls == [0, 1]


def test_stop_on_failed_result_does_not_start_next_run(tmp_path: Path) -> None:
    """A failed certificate stops an explicitly fail-fast batch."""
    calls: list[int] = []

    def runner(run: int, path: Path, version: str) -> bool:
        calls.append(run)
        return False

    with pytest.raises(RuntimeError, match="result check failed"):
        batch.run_batch(
            experiment_id="test", title="TEST", runs=2, exp_path=tmp_path,
            runner=runner, cleanup_before_run=False, stop_on_error=True,
        )
    assert calls == [0]


def test_stop_on_error_preserves_retry_count_then_raises(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(batch, "RETRIES", 2)
    calls = 0

    def runner(*args: object) -> bool:
        nonlocal calls
        calls += 1
        raise ValueError("failed")

    with pytest.raises(ValueError, match="failed"):
        batch.run_batch(
            experiment_id="test",
            title="TEST",
            runs=1,
            exp_path=tmp_path,
            runner=runner,
            cleanup_before_run=False,
            stop_on_error=True,
        )
    assert calls == 2


def test_safe_temporary_credential_failure_can_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(batch, "RETRIES", 2)
    calls = 0

    def runner(*args: object) -> bool:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise TemporaryCredentialError(
                "not created",
                service_account_name="account",
                cleanup_status="not_created",
            )
        return True

    batch.run_batch(
        experiment_id="test",
        title="TEST",
        runs=1,
        exp_path=tmp_path,
        runner=runner,
        cleanup_before_run=False,
        stop_on_error=True,
    )
    assert calls == 2


@pytest.mark.parametrize(
    "error",
    [
        batch.DockerCleanupError("post_run", {}),
        TemporaryCredentialSecurityError(
            "possible leak",
            service_account_name="account",
            possible_leak=True,
            cleanup_status="failed",
        ),
        RuntimeSecretCleanupError(
            "creation_cleanup",
            "PermissionError",
            Path("C:/temporary/mhagent-openai.json"),
        ),
    ],
)
def test_nonretryable_failure_stops_immediately(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    error: Exception,
) -> None:
    monkeypatch.setattr(batch, "RETRIES", 3)
    calls = 0

    def runner(*args: object) -> bool:
        nonlocal calls
        calls += 1
        raise error

    with pytest.raises(type(error)):
        batch.run_batch(
            experiment_id="test",
            title="TEST",
            runs=2,
            exp_path=tmp_path,
            runner=runner,
            cleanup_before_run=False,
        )
    assert calls == 1
