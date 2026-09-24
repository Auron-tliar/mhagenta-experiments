from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from mha_exp_common import paid_run
from mha_exp_common.batch import NonRetryableBatchError
from mha_exp_common.runtime_secret import RuntimeSecretCleanupError


class FakeSecret:
    def __init__(self, calls: list[str]) -> None:
        self.calls = calls
        self.deleted = False

    def cleanup(self) -> None:
        self.calls.append("secret_cleanup")
        self.deleted = True


def _cleanup(calls: list[str], resource: str):
    def run(agent_id: str, environment_id: str, *, phase: str) -> dict[str, Any]:
        calls.append(f"{phase}_{resource}")
        return {"status": "succeeded", "phase": phase, "resource": resource}

    return run


def test_configured_paid_run_has_one_exact_redacted_lifecycle(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[str] = []
    secret = FakeSecret(calls)
    monkeypatch.setattr(
        paid_run, "cleanup_run_containers", _cleanup(calls, "containers")
    )
    monkeypatch.setattr(paid_run, "cleanup_run_images", _cleanup(calls, "images"))

    def create(**kwargs: Any) -> FakeSecret:
        assert kwargs["encoded_key"] == "encoded-secret"
        calls.append("secret_create")
        return secret

    monkeypatch.setattr(paid_run, "create_runtime_secret_file", create)

    def body(handle: Any, mounted: Any) -> bool:
        assert handle is secret
        calls.append("body")
        mounted()
        return True

    evidence_path = tmp_path / "paid.json"
    assert paid_run.configured_paid_run(
        encoded_key="encoded-secret",
        evidence_path=evidence_path,
        agent_id="agent-0",
        environment_id="environment-0",
        forbidden_roots=(tmp_path,),
        body=body,
    )
    assert calls == [
        "pre_run_containers",
        "pre_run_images",
        "secret_create",
        "body",
        "post_run_containers",
        "secret_cleanup",
        "post_run_images",
    ]
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    assert evidence["cleanup_success"] is True
    assert evidence["runtime_secret"] == {
        "created": True,
        "mounted": True,
        "deleted": True,
    }
    assert "encoded-secret" not in evidence_path.read_text(encoding="utf-8")


def test_configured_paid_run_attempts_post_cleanup_after_body_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(
        paid_run, "cleanup_run_containers", _cleanup(calls, "containers")
    )
    monkeypatch.setattr(paid_run, "cleanup_run_images", _cleanup(calls, "images"))
    monkeypatch.setattr(
        paid_run,
        "create_runtime_secret_file",
        lambda **_: FakeSecret(calls),
    )

    def body(secret: Any, mounted: Any) -> bool:
        calls.append("body")
        mounted()
        raise ValueError("sensitive detail")

    evidence_path = tmp_path / "paid.json"
    with pytest.raises(ValueError, match="sensitive detail"):
        paid_run.configured_paid_run(
            encoded_key="encoded-secret",
            evidence_path=evidence_path,
            agent_id="agent-0",
            environment_id="environment-0",
            forbidden_roots=(tmp_path,),
            body=body,
        )
    assert calls[-3:] == [
        "post_run_containers",
        "secret_cleanup",
        "post_run_images",
    ]
    serialized = evidence_path.read_text(encoding="utf-8")
    assert "sensitive detail" not in serialized
    assert json.loads(serialized)["body"]["status"] == "failed"


def test_configured_paid_run_preserves_interrupt_and_finishes_cleanup(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(
        paid_run, "cleanup_run_containers", _cleanup(calls, "containers")
    )
    monkeypatch.setattr(paid_run, "cleanup_run_images", _cleanup(calls, "images"))
    monkeypatch.setattr(
        paid_run, "create_runtime_secret_file", lambda **_: FakeSecret(calls)
    )

    def interrupt(secret: Any, mounted: Any) -> bool:
        calls.append("body")
        mounted()
        raise KeyboardInterrupt("interrupt detail")

    evidence_path = tmp_path / "paid.json"
    with pytest.raises(KeyboardInterrupt):
        paid_run.configured_paid_run(
            encoded_key="encoded-secret",
            evidence_path=evidence_path,
            agent_id="agent-0",
            environment_id="environment-0",
            forbidden_roots=(tmp_path,),
            body=interrupt,
        )
    assert calls[-3:] == [
        "post_run_containers",
        "secret_cleanup",
        "post_run_images",
    ]
    assert "interrupt detail" not in evidence_path.read_text(encoding="utf-8")


def test_configured_paid_run_secret_cleanup_failure_is_nonretryable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(
        paid_run, "cleanup_run_containers", _cleanup(calls, "containers")
    )
    monkeypatch.setattr(paid_run, "cleanup_run_images", _cleanup(calls, "images"))

    class BadSecret(FakeSecret):
        def cleanup(self) -> None:
            calls.append("secret_cleanup")
            raise RuntimeSecretCleanupError("delete", "OSError", tmp_path)

    monkeypatch.setattr(
        paid_run, "create_runtime_secret_file", lambda **_: BadSecret(calls)
    )
    evidence_path = tmp_path / "paid.json"
    with pytest.raises(NonRetryableBatchError):
        paid_run.configured_paid_run(
            encoded_key="encoded-secret",
            evidence_path=evidence_path,
            agent_id="agent-0",
            environment_id="environment-0",
            forbidden_roots=(tmp_path,),
            body=lambda secret, mounted: (mounted(), True)[1],
        )
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    assert evidence["cleanup_success"] is False
    assert calls[-2:] == ["secret_cleanup", "post_run_images"]


def test_configured_paid_run_cleanup_failure_skips_body_but_attempts_safe_remainder(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[str] = []

    def containers(agent_id: str, environment_id: str, *, phase: str) -> dict[str, Any]:
        calls.append(f"{phase}_containers")
        if phase == "pre_run":
            raise NonRetryableBatchError("unsafe cleanup")
        return {"status": "succeeded"}

    monkeypatch.setattr(paid_run, "cleanup_run_containers", containers)
    monkeypatch.setattr(paid_run, "cleanup_run_images", _cleanup(calls, "images"))
    monkeypatch.setattr(
        paid_run,
        "create_runtime_secret_file",
        lambda **_: pytest.fail("secret must not be created after failed pre-cleanup"),
    )
    evidence_path = tmp_path / "paid.json"
    with pytest.raises(NonRetryableBatchError):
        paid_run.configured_paid_run(
            encoded_key="encoded-secret",
            evidence_path=evidence_path,
            agent_id="agent-0",
            environment_id="environment-0",
            forbidden_roots=(tmp_path,),
            body=lambda *_: pytest.fail("body must not run"),
        )
    assert calls == [
        "pre_run_containers",
        "pre_run_images",
        "post_run_containers",
        "post_run_images",
    ]
