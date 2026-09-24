"""Focused tests for runtime-only credential transport."""

from __future__ import annotations

import asyncio
import inspect
import json
import os
import stat
from pathlib import Path
from types import SimpleNamespace

import pytest
from docker.errors import NotFound
from mha_exp_common import runtime_secret as runtime_secret_module
from mha_exp_common.runtime_secret import (
    RUNTIME_CREDENTIAL_PATH,
    RuntimeSecretCleanupError,
    RuntimeSecretError,
    RuntimeSecretFile,
    RuntimeSecretOrchestrator,
    create_runtime_secret_file,
    load_runtime_credentials,
    write_runtime_secret_diagnostic,
)
from mhagenta.core.orchestrator import AgentEntry, Orchestrator


def _traceback_locals(error: BaseException, function_name: str) -> dict[str, object]:
    traceback = error.__traceback__
    while traceback is not None:
        if traceback.tb_frame.f_code.co_name == function_name:
            return dict(traceback.tb_frame.f_locals)
        traceback = traceback.tb_next
    raise AssertionError(f"No {function_name!r} frame in traceback")


def test_estimated_secret_schema_load_and_cleanup(tmp_path: Path) -> None:
    secret = create_runtime_secret_file(
        encoded_key="encoded-key",
        forbidden_roots=(tmp_path,),
    )
    payload = json.loads(secret.path.read_text(encoding="utf-8"))
    assert payload == {
        "schema_version": "2-7-runtime-credential-v1",
        "encoded_key": "encoded-key",
    }
    assert load_runtime_credentials(
        secret.path,
        budget_source="estimated",
    ) == ("encoded-key", None)
    if os.name != "nt":
        assert stat.S_IMODE(secret.path.parent.stat().st_mode) & 0o777 == 0o700
        assert stat.S_IMODE(secret.path.stat().st_mode) & 0o777 == 0o600
    secret.cleanup()
    secret.cleanup()
    assert secret.deleted
    assert not secret.path.exists()


def test_organization_secret_requires_and_loads_admin_key() -> None:
    secret = create_runtime_secret_file(
        encoded_key="encoded-key",
        encoded_admin_key="encoded-admin",
    )
    try:
        assert load_runtime_credentials(
            secret.path,
            budget_source="organization",
        ) == ("encoded-key", "encoded-admin")
        with pytest.raises(RuntimeSecretError):
            load_runtime_credentials(secret.path, budget_source="estimated")
    finally:
        secret.cleanup()


def test_loader_rejects_wrong_mode_and_extra_fields_without_echoing_values() -> None:
    marker = "encoded-marker-that-must-not-appear"
    secret = create_runtime_secret_file(encoded_key=marker)
    try:
        with pytest.raises(RuntimeSecretError) as missing_admin:
            load_runtime_credentials(secret.path, budget_source="organization")
        payload = json.loads(secret.path.read_text(encoding="utf-8"))
        payload["unexpected"] = marker
        secret.path.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(RuntimeSecretError) as extra_field:
            load_runtime_credentials(secret.path, budget_source="estimated")
        assert marker not in str(missing_admin.value)
        assert marker not in str(extra_field.value)
    finally:
        secret.cleanup()


def test_loader_failure_clears_credential_values_from_traceback() -> None:
    configured = "encoded-configured-loader-marker"
    admin = "encoded-admin-loader-marker"
    secret = create_runtime_secret_file(
        encoded_key=configured,
        encoded_admin_key=admin,
    )
    try:
        with pytest.raises(RuntimeSecretError) as caught:
            load_runtime_credentials(secret.path, budget_source="estimated")
    finally:
        secret.cleanup()

    frame = _traceback_locals(caught.value, "load_runtime_credentials")
    assert frame["payload"] == {}
    assert frame["encoded_key"] == ""
    assert frame["encoded_admin_key"] is None
    assert configured not in repr(frame)
    assert admin not in repr(frame)


def test_creation_failure_removes_factory_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    directory = tmp_path / "factory-directory"

    def make_directory(*args: object, **kwargs: object) -> str:
        directory.mkdir()
        return str(directory)

    monkeypatch.setattr("mha_exp_common.runtime_secret.tempfile.mkdtemp", make_directory)
    monkeypatch.setattr(
        "mha_exp_common.runtime_secret.os.open",
        lambda *args, **kwargs: (_ for _ in ()).throw(PermissionError()),
    )

    with pytest.raises(RuntimeSecretError):
        create_runtime_secret_file(encoded_key="encoded-key")

    assert not directory.exists()


def test_creation_interrupt_cleans_resources_and_clears_payload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    directory = tmp_path / "interrupted-factory"
    interrupt = KeyboardInterrupt("creation interrupted")
    configured = "encoded-configured-marker"
    admin = "encoded-admin-marker"

    def make_directory(*args: object, **kwargs: object) -> str:
        directory.mkdir()
        return str(directory)

    monkeypatch.setattr("mha_exp_common.runtime_secret.tempfile.mkdtemp", make_directory)
    monkeypatch.setattr(
        "mha_exp_common.runtime_secret.os.open",
        lambda *args, **kwargs: (_ for _ in ()).throw(interrupt),
    )

    with pytest.raises(KeyboardInterrupt) as caught:
        create_runtime_secret_file(
            encoded_key=configured,
            encoded_admin_key=admin,
        )

    assert caught.value is interrupt
    assert not directory.exists()
    frame = _traceback_locals(caught.value, "create_runtime_secret_file")
    assert frame["encoded_key"] == ""
    assert frame["encoded_admin_key"] is None
    assert frame["payload"] == {}
    assert configured not in repr(frame)
    assert admin not in repr(frame)


def test_creation_interrupt_with_uncertain_cleanup_is_nonretryable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    directory = tmp_path / "uncertain-factory"
    interrupt = KeyboardInterrupt("creation interrupted")
    original_rmdir = Path.rmdir

    def make_directory(*args: object, **kwargs: object) -> str:
        directory.mkdir()
        return str(directory)

    def fail_rmdir(path: Path) -> None:
        if path == directory:
            raise PermissionError("blocked")
        original_rmdir(path)

    monkeypatch.setattr("mha_exp_common.runtime_secret.tempfile.mkdtemp", make_directory)
    monkeypatch.setattr(
        "mha_exp_common.runtime_secret.os.open",
        lambda *args, **kwargs: (_ for _ in ()).throw(interrupt),
    )
    monkeypatch.setattr(Path, "rmdir", fail_rmdir)

    with pytest.raises(RuntimeSecretCleanupError) as caught:
        create_runtime_secret_file(encoded_key="encoded-marker")

    assert caught.value.__cause__ is interrupt
    assert caught.value.operation == "creation_cleanup"
    assert directory.exists()
    monkeypatch.setattr(Path, "rmdir", original_rmdir)
    directory.rmdir()


def test_cleanup_interrupt_is_preserved_when_absence_is_verified(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret = create_runtime_secret_file(encoded_key="encoded-key")
    directory = secret.path.parent
    interrupt = KeyboardInterrupt("cleanup interrupted")
    original_rmdir = Path.rmdir

    def remove_then_interrupt(path: Path) -> None:
        original_rmdir(path)
        if path == directory:
            raise interrupt

    monkeypatch.setattr(Path, "rmdir", remove_then_interrupt)

    with pytest.raises(KeyboardInterrupt) as caught:
        secret.cleanup()

    assert caught.value is interrupt
    assert secret.deleted is True
    assert not secret.path.exists()
    assert not directory.exists()


def test_cleanup_interrupt_with_remaining_secret_is_nonretryable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret = create_runtime_secret_file(encoded_key="encoded-key")
    interrupt = KeyboardInterrupt("cleanup interrupted")
    original_unlink = Path.unlink

    def interrupt_unlink(path: Path, *args: object, **kwargs: object) -> None:
        if path == secret.path:
            raise interrupt
        original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", interrupt_unlink)
    with pytest.raises(RuntimeSecretCleanupError) as caught:
        secret.cleanup()

    assert caught.value.__cause__ is interrupt
    assert secret.path.exists()
    monkeypatch.setattr(Path, "unlink", original_unlink)
    secret.cleanup()


def test_handle_is_factory_owned_and_properties_are_read_only() -> None:
    with pytest.raises((TypeError, RuntimeSecretError)):
        RuntimeSecretFile()  # type: ignore[call-arg]
    secret = create_runtime_secret_file(encoded_key="encoded-key")
    try:
        with pytest.raises(AttributeError):
            secret.path = Path("replacement")  # type: ignore[misc]
    finally:
        secret.cleanup()


def test_cleanup_fails_closed_on_unexpected_entry() -> None:
    secret = create_runtime_secret_file(encoded_key="encoded-key")
    extra = secret.path.parent / "unexpected"
    extra.write_text("keep", encoding="utf-8")
    with pytest.raises(RuntimeSecretCleanupError) as caught:
        secret.cleanup()
    assert secret.path.exists()
    assert extra.exists()
    assert str(secret.path) not in str(caught.value)
    extra.unlink()
    secret.cleanup()


def test_failure_diagnostic_is_separate_and_contains_exact_path(
    tmp_path: Path,
) -> None:
    error = RuntimeSecretCleanupError(
        "cleanup",
        "PermissionError",
        Path("C:/temporary/secret.json"),
    )
    target = tmp_path / "diagnostics" / "failure.json"
    write_runtime_secret_diagnostic(
        target,
        error=error,
        experiment="2-7-BW",
        run=1,
        attempt_id=None,
    )
    payload = json.loads(target.read_text(encoding="utf-8"))
    assert payload["host_path"] == str(error.diagnostic_path)
    assert payload["error_type"] == "PermissionError"


@pytest.mark.parametrize("gpu_device_ids", ["none", "all", "any", ["0", "1"]])
def test_mount_adapter_matches_signature_and_preserves_gpu_selection(
    tmp_path: Path,
    gpu_device_ids: str | list[str],
) -> None:
    """Keep secret mounts and GPU requests intact when launching an agent."""
    assert inspect.signature(RuntimeSecretOrchestrator._run_agent) == inspect.signature(
        Orchestrator._run_agent
    )
    secret = create_runtime_secret_file(encoded_key="encoded-key")
    calls: list[dict[str, object]] = []

    class Containers:
        def get(self, name: str) -> object:
            raise NotFound("missing")

        def run(self, **kwargs: object) -> object:
            calls.append(kwargs)
            return object()

    orchestrator = object.__new__(RuntimeSecretOrchestrator)
    orchestrator._runtime_secret_mounts = {"agent": secret}
    mounted: list[str] = []
    orchestrator._runtime_secret_mounted_callback = mounted.append
    orchestrator._docker_client = SimpleNamespace(containers=Containers())
    orchestrator._mas_rmq_uri_internal = None
    orchestrator._log_level = Orchestrator.INFO
    orchestrator._log_parser = SimpleNamespace(add_container=lambda agent: None)
    agent = AgentEntry(
        kwargs={},
        agent_id="agent",
        num_copies=1,
        dir=tmp_path / "agent",
        image="image",  # type: ignore[arg-type]
        gpu_device_ids=gpu_device_ids,
    )
    try:
        asyncio.run(orchestrator._run_agent(agent))
        volumes = calls[0]["volumes"]
        assert isinstance(volumes, dict)
        assert volumes[str(secret.path)] == {
            "bind": RUNTIME_CREDENTIAL_PATH,
            "mode": "ro",
        }
        assert mounted == ["agent"]
        requests = calls[0]["device_requests"]
        if gpu_device_ids == "none":
            assert requests is None
        else:
            assert len(requests) == 1
            assert requests[0]["Driver"] == "nvidia"
            assert requests[0]["Capabilities"] == [["gpu"]]
            if isinstance(gpu_device_ids, list):
                assert requests[0]["DeviceIDs"] == gpu_device_ids
            else:
                assert requests[0]["Count"] == (-1 if gpu_device_ids == "all" else 1)
        serialized = json.dumps(calls[0], default=str)
        assert "encoded-key" not in serialized
    finally:
        secret.cleanup()


def test_log_parser_retries_windows_pipe_busy_error() -> None:
    attempts = 0

    class Parser:
        async def run(self) -> None:
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise Exception(231, "CreateFile", "All pipe instances are busy")

        def add_container(self, source: object) -> None:
            pass

    parser = runtime_secret_module._RetryingLogParser(Parser())

    asyncio.run(parser.run())

    assert attempts == 2


def test_log_parser_does_not_retry_other_errors() -> None:
    class Parser:
        async def run(self) -> None:
            raise RuntimeError("docker daemon unavailable")

        def add_container(self, source: object) -> None:
            pass

    parser = runtime_secret_module._RetryingLogParser(Parser())

    with pytest.raises(RuntimeError, match="docker daemon unavailable"):
        asyncio.run(parser.run())


def test_docker_pipe_wait_timeout_is_transient() -> None:
    error = RuntimeError("docker connection failed")
    error.__cause__ = Exception(
        121,
        "WaitNamedPipe",
        "The semaphore timeout period has expired",
    )

    assert runtime_secret_module._docker_pipe_is_busy(error) is True


def test_orchestrator_initialization_retries_windows_pipe_busy_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempts = 0

    def initialize(orchestrator: object, *args: object, **kwargs: object) -> None:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise Exception(231, "CreateFile", "All pipe instances are busy")
        orchestrator._log_parser = SimpleNamespace()

    monkeypatch.setattr(Orchestrator, "__init__", initialize)
    monkeypatch.setattr(runtime_secret_module.time, "sleep", lambda _: None)

    RuntimeSecretOrchestrator(save_dir="unused")

    assert attempts == 2


@pytest.mark.skipif(os.name != "nt", reason="Windows named pipes only")
def test_windows_docker_connections_wait_for_pipe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import win32pipe
    from docker.transport import npipesocket

    calls: list[tuple[object, ...]] = []

    def original(self: object, address: str, retry_count: int = 0) -> str:
        calls.append(("connect", address, retry_count))
        return "connected"

    monkeypatch.setattr(npipesocket.NpipeSocket, "connect", original)
    monkeypatch.setattr(
        win32pipe,
        "WaitNamedPipe",
        lambda address, timeout: calls.append(("wait", address, timeout)),
    )

    runtime_secret_module._install_windows_docker_pipe_wait()
    result = npipesocket.NpipeSocket.connect(object(), "docker-pipe", 2)

    assert result == "connected"
    assert calls == [
        ("wait", "docker-pipe", 30_000),
        ("connect", "docker-pipe", 2),
    ]


def test_mount_confirmation_is_not_emitted_when_container_launch_fails(
    tmp_path: Path,
) -> None:
    secret = create_runtime_secret_file(encoded_key="encoded-key")
    mounted: list[str] = []

    class Containers:
        def get(self, name: str) -> object:
            raise NotFound("missing")

        def run(self, **kwargs: object) -> object:
            raise RuntimeError("launch failed")

    orchestrator = object.__new__(RuntimeSecretOrchestrator)
    orchestrator._runtime_secret_mounts = {"agent": secret}
    orchestrator._runtime_secret_mounted_callback = mounted.append
    orchestrator._docker_client = SimpleNamespace(containers=Containers())
    orchestrator._mas_rmq_uri_internal = None
    orchestrator._log_level = Orchestrator.INFO
    orchestrator._log_parser = SimpleNamespace(add_container=lambda agent: None)
    agent = AgentEntry(
        kwargs={},
        agent_id="agent",
        num_copies=1,
        dir=tmp_path / "agent",
        image="image",  # type: ignore[arg-type]
    )
    try:
        with pytest.raises(RuntimeError, match="launch failed"):
            asyncio.run(orchestrator._run_agent(agent))
        assert mounted == []
    finally:
        secret.cleanup()
