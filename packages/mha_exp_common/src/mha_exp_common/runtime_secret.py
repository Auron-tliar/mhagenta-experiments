"""Runtime-only credential files and the MHAgentA 1.4.12 mount adapter."""

import asyncio
import json
import os
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from docker.errors import NotFound
from mhagenta.core.orchestrator import AgentEntry, Orchestrator

from mha_exp_common.batch import NonRetryableBatchError

RUNTIME_CREDENTIAL_SCHEMA = "2-7-runtime-credential-v1"
RUNTIME_CREDENTIAL_PATH = "/run/secrets/mhagent-openai.json"
_CREDENTIAL_FILENAME = "mhagent-openai.json"
_FACTORY_SENTINEL = object()
_DOCKER_PIPE_BUSY = 231
_DOCKER_PIPE_TIMEOUT = 121
_LOG_PIPE_RETRY_LIMIT = 60
_LOG_PIPE_RETRY_SECONDS = 0.25
_DOCKER_PIPE_WAIT_MILLISECONDS = 30_000


class RuntimeSecretError(RuntimeError):
    """Base error for sanitized runtime-secret failures."""


class RuntimeSecretCleanupError(NonRetryableBatchError):
    """Report a host runtime-secret file that could not be removed safely."""

    __slots__ = ("_diagnostic_path", "_error_type", "_operation")

    def __init__(
        self,
        operation: str,
        error_type: str,
        diagnostic_path: Path,
    ) -> None:
        super().__init__(f"runtime-secret {operation} failed ({error_type})")
        self._operation = operation
        self._error_type = error_type
        self._diagnostic_path = diagnostic_path

    @property
    def operation(self) -> str:
        """Return the failed cleanup operation."""

        return self._operation

    @property
    def error_type(self) -> str:
        """Return the sanitized underlying error type."""

        return self._error_type

    @property
    def diagnostic_path(self) -> Path:
        """Return the exact host path needed for manual remediation."""

        return self._diagnostic_path


def _docker_pipe_is_busy(error: BaseException) -> bool:
    """Return whether an exception chain contains Windows pipe-busy error 231."""

    current: BaseException | None = error
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if current.args and current.args[0] in {
            _DOCKER_PIPE_BUSY,
            _DOCKER_PIPE_TIMEOUT,
        }:
            return True
        current = current.__cause__ or current.__context__
    return False


def _install_windows_docker_pipe_wait() -> None:
    """Make docker-py wait for an available Docker Desktop pipe on Windows."""

    if os.name != "nt":
        return
    import win32pipe
    from docker.transport.npipesocket import NpipeSocket

    original = NpipeSocket.connect
    if getattr(original, "_mha_waits_for_pipe", False):
        return

    def connect(self: Any, address: str, retry_count: int = 0) -> Any:
        win32pipe.WaitNamedPipe(address, _DOCKER_PIPE_WAIT_MILLISECONDS)
        return original(self, address, retry_count)

    setattr(connect, "_mha_waits_for_pipe", True)
    setattr(NpipeSocket, "connect", connect)


class _RetryingLogParser:
    """Retry transient Docker Desktop named-pipe exhaustion while collecting logs."""

    def __init__(self, delegate: Any) -> None:
        self._delegate = delegate

    def add_container(self, source: Any) -> None:
        """Register a container with the wrapped MHAgentA log parser."""

        self._delegate.add_container(source)

    async def run(self) -> None:
        """Run the parser, retrying only Windows pipe-busy failures."""

        retries = 0
        while True:
            try:
                await self._delegate.run()
                return
            except Exception as error:
                if not _docker_pipe_is_busy(error):
                    raise
                retries += 1
                if retries >= _LOG_PIPE_RETRY_LIMIT:
                    raise
                await asyncio.sleep(_LOG_PIPE_RETRY_SECONDS)


def _identity(path: Path) -> tuple[int, int]:
    stat = path.lstat()
    return stat.st_dev, stat.st_ino


class RuntimeSecretFile:
    """Factory-owned handle for one exact runtime credential file."""

    __slots__ = (
        "_container_path",
        "_deleted",
        "_directory",
        "_directory_identity",
        "_file_identity",
        "_path",
    )

    def __init__(
        self,
        sentinel: object,
        *,
        directory: Path,
        path: Path,
        directory_identity: tuple[int, int],
        file_identity: tuple[int, int],
    ) -> None:
        if sentinel is not _FACTORY_SENTINEL:
            raise RuntimeSecretError("runtime-secret handles require the factory")
        self._directory = directory
        self._path = path
        self._container_path = RUNTIME_CREDENTIAL_PATH
        self._deleted = False
        self._directory_identity = directory_identity
        self._file_identity = file_identity

    @property
    def path(self) -> Path:
        """Return the exact host credential-file path."""

        return self._path

    @property
    def container_path(self) -> str:
        """Return the fixed read-only path used inside an agent container."""

        return self._container_path

    @property
    def deleted(self) -> bool:
        """Return whether cleanup verified file and directory absence."""

        return self._deleted

    def cleanup(self) -> None:
        """Delete only the factory-created file and its empty directory."""

        if self._deleted:
            return
        try:
            if not self._path.exists() and not self._directory.exists():
                self._deleted = True
                return
            if self._path.parent != self._directory:
                raise RuntimeError("ownership")
            if self._path.name != _CREDENTIAL_FILENAME:
                raise RuntimeError("filename")
            if self._path.is_symlink() or self._directory.is_symlink():
                raise RuntimeError("symlink")
            if not self._path.is_file() or not self._directory.is_dir():
                raise RuntimeError("missing")
            if _identity(self._directory) != self._directory_identity:
                raise RuntimeError("directory identity")
            if _identity(self._path) != self._file_identity:
                raise RuntimeError("file identity")
            entries = list(self._directory.iterdir())
            if entries != [self._path]:
                raise RuntimeError("unexpected directory contents")
            self._path.unlink()
            self._directory.rmdir()
            if self._path.exists() or self._directory.exists():
                raise RuntimeError("verification")
            self._deleted = True
        except BaseException as error:
            try:
                absent = not self._path.exists() and not self._directory.exists()
            except BaseException:  # noqa: BLE001
                absent = False
            if absent:
                self._deleted = True
                raise
            raise RuntimeSecretCleanupError(
                "cleanup",
                type(error).__name__,
                self._path,
            ) from error


def _is_within(path: Path, root: Path) -> bool:
    return path == root or path.is_relative_to(root)


def create_runtime_secret_file(
    *,
    encoded_key: str,
    encoded_admin_key: str | None = None,
    forbidden_roots: Sequence[Path] = (),
) -> RuntimeSecretFile:
    """Create a restricted host credential file outside build/output roots."""

    payload: dict[str, str] = {}
    directory: Path | None = None
    path: Path | None = None
    try:
        if not isinstance(encoded_key, str) or not encoded_key:
            raise RuntimeSecretError("encoded execution key is required")
        if encoded_admin_key is not None and (
            not isinstance(encoded_admin_key, str) or not encoded_admin_key
        ):
            raise RuntimeSecretError("encoded Admin key must be nonempty")
        directory = Path(tempfile.mkdtemp(prefix="mha-2-7-secret-")).resolve()
        roots = tuple(root.resolve() for root in forbidden_roots)
        if any(_is_within(directory, root) for root in roots):
            raise RuntimeError("forbidden root")
        os.chmod(directory, 0o700)
        path = directory / _CREDENTIAL_FILENAME
        payload.update(
            schema_version=RUNTIME_CREDENTIAL_SCHEMA,
            encoded_key=encoded_key,
        )
        if encoded_admin_key is not None:
            payload["encoded_admin_key"] = encoded_admin_key
        flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
        descriptor = os.open(path, flags, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, separators=(",", ":"))
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(path, 0o600)
        return RuntimeSecretFile(
            _FACTORY_SENTINEL,
            directory=directory,
            path=path,
            directory_identity=_identity(directory),
            file_identity=_identity(path),
        )
    except BaseException as error:
        cleanup_error: BaseException | None = None
        if directory is not None:
            try:
                if path is not None and path.exists() and not path.is_symlink():
                    path.unlink()
                directory.rmdir()
            except BaseException as caught_cleanup_error:  # noqa: BLE001
                cleanup_error = caught_cleanup_error
        try:
            cleanup_verified = directory is None or (
                not directory.exists() and (path is None or not path.exists())
            )
        except BaseException as verification_error:  # noqa: BLE001
            cleanup_verified = False
            if cleanup_error is None:
                cleanup_error = verification_error
        if not cleanup_verified:
            raise RuntimeSecretCleanupError(
                "creation_cleanup",
                type(cleanup_error or error).__name__,
                path or directory,
            ) from error
        if not isinstance(error, Exception):
            raise
        raise RuntimeSecretError(
            f"runtime-secret creation failed ({type(error).__name__})"
        ) from None
    finally:
        payload.clear()
        encoded_key = ""
        encoded_admin_key = None


def load_runtime_credentials(
    path: str | os.PathLike[str],
    *,
    budget_source: str,
) -> tuple[str, str | None]:
    """Load and validate encoded credentials from the runtime-only mount."""

    payload: Any = {}
    encoded_key = ""
    encoded_admin_key: str | None = None
    try:
        with Path(path).open("r", encoding="utf-8") as stream:
            payload = json.load(stream)
        if budget_source not in {"estimated", "organization"}:
            raise ValueError("budget source")
        expected = {"schema_version", "encoded_key"}
        if budget_source == "organization":
            expected.add("encoded_admin_key")
        if not isinstance(payload, dict) or set(payload) != expected:
            raise ValueError("schema fields")
        if payload["schema_version"] != RUNTIME_CREDENTIAL_SCHEMA:
            raise ValueError("schema version")
        encoded_key = payload["encoded_key"]
        encoded_admin_key = payload.get("encoded_admin_key")
        if not isinstance(encoded_key, str) or not encoded_key:
            raise ValueError("execution key")
        if budget_source == "organization" and (
            not isinstance(encoded_admin_key, str) or not encoded_admin_key
        ):
            raise ValueError("Admin key")
        return encoded_key, encoded_admin_key
    except Exception as error:  # noqa: BLE001
        raise RuntimeSecretError(
            f"runtime credential loading failed ({type(error).__name__})"
        ) from None
    finally:
        if isinstance(payload, (dict, list)):
            payload.clear()
        encoded_key = ""
        encoded_admin_key = None


def write_runtime_secret_diagnostic(
    target: Path,
    *,
    error: RuntimeSecretCleanupError,
    experiment: str,
    run: int,
    attempt_id: str | None,
) -> None:
    """Persist the failure-only host path needed for manual secret cleanup."""

    target = target.resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "2-7-runtime-secret-diagnostic-v1",
        "timestamp": datetime.now(UTC).isoformat(),
        "experiment": experiment,
        "run": run,
        "attempt_id": attempt_id,
        "stage": "runtime_secret_cleanup",
        "operation": error.operation,
        "error_type": error.error_type,
        "host_path": str(error.diagnostic_path),
    }
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.",
        suffix=".tmp",
        dir=target.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(target)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


class RuntimeSecretOrchestrator(Orchestrator):
    """MHAgentA 1.4.12 adapter that excludes secrets from image construction."""

    def __init__(
        self,
        *args: Any,
        runtime_secret_mounts: Mapping[str, RuntimeSecretFile] | None = None,
        runtime_secret_mounted_callback: Callable[[str], None] | None = None,
        **kwargs: Any,
    ) -> None:
        _install_windows_docker_pipe_wait()
        for attempt in range(_LOG_PIPE_RETRY_LIMIT):
            try:
                super().__init__(*args, **kwargs)
                break
            except Exception as error:
                if (
                    not _docker_pipe_is_busy(error)
                    or attempt + 1 >= _LOG_PIPE_RETRY_LIMIT
                ):
                    raise
                time.sleep(_LOG_PIPE_RETRY_SECONDS)
        self._log_parser = _RetryingLogParser(self._log_parser)
        self._runtime_secret_mounts = dict(runtime_secret_mounts or {})
        self._runtime_secret_mounted_callback = runtime_secret_mounted_callback

    async def _run_agent(
        self,
        agent: AgentEntry,
        force_run: bool = False,
    ) -> None:
        secret = self._runtime_secret_mounts.get(agent.agent_id)
        if secret is not None and agent.num_copies != 1:
            raise RuntimeSecretError("runtime-secret mounts require one agent copy")

        if agent.num_copies == 1:
            print(
                f'===== RUNNING AGENT IMAGE "mhagent:{agent.agent_id}" '
                f'AS CONTAINER "{agent.agent_id}" ====='
            )
        else:
            print(
                f'===== RUNNING AGENT IMAGE "mhagent:{agent.agent_id}" AS '
                f'{agent.num_copies} CONTAINERS "{agent.agent_id}_#" ====='
            )
        agent.containers = {}
        for index in range(agent.num_copies):
            if agent.num_copies == 1:
                agent_name = agent.agent_id
                agent_dir = (agent.dir / self.SAVE_SUBDIR).resolve()
            else:
                agent_name = f"{agent.agent_id}_{index}"
                agent_dir = (agent.dir.with_name(agent_name) / self.SAVE_SUBDIR).resolve()

            agent_dir.mkdir(parents=True, exist_ok=True)
            try:
                container = self._docker_client.containers.get(agent_name)
                if force_run:
                    container.remove(force=True)
                else:
                    raise NameError(f"Container {agent_name} already exists")
            except NotFound:
                pass

            if self._mas_rmq_uri_internal is not None:
                host, raw_port = self._mas_rmq_uri_internal.split(":")
                port = int(raw_port) + 10_000
            else:
                host, port = None, None

            volumes: dict[str, dict[str, str]] = {
                str(agent_dir): {"bind": f"/{self.SAVE_SUBDIR}", "mode": "rw"}
            }
            if secret is not None:
                if not secret.path.is_file() or secret.path.is_symlink():
                    raise RuntimeSecretError("runtime-secret file is unavailable")
                agent_root = agent.dir.resolve()
                if _is_within(secret.path.resolve(), agent_root):
                    raise RuntimeSecretError("runtime-secret file is inside agent output")
                if any(
                    volume["bind"] == secret.container_path
                    for volume in volumes.values()
                ):
                    raise RuntimeSecretError("runtime-secret mount destination collision")
                volumes[str(secret.path)] = {
                    "bind": secret.container_path,
                    "mode": "ro",
                }

            assert agent.containers is not None
            agent.containers[agent_name] = self._docker_client.containers.run(
                image=agent.image,
                detach=True,
                name=agent_name,
                environment={
                    "AGENT_ID": agent_name,
                    "DOCKER_NAME": agent_name,
                    "RMQ_HOST": host,
                    "RMQ_PORT": port,
                    "VERBOSE": "true" if self._log_level <= self.PROGRESS else "false",
                },
                volumes=volumes,
                extra_hosts={"host.docker.internal": "host-gateway"},
                ports=agent.port_mapping,
                device_requests=self._resolve_gpu_ids(agent.gpu_device_ids),
            )
            if secret is not None and self._runtime_secret_mounted_callback is not None:
                self._runtime_secret_mounted_callback(agent_name)
        self._log_parser.add_container(agent)
