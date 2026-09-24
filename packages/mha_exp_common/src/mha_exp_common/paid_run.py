"""Narrow configured-key execution boundary for paid experiments."""

from __future__ import annotations

import base64
import json
import os
import tempfile
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from .batch import (
    NonRetryableBatchError,
    cleanup_run_containers,
    cleanup_run_images,
)
from .runtime_secret import (
    RuntimeSecretCleanupError,
    RuntimeSecretFile,
    create_runtime_secret_file,
)


def encode_api_key(api_key: str) -> str:
    """Encode a key for the existing runtime-secret transport."""

    try:
        return base64.urlsafe_b64encode(api_key.encode("utf-8")).decode("ascii")
    finally:
        api_key = ""


def decode_api_key(encoded_key: str) -> str:
    """Decode a runtime-secret key in the consuming process."""

    try:
        return base64.urlsafe_b64decode(encoded_key.encode("ascii")).decode("utf-8")
    finally:
        encoded_key = ""


def read_windows_credential(
    target: str = "mhagenta/openai-exp",
    username: str = "default",
) -> str:
    """Read one Windows generic credential without exposing its value."""

    try:
        import win32cred
    except ImportError as error:  # pragma: no cover - platform-specific
        raise RuntimeError("pywin32 is required to read the experiment credential") from error

    credential: dict[str, Any] = {}
    secret: Any = ""
    try:
        try:
            credential = win32cred.CredRead(target, win32cred.CRED_TYPE_GENERIC, 0)
        except Exception as error:  # noqa: BLE001 - sanitize credential errors
            code = getattr(error, "winerror", None)
            suffix = f" (Windows error {code})" if code is not None else ""
            raise RuntimeError(f"Credential {target!r} was not found{suffix}") from None
        actual_username = str(credential.get("UserName", ""))
        if actual_username != username:
            raise RuntimeError(
                f"Credential {target!r} belongs to {actual_username!r}, not {username!r}"
            )
        secret = credential.get("CredentialBlob", b"")
        if isinstance(secret, bytes):
            secret = secret.decode("utf-16-le" if b"\x00" in secret else "utf-8")
        if not secret:
            raise RuntimeError(f"Credential {target!r} contains an empty secret")
        return str(secret)
    finally:
        credential.clear()
        secret = ""


def read_experiment_credential() -> str:
    """Read Windows Credential Manager or an owner-only Linux service credential."""
    if os.name == "nt":
        return read_windows_credential()
    directory = os.environ.get("CREDENTIALS_DIRECTORY")
    if not directory:
        raise RuntimeError("CREDENTIALS_DIRECTORY is required on Linux")
    path = Path(directory) / "openai-exp"
    metadata = path.lstat()
    import stat
    if (not stat.S_ISREG(metadata.st_mode) or metadata.st_uid != os.getuid()
            or metadata.st_mode & 0o077):
        raise RuntimeError("Experiment credential must be an owner-only regular file")
    key = path.read_text(encoding="utf-8").strip()
    if not key:
        raise RuntimeError("Experiment credential is empty")
    return key


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    """Atomically write one JSON evidence document."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _safe_failure(error: BaseException) -> dict[str, str]:
    return {"kind": type(error).__name__, "message": "operation failed"}


def configured_paid_run(
    *,
    encoded_key: str,
    evidence_path: Path,
    agent_id: str,
    environment_id: str,
    forbidden_roots: Sequence[Path],
    body: Callable[[RuntimeSecretFile, Callable[[], None]], bool],
) -> bool:
    """Run one configured-key body with exact cleanup and redacted evidence.

    Cleanup order is fixed: pre-run containers/images, runtime-secret creation,
    body, post-run containers, secret deletion, then post-run images. Every
    cleanup stage is attempted after body entry, even when an earlier stage
    fails.
    """

    if not encoded_key:
        raise ValueError("encoded_key is required")
    record: dict[str, Any] = {
        "schema_version": "configured-paid-run-v1",
        "agent_id": agent_id,
        "environment_id": environment_id,
        "credential_mode": "configured_key_runtime_mount",
        "pre_run_containers": {"status": "not_run"},
        "pre_run_images": {"status": "not_run"},
        "runtime_secret": {"created": False, "mounted": False, "deleted": False},
        "body": {"status": "not_run"},
        "post_run_containers": {"status": "not_run"},
        "post_run_images": {"status": "not_run"},
        "cleanup_success": False,
        "failures": [],
    }
    atomic_json(evidence_path, record)
    secret: RuntimeSecretFile | None = None
    first_error: BaseException | None = None
    unsafe_error: NonRetryableBatchError | None = None
    result = False

    def fail(stage: str, error: BaseException) -> None:
        nonlocal first_error
        record["failures"].append({"stage": stage, **_safe_failure(error)})
        if first_error is None:
            first_error = error

    def mounted() -> None:
        record["runtime_secret"]["mounted"] = True
        atomic_json(evidence_path, record)

    try:
        for field, cleanup in (
            ("pre_run_containers", cleanup_run_containers),
            ("pre_run_images", cleanup_run_images),
        ):
            try:
                record[field] = cleanup(agent_id, environment_id, phase="pre_run")
            except BaseException as error:  # noqa: BLE001 - preserve cleanup ordering
                fail(field, error)
                if isinstance(error, NonRetryableBatchError):
                    unsafe_error = error
            atomic_json(evidence_path, record)
        if first_error is None:
            try:
                secret = create_runtime_secret_file(
                    encoded_key=encoded_key,
                    forbidden_roots=forbidden_roots,
                )
                record["runtime_secret"]["created"] = True
                atomic_json(evidence_path, record)
                result = bool(body(secret, mounted))
                record["body"] = {"status": "succeeded", "result": result}
            except BaseException as error:  # noqa: BLE001 - cleanup must still run
                record["body"] = {"status": "failed", **_safe_failure(error)}
                fail("body", error)
    finally:
        encoded_key = ""
        try:
            record["post_run_containers"] = cleanup_run_containers(
                agent_id, environment_id, phase="post_run"
            )
        except BaseException as error:  # noqa: BLE001
            fail("post_run_containers", error)
            if isinstance(error, NonRetryableBatchError):
                unsafe_error = error
        if secret is not None:
            try:
                secret.cleanup()
                record["runtime_secret"]["deleted"] = secret.deleted
            except RuntimeSecretCleanupError as error:
                fail("runtime_secret_cleanup", error)
                unsafe_error = error
        try:
            record["post_run_images"] = cleanup_run_images(
                agent_id, environment_id, phase="post_run"
            )
        except BaseException as error:  # noqa: BLE001
            fail("post_run_images", error)
            if isinstance(error, NonRetryableBatchError):
                unsafe_error = error
        record["cleanup_success"] = bool(
            record["pre_run_containers"].get("status") == "succeeded"
            and record["pre_run_images"].get("status") == "succeeded"
            and record["post_run_containers"].get("status") == "succeeded"
            and record["runtime_secret"].get("deleted") is True
            and record["post_run_images"].get("status") == "succeeded"
        )
        atomic_json(evidence_path, record)

    if unsafe_error is not None:
        raise unsafe_error
    if first_error is not None:
        raise first_error
    return result and bool(record["cleanup_success"])


__all__ = [
    "atomic_json",
    "configured_paid_run",
    "decode_api_key",
    "encode_api_key",
    "read_windows_credential",
    "read_experiment_credential",
]
