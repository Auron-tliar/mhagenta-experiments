"""Security boundary tests for portable experiment credentials."""
from __future__ import annotations

import stat
from types import SimpleNamespace

import pytest
from mha_exp_common import paid_run


@pytest.mark.parametrize("mode,owner,accepted", [
    (stat.S_IFREG | 0o400, 1000, True),
    (stat.S_IFREG | 0o600, 1000, True),
    (stat.S_IFREG | 0o640, 1000, False),
    (stat.S_IFREG | 0o600, 1001, False),
    (stat.S_IFLNK | 0o600, 1000, False),
])
def test_linux_credential_requires_owner_only_regular_file(monkeypatch, mode, owner, accepted):
    """Reject credentials exposed to other users or redirected by a symlink."""
    class FakePath:
        def __truediv__(self, name):
            assert name == "openai-exp"
            return self
        def lstat(self):
            return SimpleNamespace(st_mode=mode, st_uid=owner)
        def read_text(self, **kwargs):
            return "synthetic-test-key"
    monkeypatch.setattr(paid_run, "os", SimpleNamespace(name="posix", environ={"CREDENTIALS_DIRECTORY": "/credentials"}, getuid=lambda: 1000))
    monkeypatch.setattr(paid_run, "Path", lambda _: FakePath())
    if accepted:
        assert paid_run.read_experiment_credential() == "synthetic-test-key"
    else:
        with pytest.raises(RuntimeError, match="owner-only"):
            paid_run.read_experiment_credential()


def test_linux_credential_has_no_environment_key_fallback(monkeypatch):
    """An API-key environment variable cannot substitute for a protected file."""
    monkeypatch.setattr(paid_run, "os", SimpleNamespace(name="posix", environ={"OPENAI_API_KEY": "synthetic-test-key"}))
    with pytest.raises(RuntimeError, match="CREDENTIALS_DIRECTORY"):
        paid_run.read_experiment_credential()
