"""Focused tests for experiment-specific CLI defaults."""

from pathlib import Path
import json
from typing import Any

import pytest

from mha_exp_cli import cli
from mha_exp_cli import __main__ as cli_main
from mha_exp_cli.__main__ import build_parser


def test_cli_default_delegates_to_entry_point_default(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    class EntryPoint:
        def load(self):
            def run(
                *,
                mha_version: str = "1.4.12-torch13.0",
                **kwargs: Any,
            ) -> None:
                captured.update(kwargs)
                captured["resolved_mha_version"] = mha_version

            return run

    args = build_parser().parse_args(["2.bw.6"])
    assert args.num_runs is None
    assert args.process_only is False
    monkeypatch.setattr(cli, "available_experiments", lambda: {"2.bw.6": EntryPoint()})
    cli.run_experiment("2.bw.6", runs=args.num_runs, work_dir=tmp_path)

    assert "runs" not in captured
    assert "mha_version" not in captured
    assert captured["resolved_mha_version"] == "1.4.12-torch13.0"
    assert captured["process_only"] is False

    captured.clear()
    cli.run_experiment(
        "2.bw.6",
        runs=[2, 0],
        work_dir=tmp_path,
        mha_version="1.4.12-torch13.0",
        process_only=True,
    )
    assert captured["runs"] == [2, 0]
    assert captured["resolved_mha_version"] == "1.4.12-torch13.0"
    assert captured["process_only"] is True


def test_cli_parses_process_only_flag() -> None:
    args = build_parser().parse_args(["2.bw.3", "--process-only"])

    assert args.process_only is True


def test_cli_omits_default_mhagenta_version() -> None:
    args = build_parser().parse_args(["2.bw.5"])

    assert args.mha_version is None


def test_main_forwards_process_only_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}
    monkeypatch.setattr(cli_main, "available_experiments", lambda: {"2.bw.3": object()})
    monkeypatch.setattr(
        cli_main,
        "run_experiment",
        lambda **kwargs: captured.update(kwargs),
    )

    result = cli_main.main(["2.bw.3", "--process-only"])

    assert result == 0
    assert captured["mha_version"] is None
    assert captured["process_only"] is True


def test_cli_forwards_json_workload_without_changing_run_selection(tmp_path, monkeypatch) -> None:
    """Explicit treatment options reach the selected batch entry point."""
    captured = {}
    class EntryPoint:
        def load(self):
            return lambda **kwargs: captured.update(kwargs)
    monkeypatch.setattr(cli, 'available_experiments', lambda: {'2.cr.2': EntryPoint()})
    config = tmp_path / 'config.json'
    options = {'workload': {'synchronized_training': False, 'action_masking': True}, 'gpu_device': 1}
    config.write_text(json.dumps(options))
    cli.run_experiment('2.cr.2', runs=[7], work_dir=tmp_path / 'output', config_file=config)
    assert captured['runs'] == [7]
    assert captured['workload'] == options['workload']
    assert captured['gpu_device'] == 1
    assert build_parser().parse_args(['2.cr.2', '--config-file', str(config)]).config_file == config
    config.write_text(json.dumps({'exp_path': '/unintended'}))
    with pytest.raises(ValueError, match='CLI flags'):
        cli.run_experiment('2.cr.2', runs=[7], config_file=config)
