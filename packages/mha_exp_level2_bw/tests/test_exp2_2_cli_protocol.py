"""Validate CLI protocol reconstruction without launching experiment containers."""

import json

import pytest

from mha_exp_cli import cli
from mha_exp_level2_bw.exp2_2 import runner
from mha_exp_level2_bw.exp2_2.protocol import DQNProtocol


def test_cli_preserves_spatial_protocol(tmp_path, monkeypatch):
    """The CLI must forward the exact treatment and GPU through the batch runner."""
    protocol = DQNProtocol(synchronized_training=False, action_masking=True,
                           min_updates_per_transition=0.25,
                           evaluation_interval_seconds=900,
                           network_architecture='goal_spatial_cnn_v1')
    captured = {}

    class EntryPoint:
        def load(self):
            return runner.run_batch

    monkeypatch.setattr(cli, 'available_experiments', lambda: {'2.bw.2': EntryPoint()})
    monkeypatch.setattr(runner, 'run_experiment_batch', lambda **kwargs: captured.update(kwargs))
    monkeypatch.setattr(runner, 'aggregate_run_directory', lambda *args, **kwargs: None)
    monkeypatch.setattr(runner, 'process_execution_metrics', lambda *args, **kwargs: None)
    config = tmp_path / 'config.json'
    config.write_text(json.dumps({'protocol': protocol.record(), 'gpu_device': 0}))
    cli.run_experiment('2.bw.2', runs=[7], work_dir=tmp_path / 'results', config_file=config)
    assert captured['runner'].keywords == {'protocol': protocol, 'gpu_device': 0}
    assert list(captured['runs']) == [7]
    assert captured['cleanup_before_run'] is False
    invalid = protocol.record()
    invalid['scheduling_mode'] = 'synchronous'
    config.write_text(json.dumps({'protocol': invalid, 'gpu_device': 0}))
    with pytest.raises(ValueError, match='Inconsistent'):
        cli.run_experiment('2.bw.2', runs=[7], config_file=config)
