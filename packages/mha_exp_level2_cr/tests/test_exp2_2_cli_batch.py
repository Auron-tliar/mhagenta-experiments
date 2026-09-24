"""Validate main-CLI treatment conversion and batch failure propagation."""

import pytest

from mha_exp_level2_cr.exp2_2 import runner
from mha_exp_level2_cr.exp2_2.treatment import DQNWorkload


def test_json_workload_preserves_async_treatment_and_stops_on_failure(tmp_path, monkeypatch):
    """The CLI adapter must neither revert defaults nor hide failed runs."""
    workload = DQNWorkload(synchronized_training=False, action_masking=True, duration_seconds=3930)
    captured = {}
    def batch(**kwargs):
        captured.update(kwargs)
        raise RuntimeError('run failed')
    monkeypatch.setattr(runner, 'run_experiment_batch', batch)
    monkeypatch.setattr(runner, 'process_execution_metrics', lambda *args, **kwargs: None)
    with pytest.raises(RuntimeError, match='run failed'):
        runner.run_batch(runs=[1], exp_path=tmp_path, workload=workload.dump(), gpu_device=1)
    assert captured['stop_on_error'] is True
    assert captured['cleanup_before_run'] is False
    assert captured['runner'].keywords['workload'] == workload
    assert captured['runner'].keywords['gpu_device'] == 1
