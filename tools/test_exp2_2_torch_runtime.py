"""Check 2-2 runtime provenance and rejection of unsuitable GPU images."""

from types import SimpleNamespace

import pytest
import torch
from mhagenta.outboxes import LearnerOutbox
from mhagenta.utils import State

from mha_exp_level2_bw.exp2_2 import runner as bw
from mha_exp_level2_cr.exp2_2 import runner as cr
import exp2_2_bw_remote_preflight as preflight


@pytest.mark.parametrize("runner", [bw, cr])
def test_learner_persists_actual_torch_runtime(runner) -> None:
    """Runtime metadata must survive real MHAgentA state serialization."""
    assert torch.__version__.split("+", 1)[0] == "2.14.0"
    learner = runner.TestLearner("learner", {})
    learner.on_init(seed=0, device="cpu")
    state = State(
        agent_id="agent", module_id="learner", time_func=lambda: 0.0,
        directory=SimpleNamespace(internal=SimpleNamespace(
            ll_reasoning=[SimpleNamespace(module_id="reasoner")],
            memory=[SimpleNamespace(module_id="memory")],
        )),
        outbox=LearnerOutbox(), **runner.initial_states()["learner"],
    )
    learner.on_first(state)
    saved = state.dump()
    assert saved["torch_version"] == str(torch.__version__)
    assert saved["cuda_runtime"] == torch.version.cuda
    assert saved["device"] == "cpu"


@pytest.mark.parametrize("override, message", [
    ({"torch": "2.13.0+cu130"}, "requires Torch 2.14.0"),
    ({"torch": "2.14.0.dev1+cu130"}, "requires Torch 2.14.0"),
    ({"cuda_available": False}, "assigned CUDA GPU"),
    ({"device_count": 2}, "assigned CUDA GPU"),
    ({"device_name": ""}, "GPU name and CUDA computation"),
    ({"cuda_compute_result": 0.0}, "GPU name and CUDA computation"),
    ({}, None),
])
def test_remote_preflight_requires_selected_runtime_and_working_cuda(
    monkeypatch: pytest.MonkeyPatch, override: dict, message: str | None,
) -> None:
    """A visible GPU alone cannot certify the selected Torch runtime."""
    monkeypatch.setattr(preflight, "_host_gpus", lambda: [
        {"index": index, "name": "RTX PRO 4500", "memory_mib": 32768,
         "driver_version": "595.0"} for index in (0, 1)
    ])
    evidence = {
        "torch": "2.14.0+cu130", "cuda_runtime": "13.0", "cuda_available": True,
        "device_count": 1, "device_name": "RTX PRO 4500", "cuda_compute_result": 4096.0,
        **override,
    }
    monkeypatch.setattr(preflight, "_container_probe", lambda *_: evidence)
    if message:
        with pytest.raises(RuntimeError, match=message):
            preflight.preflight(0, preflight.DEFAULT_IMAGE)
    else:
        assert preflight.preflight(0, preflight.DEFAULT_IMAGE)["status"] == "passed"


@pytest.mark.parametrize("runner", [bw, cr])
@pytest.mark.parametrize("module_name", ["TestLLReasoner", "TestLearner"])
def test_torch_modules_limit_cpu_threads(runner, module_name: str) -> None:
    """Each actor/learner must correct an oversized inherited CPU thread pool."""
    previous = torch.get_num_threads()
    try:
        torch.set_num_threads(2)
        module = getattr(runner, module_name)("module", {})
        module.on_init(seed=0, device="cpu")
        assert torch.get_num_threads() == 1
    finally:
        torch.set_num_threads(previous)
