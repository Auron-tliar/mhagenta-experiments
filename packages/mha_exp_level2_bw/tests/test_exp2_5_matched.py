"""Task provenance, native-size execution and frozen-policy checks for the matched batch."""

from concurrent.futures import Future
from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from mha_exp_level2_bw.exp2_5.matched import matched_task
from mha_exp_level2_bw.exp2_5.resized import frozen_artifact
from mha_exp_level2_bw.exp2_6_direct import batch, runner, runtime
from test_exp2_6_direct import exercise


PLANS = [
    [("b3", "b2", "t0", "t3", "t0"), ("b5", "b4", "b3", "t2", "t0"), ("b4", "b1", "b2", "t2", "t3")],
    [("b7", "b6", "b5", "t4", "t1")],
    [("b06", "b02", "b11", "t1", "t0"), ("b10", "b05", "t3", "t2", "t3"), ("b02", "t1", "b05", "t1", "t2")],
]


def test_all_fifty_tasks_reconstruct_the_comparator_worlds() -> None:
    """Reconstruct every recorded digest with the original seed and correctly padded names."""
    sizes = []
    seeds = []
    for run in range(50):
        task, source = matched_task(run)
        treatment = source["source_treatment"]
        assert task["goal"] == treatment["goal"] and source["source_run"] == run
        sizes.append((treatment["table_len"], treatment["num_blocks"]))
        seeds.append(task["seed"])
        _, artifact = frozen_artifact(*sizes[-1])
        assert artifact["checkpoint_sha256"]
    assert seeds == list(range(23000, 23050))
    assert [sizes.count(size) for size in ((4, 6), (5, 8), (7, 12))] == [17, 17, 16]


@pytest.mark.parametrize("run", [0, 1, 2])
def test_complete_six_module_matched_execution(run, tmp_path, monkeypatch):
    """Execute the actual module message flow using a previously validated symbolic plan."""
    class InlinePool:
        def __init__(self, **kwargs): pass
        def submit(self, function, *args):
            future = Future()
            future.set_result(function(*args))
            return future
        def shutdown(self, **kwargs): pass
    keys = ("block", "source_support", "destination_support", "source", "destination")
    plan = [dict(zip(keys, row)) for row in PLANS[run]]
    monkeypatch.setattr(runtime, "ThreadPoolExecutor", InlinePool)
    seen = []
    def planner(**dimensions):
        seen.append(dimensions)
        return SimpleNamespace(solve=lambda *args: SimpleNamespace(
            actions=plan, accepted=True, elapsed_seconds=0.01, engine="fixture"))
    monkeypatch.setattr(runtime, "production_planner", planner)
    monkeypatch.setattr(runner, "reference", lambda *args: pytest.fail("Matched Transfer needs no AchieveOn checkpoint"))
    config = runner.prepare(tmp_path, tmp_path, tmp_path / "prepared", run, "matched", "cpu", 0, True,
        profile_override=runner.Profile(0, 1, 0, 0, 0, 1, 600, action_cap=None),
        frozen_policy="transfer", matched_2_4=True)
    states, env, config = exercise(tmp_path, True, production=True, frozen_policy="transfer", config_override=config)
    assert seen == [config["dimensions"]]
    assert len(states) == 6 and env["resets"] == 1 and env["closed"]
    assert states["hlreasoner"]["results"][0]["success"]
    assert not states["llreasoner"]["rows"] and not states["llreasoner"]["model_installs"]
    result = runner.check_results(states, env, config, tmp_path)
    assert result["operationally_valid"], result
    assert result["production"]["accomplished"] == 1
    changed = deepcopy(config)
    changed["matched_2_4"]["source_treatment"]["seed"] += 1
    assert "matched-task-or-artifact-identity" in runner.check_results(states, env, changed, tmp_path)["failures"]


def test_matched_batch_stops_on_an_operational_failure(tmp_path, monkeypatch):
    """Do not spend the remaining batch budget when the first run fails validation."""
    monkeypatch.setattr(runner, "source_identity", lambda: {})
    calls = []
    def prepare(*args, **kwargs):
        args[2].mkdir()
        calls.append(args[3])
        return {}
    monkeypatch.setattr(runner, "prepare", prepare)
    monkeypatch.setattr(runner, "execute", lambda *args: {"operationally_valid": False, "failures": ["fixture"]})
    with pytest.raises(RuntimeError, match="remaining runs"):
        batch.run_batch(50, tmp_path / "batch", duration_seconds=600, episode_limit=1,
                        frozen_policy="transfer", matched_2_4=True)
    assert calls == [0]
    assert json.loads((tmp_path / "batch/batch.json").read_text())["status"] == "blocked-operational-failure"


def test_matched_ll_timeout_returns_a_terminal_result_without_another_action(monkeypatch):
    """The task deadline permits acknowledged shutdown before the global 600-second cap."""
    from mhagenta import Observation
    from mha_env_blocksworld import BlocksWorldEnv
    from test_exp2_6_direct import State
    ll = runtime.LLReasoner(module_id="llreasoner", initial_state=runtime.initial_states()["llreasoner"])
    ll.on_init(seed=1, action_cap=None, training_episodes=1, frozen_policy="transfer",
               dimensions={"table_len": 4, "num_blocks": 6})
    task, _ = matched_task(0)
    env = BlocksWorldEnv(table_len=4, num_blocks=6, symbolic=False)
    observation, _ = env.reset(seed=task["seed"])
    env.close()
    state = State(deepcopy(runtime.initial_states()["llreasoner"]))
    state.active, state.pending = task, "observation-1"
    state.directory = SimpleNamespace(internal=SimpleNamespace(perception=[SimpleNamespace(module_id="perceptor")]))
    ll._deadline = -1
    observed = []
    monkeypatch.setattr(ll, "_finish", lambda state, grounded, success, reason=None: observed.append((success, reason)) or state)
    monkeypatch.setattr(ll, "_act", lambda *args, **kwargs: pytest.fail("Must not act after deadline"))
    ll.on_observation(state, "perceptor", Observation(observation), request_id="observation-1")
    assert observed == [(False, "time-limit")]
