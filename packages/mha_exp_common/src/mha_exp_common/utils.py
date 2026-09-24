from os import PathLike
from pathlib import Path
from typing import Any
import json
from warnings import warn


def gather_states(root: str | PathLike[str], single_agent: bool, no_warnings: bool = False) -> dict[str, dict[str, dict[str, Any]]]:
    root = Path(root).resolve()
    data: dict[str, dict[str, dict[ str, Any]]] = dict()
    save_dir: Path
    agent_id: str
    module_id: str
    if single_agent:
        iterator = [root]
    else:
        iterator = root.iterdir()
    for agent_dir in iterator:
        if not agent_dir.is_dir():
            if not agent_dir.suffix == '.log':
                warn(f'{agent_dir} is not a directory, skipping...')
            continue
        if agent_dir.name == 'tmp':
            continue  # check for the source of this artifact
        agent_id = agent_dir.name
        data[agent_id] = dict()
        save_dir = agent_dir / 'out'
        for module_file in save_dir.iterdir():
            if module_file.is_dir():
                if not no_warnings:
                    warn(f'Unexpected directory {module_file} in {agent_id}\'s save folder, skipping...')
                continue
            if module_file.suffix != '.json':
                if not no_warnings:
                    warn(f'Unexpected file {module_file} in {agent_id}\'s save folder, skipping...')
                continue
            module_id = module_file.stem
            if '.' in module_id:
                module_id = module_id.split('.')[1]
            with open(module_file, 'r', encoding='utf-8') as f:
                data[agent_id][module_id] = json.load(f)
    return data


class Seeder:
    _orc_mod = 100
    _env_mod = 1_000
    _act_mod = 10_000
    _per_mod = 20_000
    _llr_mod = 30_000
    _knw_mod = 40_000
    _hlr_mod = 50_000
    _ggr_mod = 60_000
    _mem_mod = 70_000
    _lea_mod = 80_000

    MODULE_MULTIPLIER = 100
    AGENT_MULTIPLIER = 1000
    ENV_MULTIPLIER = 100

    def __init__(self, run: int = 0) -> None:
        self._run = run

    @property
    def run(self) -> int:
        return self._run

    @run.setter
    def run(self, run: int) -> None:
        self._run = run

    @property
    def orchestrator(self) -> int:
        return self._run + self._orc_mod

    @property
    def environment(self) -> int:
        return self._run + self._env_mod

    @property
    def actuator(self) -> int:
        return self._run + self._act_mod

    @property
    def perceptor(self) -> int:
        return self._run + self._per_mod

    @property
    def ll_reasoner(self) -> int:
        return self._run + self._llr_mod

    @property
    def knowledge(self) -> int:
        return self._run + self._knw_mod

    @property
    def hl_reasoner(self) -> int:
        return self._run + self._hlr_mod

    @property
    def goal_graph(self) -> int:
        return self._run + self._ggr_mod

    @property
    def memory(self) -> int:
        return self._run + self._mem_mod

    @property
    def learner(self) -> int:
        return self._run + self._lea_mod


def agent_name(run: int, exp: str) -> str:
    return f'exp_agent{exp}_{run}'


def env_name(run: int, exp: str) -> str:
    return f'exp_env{exp}_{run}'


def module_name(mod_type: str, num: int) -> str:
    return f'{mod_type}_{num}'
