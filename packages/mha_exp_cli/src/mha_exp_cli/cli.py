from importlib.metadata import EntryPoint, entry_points
from pathlib import Path
import json
import re


ENTRY_POINT_GROUP = "mhagenta_experiments"


def available_experiments() -> dict[str, EntryPoint]:
    return {
        ep.name: ep
        for ep in entry_points(group=ENTRY_POINT_GROUP)
    }


def parse_run_range(value: str) -> list[int]:
    if not value:
        raise ValueError("run range cannot be empty")

    result: list[int] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            raise ValueError("run range cannot contain empty segments")

        if re.fullmatch(r"\d+", part):
            result.append(int(part))
            continue

        match = re.fullmatch(r"(\d+)-(\d+)", part)
        if not match:
            raise ValueError(f"invalid run range segment: {part!r}")

        start = int(match.group(1))
        end = int(match.group(2))
        if start > end:
            raise ValueError(f"run range start cannot exceed end: {part!r}")
        result.extend(range(start, end + 1))

    return result


def normalize_experiment_id(name: str) -> str:
    normalized = re.sub(r"[^0-9A-Za-z]+", "_", name).strip("_")
    return f"exp{normalized}"


def find_project_root(start: Path = Path.cwd()) -> Path:
    start = start.resolve()
    candidates = [start, *start.parents]
    for candidate in candidates:
        pyproject = candidate / "pyproject.toml"
        if pyproject.is_file() and "[tool.uv.workspace]" in pyproject.read_text():
            return candidate
    return Path.cwd().resolve()


def default_work_dir(experiment: str) -> Path:
    return find_project_root() / "agents" / normalize_experiment_id(experiment)


def run_experiment(
    name: str,
    *,
    runs: int | list[int] | None,
    work_dir: Path | None = None,
    mha_version: str | None = None,
    process_only: bool = False,
    config_file: Path | None = None,
):
    experiments = available_experiments()

    if name not in experiments:
        available = ", ".join(sorted(experiments))
        raise ValueError(f"Unknown experiment {name!r}. Available: {available}")

    run = experiments[name].load()
    exp_path = work_dir.resolve() if work_dir is not None else default_work_dir(name)
    kwargs = {
        "exp_path": exp_path,
        "process_only": process_only,
    }
    if runs is not None:
        kwargs["runs"] = runs
    if mha_version is not None:
        kwargs["mha_version"] = mha_version
    if config_file is not None:
        config = json.loads(config_file.read_text())
        if not isinstance(config, dict):
            raise ValueError("Experiment configuration must be a JSON object")
        reserved = {"runs", "exp_path", "process_only", "mha_version"}
        if reserved.intersection(config):
            raise ValueError("Use CLI flags for run selection, output, processing and version")
        kwargs.update(config)
    return run(**kwargs)
