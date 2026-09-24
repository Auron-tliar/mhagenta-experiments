# MHAgentA experiments

Source snapshot of the MHAgentA experiments in Blocks World and Crafter. The
repository covers the thesis's level-1 framework validation and seven level-2
agent designs in each environment. It contains the agents, environments,
experiment runners, result checkers, fixed task definitions and runtime policy
bundles. Collected results and recordings are distributed separately in the
[published results release](#published-results).

## Published results

**Results version 1.0** is available as
[MHAgentA experimental validation results 1.0 on Zenodo](https://zenodo.org/records/22912504),
DOI: [10.5281/zenodo.22912504](https://doi.org/10.5281/zenodo.22912504).

This data release is paired with **this frozen `mhagenta-experiments` source
snapshot**, whose workspace version is **0.1.0**, using MHAgentA **1.4.12**.
Use this snapshot's experiment implementations, fixed task definitions,
bundled policies and result processors with results version 1.0.

### Download and extraction

The dataset is **one ZIP archive split into 11 parts**, named
`mhagenta-results-v1.0.zip.001` through `mhagenta-results-v1.0.zip.011`.
All eleven parts are required. Download them into the same directory and
keep their original filenames.

With [7-Zip](https://www.7-zip.org/), open
`mhagenta-results-v1.0.zip.001` and choose **Extract**. It reads the remaining
parts automatically. To check and extract from the command line, run the
following from the directory containing all parts, with `7z` available:

```sh
7z t mhagenta-results-v1.0.zip.001
7z x mhagenta-results-v1.0.zip.001 -oresults-v1.0
```

The first command checks archive integrity; the second extracts the dataset
into `results-v1.0/`. Use `7zz` in place of `7z` on systems that provide that
executable name. Zenodo also lists a checksum for each downloadable part.

Keep the extracted directory structure intact. When processing existing
results, point `--work-dir` at the relevant extracted experiment batch
directory and use `--process-only`, as described below.

## Repository layout

```text
packages/
  mha_env_blocksworld/   Blocks World environment
  mha_env_crafter/       Crafter environment and vendored game assets
  mha_exp_cli/           Shared experiment command line
  mha_exp_common/        Batch execution, metrics and runtime utilities
  mha_exp_level1/        Framework lifecycle and communication experiments
  mha_exp_level2_bw/     Agent architectures in Blocks World
  mha_exp_level2_cr/     Agent architectures in Crafter
tools/                  Offline policy preparation and validation utilities
pyproject.toml          Workspace membership and dependency groups
uv.lock                 Locked Python dependencies
```

Each package has its own `pyproject.toml` and `src/` tree. Focused checks live
in package `tests/` directories and alongside the relevant tools. Level 3 and
superseded hierarchical 2-6 prototypes are outside the release.

## Setup

Use Python 3.12 or later, `uv`, and the MHAgentA **1.4.12 source checkout** in
this sibling layout:

```text
parent/
  mhagenta/
  mhagenta-experiments/
```

The workspace installs `../mhagenta` as an editable dependency. Experiment
runners also use that checkout to assemble container code, so retain this
layout even when the Python packages are installed.

From `mhagenta-experiments/`:

```sh
uv sync --locked --all-packages
uv run --no-sync mha-exp --list
```

Executing experiments requires Docker with Linux containers. MHAgentA manages
the experiment containers and RabbitMQ communication. Neural experiments
select the `1.4.12-torch13.0` MHAgentA image; other experiments use `1.4.12`.
GPU treatments require a compatible NVIDIA driver and GPU access from Docker.
Symbolic planning uses the declared Unified Planning engines; Java is needed
for ENHSP. Container requirements and initialization scripts are kept beside
the owning experiment.

For host-side policy preparation and policy checks, install the Torch 2.13
extras:

```sh
uv sync --locked --all-packages --extra policies
```

Experiments 2-2 and 2-6 use Torch **2.14.0** in their runtime requirements.
Install the separate `exp2-2` host dependency group for those runtime checks:

```sh
uv sync --locked --all-packages --group exp2-2
```

These two Torch selections are mutually exclusive. The container image tag
alone does not establish the installed Torch version; the experiment-specific
requirements determine it. Use `--no-sync` for subsequent commands to keep
the environment selection you just installed.

## Included experiments

`bw` denotes Blocks World and `cr` denotes Crafter. Experiment IDs are those
accepted by `mha-exp`.

| IDs | Purpose |
| --- | --- |
| `1.1` | Module multiplicity and lifecycle |
| `1.2` | Agent-to-agent and agent-to-environment communication |
| `1.3` | Internal typed communication between modules |
| `1.4` | Concurrent execution and saturation under load |
| `2.bw.1`, `2.cr.1` | Rule-based reactive agents |
| `2.bw.2`, `2.cr.2` | DQN learning agents |
| `2.bw.3`, `2.cr.3` | Symbolic BDI agents |
| `2.bw.4`, `2.cr.4` | Hybrid symbolic agents |
| `2.bw.5`, `2.cr.5` | Hybrid agents with pretrained policies |
| `2.bw.6`, `2.cr.6` | Hybrid agents with runtime learning |
| `2.bw.7`, `2.cr.7` | Hybrid agents with LLM modules |
| `2.bw.5.matched` | Fixed-goal pretrained Blocks World comparison |
| `2.cr.7.extended` | Extended Crafter LLM demonstration |

The public 2-6 entry points select the direct AchieveOn implementation in BW
and continuing native-policy learning in CR. Their result validators are
`exp2_6_direct/results.py` and `exp2_6/online_runner.py`, respectively.

## Running and processing results

Run one level-1 experiment in a fresh output directory:

```sh
uv run --no-sync mha-exp 1.1 --num-runs 1 --work-dir results/level1-example
```

Omitting `--num-runs` uses the selected experiment's batch size. Use
`--run-range 0,2,5-7` to select zero-based run IDs explicitly. Default output
directories are under `agents/`; both `agents/` and `results/` are ignored by
Git.

Use a new work directory for each execution. Several legacy batch runners
clear their target directory before starting; the newer runners refuse an
existing directory. Preserve completed runs when recovering an interrupted
batch, and select only the remaining run IDs in a new directory.

To regenerate reports from saved evidence without launching containers:

```sh
uv run --no-sync mha-exp 1.1 --process-only --work-dir results/level1-example
```

Additional batch parameters can be supplied as a JSON object with
`--config-file path/to/config.json`. Its keys must match the selected runner's
`run_batch` parameters. Run IDs, work directory, version and processing mode
remain command-line options. Frozen treatments enforce their own durations,
devices and run ranges.

The thesis's final 2-5-BW comparison uses fifty fixed tasks at three world
sizes and the qualified three-size Transfer family. Its explicit configuration
is included:

```sh
uv run --no-sync mha-exp 2.bw.5 --run-range 0-49 --config-file tools/configs/2-5-bw.json --work-dir results/2-5-bw
```

Default batch sizes and optional treatments are not a substitute for the
thesis's per-experiment run selection. Reproducing a retained campaign also
requires its saved configuration and seed assignment from the results bundle.

The optional offline AchieveOn condition of 2-5-BW requires separately
prepared training and assessment directories, supplied as `pretrained_run`
and `assessment` in the configuration. Those outputs are not bundled. The
default Transfer policies and the initialization used by 2-6-BW are included.

LLM experiments make paid API calls. On Windows they read the generic
Credential Manager entry `mhagenta/openai-exp`, username `default`. On Linux
they read an owner-only regular file named `openai-exp` in
`CREDENTIALS_DIRECTORY`. Credentials are supplied locally and are never part
of the repository. Processing existing results does not require API calls.

## Validation and preparation

A small check of the shared runner and command-line behavior:

```sh
uv run --no-sync python -m pytest packages/mha_exp_cli/tests packages/mha_exp_common/tests tools/tests/test_exp1_3_channel_validation.py -q
```

Experiment-specific tests live in the corresponding package. Neural tests
need the appropriate Torch selection described above. These tests are
separate from Docker/GPU experiment runs.

See [tools/README.md](tools/README.md) for retained preparation, preflight,
monitoring and export utilities. New training outputs belong under
`results/`, outside the published source tree.

## Snapshot contents

Runtime assets stay beside their consumers: PDDL domains, fixed task and
schedule JSON, Crafter textures, selected policy weights, perception templates
and checksum-bound qualification evidence. The detailed assessment files in
the BW policy-family bundle are required by its runtime validator.

Git excludes collected results, run logs, recordings, archived checkpoints,
temporary experiments, development plans, implementation reviews, coding-agent
instructions and host-specific collection campaigns. These exclusions leave
local files intact. The root allowlist in `.gitignore` makes the publication
boundary explicit.

Crafter is vendored with project-specific changes. Its original
[README](packages/mha_env_crafter/src/mha_env_crafter/README.md) and
[MIT license](packages/mha_env_crafter/src/mha_env_crafter/LICENSE) are retained.
