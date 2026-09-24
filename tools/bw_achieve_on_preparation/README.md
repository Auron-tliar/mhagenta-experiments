# Direct AchieveOn policy preparation

These tools prepare an offline policy for the optional AchieveOn condition of
2-5-BW. The policy receives a final `On(top, bottom)` goal and selects atomic
actions. The shared policy and learning implementation also supports the
runtime-learning 2-6-BW agent.

Install the workspace with the Torch 2.13 `policies` extras, then run commands
from the repository root. Offline preparation needs the BW planning engines
and their Java dependency, but does not launch Docker or RabbitMQ.

## Initialization and training

Create an untrained CPU initialization from the bundled Transfer weights:

```sh
uv run --no-sync python -m tools.bw_achieve_on_preparation.prepare --output results/achieve-on-initial
```

Train into a fresh output directory:

```sh
uv run --no-sync python -m tools.bw_achieve_on_preparation.train --device cuda --seed 2605 --output results/achieve-on-training
```

The defaults are 600 attempted demonstration tasks, 6,000 warm updates and
50,000 online actions. Use `--help` for explicit budget, architecture and
selection options. `--device cpu` is also supported.

Initialization copies the Transfer network and adds a goal-bottom identity
feature with zero initial weights. Demonstrations use the symbolic planner
and frozen Transfer policy; every learning target is the final AchieveOn
goal. Candidate execution selects atomic actions directly. Completion
requires both `On(top, bottom)` and an empty hand.

The trainer records attempted demonstrations, outcomes, selection results,
source hashes and checkpoints. Failed teacher attempts remain in its
baseline evidence. Checkpoints contain weights, not a complete optimizer,
replay and random-state continuation. An interrupted training run remains
incomplete; a new run needs a fresh output directory.

## Assessment and checks

Assess the selected candidate on the separate assessment cases:

```sh
uv run --no-sync python -m tools.bw_achieve_on_preparation.assess --candidate results/achieve-on-training --output results/achieve-on-assessment --cases 200
uv run --no-sync python -m pytest tools/bw_achieve_on_preparation -q
```

Assessment compares the frozen candidate and symbolic-plus-Transfer baseline
on identical initial states and goals. Keep these cases separate from method
selection. `compare.py` additionally compares an assessed Conv3d candidate
with an existing MLP reference; its inputs are explicit command-line paths.

Outputs remain under `results/`. To use an assessed offline policy in 2-5-BW,
provide its training directory as `pretrained_run`, its assessment directory
as `assessment`, and `policy: "achieve_on"` with a duration in the batch
configuration. The runtime-learning 2-6-BW treatment initializes from the
bundled Transfer weights independently of this offline-trained candidate.
