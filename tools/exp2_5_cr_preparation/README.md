# Five-policy Crafter preparation

Independent Explore, NavigateTo, GetResource, EatTarget, and EatCow models.
Run from the repository root with the Torch 2.13 `policies` extras installed.
`prepare_grounding.py OUTPUT` creates a new perception-template bundle; the
selected runtime templates are already included in the experiment package.

## Commands

```powershell
uv run --no-sync python tools/exp2_5_cr_preparation/train.py train results/my-explore --device cuda --policy explore
uv run --no-sync python tools/exp2_5_cr_preparation/train.py train results/my-explore --device cuda --policy explore --resume
uv run --no-sync python tools/exp2_5_cr_preparation/train.py train results/my-resource --device cuda --policy get_resource
uv run --no-sync python tools/exp2_5_cr_preparation/train.py pilot results/my-pilot --device cpu --policy explore
uv run --no-sync python tools/exp2_5_cr_preparation/train.py evaluate path/to/explore-policy.pt --policy explore --device cpu --report results/explore-evaluation.json
```

Omitting `--policy` trains all five sequentially and independently.
Initialization is fresh unless a current-format `--init-checkpoint` is
supplied. Warm starts copy weights only, may cross policy identities, and
require a single policy. They cannot be combined with resume.
GetResource remediation always starts fresh and rejects warm starts.

Resume restores model/target weights, optimizer, replay, RNG, curriculum
cursors, and counters. It requires the same policy selection, device, PyTorch
version, and configuration. Historical working formats are unsupported.

A diagnostic pilot collects approximately 256 demonstration and 512 online
steps per policy, then evaluates a smaller schedule. It does not certify a
model or gate a later training command.

## Training and reporting

The existing DQfD-lite optimizer and curricula are retained: one-step and
three-step Double-DQN losses, prioritized replay with protected demonstrations,
demonstration-only margin loss, and target synchronization every 500 updates.
The other four policies collect approximately 2,000 demonstration steps and select at
5,000/10,000/15,000/20,000 online steps, committing complete episodes.

Rewards are +1 for the task, −0.01 otherwise, and +0.3 for consuming another cow
during EatTarget training without terminating the target's episode.
All models retain their static action masks. EatCow additionally masks
unavailable repeated movement and DO without an adjacent faced cow.
The additional execution mask does not change the retained TD/margin losses.

Flushed JSON events report step counts, separate losses, successes, illegal/
lethal actions, target losses, selections, and elapsed time. Per-policy
progress and working state are saved every 25 episodes, at selection, and
after requested interruption. First Ctrl+C commits the current episode; a
second interrupts immediately. Abrupt failures resume from the last save.

## GetResource remediation

GetResource zeros RGB rows 49–63 inside the network, consistently during
demonstrations, optimization, bootstrapping, evaluation, and runtime. The full
image still goes to grounding and HLR. Success requires a legal DO facing the
specified cell and its corresponding collection event, not a net inventory
increase. A wrong resource gives −0.01 and does not complete the goal.

Cases use intact randomized worlds: 50% fresh, 25% after a table, and 25%
after a furnace. Later states come from native 2-1-CR policy rollouts, not
six-action imitation of crafting or sleeping. All six resource kinds, four
approach-distance bands (0–1, 2–3, 4–6, 7–10), facing, pose, valid inventories,
and needs vary. Descriptors save the world seed, native prefix, pose, target,
and inventory for replay. A world supplies at most eight starting cases.
The resource-only environment orders creature sets before native random
despawn selection, preventing object-address-dependent replay differences.
Native probabilities and mechanics are unchanged; runtime and other-policy
environments are untouched. Older case/trajectory files are rejected.
If a moving cow blocks every route after an expert has started, retain its
valid transition prefix as a failed terminal demonstration and log
`expert_route_blocked`. Do not substitute another case. An invalid initial
expert route still raises a harness error; neural evaluations are unchanged.

Training collects 10,000 demonstration transitions and at most approximately
100,000 online transitions (complete episodes). Validation uses 240 fixed
cases every 5,000 steps, with a disjoint world-seed range. Selection maximizes
success rate, then minimizes mean episode cost (failures cost 32), then prefers
the earlier checkpoint. Four checks without improvement stop training after
at least 20,000 steps. Epsilon decays over 80,000 steps; replay beta over 100,000.
The effective settings are recorded separately as `resource_configuration`.

The chosen checkpoint must reach 95% overall and 90% for each resource and
state source on validation, then on a single untouched 600-case final test.
Generation/harness errors abort evaluation; failed neural cases are never
resampled. Failed gates return a nonzero CLI status and do not export a
candidate. The standalone `evaluate --policy get_resource` command uses the
fixed validation set; the final test is owned only by the training workflow.
`--seed` does not override GetResource's fixed split assignment.

Eight transitions confined to at most two cells, without new visible cells
or exact-target collection, terminate GetResource as stagnation. The same
rule applies in training and runtime. Progress includes losses, outcomes,
source/kind/distance coverage, unique worlds, and generation rejections.

## Export

```powershell
uv run --no-sync python tools/exp2_5_cr_preparation/train.py export path/to/new-bundle --checkpoint explore=path/to/explore-policy.pt --checkpoint navigate_to=path/to/navigate-to-policy.pt --checkpoint get_resource=path/to/get-resource-policy.pt --checkpoint eat_target=path/to/eat-target-policy.pt --checkpoint eat_cow=path/to/eat-cow-policy.pt
```

Exactly five current-format models are required. The destination must not
exist. Export validates models, copies checkpoint bytes, and writes a compact
manifest. Training never replaces runtime models automatically.
