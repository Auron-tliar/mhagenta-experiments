# Blocks World Transfer policy preparation

Offline preparation for the selected structured DQfD-lite Transfer policy.
Runtime experiment code does not import these tools. Run from the repository
root with the Torch 2.13 `policies` extras installed.

## Original-size policy

Train the fixed seed-2505, 250,000-step candidate into a new directory:

```sh
uv run --no-sync python tools/exp2_5_policy_preparation/train.py train --output-dir results/transfer-candidate
```

Certify the selected candidate and write a compact bundle separately from the
installed runtime policy:

```sh
uv run --no-sync python tools/exp2_5_policy_preparation/train.py certify --candidate-dir results/transfer-candidate --artifact-dir results/transfer-bundle --results-dir results/transfer-certification
```

The certification rule requires full success on 100 fixed single-transfer
cases, one integration case, twenty ten-transfer sequences and one
planner-reachable regression. Both phases refuse existing output directories.

## Resized policies

`train_resized.py` fine-tunes native 4-column/6-block and 7-column/12-block
networks from the bundled original-size policy:

```sh
uv run --no-sync python tools/exp2_5_policy_preparation/train_resized.py --columns 4 --blocks 6 --output results/transfer-4x6
```

`audit_resized.py` independently replays assessment actions through symbolic
Blocks World. Use `--help` for its input paths and the trainer's budget options.

## Final three-size family

The thesis's matched experiment uses `structured-dqfd-family-20260920-v1`.
`train_family.py`, `qualify_family.py` and `family_support.py` retain its
training, selection, seed isolation and independent qualification procedure.
The 5-column/8-block member starts from random weights; each resized member
starts independently from the qualified 5-column/8-block member.

This procedure uses the historical failed cases as development evidence.
Restore `results/2-5-bw-5x8-extended-assessment-20260920-1/` from the separately
distributed results before running it. Continuing an existing campaign also
requires its original `results/2-5-bw-policy-family-20260920/` seed manifest,
consumed-cohort records and historical artifacts named in the manifest.
Preserve these records: the qualifier refuses
to reuse a final assessment cohort. It also verifies the baseline files
recorded in the manifest.

Inspect the training options without starting a run:

```sh
uv run --no-sync python tools/exp2_5_policy_preparation/train_family.py --help
uv run --no-sync python tools/exp2_5_policy_preparation/qualify_family.py --help
```

New preparation never replaces the bundled runtime weights automatically.
The selected family and its checksum-bound qualification evidence are already
included under the experiment's `artifacts/` directory. Host deployment and
campaign publication helpers are outside this snapshot.
