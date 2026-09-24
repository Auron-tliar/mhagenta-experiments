# Experiment tools

Run these tools from the repository root after installing the workspace.
Policy preparation uses the Torch 2.13 `policies` extras; 2-2 and 2-6 runtime
checks use the Torch 2.14 `exp2-2` group. See the root [README](../README.md).

| Tool | Purpose |
| --- | --- |
| [exp2_5_policy_preparation](exp2_5_policy_preparation/README.md) | Train and validate BW Transfer policies, including resized policies |
| [bw_achieve_on_preparation](bw_achieve_on_preparation/README.md) | Initialize, train and assess direct BW AchieveOn policies |
| [exp2_5_cr_preparation](exp2_5_cr_preparation/README.md) | Prepare Crafter grounding and five independent native-action policies |
| `configs/2-5-bw.json` | Select the final three-size, fixed-goal BW policy-family treatment |
| `exp2_2_bw_remote_preflight.py` | Check the selected GPU and Torch 2.14 runtime before a 2-2 batch |
| `preflight_exp2_6_cr.py` | Execute a bounded CR runtime-learning preflight |
| `monitor_exp2_6_cr.py` | Read batch, module and learner progress without modifying a run |
| `heartbeat_exp2_6_cr.py` | Monitor a named Linux `cr26-` service and stop its assigned job on execution faults |
| `export_exp2_6_cr_final.py` | Export completed CR runs and final policy evidence to a new local archive |

Every command-line tool accepts `--help`. Training, preflight and service
monitoring are explicit operations; importing the packages does not start
them. Preparation outputs and exported results are separate from the source
snapshot.

`test_exp2_2_parallel_runtime.py`, `test_exp2_2_torch_runtime.py` and
`tests/test_exp1_3_channel_validation.py` check runtime assembly without
launching experiment batches. Preparation directories also contain their
focused tests.

Host assignments, deployment archives, dated campaign controllers and
superseded experiment prototypes remain local. The selected runtime
policy-family bundle is retained with its validation evidence; its preparation
code requires the original development inputs described in its README.
