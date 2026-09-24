# Level 1 experiments

Framework validation experiments for MHAgentA 1.4.12. See the root
[README](../../README.md) for installation and execution requirements.

```sh
uv run --no-sync mha-exp 1.1 --num-runs 1 --work-dir results/level1-example
uv run --no-sync mha-exp 1.1 --process-only --work-dir results/level1-example
```

Each batch writes an aggregate `summary.json` and generates its figures in both
SVG and PNG. Passing `--process-only` regenerates these artifacts from existing
results without rerunning containers. Results produced by older schemas require
a new run.

## Claim coverage

| Claim | Experiment | Evidence | Figure |
| --- | --- | --- | --- |
| Module multiplicity and init/first/step/last lifecycle | 1-1 | Exact module, lifecycle, counter, and log checks | `module-counts` |
| Valid partial agent topologies | 1-1 | One optional module type is omitted in every run | `module-counts` |
| Exact external agent and environment communication | 1-2 | Multiset delivery accounting and channel latency | `communication-flow`, `latency-ecdf` |
| Exact internal typed communication topology | 1-3 | Multiset accounting for all supported edges | `communication-heatmap`, `latency-ecdf` |
| Concurrent execution and saturation under load | 1-4 | Run-level CPU-time shares and confidence intervals | `concurrency-saturation`, `finish-spread` |
