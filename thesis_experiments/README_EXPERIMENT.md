# Experiment Guide — Reproducing the BACKPAS vs. Baseline Study

This guide explains, step by step, how to reproduce the BASELINE and BACKPAS
experiments reported in the paper. For a high-level overview of the project,
see [`README.md`](README.md).

## The experiment in one paragraph

We solve each MIS instance to **proven optimality** in two ways and compare them.
**BASELINE** is plain Gurobi. **BACKPAS** is a two-phase scheme: Phase 1 runs
Gurobi under a GNN-derived **trust region** (a short, time-boxed search that
yields a strong incumbent), and Phase 2 runs Gurobi on the **original** instance
using that incumbent as a **warm-start**, so global optimality is still
certified. Every instance is run with several seeds; since the GNN predictions
are deterministic, all variability comes from Gurobi itself.

## Prerequisites

```bash
# From the repository root, with your environment activated
python -c "import torch, gurobipy, networkx; print('OK')"
```

You need:

- A valid **Gurobi** license.
- The trained model at
  `wkdir/MIS/ml_training/graph_with_literals_8_GTR/best_model.pth`.
- A set of MIS test instances (`.lp` or `.mps`).

## Paper hyperparameters (MIS)

| Parameter             | Symbol | Value      | Description                                |
|-----------------------|--------|------------|--------------------------------------------|
| `--threshold`         | θ      | `0.6237`   | Confidence cutoff for backbone selection   |
| `--alpha`             | α      | `-0.8419`  | Adaptive trust-region tolerance            |
| `--trust_region_time` | —      | `300`      | Phase-1 duration (seconds)                 |
| `--time_limit`        | —      | `3600`     | Total per-instance budget (seconds)        |
| `--threads`           | —      | `1`        | Single-threaded (deterministic, fair)      |

These were obtained by grid search on the validation set.

## Recommended workflow: multi-seed runner

`run_multi_seed.py` runs an entire instance set across several seeds and writes
one CSV per seed for each method. This is the entry point used for the paper.

### Run both methods (baseline + BACKPAS)

```bash
python src/run_multi_seed.py \
    --instance_dir <path/to/test/instances> \
    --model wkdir/MIS/ml_training/graph_with_literals_8_GTR/best_model.pth \
    --seeds 0 1 2 3 4 \
    --output_dir results/metrics/my_run \
    --threshold 0.6237 --alpha -0.8419 \
    --trust_region_time 300 --time_limit 3600
```

This produces, inside `results/metrics/my_run/`:

```
baseline_seed0.csv  baseline_seed1.csv  ...  baseline_seed4.csv
backpas_seed0.csv   backpas_seed1.csv   ...  backpas_seed4.csv
```

### Run only one method

```bash
python src/run_multi_seed.py ... --baseline   # baseline only
python src/run_multi_seed.py ... --backpas    # BACKPAS only
```

### `run_multi_seed.py` arguments

| Argument              | Description                                   | Default       |
|-----------------------|-----------------------------------------------|---------------|
| `--instance_dir`      | Directory of `.lp`/`.mps` instances (required)| —             |
| `--model`             | Path to the trained `.pth` model (required)   | —             |
| `--output_dir`        | Where per-seed CSVs are written (required)    | —             |
| `--seeds`             | Gurobi seeds                                  | `0 1 2 3 4`   |
| `--threads`           | Gurobi threads                               | `1`           |
| `--time_limit`        | Total per-instance budget (s)                | `3600`        |
| `--trust_region_time` | Phase-1 budget (s)                           | `300`         |
| `--threshold`         | Confidence threshold θ                        | `0.7`         |
| `--alpha`             | Trust-region tolerance α                      | `0.0`         |
| `--log_dir`           | Directory for Gurobi logs                     | `None`        |
| `--baseline`          | Run baseline only                            | both          |
| `--backpas`           | Run BACKPAS only                             | both          |

## Single-instance runs (debugging)

`run_gurobi_experiment.py` is the underlying driver; use it to test one instance.

```bash
# Baseline
python src/run_gurobi_experiment.py \
    --instance <path/to/instance.mps> \
    --output_csv results/metrics/baseline.csv \
    --time_limit 3600

# BACKPAS
python src/run_gurobi_experiment.py \
    --instance <path/to/instance.mps> \
    --backpas \
    --model_path wkdir/MIS/ml_training/graph_with_literals_8_GTR/best_model.pth \
    --trust_region_time 300 \
    --threshold 0.6237 --alpha -0.8419 \
    --output_csv results/metrics/backpas.csv \
    --time_limit 3600
```

### Phase-2 variants

The driver implements two ways of setting up Phase 2 (both certify the original
instance using the same Phase-1 incumbent):

- **Model resumption (opA)** — reuse the Phase-1 model, keep the generated cuts,
  remove the trust-region constraints.
- **Clean reload (opB)** — reload the original instance from disk and apply the
  incumbent via the `Start` attribute.

Within Phase 2 we additionally contrast Gurobi's `Heuristics=0` (trusts the
warm-start; **H0**) against the default heuristics (**HDefault**), and a
`MIPFocus=2` variant. These correspond to the result groups in the paper.

## Generating the result tables

After the per-seed CSVs exist:

### Runtime table

```bash
# Reads baseline_seed*.csv and backpas_seed*.csv -> time_table.csv
python src/analytics/make_table.py --dir results/metrics/my_run

# Only one method:
python src/analytics/make_table.py --dir results/metrics/my_run --baseline
python src/analytics/make_table.py --dir results/metrics/my_run --backpas
```

### Primal integral table (+ Wilcoxon)

This aggregates the primal integral per variant (mean across seeds per instance,
then mean over all instances), runs a paired Wilcoxon test (baseline − variant),
and reports the effect size `r = Z / sqrt(N)`. It writes `primal_table.csv`.

```bash
python src/analytics/make_primal_table.py --base_dir results/metrics
```

> Edit the `VARIANTS` mapping at the top of `make_primal_table.py` to point at
> your own result folders. opA variants reuse a single baseline folder because
> plain Gurobi is unaffected by Phase-2 parameters.

## Interpreting the output CSVs

Key columns shared by both methods:

| Column            | Description                                          |
|-------------------|------------------------------------------------------|
| `method`          | `baseline` or `backpas`                              |
| `runtime`         | Total wall-clock time (Python, GNN overhead included)|
| `gurobi_runtime`  | Solver-internal time                                 |
| `primal_integral` | Anytime metric on the solver clock (lower is better) |
| `obj_val`         | Objective value of the returned solution            |
| `obj_bound`       | Best dual bound                                      |
| `mip_gap`         | Final optimality gap                                |
| `status_name`     | `OPTIMAL` / `TIME_LIMIT` / ...                       |

Additional BACKPAS columns:

| Column          | Description                              |
|-----------------|------------------------------------------|
| `phase1_time`   | Phase-1 (trust region) duration          |
| `phase2_time`   | Phase-2 (warm-started) duration          |
| `threshold`     | θ used                                    |
| `alpha`         | α used                                    |

### When is BACKPAS successful?

1. **Same optimum** — optimality is never compromised (correctness first).
2. **Lower primal integral** — it reaches good solutions much faster.
3. **Less algorithmic work and/or wall-clock** — especially on hard instances,
   where the warm-start absorbs the preprocessing overhead.

## Troubleshooting

**`Module BACKPAS not available` / torch import errors.**
Install PyTorch and PyTorch Geometric following the official guide (it covers
CPU-only and Apple Silicon setups):
<https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html>

**`Version number is X, license is for version Y`.**
Install the `gurobipy` build matching your license:

```bash
pip uninstall gurobipy
pip install gurobipy==<your_version>
```

**`No .lp/.mps files found`.**
The driver accepts `.lp` and `.mps`. Verify the directory contents:

```bash
ls <directory>/*.lp <directory>/*.mps
```

## Scripts at a glance

| Script                              | Purpose                                          |
|-------------------------------------|--------------------------------------------------|
| `src/run_gurobi_experiment.py`      | Core driver — BASELINE and BACKPAS modes         |
| `src/run_gurobi_baseline.py`        | Baseline-only convenience driver                 |
| `src/run_multi_seed.py`             | Run an instance set across several seeds         |
| `src/analytics/make_table.py`       | Aggregate runtimes into a table                  |
| `src/analytics/make_primal_table.py`| Aggregate primal integral + paired Wilcoxon      |
| `src/analytics/combine_seeds.py`    | Merge per-seed CSVs                              |
| `src/analytics/analyze_results.py`  | Statistical analysis helpers                     |

## Acknowledgments

This experiment suite is built on top of the original **BACKPAS** implementation
by Bryan Alvarado Ulloa:

> <https://github.com/bryan-alvarado-ulloa/backpas>

The GNN architecture, trust-region construction, and trained model come from that
repository. We thank the author for making it available.
