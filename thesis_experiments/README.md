# Thesis Experiments — Warmstarting Gurobi through BACKPAS (MIS)

This directory contains the code, data layout, and instructions to reproduce the
experiments of the paper:

> **Warmstarting Gurobi through BACKPAS: An experimental evaluation on MIS**

The study evaluates whether a GNN-guided, two-phase warm-start can accelerate
**optimality certification** (proving the optimum, not merely finding good
solutions) for the Maximum Independent Set (MIS) problem solved with Gurobi.

## What the experiment does

We compare two ways of solving each MIS instance to proven optimality:

1. **Baseline** — plain Gurobi on the original instance.
2. **BACKPAS** — a two-phase scheme:
   - **Phase 1**: a GNN predicts which variables belong to the *backbone*
     (variables fixed across all optimal solutions). These predictions define a
     **trust region** that restricts the search, and Gurobi solves the
     restricted instance under a short time budget (`--trust_region_time`),
     producing a strong incumbent.
   - **Phase 2**: Gurobi solves the **original** instance using the Phase-1
     incumbent as a **warm-start**, recovering the global optimality guarantee
     that the restricted instance cannot provide on its own.

Each instance is run with multiple seeds so that the only source of variability
is Gurobi's internal stochasticity — the GNN predictions are deterministic.

We report two complementary views:

- **Algorithmic work** (solver time only) — does the trust region + warm-start
  help Gurobi close the gap faster?
- **Wall-clock** (end-to-end, GNN overhead included) — is BACKPAS worth it in
  practice?

plus the **primal integral** (anytime behavior), **Wilcoxon** significance tests,
**Dolan–Moré performance profiles**, and **cactus plots**.

## Directory layout

```
thesis_experiments/
├── README.md                        # This file (overview)
├── README_EXPERIMENT.md             # Detailed step-by-step run guide
├── src/
│   ├── run_gurobi_experiment.py     # Core driver (BASELINE and BACKPAS modes)
│   ├── run_gurobi_baseline.py       # Baseline-only convenience driver
│   ├── run_multi_seed.py            # Runs an instance set across several seeds
│   └── analytics/
│       ├── make_table.py            # Aggregate runtimes -> time_table.csv
│       ├── make_primal_table.py     # Aggregate primal integral + Wilcoxon
│       ├── combine_seeds.py         # Merge per-seed CSVs
│       └── analyze_results.py       # Statistical analysis helpers
├── results/
│   ├── metrics/                     # Per-run CSVs and aggregated tables
│   ├── logs/                        # Gurobi logs
│   └── analysis/                    # Derived analysis artifacts
└── thesis/                          # LaTeX sources, figures, and tables
    ├── chapters/
    ├── figures/
    └── tables/
```

## Requirements

### Python dependencies

```bash
pip install -r ../requirements.txt
```

`torch_scatter` and `torch_sparse` must match your installed torch + CUDA build:

For CPU-only setups (e.g. Apple Silicon), follow the official PyTorch Geometric
installation guide: <https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html>

### Gurobi license

A valid Gurobi license is required. :

The experiments in the paper used **Gurobi 13.0.0**, **Python 3.12**, and
**PyTorch 2.8**.

### Trained model

BACKPAS needs the trained GNN checkpoint:

```
wkdir/MIS/ml_training/graph_with_literals_8_GTR/best_model.pth
```

## Quick start (reproduce the paper)

The full reproduction recipe — with the exact tuned hyperparameters, the
multi-seed runner, and the table generators — lives in
[`README_EXPERIMENT.md`](README_EXPERIMENT.md). The short version:

```bash
# 1) Run both Baseline and BACKPAS over an instance set across 5 seeds
python src/run_multi_seed.py \
    --instance_dir <path/to/test/instances> \
    --model wkdir/MIS/ml_training/graph_with_literals_8_GTR/best_model.pth \
    --seeds 0 1 2 3 4 \
    --output_dir results/metrics/my_run \
    --threshold 0.6237 --alpha -0.8419 \
    --trust_region_time 300 --time_limit 3600

# 2) Aggregate runtimes into a table
python src/analytics/make_table.py --dir results/metrics/my_run

# 3) Aggregate the primal integral and run the paired Wilcoxon test
python src/analytics/make_primal_table.py --base_dir results/metrics
```

## Tuned hyperparameters (paper values)

| Parameter             | Symbol | Value      | Meaning                                            |
|-----------------------|--------|------------|----------------------------------------------------|
| `--threshold`         | θ      | `0.6237`   | Confidence cutoff for selecting backbone variables |
| `--alpha`             | α      | `-0.8419`  | Trust-region tolerance                             |
| `--trust_region_time` | —      | `300`      | Phase-1 duration (seconds)                         |
| `--time_limit`        | —      | `3600`     | Total per-instance budget (seconds)                |
| `--threads`           | —      | `1`        | Single-threaded for deterministic, fair runs       |

> `MIPGap` is fixed to `0` so the solver must **prove** optimality rather than
> stop at a tolerance, and `Threads=1` keeps the search reproducible.

> To train the model from scratch (rather than using the provided checkpoint), refer to the original BACKPAS documentation: <https://github.com/bryan-alvarado-ulloa/backpas>.

## Collected metrics

| Metric             | Description                                                        |
|--------------------|-------------------------------------------------------------------|
| `runtime`          | Total wall-clock time (Python), including all GNN overhead         |
| `gurobi_runtime`   | Solver-internal time reported by Gurobi                            |
| `phase1_time`      | Phase-1 (trust region) duration — BACKPAS only                     |
| `phase2_time`      | Phase-2 (warm-started original instance) duration — BACKPAS only   |
| `primal_integral`  | Time-integral of the normalized primal gap (lower is better)       |
| `obj_val`          | Objective value of the returned solution                          |
| `obj_bound`        | Best dual bound                                                    |
| `mip_gap`          | Final optimality gap                                              |
| `status_name`      | `OPTIMAL` / `TIME_LIMIT` / ...                                     |

> The **primal integral** is measured on the solver's internal clock and
> therefore **excludes** the GNN preprocessing (~0.35 s). It belongs with the
> *algorithmic-work* comparison, not the wall-clock one.

For the thesis tables: baseline uses `gurobi_runtime`; BACKPAS uses `runtime`
(total wall-clock). Standard deviation across seeds uses `ddof=1` (sample std).

## Acknowledgments

This work builds directly on the original **BACKPAS** implementation by
Bryan Alvarado Ulloa:

> <https://github.com/bryan-alvarado-ulloa/backpas>

We are grateful for that codebase, which provided the GNN architecture
(literal-based bipartite Graph Transformer), the trust-region construction, and
the trained model that make these experiments possible. This repository extends
it with the two-phase warm-start driver, the multi-seed experiment runner, and
the statistical analysis used in the paper.
