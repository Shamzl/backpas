#!/usr/bin/env python3
"""
Ejecuta experimentos con múltiples seeds y agrega los resultados.

Para cada seed corre baseline y BACKPAS sobre el mismo dataset,
luego calcula media ± desviación estándar por instancia.

Uso:
  python run_multi_seed.py \\
      --instance_dir dataset/MIS/instance/valid_barabasi_albert_1333 \\
      --model wkdir/MIS/ml_training/graph_with_literals_8_GTR/best_model.pth \\
      --seeds 0 1 2 3 4 \\
      --output_dir thesis_experiments/results/metrics/multi_seed \\
      --threshold 0.6237 --alpha -0.8419 --trust_region_time 300
"""

import argparse
import os
import pandas as pd
import numpy as np
from pathlib import Path

from run_gurobi_experiment import run_batch_experiment


def run_multi_seed(
    instance_dir: str,
    model_path: str,
    seeds: list,
    output_dir: str,
    threads: int = 1,
    time_limit: float = 3600,
    trust_region_time: float = 300,
    threshold: float = 0.7,
    alpha: float = 0.0,
    log_dir: str = None,
):
    os.makedirs(output_dir, exist_ok=True)

    baseline_csvs = []
    backpas_csvs = []

    for seed in seeds:
        print(f"\n{'#'*60}")
        print(f"# SEED {seed}")
        print(f"{'#'*60}")

        # ── Baseline ──────────────────────────────────────────────
        baseline_csv = os.path.join(output_dir, f"baseline_seed{seed}.csv")
        seed_log_dir = os.path.join(log_dir, f"seed{seed}") if log_dir else None

        run_batch_experiment(
            instance_dir=instance_dir,
            output_csv=baseline_csv,
            threads=threads,
            time_limit=time_limit,
            log_dir=seed_log_dir,
            seed=seed,
            use_backpas=False,
        )
        baseline_csvs.append(baseline_csv)

        # ── BACKPAS ───────────────────────────────────────────────
        backpas_csv = os.path.join(output_dir, f"backpas_seed{seed}.csv")

        run_batch_experiment(
            instance_dir=instance_dir,
            output_csv=backpas_csv,
            threads=threads,
            time_limit=time_limit,
            log_dir=seed_log_dir,
            seed=seed,
            use_backpas=True,
            model_path=model_path,
            trust_region_time=trust_region_time,
            threshold=threshold,
            alpha=alpha,
        )
        backpas_csvs.append(backpas_csv)

    # ── Agregar resultados ─────────────────────────────────────────
    print(f"\n{'='*60}")
    print("AGREGANDO RESULTADOS")
    print(f"{'='*60}")

    aggregate(baseline_csvs, os.path.join(output_dir, "baseline_aggregated.csv"), "baseline")
    aggregate(backpas_csvs, os.path.join(output_dir, "backpas_aggregated.csv"), "backpas")

    print(f"\nArchivos guardados en: {output_dir}")


def aggregate(csv_paths: list, output_path: str, label: str):
    """
    Carga todos los CSVs de distintas seeds y calcula media ± std
    por instancia para las métricas numéricas relevantes.
    """
    dfs = []
    for i, path in enumerate(csv_paths):
        if not os.path.exists(path):
            print(f"  ADVERTENCIA: no existe {path}, se omite")
            continue
        df = pd.read_csv(path)
        df['seed'] = i
        dfs.append(df)

    if not dfs:
        print(f"  ERROR: no hay datos para agregar ({label})")
        return

    all_data = pd.concat(dfs, ignore_index=True)

    # Columnas numéricas a agregar
    numeric_cols = [
        'runtime', 'gurobi_runtime', 'obj_val', 'n_nodes',
        'primal_integral', 'phase1_time', 'phase2_time',
        'phase1_obj', 'phase2_obj', 'total_nodes',
    ]
    numeric_cols = [c for c in numeric_cols if c in all_data.columns]

    agg_mean = all_data.groupby('instance_name')[numeric_cols].mean()
    agg_std  = all_data.groupby('instance_name')[numeric_cols].std()
    agg_min  = all_data.groupby('instance_name')[numeric_cols].min()
    agg_max  = all_data.groupby('instance_name')[numeric_cols].max()

    # Renombrar columnas
    agg_mean.columns = [f"{c}_mean" for c in agg_mean.columns]
    agg_std.columns  = [f"{c}_std"  for c in agg_std.columns]
    agg_min.columns  = [f"{c}_min"  for c in agg_min.columns]
    agg_max.columns  = [f"{c}_max"  for c in agg_max.columns]

    result = pd.concat([agg_mean, agg_std, agg_min, agg_max], axis=1)
    result = result.reset_index()

    result.to_csv(output_path, index=False)

    # Resumen en pantalla
    print(f"\n  {label.upper()} — {len(dfs)} seeds, {len(result)} instancias")
    if 'runtime_mean' in result.columns:
        print(f"  Tiempo medio:   {result['runtime_mean'].mean():.2f}s "
              f"(± {result['runtime_std'].mean():.2f}s)")
    if 'primal_integral_mean' in result.columns:
        pi_valid = result['primal_integral_mean'].dropna()
        if len(pi_valid) > 0:
            print(f"  Primal integral medio: {pi_valid.mean():.4f} "
                  f"(± {result['primal_integral_std'].mean():.4f})")
    print(f"  Guardado en: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Ejecuta experimentos con múltiples seeds y agrega resultados",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplo:
  python run_multi_seed.py \\
      --instance_dir dataset/MIS/instance/valid_barabasi_albert_1333 \\
      --model wkdir/MIS/ml_training/graph_with_literals_8_GTR/best_model.pth \\
      --seeds 0 1 2 3 4 \\
      --output_dir thesis_experiments/results/metrics/multi_seed \\
      --threshold 0.6237 --alpha -0.8419 --trust_region_time 300
        """
    )

    parser.add_argument("--instance_dir", type=str, required=True,
                        help="Directorio con instancias")
    parser.add_argument("--model", type=str, required=True,
                        help="Ruta al modelo GNN .pth")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4],
                        help="Lista de seeds (default: 0 1 2 3 4)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directorio para guardar resultados por seed y agregados")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--time_limit", type=float, default=3600)
    parser.add_argument("--trust_region_time", type=float, default=300)
    parser.add_argument("--threshold", type=float, default=0.7)
    parser.add_argument("--alpha", type=float, default=0.0)
    parser.add_argument("--log_dir", type=str, default=None,
                        help="Directorio base para logs (se crea subdirectorio por seed)")

    args = parser.parse_args()

    run_multi_seed(
        instance_dir=args.instance_dir,
        model_path=args.model,
        seeds=args.seeds,
        output_dir=args.output_dir,
        threads=args.threads,
        time_limit=args.time_limit,
        trust_region_time=args.trust_region_time,
        threshold=args.threshold,
        alpha=args.alpha,
        log_dir=args.log_dir,
    )


if __name__ == "__main__":
    main()
