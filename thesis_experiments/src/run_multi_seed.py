#!/usr/bin/env python3
"""
Ejecuta experimentos con múltiples seeds sobre el mismo dataset.

Para cada seed corre baseline y BACKPAS, guardando un CSV por seed.
Para agregar los resultados usa analyze_results.py.

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
from typing import Optional

from run_gurobi_experiment import run_batch_experiment


def run_multi_seed(
    instance_dir: str,
    model_path: str,
    seeds: list,
    output_dir: str,
    mode: str = "both",
    threads: int = 1,
    time_limit: float = 3600,
    trust_region_time: float = 300,
    threshold: float = 0.7,
    alpha: float = 0.0,
    log_dir: Optional[str] = None,
):
    os.makedirs(output_dir, exist_ok=True)

    run_baseline = mode in ("both", "baseline")
    run_backpas  = mode in ("both", "backpas")

    for seed in seeds:
        print(f"\n{'#'*60}")
        print(f"# SEED {seed}")
        print(f"{'#'*60}")

        seed_log_dir = os.path.join(log_dir, f"seed{seed}") if log_dir else None

        # ── Baseline ──────────────────────────────────────────────
        if run_baseline:
            run_batch_experiment(
                instance_dir=instance_dir,
                output_csv=os.path.join(output_dir, f"baseline_seed{seed}.csv"),
                threads=threads,
                time_limit=time_limit,
                log_dir=seed_log_dir,
                seed=seed,
                use_backpas=False,
            )

        # ── BACKPAS ───────────────────────────────────────────────
        if run_backpas:
            run_batch_experiment(
                instance_dir=instance_dir,
                output_csv=os.path.join(output_dir, f"backpas_seed{seed}.csv"),
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

    print(f"\nSeed(s) {seeds} completadas en: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Ejecuta experimentos con múltiples seeds",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  # Ejecutar seeds 0 y 1
  python run_multi_seed.py \\
      --instance_dir dataset/MIS/instance/valid_barabasi_albert_1333 \\
      --model wkdir/MIS/ml_training/graph_with_literals_8_GTR/best_model.pth \\
      --seeds 0 1 --output_dir thesis_experiments/results/metrics/multi_seed \\
      --threshold 0.6237 --alpha -0.8419

  # Ejecutar seeds 2 3 4 más tarde
  python run_multi_seed.py ... --seeds 2 3 4 --output_dir ...
        """
    )

    parser.add_argument("--instance_dir", type=str, required=True,
                        help="Directorio con instancias")
    parser.add_argument("--model", type=str, required=True,
                        help="Ruta al modelo GNN .pth")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4],
                        help="Lista de seeds (default: 0 1 2 3 4)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directorio para guardar resultados por seed")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--time_limit", type=float, default=3600)
    parser.add_argument("--trust_region_time", type=float, default=300)
    parser.add_argument("--threshold", type=float, default=0.7)
    parser.add_argument("--alpha", type=float, default=0.0)
    parser.add_argument("--log_dir", type=str, default=None,
                        help="Directorio base para logs (se crea subdirectorio por seed)")

    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--baseline", action="store_true",
                            help="Ejecutar solo baseline")
    mode_group.add_argument("--backpas", action="store_true",
                            help="Ejecutar solo BACKPAS")

    args = parser.parse_args()

    if args.baseline:
        mode = "baseline"
    elif args.backpas:
        mode = "backpas"
    else:
        mode = "both"

    run_multi_seed(
        instance_dir=args.instance_dir,
        model_path=args.model,
        seeds=args.seeds,
        output_dir=args.output_dir,
        mode=mode,
        threads=args.threads,
        time_limit=args.time_limit,
        trust_region_time=args.trust_region_time,
        threshold=args.threshold,
        alpha=args.alpha,
        log_dir=args.log_dir,
    )


if __name__ == "__main__":
    main()
