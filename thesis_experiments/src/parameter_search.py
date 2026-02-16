#!/usr/bin/env python3
"""
Script para búsqueda de parámetros óptimos de BACKPAS.

Ejecuta BACKPAS con una grilla de valores (threshold, alpha) sobre un conjunto
de instancias y genera un resumen comparativo para encontrar la mejor configuración.

Uso:
  python parameter_search.py \
      --instance_dir ../instances/calibration \
      --model_path ../../wkdir/MIS/ml_training/graph_with_literals_8_GTR/best_model.pth \
      --output_dir ../results/param_search \
      --time_limit 3600

  # Con grilla personalizada
  python parameter_search.py \
      --instance_dir ../instances/calibration \
      --model_path ../../wkdir/MIS/ml_training/graph_with_literals_8_GTR/best_model.pth \
      --output_dir ../results/param_search \
      --thresholds 0.5 0.6 0.7 0.8 0.9 \
      --alphas 0.0 0.05 0.1 0.2 \
      --trust_region_times 60 180 300 600
"""

import argparse
import os
import sys
import csv
import itertools
import pandas as pd
from pathlib import Path
from glob import glob
from datetime import datetime

# Agregar src/ al path
SCRIPT_DIR = Path(__file__).parent
REPO_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(SCRIPT_DIR))

from run_gurobi_experiment import GurobiMISExperiment, save_metrics_to_csv


def run_parameter_search(
    instance_dir: str,
    model_path: str,
    output_dir: str,
    thresholds: list,
    alphas: list,
    trust_region_times: list,
    time_limit: float = 3600,
    threads: int = 1,
):
    """
    Ejecuta búsqueda de parámetros sobre una grilla.

    Args:
        instance_dir: Directorio con instancias .lp
        model_path: Ruta al modelo .pth
        output_dir: Directorio para guardar resultados
        thresholds: Lista de valores θ a probar
        alphas: Lista de valores α a probar
        trust_region_times: Lista de tiempos de trust region a probar
        time_limit: Tiempo límite por instancia
        threads: Número de hilos
    """
    # Encontrar instancias
    instance_files = sorted(glob(os.path.join(instance_dir, "*.lp")))
    if not instance_files:
        print(f"ERROR: No se encontraron archivos .lp en {instance_dir}")
        return

    os.makedirs(output_dir, exist_ok=True)

    # Generar todas las combinaciones
    configs = list(itertools.product(thresholds, alphas, trust_region_times))
    total_runs = len(configs) * len(instance_files)

    print("=" * 70)
    print("BÚSQUEDA DE PARÁMETROS BACKPAS")
    print("=" * 70)
    print(f"Instancias: {len(instance_files)}")
    print(f"Configuraciones: {len(configs)}")
    print(f"  - Thresholds (θ): {thresholds}")
    print(f"  - Alphas (α): {alphas}")
    print(f"  - Trust region times: {trust_region_times}")
    print(f"Ejecuciones totales: {total_runs}")
    print(f"Tiempo límite por instancia: {time_limit}s")
    print(f"Directorio de salida: {output_dir}")
    print("=" * 70)

    # También ejecutar baseline para comparación
    print("\n[0/{0}] Ejecutando BASELINE como referencia...".format(len(configs)))
    baseline_metrics = []
    baseline_experiment = GurobiMISExperiment(
        threads=threads,
        time_limit=time_limit,
        use_backpas=False,
    )

    for i, instance_path in enumerate(instance_files):
        print(f"  BASELINE [{i+1}/{len(instance_files)}] {Path(instance_path).stem}")
        metrics = baseline_experiment.run_instance(instance_path, verbose=False)
        metrics_csv = {k: v for k, v in metrics.items() if k != 'incumbent_history'}
        baseline_metrics.append(metrics_csv)

    baseline_csv = os.path.join(output_dir, "baseline.csv")
    save_metrics_to_csv(baseline_metrics, baseline_csv)
    print(f"  Baseline guardado en: {baseline_csv}")

    # Ejecutar cada configuración
    all_summaries = []

    for config_idx, (threshold, alpha, tr_time) in enumerate(configs):
        config_name = f"th{threshold}_a{alpha}_tr{int(tr_time)}"
        print(f"\n[{config_idx+1}/{len(configs)}] Configuración: θ={threshold}, α={alpha}, TR={tr_time}s")

        experiment = GurobiMISExperiment(
            threads=threads,
            time_limit=time_limit,
            use_backpas=True,
            model_path=model_path,
            trust_region_time=tr_time,
            threshold=threshold,
            alpha=alpha,
        )

        config_metrics = []
        for i, instance_path in enumerate(instance_files):
            instance_name = Path(instance_path).stem
            print(f"  [{i+1}/{len(instance_files)}] {instance_name}", end=" ")

            try:
                metrics = experiment.run_instance(instance_path, verbose=False)
                metrics_csv = {k: v for k, v in metrics.items() if k != 'incumbent_history'}
                config_metrics.append(metrics_csv)
                status = metrics['status_name']
                obj = metrics.get('obj_val', 'N/A')
                pi = metrics.get('primal_integral', 'N/A')
                if isinstance(pi, float):
                    pi = f"{pi:.4f}"
                print(f"-> {status}, obj={obj}, PI={pi}")
            except Exception as e:
                print(f"-> ERROR: {e}")
                config_metrics.append({
                    'instance_name': instance_name,
                    'method': 'backpas',
                    'status_name': 'ERROR',
                    'runtime': None,
                    'obj_val': None,
                    'primal_integral': None,
                    'threshold': threshold,
                    'alpha': alpha,
                    'trust_region_time': tr_time,
                    'error': str(e),
                })

        # Guardar resultados de esta configuración
        config_csv = os.path.join(output_dir, f"backpas_{config_name}.csv")
        save_metrics_to_csv(config_metrics, config_csv)

        # Calcular resumen
        df = pd.DataFrame(config_metrics)
        valid = df[df['status_name'].isin(['OPTIMAL', 'TIME_LIMIT'])]

        summary = {
            'threshold': threshold,
            'alpha': alpha,
            'trust_region_time': tr_time,
            'config_name': config_name,
            'n_instances': len(instance_files),
            'n_optimal': len(df[df['status_name'] == 'OPTIMAL']),
            'n_errors': len(df[df['status_name'] == 'ERROR']),
        }

        if len(valid) > 0:
            summary['mean_runtime'] = valid['runtime'].mean()
            summary['median_runtime'] = valid['runtime'].median()
            summary['mean_primal_integral'] = valid['primal_integral'].mean() if 'primal_integral' in valid.columns else None
            summary['median_primal_integral'] = valid['primal_integral'].median() if 'primal_integral' in valid.columns else None
            summary['mean_obj_val'] = valid['obj_val'].mean() if 'obj_val' in valid.columns else None
        else:
            summary['mean_runtime'] = None
            summary['median_runtime'] = None
            summary['mean_primal_integral'] = None
            summary['median_primal_integral'] = None
            summary['mean_obj_val'] = None

        all_summaries.append(summary)

    # Agregar baseline al resumen
    df_base = pd.DataFrame(baseline_metrics)
    valid_base = df_base[df_base['status_name'].isin(['OPTIMAL', 'TIME_LIMIT'])]
    baseline_summary = {
        'threshold': '-',
        'alpha': '-',
        'trust_region_time': '-',
        'config_name': 'BASELINE',
        'n_instances': len(instance_files),
        'n_optimal': len(df_base[df_base['status_name'] == 'OPTIMAL']),
        'n_errors': 0,
    }
    if len(valid_base) > 0:
        baseline_summary['mean_runtime'] = valid_base['runtime'].mean()
        baseline_summary['median_runtime'] = valid_base['runtime'].median()
        baseline_summary['mean_primal_integral'] = valid_base['primal_integral'].mean()
        baseline_summary['median_primal_integral'] = valid_base['primal_integral'].median()
        baseline_summary['mean_obj_val'] = valid_base['obj_val'].mean()
    all_summaries.insert(0, baseline_summary)

    # Guardar resumen
    summary_csv = os.path.join(output_dir, "parameter_search_summary.csv")
    summary_df = pd.DataFrame(all_summaries)
    summary_df.to_csv(summary_csv, index=False)

    # Imprimir resumen
    print("\n" + "=" * 70)
    print("RESUMEN DE BÚSQUEDA DE PARÁMETROS")
    print("=" * 70)

    # Ordenar por primal integral (menor es mejor)
    display_cols = [
        'config_name', 'threshold', 'alpha', 'trust_region_time',
        'n_optimal', 'mean_runtime', 'mean_primal_integral', 'mean_obj_val'
    ]
    available_cols = [c for c in display_cols if c in summary_df.columns]
    print(summary_df[available_cols].to_string(index=False))

    # Encontrar mejor configuración
    backpas_summaries = summary_df[summary_df['config_name'] != 'BASELINE']
    if len(backpas_summaries) > 0:
        valid_configs = backpas_summaries.dropna(subset=['mean_primal_integral'])
        if len(valid_configs) > 0:
            best_pi = valid_configs.loc[valid_configs['mean_primal_integral'].idxmin()]
            print(f"\nMEJOR CONFIG (primal integral): θ={best_pi['threshold']}, "
                  f"α={best_pi['alpha']}, TR={best_pi['trust_region_time']}s "
                  f"(PI={best_pi['mean_primal_integral']:.4f})")

            best_time = valid_configs.loc[valid_configs['mean_runtime'].idxmin()]
            print(f"MEJOR CONFIG (tiempo):          θ={best_time['threshold']}, "
                  f"α={best_time['alpha']}, TR={best_time['trust_region_time']}s "
                  f"(t={best_time['mean_runtime']:.2f}s)")

    print(f"\nResumen guardado en: {summary_csv}")
    print(f"CSVs individuales en: {output_dir}/")

    return summary_df


def main():
    parser = argparse.ArgumentParser(
        description="Búsqueda de parámetros óptimos para BACKPAS",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  # Búsqueda con grilla por defecto
  python parameter_search.py \\
      --instance_dir ../instances/calibration \\
      --model_path ../../wkdir/MIS/ml_training/graph_with_literals_8_GTR/best_model.pth \\
      --output_dir ../results/param_search

  # Grilla personalizada
  python parameter_search.py \\
      --instance_dir ../instances/calibration \\
      --model_path ../../wkdir/MIS/ml_training/graph_with_literals_8_GTR/best_model.pth \\
      --output_dir ../results/param_search \\
      --thresholds 0.5 0.6 0.7 0.8 0.9 \\
      --alphas 0.0 0.1 0.2 \\
      --trust_region_times 60 300 600
        """
    )

    parser.add_argument("--instance_dir", type=str, required=True,
                        help="Directorio con instancias .lp")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Ruta al modelo .pth")
    parser.add_argument("--output_dir", type=str, default="../results/param_search",
                        help="Directorio para resultados")
    parser.add_argument("--thresholds", type=float, nargs="+",
                        default=[0.5, 0.6, 0.7, 0.8, 0.9],
                        help="Valores de threshold θ (default: 0.5 0.6 0.7 0.8 0.9)")
    parser.add_argument("--alphas", type=float, nargs="+",
                        default=[0.0, 0.05, 0.1, 0.2],
                        help="Valores de alpha α (default: 0.0 0.05 0.1 0.2)")
    parser.add_argument("--trust_region_times", type=float, nargs="+",
                        default=[300],
                        help="Tiempos de trust region en segundos (default: 300)")
    parser.add_argument("--time_limit", type=float, default=3600,
                        help="Tiempo límite por instancia (default: 3600)")
    parser.add_argument("--threads", type=int, default=1,
                        help="Número de hilos (default: 1)")

    args = parser.parse_args()

    run_parameter_search(
        instance_dir=args.instance_dir,
        model_path=args.model_path,
        output_dir=args.output_dir,
        thresholds=args.thresholds,
        alphas=args.alphas,
        trust_region_times=args.trust_region_times,
        time_limit=args.time_limit,
        threads=args.threads,
    )


if __name__ == "__main__":
    main()
