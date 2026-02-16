#!/usr/bin/env python3
"""
Script para análisis estadístico de resultados experimentales.

Compara resultados entre baseline y BACKPAS, genera estadísticas
y realiza tests estadísticos.

"""

import argparse
import pandas as pd
import numpy as np
from scipy import stats
import os
from pathlib import Path
from typing import Tuple, Optional


def load_results(csv_path: str) -> pd.DataFrame:
    """Carga resultados desde CSV."""
    df = pd.read_csv(csv_path)
    return df


def compare_experiments(
    baseline_csv: str,
    backbone_csv: str,
    output_dir: Optional[str] = None
) -> dict:
    """
    Compara resultados entre experimentos baseline y BACKPAS.

    Args:
        baseline_csv: Ruta al CSV de resultados baseline
        backbone_csv: Ruta al CSV de resultados BACKPAS
        output_dir: Directorio para guardar reportes

    Returns:
        Diccionario con resultados de la comparación
    """
    df_base = load_results(baseline_csv)
    df_back = load_results(backbone_csv)

    print("=" * 70)
    print("ANÁLISIS COMPARATIVO: BASELINE vs BACKPAS")
    print("=" * 70)

    print(f"\nInstancias baseline: {len(df_base)}")
    print(f"Instancias BACKPAS: {len(df_back)}")

    # Merge por nombre de instancia
    df_merged = pd.merge(
        df_base, df_back,
        on='instance_name',
        suffixes=('_baseline', '_backpas')
    )

    print(f"Instancias pareadas: {len(df_merged)}")

    if len(df_merged) == 0:
        print("ERROR: No hay instancias comunes entre ambos experimentos")
        return {}

    results = {}

    # 0. Verificar que ambos métodos alcanzan el mismo óptimo
    if 'obj_val_baseline' in df_merged.columns and 'obj_val_backpas' in df_merged.columns:
        df_obj = df_merged.dropna(subset=['obj_val_baseline', 'obj_val_backpas'])
        if len(df_obj) > 0:
            df_obj['obj_match'] = np.isclose(df_obj['obj_val_baseline'], df_obj['obj_val_backpas'], rtol=1e-6)
            n_match = df_obj['obj_match'].sum()
            n_total = len(df_obj)

            results['objective'] = {
                'instances_matching': int(n_match),
                'instances_total': n_total,
                'all_match': n_match == n_total,
                'mean_baseline': df_obj['obj_val_baseline'].mean(),
                'mean_backpas': df_obj['obj_val_backpas'].mean(),
            }

            print("\n" + "-" * 50)
            print("VALOR OBJETIVO")
            print("-" * 50)
            print(f"  Media baseline: {results['objective']['mean_baseline']:.2f}")
            print(f"  Media BACKPAS:  {results['objective']['mean_backpas']:.2f}")
            print(f"  Coinciden: {n_match}/{n_total}")
            if n_match < n_total:
                mismatches = df_obj[~df_obj['obj_match']]
                print("  ADVERTENCIA: Diferencias en:")
                for _, row in mismatches.iterrows():
                    print(f"    {row['instance_name']}: baseline={row['obj_val_baseline']}, backpas={row['obj_val_backpas']}")

    # 1. Mejora en tiempo
    if 'runtime_baseline' in df_merged.columns:
        # Evitar division por zero
        valid_time = df_merged[df_merged['runtime_baseline'] > 0].copy()

        if len(valid_time) > 0:
            valid_time['speedup'] = valid_time['runtime_baseline'] / valid_time['runtime_backpas']
            valid_time['time_reduction'] = (valid_time['runtime_baseline'] - valid_time['runtime_backpas']) / valid_time['runtime_baseline'] * 100

            results['time'] = {
                'mean_baseline': valid_time['runtime_baseline'].mean(),
                'mean_backpas': valid_time['runtime_backpas'].mean(),
                'mean_speedup': valid_time['speedup'].mean(),
                'median_speedup': valid_time['speedup'].median(),
                'mean_time_reduction_pct': valid_time['time_reduction'].mean(),
                'instances_improved': int((valid_time['runtime_backpas'] < valid_time['runtime_baseline']).sum()),
                'instances_total': len(valid_time)
            }

            print("\n" + "-" * 50)
            print("TIEMPO DE EJECUCIÓN")
            print("-" * 50)
            print(f"  Media baseline: {results['time']['mean_baseline']:.2f} seg")
            print(f"  Media BACKPAS:  {results['time']['mean_backpas']:.2f} seg")
            print(f"  Speedup medio:  {results['time']['mean_speedup']:.2f}x")
            print(f"  Speedup mediano: {results['time']['median_speedup']:.2f}x")
            print(f"  Reducción media: {results['time']['mean_time_reduction_pct']:.1f}%")
            print(f"  Instancias mejoradas: {results['time']['instances_improved']}/{results['time']['instances_total']}")

    # 2. Mejora en nodos explorados
    # Para BACKPAS, usar total_nodes si está disponible (phase1 + phase2)
    nodes_col_backpas = 'total_nodes_backpas' if 'total_nodes_backpas' in df_merged.columns else 'n_nodes_backpas'

    if 'n_nodes_baseline' in df_merged.columns and nodes_col_backpas in df_merged.columns:
        valid_nodes = df_merged[df_merged['n_nodes_baseline'] > 0].copy()

        if len(valid_nodes) > 0:
            valid_nodes['node_reduction'] = (
                (valid_nodes['n_nodes_baseline'] - valid_nodes[nodes_col_backpas])
                / valid_nodes['n_nodes_baseline'] * 100
            )

            results['nodes'] = {
                'mean_baseline': valid_nodes['n_nodes_baseline'].mean(),
                'mean_backpas': valid_nodes[nodes_col_backpas].mean(),
                'mean_reduction_pct': valid_nodes['node_reduction'].mean(),
                'instances_improved': int((valid_nodes[nodes_col_backpas] < valid_nodes['n_nodes_baseline']).sum()),
                'instances_total': len(valid_nodes),
                'nodes_column_used': nodes_col_backpas,
            }

            print("\n" + "-" * 50)
            print("NODOS EXPLORADOS")
            print("-" * 50)
            print(f"  Media baseline: {results['nodes']['mean_baseline']:.0f}")
            print(f"  Media BACKPAS:  {results['nodes']['mean_backpas']:.0f}")
            print(f"  Reducción media: {results['nodes']['mean_reduction_pct']:.1f}%")
            print(f"  Instancias mejoradas: {results['nodes']['instances_improved']}/{results['nodes']['instances_total']}")

    # 3. Primal integral
    if 'primal_integral_baseline' in df_merged.columns and 'primal_integral_backpas' in df_merged.columns:
        df_pi = df_merged.dropna(subset=['primal_integral_baseline', 'primal_integral_backpas'])
        df_pi = df_pi[df_pi['primal_integral_baseline'] > 0].copy()

        if len(df_pi) > 0:
            df_pi['pi_reduction'] = (
                (df_pi['primal_integral_baseline'] - df_pi['primal_integral_backpas'])
                / df_pi['primal_integral_baseline'] * 100
            )
            df_pi['pi_ratio'] = df_pi['primal_integral_baseline'] / df_pi['primal_integral_backpas']

            results['primal_integral'] = {
                'mean_baseline': df_pi['primal_integral_baseline'].mean(),
                'mean_backpas': df_pi['primal_integral_backpas'].mean(),
                'mean_reduction_pct': df_pi['pi_reduction'].mean(),
                'median_reduction_pct': df_pi['pi_reduction'].median(),
                'mean_ratio': df_pi['pi_ratio'].mean(),
                'instances_improved': int((df_pi['primal_integral_backpas'] < df_pi['primal_integral_baseline']).sum()),
                'instances_with_data': len(df_pi)
            }

            print("\n" + "-" * 50)
            print("PRIMAL INTEGRAL (menor = mejor)")
            print("-" * 50)
            print(f"  Media baseline: {results['primal_integral']['mean_baseline']:.4f}")
            print(f"  Media BACKPAS:  {results['primal_integral']['mean_backpas']:.4f}")
            print(f"  Reducción media: {results['primal_integral']['mean_reduction_pct']:.1f}%")
            print(f"  Ratio medio (baseline/backpas): {results['primal_integral']['mean_ratio']:.2f}x")
            print(f"  Instancias mejoradas: {results['primal_integral']['instances_improved']}/{results['primal_integral']['instances_with_data']}")

    # 4. Detalles BACKPAS (fase 1 vs fase 2)
    if 'phase1_obj_backpas' in df_merged.columns:
        df_phases = df_merged.dropna(subset=['phase1_obj_backpas'])
        if len(df_phases) > 0:
            n_improved = 0
            if 'phase2_improved_backpas' in df_phases.columns:
                n_improved = int(df_phases['phase2_improved_backpas'].sum())

            results['phases'] = {
                'mean_phase1_time': df_phases['phase1_time_backpas'].mean() if 'phase1_time_backpas' in df_phases.columns else None,
                'mean_phase2_time': df_phases['phase2_time_backpas'].mean() if 'phase2_time_backpas' in df_phases.columns else None,
                'phase2_improved_count': n_improved,
                'total_instances': len(df_phases),
            }

            print("\n" + "-" * 50)
            print("DETALLES BACKPAS (FASES)")
            print("-" * 50)
            if results['phases']['mean_phase1_time'] is not None:
                print(f"  Tiempo medio Fase 1: {results['phases']['mean_phase1_time']:.2f} seg")
            if results['phases']['mean_phase2_time'] is not None:
                print(f"  Tiempo medio Fase 2: {results['phases']['mean_phase2_time']:.2f} seg")
            print(f"  Fase 2 mejoró Fase 1: {n_improved}/{len(df_phases)}")

    # 5. Tests estadísticos
    print("\n" + "-" * 50)
    print("TESTS ESTADÍSTICOS (Wilcoxon signed-rank)")
    print("-" * 50)

    results['statistical_tests'] = {}

    if len(df_merged) >= 5:
        # Test para tiempo
        try:
            stat, p_value = stats.wilcoxon(
                df_merged['runtime_baseline'],
                df_merged['runtime_backpas'],
                alternative='greater'  # H1: baseline > backpas (backpas es más rápido)
            )
            results['statistical_tests']['runtime'] = {
                'statistic': stat,
                'p_value': p_value,
                'significant_005': p_value < 0.05,
                'significant_001': p_value < 0.01
            }
            sig = "**" if p_value < 0.01 else ("*" if p_value < 0.05 else "")
            print(f"  Tiempo: W={stat:.2f}, p={p_value:.4f} {sig}")
        except ValueError as e:
            print(f"  Tiempo: no se pudo calcular ({e})")

        # Test para primal integral
        if 'primal_integral_baseline' in df_merged.columns and 'primal_integral_backpas' in df_merged.columns:
            df_pi_test = df_merged.dropna(subset=['primal_integral_baseline', 'primal_integral_backpas'])
            if len(df_pi_test) >= 5:
                try:
                    stat, p_value = stats.wilcoxon(
                        df_pi_test['primal_integral_baseline'],
                        df_pi_test['primal_integral_backpas'],
                        alternative='greater'  # H1: baseline > backpas (backpas tiene menor PI)
                    )
                    results['statistical_tests']['primal_integral'] = {
                        'statistic': stat,
                        'p_value': p_value,
                        'significant_005': p_value < 0.05,
                        'significant_001': p_value < 0.01
                    }
                    sig = "**" if p_value < 0.01 else ("*" if p_value < 0.05 else "")
                    print(f"  Primal integral: W={stat:.2f}, p={p_value:.4f} {sig}")
                except ValueError as e:
                    print(f"  Primal integral: no se pudo calcular ({e})")

        # Test para nodos
        if nodes_col_backpas in df_merged.columns:
            try:
                stat, p_value = stats.wilcoxon(
                    df_merged['n_nodes_baseline'],
                    df_merged[nodes_col_backpas],
                    alternative='greater'
                )
                results['statistical_tests']['nodes'] = {
                    'statistic': stat,
                    'p_value': p_value,
                    'significant_005': p_value < 0.05
                }
                sig = "**" if p_value < 0.01 else ("*" if p_value < 0.05 else "")
                print(f"  Nodos: W={stat:.2f}, p={p_value:.4f} {sig}")
            except ValueError as e:
                print(f"  Nodos: no se pudo calcular ({e})")
    else:
        print("  (Insuficientes datos para test estadístico, se requieren >= 5 pares)")

    print("\n* = significativo al 5%, ** = significativo al 1%")

    # Guardar resultados detallados
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

        merged_path = os.path.join(output_dir, "comparison_detailed.csv")
        df_merged.to_csv(merged_path, index=False)
        print(f"\nResultados detallados guardados en: {merged_path}")

        summary_path = os.path.join(output_dir, "comparison_summary.txt")
        with open(summary_path, 'w') as f:
            f.write("RESUMEN DE COMPARACIÓN BASELINE vs BACKPAS\n")
            f.write("=" * 50 + "\n\n")

            for section_name, section_data in results.items():
                if isinstance(section_data, dict):
                    f.write(f"{section_name.upper()}:\n")
                    for k, v in section_data.items():
                        if isinstance(v, dict):
                            f.write(f"  {k}:\n")
                            for k2, v2 in v.items():
                                f.write(f"    {k2}: {v2}\n")
                        else:
                            f.write(f"  {k}: {v}\n")
                    f.write("\n")

        print(f"Resumen guardado en: {summary_path}")

    return results


def generate_latex_table(
    baseline_csv: str,
    backbone_csv: str,
    output_path: Optional[str] = None
) -> str:
    """
    Genera una tabla LaTeX con los resultados comparativos.

    Args:
        baseline_csv: CSV de resultados baseline
        backbone_csv: CSV de resultados BACKPAS
        output_path: Ruta para guardar la tabla

    Returns:
        String con la tabla LaTeX
    """
    df_base = load_results(baseline_csv)
    df_back = load_results(backbone_csv)

    df_merged = pd.merge(
        df_base, df_back,
        on='instance_name',
        suffixes=('_base', '_back')
    )

    n = len(df_merged)

    # Calcular métricas
    mean_base_t = df_merged['runtime_base'].mean()
    mean_back_t = df_merged['runtime_back'].mean()
    mejora_t = (mean_base_t - mean_back_t) / mean_base_t * 100 if mean_base_t > 0 else 0

    # Nodos - usar total_nodes para BACKPAS si disponible
    nodes_col = 'total_nodes_back' if 'total_nodes_back' in df_merged.columns else 'n_nodes_back'
    mean_base_n = df_merged['n_nodes_base'].mean()
    mean_back_n = df_merged[nodes_col].mean()
    mejora_n = (mean_base_n - mean_back_n) / mean_base_n * 100 if mean_base_n > 0 else 0

    # Primal integral
    has_pi = ('primal_integral_base' in df_merged.columns and
              'primal_integral_back' in df_merged.columns)
    if has_pi:
        df_pi = df_merged.dropna(subset=['primal_integral_base', 'primal_integral_back'])
        mean_base_pi = df_pi['primal_integral_base'].mean()
        mean_back_pi = df_pi['primal_integral_back'].mean()
        mejora_pi = (mean_base_pi - mean_back_pi) / mean_base_pi * 100 if mean_base_pi > 0 else 0

    # Valor objetivo
    has_obj = ('obj_val_base' in df_merged.columns and 'obj_val_back' in df_merged.columns)
    if has_obj:
        df_obj = df_merged.dropna(subset=['obj_val_base', 'obj_val_back'])
        mean_base_obj = df_obj['obj_val_base'].mean()
        mean_back_obj = df_obj['obj_val_back'].mean()

    # p-valores
    p_t = p_n = p_pi = float('nan')
    if n >= 5:
        try:
            _, p_t = stats.wilcoxon(df_merged['runtime_base'], df_merged['runtime_back'], alternative='greater')
        except ValueError:
            pass
        try:
            _, p_n = stats.wilcoxon(df_merged['n_nodes_base'], df_merged[nodes_col], alternative='greater')
        except ValueError:
            pass
        if has_pi and len(df_pi) >= 5:
            try:
                _, p_pi = stats.wilcoxon(df_pi['primal_integral_base'], df_pi['primal_integral_back'], alternative='greater')
            except ValueError:
                pass

    def fmt_p(p):
        if np.isnan(p):
            return "--"
        if p < 0.001:
            return f"$<$0.001"
        return f"{p:.3f}"

    def sig_mark(p):
        if np.isnan(p):
            return ""
        if p < 0.01:
            return "$^{**}$"
        if p < 0.05:
            return "$^{*}$"
        return ""

    latex = r"""\begin{table}[htbp]
\centering
\caption{Comparación de rendimiento: Baseline vs BACKPAS (""" + str(n) + r""" instancias)}
\label{tab:comparison}
\begin{tabular}{lrrrr}
\toprule
\textbf{Métrica} & \textbf{Baseline} & \textbf{BACKPAS} & \textbf{Mejora (\%)} & \textbf{p-valor} \\
\midrule
"""

    # Valor objetivo
    if has_obj:
        latex += f"Valor objetivo & {mean_base_obj:.1f} & {mean_back_obj:.1f} & -- & -- \\\\\n"

    # Tiempo
    latex += f"Tiempo (seg) & {mean_base_t:.2f} & {mean_back_t:.2f} & {mejora_t:.1f} & {fmt_p(p_t)}{sig_mark(p_t)} \\\\\n"

    # Nodos
    latex += f"Nodos explorados & {mean_base_n:.0f} & {mean_back_n:.0f} & {mejora_n:.1f} & {fmt_p(p_n)}{sig_mark(p_n)} \\\\\n"

    # Primal integral
    if has_pi:
        latex += f"Primal integral & {mean_base_pi:.4f} & {mean_back_pi:.4f} & {mejora_pi:.1f} & {fmt_p(p_pi)}{sig_mark(p_pi)} \\\\\n"

    latex += r"""\bottomrule
\end{tabular}
\vspace{0.5em}
\footnotesize{$^{*}$ $p < 0.05$, $^{**}$ $p < 0.01$ (Wilcoxon signed-rank, one-sided)}
\end{table}
"""

    if output_path:
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        with open(output_path, 'w') as f:
            f.write(latex)
        print(f"Tabla LaTeX guardada en: {output_path}")

    return latex


def main():
    parser = argparse.ArgumentParser(
        description="Análisis estadístico de resultados experimentales",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:
  # Comparar baseline vs BACKPAS
  python analyze_results.py \\
      --baseline ../results/metrics/baseline.csv \\
      --backbone ../results/metrics/backpas.csv \\
      --output_dir ../results/analysis

  # Generar tabla LaTeX
  python analyze_results.py \\
      --baseline ../results/metrics/baseline.csv \\
      --backbone ../results/metrics/backpas.csv \\
      --latex_table ../thesis/figures/comparison_table.tex
        """
    )

    parser.add_argument("--baseline", type=str, required=True,
                        help="CSV con resultados baseline")
    parser.add_argument("--backbone", type=str, required=True,
                        help="CSV con resultados BACKPAS")
    parser.add_argument("--output_dir", type=str, default="../results/analysis",
                        help="Directorio para guardar análisis")
    parser.add_argument("--latex_table", type=str,
                        help="Ruta para guardar tabla LaTeX")

    args = parser.parse_args()

    # Ejecutar comparación
    results = compare_experiments(
        baseline_csv=args.baseline,
        backbone_csv=args.backbone,
        output_dir=args.output_dir
    )

    # Generar tabla LaTeX si se solicita
    if args.latex_table:
        generate_latex_table(
            baseline_csv=args.baseline,
            backbone_csv=args.backbone,
            output_path=args.latex_table
        )


if __name__ == "__main__":
    main()
