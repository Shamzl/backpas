#!/usr/bin/env python3
"""
Script para análisis estadístico de resultados experimentales.

Compara resultados entre baseline y backbone, genera estadísticas
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


def compute_summary_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula estadísticas resumen de los resultados.
    
    Args:
        df: DataFrame con resultados
    
    Returns:
        DataFrame con estadísticas
    """
    numeric_cols = ['runtime', 'gurobi_runtime', 'n_nodes', 'mip_gap', 'primal_integral']
    
    stats_dict = {}
    for col in numeric_cols:
        if col in df.columns:
            stats_dict[col] = {
                'mean': df[col].mean(),
                'std': df[col].std(),
                'median': df[col].median(),
                'min': df[col].min(),
                'max': df[col].max(),
                'count': df[col].count()
            }
    
    return pd.DataFrame(stats_dict).T


def compare_experiments(
    baseline_csv: str,
    backbone_csv: str,
    output_dir: Optional[str] = None
) -> dict:
    """
    Compara resultados entre experimentos baseline y backbone.
    
    Args:
        baseline_csv: Ruta al CSV de resultados baseline
        backbone_csv: Ruta al CSV de resultados con backbone
        output_dir: Directorio para guardar reportes
    
    Returns:
        Diccionario con resultados de la comparación
    """
    # Cargar datos
    df_base = load_results(baseline_csv)
    df_back = load_results(backbone_csv)
    
    print("="*70)
    print("ANÁLISIS COMPARATIVO: BASELINE vs BACKBONE")
    print("="*70)
    
    print(f"\nInstancias baseline: {len(df_base)}")
    print(f"Instancias backbone: {len(df_back)}")
    
    # Merge por nombre de instancia
    df_merged = pd.merge(
        df_base, df_back,
        on='instance_name',
        suffixes=('_baseline', '_backbone')
    )
    
    print(f"Instancias pareadas: {len(df_merged)}")
    
    if len(df_merged) == 0:
        print("ERROR: No hay instancias comunes entre ambos experimentos")
        return {}
    
    # Calcular mejoras
    results = {}
    
    # 1. Mejora en tiempo
    if 'runtime_baseline' in df_merged.columns:
        df_merged['speedup'] = df_merged['runtime_baseline'] / df_merged['runtime_backbone']
        df_merged['time_reduction'] = (df_merged['runtime_baseline'] - df_merged['runtime_backbone']) / df_merged['runtime_baseline'] * 100
        
        results['time'] = {
            'mean_baseline': df_merged['runtime_baseline'].mean(),
            'mean_backbone': df_merged['runtime_backbone'].mean(),
            'mean_speedup': df_merged['speedup'].mean(),
            'median_speedup': df_merged['speedup'].median(),
            'mean_time_reduction_pct': df_merged['time_reduction'].mean(),
            'instances_improved': (df_merged['runtime_backbone'] < df_merged['runtime_baseline']).sum(),
            'instances_total': len(df_merged)
        }
        
        print("\n" + "-"*50)
        print("TIEMPO DE EJECUCIÓN")
        print("-"*50)
        print(f"  Media baseline: {results['time']['mean_baseline']:.2f} seg")
        print(f"  Media backbone: {results['time']['mean_backbone']:.2f} seg")
        print(f"  Speedup medio: {results['time']['mean_speedup']:.2f}x")
        print(f"  Speedup mediano: {results['time']['median_speedup']:.2f}x")
        print(f"  Reducción media: {results['time']['mean_time_reduction_pct']:.1f}%")
        print(f"  Instancias mejoradas: {results['time']['instances_improved']}/{results['time']['instances_total']}")
    
    # 2. Mejora en nodos explorados
    if 'n_nodes_baseline' in df_merged.columns:
        df_merged['node_reduction'] = (df_merged['n_nodes_baseline'] - df_merged['n_nodes_backbone']) / df_merged['n_nodes_baseline'] * 100
        
        results['nodes'] = {
            'mean_baseline': df_merged['n_nodes_baseline'].mean(),
            'mean_backbone': df_merged['n_nodes_backbone'].mean(),
            'mean_reduction_pct': df_merged['node_reduction'].mean(),
            'instances_improved': (df_merged['n_nodes_backbone'] < df_merged['n_nodes_baseline']).sum()
        }
        
        print("\n" + "-"*50)
        print("NODOS EXPLORADOS")
        print("-"*50)
        print(f"  Media baseline: {results['nodes']['mean_baseline']:.0f}")
        print(f"  Media backbone: {results['nodes']['mean_backbone']:.0f}")
        print(f"  Reducción media: {results['nodes']['mean_reduction_pct']:.1f}%")
    
    # 3. Primal integral
    if 'primal_integral_baseline' in df_merged.columns and 'primal_integral_backbone' in df_merged.columns:
        # Filtrar NaN
        df_pi = df_merged.dropna(subset=['primal_integral_baseline', 'primal_integral_backbone'])
        
        if len(df_pi) > 0:
            df_pi['pi_reduction'] = (df_pi['primal_integral_baseline'] - df_pi['primal_integral_backbone']) / df_pi['primal_integral_baseline'] * 100
            
            results['primal_integral'] = {
                'mean_baseline': df_pi['primal_integral_baseline'].mean(),
                'mean_backbone': df_pi['primal_integral_backbone'].mean(),
                'mean_reduction_pct': df_pi['pi_reduction'].mean(),
                'instances_with_data': len(df_pi)
            }
            
            print("\n" + "-"*50)
            print("PRIMAL INTEGRAL")
            print("-"*50)
            print(f"  Media baseline: {results['primal_integral']['mean_baseline']:.4f}")
            print(f"  Media backbone: {results['primal_integral']['mean_backbone']:.4f}")
            print(f"  Reducción media: {results['primal_integral']['mean_reduction_pct']:.1f}%")
    
    # 4. Tests estadísticos
    print("\n" + "-"*50)
    print("TESTS ESTADÍSTICOS (Wilcoxon signed-rank)")
    print("-"*50)
    
    results['statistical_tests'] = {}
    
    # Test para tiempo
    if len(df_merged) >= 5:
        stat, p_value = stats.wilcoxon(
            df_merged['runtime_baseline'],
            df_merged['runtime_backbone'],
            alternative='greater'  # H1: baseline > backbone
        )
        results['statistical_tests']['runtime'] = {
            'statistic': stat,
            'p_value': p_value,
            'significant_005': p_value < 0.05,
            'significant_001': p_value < 0.01
        }
        print(f"  Tiempo: W={stat:.2f}, p={p_value:.4f} {'*' if p_value < 0.05 else ''}")
        
        # Test para nodos
        if 'n_nodes_baseline' in df_merged.columns:
            stat, p_value = stats.wilcoxon(
                df_merged['n_nodes_baseline'],
                df_merged['n_nodes_backbone'],
                alternative='greater'
            )
            results['statistical_tests']['nodes'] = {
                'statistic': stat,
                'p_value': p_value,
                'significant_005': p_value < 0.05
            }
            print(f"  Nodos: W={stat:.2f}, p={p_value:.4f} {'*' if p_value < 0.05 else ''}")
    else:
        print("  (Insuficientes datos para test estadístico, se requieren >= 5 pares)")
    
    print("\n* = significativo al 5%")
    
    # Guardar resultados detallados
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
        # Guardar merged dataframe
        merged_path = os.path.join(output_dir, "comparison_detailed.csv")
        df_merged.to_csv(merged_path, index=False)
        print(f"\nResultados detallados guardados en: {merged_path}")
        
        # Guardar resumen
        summary_path = os.path.join(output_dir, "comparison_summary.txt")
        with open(summary_path, 'w') as f:
            f.write("RESUMEN DE COMPARACIÓN BASELINE vs BACKBONE\n")
            f.write("="*50 + "\n\n")
            
            if 'time' in results:
                f.write("TIEMPO:\n")
                for k, v in results['time'].items():
                    f.write(f"  {k}: {v}\n")
                f.write("\n")
            
            if 'nodes' in results:
                f.write("NODOS:\n")
                for k, v in results['nodes'].items():
                    f.write(f"  {k}: {v}\n")
                f.write("\n")
            
            if 'statistical_tests' in results:
                f.write("TESTS ESTADÍSTICOS:\n")
                for test_name, test_results in results['statistical_tests'].items():
                    f.write(f"  {test_name}:\n")
                    for k, v in test_results.items():
                        f.write(f"    {k}: {v}\n")
        
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
        backbone_csv: CSV de resultados backbone
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
    
    # Calcular estadísticas
    stats_base = df_merged[['runtime_base', 'n_nodes_base']].agg(['mean', 'std', 'median'])
    stats_back = df_merged[['runtime_back', 'n_nodes_back']].agg(['mean', 'std', 'median'])
    
    # Generar tabla
    latex = r"""
\begin{table}[h]
\centering
\caption{Comparación de rendimiento: Baseline vs Backbone}
\label{tab:comparison}
\begin{tabular}{lrrrr}
\toprule
\textbf{Métrica} & \textbf{Baseline} & \textbf{Backbone} & \textbf{Mejora (\%)} & \textbf{p-valor} \\
\midrule
"""
    
    # Tiempo
    mean_base_t = df_merged['runtime_base'].mean()
    mean_back_t = df_merged['runtime_back'].mean()
    mejora_t = (mean_base_t - mean_back_t) / mean_base_t * 100
    
    if len(df_merged) >= 5:
        _, p_t = stats.wilcoxon(df_merged['runtime_base'], df_merged['runtime_back'], alternative='greater')
    else:
        p_t = float('nan')
    
    latex += f"Tiempo (seg) & {mean_base_t:.2f} & {mean_back_t:.2f} & {mejora_t:.1f} & {p_t:.4f} \\\\\n"
    
    # Nodos
    if 'n_nodes_base' in df_merged.columns:
        mean_base_n = df_merged['n_nodes_base'].mean()
        mean_back_n = df_merged['n_nodes_back'].mean()
        mejora_n = (mean_base_n - mean_back_n) / mean_base_n * 100
        
        if len(df_merged) >= 5:
            _, p_n = stats.wilcoxon(df_merged['n_nodes_base'], df_merged['n_nodes_back'], alternative='greater')
        else:
            p_n = float('nan')
        
        latex += f"Nodos & {mean_base_n:.0f} & {mean_back_n:.0f} & {mejora_n:.1f} & {p_n:.4f} \\\\\n"
    
    latex += r"""
\bottomrule
\end{tabular}
\end{table}
"""
    
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
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
  # Comparar baseline vs backbone
  python analyze_results.py \\
      --baseline ../results/metrics/baseline.csv \\
      --backbone ../results/metrics/backbone.csv \\
      --output_dir ../results/analysis

  # Generar tabla LaTeX
  python analyze_results.py \\
      --baseline ../results/metrics/baseline.csv \\
      --backbone ../results/metrics/backbone.csv \\
      --latex_table ../thesis/figures/comparison_table.tex
        """
    )
    
    parser.add_argument("--baseline", type=str, required=True,
                        help="CSV con resultados baseline")
    parser.add_argument("--backbone", type=str, required=True,
                        help="CSV con resultados backbone")
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
