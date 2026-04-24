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
from typing import Optional


def load_results(csv_path: str) -> pd.DataFrame:
    """Carga resultados desde CSV."""
    df = pd.read_csv(csv_path)
    return df


def compare_experiments(
    baseline_csv: str,
    backpas_csv: str,
    output_dir: Optional[str] = None
) -> dict:
    """
    Compara resultados entre experimentos baseline y BACKPAS.

    Args:
        baseline_csv: Ruta al CSV de resultados baseline
        backpas_csv: Ruta al CSV de resultados BACKPAS
        output_dir: Directorio para guardar reportes

    Returns:
        Diccionario con resultados de la comparación
    """
    df_base = load_results(baseline_csv)
    df_back = load_results(backpas_csv)

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
    backpas_csv: str,
    output_path: Optional[str] = None
) -> str:
    """
    Genera una tabla LaTeX con los resultados comparativos.

    Args:
        baseline_csv: CSV de resultados baseline
        backpas_csv: CSV de resultados BACKPAS
        output_path: Ruta para guardar la tabla

    Returns:
        String con la tabla LaTeX
    """
    df_base = load_results(baseline_csv)
    df_back = load_results(backpas_csv)

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


def generate_detailed_table(
    baseline_csv: str,
    backpas_csv: str,
    output_path: Optional[str] = None
) -> str:
    """
    Genera una tabla detallada por instancia con tiempos de Baseline,
    Phase 1, Phase 2 y total BACKPAS, para comparar directamente.

    Columnas: instancia | baseline | phase1 | phase2 | backpas_total | speedup
    """
    df_base = load_results(baseline_csv)
    df_back = load_results(backpas_csv)

    df = pd.merge(df_base, df_back, on='instance_name', suffixes=('_base', '_back'))

    if len(df) == 0:
        print("ERROR: No hay instancias comunes entre ambos experimentos")
        return ""

    # Columnas de tiempo
    has_phases = 'phase1_time_back' in df.columns and 'phase2_time_back' in df.columns
    time_base = df['runtime_base']
    time_p1   = df['phase1_time_back'] if has_phases else pd.Series([float('nan')] * len(df))
    time_p2   = df['phase2_time_back'] if has_phases else pd.Series([float('nan')] * len(df))
    time_back = df['runtime_back']
    speedup   = time_base / time_back.replace(0, float('nan'))

    # ── Tabla texto ──────────────────────────────────────────────────────────
    col_w = 30
    header = (f"{'Instancia':<{col_w}} {'Baseline':>10} {'Fase 1':>10} "
              f"{'Fase 2':>10} {'BACKPAS':>10} {'Speedup':>9}")
    sep = "-" * len(header)
    rows = [header, sep]

    for i, row in df.iterrows():
        p1 = f"{time_p1.iloc[i - df.index[0]]:.2f}" if has_phases else "--"
        p2 = f"{time_p2.iloc[i - df.index[0]]:.2f}" if has_phases else "--"
        sp = f"{speedup.iloc[i - df.index[0]]:.2f}x"
        rows.append(
            f"{row['instance_name']:<{col_w}} "
            f"{time_base.iloc[i - df.index[0]]:>10.2f} "
            f"{p1:>10} {p2:>10} "
            f"{time_back.iloc[i - df.index[0]]:>10.2f} "
            f"{sp:>9}"
        )

    rows.append(sep)
    rows.append(
        f"{'Media':<{col_w}} "
        f"{time_base.mean():>10.2f} "
        f"{time_p1.mean():>10.2f} "
        f"{time_p2.mean():>10.2f} "
        f"{time_back.mean():>10.2f} "
        f"{speedup.mean():>8.2f}x"
    )
    rows.append(
        f"{'Mediana':<{col_w}} "
        f"{time_base.median():>10.2f} "
        f"{time_p1.median():>10.2f} "
        f"{time_p2.median():>10.2f} "
        f"{time_back.median():>10.2f} "
        f"{speedup.median():>8.2f}x"
    )

    table_text = "\n".join(rows)
    print("\n" + "=" * 70)
    print("TABLA DETALLADA DE TIEMPOS (segundos)")
    print("=" * 70)
    print(table_text)

    # ── Tabla LaTeX ──────────────────────────────────────────────────────────
    n = len(df)
    latex_rows = ""
    for idx, row in df.iterrows():
        arr = idx - df.index[0]
        p1_str = f"{time_p1.iloc[arr]:.2f}" if has_phases else "--"
        p2_str = f"{time_p2.iloc[arr]:.2f}" if has_phases else "--"
        sp_str = f"{speedup.iloc[arr]:.2f}x"
        # Nombre corto: solo el número al final
        short_name = row['instance_name'].split('_')[-1]
        latex_rows += (
            f"  {short_name} & {time_base.iloc[arr]:.2f} & "
            f"{p1_str} & {p2_str} & "
            f"{time_back.iloc[arr]:.2f} & {sp_str} \\\\\n"
        )

    latex = (
        r"\begin{table}[htbp]" + "\n"
        r"\centering" + "\n"
        r"\caption{Tiempos de ejecución por instancia: Baseline vs BACKPAS ("
        + str(n) + r" instancias)}" + "\n"
        r"\label{tab:detailed_times}" + "\n"
        r"\begin{tabular}{lrrrrr}" + "\n"
        r"\toprule" + "\n"
        r"\textbf{Inst.} & \textbf{Baseline (s)} & \textbf{Fase 1 (s)} & "
        r"\textbf{Fase 2 (s)} & \textbf{BACKPAS (s)} & \textbf{Speedup} \\" + "\n"
        r"\midrule" + "\n"
        + latex_rows
        + r"\midrule" + "\n"
        + f"  Media & {time_base.mean():.2f} & {time_p1.mean():.2f} & "
          f"{time_p2.mean():.2f} & {time_back.mean():.2f} & "
          f"{speedup.mean():.2f}x \\\\\n"
        + f"  Mediana & {time_base.median():.2f} & {time_p1.median():.2f} & "
          f"{time_p2.median():.2f} & {time_back.median():.2f} & "
          f"{speedup.median():.2f}x \\\\\n"
        + r"\bottomrule" + "\n"
        + r"\end{tabular}" + "\n"
        + r"\end{table}" + "\n"
    )

    if output_path:
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        with open(output_path, 'w') as f:
            f.write(latex)
        print(f"\nTabla LaTeX guardada en: {output_path}")

    return latex


def generate_time_comparison_table(
    baseline_csv: str,
    backpas_csv: str,
    output_path: Optional[str] = None,
    label: str = "BACKPAS",
) -> str:
    """
    Genera tabla de tiempos promedio ± std: Gurobi (baseline) vs BACKPAS.

    Para BACKPAS usa phase1_time + phase2_time como tiempo total de Gurobi.
    Acepta tanto CSVs individuales como agregados (salida de aggregate_seeds).

    Args:
        baseline_csv: CSV baseline (individual o aggregated).
        backpas_csv:  CSV BACKPAS  (individual o aggregated).
        output_path:  Ruta opcional para guardar tabla LaTeX.
        label:        Nombre del método BACKPAS en los encabezados.

    Returns:
        String con la tabla LaTeX.
    """
    df_base = pd.read_csv(baseline_csv)
    df_back = pd.read_csv(backpas_csv)

    df = pd.merge(df_base, df_back, on='instance_name', suffixes=('_base', '_back'))
    n = len(df)

    if n == 0:
        print("ERROR: No hay instancias comunes entre ambos experimentos")
        return ""

    # ── Detectar si son CSVs agregados (tienen columnas _mean/_std) ─────────
    agg_base = 'runtime_mean_base' in df.columns
    agg_back = 'phase1_time_mean_back' in df.columns

    # Baseline: usar gurobi_runtime
    if agg_base:
        t_base_mean = df['gurobi_runtime_mean_base']
        t_base_std  = df['gurobi_runtime_std_base'].fillna(0.0)
    else:
        t_base_mean = df['gurobi_runtime_base']
        t_base_std  = pd.Series([float('nan')] * n, index=df.index)

    # BACKPAS: usar phase1_time + phase2_time
    if agg_back:
        p1_mean     = df['phase1_time_mean_back'].fillna(0.0)
        p2_mean     = df['phase2_time_mean_back'].fillna(0.0)
        p1_std      = df['phase1_time_std_back'].fillna(0.0)
        p2_std      = df['phase2_time_std_back'].fillna(0.0)
        t_back_mean = p1_mean + p2_mean
        t_back_std  = np.sqrt(p1_std**2 + p2_std**2)
    elif 'phase1_time_back' in df.columns and 'phase2_time_back' in df.columns:
        t_back_mean = df['phase1_time_back'] + df['phase2_time_back']
        t_back_std  = pd.Series([float('nan')] * n, index=df.index)
    else:
        # fallback: runtime total
        t_back_mean = df['runtime_back']
        t_back_std  = pd.Series([float('nan')] * n, index=df.index)

    speedup = t_base_mean / t_back_mean.replace(0, float('nan'))

    def _fmt(mean: float, std: float) -> str:
        if np.isnan(std) or std < 1e-9:
            return f"{mean:.2f}"
        return f"{mean:.2f} ± {std:.2f}"

    # ── Tabla texto ──────────────────────────────────────────────────────────
    col_w = 35
    header = (f"{'Instancia':<{col_w}} {'Gurobi (s)':>22} {label+' (s)':>22} {'Speedup':>9}")
    sep = "-" * len(header)
    rows = [header, sep]

    for i, row in df.iterrows():
        idx = df.index.get_loc(i)
        rows.append(
            f"{row['instance_name']:<{col_w}} "
            f"{_fmt(t_base_mean.iloc[idx], t_base_std.iloc[idx]):>22} "
            f"{_fmt(t_back_mean.iloc[idx], t_back_std.iloc[idx]):>22} "
            f"{speedup.iloc[idx]:>8.2f}x"
        )

    rows.append(sep)
    rows.append(
        f"{'Media':<{col_w}} "
        f"{_fmt(t_base_mean.mean(), t_base_std.mean()):>22} "
        f"{_fmt(t_back_mean.mean(), t_back_std.mean()):>22} "
        f"{speedup.mean():>8.2f}x"
    )
    rows.append(
        f"{'Mediana':<{col_w}} "
        f"{_fmt(t_base_mean.median(), float('nan')):>22} "
        f"{_fmt(t_back_mean.median(), float('nan')):>22} "
        f"{speedup.median():>8.2f}x"
    )

    table_text = "\n".join(rows)
    print("\n" + "=" * 70)
    print(f"TABLA DE TIEMPOS: GUROBI vs {label.upper()} (segundos)")
    print(f"Baseline = gurobi_runtime | BACKPAS = phase1_time + phase2_time")
    print("=" * 70)
    print(table_text)

    # ── Tabla LaTeX ──────────────────────────────────────────────────────────
    latex_rows = ""
    for i, row in df.iterrows():
        idx = df.index.get_loc(i)
        short      = row['instance_name'].split('_')[-1]
        base_str   = _fmt(t_base_mean.iloc[idx], t_base_std.iloc[idx])
        back_str   = _fmt(t_back_mean.iloc[idx], t_back_std.iloc[idx])
        sp_str     = f"{speedup.iloc[idx]:.2f}x"
        latex_rows += f"  {short} & {base_str} & {back_str} & {sp_str} \\\\\n"

    latex = (
        r"\begin{table}[htbp]" + "\n"
        r"\centering" + "\n"
        r"\caption{Tiempos de ejecución: Gurobi vs " + label
        + r" (" + str(n) + r" instancias, media $\pm$ std)}" + "\n"
        r"\label{tab:time_comparison}" + "\n"
        r"\begin{tabular}{lrrr}" + "\n"
        r"\toprule" + "\n"
        r"\textbf{Inst.} & \textbf{Gurobi (s)} & \textbf{"
        + label + r" (s)} & \textbf{Speedup} \\" + "\n"
        r"\midrule" + "\n"
        + latex_rows
        + r"\midrule" + "\n"
        + f"  Media & {_fmt(t_base_mean.mean(), t_base_std.mean())} & "
          f"{_fmt(t_back_mean.mean(), t_back_std.mean())} & "
          f"{speedup.mean():.2f}x \\\\\n"
        + f"  Mediana & {_fmt(t_base_mean.median(), float('nan'))} & "
          f"{_fmt(t_back_mean.median(), float('nan'))} & "
          f"{speedup.median():.2f}x \\\\\n"
        + r"\bottomrule" + "\n"
        + r"\end{tabular}" + "\n"
        + r"\end{table}" + "\n"
    )

    if output_path:
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        with open(output_path, 'w') as f:
            f.write(latex)
        print(f"\nTabla LaTeX guardada en: {output_path}")

    return latex


def aggregate_seeds(input_dir: str) -> None:
    """
    Detecta todos los baseline_seed*.csv y backpas_seed*.csv en input_dir,
    calcula media ± std ± min ± max por instancia entre seeds, y guarda
    baseline_aggregated.csv y backpas_aggregated.csv en el mismo directorio.
    """
    import glob

    numeric_cols = [
        'runtime', 'gurobi_runtime', 'obj_val', 'n_nodes', 'n_solutions',
        'primal_integral', 'phase1_time', 'phase2_time', 'phase1_obj',
        'phase2_obj', 'phase1_nodes', 'phase2_nodes', 'total_nodes',
    ]

    def _aggregate_group(csv_paths: list, output_path: str, label: str) -> None:
        dfs = []
        for i, path in enumerate(csv_paths):
            df = pd.read_csv(path)
            df['seed'] = i
            dfs.append(df)

        all_data = pd.concat(dfs, ignore_index=True)
        cols = [c for c in numeric_cols if c in all_data.columns]

        grouped = all_data.groupby('instance_name')

        agg_mean  = grouped[cols].mean().add_suffix('_mean')
        agg_std   = grouped[cols].std().add_suffix('_std')
        agg_min   = grouped[cols].min().add_suffix('_min')
        agg_max   = grouped[cols].max().add_suffix('_max')

        # Contar cuántas seeds llegaron a OPTIMAL
        if 'status_name' in all_data.columns:
            n_optimal = grouped['status_name'].apply(
                lambda x: (x == 'OPTIMAL').sum()
            ).rename('n_optimal')
        else:
            n_optimal = pd.Series(dtype=int)

        result = pd.concat([agg_mean, agg_std, agg_min, agg_max], axis=1)
        if len(n_optimal):
            result = result.join(n_optimal)
        result['n_seeds'] = grouped['seed'].count()
        result = result.reset_index()
        result.to_csv(output_path, index=False)

        # Resumen en pantalla
        n_seeds = len(dfs)
        n_inst  = len(result)
        print(f"\n  {label.upper()} — {n_seeds} seeds, {n_inst} instancias")
        if 'runtime_mean' in result.columns:
            print(f"  Runtime:         {result['runtime_mean'].mean():.2f}s "
                  f"(± {result['runtime_std'].mean():.2f}s)")
        if 'primal_integral_mean' in result.columns:
            pi = result['primal_integral_mean'].dropna()
            if len(pi):
                print(f"  Primal integral: {pi.mean():.4f} "
                      f"(± {result['primal_integral_std'].mean():.4f})")
        if 'phase1_time_mean' in result.columns:
            print(f"  Fase 1 media:    {result['phase1_time_mean'].mean():.2f}s")
        if 'phase2_time_mean' in result.columns:
            print(f"  Fase 2 media:    {result['phase2_time_mean'].mean():.2f}s")
        if 'n_optimal' in result.columns:
            fully_optimal = (result['n_optimal'] == n_seeds).sum()
            print(f"  OPTIMAL en todas las seeds: {fully_optimal}/{n_inst}")
        print(f"  Guardado en: {output_path}")

    print(f"\n{'='*60}")
    print("AGREGANDO RESULTADOS POR SEED")
    print(f"{'='*60}")
    print(f"  Directorio: {input_dir}")

    baseline_csvs = sorted(glob.glob(os.path.join(input_dir, "baseline_seed*.csv")))
    backpas_csvs  = sorted(glob.glob(os.path.join(input_dir, "backpas_seed*.csv")))

    print(f"  baseline_seed*.csv encontrados: {len(baseline_csvs)}")
    print(f"  backpas_seed*.csv  encontrados: {len(backpas_csvs)}")

    if not baseline_csvs and not backpas_csvs:
        print("  ERROR: no se encontraron CSVs en el directorio")
        return

    if baseline_csvs:
        _aggregate_group(
            baseline_csvs,
            os.path.join(input_dir, "baseline_aggregated.csv"),
            "baseline"
        )
    if backpas_csvs:
        _aggregate_group(
            backpas_csvs,
            os.path.join(input_dir, "backpas_aggregated.csv"),
            "backpas"
        )


def _find_csv(directory: str, prefix: str) -> str:
    """
    Busca el CSV más apropiado en `directory` cuyo nombre empiece con `prefix`.

    Prioridad:
      1. <prefix>_aggregated.csv  (resultado de aggregate_seeds)
      2. El único archivo <prefix>*.csv si hay exactamente uno
      3. El primero en orden alfabético si hay varios

    Raises FileNotFoundError si no encuentra ninguno.
    """
    import glob
    pattern = os.path.join(directory, f"{prefix}*.csv")
    candidates = sorted(glob.glob(pattern))
    if not candidates:
        raise FileNotFoundError(
            f"No se encontró ningún archivo '{prefix}*.csv' en {directory}"
        )
    aggregated = [c for c in candidates if os.path.basename(c) == f"{prefix}_aggregated.csv"]
    if aggregated:
        return aggregated[0]
    return candidates[0]


def main():
    parser = argparse.ArgumentParser(
        description="Análisis estadístico de resultados experimentales",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:
  # Comparar baseline vs BACKPAS (auto-detecta baseline*.csv y backpas*.csv)
  python analyze_results.py --dir ../results/metrics/multi_seed

  # Generar tabla de tiempos LaTeX
  python analyze_results.py \\
      --dir ../results/metrics/multi_seed \\
      --time_table ../thesis/figures/time_table.tex

  # Agregar seeds y luego comparar
  python analyze_results.py --dir ../results/metrics/multi_seed --aggregate
  python analyze_results.py --dir ../results/metrics/multi_seed --time_table tabla.tex
        """
    )

    parser.add_argument("--dir", type=str, default=None,
                        help="Directorio con los CSVs. Se auto-detectan baseline*.csv "
                             "y backpas*.csv (prefiere *_aggregated.csv si existe)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Directorio para guardar análisis (default: mismo que --dir)")
    parser.add_argument("--latex_table", type=str, default=None,
                        help="Ruta para guardar tabla LaTeX de resumen estadístico")
    parser.add_argument("--detailed_table", type=str, default=None,
                        help="Ruta para guardar tabla LaTeX detallada por instancia")
    parser.add_argument("--time_table", type=str, default=None,
                        help="Ruta para guardar tabla LaTeX de tiempos (mean±std). "
                             "Para BACKPAS usa phase1_time+phase2_time.")
    parser.add_argument("--time_table_label", type=str, default="BACKPAS",
                        help="Etiqueta del método BACKPAS en la tabla de tiempos (default: BACKPAS)")
    parser.add_argument("--aggregate", action="store_true",
                        help="Agregar CSVs por seed antes de comparar "
                             "(equivalente a --aggregate_dir con el mismo directorio)")

    args = parser.parse_args()

    if not args.dir:
        parser.error("--dir es requerido")

    # Modo agregación
    if args.aggregate:
        aggregate_seeds(args.dir)

    # Auto-detectar CSVs
    try:
        baseline_csv = _find_csv(args.dir, "baseline")
        backpas_csv  = _find_csv(args.dir, "backpas")
    except FileNotFoundError as e:
        parser.error(str(e))

    print(f"baseline : {os.path.basename(baseline_csv)}")
    print(f"backpas  : {os.path.basename(backpas_csv)}")

    output_dir = args.output_dir or args.dir

    results = compare_experiments(
        baseline_csv=baseline_csv,
        backpas_csv=backpas_csv,
        output_dir=output_dir,
    )

    if args.latex_table:
        generate_latex_table(
            baseline_csv=baseline_csv,
            backpas_csv=backpas_csv,
            output_path=args.latex_table,
        )

    if args.detailed_table:
        generate_detailed_table(
            baseline_csv=baseline_csv,
            backpas_csv=backpas_csv,
            output_path=args.detailed_table,
        )

    if args.time_table:
        generate_time_comparison_table(
            baseline_csv=baseline_csv,
            backpas_csv=backpas_csv,
            output_path=args.time_table,
            label=args.time_table_label,
        )


if __name__ == "__main__":
    main()
