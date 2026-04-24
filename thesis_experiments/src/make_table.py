#!/usr/bin/env python3
"""
Genera tabla de tiempos: Gurobi (baseline) vs BACKPAS.

Lee baseline_seed*.csv y/o backpas_seed*.csv de un directorio y calcula
media ± desviación estándar por instancia entre seeds.

  Baseline : gurobi_runtime
  BACKPAS  : phase1_time + phase2_time

Uso:
  # Ambos métodos (genera speedup)
  python make_table.py --dir results/multi_seed_1333_v3

  # Solo baseline
  python make_table.py --dir results/multi_seed_1333_v3 --baseline

  # Solo BACKPAS
  python make_table.py --dir results/multi_seed_1333_v3 --backpas
"""

import argparse
import glob
import os

import pandas as pd


def _load_seeds(directory: str, prefix: str) -> pd.DataFrame:
    """Carga todos los <prefix>_seed*.csv y los concatena."""
    paths = sorted(glob.glob(os.path.join(directory, f"{prefix}_seed*.csv")))
    if not paths:
        raise FileNotFoundError(
            f"No se encontraron archivos '{prefix}_seed*.csv' en {directory}"
        )
    frames = []
    for i, path in enumerate(paths):
        df = pd.read_csv(path)
        df["seed"] = i
        frames.append(df)
    print(f"  {prefix}: {len(paths)} seed(s)")
    return pd.concat(frames, ignore_index=True)


def _agg_baseline(directory: str) -> pd.DataFrame:
    df = _load_seeds(directory, "baseline")
    grp = df.groupby("instance_name")
    return pd.DataFrame({
        "baseline_mean":      grp["gurobi_runtime"].mean(),
        "baseline_std":       grp["gurobi_runtime"].std(ddof=1).fillna(0.0),
        "baseline_n_optimal": grp["status_name"].apply(lambda x: (x == "OPTIMAL").sum()),
    }).reset_index()


def _agg_backpas(directory: str) -> pd.DataFrame:
    df = _load_seeds(directory, "backpas")
    df["backpas_time"] = df["phase1_time"] + df["phase2_time"]
    grp = df.groupby("instance_name")
    return pd.DataFrame({
        "backpas_mean":      grp["backpas_time"].mean(),
        "backpas_std":       grp["backpas_time"].std(ddof=1).fillna(0.0),
        "backpas_n_optimal": grp["status_name"].apply(lambda x: (x == "OPTIMAL").sum()),
    }).reset_index()


def build_table(directory: str, only_baseline: bool = False, only_backpas: bool = False) -> None:
    print(f"\nDirectorio: {directory}")

    if only_baseline:
        result = _agg_baseline(directory)
        output = os.path.join(directory, "time_table_baseline.csv")
    elif only_backpas:
        result = _agg_backpas(directory)
        output = os.path.join(directory, "time_table_backpas.csv")
    else:
        base = _agg_baseline(directory)
        back = _agg_backpas(directory)
        result = pd.merge(base, back, on="instance_name")
        result["speedup"] = result["baseline_mean"] / result["backpas_mean"].replace(0, float("nan"))
        output = os.path.join(directory, "time_table.csv")

    result.to_csv(output, index=False, float_format="%.4f")
    print(f"Tabla guardada en: {output}")
    print(f"Instancias: {len(result)}")
    if "speedup" in result.columns:
        print(f"Speedup medio: {result['speedup'].mean():.2f}x")


def main():
    parser = argparse.ArgumentParser(
        description="Genera tabla de tiempos Gurobi vs BACKPAS desde CSVs por seed.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  python make_table.py --dir results/multi_seed_1333_v3
  python make_table.py --dir results/multi_seed_1333_v3 --baseline
  python make_table.py --dir results/multi_seed_1333_v3 --backpas
        """,
    )
    parser.add_argument("--dir", required=True,
                        help="Directorio con baseline_seed*.csv y/o backpas_seed*.csv")

    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--baseline", action="store_true",
                      help="Solo procesar archivos baseline_seed*.csv")
    mode.add_argument("--backpas", action="store_true",
                      help="Solo procesar archivos backpas_seed*.csv")

    args = parser.parse_args()
    build_table(args.dir, only_baseline=args.baseline, only_backpas=args.backpas)


if __name__ == "__main__":
    main()
