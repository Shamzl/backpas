#!/usr/bin/env python3
"""
Tabla de primal integral: Gurobi (baseline) vs BACKPAS, por variante.

Misma logica de agregacion que make_table.py (media entre seeds por instancia),
pero sobre la columna 'primal_integral'. Ademas corre un Wilcoxon pareado por
instancia (diferencia = baseline - backpas; un valor positivo => backpas mejor,
ya que MENOR primal integral es mejor) y reporta tamano de efecto r = Z/sqrt(N).

NOTA: el primal integral se mide sobre el reloj interno del solver y por tanto
NO incluye el overhead de la GNN (ver run_gurobi_experiment.py). Es la lectura
"anytime / trabajo algoritmico", comparable a backpas_gurobi_mean.

Uso:
  python make_primal_table.py --base_dir ../../results/metrics
"""

import argparse
import glob
import os

import numpy as np
import pandas as pd
from scipy.stats import norm, wilcoxon

# Mapeo: etiqueta -> (carpeta backpas, carpeta baseline).
# Algunas variantes no re-corrieron baseline (Gurobi puro no cambia con params
# de Fase 2), asi que reutilizan el baseline de su misma familia op.
_OPA_BASE = "multi_seed_test_1333_V8_opA_Heurisitics0"
_OPB_BASE = "multi_seed_test_1333_V9_opB_H0"
VARIANTS = [
    ("opA_H0",       "multi_seed_test_1333_V8_opA_Heurisitics0",       _OPA_BASE),
    ("opA_HDefault", "multi_seed_test_1333_V8_opA_HeurisiticsDefault", _OPA_BASE),
    ("opB_H0",       "multi_seed_test_1333_V9_opB_H0",                 _OPB_BASE),
    ("opB_HDefault", "multi_seed_test_1333_V9_opB_HDefault",           _OPB_BASE),
    ("opA_MIPF2",    "multi_seed_test_1333_V9_opA_H0_MIPF2",           _OPA_BASE),
]


def _load_pi(directory: str, prefix: str) -> pd.DataFrame:
    """Carga <prefix>_seed*.csv y devuelve primal_integral por instancia/seed."""
    paths = sorted(glob.glob(os.path.join(directory, f"{prefix}_seed*.csv")))
    if not paths:
        raise FileNotFoundError(f"No hay '{prefix}_seed*.csv' en {directory}")
    frames = []
    for i, path in enumerate(paths):
        df = pd.read_csv(path)[["instance_name", "primal_integral"]].copy()
        df["seed"] = i
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    out["primal_integral"] = pd.to_numeric(out["primal_integral"], errors="coerce")
    return out


def _agg_per_instance(directory: str, prefix: str) -> pd.Series:
    df = _load_pi(directory, prefix)
    return df.groupby("instance_name")["primal_integral"].mean()


def process_variant(back_dir: str, base_dir: str, label: str) -> dict:
    base = _agg_per_instance(base_dir, "baseline").rename("baseline")
    back = _agg_per_instance(back_dir, "backpas").rename("backpas")
    m = pd.concat([base, back], axis=1).dropna()

    n = len(m)
    diff = m["baseline"] - m["backpas"]          # >0 => backpas mejor (menor PI)

    # Wilcoxon de dos colas; H1 una cola: backpas mas rapido (diff>0)
    try:
        stat, p_two = wilcoxon(m["baseline"], m["backpas"],
                               zero_method="wilcox", alternative="two-sided")
    except ValueError:
        p_two = float("nan")
    # z a partir del p-value de dos colas, con signo segun la mediana de diff
    if np.isnan(p_two) or p_two <= 0:
        z = float("nan")
    else:
        z = norm.isf(p_two / 2) * (1 if diff.median() >= 0 else -1)
    r = z / np.sqrt(n) if n > 0 else float("nan")

    return {
        "variant":       label,
        "n":             n,
        "pi_baseline":   m["baseline"].mean(),
        "pi_backpas":    m["backpas"].mean(),
        "reduction_x":   m["baseline"].mean() / m["backpas"].mean()
                         if m["backpas"].mean() else float("nan"),
        "median_diff":   diff.median(),
        "p_two_sided":   p_two,
        "r":             r,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_dir", required=True,
                    help="Directorio que contiene las carpetas multi_seed_test_1333_*")
    ap.add_argument("--output", default=None,
                    help="Ruta del CSV de salida (default: <base_dir>/primal_table.csv)")
    args = ap.parse_args()

    rows = []
    for label, back_folder, base_folder in VARIANTS:
        back_dir = os.path.join(args.base_dir, back_folder)
        base_dir = os.path.join(args.base_dir, base_folder)
        if not os.path.isdir(back_dir) or not os.path.isdir(base_dir):
            print(f"  [skip] falta carpeta para {label}")
            continue
        rows.append(process_variant(back_dir, base_dir, label))

    res = pd.DataFrame(rows)
    pd.set_option("display.float_format", lambda x: f"{x:.4f}")
    print(res.to_string(index=False))

    output = args.output or os.path.join(args.base_dir, "primal_table2.csv")
    res.to_csv(output, index=False, float_format="%.6g")
    print(f"\nTabla guardada en: {output}")


if __name__ == "__main__":
    main()
