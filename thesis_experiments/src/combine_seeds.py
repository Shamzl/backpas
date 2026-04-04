#!/usr/bin/env python3
"""
Combina los CSVs de múltiples seeds en un único archivo por método.

Lee baseline_seed*.csv y backpas_seed*.csv (excluye *_aggregated.csv),
agrega una columna 'seed' a cada uno y los concatena.

Resultado:
  <output_dir>/baseline_combined.csv  — todas las seeds del baseline juntas
  <output_dir>/backpas_combined.csv   — todas las seeds de BACKPAS juntas

Uso:
  python combine_seeds.py \\
      --input_dir thesis_experiments/results/metrics/multi_seed_1333_v2 \\
      --output_dir thesis_experiments/results/metrics/multi_seed_1333_v2
"""

import argparse
import re
from pathlib import Path

import pandas as pd


def combine_seeds(input_dir: str, output_dir: str) -> None:
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for prefix in ("baseline", "backpas"):
        # Buscar archivos seed_N, excluir aggregated
        pattern = f"{prefix}_seed*.csv"
        files = sorted(
            f for f in input_path.glob(pattern)
            if "aggregated" not in f.name
        )

        if not files:
            print(f"[{prefix}] No se encontraron archivos con patrón {pattern}")
            continue

        frames = []
        for csv_file in files:
            # Extraer número de seed desde el nombre del archivo
            match = re.search(r"seed(\d+)", csv_file.stem)
            seed_num = int(match.group(1)) if match else -1

            df = pd.read_csv(csv_file)
            df.insert(0, "seed", seed_num)  # columna seed al inicio
            frames.append(df)
            print(f"  Cargado: {csv_file.name} ({len(df)} instancias, seed={seed_num})")

        combined = pd.concat(frames, ignore_index=True)
        combined.sort_values(["instance_name", "seed"], inplace=True)
        combined.reset_index(drop=True, inplace=True)

        out_file = output_path / f"{prefix}_combined.csv"
        combined.to_csv(out_file, index=False)
        print(f"[{prefix}] Guardado: {out_file}  ({len(combined)} filas)")


def main():
    parser = argparse.ArgumentParser(
        description="Combina CSVs de múltiples seeds en un único archivo por método",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplo:
  python combine_seeds.py \\
      --input_dir thesis_experiments/results/metrics/multi_seed_1333_v2
        """
    )
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Directorio con baseline_seed*.csv y backpas_seed*.csv")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Directorio de salida (default: mismo que input_dir)")

    args = parser.parse_args()
    output_dir = args.output_dir if args.output_dir else args.input_dir

    combine_seeds(args.input_dir, output_dir)


if __name__ == "__main__":
    main()
