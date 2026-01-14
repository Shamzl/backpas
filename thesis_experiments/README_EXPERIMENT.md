# Guía de Experimentos - Tesis

Este documento explica cómo ejecutar los experimentos BASELINE y BACKPAS para la tesis.

## Descripción del Experimento

El objetivo es comparar dos métodos:

1. **BASELINE**: Gurobi ejecutando solo, sin ninguna ayuda
2. **BACKPAS**: Gurobi + Trust Region Temporal
   - Usa predicciones de GNN para crear una trust region
   - La trust region se mantiene por X segundos (configurable)
   - Después, se eliminan las restricciones y Gurobi continúa en el espacio completo
   - Esto permite encontrar soluciones buenas rápido + demostrar optimalidad global

## Prerrequisitos

```bash
# En la VM, asegúrate de tener el entorno activado
cd ~/backpas
source .venv/bin/activate

# Verificar que todo está instalado
python -c "import torch, gurobipy, networkx; print('OK')"
```

## Comandos de Uso

### Modo BASELINE (Gurobi solo)

```bash
# Una sola instancia
python thesis_experiments/src/run_gurobi_experiment.py \
    --instance thesis_experiments/instances/test/mis_50n_000.lp \
    --output_csv thesis_experiments/results/baseline_test.csv \
    --time_limit 3600

# Múltiples instancias
python thesis_experiments/src/run_gurobi_experiment.py \
    --instance_dir thesis_experiments/instances/test \
    --output_csv thesis_experiments/results/baseline.csv \
    --time_limit 3600
```

### Modo BACKPAS (Trust Region Temporal)

```bash
# Una sola instancia con trust region de 5 minutos
python thesis_experiments/src/run_gurobi_experiment.py \
    --instance dataset/MIS/instance/train_easy_instance_338.lp \
    --backpas \
    --model_path wkdir/MIS/ml_training/graph_with_literals_8_GTR/best_model.pth \
    --trust_region_time 300 \
    --threshold 0.7 \
    --alpha 0.0 \
    --output_csv thesis_experiments/results/backpas_easy_instance_338s.csv \
    --time_limit 3600

# Múltiples instancias
python thesis_experiments/src/run_gurobi_experiment.py \
    --instance_dir thesis_experiments/instances/test \
    --backpas \
    --model_path wkdir/MIS/ml_training/graph_with_literals_8_GTR/best_model.pth \
    --trust_region_time 300 \
    --output_csv thesis_experiments/results/backpas.csv \
    --time_limit 3600
```

## Parámetros Importantes

| Parámetro | Descripción | Valor por defecto |
|-----------|-------------|-------------------|
| `--time_limit` | Tiempo total (segundos) | 3600 (1 hora) |
| `--trust_region_time` | Tiempo con trust region (BACKPAS) | 300 (5 minutos) |
| `--threshold` | Umbral θ de confianza | 0.7 |
| `--alpha` | Tolerancia adaptativa α | 0.0 |
| `--threads` | Número de hilos Gurobi | 1 |

## Experimentos Sugeridos

### Experimento 1: Comparar BASELINE vs BACKPAS

```bash
# BASELINE
python thesis_experiments/src/run_gurobi_experiment.py \
    --instance_dir thesis_experiments/instances/test \
    --output_csv thesis_experiments/results/baseline.csv \
    --time_limit 3600

# BACKPAS (trust region 5 minutos)
python thesis_experiments/src/run_gurobi_experiment.py \
    --instance_dir thesis_experiments/instances/test \
    --backpas \
    --model_path wkdir/MIS/ml_training/graph_with_literals_8_GTR/best_model.pth \
    --trust_region_time 300 \
    --output_csv thesis_experiments/results/backpas_300s.csv \
    --time_limit 3600
```

### Experimento 2: Evaluar diferentes tiempos de trust region

```bash
# Trust region 1 minuto
python thesis_experiments/src/run_gurobi_experiment.py \
    --instance_dir thesis_experiments/instances/test \
    --backpas \
    --model_path wkdir/MIS/ml_training/graph_with_literals_8_GTR/best_model.pth \
    --trust_region_time 60 \
    --output_csv thesis_experiments/results/backpas_60s.csv

# Trust region 5 minutos
python thesis_experiments/src/run_gurobi_experiment.py \
    --instance_dir thesis_experiments/instances/test \
    --backpas \
    --model_path wkdir/MIS/ml_training/graph_with_literals_8_GTR/best_model.pth \
    --trust_region_time 300 \
    --output_csv thesis_experiments/results/backpas_300s.csv

# Trust region 10 minutos
python thesis_experiments/src/run_gurobi_experiment.py \
    --instance_dir thesis_experiments/instances/test \
    --backpas \
    --model_path wkdir/MIS/ml_training/graph_with_literals_8_GTR/best_model.pth \
    --trust_region_time 600 \
    --output_csv thesis_experiments/results/backpas_600s.csv
```

## Interpretación de Resultados

El archivo CSV contendrá las siguientes columnas clave:

| Columna | Descripción |
|---------|-------------|
| `method` | 'baseline' o 'backpas' |
| `runtime` | Tiempo total de ejecución |
| `obj_val` | Valor objetivo de la solución |
| `primal_integral` | Métrica de anytime performance (menor es mejor) |
| `trust_region_removed` | Si se quitó la trust region (solo BACKPAS) |
| `k_0`, `k_1`, `Delta` | Parámetros de trust region (solo BACKPAS) |

### Métricas de Éxito

BACKPAS es exitoso si:
1. **Primal Integral menor** que baseline → Encuentra buenas soluciones más rápido
2. **Mismo valor objetivo** final → No pierde optimalidad
3. **Tiempo similar o menor** → No agrega overhead significativo

## Troubleshooting

### Error: "Módulos BACKPAS no disponibles"
```bash
# Verificar que torch está instalado
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install torch_geometric torch_scatter torch_sparse
```

### Error: "Version number is 13.0, license is for version 12.0"
```bash
pip uninstall gurobipy
pip install gurobipy==12.0.0
```

### Error: "No module named 'pyscipopt'"
```bash
pip install pyscipopt
```

## Notas

- El script **reutiliza** toda la lógica existente de `run_gurobi_experiment.py`
- El modo BACKPAS es **opcional** (flag `--backpas`)
- Las restricciones de trust region se agregan **dinámicamente** en memoria (no se modifican archivos)
- El callback de Gurobi **elimina automáticamente** las restricciones después del tiempo especificado
