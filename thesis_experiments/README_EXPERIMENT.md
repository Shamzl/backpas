# Guía de Experimentos - Tesis

Este documento explica cómo ejecutar los experimentos BASELINE y BACKPAS para la tesis.

## Descripción del Experimento

El objetivo es comparar dos métodos:

1. **BASELINE**: Gurobi ejecutando solo, sin ninguna ayuda
2. **BACKPAS**: Gurobi + Trust Region en dos fases
   - **Fase 1**: Predicciones de GNN crean una trust region que restringe el espacio de búsqueda
   - Gurobi encuentra buenas soluciones rápido dentro de la trust region
   - La solución se guarda como warmstart
   - **Fase 2**: Se crea un modelo nuevo SIN trust region + warmstart de Fase 1
   - Fase 2 SIEMPRE se ejecuta para verificar optimalidad global

## Prerrequisitos

```bash
# En la VM, asegúrate de tener el entorno activado
cd ~/backpas
source .venv/bin/activate

# Verificar que todo está instalado
python -c "import torch, gurobipy, networkx; print('OK')"
```

## Parámetros del Paper (MIS)

Valores óptimos extraídos del paper de referencia:

| Parámetro | Valor | Descripción |
|-----------|-------|-------------|
| `--threshold` (θ) | `0.6161128065675395` | Umbral de confianza para selección de variables |
| `--alpha` (α) | `-0.9946887313825644` | Tolerancia adaptativa |
| `--trust_region_time` | `300` | Tiempo de Fase 1 (segundos) |
| `--time_limit` | `3600` | Tiempo total (segundos) |
| `--threads` | `1` | Hilos de Gurobi |

## Comandos de Uso

### Modo BASELINE (Gurobi solo)

```bash
# Una sola instancia
python thesis_experiments/src/run_gurobi_experiment.py \
    --instance dataset/MIS/instance/instancia.mps \
    --output_csv thesis_experiments/results/metrics/baseline.csv \
    --time_limit 3600

# Múltiples instancias (background)
nohup python thesis_experiments/src/run_gurobi_experiment.py \
    --instance_dir dataset/MIS/instance/param_search \
    --output_csv thesis_experiments/results/metrics/baseline.csv \
    --log_dir thesis_experiments/results/logs/baseline \
    --time_limit 3600 > baseline.log 2>&1 &
```

### Modo BACKPAS (parámetros del paper)

```bash
# Una sola instancia
python thesis_experiments/src/run_gurobi_experiment.py \
    --instance dataset/MIS/instance/instancia.mps \
    --backpas \
    --model_path wkdir/MIS/ml_training/graph_with_literals_8_GTR/best_model.pth \
    --trust_region_time 300 \
    --threshold 0.6161128065675395 \
    --alpha -0.9946887313825644 \
    --output_csv thesis_experiments/results/metrics/backpas.csv \
    --time_limit 3600

# Múltiples instancias (background)
nohup python thesis_experiments/src/run_gurobi_experiment.py \
    --instance_dir dataset/MIS/instance/param_search \
    --backpas \
    --model_path wkdir/MIS/ml_training/graph_with_literals_8_GTR/best_model.pth \
    --trust_region_time 300 \
    --threshold 0.6161128065675395 \
    --alpha -0.9946887313825644 \
    --output_csv thesis_experiments/results/metrics/backpas.csv \
    --log_dir thesis_experiments/results/logs/backpas \
    --time_limit 3600 > backpas.log 2>&1 &
```

### Analizar resultados

```bash
python thesis_experiments/src/analyze_results.py \
    --baseline thesis_experiments/results/metrics/baseline.csv \
    --backbone thesis_experiments/results/metrics/backpas.csv \
    --output_dir thesis_experiments/results/analysis \
    --latex_table thesis_experiments/thesis/figures/comparison_table.tex
```

### Búsqueda de parámetros (opcional)

```bash
python thesis_experiments/src/parameter_search.py \
    --instance_dir dataset/MIS/instance/param_search \
    --model_path wkdir/MIS/ml_training/graph_with_literals_8_GTR/best_model.pth \
    --output_dir thesis_experiments/results/param_search \
    --thresholds 0.5 0.6 0.7 0.8 0.9 \
    --alphas -0.99 -0.9 -0.5 0.0 \
    --trust_region_times 300 \
    --time_limit 3600
```

## Parámetros Disponibles

| Parámetro | Descripción | Valor por defecto |
|-----------|-------------|-------------------|
| `--instance` | Ruta a una instancia .lp o .mps | - |
| `--instance_dir` | Directorio con múltiples instancias | - |
| `--time_limit` | Tiempo total (segundos) | 3600 |
| `--threads` | Número de hilos Gurobi | 1 |
| `--backpas` | Activar modo BACKPAS | False |
| `--model_path` | Ruta al modelo .pth (requerido con --backpas) | - |
| `--trust_region_time` | Tiempo de Fase 1 con trust region | 300 |
| `--threshold` | Umbral θ de confianza | 0.7 |
| `--alpha` | Tolerancia adaptativa α | 0.0 |
| `--output_csv` | Archivo CSV para resultados | results.csv |
| `--log_dir` | Directorio para logs de Gurobi | None |

## Formatos de Instancia Soportados

- `.lp` - LP format (Gurobi nativo)
- `.mps` - MPS format (estándar de la industria)

## Interpretación de Resultados

El archivo CSV contendrá las siguientes columnas clave:

| Columna | Descripción |
|---------|-------------|
| `method` | 'baseline' o 'backpas' |
| `runtime` | Tiempo total de ejecución |
| `obj_val` | Valor objetivo de la solución |
| `obj_bound` | Mejor cota (bound) |
| `mip_gap` | Gap de optimalidad |
| `primal_integral` | Métrica de anytime performance (menor es mejor) |
| `n_nodes` | Nodos del árbol explorados |

Columnas adicionales para BACKPAS:

| Columna | Descripción |
|---------|-------------|
| `phase1_time` | Tiempo de Fase 1 |
| `phase1_obj` | Mejor solución en Fase 1 |
| `phase2_time` | Tiempo de Fase 2 |
| `phase2_obj` | Mejor solución en Fase 2 |
| `total_nodes` | Nodos totales (Fase 1 + Fase 2) |
| `phase2_improved` | Si Fase 2 mejoró el resultado de Fase 1 |
| `warmstart_used` | Si se usó warmstart |
| `k_0`, `k_1`, `Delta` | Parámetros de trust region |

### Métricas de Éxito

BACKPAS es exitoso si:
1. **Primal integral menor** que baseline - Encuentra buenas soluciones más rápido
2. **Mismo valor objetivo** final - No pierde optimalidad
3. **Tiempo similar o menor** - No agrega overhead significativo

## Monitoreo de Procesos en Background

```bash
# Ver si sigue corriendo
ps aux | grep run_gurobi

# Ver output en tiempo real
tail -f baseline.log

# Ver cuántas instancias completadas
grep "Ejecutando:" baseline.log | wc -l

# Ver último resultado
tail -20 baseline.log

# Detener proceso
kill %1          # Por job number
kill <PID>       # Por PID
```

## Troubleshooting

### Error: "Módulos BACKPAS no disponibles"
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install torch_geometric torch_scatter torch_sparse
```

### Error: "Version number is 13.0, license is for version 12.0"
```bash
pip uninstall gurobipy
pip install gurobipy==12.0.0
```

### Error: "No se encontraron archivos .lp"
El script busca archivos `.lp` y `.mps`. Verifica que el directorio contiene
instancias en alguno de estos formatos:
```bash
ls directorio/*.lp directorio/*.mps
```

## Scripts Disponibles

| Script | Propósito |
|--------|-----------|
| `run_gurobi_experiment.py` | Script principal - BASELINE y BACKPAS |
| `parameter_search.py` | Grid search para encontrar mejores θ y α |
| `analyze_results.py` | Análisis estadístico + tabla LaTeX |
| `generate_mis_instances.py` | Generador de instancias MIS aleatorias |
