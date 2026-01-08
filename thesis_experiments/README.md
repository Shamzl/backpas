# Experimentos de Tesis: Aceleración de Gurobi con Backbone

Este directorio contiene los scripts y datos para los experimentos de la tesis:

**"Aceleración de la Demostración de Optimalidad en Gurobi mediante Predicción de Variables Backbone para el Problema de Maximum Independent Set"**

## Estructura del Directorio

```
thesis_experiments/
├── README.md                 # Este archivo
├── src/
│   ├── generate_mis_instances.py   # Generador de instancias MIS
│   ├── run_gurobi_experiment.py    # Ejecutor de experimentos Gurobi
│   ├── generate_trust_region.py    # Generador de trust regions
│   └── analyze_results.py          # Análisis estadístico
├── instances/
│   ├── baseline/             # Instancias MIS originales
│   └── with_backbone/        # Instancias con trust region
├── results/
│   ├── logs/                 # Logs de Gurobi
│   └── metrics/              # CSVs con métricas
├── thesis/                   # Documento LaTeX
│   ├── chapters/
│   └── figures/
└── scripts/
    └── run_experiments.sh    # Script para servidor SSH
```

## Requisitos

### Dependencias Python
```bash
pip install networkx gurobipy pandas numpy scipy
```

### Dependencias para Trust Region (opcional)
Si vas a generar trust regions, necesitas el entorno del repositorio BACKPAS:
```bash
pip install torch pyscipopt
```

### Licencia Gurobi
Necesitas una licencia de Gurobi válida. Para académicos:
https://www.gurobi.com/academia/academic-program-and-licenses/

## Guía Rápida

### Paso 1: Generar Instancias MIS

```bash
cd src/

# Generar 5 instancias pequeñas para prueba (100 nodos)
python generate_mis_instances.py \
    --n_instances 5 \
    --n_nodes 100 \
    --output_dir ../instances/baseline/small \
    --seed 42

# Generar 10 instancias medianas (500 nodos)
python generate_mis_instances.py \
    --n_instances 10 \
    --n_nodes 500 \
    --output_dir ../instances/baseline/medium \
    --seed 42

# Generar 20 instancias grandes para experimentos finales (2000 nodos)
python generate_mis_instances.py \
    --n_instances 20 \
    --n_nodes 2000 \
    --output_dir ../instances/baseline/large \
    --seed 42
```

### Paso 2: Ejecutar Baseline (sin backbone)

```bash
# Ejecutar una sola instancia (para probar)
python run_gurobi_experiment.py \
    --instance ../instances/baseline/small/mis_100n_000.lp \
    --time_limit 60

# Ejecutar batch de instancias
python run_gurobi_experiment.py \
    --instance_dir ../instances/baseline/small \
    --output_csv ../results/metrics/baseline_small.csv \
    --time_limit 300 \
    --threads 1
```

### Paso 3: Generar Trust Regions (con modelo BACKPAS)

```bash
# Procesar instancias con el modelo backbone
python generate_trust_region.py \
    --model_path /ruta/a/best_model.pth \
    --input_dir ../instances/baseline/small \
    --output_dir ../instances/with_backbone/small \
    --method thresholded_expected_error \
    --threshold 0.7 \
    --alpha 0.0
```

### Paso 4: Ejecutar con Backbone

```bash
python run_gurobi_experiment.py \
    --instance_dir ../instances/with_backbone/small \
    --output_csv ../results/metrics/backbone_small.csv \
    --time_limit 300 \
    --threads 1
```

### Paso 5: Analizar Resultados

```bash
python analyze_results.py \
    --baseline ../results/metrics/baseline_small.csv \
    --backbone ../results/metrics/backbone_small.csv \
    --output_dir ../results/analysis
```

## Parámetros Importantes

### Generador de Instancias (`generate_mis_instances.py`)

| Parámetro | Descripción | Default |
|-----------|-------------|---------|
| `--n_instances` | Número de instancias | 10 |
| `--n_nodes` | Nodos por instancia | 500 |
| `--graph_type` | `erdos_renyi` o `barabasi_albert` | `erdos_renyi` |
| `--edge_prob` | Probabilidad de arista | 0.5 |
| `--seed` | Semilla para reproducibilidad | None |

### Ejecutor Gurobi (`run_gurobi_experiment.py`)

| Parámetro | Descripción | Default |
|-----------|-------------|---------|
| `--threads` | Hilos de Gurobi | 1 |
| `--time_limit` | Límite de tiempo (seg) | 3600 |
| `--mip_gap` | Gap objetivo | 0.0 |

### Trust Region (`generate_trust_region.py`)

| Parámetro | Descripción | Default |
|-----------|-------------|---------|
| `--method` | Método de construcción | `thresholded_expected_error` |
| `--threshold` | Umbral θ | 0.7 |
| `--alpha` | Parámetro α | 0.0 |

## Calibración de Tamaño de Instancias

Para encontrar el tamaño de instancia donde Gurobi tarde ~1 hora:

```bash
# Probar con diferentes tamaños
for nodes in 500 1000 1500 2000 2500; do
    python generate_mis_instances.py \
        --n_instances 3 \
        --n_nodes $nodes \
        --output_dir ../instances/calibration/${nodes}n \
        --seed 42
    
    python run_gurobi_experiment.py \
        --instance_dir ../instances/calibration/${nodes}n \
        --output_csv ../results/metrics/calibration_${nodes}n.csv \
        --time_limit 3600
done
```

## Ejecutar en Servidor SSH

1. Copiar este directorio al servidor:
```bash
scp -r thesis_experiments/ usuario@servidor:/ruta/destino/
```

2. Ejecutar en background con nohup:
```bash
nohup python run_gurobi_experiment.py \
    --instance_dir ../instances/baseline/large \
    --output_csv ../results/metrics/baseline_large.csv \
    --time_limit 3600 \
    > experiment.log 2>&1 &
```

3. Verificar progreso:
```bash
tail -f experiment.log
```

## Métricas Recopiladas

| Métrica | Descripción |
|---------|-------------|
| `runtime` | Tiempo total de ejecución |
| `gurobi_runtime` | Tiempo reportado por Gurobi |
| `obj_val` | Valor objetivo óptimo |
| `obj_bound` | Cota del valor objetivo |
| `mip_gap` | Gap de optimalidad |
| `n_nodes` | Nodos del árbol explorados |
| `n_solutions` | Soluciones factibles encontradas |
| `primal_integral` | Integral primal (aproximado) |

## Notas

- **Un solo hilo**: Todos los experimentos deben usar `--threads 1` para comparación justa.
- **Gap = 0**: Usamos `--mip_gap 0.0` para garantizar optimalidad demostrada.
- **Reproducibilidad**: Usar siempre la misma semilla (`--seed 42`) para generar instancias.
- **Time limit**: Ajustar según la capacidad de cómputo disponible.

## Contacto

