#!/bin/bash
# =============================================================================
# Script para ejecutar el experimento completo en servidor SSH
#
# Tesis: Aceleración de la Demostración de Optimalidad en Gurobi mediante
#        Predicción de Variables Backbone para el Problema de Maximum Independent Set
#
# Uso:
#   chmod +x run_full_experiment.sh
#   ./run_full_experiment.sh
# =============================================================================

set -e  # Salir si hay error

# Colores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuración
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
SRC_DIR="${PROJECT_DIR}/src"
INSTANCES_DIR="${PROJECT_DIR}/instances"
RESULTS_DIR="${PROJECT_DIR}/results"

# Parámetros del experimento
N_INSTANCES=20
N_NODES=500          # Ajustar después de calibración
TIME_LIMIT=3600      # 1 hora
THREADS=1
SEED=42

# Parámetros BACKPAS
TRUST_REGION_TIME=300
THRESHOLD=0.7
ALPHA=0.0

# Ruta al modelo BACKPAS (AJUSTAR SEGÚN TU CONFIGURACIÓN)
MODEL_PATH="${PROJECT_DIR}/../wkdir/MIS/ml_training/graph_with_literals_8_GTR/best_model.pth"

echo -e "${GREEN}============================================${NC}"
echo -e "${GREEN}   EXPERIMENTO COMPLETO - TESIS BACKBONE   ${NC}"
echo -e "${GREEN}============================================${NC}"
echo ""
echo "Configuración:"
echo "  - Instancias: ${N_INSTANCES}"
echo "  - Nodos: ${N_NODES}"
echo "  - Tiempo límite: ${TIME_LIMIT}s"
echo "  - Hilos: ${THREADS}"
echo "  - Semilla: ${SEED}"
echo "  - Trust region time: ${TRUST_REGION_TIME}s"
echo "  - Threshold (θ): ${THRESHOLD}"
echo "  - Alpha (α): ${ALPHA}"
echo ""

# =============================================================================
# PASO 1: Generar instancias
# =============================================================================
echo -e "${YELLOW}[PASO 1/4] Generando instancias MIS...${NC}"

INSTANCE_OUTPUT_DIR="${INSTANCES_DIR}/n${N_NODES}_seed${SEED}"

if [ -d "${INSTANCE_OUTPUT_DIR}" ] && [ "$(ls -A ${INSTANCE_OUTPUT_DIR}/*.lp 2>/dev/null)" ]; then
    echo -e "${GREEN}Instancias ya existen en: ${INSTANCE_OUTPUT_DIR}. Saltando generación.${NC}"
else
    python "${SRC_DIR}/generate_mis_instances.py" \
        --n_instances ${N_INSTANCES} \
        --n_nodes ${N_NODES} \
        --output_dir "${INSTANCE_OUTPUT_DIR}" \
        --prefix "mis" \
        --seed ${SEED}
    echo -e "${GREEN}Instancias generadas en: ${INSTANCE_OUTPUT_DIR}${NC}"
fi
echo ""

# =============================================================================
# PASO 2: Ejecutar baseline (sin backbone)
# =============================================================================
echo -e "${YELLOW}[PASO 2/4] Ejecutando experimentos BASELINE...${NC}"
echo "Esto puede tomar varias horas..."

BASELINE_CSV="${RESULTS_DIR}/metrics/baseline_n${N_NODES}.csv"

python "${SRC_DIR}/run_gurobi_experiment.py" \
    --instance_dir "${INSTANCE_OUTPUT_DIR}" \
    --output_csv "${BASELINE_CSV}" \
    --log_dir "${RESULTS_DIR}/logs/baseline" \
    --threads ${THREADS} \
    --time_limit ${TIME_LIMIT}

echo -e "${GREEN}Baseline completado.${NC}"
echo ""

# =============================================================================
# PASO 3: Ejecutar BACKPAS (con trust region en dos fases)
# =============================================================================
echo -e "${YELLOW}[PASO 3/4] Ejecutando experimentos BACKPAS...${NC}"
echo "Esto puede tomar varias horas..."

BACKPAS_CSV="${RESULTS_DIR}/metrics/backpas_n${N_NODES}_t${TRUST_REGION_TIME}_th${THRESHOLD}_a${ALPHA}.csv"

if [ ! -f "${MODEL_PATH}" ]; then
    echo -e "${RED}ERROR: Modelo no encontrado en ${MODEL_PATH}${NC}"
    echo -e "${RED}Ajusta la variable MODEL_PATH en este script${NC}"
    exit 1
fi

python "${SRC_DIR}/run_gurobi_experiment.py" \
    --instance_dir "${INSTANCE_OUTPUT_DIR}" \
    --backpas \
    --model_path "${MODEL_PATH}" \
    --trust_region_time ${TRUST_REGION_TIME} \
    --threshold ${THRESHOLD} \
    --alpha ${ALPHA} \
    --output_csv "${BACKPAS_CSV}" \
    --log_dir "${RESULTS_DIR}/logs/backpas" \
    --threads ${THREADS} \
    --time_limit ${TIME_LIMIT}

echo -e "${GREEN}BACKPAS completado.${NC}"
echo ""

# =============================================================================
# PASO 4: Analizar resultados
# =============================================================================
echo -e "${YELLOW}[PASO 4/4] Analizando resultados...${NC}"

python "${SRC_DIR}/analyze_results.py" \
    --baseline "${BASELINE_CSV}" \
    --backbone "${BACKPAS_CSV}" \
    --output_dir "${RESULTS_DIR}/analysis" \
    --latex_table "${PROJECT_DIR}/thesis/figures/comparison_table.tex"

echo ""
echo -e "${GREEN}============================================${NC}"
echo -e "${GREEN}   EXPERIMENTO COMPLETADO EXITOSAMENTE     ${NC}"
echo -e "${GREEN}============================================${NC}"
echo ""
echo "Resultados guardados en:"
echo "  - Métricas baseline: ${BASELINE_CSV}"
echo "  - Métricas BACKPAS:  ${BACKPAS_CSV}"
echo "  - Análisis:          ${RESULTS_DIR}/analysis/"
echo "  - Tabla LaTeX:       ${PROJECT_DIR}/thesis/figures/comparison_table.tex"
echo ""
