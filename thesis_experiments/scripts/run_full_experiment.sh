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
echo ""

# =============================================================================
# PASO 1: Generar instancias
# =============================================================================
echo -e "${YELLOW}[PASO 1/5] Generando instancias MIS...${NC}"

python "${SRC_DIR}/generate_mis_instances.py" \
    --n_instances ${N_INSTANCES} \
    --n_nodes ${N_NODES} \
    --output_dir "${INSTANCES_DIR}/baseline" \
    --prefix "mis" \
    --seed ${SEED}

echo -e "${GREEN}Instancias generadas en: ${INSTANCES_DIR}/baseline${NC}"
echo ""

# =============================================================================
# PASO 2: Ejecutar baseline (sin backbone)
# =============================================================================
echo -e "${YELLOW}[PASO 2/5] Ejecutando experimentos BASELINE...${NC}"
echo "Esto puede tomar varias horas..."

python "${SRC_DIR}/run_gurobi_experiment.py" \
    --instance_dir "${INSTANCES_DIR}/baseline" \
    --output_csv "${RESULTS_DIR}/metrics/baseline_results.csv" \
    --log_dir "${RESULTS_DIR}/logs/baseline" \
    --threads ${THREADS} \
    --time_limit ${TIME_LIMIT}

echo -e "${GREEN}Baseline completado.${NC}"
echo ""

# =============================================================================
# PASO 3: Generar trust regions
# =============================================================================
echo -e "${YELLOW}[PASO 3/5] Generando Trust Regions con modelo BACKPAS...${NC}"

if [ -f "${MODEL_PATH}" ]; then
    python "${SRC_DIR}/generate_trust_region.py" \
        --model_path "${MODEL_PATH}" \
        --input_dir "${INSTANCES_DIR}/baseline" \
        --output_dir "${INSTANCES_DIR}/with_backbone" \
        --method thresholded_expected_error \
        --threshold 0.7 \
        --alpha 0.0
    
    echo -e "${GREEN}Trust regions generadas en: ${INSTANCES_DIR}/with_backbone${NC}"
else
    echo -e "${RED}ERROR: Modelo no encontrado en ${MODEL_PATH}${NC}"
    echo -e "${RED}Ajusta la variable MODEL_PATH en este script${NC}"
    exit 1
fi
echo ""

# =============================================================================
# PASO 4: Ejecutar con backbone
# =============================================================================
echo -e "${YELLOW}[PASO 4/5] Ejecutando experimentos CON BACKBONE...${NC}"
echo "Esto puede tomar varias horas..."

python "${SRC_DIR}/run_gurobi_experiment.py" \
    --instance_dir "${INSTANCES_DIR}/with_backbone" \
    --output_csv "${RESULTS_DIR}/metrics/backbone_results.csv" \
    --log_dir "${RESULTS_DIR}/logs/backbone" \
    --threads ${THREADS} \
    --time_limit ${TIME_LIMIT}

echo -e "${GREEN}Backbone completado.${NC}"
echo ""

# =============================================================================
# PASO 5: Analizar resultados
# =============================================================================
echo -e "${YELLOW}[PASO 5/5] Analizando resultados...${NC}"

python "${SRC_DIR}/analyze_results.py" \
    --baseline "${RESULTS_DIR}/metrics/baseline_results.csv" \
    --backbone "${RESULTS_DIR}/metrics/backbone_results.csv" \
    --output_dir "${RESULTS_DIR}/analysis" \
    --latex_table "${PROJECT_DIR}/thesis/figures/comparison_table.tex"

echo ""
echo -e "${GREEN}============================================${NC}"
echo -e "${GREEN}   EXPERIMENTO COMPLETADO EXITOSAMENTE     ${NC}"
echo -e "${GREEN}============================================${NC}"
echo ""
echo "Resultados guardados en:"
echo "  - Métricas baseline: ${RESULTS_DIR}/metrics/baseline_results.csv"
echo "  - Métricas backbone: ${RESULTS_DIR}/metrics/backbone_results.csv"
echo "  - Análisis: ${RESULTS_DIR}/analysis/"
echo "  - Tabla LaTeX: ${PROJECT_DIR}/thesis/figures/comparison_table.tex"
echo ""
