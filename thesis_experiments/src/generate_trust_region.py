#!/usr/bin/env python3
"""
Script wrapper para generar Trust Regions usando el modelo BACKPAS existente.

Este script toma instancias MIS baseline y genera versiones con restricciones
de trust region basadas en predicciones backbone del modelo de ML.

"""

import argparse
import sys
import os
from pathlib import Path
from glob import glob
from typing import Optional

# Agregar el directorio src del repositorio principal al path
REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import torch

# Importar módulos del repositorio BACKPAS
try:
    from GCN import BackbonePredictor
    from create_trust_region import (
        ThresholdedExpectedErrorTrustRegionConstructor,
        ThresholdedWeightedBudgetTrustRegionConstructor,
        FixedTwoRatiosTrustRegionConstructor,
        FixedThreeRatiosTrustRegionConstructor
    )
    from constants import LITERALS_GRAPH, VARIABLES_GRAPH
    BACKPAS_AVAILABLE = True
except ImportError as e:
    print(f"Advertencia: No se pudieron importar módulos BACKPAS: {e}")
    print("Asegúrate de que el repositorio BACKPAS esté correctamente configurado.")
    BACKPAS_AVAILABLE = False


def load_model(
    model_path: str,
    graph_type: str = LITERALS_GRAPH,
    num_layers: int = 8,
    layer_type: str = "GTR",
    use_literals_message: bool = False,
    device: str = "cpu"
) -> "BackbonePredictor":
    """
    Carga el modelo de predicción backbone.
    
    Args:
        model_path: Ruta al archivo .pth del modelo
        graph_type: Tipo de grafo ('literals' o 'variables')
        num_layers: Número de capas de la GNN
        layer_type: Tipo de capa ('GTR', 'GCN', etc.)
        use_literals_message: Si usar message passing de literales
        device: Dispositivo ('cpu' o 'cuda')
    
    Returns:
        Modelo cargado y en modo evaluación
    """
    if not BACKPAS_AVAILABLE:
        raise RuntimeError("Módulos BACKPAS no disponibles")
    
    model = BackbonePredictor(
        graph_type=graph_type,
        num_layers=num_layers,
        layer_type=layer_type,
        use_literals_message=use_literals_message
    ).to(device)
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    print(f"Modelo cargado: {model_path}")
    print(f"  - Tipo de grafo: {graph_type}")
    print(f"  - Capas: {num_layers}")
    print(f"  - Tipo de capa: {layer_type}")
    print(f"  - Dispositivo: {device}")
    
    return model


def generate_trust_region(
    model,
    instance_input_path: str,
    instance_output_path: str,
    method: str = "thresholded_expected_error",
    graph_type: str = LITERALS_GRAPH,
    threshold: float = 0.7,
    alpha: float = 0.0,
    budget_ratio: float = 0.3,
    k_ratio: float = 0.5,
    value_0_ratio: float = 0.5,
    delta_ratio: float = 0.1,
    log_file: Optional[str] = None
):
    """
    Genera una instancia con trust region.
    
    Args:
        model: Modelo de predicción backbone cargado
        instance_input_path: Ruta a la instancia original
        instance_output_path: Ruta donde guardar la instancia con trust region
        method: Método de construcción de trust region
        graph_type: Tipo de grafo
        threshold: Umbral de confianza (para métodos thresholded)
        alpha: Parámetro alpha (para thresholded_expected_error)
        budget_ratio: Ratio de presupuesto (para thresholded_weighted_budget)
        k_ratio: Ratio k (para métodos fixed)
        value_0_ratio: Ratio de valor 0 (para fixed_three_ratios)
        delta_ratio: Ratio delta (para métodos fixed)
        log_file: Archivo de log
    """
    if not BACKPAS_AVAILABLE:
        raise RuntimeError("Módulos BACKPAS no disponibles")
    
    # Crear archivo de log temporal si no se especificó
    if log_file is None:
        log_file = instance_output_path + ".construction_log.txt"
    
    # Crear constructor según el método
    if method == "thresholded_expected_error":
        constructor = ThresholdedExpectedErrorTrustRegionConstructor(
            ml_model=model,
            graph_type=graph_type,
            threshold=threshold,
            alpha=alpha,
            log_file=log_file
        )
    elif method == "thresholded_weighted_budget":
        constructor = ThresholdedWeightedBudgetTrustRegionConstructor(
            ml_model=model,
            graph_type=graph_type,
            threshold=threshold,
            budget=budget_ratio,
            log_file=log_file
        )
    elif method == "fixed_two_ratios":
        constructor = FixedTwoRatiosTrustRegionConstructor(
            ml_model=model,
            graph_type=graph_type,
            k_ratio=k_ratio,
            Delta_ratio=delta_ratio,
            log_file=log_file
        )
    elif method == "fixed_three_ratios":
        constructor = FixedThreeRatiosTrustRegionConstructor(
            ml_model=model,
            graph_type=graph_type,
            k_ratio=k_ratio,
            value_0_ratio=value_0_ratio,
            Delta_ratio=delta_ratio,
            log_file=log_file
        )
    else:
        raise ValueError(f"Método no soportado: {method}")
    
    # Procesar instancia
    constructor.process_instance(instance_input_path, instance_output_path)


def process_batch(
    model,
    input_dir: str,
    output_dir: str,
    method: str = "thresholded_expected_error",
    graph_type: str = LITERALS_GRAPH,
    pattern: str = "*.lp",
    **method_params
):
    """
    Procesa múltiples instancias en batch.
    
    Args:
        model: Modelo cargado
        input_dir: Directorio con instancias originales
        output_dir: Directorio de salida
        method: Método de trust region
        graph_type: Tipo de grafo
        pattern: Patrón de archivos
        **method_params: Parámetros específicos del método
    """
    # Encontrar instancias
    instance_files = sorted(glob(os.path.join(input_dir, pattern)))
    
    if not instance_files:
        print(f"No se encontraron archivos {pattern} en {input_dir}")
        return
    
    print(f"\nProcesando {len(instance_files)} instancias")
    print(f"Método: {method}")
    print(f"Salida: {output_dir}")
    
    # Crear directorio de salida
    os.makedirs(output_dir, exist_ok=True)
    
    # Log global
    log_file = os.path.join(output_dir, "batch_construction.log")
    
    for i, input_path in enumerate(instance_files):
        instance_name = Path(input_path).name
        output_path = os.path.join(output_dir, instance_name)
        
        print(f"[{i+1}/{len(instance_files)}] {instance_name}...", end=" ")
        
        try:
            generate_trust_region(
                model=model,
                instance_input_path=input_path,
                instance_output_path=output_path,
                method=method,
                graph_type=graph_type,
                log_file=log_file,
                **method_params
            )
            print("OK")
        except Exception as e:
            print(f"ERROR: {e}")
    
    print(f"\nProcesamiento completado. Resultados en: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Generar Trust Regions usando modelo BACKPAS",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:
  # Procesar una instancia
  python generate_trust_region.py \\
      --model_path /path/to/best_model.pth \\
      --instance ../instances/baseline/mis_500n_001.lp \\
      --output ../instances/with_backbone/mis_500n_001.lp

  # Procesar múltiples instancias
  python generate_trust_region.py \\
      --model_path /path/to/best_model.pth \\
      --input_dir ../instances/baseline \\
      --output_dir ../instances/with_backbone

  # Usar diferentes parámetros de trust region
  python generate_trust_region.py \\
      --model_path /path/to/best_model.pth \\
      --input_dir ../instances/baseline \\
      --output_dir ../instances/with_backbone \\
      --method thresholded_expected_error \\
      --threshold 0.8 \\
      --alpha 0.1

Métodos disponibles:
  - thresholded_expected_error: Usa umbral θ y parámetro α
  - thresholded_weighted_budget: Usa umbral θ y ratio de presupuesto
  - fixed_two_ratios: Usa ratios k y Δ
  - fixed_three_ratios: Usa ratios k, valor_0 y Δ
        """
    )
    
    # Modelo
    parser.add_argument("--model_path", type=str, required=True,
                        help="Ruta al archivo .pth del modelo")
    parser.add_argument("--graph_type", type=str, default="literals",
                        choices=["literals", "variables"],
                        help="Tipo de grafo (default: literals)")
    parser.add_argument("--num_layers", type=int, default=8,
                        help="Número de capas GNN (default: 8)")
    parser.add_argument("--layer_type", type=str, default="GTR",
                        help="Tipo de capa (default: GTR)")
    parser.add_argument("--use_cuda", action="store_true",
                        help="Usar CUDA si está disponible")
    
    # Input/Output
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--instance", type=str,
                       help="Ruta a una sola instancia")
    group.add_argument("--input_dir", type=str,
                       help="Directorio con múltiples instancias")
    
    parser.add_argument("--output", type=str,
                        help="Ruta de salida (para una instancia)")
    parser.add_argument("--output_dir", type=str,
                        help="Directorio de salida (para batch)")
    
    # Método de trust region
    parser.add_argument("--method", type=str, default="thresholded_expected_error",
                        choices=["thresholded_expected_error", "thresholded_weighted_budget",
                                "fixed_two_ratios", "fixed_three_ratios"],
                        help="Método de construcción (default: thresholded_expected_error)")
    
    # Parámetros de trust region
    parser.add_argument("--threshold", type=float, default=0.7,
                        help="Umbral de confianza θ (default: 0.7)")
    parser.add_argument("--alpha", type=float, default=0.0,
                        help="Parámetro α para expected_error (default: 0.0)")
    parser.add_argument("--budget_ratio", type=float, default=0.3,
                        help="Ratio de presupuesto para weighted_budget (default: 0.3)")
    parser.add_argument("--k_ratio", type=float, default=0.5,
                        help="Ratio k para métodos fixed (default: 0.5)")
    parser.add_argument("--value_0_ratio", type=float, default=0.5,
                        help="Ratio valor_0 para fixed_three (default: 0.5)")
    parser.add_argument("--delta_ratio", type=float, default=0.1,
                        help="Ratio Δ para métodos fixed (default: 0.1)")
    
    args = parser.parse_args()
    
    if not BACKPAS_AVAILABLE:
        print("ERROR: No se pueden importar los módulos BACKPAS.")
        print("Verifica que estés en el entorno correcto y que el repositorio esté configurado.")
        sys.exit(1)
    
    # Determinar dispositivo
    device = "cuda" if args.use_cuda and torch.cuda.is_available() else "cpu"
    
    # Mapear tipo de grafo
    graph_type = LITERALS_GRAPH if args.graph_type == "literals" else VARIABLES_GRAPH
    
    # Cargar modelo
    print("="*60)
    print("GENERADOR DE TRUST REGIONS")
    print("="*60)
    
    model = load_model(
        model_path=args.model_path,
        graph_type=graph_type,
        num_layers=args.num_layers,
        layer_type=args.layer_type,
        device=device
    )
    
    # Preparar parámetros del método
    method_params = {
        "threshold": args.threshold,
        "alpha": args.alpha,
        "budget_ratio": args.budget_ratio,
        "k_ratio": args.k_ratio,
        "value_0_ratio": args.value_0_ratio,
        "delta_ratio": args.delta_ratio
    }
    
    if args.instance:
        # Procesar una instancia
        output_path = args.output or args.instance.replace("baseline", "with_backbone")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        print(f"\nProcesando: {args.instance}")
        print(f"Salida: {output_path}")
        
        generate_trust_region(
            model=model,
            instance_input_path=args.instance,
            instance_output_path=output_path,
            method=args.method,
            graph_type=graph_type,
            **method_params
        )
        print("Completado.")
    else:
        # Procesar batch
        output_dir = args.output_dir or args.input_dir.replace("baseline", "with_backbone")
        
        process_batch(
            model=model,
            input_dir=args.input_dir,
            output_dir=output_dir,
            method=args.method,
            graph_type=graph_type,
            **method_params
        )


if __name__ == "__main__":
    main()
