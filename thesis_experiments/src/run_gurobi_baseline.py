#!/usr/bin/env python3
"""
Script para ejecutar experimentos BASELINE con Gurobi sobre instancias MIS.

Este script ejecuta Gurobi sin ninguna ayuda del modelo ML (baseline puro).
Sirve como punto de comparación para evaluar el método BACKPAS.

Métricas recopiladas:
- Tiempo de ejecución
- Valor óptimo / mejor solución
- Gap de optimalidad
- Nodos explorados
- Primal integral
- Historial de incumbentes (para análisis detallado)

"""

import argparse
import gurobipy as gp
from gurobipy import GRB
import os
import csv
import time
import json
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List
from glob import glob


class GurobiBaselineExperiment:
    """Clase para ejecutar experimentos MIS baseline con Gurobi."""
    
    def __init__(
        self,
        threads: int = 1,
        time_limit: float = 3600,
        mip_gap: float = 0.0,
        log_dir: Optional[str] = None,
        save_history: bool = True
    ):
        """
        Inicializa la configuración del experimento baseline.
        
        Args:
            threads: Número de hilos (default: 1)
            time_limit: Límite de tiempo en segundos (default: 3600 = 1 hora)
            mip_gap: Gap de optimalidad objetivo (default: 0.0 = óptimo exacto)
            log_dir: Directorio para logs de Gurobi
            save_history: Si guardar historial de incumbentes
        """
        self.threads = threads
        self.time_limit = time_limit
        self.mip_gap = mip_gap
        self.log_dir = log_dir
        self.save_history = save_history
        
        # Para tracking del primal integral
        self.incumbent_history = []
        
    def _callback(self, model, where):
        """Callback para capturar el progreso durante la optimización."""
        if where == GRB.Callback.MIP:
            try:
                obj_best = model.cbGet(GRB.Callback.MIP_OBJBST)
                obj_bound = model.cbGet(GRB.Callback.MIP_OBJBND)
                runtime = model.cbGet(GRB.Callback.RUNTIME)
                
                self.incumbent_history.append({
                    'time': runtime,
                    'incumbent': obj_best,
                    'bound': obj_bound
                })
            except:
                pass
    
    def run_instance(self, instance_path: str, verbose: bool = True) -> Dict:
        """
        Ejecuta Gurobi (baseline) sobre una instancia MIS.
        
        Args:
            instance_path: Ruta al archivo .lp
            verbose: Si mostrar progreso
        
        Returns:
            Diccionario con métricas del experimento
        """
        instance_name = Path(instance_path).stem
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"BASELINE - Ejecutando: {instance_name}")
            print(f"{'='*60}")
        
        # Resetear historial
        self.incumbent_history = []
        
        # Crear modelo
        model = gp.read(instance_path)
        
        # Configurar parámetros
        model.Params.Threads = self.threads
        model.Params.TimeLimit = self.time_limit
        model.Params.MIPGap = self.mip_gap
        
        # Configurar log si se especificó directorio
        if self.log_dir:
            os.makedirs(self.log_dir, exist_ok=True)
            log_file = os.path.join(self.log_dir, f"{instance_name}_baseline.log")
            model.Params.LogFile = log_file
        
        if not verbose:
            model.Params.OutputFlag = 0
        
        # Ejecutar optimización con callback
        start_time = time.time()
        model.optimize(self._callback)
        end_time = time.time()
        
        # Recopilar métricas
        metrics = {
            'instance_name': instance_name,
            'instance_path': instance_path,
            'method': 'baseline',
            'status': model.Status,
            'status_name': self._status_to_string(model.Status),
            'runtime': end_time - start_time,
            'gurobi_runtime': model.Runtime,
            'n_vars': model.NumVars,
            'n_constrs': model.NumConstrs,
            'n_nodes': model.NodeCount,
            'n_solutions': model.SolCount,
            'mip_gap': model.MIPGap if model.SolCount > 0 else float('inf'),
            'obj_val': model.ObjVal if model.SolCount > 0 else None,
            'obj_bound': model.ObjBound,
            'threads': self.threads,
            'time_limit': self.time_limit,
            'timestamp': datetime.now().isoformat()
        }
        
        # Calcular primal integral
        if len(self.incumbent_history) > 0:
            metrics['primal_integral'] = self._compute_primal_integral(
                metrics['obj_bound'], 
                self.time_limit
            )
        else:
            metrics['primal_integral'] = None
        
        # Guardar historial si se solicita
        if self.save_history:
            metrics['incumbent_history'] = self.incumbent_history
        
        if verbose:
            self._print_metrics(metrics)
        
        return metrics
    
    def _compute_primal_integral(self, best_known: float, time_limit: float) -> float:
        """
        Calcula el primal integral.
        
        El primal integral mide el área entre la curva del incumbent
        y el mejor bound conocido a lo largo del tiempo.
        
        Args:
            best_known: Mejor valor conocido (bound)
            time_limit: Límite de tiempo total
        
        Returns:
            Valor del primal integral normalizado
        """
        if len(self.incumbent_history) < 2:
            return 0.0
        
        integral = 0.0
        for i in range(1, len(self.incumbent_history)):
            t_prev = self.incumbent_history[i-1]['time']
            t_curr = self.incumbent_history[i]['time']
            inc_prev = self.incumbent_history[i-1]['incumbent']
            
            if inc_prev != GRB.INFINITY and best_known != 0:
                gap = abs(best_known - inc_prev) / abs(best_known)
                dt = t_curr - t_prev
                integral += gap * dt
        
        return integral
    
    def _status_to_string(self, status: int) -> str:
        """Convierte código de estado de Gurobi a string."""
        status_map = {
            GRB.OPTIMAL: "OPTIMAL",
            GRB.INFEASIBLE: "INFEASIBLE",
            GRB.INF_OR_UNBD: "INF_OR_UNBD",
            GRB.UNBOUNDED: "UNBOUNDED",
            GRB.CUTOFF: "CUTOFF",
            GRB.ITERATION_LIMIT: "ITERATION_LIMIT",
            GRB.NODE_LIMIT: "NODE_LIMIT",
            GRB.TIME_LIMIT: "TIME_LIMIT",
            GRB.SOLUTION_LIMIT: "SOLUTION_LIMIT",
            GRB.INTERRUPTED: "INTERRUPTED",
            GRB.NUMERIC: "NUMERIC",
            GRB.SUBOPTIMAL: "SUBOPTIMAL",
        }
        return status_map.get(status, f"UNKNOWN({status})")
    
    def _print_metrics(self, metrics: Dict):
        """Imprime métricas de forma formateada."""
        print(f"\nResultados BASELINE:")
        print(f"  Estado: {metrics['status_name']}")
        print(f"  Tiempo: {metrics['runtime']:.2f} seg")
        print(f"  Valor objetivo: {metrics['obj_val']}")
        print(f"  Bound: {metrics['obj_bound']}")
        print(f"  Gap: {metrics['mip_gap']*100:.4f}%")
        print(f"  Nodos explorados: {metrics['n_nodes']}")
        print(f"  Soluciones encontradas: {metrics['n_solutions']}")
        if metrics['primal_integral'] is not None:
            print(f"  Primal integral: {metrics['primal_integral']:.4f}")


def save_metrics_to_csv(metrics_list: List[Dict], output_path: str):
    """Guarda métricas en archivo CSV."""
    if not metrics_list:
        return
    
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    
    fieldnames = [
        'instance_name', 'method', 'status_name', 'runtime', 'gurobi_runtime',
        'obj_val', 'obj_bound', 'mip_gap', 'n_nodes', 'n_solutions',
        'n_vars', 'n_constrs', 'primal_integral', 'threads', 'time_limit', 'timestamp'
    ]
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(metrics_list)


def save_history_to_json(metrics: Dict, output_dir: str):
    """Guarda el historial de incumbentes en JSON para análisis posterior."""
    if 'incumbent_history' not in metrics or not metrics['incumbent_history']:
        return
    
    os.makedirs(output_dir, exist_ok=True)
    history_file = os.path.join(output_dir, f"{metrics['instance_name']}_baseline_history.json")
    
    with open(history_file, 'w') as f:
        json.dump({
            'instance_name': metrics['instance_name'],
            'method': 'baseline',
            'history': metrics['incumbent_history']
        }, f, indent=2)


def run_batch_experiment(
    instance_dir: str,
    output_csv: str,
    threads: int = 1,
    time_limit: float = 3600,
    mip_gap: float = 0.0,
    log_dir: Optional[str] = None,
    history_dir: Optional[str] = None,
    pattern: str = "*.lp"
) -> List[Dict]:
    """
    Ejecuta experimentos baseline sobre múltiples instancias.
    
    Args:
        instance_dir: Directorio con instancias .lp
        output_csv: Archivo CSV para guardar resultados
        threads: Número de hilos
        time_limit: Límite de tiempo por instancia
        mip_gap: Gap objetivo
        log_dir: Directorio para logs
        history_dir: Directorio para guardar historiales JSON
        pattern: Patrón para filtrar archivos
    
    Returns:
        Lista de diccionarios con métricas de cada instancia
    """
    # Encontrar instancias
    instance_files = sorted(glob(os.path.join(instance_dir, pattern)))
    
    if not instance_files:
        print(f"No se encontraron archivos {pattern} en {instance_dir}")
        return []
    
    print(f"\n{'='*60}")
    print(f"EXPERIMENTO BASELINE")
    print(f"{'='*60}")
    print(f"Instancias encontradas: {len(instance_files)}")
    print(f"Configuración: threads={threads}, time_limit={time_limit}s, mip_gap={mip_gap}")
    print(f"{'='*60}")
    
    # Crear experimento
    experiment = GurobiBaselineExperiment(
        threads=threads,
        time_limit=time_limit,
        mip_gap=mip_gap,
        log_dir=log_dir,
        save_history=history_dir is not None
    )
    
    # Ejecutar cada instancia
    all_metrics = []
    for i, instance_path in enumerate(instance_files):
        print(f"\n[{i+1}/{len(instance_files)}] ", end="")
        metrics = experiment.run_instance(instance_path, verbose=True)
        
        # Guardar historial si se especificó directorio
        if history_dir:
            save_history_to_json(metrics, history_dir)
        
        # Remover incumbent_history para el CSV (es muy largo)
        metrics_for_csv = {k: v for k, v in metrics.items() if k != 'incumbent_history'}
        all_metrics.append(metrics_for_csv)
        
        # Guardar resultados incrementalmente
        save_metrics_to_csv(all_metrics, output_csv)
    
    print(f"\n{'='*60}")
    print(f"Experimento BASELINE completado: {len(all_metrics)} instancias")
    print(f"Resultados guardados en: {output_csv}")
    print(f"{'='*60}")
    
    return all_metrics


def main():
    parser = argparse.ArgumentParser(
        description="Ejecutar experimentos BASELINE con Gurobi sobre instancias MIS",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:
  # Ejecutar una sola instancia
  python run_gurobi_baseline.py --instance ../instances/test/mis_50n_000.lp
  
  # Ejecutar todas las instancias en un directorio
  python run_gurobi_baseline.py --instance_dir ../instances/test --output_csv ../results/baseline.csv
  
  # Con límite de tiempo de 1 hora y guardando historial
  python run_gurobi_baseline.py --instance_dir ../instances/test --time_limit 3600 --history_dir ../results/history
        """
    )
    
    # Modo de ejecución
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--instance", type=str,
                       help="Ruta a una sola instancia .lp")
    group.add_argument("--instance_dir", type=str,
                       help="Directorio con múltiples instancias .lp")
    
    # Configuración de Gurobi
    parser.add_argument("--threads", type=int, default=1,
                        help="Número de hilos (default: 1)")
    parser.add_argument("--time_limit", type=float, default=3600,
                        help="Límite de tiempo en segundos (default: 3600)")
    parser.add_argument("--mip_gap", type=float, default=0.0,
                        help="Gap de optimalidad objetivo (default: 0.0)")
    
    # Salida
    parser.add_argument("--output_csv", type=str, default="../results/metrics/baseline.csv",
                        help="Archivo CSV para resultados")
    parser.add_argument("--log_dir", type=str, default=None,
                        help="Directorio para logs de Gurobi")
    parser.add_argument("--history_dir", type=str, default=None,
                        help="Directorio para guardar historial de incumbentes (JSON)")
    
    args = parser.parse_args()
    
    if args.instance:
        # Ejecutar una sola instancia
        experiment = GurobiBaselineExperiment(
            threads=args.threads,
            time_limit=args.time_limit,
            mip_gap=args.mip_gap,
            log_dir=args.log_dir,
            save_history=args.history_dir is not None
        )
        metrics = experiment.run_instance(args.instance, verbose=True)
        
        # Guardar historial si se especificó
        if args.history_dir:
            save_history_to_json(metrics, args.history_dir)
        
        # Guardar resultado
        metrics_for_csv = {k: v for k, v in metrics.items() if k != 'incumbent_history'}
        save_metrics_to_csv([metrics_for_csv], args.output_csv)
        
    else:
        # Ejecutar batch
        run_batch_experiment(
            instance_dir=args.instance_dir,
            output_csv=args.output_csv,
            threads=args.threads,
            time_limit=args.time_limit,
            mip_gap=args.mip_gap,
            log_dir=args.log_dir,
            history_dir=args.history_dir
        )


if __name__ == "__main__":
    main()
