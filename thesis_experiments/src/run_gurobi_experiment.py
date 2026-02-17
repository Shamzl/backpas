#!/usr/bin/env python3
"""
Script para ejecutar experimentos con Gurobi sobre instancias MIS.

Este script ejecuta Gurobi sobre instancias MIS y recopila métricas como:
- Tiempo de ejecución
- Valor óptimo
- Gap de optimalidad
- Nodos explorados
- Primal integral (aproximado)

Modos de ejecución:
- BASELINE: Gurobi sin ayuda (por defecto)
- BACKPAS: Gurobi + Trust Region en dos fases (--backpas)
    - Fase 1: Con trust region (tiempo limitado por --trust_region_time)
    - Fase 2: Sin trust region + warmstart (tiempo restante)

"""

import argparse
import gurobipy as gp
from gurobipy import GRB
import os
import csv
import time
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List, Tuple
import json
import math

# Agregar src/ al path para importar módulos BACKPAS
REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

# Importaciones opcionales para modo BACKPAS
BACKPAS_AVAILABLE = False
try:
    import torch
    import torch.nn.functional as F
    from GCN import BackbonePredictor, GraphBackboneDataset
    from constants import LITERALS_GRAPH, VARIABLES_GRAPH
    from get_bipartite_graph import get_standard_bipartite_graph
    BACKPAS_AVAILABLE = True
except ImportError as e:
    pass  # BACKPAS no disponible, solo modo baseline


class GurobiMISExperiment:
    """Clase para ejecutar y monitorear experimentos MIS con Gurobi."""

    def __init__(
        self,
        threads: int = 1,
        time_limit: float = 3600,
        mip_gap: float = 0.0,
        log_dir: Optional[str] = None,
        # Parámetros BACKPAS
        use_backpas: bool = False,
        model_path: Optional[str] = None,
        trust_region_time: float = 300,
        threshold: float = 0.7,
        alpha: float = 0.0,
        graph_type: str = "literals",
        num_layers: int = 8,
        layer_type: str = "GTR"
    ):
        """
        Inicializa la configuración del experimento.

        Args:
            threads: Número de hilos (default: 1)
            time_limit: Límite de tiempo en segundos (default: 3600 = 1 hora)
            mip_gap: Gap de optimalidad objetivo (default: 0.0 = óptimo exacto)
            log_dir: Directorio para logs de Gurobi
            use_backpas: Si usar modo BACKPAS (default: False)
            model_path: Ruta al modelo .pth (requerido si use_backpas=True)
            trust_region_time: Tiempo en segundos para Fase 1 con trust region (default: 300)
            threshold: Umbral θ para selección de variables (default: 0.7)
            alpha: Parámetro α para tolerancia (default: 0.0)
            graph_type: Tipo de grafo ('literals' o 'variables')
            num_layers: Número de capas GNN (default: 8)
            layer_type: Tipo de capa GNN (default: 'GTR')
        """
        self.threads = threads
        self.time_limit = time_limit
        self.mip_gap = mip_gap
        self.log_dir = log_dir

        # Parámetros BACKPAS
        self.use_backpas = use_backpas
        self.model_path = model_path
        self.trust_region_time = trust_region_time
        self.threshold = threshold
        self.alpha = alpha
        self.graph_type = graph_type
        self.num_layers = num_layers
        self.layer_type = layer_type

        # Validación
        if self.use_backpas:
            if not BACKPAS_AVAILABLE:
                raise RuntimeError("Modo BACKPAS requiere PyTorch y módulos BACKPAS instalados")
            if not self.model_path:
                raise ValueError("model_path es requerido cuando use_backpas=True")
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Modelo no encontrado: {self.model_path}")

        # Para tracking del primal integral
        self.incumbent_history = []

    def _load_backpas_model(self):
        """Carga el modelo BACKPAS para predicciones."""
        if not BACKPAS_AVAILABLE:
            raise RuntimeError("BACKPAS no disponible")

        device = "cpu"
        graph_type_const = LITERALS_GRAPH if self.graph_type == "literals" else VARIABLES_GRAPH

        model = BackbonePredictor(
            graph_type=graph_type_const,
            num_layers=self.num_layers,
            layer_type=self.layer_type,
            use_literals_message=False
        ).to(device)

        model.load_state_dict(torch.load(self.model_path, map_location=device))
        model.eval()

        return model

    def _get_backbone_predictions(self, instance_path: str, model) -> Tuple:
        """
        Obtiene predicciones de backbone del modelo GNN.

        Returns:
            (pred_probs, v_map): Probabilidades y mapeo de variables
        """
        device = next(model.parameters()).device
        graph_type_const = LITERALS_GRAPH if self.graph_type == "literals" else VARIABLES_GRAPH

        # Construir grafo bipartito
        A, v_map, l_nodes, c_nodes = get_standard_bipartite_graph(instance_path, graph_type_const)
        constraint_features, edge_indices, edge_features, variable_features = \
            GraphBackboneDataset.get_graph_components(A, l_nodes, c_nodes)

        # Predicción
        with torch.no_grad():
            BD = model(
                constraint_features.float().to(device),
                edge_indices.long().to(device),
                edge_features.float().to(device),
                variable_features.float().to(device),
            )
            pred_probs = F.softmax(BD, dim=1).cpu().squeeze()

        return pred_probs, v_map

    def _compute_trust_region_params(self, pred_probs) -> Tuple:
        """
        Calcula los parámetros de la trust region basados en predicciones.

        Returns:
            (selected_indices, assigned_classes, Delta, k_0, k_1)
        """
        P1 = pred_probs[:, 0]  # Probabilidad de clase 0
        P2 = pred_probs[:, 1]  # Probabilidad de clase 1

        # Seleccionar variables con confianza >= threshold
        selected_mask = (torch.max(P1, P2) > self.threshold)
        selected_indices = torch.nonzero(selected_mask).squeeze(1)

        if selected_indices.numel() == 0:
            raise RuntimeError("No se seleccionaron variables para trust region")

        # Determinar clase asignada
        selected_pred_probs = pred_probs[selected_indices]
        assigned_classes = torch.argmax(selected_pred_probs[:, :2], dim=1)

        # Calcular Delta (tolerancia adaptativa)
        assigned_probabilities = selected_pred_probs[:, :2].max(dim=1)[0]
        expected_errors = (1 - assigned_probabilities).sum().item()

        k_0 = (assigned_classes == 0).sum().item()
        k_1 = (assigned_classes == 1).sum().item()

        if self.alpha <= 0:
            Delta = expected_errors * (1 + self.alpha)
        else:
            Delta = (k_0 + k_1 - expected_errors) * self.alpha + expected_errors
        Delta = math.ceil(Delta)

        return selected_indices, assigned_classes, Delta, k_0, k_1

    def _callback(self, model, where):
        """Callback para capturar el progreso durante la optimización."""
        if where == GRB.Callback.MIP:
            # Capturar incumbent y bound actual
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

    def _add_trust_region_constraints(
        self,
        model,
        selected_indices,
        assigned_classes,
        Delta,
        v_map
    ):
        """
        Agrega restricciones de trust region al modelo Gurobi.

        Args:
            model: Modelo Gurobi
            selected_indices: Índices de variables seleccionadas
            assigned_classes: Clases asignadas (0 o 1) para cada variable
            Delta: Tolerancia máxima de desviaciones
            v_map: Mapeo de nombres de variables a índices
        """
        # Crear mapeo índice -> nombre de variable
        index_to_var_name = {v_map[var_name]: var_name for var_name in v_map}

        # Crear mapeo nombre -> objeto variable de Gurobi
        gurobi_var_map = {var.VarName: var for var in model.getVars()}

        # Variables delta para cada variable seleccionada
        delta_vars = []

        for j, original_var_idx in enumerate(selected_indices):
            tar_var_name = index_to_var_name[original_var_idx.item()]
            delta_var = model.addVar(name=f"delta_{tar_var_name}", vtype=GRB.BINARY)
            delta_vars.append(delta_var)

            predicted_class = assigned_classes[j].item()

            if predicted_class == 0:  # Predicción: x = 0
                model.addConstr(
                    gurobi_var_map[tar_var_name] <= delta_var,
                    name=f"tr_{tar_var_name}_0"
                )
            elif predicted_class == 1:  # Predicción: x = 1
                model.addConstr(
                    1 - gurobi_var_map[tar_var_name] <= delta_var,
                    name=f"tr_{tar_var_name}_1"
                )

        # Restricción de suma de deltas
        if delta_vars:
            model.addConstr(
                sum(delta_vars) <= Delta,
                name="tr_delta_sum"
            )

        model.update()

    def _collect_metrics(
        self,
        model,
        instance_path: str,
        start_time: float,
        end_time: float,
        method: str = 'baseline'
    ) -> Dict:
        """
        Recopila métricas del modelo optimizado.

        Args:
            model: Modelo Gurobi después de optimizar
            instance_path: Ruta a la instancia
            start_time: Tiempo de inicio
            end_time: Tiempo de fin
            method: 'baseline' o 'backpas'

        Returns:
            Diccionario con métricas
        """
        instance_name = Path(instance_path).stem

        metrics = {
            'instance_name': instance_name,
            'instance_path': instance_path,
            'method': method,
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
            'timestamp': datetime.now().isoformat(),
        }

        # Calcular primal integral
        if len(self.incumbent_history) > 0:
            metrics['primal_integral'] = self._compute_primal_integral(metrics['obj_bound'])
        else:
            metrics['primal_integral'] = None

        return metrics

    def run_instance(self, instance_path: str, verbose: bool = True) -> Dict:
        """
        Ejecuta Gurobi sobre una instancia MIS.

        Si use_backpas=True, ejecuta en dos fases:
          - Fase 1: Con trust region (trust_region_time segundos)
          - Fase 2: Sin trust region (tiempo restante), usando warmstart

        Args:
            instance_path: Ruta al archivo .lp
            verbose: Si mostrar progreso

        Returns:
            Diccionario con métricas del experimento
        """
        instance_name = Path(instance_path).stem
        method_name = "BACKPAS" if self.use_backpas else "BASELINE"

        if verbose:
            print(f"\n{'='*60}")
            print(f"[{method_name}] Ejecutando: {instance_name}")
            print(f"{'='*60}")

        # Resetear historial
        self.incumbent_history = []

        if self.use_backpas:
            # ============== MODO BACKPAS: DOS FASES ==============
            metrics = self._run_two_phase(instance_path, verbose)
        else:
            # ============== MODO BASELINE: UNA FASE ==============
            metrics = self._run_single_phase(instance_path, verbose)

        if verbose:
            self._print_metrics(metrics)

        return metrics

    def _run_single_phase(self, instance_path: str, verbose: bool) -> Dict:
        """
        Ejecuta Gurobi sin trust region (modo baseline).

        Args:
            instance_path: Ruta al archivo .lp
            verbose: Si mostrar progreso

        Returns:
            Diccionario con métricas
        """
        instance_name = Path(instance_path).stem
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
        metrics = self._collect_metrics(model, instance_path, start_time, end_time, 'baseline')
        metrics['incumbent_history'] = self.incumbent_history

        return metrics

    def _run_two_phase(self, instance_path: str, verbose: bool) -> Dict:
        """
        Ejecuta BACKPAS en dos fases: con trust region → sin trust region.

        Fase 1: Optimiza con trust region por trust_region_time segundos
        Fase 2: Si no es óptimo, continúa sin trust region usando warmstart

        Args:
            instance_path: Ruta al archivo .lp
            verbose: Si mostrar progreso

        Returns:
            Diccionario con métricas
        """
        instance_name = Path(instance_path).stem
        start_time_total = time.time()

        # ==================== PREPARACIÓN: CARGAR MODELO Y PREDICCIONES ====================
        if verbose:
            print(f"\n[BACKPAS] === PREPARACIÓN ===")
            print(f"[BACKPAS] Cargando modelo GNN...")

        ml_model = self._load_backpas_model()

        if verbose:
            print(f"[BACKPAS] Generando predicciones...")

        pred_probs, v_map = self._get_backbone_predictions(instance_path, ml_model)

        if verbose:
            print(f"[BACKPAS] Calculando parámetros de trust region...")

        selected_indices, assigned_classes, Delta, k_0, k_1 = \
            self._compute_trust_region_params(pred_probs)

        trust_region_info = {
            'k_0': k_0,
            'k_1': k_1,
            'Delta': Delta,
            'n_selected': len(selected_indices),
            'threshold': self.threshold,
            'alpha': self.alpha,
            'trust_region_time': self.trust_region_time
        }

        if verbose:
            print(f"[BACKPAS] Trust region configurada:")
            print(f"  - Variables fijadas a 0: {k_0}")
            print(f"  - Variables fijadas a 1: {k_1}")
            print(f"  - Tolerancia (Δ): {Delta}")

        # ==================== FASE 1: CON TRUST REGION ====================
        if verbose:
            print(f"\n[BACKPAS] === FASE 1: Con Trust Region ===")
            print(f"[BACKPAS] Tiempo límite fase 1: {self.trust_region_time}s")

        # Crear modelo con trust region
        model_phase1 = gp.read(instance_path)
        self._add_trust_region_constraints(
            model_phase1, selected_indices, assigned_classes, Delta, v_map
        )

        # Configurar fase 1
        model_phase1.Params.Threads = self.threads
        model_phase1.Params.TimeLimit = self.trust_region_time
        model_phase1.Params.MIPGap = self.mip_gap

        if self.log_dir:
            os.makedirs(self.log_dir, exist_ok=True)
            log_file = os.path.join(self.log_dir, f"{instance_name}_backpas_phase1.log")
            model_phase1.Params.LogFile = log_file

        if not verbose:
            model_phase1.Params.OutputFlag = 0

        # Ejecutar fase 1
        phase1_start = time.time()
        model_phase1.optimize(self._callback)
        phase1_time = time.time() - phase1_start

        # Guardar solución para warmstart
        warmstart_solution = {}
        phase1_obj = None
        phase1_bound = model_phase1.ObjBound if hasattr(model_phase1, 'ObjBound') else None

        if model_phase1.SolCount > 0:
            phase1_obj = model_phase1.ObjVal
            # Solo guardar variables originales (no las delta)
            for v in model_phase1.getVars():
                if not v.VarName.startswith('delta_'):
                    warmstart_solution[v.VarName] = v.X

        phase1_status = model_phase1.Status
        phase1_nodes = model_phase1.NodeCount

        if verbose:
            print(f"\n[BACKPAS] Fase 1 completada:")
            print(f"  - Tiempo: {phase1_time:.2f}s")
            print(f"  - Estado: {self._status_to_string(phase1_status)}")
            print(f"  - Nodos explorados: {phase1_nodes}")
            if phase1_obj is not None:
                print(f"  - Mejor solución: {phase1_obj}")
                print(f"  - Bound: {phase1_bound}")
                if phase1_bound and phase1_obj:
                    gap = abs(phase1_bound - phase1_obj) / abs(phase1_obj) * 100
                    print(f"  - Gap: {gap:.4f}%")

        # ==================== VERIFICAR TIEMPO PARA FASE 2 ====================
        # IMPORTANTE: Siempre ejecutamos Fase 2 (si hay tiempo), incluso si Fase 1
        # terminó con OPTIMAL. El "óptimo" de Fase 1 es solo el óptimo DENTRO de la
        # trust region, no necesariamente el óptimo GLOBAL. Si las predicciones de
        # la GNN fueron incorrectas, la trust region podría haber excluido el
        # verdadero óptimo.

        # Calcular tiempo restante
        elapsed_time = time.time() - start_time_total
        remaining_time = self.time_limit - elapsed_time

        if remaining_time <= 1:  # Menos de 1 segundo restante
            if verbose:
                print(f"\n[BACKPAS] Sin tiempo restante para Fase 2 ({remaining_time:.2f}s).")
                print(f"[BACKPAS] ADVERTENCIA: No se pudo verificar si el óptimo de Fase 1 es el óptimo global.")

            end_time_total = time.time()
            metrics = self._collect_metrics(
                model_phase1, instance_path, start_time_total, end_time_total, 'backpas'
            )
            metrics.update(trust_region_info)
            metrics['phase1_time'] = phase1_time
            metrics['phase1_status'] = self._status_to_string(phase1_status)
            metrics['phase1_obj'] = phase1_obj
            metrics['phase1_nodes'] = phase1_nodes
            metrics['phase2_time'] = 0
            metrics['phase2_status'] = None
            metrics['phase2_obj'] = None
            metrics['phase2_nodes'] = 0
            metrics['trust_region_removed'] = False
            metrics['warmstart_used'] = False
            metrics['phase2_skipped'] = True  # Indicar que Fase 2 fue omitida
            metrics['incumbent_history'] = self.incumbent_history
            return metrics

        # ==================== FASE 2: SIN TRUST REGION (SIEMPRE SE EJECUTA) ====================
        if verbose:
            print(f"\n[BACKPAS] === FASE 2: Sin Trust Region (verificación de óptimo global) ===")
            print(f"[BACKPAS] Tiempo límite fase 2: {remaining_time:.2f}s")
            if warmstart_solution:
                print(f"[BACKPAS] Usando warmstart con {len(warmstart_solution)} variables")
                if phase1_obj:
                    print(f"[BACKPAS] Solución inicial (warmstart): {phase1_obj}")

        # Guardar historial de fase 1 y resetear para fase 2
        phase1_history = self.incumbent_history.copy()
        self.incumbent_history = []

        # Crear modelo limpio (sin trust region)
        model_phase2 = gp.read(instance_path)

        # Aplicar warmstart
        if warmstart_solution:
            for v in model_phase2.getVars():
                if v.VarName in warmstart_solution:
                    v.Start = warmstart_solution[v.VarName]

        # Configurar fase 2
        model_phase2.Params.Threads = self.threads
        model_phase2.Params.TimeLimit = remaining_time
        model_phase2.Params.MIPGap = self.mip_gap

        if self.log_dir:
            log_file = os.path.join(self.log_dir, f"{instance_name}_backpas_phase2.log")
            model_phase2.Params.LogFile = log_file

        if not verbose:
            model_phase2.Params.OutputFlag = 0

        # Ejecutar fase 2
        phase2_start = time.time()
        model_phase2.optimize(self._callback)
        phase2_time = time.time() - phase2_start
        end_time_total = time.time()

        phase2_status = model_phase2.Status
        phase2_obj = model_phase2.ObjVal if model_phase2.SolCount > 0 else None
        phase2_nodes = model_phase2.NodeCount

        if verbose:
            print(f"\n[BACKPAS] Fase 2 completada:")
            print(f"  - Tiempo: {phase2_time:.2f}s")
            print(f"  - Estado: {self._status_to_string(phase2_status)}")
            print(f"  - Nodos explorados: {phase2_nodes}")
            if phase2_obj is not None:
                print(f"  - Mejor solución: {phase2_obj}")

        # Combinar historiales ajustando tiempos de fase 2
        # phase1_history tiene entries con tiempos relativos a fase 1
        # self.incumbent_history tiene entries de fase 2 con tiempos relativos a fase 2
        elapsed_before_phase2 = time.time() - start_time_total - phase2_time
        combined_history = phase1_history.copy()
        for entry in self.incumbent_history:
            combined_history.append({
                'time': elapsed_before_phase2 + entry['time'],
                'incumbent': entry['incumbent'],
                'bound': entry['bound']
            })
        self.incumbent_history = combined_history

        # Recopilar métricas finales (del modelo que tenga mejor resultado)
        # Usamos phase2 ya que tiene el espacio completo
        metrics = self._collect_metrics(
            model_phase2, instance_path, start_time_total, end_time_total, 'backpas'
        )

        # Actualizar con la mejor solución encontrada
        if phase1_obj is not None and phase2_obj is not None:
            metrics['obj_val'] = max(phase1_obj, phase2_obj)  # MIS es maximización
        elif phase1_obj is not None:
            metrics['obj_val'] = phase1_obj

        # Agregar información adicional
        metrics.update(trust_region_info)
        metrics['phase1_time'] = phase1_time
        metrics['phase1_status'] = self._status_to_string(phase1_status)
        metrics['phase1_obj'] = phase1_obj
        metrics['phase1_nodes'] = phase1_nodes
        metrics['phase2_time'] = phase2_time
        metrics['phase2_status'] = self._status_to_string(phase2_status)
        metrics['phase2_obj'] = phase2_obj
        metrics['phase2_nodes'] = phase2_nodes
        metrics['trust_region_removed'] = True
        metrics['warmstart_used'] = len(warmstart_solution) > 0
        metrics['total_nodes'] = phase1_nodes + phase2_nodes
        metrics['phase2_skipped'] = False
        metrics['phase2_improved'] = phase2_obj is not None and phase1_obj is not None and phase2_obj > phase1_obj
        metrics['incumbent_history'] = self.incumbent_history

        return metrics

    def _compute_primal_integral(self, best_known: float) -> float:
        """
        Calcula el primal integral aproximado.

        El primal integral mide el área entre la curva del incumbent
        y el valor óptimo a lo largo del tiempo.

        Args:
            best_known: Mejor valor conocido (bound)

        Returns:
            Valor del primal integral
        """
        if len(self.incumbent_history) < 2:
            return 0.0

        integral = 0.0
        for i in range(1, len(self.incumbent_history)):
            t_prev = self.incumbent_history[i-1]['time']
            t_curr = self.incumbent_history[i]['time']
            inc_prev = self.incumbent_history[i-1]['incumbent']

            if inc_prev != GRB.INFINITY and best_known != 0:
                # Gap normalizado
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
        print(f"\n{'='*60}")
        print(f"RESULTADOS FINALES")
        print(f"{'='*60}")
        print(f"  Estado: {metrics['status_name']}")
        print(f"  Tiempo total: {metrics['runtime']:.2f} seg")
        print(f"  Valor objetivo: {metrics['obj_val']}")
        print(f"  Bound: {metrics['obj_bound']}")
        print(f"  Gap: {metrics['mip_gap']*100:.4f}%")
        print(f"  Nodos explorados: {metrics['n_nodes']}")
        print(f"  Soluciones encontradas: {metrics['n_solutions']}")

        if metrics.get('primal_integral') is not None:
            print(f"  Primal integral: {metrics['primal_integral']:.4f}")

        # Información adicional para BACKPAS
        if metrics.get('method') == 'backpas':
            print(f"\n  --- Detalles BACKPAS ---")
            print(f"  Fase 1 (con TR): {metrics.get('phase1_time', 0):.2f}s, "
                  f"estado={metrics.get('phase1_status')}, "
                  f"obj={metrics.get('phase1_obj')}")
            if metrics.get('phase2_skipped'):
                print(f"  Fase 2: OMITIDA (sin tiempo restante)")
                print(f"  ADVERTENCIA: El resultado puede no ser el óptimo global")
            elif metrics.get('phase2_time', 0) > 0:
                print(f"  Fase 2 (sin TR): {metrics.get('phase2_time', 0):.2f}s, "
                      f"estado={metrics.get('phase2_status')}, "
                      f"obj={metrics.get('phase2_obj')}")
                if metrics.get('phase2_improved'):
                    print(f"  ** Fase 2 MEJORÓ el resultado de Fase 1 **")
                    print(f"     (Trust region había excluido el óptimo global)")
                else:
                    print(f"  Fase 2 confirmó el resultado de Fase 1")
            print(f"  Trust region removida: {metrics.get('trust_region_removed')}")
            print(f"  Warmstart usado: {metrics.get('warmstart_used')}")


def run_batch_experiment(
    instance_dir: str,
    output_csv: str,
    threads: int = 1,
    time_limit: float = 3600,
    mip_gap: float = 0.0,
    log_dir: Optional[str] = None,
    pattern: str = "*.lp",
    use_backpas: bool = False,
    model_path: Optional[str] = None,
    trust_region_time: float = 300,
    threshold: float = 0.7,
    alpha: float = 0.0,
    graph_type: str = "literals",
    num_layers: int = 8,
    layer_type: str = "GTR"
) -> List[Dict]:
    """
    Ejecuta experimentos sobre múltiples instancias.

    Args:
        instance_dir: Directorio con instancias .lp
        output_csv: Archivo CSV para guardar resultados
        threads: Número de hilos
        time_limit: Límite de tiempo por instancia
        mip_gap: Gap objetivo
        log_dir: Directorio para logs
        pattern: Patrón para filtrar archivos
        use_backpas: Si usar modo BACKPAS
        model_path: Ruta al modelo .pth
        trust_region_time: Tiempo para Fase 1 con trust region
        threshold: Umbral θ
        alpha: Parámetro α
        graph_type: Tipo de grafo
        num_layers: Capas GNN
        layer_type: Tipo de capa GNN

    Returns:
        Lista de diccionarios con métricas de cada instancia
    """
    from glob import glob

    # Encontrar instancias (.lp y .mps)
    if pattern == "*.lp":
        instance_files = sorted(
            glob(os.path.join(instance_dir, "*.lp")) +
            glob(os.path.join(instance_dir, "*.mps"))
        )
    else:
        instance_files = sorted(glob(os.path.join(instance_dir, pattern)))

    if not instance_files:
        print(f"No se encontraron archivos {pattern} en {instance_dir}")
        return []

    method_name = "BACKPAS" if use_backpas else "BASELINE"
    print(f"\n{'='*60}")
    print(f"EXPERIMENTO {method_name}")
    print(f"{'='*60}")
    print(f"Encontradas {len(instance_files)} instancias")
    print(f"Configuración: threads={threads}, time_limit={time_limit}s, mip_gap={mip_gap}")
    if use_backpas:
        print(f"BACKPAS: trust_region_time={trust_region_time}s, threshold={threshold}, alpha={alpha}")
        print(f"BACKPAS: Modo dos fases (Fase 1: con TR, Fase 2: sin TR + warmstart)")
    print(f"{'='*60}")

    # Crear experimento
    experiment = GurobiMISExperiment(
        threads=threads,
        time_limit=time_limit,
        mip_gap=mip_gap,
        log_dir=log_dir,
        use_backpas=use_backpas,
        model_path=model_path,
        trust_region_time=trust_region_time,
        threshold=threshold,
        alpha=alpha,
        graph_type=graph_type,
        num_layers=num_layers,
        layer_type=layer_type
    )

    # Ejecutar cada instancia
    all_metrics = []
    for i, instance_path in enumerate(instance_files):
        print(f"\n[{i+1}/{len(instance_files)}] ", end="")
        metrics = experiment.run_instance(instance_path, verbose=True)

        # Remover incumbent_history para el CSV (es muy largo)
        metrics_for_csv = {k: v for k, v in metrics.items() if k != 'incumbent_history'}
        all_metrics.append(metrics_for_csv)

        # Guardar resultados incrementalmente
        save_metrics_to_csv(all_metrics, output_csv)

    print(f"\n{'='*60}")
    print(f"Experimento completado: {len(all_metrics)} instancias")
    print(f"Resultados guardados en: {output_csv}")
    print(f"{'='*60}")

    return all_metrics


def save_metrics_to_csv(metrics_list: List[Dict], output_path: str):
    """Guarda métricas en archivo CSV."""
    if not metrics_list:
        return

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)

    fieldnames = [
        'instance_name', 'method', 'status_name', 'runtime', 'gurobi_runtime',
        'obj_val', 'obj_bound', 'mip_gap', 'n_nodes', 'n_solutions',
        'n_vars', 'n_constrs', 'primal_integral', 'threads', 'time_limit',
        # Campos BACKPAS
        'k_0', 'k_1', 'Delta', 'n_selected', 'threshold', 'alpha', 'trust_region_time',
        'phase1_time', 'phase1_status', 'phase1_obj', 'phase1_nodes',
        'phase2_time', 'phase2_status', 'phase2_obj', 'phase2_nodes',
        'total_nodes', 'trust_region_removed', 'warmstart_used',
        'phase2_skipped', 'phase2_improved',
        'timestamp'
    ]

    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(metrics_list)


def main():
    parser = argparse.ArgumentParser(
        description="Ejecutar experimentos Gurobi sobre instancias MIS (BASELINE o BACKPAS)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:
  # BASELINE - Ejecutar una sola instancia sin ayuda
  python run_gurobi_experiment.py --instance ../instances/test/mis_50n_000.lp

  # BASELINE - Ejecutar todas las instancias en un directorio
  python run_gurobi_experiment.py --instance_dir ../instances/test --output_csv ../results/baseline.csv

  # BACKPAS - Con trust region en dos fases
  # Fase 1: 300s con trust region
  # Fase 2: tiempo restante sin trust region + warmstart
  python run_gurobi_experiment.py \\
      --instance ../instances/test/mis_50n_000.lp \\
      --backpas \\
      --model_path ../wkdir/MIS/ml_training/graph_with_literals_8_GTR/best_model.pth \\
      --trust_region_time 300 \\
      --threshold 0.7 \\
      --alpha 0.0 \\
      --output_csv ../results/backpas.csv

  # BACKPAS - Experimento con diferentes tiempos de trust region
  python run_gurobi_experiment.py --instance_dir ../instances/test --backpas \\
      --model_path ../wkdir/MIS/ml_training/graph_with_literals_8_GTR/best_model.pth \\
      --trust_region_time 60 --output_csv ../results/backpas_60s.csv
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

    # Modo BACKPAS
    parser.add_argument("--backpas", action="store_true",
                        help="Usar modo BACKPAS con trust region en dos fases")
    parser.add_argument("--model_path", type=str,
                        help="Ruta al modelo .pth (requerido si --backpas)")
    parser.add_argument("--trust_region_time", type=float, default=300,
                        help="Tiempo (segundos) para Fase 1 con trust region (default: 300)")
    parser.add_argument("--threshold", type=float, default=0.7,
                        help="Umbral θ para selección de variables (default: 0.7)")
    parser.add_argument("--alpha", type=float, default=0.0,
                        help="Parámetro α para tolerancia adaptativa (default: 0.0)")
    parser.add_argument("--graph_type", type=str, default="literals",
                        choices=["literals", "variables"],
                        help="Tipo de grafo para GNN (default: literals)")
    parser.add_argument("--num_layers", type=int, default=8,
                        help="Número de capas GNN (default: 8)")
    parser.add_argument("--layer_type", type=str, default="GTR",
                        help="Tipo de capa GNN (default: GTR)")

    # Salida
    parser.add_argument("--output_csv", type=str, default="../results/metrics/results.csv",
                        help="Archivo CSV para resultados")
    parser.add_argument("--log_dir", type=str, default=None,
                        help="Directorio para logs de Gurobi")

    args = parser.parse_args()

    if args.instance:
        # Ejecutar una sola instancia
        experiment = GurobiMISExperiment(
            threads=args.threads,
            time_limit=args.time_limit,
            mip_gap=args.mip_gap,
            log_dir=args.log_dir,
            use_backpas=args.backpas,
            model_path=args.model_path,
            trust_region_time=args.trust_region_time,
            threshold=args.threshold,
            alpha=args.alpha,
            graph_type=args.graph_type,
            num_layers=args.num_layers,
            layer_type=args.layer_type
        )
        metrics = experiment.run_instance(args.instance, verbose=True)

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
            use_backpas=args.backpas,
            model_path=args.model_path,
            trust_region_time=args.trust_region_time,
            threshold=args.threshold,
            alpha=args.alpha,
            graph_type=args.graph_type,
            num_layers=args.num_layers,
            layer_type=args.layer_type
        )


if __name__ == "__main__":
    main()
