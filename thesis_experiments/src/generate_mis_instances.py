#!/usr/bin/env python3
"""
Generador de instancias del problema Maximum Independent Set (MIS).

Este script genera grafos aleatorios y los convierte a formulación MIP
compatible con Gurobi (.lp format).

El problema MIS se formula como:
    max  sum(x_i)  para todo nodo i
    s.t. x_i + x_j <= 1  para toda arista (i,j)
         x_i in {0,1}

"""

import argparse
import networkx as nx
import random
import os
from pathlib import Path
from typing import Optional


def generate_erdos_renyi_graph(n_nodes: int, edge_probability: float, seed: Optional[int] = None) -> nx.Graph:
    """
    Genera un grafo aleatorio usando el modelo Erdős-Rényi.
    
    Args:
        n_nodes: Número de nodos del grafo
        edge_probability: Probabilidad de que exista una arista entre dos nodos (0-1)
        seed: Semilla para reproducibilidad
    
    Returns:
        Grafo de NetworkX
    """
    if seed is not None:
        random.seed(seed)
    return nx.erdos_renyi_graph(n_nodes, edge_probability, seed=seed)


def generate_barabasi_albert_graph(n_nodes: int, m_edges: int, seed: Optional[int] = None) -> nx.Graph:
    """
    Genera un grafo aleatorio usando el modelo Barabási-Albert (preferential attachment).
    
    Args:
        n_nodes: Número de nodos del grafo
        m_edges: Número de aristas a añadir por cada nuevo nodo
        seed: Semilla para reproducibilidad
    
    Returns:
        Grafo de NetworkX
    """
    return nx.barabasi_albert_graph(n_nodes, m_edges, seed=seed)


def graph_to_mis_lp(graph: nx.Graph, instance_name: str) -> str:
    """
    Convierte un grafo a formulación LP del problema MIS.
    
    Formulación:
        maximize sum(x_i)
        subject to:
            x_i + x_j <= 1  para toda arista (i,j)
            x_i binary
    
    Args:
        graph: Grafo de NetworkX
        instance_name: Nombre de la instancia (para comentarios)
    
    Returns:
        String con el contenido del archivo .lp
    """
    lines = []
    
    # Header con comentarios
    lines.append(f"\\ Maximum Independent Set Instance: {instance_name}")
    lines.append(f"\\ Nodes: {graph.number_of_nodes()}")
    lines.append(f"\\ Edges: {graph.number_of_edges()}")
    lines.append("")
    
    # Función objetivo: maximizar suma de x_i
    lines.append("Maximize")
    obj_terms = [f"x_{i}" for i in graph.nodes()]
    # Dividir en líneas de ~10 términos para legibilidad
    chunk_size = 10
    for i in range(0, len(obj_terms), chunk_size):
        chunk = obj_terms[i:i+chunk_size]
        prefix = "  " if i == 0 else "  + "
        lines.append(prefix + " + ".join(chunk))
    
    lines.append("")
    
    # Restricciones: x_i + x_j <= 1 para cada arista
    lines.append("Subject To")
    for idx, (i, j) in enumerate(graph.edges()):
        lines.append(f"  edge_{idx}: x_{i} + x_{j} <= 1")
    
    lines.append("")
    
    # Variables binarias
    lines.append("Binary")
    for node in graph.nodes():
        lines.append(f"  x_{node}")
    
    lines.append("")
    lines.append("End")
    
    return "\n".join(lines)


def generate_mis_instance(
    n_nodes: int,
    output_path: str,
    graph_type: str = "erdos_renyi",
    edge_probability: float = 0.5,
    ba_edges: int = 4,
    seed: Optional[int] = None
) -> dict:
    """
    Genera una instancia MIS completa y la guarda en formato .lp
    
    Args:
        n_nodes: Número de nodos
        output_path: Ruta donde guardar el archivo .lp
        graph_type: "erdos_renyi" o "barabasi_albert"
        edge_probability: Probabilidad de arista (solo para Erdős-Rényi)
        ba_edges: Aristas por nodo nuevo (solo para Barabási-Albert)
        seed: Semilla para reproducibilidad
    
    Returns:
        Diccionario con estadísticas de la instancia generada
    """
    # Generar grafo
    if graph_type == "erdos_renyi":
        graph = generate_erdos_renyi_graph(n_nodes, edge_probability, seed)
    elif graph_type == "barabasi_albert":
        graph = generate_barabasi_albert_graph(n_nodes, ba_edges, seed)
    else:
        raise ValueError(f"Tipo de grafo no soportado: {graph_type}")
    
    # Extraer nombre de instancia del path
    instance_name = Path(output_path).stem
    
    # Convertir a LP
    lp_content = graph_to_mis_lp(graph, instance_name)
    
    # Crear directorio si no existe
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Guardar archivo
    with open(output_path, 'w') as f:
        f.write(lp_content)
    
    # Retornar estadísticas
    stats = {
        "instance_name": instance_name,
        "n_nodes": graph.number_of_nodes(),
        "n_edges": graph.number_of_edges(),
        "graph_type": graph_type,
        "density": nx.density(graph),
        "output_path": output_path,
        "seed": seed
    }
    
    return stats


def generate_multiple_instances(
    n_instances: int,
    n_nodes: int,
    output_dir: str,
    prefix: str = "instance",
    graph_type: str = "erdos_renyi",
    edge_probability: float = 0.5,
    ba_edges: int = 4,
    base_seed: Optional[int] = None
) -> list:
    """
    Genera múltiples instancias MIS.
    
    Args:
        n_instances: Número de instancias a generar
        n_nodes: Número de nodos por instancia
        output_dir: Directorio de salida
        prefix: Prefijo para nombres de archivo
        graph_type: Tipo de grafo
        edge_probability: Probabilidad de arista (Erdős-Rényi)
        ba_edges: Aristas por nodo (Barabási-Albert)
        base_seed: Semilla base (cada instancia usa base_seed + i)
    
    Returns:
        Lista de diccionarios con estadísticas de cada instancia
    """
    all_stats = []
    
    for i in range(n_instances):
        seed = (base_seed + i) if base_seed is not None else None
        output_path = os.path.join(output_dir, f"{prefix}_{n_nodes}n_{i:03d}.lp")
        
        stats = generate_mis_instance(
            n_nodes=n_nodes,
            output_path=output_path,
            graph_type=graph_type,
            edge_probability=edge_probability,
            ba_edges=ba_edges,
            seed=seed
        )
        
        all_stats.append(stats)
        print(f"[{i+1}/{n_instances}] Generada: {stats['instance_name']} "
              f"(nodos={stats['n_nodes']}, aristas={stats['n_edges']}, "
              f"densidad={stats['density']:.3f})")
    
    return all_stats


def main():
    parser = argparse.ArgumentParser(
        description="Generador de instancias MIS para experimentos de tesis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:
  # Generar 10 instancias pequeñas para pruebas
  python generate_mis_instances.py --n_instances 10 --n_nodes 100 --output_dir ../instances/baseline/small
  
  # Generar 5 instancias medianas
  python generate_mis_instances.py --n_instances 5 --n_nodes 500 --output_dir ../instances/baseline/medium
  
  # Generar 20 instancias grandes para experimentos finales
  python generate_mis_instances.py --n_instances 20 --n_nodes 2000 --output_dir ../instances/baseline/large
  
  # Usar modelo Barabási-Albert en vez de Erdős-Rényi
  python generate_mis_instances.py --n_instances 10 --n_nodes 500 --graph_type barabasi_albert --ba_edges 5
        """
    )
    
    parser.add_argument("--n_instances", type=int, default=10,
                        help="Número de instancias a generar (default: 10)")
    parser.add_argument("--n_nodes", type=int, default=500,
                        help="Número de nodos por instancia (default: 500)")
    parser.add_argument("--output_dir", type=str, default="../instances/baseline",
                        help="Directorio de salida (default: ../instances/baseline)")
    parser.add_argument("--prefix", type=str, default="mis",
                        help="Prefijo para nombres de archivo (default: mis)")
    parser.add_argument("--graph_type", type=str, default="erdos_renyi",
                        choices=["erdos_renyi", "barabasi_albert"],
                        help="Tipo de grafo aleatorio (default: erdos_renyi)")
    parser.add_argument("--edge_prob", type=float, default=0.5,
                        help="Probabilidad de arista para Erdős-Rényi (default: 0.5)")
    parser.add_argument("--ba_edges", type=int, default=4,
                        help="Aristas por nodo para Barabási-Albert (default: 4)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Semilla base para reproducibilidad (default: None)")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("GENERADOR DE INSTANCIAS MIS")
    print("=" * 60)
    print(f"Configuración:")
    print(f"  - Instancias: {args.n_instances}")
    print(f"  - Nodos: {args.n_nodes}")
    print(f"  - Tipo de grafo: {args.graph_type}")
    if args.graph_type == "erdos_renyi":
        print(f"  - Probabilidad de arista: {args.edge_prob}")
    else:
        print(f"  - Aristas por nodo: {args.ba_edges}")
    print(f"  - Directorio salida: {args.output_dir}")
    print(f"  - Semilla: {args.seed}")
    print("=" * 60)
    
    stats = generate_multiple_instances(
        n_instances=args.n_instances,
        n_nodes=args.n_nodes,
        output_dir=args.output_dir,
        prefix=args.prefix,
        graph_type=args.graph_type,
        edge_probability=args.edge_prob,
        ba_edges=args.ba_edges,
        base_seed=args.seed
    )
    
    print("=" * 60)
    print(f"Generación completada: {len(stats)} instancias")
    print(f"Guardadas en: {args.output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
