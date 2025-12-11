import pickle
import argparse
from get_bipartite_graph import get_standard_bipartite_graph
import sys
import argparse
import os
import torch
from constants import LITERALS_GRAPH, VARIABLES_GRAPH
def is_backbone_complete(backbone_path: str) -> bool:
    # check if file exists
    if not os.path.exists(backbone_path):
        return False
    with open(backbone_path, 'r') as f:
        #get the last line of the file
        lines = f.readlines()
        if len(lines)==0:
            return False
        last_line = lines[-1].strip()
        if last_line == "b 0":
            return True
    return False
def get_backbone_target(backbone_path: str, v_map:dict):
    # values in target:
    # 0 if variable in backbone with value 0
    # 1 if variable in backbone with value 1
    # 2 if variable not in backbone
    target = [2]*len(v_map) # assume all variables are not in the backbone
    complete_backbone = False
    
    with open(backbone_path, 'r') as f:
        lines = f.readlines()
        len_lines = len(lines)
        #iterate over lines
        for i in range(len_lines):
            line = lines[i]
            line = line.strip()
            parts =  line.split()
            if parts[0] != "b":
                continue
            backbone_found = parts[1]
            if backbone_found.startswith("-"):
                variable_name = backbone_found[1:]
                value= 0 # negative means value 0
            else:
                variable_name = backbone_found
                value = 1 # positive means value 1
            if variable_name == "0" and i == len_lines - 1: #completion signal
                complete_backbone = True
                break
            target[v_map[variable_name]] = value
    if not complete_backbone:
        raise Exception("Backbone not complete")
    target = torch.tensor(target, dtype=torch.long)
    return target
def collect(instance_path:str, backbone_path:str, filename:str ,ml_dataset_path :str, graph_type:str):
    if not filename:
        return

    instance_filepath = os.path.join(instance_path,filename)
    backbone_filepath = os.path.join(backbone_path,filename+".backbone")
    try:
        if not is_backbone_complete(backbone_filepath):
            print(f"Backbone not complete for {filename}, skipping.")
            return

        bipartite_graph = get_standard_bipartite_graph(instance_filepath, graph_type)
        v_map = bipartite_graph[1]  # Get the variable map from the bipartite graph
        
        backbone_target = get_backbone_target(backbone_filepath,v_map)
        data = {
            "bipartite_graph": bipartite_graph,
            "target": backbone_target,
            "graph_type": graph_type,
        }
        pickle.dump(data, open(os.path.join(ml_dataset_path, filename+'.pkl'), 'wb'))
    except Exception as e:
        print(f"Error procesando archivo {filename}: {e}")
    
    




def main():
    parser = argparse.ArgumentParser(description="Create the machine learning dataset from the instance and backbone files. The machine learning datasets contains the input (bipartite graph) and target (backbone) for each instance.")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the dataset directory containing instance and backbone directories.")
    parser.add_argument("--graph_type", type=str, choices=[LITERALS_GRAPH, VARIABLES_GRAPH], default=LITERALS_GRAPH, help="Type of bipartite graph to create.")
    args = parser.parse_args()
    dataset_path = args.dataset_path
    graph_type = args.graph_type
    
    instance_path=os.path.join(dataset_path,"instance")
    backbone_path=os.path.join(dataset_path,"backbone")
    ml_dataset_path=os.path.join(dataset_path,f"ml_dataset_{graph_type}")
    
    os.makedirs(ml_dataset_path,exist_ok=True)
    
    instance_filenames = os.listdir(instance_path)
    backbone_filenames = os.listdir(backbone_path)
    backbone_filenames = [f.replace(".backbone","") for f in backbone_filenames]
    filenames = list(set(instance_filenames).intersection(set(backbone_filenames)))
    
    remaining_filenames = []
    for filename in filenames:
        #Añade los que faltan
        if not os.path.exists(os.path.join(ml_dataset_path,filename+'.pkl')):
            remaining_filenames.append(filename)
    n = len(remaining_filenames)
    for i in range(n):
        print(f"Procesando {i}/{n}: {remaining_filenames[i]}")
        collect(instance_path,backbone_path,remaining_filenames[i],ml_dataset_path,graph_type)
    print(f"Total files processed: {n}")
if __name__ == '__main__':
    main()


