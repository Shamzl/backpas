from abc import ABC, abstractmethod
from GCN import GraphBackboneDataset
from constants import LITERALS_GRAPH, VARIABLES_GRAPH
from get_bipartite_graph import get_standard_bipartite_graph
import torch
import torch.nn.functional as F
import pyscipopt as scp
import math
import time
class TrustRegionNotCreatedException(Exception):
    pass
import os
import time
import shutil
import multiprocessing
from pathlib import Path

def _run_process_instance(obj, instance_input_path, instance_output_path, q):
    """Helper for multiprocessing that runs the original logic and communicates success/failure."""
    try:
        obj._process_instance_impl(instance_input_path, instance_output_path)
        q.put("success")
    except Exception as e:
        q.put(("error", str(e)))
class TrustRegionConstructor(ABC):
    def __init__(self, ml_model, graph_type:str, log_file:str):
        assert graph_type in [LITERALS_GRAPH, VARIABLES_GRAPH], f"Unknown graph type requested: {graph_type}. Choose '{LITERALS_GRAPH}' or '{VARIABLES_GRAPH}'."
        self.ml_model = ml_model
        self.device = next(ml_model.parameters()).device if hasattr(ml_model, 'parameters') else 'cpu'
        self.graph_type = graph_type
        self.log_file = log_file

    @abstractmethod
    def compute_constraints(self, instance_input_path:str, pred_probs, v_map) -> tuple:
        pass

    @abstractmethod
    def params_to_string(self) -> str:
        """Return a string representation of the parameters used for the trust region construction."""
        pass
    def add_log(self, message:str):
        """Utility method to write messages to the log file."""
        with open(self.log_file, 'a') as f:
            f.write(message + '\n')
    
    def process_instance(self, instance_input_path: str, instance_output_path: str, timelimit: int = None):
        if timelimit is None:
            self._process_instance_impl(instance_input_path, instance_output_path)
            return

        q = multiprocessing.Queue()
        p = multiprocessing.Process(
            target=_run_process_instance,
            args=(self, instance_input_path, instance_output_path, q)
        )
        p.start()
        p.join(timeout=timelimit)

        if p.is_alive():
            p.terminate()
            p.join()
            self.add_log(f"Timeout: process_instance exceeded {timelimit} seconds for {instance_input_path}")
            try:
                if instance_output_path.exists():
                    instance_output_path.unlink()
                    self.add_log(f"Partial file {instance_output_path} removed due to timeout.")
            except Exception as e:
                self.add_log(f"Error deleting partial file: {e}")
        else:
            if not q.empty():
                result = q.get()
                if isinstance(result, tuple) and result[0] == "error":
                    self.add_log(f"Error in instance processing: {result[1]}")

    def _process_instance_impl(self, instance_input_path:str, instance_output_path:str):
        self.add_log(f"Processing instance {instance_input_path} to create trust region. Parameters: {self.params_to_string()}")
        pred,v_map = self.get_predictions_and_vmap(instance_input_path)
        try:
            begin = time.time()
            constraints, scp_model, k_0, k_1, Delta = self.compute_constraints(instance_input_path, pred, v_map)
            #write to log file if provided
            self.add_log(f"Constraints computed in {time.time() - begin:.2f} seconds for instance {instance_input_path}.")
            begin = time.time()
            for c in constraints:
                scp_model.addCons(c)
            scp_model.writeProblem(instance_output_path)
            self.add_log(f"New instance written to {instance_output_path} in {time.time() - begin:.2f} seconds.")
            self.add_log(f"Trust region created for instance {instance_input_path} with {len(constraints)} constraints, k_0={k_0}, k_1={k_1}, Delta={Delta}.")
        except TrustRegionNotCreatedException as e:
            self.add_log(f"Trust region not created for instance {instance_input_path}: {e}")
        

    def get_predictions_and_vmap(self, instance_input:str):
        begin = time.time()
        A, v_map, l_nodes, c_nodes = get_standard_bipartite_graph(instance_input, self.graph_type)
        constraint_features, edge_indices, edge_features, variable_features = GraphBackboneDataset.get_graph_components(A, l_nodes, c_nodes)
        self.add_log(f"Bipartite graph constructed and loaded in {time.time() - begin:.2f} seconds for instance {instance_input}.")
        #prediction
        begin = time.time()
        with torch.no_grad():
            BD = self.ml_model(
                constraint_features.float().to(self.device),
                edge_indices.long().to(self.device),
                edge_features.float().to(self.device),
                variable_features.float().to(self.device),
            )
            pred = F.softmax(BD,dim=1).cpu().squeeze()
        
        self.add_log(f"Predictions made in {time.time() - begin:.2f} seconds for instance {instance_input}.")
        return pred, v_map


class ThresholdedExpectedErrorTrustRegionConstructor(TrustRegionConstructor):
    def __init__(self, ml_model, graph_type:str, threshold:float, alpha:float, log_file:str):
        super().__init__(ml_model, graph_type, log_file)
        self.threshold = threshold
        self.alpha = alpha
        if not (-1 < alpha < 1):
            raise ValueError(f"Alpha must be in the range (-1, 1), but got {alpha}.")
        if not (0 < threshold < 1):
            raise ValueError(f"Threshold must be in the range (0, 1), but got {threshold}.")
    def params_to_string(self) -> str:
        """Return a string representation of the parameters used for the trust region construction."""
        return f"Threshold: {self.threshold}, Alpha: {self.alpha}"
    def compute_constraints(self, instance_input_path:str, pred_probs, v_map) -> tuple:
        N = len(v_map) # Total number of nodes (variables) in the instance
        # P1, P2, P3 correspond to probabilities of class 0, 1, 2 respectively.
        P1 = pred_probs[:, 0] # Probability of class 0
        P2 = pred_probs[:, 1] # Probability of class 1
        P3 = pred_probs[:, 2] # Probability of class 2


        selected_mask = (torch.max(P1, P2) > self.threshold)
        selected_indices = torch.nonzero(selected_mask).squeeze(1) # Get the actual indices

        # Handle case where no nodes are selected
        if selected_indices.numel() == 0:
            raise TrustRegionNotCreatedException(f"No nodes selected for prediction.")


        # Filter predictions and true labels for selected nodes
        selected_pred_probs = pred_probs[selected_indices] # Only probabilities of selected nodes

        # --- Determine Assigned Class for Selected Nodes ---
        # This is argmax over P0 and P1 for selected nodes
        # assigned_classes will be 0 if P0 > P1, and 1 if P1 >= P0
        assigned_classes = torch.argmax(selected_pred_probs[:, :2], dim=1) 
        
        # --- Count fixed 0s and 1s for printing ---
        # `assigned_classes` contains 0 for P0-assigned, 1 for P1-assigned
        k_0 = (assigned_classes == 0).sum().item()
        k_1 = (assigned_classes == 1).sum().item()

        # --- New Expected Number of Errors (Delta) Calculation ---
        # Get the probability of the *assigned* class (which is the maximum of P0 and P1)
        assigned_probabilities = selected_pred_probs[:, :2].max(dim=1)[0]

        # Delta is the total expected number of errors among selected nodes
        expected_errrors = (1 - assigned_probabilities).sum().item()
        if self.alpha <= 0:
            Delta = expected_errrors * (1 + self.alpha)
        else:
            Delta = (k_0 + k_1 - expected_errrors) * self.alpha + expected_errrors
        Delta = math.ceil(Delta)

        if Delta >= (k_0 + k_1):
            raise TrustRegionNotCreatedException(f"Delta ({Delta}) is greater than or equal to the number of fixed variables ({k_0 + k_1}).")
        # --- SCIP Model Setup ---
        index_to_var_name = {v_map[var_name]: var_name for var_name in v_map}
        scp_model = scp.Model()
        scp_model.hideOutput()
        scp_model.readProblem(instance_input_path)
        scp_var_map = {}  # Map to store original variables
        
        for var in scp_model.getVars():
            scp_var_map[var.name] = var
        constraints = []
        deltas = [] # List to hold the delta variables for the sum constraint

        # Iterate over indices of selected nodes
        for j, original_var_idx in enumerate(selected_indices):
            tar_var_name = index_to_var_name[original_var_idx.item()]
            delta_var = scp_model.addVar(name=f"delta_{tar_var_name}", vtype="B")
            
            predicted_class = assigned_classes[j].item() # The predicted class for this variable (0 or 1)
            
            if predicted_class == 0: # Predicted value is 0
                constraints.append(scp_var_map[tar_var_name] <= delta_var)
            elif predicted_class == 1: # Predicted value is 1
                constraints.append(1 - scp_var_map[tar_var_name] <= delta_var)
            else:
                # This case should ideally not happen if assigned_classes is only 0 or 1
                raise Exception(f"Invalid predicted class encountered: {predicted_class}")
            deltas.append(delta_var)

        if len(deltas) > 0:
            constraints.append(sum(deltas) <= Delta)
        
        return constraints, scp_model, k_0, k_1, Delta
class ThresholdedWeightedBudgetTrustRegionConstructor(TrustRegionConstructor):
    def __init__(self, ml_model, graph_type:str, threshold:float, budget:float, log_file:str):
        super().__init__(ml_model, graph_type, log_file)
        self.threshold = threshold
        self.budget = budget
        if not (0 < threshold < 1):
            raise ValueError(f"Threshold must be in the range (0, 1), but got {threshold}.")
        if not (0 < budget < 1):
            raise ValueError(f"Budget must be in the range (0, 1), but got {budget}.")
    def params_to_string(self) -> str:
        """Return a string representation of the parameters used for the trust region construction."""
        return f"Threshold: {self.threshold}, Budget: {self.budget}"
    def compute_constraints(self, instance_input_path:str, pred_probs, v_map) -> tuple:
        N = len(v_map) # Total number of nodes (variables) in the instance
        # --- New Selection Criteria (MAX criterion) ---
        # P1, P2, P3 correspond to probabilities of class 0, 1, 2 respectively.
        P1 = pred_probs[:, 0] # Probability of class 0
        P2 = pred_probs[:, 1] # Probability of class 1
        P3 = pred_probs[:, 2] # Probability of class 2
        selected_mask = (torch.max(P1, P2) > self.threshold)
        selected_indices = torch.nonzero(selected_mask).squeeze(1) # Get the actual indices

        # Handle case where no nodes are selected
        if selected_indices.numel() == 0:
            raise TrustRegionNotCreatedException(f"No nodes selected for prediction.")


        # Filter predictions and true labels for selected nodes
        selected_pred_probs = pred_probs[selected_indices] # Only probabilities of selected nodes

        # --- Determine Assigned Class for Selected Nodes ---
        # This is argmax over P0 and P1 for selected nodes
        # assigned_classes will be 0 if P0 > P1, and 1 if P1 >= P0
        assigned_classes = torch.argmax(selected_pred_probs[:, :2], dim=1) 

        # --- Delta calculation as a ratio of the expected error (Delta) Calculation ---
        # Get the probability of the *assigned* class (which is the maximum of P0 and P1)
        assigned_probabilities = selected_pred_probs[:, :2].max(dim=1)[0]
        weights = 4**(1 - assigned_probabilities)  # Weighting factor based on the probability of error
        Delta = weights.sum().item() * self.budget

        # --- Count fixed 0s and 1s for printing ---
        # `assigned_classes` contains 0 for P0-assigned, 1 for P1-assigned
        k_0 = (assigned_classes == 0).sum().item()
        k_1 = (assigned_classes == 1).sum().item()

        # --- SCIP Model Setup ---
        index_to_var_name = {v_map[var_name]: var_name for var_name in v_map}
        scp_model = scp.Model()
        scp_model.hideOutput()
        scp_model.readProblem(instance_input_path)
        scp_var_map = {}  # Map to store original variables
        
        for var in scp_model.getVars():
            scp_var_map[var.name] = var
            
        constraints = []
        deltas = [] # List to hold the delta variables for the sum constraint
        # Iterate over indices of selected nodes
        for j, original_var_idx in enumerate(selected_indices):
            tar_var_name = index_to_var_name[original_var_idx.item()]
            delta_var = scp_model.addVar(name=f"delta_{tar_var_name}", vtype="B")
            
            predicted_class = assigned_classes[j].item() # The predicted class for this variable (0 or 1)
            
            if predicted_class == 0: # Predicted value is 0
                constraints.append(scp_var_map[tar_var_name] <= delta_var)
            elif predicted_class == 1: # Predicted value is 1
                constraints.append(1 - scp_var_map[tar_var_name] <= delta_var)
            else:
                # This case should ideally not happen if assigned_classes is only 0 or 1
                raise Exception(f"Invalid predicted class encountered: {predicted_class}")
            
            deltas.append(delta_var* weights[j].item())  # Scale delta by the weight for this variable
            
        if len(deltas) > 0:
            constraints.append(sum(deltas) <= Delta)
        
        return constraints, scp_model, k_0, k_1, Delta
    
class FixedTwoRatiosTrustRegionConstructor(TrustRegionConstructor):
    def __init__(self, ml_model, graph_type:str, k_ratio:float, Delta_ratio:float, log_file:str):
        super().__init__(ml_model, graph_type, log_file)
        self.k_ratio = k_ratio
        self.Delta_ratio = Delta_ratio
        if not (0 < k_ratio < 1):
            raise ValueError(f"k_ratio must be in the range (0, 1), but got {k_ratio}.")
        if not (0 < Delta_ratio < 1):
            raise ValueError(f"Delta_ratio must be in the range (0, 1), but got {Delta_ratio}.")
        
    def params_to_string(self) -> str:
        """Return a string representation of the parameters used for the trust region construction."""
        return f"k_ratio: {self.k_ratio}, Delta_ratio: {self.Delta_ratio}"
    
    def compute_constraints(self, instance_input_path:str, pred_probs, v_map) -> tuple:
        N = len(v_map)
        k = int(N*self.k_ratio)
        Delta = int(N*self.k_ratio*self.Delta_ratio)
        selected = torch.topk(pred_probs[:,:2].max(dim=1)[0],k)[1]
        pred_probs[:,1] = pred_probs[:,1]>=pred_probs[:,0]
        pred_probs[:,0] = pred_probs[:,1]<pred_probs[:,0]

        k_0 = pred_probs[:,0][selected].sum().item()
        k_1 = pred_probs[:,1][selected].sum().item()
        
        index_to_var_name = {v_map[var_name]:var_name for var_name in v_map}
        scp_model = scp.Model()
        scp_model.hideOutput()
        scp_model.readProblem(instance_input_path)
        scp_var_map = {}  # Map to store original variables and their complements
        # Step 1: Duplicate binary variables with complements
        for var in scp_model.getVars():
            scp_var_map[var.name] = var
        constraints = []
        deltas = []
        #iterate over indexes of selected
        for i in selected:
            tar_var = index_to_var_name[i.item()] #target variable
            delta_var = scp_model.addVar(name=f"delta_{tar_var}", vtype="B")
            if pred_probs[i,0].item():
                constraints.append(scp_var_map[tar_var]<=delta_var)
            elif pred_probs[i,1].item():
                constraints.append(1-scp_var_map[tar_var]<=delta_var)
            else:
                raise Exception("No se ha predicho un valor valido")
            deltas.append(delta_var)
        if len(deltas)>0:
            constraints.append(sum(deltas) <= Delta )
        return constraints, scp_model, k_0, k_1, Delta

class FixedThreeRatiosTrustRegionConstructor(TrustRegionConstructor):
    def __init__(self, ml_model, graph_type:str, k_ratio:float,value_0_ratio:float, Delta_ratio:float, log_file:str):
        super().__init__(ml_model, graph_type, log_file)
        self.k_ratio = k_ratio
        self.value_0_ratio = value_0_ratio
        self.Delta_ratio = Delta_ratio
        if not (0 < k_ratio < 1):
            raise ValueError(f"k_ratio must be in the range (0, 1), but got {k_ratio}.")
        if not (0 < value_0_ratio < 1):
            raise ValueError(f"value_0_ratio must be in the range (0, 1), but got {value_0_ratio}.")
        if not (0 < Delta_ratio < 1):
            raise ValueError(f"Delta_ratio must be in the range (0, 1), but got {Delta_ratio}.")
        
    def params_to_string(self) -> str:
        """Return a string representation of the parameters used for the trust region construction."""
        return f"k_ratio: {self.k_ratio}, value_0_ratio: {self.value_0_ratio}, Delta_ratio: {self.Delta_ratio}"

    def compute_constraints(self, instance_input_path:str, pred_probs, v_map) -> tuple:
        N = len(v_map)
        k_0 = int(N*self.k_ratio*(self.value_0_ratio))
        k_1 = int(N*self.k_ratio*(1-self.value_0_ratio))
        Delta = int(N*self.k_ratio*self.Delta_ratio)


        selected_0 = torch.topk(pred_probs[:,0],k_0)[1]
        selected_1 = torch.topk(pred_probs[:,1],k_1)[1]
        
        index_to_var_name = {v_map[var_name]:var_name for var_name in v_map}
        scp_model = scp.Model()
        scp_model.hideOutput()
        scp_model.readProblem(instance_input_path)
        scp_var_map = {}  # Map to store original variables and their complements
        # Step 1: Duplicate binary variables with complements
        for var in scp_model.getVars():
            scp_var_map[var.name] = var
        constraints = []
        deltas = []
        #iterate over indexes of selected
        for i in selected_0:
            tar_var = index_to_var_name[i.item()] #target variable
            delta_var = scp_model.addVar(name=f"delta_{tar_var}", vtype="B")
            constraints.append(scp_var_map[tar_var]<=delta_var)
            deltas.append(delta_var)
        for i in selected_1:
            tar_var = index_to_var_name[i.item()] #target variable
            delta_var = scp_model.addVar(name=f"delta_{tar_var}", vtype="B")
            constraints.append(1-scp_var_map[tar_var]<=delta_var)
            deltas.append(delta_var)
        if len(deltas)>0:
            constraints.append(sum(deltas) <= Delta )
        return constraints, scp_model, k_0, k_1, Delta


