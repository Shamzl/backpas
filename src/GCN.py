import torch
import torch.nn as nn
import torch_geometric
import pickle as pkl
from constants import LITERALS_GRAPH, VARIABLES_GRAPH, GCN_LAYER, GTR_LAYER

def flip(x, dim=0):
    # Move the specified dimension to the front
    x = x.transpose(0, dim)
    
    # Get the shape of the tensor
    shape = x.shape
    
    # Ensure the leading dimension is even
    if shape[0] % 2 != 0:
        raise ValueError("The size of the specified dimension must be even.")
    
    # Reshape to collapse the leading dimension to pairs
    new_shape = (shape[0] // 2, 2) + shape[1:]
    x = x.view(new_shape)
    
    # Initialize an empty tensor with the same shape as the reshaped tensor
    result = torch.empty_like(x)
    
    # Perform the swapping for related pairs
    result[:, 0] = x[:, 1]
    result[:, 1] = x[:, 0]
    
    # Reshape back to the original shape
    result = result.view(shape)
    
    # Move the dimensions back to their original order
    result = result.transpose(0, dim)
    
    return result
def inverse_join_literals(tensor):
    n, two_k = tensor.shape
    if two_k % 2 != 0:
        raise ValueError("The second dimension must be even.")
    
    k = two_k // 2
    
    # Reshape to (n, k, 2)
    tensor = tensor.view(n, k, 2)
    
    # Permute to (n, 2, k)
    tensor = tensor.permute(0, 2, 1)
    
    # Reshape to (2n, k)
    tensor = tensor.reshape(2 * n, k)
    
    return tensor

def join_literals(tensor):
    two_n, k = tensor.shape
    if two_n % 2 != 0:
        raise ValueError("The first dimension must be even.")

    n = two_n // 2

    # Reshape to (n, 2, k)
    tensor = tensor.view(n, 2, k)

    # Permute to (n, k, 2)
    tensor = tensor.permute(0, 2, 1)

    # Reshape to (n, 2k)
    tensor = tensor.reshape(n, 2 * k)
    
    return tensor
    
class BackbonePredictor(torch.nn.Module):
    def __init__(self,graph_type:str, layer_type:str, num_layers:int, use_literals_message:bool, emb_size:int=64):
        super().__init__()
        cons_nfeats = 4
        edge_nfeats = 1
        var_nfeats = 6
        self.num_layers = num_layers
        self.graph_type = graph_type
        
        # CONSTRAINT EMBEDDING
        self.cons_embedding = torch.nn.Sequential(
            torch.nn.LayerNorm(cons_nfeats),
            torch.nn.Linear(cons_nfeats, emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.ReLU(),
        )

        # EDGE EMBEDDING
        self.edge_embedding = torch.nn.Sequential(
            torch.nn.LayerNorm(edge_nfeats),
        )

        # VARIABLE EMBEDDING
        self.var_embedding = torch.nn.Sequential(
            torch.nn.LayerNorm(var_nfeats),
            torch.nn.Linear(var_nfeats, emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.ReLU(),
        )

        if layer_type == GCN_LAYER:
            self.conv_v_to_c = torch.nn.ModuleList([
                BipartiteGraphConvolution(emb_size) for _ in range(num_layers)
            ])
            self.conv_c_to_v = torch.nn.ModuleList([
                BipartiteGraphConvolution(emb_size) for _ in range(num_layers)
            ])
        elif layer_type == GTR_LAYER:
            self.conv_v_to_c = torch.nn.ModuleList([
                BipartiteTransformerLayer(emb_size) for _ in range(num_layers)
            ])
            self.conv_c_to_v = torch.nn.ModuleList([
                BipartiteTransformerLayer(emb_size) for _ in range(num_layers)
            ])
        else:
            raise ValueError(f"Unknown layer_type requested. Choose '{GCN_LAYER}' or '{GTR_LAYER}'.")
        
        if graph_type == LITERALS_GRAPH:
            output_emb_size = 2 * emb_size
        elif graph_type == VARIABLES_GRAPH:
            output_emb_size = emb_size
        else:
            raise ValueError(f"Unknown graph_type requested: {graph_type}. Choose '{LITERALS_GRAPH}' or '{VARIABLES_GRAPH}'.")
        
        if use_literals_message:
            assert graph_type == LITERALS_GRAPH, "Literals message passing is only supported for literals graph type."
            self.l_to_l = torch.nn.ModuleList([
                LiteralsMessage(emb_size) for _ in range(num_layers)
            ])

        self.output_module = nn.Sequential(
            nn.Linear(output_emb_size, output_emb_size),
            nn.ReLU(),
            nn.Linear(output_emb_size, 3, bias=False),
        )


    def forward(
        self, constraint_features, edge_indices, edge_features, variable_features
    ):
        reversed_edge_indices = torch.stack([edge_indices[1], edge_indices[0]], dim=0)

        # First step: linear embedding layers to a common dimension
        constraint_features = self.cons_embedding(constraint_features)
        edge_features = self.edge_embedding(edge_features)
        variable_features = self.var_embedding(variable_features)

        # Iterate over the gnn layers
        for i in range(self.num_layers):
            # Apply the gnn layer from variable to constraint
            constraint_features = self.conv_v_to_c[i](variable_features, reversed_edge_indices, edge_features, constraint_features)
            # Apply the gnn layer from constraint to variable
            variable_features = self.conv_c_to_v[i](constraint_features, edge_indices, edge_features, variable_features)
            # Apply the literals message passing
            if hasattr(self, 'l_to_l'):
                # If using literals message passing, apply it
                variable_features = self.l_to_l[i](variable_features)

        if self.graph_type == LITERALS_GRAPH:
            # If the graph type is literals, we need to join the literals
            variable_features = join_literals(variable_features)

        return self.output_module(variable_features)
    
# Message Passing Layers (LiteralsMessage, BipartiteGraphConvolution, BipartiteTransformerLayer)
class LiteralsMessage(torch.nn.Module):
    def __init__(self,size):
        super().__init__()
        self.a = torch.nn.Parameter(torch.randn(size))
        self.b = torch.nn.Parameter(torch.randn(size))
        self.activation = torch.nn.ReLU()
    def forward(self,literals):
        return self.activation(self.a*literals + self.b*flip(literals))
    
class BipartiteGraphConvolution(torch_geometric.nn.MessagePassing):
    """
    The bipartite graph convolution is already provided by pytorch geometric and we merely need
    to provide the exact form of the messages being passed.
    """

    def __init__(self,emb_size = 64):
        super().__init__("add")
        

        self.feature_module_left = torch.nn.Sequential(
            torch.nn.Linear(emb_size, emb_size)
        )
        self.feature_module_edge = torch.nn.Sequential(
            torch.nn.Linear(1, emb_size, bias=False)
        )
        self.feature_module_right = torch.nn.Sequential(
            torch.nn.Linear(emb_size, emb_size, bias=False)
        )
        self.feature_module_final = torch.nn.Sequential(
            torch.nn.LayerNorm(emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, emb_size),
        )

        self.post_conv_module = torch.nn.Sequential(torch.nn.LayerNorm(emb_size))

        # output_layers
        self.output_module = torch.nn.Sequential(
            torch.nn.Linear(2 * emb_size, emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, emb_size),
        )

    def forward(self, left_features, edge_indices, edge_features, right_features):
        """
        This method sends the messages, computed in the message method.
        """


        output = self.propagate(
            edge_indices,
            size=(left_features.shape[0], right_features.shape[0]),
            node_features=(left_features, right_features),
            edge_features=edge_features,
        )

        return self.output_module(
            torch.cat([self.post_conv_module(output), right_features], dim=-1)
        )


    def message(self, node_features_i, node_features_j, edge_features):

        output = self.feature_module_final(
            self.feature_module_left(node_features_i)
            + self.feature_module_edge(edge_features)
            + self.feature_module_right(node_features_j)
        )

        return output
    
class BipartiteTransformerLayer(torch_geometric.nn.MessagePassing):
    def __init__(self, emb_size, heads=4):
        super().__init__(aggr='add')
        self.heads = heads
        self.emb_size = emb_size
        self.head_dim = emb_size // heads

        self.q_lin = nn.Linear(emb_size, emb_size)
        self.k_lin = nn.Linear(emb_size, emb_size)
        self.v_lin = nn.Linear(emb_size, emb_size)
        self.edge_lin = nn.Linear(1, emb_size)

        self.out_lin = nn.Linear(emb_size, emb_size)
        self.norm = nn.LayerNorm(emb_size)

    def forward(self, x_src, edge_index, edge_attr, x_tgt):
        out = self.propagate(edge_index, x=(x_src, x_tgt), edge_attr=edge_attr)
        return self.norm(x_tgt + self.out_lin(out))  # residual

    def message(self, x_i, x_j, edge_attr, index):
        # Attention over neighbors j -> i
        Q = self.q_lin(x_i).view(-1, self.heads, self.head_dim)
        K = self.k_lin(x_j).view(-1, self.heads, self.head_dim)
        V = self.v_lin(x_j).view(-1, self.heads, self.head_dim)
        E = self.edge_lin(edge_attr).view(-1, self.heads, self.head_dim)

        scores = (Q * (K + E)).sum(-1) / self.head_dim**0.5  # [E, H]
        alpha = torch_geometric.utils.softmax(scores, index)

        return (V * alpha.unsqueeze(-1)).view(-1, self.emb_size)
# GraphDataset
class GraphBackboneDataset(torch_geometric.data.Dataset):

    def __init__(self, sample_files: list, graph_type: str):
        super().__init__(root=None, transform=None, pre_transform=None)
        self.sample_files = sample_files
        self.graph_type = graph_type
        assert graph_type in [LITERALS_GRAPH, VARIABLES_GRAPH], f"Unknown graph type requested: {graph_type}. Choose '{LITERALS_GRAPH}' or '{VARIABLES_GRAPH}'."

    def len(self):
        return len(self.sample_files)

    def process_sample(self,filepath):
        with open(filepath, "rb") as f:
            data = pkl.load(f)
        bipartite_graph =  data['bipartite_graph']
        target = data['target']
        assert data["graph_type"] == self.graph_type, f"Graph type mismatch: expected {self.graph_type}, got {data['graph_type']}"
        return bipartite_graph, target

    def get_graph_components(A, v_nodes, c_nodes):
        constraint_features = c_nodes
        edge_indices = A._indices()

        variable_features = v_nodes
        edge_features = A._values().unsqueeze(1)
        #edge_features = torch.ones(edge_features.shape)
        constraint_features = torch.nan_to_num(constraint_features,1)
        
        return constraint_features, edge_indices, edge_features, variable_features

    def get(self, index):
        """
        This method loads a node bipartite graph observation as saved on the disk during data collection.
        """

        bipartite_graph, target = self.process_sample(self.sample_files[index])
        A, v_map, v_nodes, c_nodes = bipartite_graph
        constraint_features, edge_indices, edge_features, variable_features = GraphBackboneDataset.get_graph_components(A, v_nodes, c_nodes)    

        graph = BipartiteNodeData(
            constraint_features.cpu().float(),
            edge_indices.cpu().long(),
            edge_features.cpu().float(),
            variable_features.cpu().float(),
        )
        
        
        # We must tell pytorch geometric how many nodes there are, for indexing purposes
        graph.num_nodes = constraint_features.shape[0] + variable_features.shape[0]
        graph.target = target
        if self.graph_type == LITERALS_GRAPH:
            graph.num_original_bin_variables = variable_features.shape[0] // 2
        else:
            graph.num_original_bin_variables = variable_features.shape[0]
        #graph.varNames = [v for _,v in sorted(v_map.items(), key=lambda item: item[1])]

        return graph

class BipartiteNodeData(torch_geometric.data.Data):
    """
    This class encode a node bipartite graph observation as returned by the `ecole.observation.NodeBipartite`
    observation function in a format understood by the pytorch geometric data handlers.
    """

    def __init__(
            self,
            constraint_features,
            edge_indices,
            edge_features,
            variable_features,

    ):
        super().__init__()
        self.constraint_features = constraint_features
        self.edge_index = edge_indices
        self.edge_attr = edge_features
        self.variable_features = variable_features



    def __inc__(self, key, value, store, *args, **kwargs):
        """
        We overload the pytorch geometric method that tells how to increment indices when concatenating graphs
        for those entries (edge index, candidates) for which this is not obvious.
        """
        if key == "edge_index":
            return torch.tensor(
                [[self.constraint_features.size(0)], [self.variable_features.size(0)]]
            )
        elif key == "candidates": # This is not used anymore
            return self.variable_features.size(0)
        else:
            return super().__inc__(key, value, *args, **kwargs)