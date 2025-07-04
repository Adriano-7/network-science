import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv 
import numpy as np 
import networkx as nx
from torch_geometric.utils import to_networkx
from torch_geometric.data import Data 
import random

# GCN Model
class GCN(torch.nn.Module):
    def __init__(self, num_node_features, num_classes, hidden_channels=16):
        super().__init__()
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, num_classes)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        embeddings = x 
        x = self.conv2(x, edge_index) 
        return x, embeddings 
    
# GraphSage Model
class GraphSage(torch.nn.Module):
    def __init__(self, num_node_features, num_classes, hidden_channels=16):
        super().__init__()
        self.conv1 = SAGEConv(num_node_features, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, num_classes)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        embeddings = x 
        x = self.conv2(x, edge_index) 
        return x, embeddings 

def run_gnn(model_class, data_pyg, num_classes, num_epochs=200):
    """
    Trains a GNN model (of model_class type) for node classification and returns predicted communities.
    Assumes data_pyg has x, edge_index, y, train_mask.
    Also returns the learned node embeddings.
    """
    device = torch.device('cpu') 
    data_pyg = data_pyg.to(device)

    model = model_class(
        num_node_features=data_pyg.num_node_features,
        num_classes=num_classes, 
        hidden_channels=16 
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    criterion = torch.nn.CrossEntropyLoss()

    model.train()
    for epoch in range(1, num_epochs + 1):
        optimizer.zero_grad()
        out_logits, _ = model(data_pyg.x, data_pyg.edge_index) # Only use logits for loss
        loss = criterion(out_logits[data_pyg.train_mask], data_pyg.y[data_pyg.train_mask])
        loss.backward()
        optimizer.step()
    
    model.eval()
    with torch.no_grad():
        out_logits, embeddings = model(data_pyg.x, data_pyg.edge_index) 
        predicted_classes = out_logits.argmax(dim=1).cpu().numpy()

    partition = {i: predicted_classes[i] for i in range(len(predicted_classes))}
    return partition, embeddings.cpu().numpy()

def calculate_graphlet_features(G):
    """
    Calculates graphlet-based features for each node in the graph.
    
    As a starting point, this function calculates the number of triangles
    each node is part of. This is a basic but powerful structural feature.
    More complex implementations could count orbits of 4-node graphlets.

    Args:
        G (nx.Graph): A NetworkX graph.

    Returns:
        torch.Tensor: A tensor of shape (num_nodes, 1) where each element
                      is the triangle count for the corresponding node.
    """
    print("  - Calculating graphlet features (triangle counts)...")
    
    triangle_counts = nx.triangles(G)
    
    feature_vector = [triangle_counts.get(node, 0) for node in G.nodes()]
    
    gdv_tensor = torch.tensor(feature_vector, dtype=torch.float).view(-1, 1)

    mean = gdv_tensor.mean(dim=0, keepdim=True)
    std = gdv_tensor.std(dim=0, keepdim=True)
    gdv_tensor = (gdv_tensor - mean) / (std + 1e-8)

    print(f"    Graphlet feature calculation complete. Feature shape: {gdv_tensor.shape}")
    return gdv_tensor


def get_cora_subgraph_pyg(full_data_pyg, num_nodes_subgraph=150):
    """
    Extracts a connected subgraph of a specified number of nodes from the full Cora graph.
    Returns a PyTorch Geometric Data object with relabeled nodes (0-indexed),
    node features, ground truth, and randomly generated train/val/test masks.
    """
    G_full = to_networkx(full_data_pyg, to_undirected=True)
    
    nodes_in_full_graph = list(G_full.nodes())
    if len(nodes_in_full_graph) < num_nodes_subgraph:
        print(f"Warning: Full graph (Cora) has fewer nodes ({len(nodes_in_full_graph)}) than requested subgraph size ({num_nodes_subgraph}). Returning full graph data.")
        return full_data_pyg

    subgraph_nodes = set()
    
    attempts = 0
    max_attempts = 10 
    while len(subgraph_nodes) < num_nodes_subgraph and attempts < max_attempts:
        attempts += 1
        start_node = random.choice(nodes_in_full_graph)
        current_subgraph_nodes = set()
        queue = [start_node]
        visited = {start_node}

        while queue and len(current_subgraph_nodes) < num_nodes_subgraph:
            current_node = queue.pop(0)
            current_subgraph_nodes.add(current_node)
            for neighbor in G_full.neighbors(current_node):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        
        if len(current_subgraph_nodes) >= num_nodes_subgraph * 0.8: 
            subgraph_nodes = current_subgraph_nodes
            break
        else: 
            subgraph_nodes = set()
            
    if len(subgraph_nodes) < num_nodes_subgraph:
        print(f"Warning: BFS method struggled to find a connected subgraph of {num_nodes_subgraph} nodes after {max_attempts} attempts. Taking a random sample of {num_nodes_subgraph} nodes instead (might be disconnected).")
        subgraph_nodes = set(random.sample(nodes_in_full_graph, num_nodes_subgraph))


    subgraph = G_full.subgraph(subgraph_nodes).copy() 
    
    old_to_new_node_map = {old_id: new_id for new_id, old_id in enumerate(sorted(subgraph_nodes))}
    
    subgraph_edges = []
    for u, v in G_full.edges():
        if u in subgraph_nodes and v in subgraph_nodes:
            subgraph_edges.append([old_to_new_node_map[u], old_to_new_node_map[v]])
    if not subgraph_edges:
        edge_index_subgraph = torch.empty((2, 0), dtype=torch.long)
    else:
        edge_index_subgraph = torch.tensor(subgraph_edges, dtype=torch.long).t().contiguous()

    subgraph_x = torch.stack([full_data_pyg.x[old_node_id] for old_node_id in sorted(subgraph_nodes)])
    subgraph_y = torch.stack([full_data_pyg.y[old_node_id] for old_node_id in sorted(subgraph_nodes)])

    num_nodes_actual = len(subgraph_nodes)
    idx = np.arange(num_nodes_actual)
    np.random.shuffle(idx)

    train_size = int(0.6 * num_nodes_actual)
    val_size = int(0.2 * num_nodes_actual)
    
    train_mask = torch.zeros(num_nodes_actual, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes_actual, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes_actual, dtype=torch.bool)

    train_mask[idx[:train_size]] = True
    val_mask[idx[train_size : train_size + val_size]] = True
    test_mask[idx[train_size + val_size :]] = True

    subgraph_data = Data(x=subgraph_x, edge_index=edge_index_subgraph, y=subgraph_y,
                         train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)
    
    print(f"Created PyG Cora subgraph with {subgraph_data.num_nodes} nodes and {subgraph_data.num_edges} edges.")
    return subgraph_data