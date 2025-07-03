import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv 
import numpy as np 
import networkx as nx

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