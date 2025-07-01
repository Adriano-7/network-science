import numpy as np
import networkx as nx
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
import torch

def _count_node_four_cycles(G: nx.Graph, node: int) -> int:
    count = 0
    neighbors_of_node = list(G.neighbors(node))
    if len(neighbors_of_node) < 2:
        return 0

    for i in range(len(neighbors_of_node)):
        for j in range(i + 1, len(neighbors_of_node)):
            v = neighbors_of_node[i]
            x = neighbors_of_node[j]
            common_neighbors_vx = len(list(nx.common_neighbors(G, v, x)))
            if common_neighbors_vx > 1:
                count += (common_neighbors_vx - 1)
    return count

def get_graphlet_features(data: Data, scale: bool = True) -> np.ndarray:
    print("Extracting a set of node-level graphlet features (this may take a while)...")
    
    G = to_networkx(data, to_undirected=True)
    node_list = sorted(list(G.nodes()))
    features = {}

    print("Pre-calculating triangles for all nodes...")
    all_triangles = nx.triangles(G)
    
    print("Calculating graphlet features for each node...")
    for node in tqdm(node_list, desc="Extracting Graphlet Features"):
        degree = G.degree(node)
        num_triangles = all_triangles.get(node, 0)
        num_wedges_centered = (degree * (degree - 1) // 2) - num_triangles
        num_4_cycles = _count_node_four_cycles(G, node)
        features[node] = [num_triangles, num_wedges_centered, num_4_cycles]
    
    feature_matrix = np.array([features[node] for node in node_list])
    feature_matrix = np.nan_to_num(feature_matrix, nan=0.0, posinf=0.0, neginf=0.0)

    if scale:
        scaler = StandardScaler()
        feature_matrix = scaler.fit_transform(feature_matrix)
    
    print(f"Graphlet feature extraction complete. Matrix shape: {feature_matrix.shape}")
    return feature_matrix