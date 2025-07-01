import torch
import numpy as np
import networkx as nx
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx
from tqdm import tqdm

from .RoleDiscoveryModel import RoleDiscoveryModel

class FeatureBasedRolesGraphlets(RoleDiscoveryModel):
    """
    A role discovery model that uses graphlet-based features for each node.
    Specifically, it counts the number of 3-node (triangles, wedges) and 4-node
    (4-cycles) graphlets each node participates in. This provides a rich
    topological signature for clustering.
    
    Note: This model requires the `tqdm` library for progress bars during feature
    extraction. You can install it via: pip install tqdm
    """
    def __init__(self):
        print("Initialized Feature-Based Role Discovery Model (Graphlets).")
        self.scaler = StandardScaler()
        self.node_features = None

    def _count_node_four_cycles(self, G: nx.Graph, node: int) -> int:
        """
        Counts the number of simple 4-cycles a specific node is a part of.
        A 4-cycle is a path of 4 nodes u-v-w-x-u.
        This counts the number of such cycles where the given 'node' is 'u'.
        """
        count = 0
        neighbors_of_node = list(G.neighbors(node))
        if len(neighbors_of_node) < 2:
            return 0

        # Iterate over all unique pairs of neighbors (v, x) of the target 'node'
        for i in range(len(neighbors_of_node)):
            for j in range(i + 1, len(neighbors_of_node)):
                v = neighbors_of_node[i]
                x = neighbors_of_node[j]
                
                # A 4-cycle exists if v and x share a common neighbor other than 'node'.
                # The number of such common neighbors 'w' is what we add to the count.
                common_neighbors_vx = len(list(nx.common_neighbors(G, v, x)))
                if common_neighbors_vx > 1:
                    count += (common_neighbors_vx - 1)
        return count

    def _extract_node_features(self, data: Data) -> np.ndarray:
        # --- CORRECTED LINE ---
        if self.node_features is not None:
            return self.node_features

        print("Extracting a set of node-level graphlet features (this may take a while)...")
        
        G = to_networkx(data, to_undirected=True)
        node_list = sorted(list(G.nodes()))
        features = {}

        # Pre-calculate all triangles at once for efficiency
        print("Pre-calculating triangles for all nodes...")
        all_triangles = nx.triangles(G)
        
        print("Calculating graphlet features for each node...")
        for node in tqdm(node_list, desc="Extracting Graphlet Features"):
            degree = G.degree(node)
            
            # --- 3-node graphlets ---
            # Orbit: Node in a triangle (3-clique)
            num_triangles = all_triangles.get(node, 0)
            
            # Orbit: Node is the center of a wedge (2-path)
            # Total pairs of neighbors minus the pairs that form a triangle
            num_wedges_centered = (degree * (degree - 1) // 2) - num_triangles

            # --- 4-node graphlets (simplified) ---
            # We will count the number of 4-cycles the node is part of as a representative
            # complex feature.
            num_4_cycles = self._count_node_four_cycles(G, node)

            features[node] = [
                num_triangles,
                num_wedges_centered,
                num_4_cycles,
            ]
        
        feature_matrix = np.array([features[node] for node in node_list])
        feature_matrix = np.nan_to_num(feature_matrix, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Scaling is crucial as feature counts can have vastly different ranges
        self.node_features = self.scaler.fit_transform(feature_matrix)
        
        print(f"Feature extraction complete. Matrix shape: {self.node_features.shape}")
        return self.node_features

    def predict(self, data: Data, k: int) -> tuple[torch.Tensor, torch.Tensor]:
        scaled_features = self._extract_node_features(data)
        
        print(f"Clustering {scaled_features.shape[0]} nodes into {k} roles using KMeans...")
        kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto', verbose=0)
        role_labels = kmeans.fit_predict(scaled_features)
        
        return torch.from_numpy(scaled_features).float(), torch.from_numpy(role_labels).int()