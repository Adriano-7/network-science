import torch
import numpy as np
import networkx as nx
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx

from .RoleDiscoveryModel import RoleDiscoveryModel

class FeatureBasedRoles(RoleDiscoveryModel):
    def __init__(self):
        print("Initialized Feature-Based Role Discovery Model.")
        self.scaler = StandardScaler()

    def _extract_node_features(self, G: nx.Graph) -> np.ndarray:
        """Extracts structural features for each node in the graph."""
        print("Extracting node-level structural features...")
        features = {}
        
        degree_centrality = nx.degree_centrality(G)
        betweenness_centrality = nx.betweenness_centrality(G)
        closeness_centrality = nx.closeness_centrality(G)
        
        clustering_coefficient = nx.clustering(G)
        
        for node in G.nodes():
            features[node] = [
                degree_centrality.get(node, 0),
                betweenness_centrality.get(node, 0),
                closeness_centrality.get(node, 0),
                clustering_coefficient.get(node, 0)
            ]
        
        feature_matrix = np.array([features[i] for i in sorted(G.nodes())])
        return feature_matrix

    def predict(self, graph_data: Data, k: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Extracts node features, clusters them with KMeans, and returns role assignments.
        """
        nx_graph = to_networkx(graph_data, to_undirected=True, node_attrs=['x'])
        
        node_features = self._extract_node_features(nx_graph)
        scaled_features = self.scaler.fit_transform(node_features)
        
        print(f"Clustering {scaled_features.shape[0]} nodes into {k} roles using KMeans...")
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        role_labels = kmeans.fit_predict(scaled_features)
        
        return torch.from_numpy(scaled_features).float(), torch.from_numpy(role_labels).int()