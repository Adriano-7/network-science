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
        self.node_features = None

    def _extract_node_features(self, G: nx.Graph) -> np.ndarray:
        if self.node_features is not None:
            return self.node_features

        print("Extracting node-level structural features (Degree, Betweenness, PageRank, Clustering Coeff)...")
        features = {}
        
        degree_centrality = nx.degree_centrality(G)
        betweenness_centrality = nx.betweenness_centrality(G)
        pagerank = nx.pagerank(G)
        
        clustering_coefficient = nx.clustering(G)
        
        for node in sorted(G.nodes()):
            features[node] = [
                degree_centrality.get(node, 0),
                betweenness_centrality.get(node, 0),
                pagerank.get(node, 0),
                clustering_coefficient.get(node, 0)
            ]
        
        feature_matrix = np.array([features[node] for node in sorted(G.nodes())])
        
        self.node_features = self.scaler.fit_transform(feature_matrix)
        return self.node_features

    def predict(self, graph_data: Data, k: int) -> tuple[torch.Tensor, torch.Tensor]:
        nx_graph = to_networkx(graph_data, to_undirected=True)
        
        scaled_features = self._extract_node_features(nx_graph)
        
        print(f"Clustering {scaled_features.shape[0]} nodes into {k} roles using KMeans...")
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10, verbose=0)
        role_labels = kmeans.fit_predict(scaled_features)
        
        return torch.from_numpy(scaled_features).float(), torch.from_numpy(role_labels).int()