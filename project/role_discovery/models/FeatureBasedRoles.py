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

    def _extract_node_features(self, data: Data) -> np.ndarray:
        if self.node_features is not None:
            return self.node_features

        print("Extracting a comprehensive set of node-level structural features...")
        
        G_directed = to_networkx(data, to_undirected=False)
        G_undirected = to_networkx(data, to_undirected=True)

        ##### Feature Extraction #####
        # Basic Centrality Measures
        degree_centrality = nx.degree_centrality(G_undirected)
        closeness_centrality = nx.closeness_centrality(G_undirected)
        betweenness_centrality = nx.betweenness_centrality(G_undirected, k=min(100, len(G_undirected)-1)) # k for approximation
        
        # Influence-based Centrality
        eigenvector_centrality = nx.eigenvector_centrality_numpy(G_undirected)
        
        pagerank = nx.pagerank(G_undirected, alpha=0.85)

        # Local Structure Measures
        clustering_coefficient = nx.clustering(G_undirected)
        
        #  Directed Graph Features (if applicable)
        if data.is_directed():
            in_degree_centrality = {n: d for n, d in G_directed.in_degree(weight=None)}
            out_degree_centrality = {n: d for n, d in G_directed.out_degree(weight=None)}
        else: 
            in_degree_centrality = {n: 0 for n in G_undirected.nodes()}
            out_degree_centrality = {n: 0 for n in G_undirected.nodes()}

        features = {}
        for node in sorted(G_undirected.nodes()):
            features[node] = [
                degree_centrality.get(node, 0),
                closeness_centrality.get(node, 0),
                betweenness_centrality.get(node, 0),
                eigenvector_centrality.get(node, 0),
                pagerank.get(node, 0),
                clustering_coefficient.get(node, 0),
                in_degree_centrality.get(node, 0),
                out_degree_centrality.get(node, 0),
            ]
        
        feature_matrix = np.array([features[node] for node in sorted(G_undirected.nodes())])
        
        feature_matrix = np.nan_to_num(feature_matrix, nan=0.0, posinf=0.0, neginf=0.0)
        
        self.node_features = self.scaler.fit_transform(feature_matrix)
        
        print(f"Feature extraction complete. Matrix shape: {self.node_features.shape}")
        return self.node_features

    def predict(self, data: Data, k: int) -> tuple[torch.Tensor, torch.Tensor]:
        scaled_features = self._extract_node_features(data)
        
        print(f"Clustering {scaled_features.shape[0]} nodes into {k} roles using KMeans...")
        kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto', verbose=0)
        role_labels = kmeans.fit_predict(scaled_features)
        
        return torch.from_numpy(scaled_features).float(), torch.from_numpy(role_labels).int()