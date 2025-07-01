import torch
import numpy as np
from sklearn.cluster import KMeans
from torch_geometric.data import Data

from .RoleDiscoveryModel import RoleDiscoveryModel
from ..utils.feature_engineering import get_graphlet_features

class FeatureBasedRolesGraphlets(RoleDiscoveryModel):

    def __init__(self):
        print("Initialized Feature-Based Role Discovery Model (Graphlets).")
        self.node_features = None

    def _extract_node_features(self, data: Data) -> np.ndarray:
        if self.node_features is not None:
            return self.node_features
        self.node_features = get_graphlet_features(data, scale=True)

        return self.node_features

    def predict(self, data: Data, k: int) -> tuple[torch.Tensor, torch.Tensor]:
        scaled_features = self._extract_node_features(data)
        
        print(f"Clustering {scaled_features.shape[0]} nodes into {k} roles using KMeans...")
        kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto', verbose=0)
        role_labels = kmeans.fit_predict(scaled_features)
        
        return torch.from_numpy(scaled_features).float(), torch.from_numpy(role_labels).int()