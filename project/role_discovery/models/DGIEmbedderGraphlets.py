import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.nn.models import DeepGraphInfomax
from sklearn.cluster import KMeans
from pathlib import Path
import numpy as np

from .RoleDiscoveryModel import RoleDiscoveryModel
from .DGIEmbedder import Encoder, corruption # Re-use from the original DGI
from ..utils.feature_engineering import get_graphlet_features

class DGIEmbedderGraphlets(RoleDiscoveryModel):
    def __init__(self, hidden_channels, epochs=300, lr=0.001,
                 force_retrain: bool = False, model_path: str = None):
        self.hidden_channels = hidden_channels
        self.epochs = epochs
        self.lr = lr
        self.model = None
        self.embeddings = None
        self.force_retrain = force_retrain
        self.model_path = model_path
        print("Initialized DGI Embedder with GRAPHLET features.")

    def train(self, data: Data):
        # Use graphlet features instead of node degrees
        graphlet_feats_np = get_graphlet_features(data, scale=False)
        graphlet_feats_tensor = torch.from_numpy(graphlet_feats_np).float()
        
        structural_data = Data(x=graphlet_feats_tensor, edge_index=data.edge_index)
        
        encoder = Encoder(structural_data.num_features, self.hidden_channels)
        self.model = DeepGraphInfomax(
            encoder=encoder,
            summary=lambda z, *args, **kwargs: z.mean(dim=0),
            corruption=corruption,
            hidden_channels=self.hidden_channels
        )

        if not self.force_retrain and self.model_path and Path(self.model_path).exists():
            print(f"Loading pre-trained DGI (Graphlet) model from {self.model_path}")
            self.model.load_state_dict(torch.load(self.model_path))
            with torch.no_grad():
                self.model.eval()
                self.embeddings = self.model.encoder(structural_data.x, structural_data.edge_index).detach()
            print("Embeddings generated from pre-trained DGI (Graphlet) model.")
            return

        print("Starting DGI training with graphlet features...")
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=5e-4)

        for epoch in range(1, self.epochs + 1):
            self.model.train()
            optimizer.zero_grad()
            pos_z, neg_z, summary = self.model(structural_data.x, structural_data.edge_index)
            loss = self.model.loss(pos_z, neg_z, summary)
            loss.backward()
            optimizer.step()

            if epoch % 50 == 0:
                print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')
        
        if self.model_path:
            Path(self.model_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save(self.model.state_dict(), self.model_path)
            print(f"Trained DGI (Graphlet) model saved to {self.model_path}")

        with torch.no_grad():
            self.model.eval()
            self.embeddings = self.model.encoder(structural_data.x, structural_data.edge_index).detach()

    def predict(self, data: Data, k: int) -> tuple[torch.Tensor, torch.Tensor]:
        if self.embeddings is None:
            self.train(data)
        kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
        role_labels = kmeans.fit_predict(self.embeddings.cpu().numpy())
        return self.embeddings, torch.from_numpy(role_labels).int()