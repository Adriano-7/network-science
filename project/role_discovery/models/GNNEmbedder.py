import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans
from torch_geometric.data import Data
from torch_geometric.nn import GAE, GCNConv
import torch_geometric.transforms as T
from torch_geometric.utils import degree

from .RoleDiscoveryModel import RoleDiscoveryModel

class GCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels, cached=True)
        self.conv2 = GCNConv(hidden_channels, out_channels, cached=True)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv2(x, edge_index)

class GNNEmbedder(RoleDiscoveryModel):
    def __init__(self, hidden_channels: int = 128, emb_dim: int = 32,
                 epochs: int = 400, lr: float = 0.005):
        print("Initialized GNN Embedder (GAE) for Role Discovery.")
        self.epochs = epochs
        self.lr = lr
        self.hidden_channels = hidden_channels
        self.emb_dim = emb_dim
        self.model = None
        self.embeddings = None

    def train(self, graph_data: Data):
        print("Starting GAE training...")
        
        # 1. Use node degrees as features to focus on structure
        deg = degree(graph_data.edge_index[0], graph_data.num_nodes).view(-1, 1)
        
        # 2. Apply feature normalization
        transform = T.NormalizeFeatures()
        structural_data = Data(x=deg, edge_index=graph_data.edge_index)
        structural_data = transform(structural_data)

        # 3. Initialize the GAE model
        in_channels = structural_data.num_features
        encoder = GCNEncoder(in_channels, self.hidden_channels, self.emb_dim)
        self.model = GAE(encoder)
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        
        for epoch in range(1, self.epochs + 1):
            self.model.train()
            optimizer.zero_grad()
            
            z = self.model.encode(structural_data.x, structural_data.edge_index)
            loss = self.model.recon_loss(z, structural_data.edge_index)
            
            loss.backward()
            optimizer.step()
            
            if epoch % 50 == 0:
                print(f'Epoch: {epoch:03d}, Reconstruction Loss: {loss:.4f}')
        
        print("GAE training finished.")
        with torch.no_grad():
            self.model.eval()
            self.embeddings = self.model.encode(structural_data.x, structural_data.edge_index).detach()

    def predict(self, graph_data: Data, k: int) -> tuple[torch.Tensor, torch.Tensor]:
        if self.embeddings is None:
            self.train(graph_data)

        print(f"Clustering GAE embeddings into {k} roles using KMeans...")
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10, verbose=0)
        role_labels = kmeans.fit_predict(self.embeddings.cpu().numpy())
        
        return self.embeddings, torch.from_numpy(role_labels).int()