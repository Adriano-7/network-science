import torch
from sklearn.cluster import KMeans
from torch_geometric.data import Data
from torch_geometric.nn import GAE, GCNConv

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
    """
    Discovers roles by learning node embeddings with a Graph Autoencoder (GAE)
    and then clustering them with KMeans.
    """
    def __init__(self, in_channels: int, hidden_channels: int = 256, emb_dim: int = 64,
                 epochs: int = 200, lr: float = 0.01):
        print("Initialized GNN Embedder (GAE) for Role Discovery.")
        self.epochs = epochs
        self.lr = lr
        
        encoder = GCNEncoder(in_channels, hidden_channels, emb_dim)
        self.model = GAE(encoder)
        
        self.embeddings = None

    def train(self, graph_data: Data):
        """Trains the Graph Autoencoder model on the full graph structure."""
        print("Starting GAE training...")
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        
        x = graph_data.x
        train_pos_edge_index = graph_data.edge_index

        for epoch in range(1, self.epochs + 1):
            self.model.train()
            optimizer.zero_grad()
            
            z = self.model.encode(x, train_pos_edge_index)
            
            loss = self.model.recon_loss(z, train_pos_edge_index)
            
            loss.backward()
            optimizer.step()
            
            if epoch % 20 == 0:
                print(f'Epoch: {epoch:03d}, Reconstruction Loss: {loss:.4f}')
        
        print("GAE training finished.")
        with torch.no_grad():
            self.model.eval()
            self.embeddings = self.model.encode(x, train_pos_edge_index).detach()

    def predict(self, graph_data: Data, k: int) -> tuple[torch.Tensor, torch.Tensor]:
        if self.embeddings is None:
            raise RuntimeError("Model has not been trained. Call train() before predict().")

        print(f"Clustering GAE embeddings into {k} roles using KMeans...")
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        role_labels = kmeans.fit_predict(self.embeddings.cpu().numpy())
        
        return self.embeddings, torch.from_numpy(role_labels).int()