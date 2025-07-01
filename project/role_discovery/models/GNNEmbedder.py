import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans
from torch_geometric.data import Data
from torch_geometric.nn import GAE, GCNConv
import torch_geometric.transforms as T
from pathlib import Path
from torch_geometric.utils import degree, negative_sampling
import copy
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
                 epochs: int = 500, lr: float = 0.01,
                 val_ratio: float = 0.1, test_ratio: float = 0.05,
                 patience: int = 20, force_retrain: bool = False,
                 model_path: str = None, seed: int = 42):
        self.epochs = epochs
        self.lr = lr
        self.hidden_channels = hidden_channels
        self.emb_dim = emb_dim
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.patience = patience
        self.model = None
        self.embeddings = None
        self.force_retrain = force_retrain
        self.model_path = model_path
        self.seed = seed
        print(f"Initialized GNN Embedder (GAE) with params: hidden_channels={hidden_channels}, emb_dim={emb_dim}, lr={lr}, seed={seed}.")


    def train(self, graph_data: Data):
        deg = degree(graph_data.edge_index[0], graph_data.num_nodes).view(-1, 1)
        transform = T.NormalizeFeatures()
        structural_data = Data(x=deg, edge_index=graph_data.edge_index)
        structural_data = transform(structural_data)

        in_channels = structural_data.num_features
        encoder = GCNEncoder(in_channels, self.hidden_channels, self.emb_dim)
        self.model = GAE(encoder)

        if not self.force_retrain and self.model_path and Path(self.model_path).exists():
            print(f"Loading pre-trained model from {self.model_path}")
            self.model.load_state_dict(torch.load(self.model_path))
            with torch.no_grad():
                self.model.eval()
                self.embeddings = self.model.encode(structural_data.x, structural_data.edge_index).detach()
            print("Embeddings generated from pre-trained model.")
            return

        print(f"Starting GAE training with validation and early stopping (seed={self.seed})...")
        torch.manual_seed(self.seed)
        split_transform = T.RandomLinkSplit(
            num_val=self.val_ratio, num_test=self.test_ratio,
            is_undirected=True, add_negative_train_samples=True,
            split_labels=True  
        )
        train_data, val_data, _ = split_transform(structural_data)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        best_val_auc, patience_counter = 0, 0
        best_model_state = None

        for epoch in range(1, self.epochs + 1):
            self.model.train()
            optimizer.zero_grad()
            z = self.model.encode(train_data.x, train_data.edge_index)
            loss = self.model.recon_loss(z, train_data.pos_edge_label_index)
            loss.backward()
            optimizer.step()

            self.model.eval()
            with torch.no_grad():
                z = self.model.encode(val_data.x, val_data.edge_index)
                val_auc, _ = self.model.test(z, val_data.pos_edge_label_index, val_data.neg_edge_label_index)

            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_model_state = copy.deepcopy(self.model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1

            if epoch % 10 == 0:
                print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Val AUC: {val_auc:.4f}')

            if patience_counter >= self.patience:
                print(f"Early stopping at epoch {epoch}.")
                break

        if best_model_state:
            print(f"Training finished. Loading best model with Val AUC: {best_val_auc:.4f}")
            self.model.load_state_dict(best_model_state)
            if self.model_path:
                Path(self.model_path).parent.mkdir(parents=True, exist_ok=True)
                torch.save(best_model_state, self.model_path)
                print(f'Best model saved to {self.model_path}')
        else:
            print("Warning: Training finished, but no best model was saved.")

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