import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans
from torch_geometric.data import Data
from torch_geometric.nn import GAE, GCNConv
import torch_geometric.transforms as T
import matplotlib.pyplot as plt
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
                     patience: int = 20): 
        print("Initialized GNN Embedder (GAE) for Role Discovery.")
        self.epochs = epochs
        self.lr = lr
        self.hidden_channels = hidden_channels
        self.emb_dim = emb_dim
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.patience = patience 
        self.model = None
        self.embeddings = None
        self.train_losses = []
        self.val_aucs = []

    def train(self, graph_data: Data):
        print("Starting GAE training with validation and early stopping...")

        deg = degree(graph_data.edge_index[0], graph_data.num_nodes).view(-1, 1)
        
        transform = T.NormalizeFeatures()
        structural_data = Data(x=deg, edge_index=graph_data.edge_index)
        structural_data = transform(structural_data)

        split_transform = T.RandomLinkSplit(
            num_val=self.val_ratio,
            num_test=self.test_ratio,
            is_undirected=True,
            add_negative_train_samples=True
        )
        train_data, val_data, test_data = split_transform(structural_data)

        in_channels = train_data.num_features
        encoder = GCNEncoder(in_channels, self.hidden_channels, self.emb_dim)
        self.model = GAE(encoder)
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        
        best_val_auc = 0
        best_model_state = None
        patience_counter = 0 

        for epoch in range(1, self.epochs + 1):
            self.model.train()
            optimizer.zero_grad()
            z = self.model.encode(train_data.x, train_data.edge_index)
            loss = self.model.recon_loss(z, train_data.edge_label_index)
            loss.backward()
            optimizer.step()
            
            self.train_losses.append(loss.item())

            self.model.eval()
            with torch.no_grad():
                z = self.model.encode(val_data.x, val_data.edge_index)
                if not hasattr(val_data, 'neg_edge_label_index'):
                    num_neg_edges = val_data.edge_label_index.size(1)
                    neg_edge_index = negative_sampling(
                        edge_index=val_data.edge_index,
                        num_nodes=val_data.num_nodes,
                        num_neg_samples=num_neg_edges,
                    )
                    val_auc, _ = self.model.test(z, val_data.edge_label_index, neg_edge_index)
                else:
                    val_auc, _ = self.model.test(z, val_data.edge_label_index, val_data.neg_edge_label_index)
                self.val_aucs.append(val_auc)

            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_model_state = copy.deepcopy(self.model.state_dict()) 
                patience_counter = 0
            else:
                patience_counter += 1
            
            if epoch % 10 == 0: 
                print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Val AUC: {val_auc:.4f}')

            if patience_counter >= self.patience:
                print(f"Early stopping at epoch {epoch} due to no improvement in Val AUC for {self.patience} checks.")
                break
        
        if best_model_state:
            print(f"Training finished. Loading best model with Val AUC: {best_val_auc:.4f}")
            self.model.load_state_dict(best_model_state)
        else:
            print("Warning: Training finished, but no best model was saved. Using the last state.")

        if best_model_state is not None:
            project_root = Path(__file__).resolve().parents[2]
            model_save_path = project_root / 'best_gae_model.pt'
            torch.save(best_model_state, model_save_path)
            print(f'Best model saved to {model_save_path} with Val AUC: {best_val_auc:.4f}')

        fig, ax1 = plt.subplots(figsize=(12, 6))
        ax1.plot(self.train_losses, 'b-', label='Training Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss', color='b')
        ax1.tick_params('y', colors='b')

        ax2 = ax1.twinx()
        ax2.plot(range(1, len(self.val_aucs) + 1), self.val_aucs, 'r-', label='Validation AUC')
        ax2.set_ylabel('AUC', color='r')
        ax2.tick_params('y', colors='r')

        fig.tight_layout()
        plt.title('GAE Training Metrics')
        plot_save_path = project_root / 'training_metrics.png'
        plt.savefig(plot_save_path)
        print(f'Training metrics plot saved to {plot_save_path}')
        plt.close()

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