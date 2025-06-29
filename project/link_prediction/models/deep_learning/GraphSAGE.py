import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv
from sklearn.metrics import roc_auc_score
from ..LinkPredModel import LinkPredictionModel

if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

class GraphSAGEEncoder(torch.nn.Module):
    """
    GraphSAGE-based encoder.
    GraphSAGE is an inductive model that learns aggregator functions, allowing it to
    generate embeddings for nodes not seen during training[cite: 42, 44, 45].
    """
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.5):
        super().__init__()
        self.dropout = dropout
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x

class MLPDecoder(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels=1, dropout=0.5):
        super().__init__()
        self.lin1 = torch.nn.Linear(2 * in_channels, hidden_channels)
        self.lin2 = torch.nn.Linear(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, z, edge_label_index):
        z_src = z[edge_label_index[0]]
        z_dst = z[edge_label_index[1]]
        x = torch.cat([z_src, z_dst], dim=-1)
        x = self.lin1(x).relu()
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)
        return x.squeeze()

class GraphSAGEModel(LinkPredictionModel):
    def __init__(self, in_channels: int, hidden_channels: int = 128, out_channels: int = 64, 
                 epochs: int = 100, lr: float = 0.01, dropout: float = 0.5):
        
        print("Initialized GraphSAGE Link Prediction Model.")
        print(f"Using device: {device}")
        self.epochs = epochs
        
        self.model = torch.nn.Module()
        self.model.encoder = GraphSAGEEncoder(in_channels, hidden_channels, out_channels, dropout)
        self.model.decoder = MLPDecoder(out_channels, hidden_channels, 1, dropout)
        self.model.to(device)

        self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=lr)
        self.criterion = torch.nn.BCEWithLogitsLoss()

    def train(self, train_data: Data, val_data: Data):
        print("Starting GraphSAGE training...")
        train_data = train_data.to(device)
        val_data = val_data.to(device)
        
        for epoch in range(1, self.epochs + 1):
            self.model.train()
            self.optimizer.zero_grad()
            
            z = self.model.encoder(train_data.x, train_data.edge_index)
            
            train_edges = torch.cat([train_data.pos_edge_label_index, train_data.neg_edge_label_index], dim=-1)
            train_labels = torch.cat([train_data.pos_edge_label, train_data.neg_edge_label], dim=0)

            out = self.model.decoder(z, train_edges)
            loss = self.criterion(out, train_labels)
            
            loss.backward()
            self.optimizer.step()

            if epoch % 10 == 0:
                val_auc = self.test_on_data(val_data)
                print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Val AUC: {val_auc:.4f}')

    @torch.no_grad()
    def test_on_data(self, data: Data) -> float:
        self.model.eval()
        z = self.model.encoder(data.x, data.edge_index)
        val_edges = torch.cat([data.pos_edge_label_index, data.neg_edge_label_index], dim=-1)
        val_labels = torch.cat([data.pos_edge_label, data.neg_edge_label], dim=0)
        out = self.model.decoder(z, val_edges).sigmoid()
        return roc_auc_score(val_labels.cpu().numpy(), out.cpu().numpy())

    def predict(self, graph_data: Data, edges_to_predict: torch.Tensor) -> torch.Tensor:
        print("Generating predictions with trained GraphSAGE...")
        self.model.eval()
        with torch.no_grad():
            graph_data = graph_data.to(device)
            edges_to_predict = edges_to_predict.to(device)
            z = self.model.encoder(graph_data.x, graph_data.edge_index)
            scores = self.model.decoder(z, edges_to_predict)
            return scores.sigmoid().cpu()