import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv
from ..LinkPredModel import LinkPredictionModel
import copy

if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

class GraphSAGEEncoder(torch.nn.Module):
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

class DotProductDecoder(torch.nn.Module):
    def forward(self, z, edge_label_index):
        z_src = z[edge_label_index[0]]
        z_dst = z[edge_label_index[1]]
        return (z_src * z_dst).sum(dim=-1)

class GraphSAGEModel2(LinkPredictionModel):
    def __init__(self, in_channels: int, hidden_channels: int, emb_dim: int,
                 epochs: int, lr: float, dropout: float, patience: int):
        
        print("Initialized GraphSAGE Link Prediction Model.")
        print(f"Using device: {device}")
        self.epochs = epochs
        self.patience = patience
        self.best_val_mrr = 0
        self.best_model_state = None       
        self.model = torch.nn.Module()
        self.model.encoder = GraphSAGEEncoder(in_channels, hidden_channels, emb_dim, dropout)
        self.model.decoder = DotProductDecoder()
        self.model.to(device)

        self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=lr)
        self.criterion = torch.nn.BCEWithLogitsLoss()

    def train(self, train_data: Data, val_data: Data):
        print("Starting GraphSAGE training with early stopping based on Val MRR...")
        train_data = train_data.to(device)
        val_data = val_data.to(device)
        
        patience_counter = 0

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
                val_mrr = self._calculate_mrr_for_validation(train_data, val_data)
                print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Val MRR: {val_mrr:.4f}')

                # --- Early stopping logic based on MRR ---
                if val_mrr > self.best_val_mrr:
                    self.best_val_mrr = val_mrr
                    self.best_model_state = copy.deepcopy(self.model.state_dict())
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                if patience_counter >= self.patience:
                    print(f"Early stopping at epoch {epoch} due to no improvement in Val MRR for {self.patience} checks.")
                    break
        
        if self.best_model_state:
            print(f"Training finished. Loading best model with Val MRR: {self.best_val_mrr:.4f}")
            self.model.load_state_dict(self.best_model_state)
        else:
            print("Warning: Training finished, but no best model was saved. Using the last state.")

    @torch.no_grad()
    def _calculate_mrr_for_validation(self, train_data_for_emb: Data, val_data_to_eval: Data) -> float:
        self.model.eval()
        z = self.model.encoder(train_data_for_emb.x, train_data_for_emb.edge_index)
        
        val_edges = torch.cat([val_data_to_eval.pos_edge_label_index, val_data_to_eval.neg_edge_label_index], dim=-1)
        val_labels = torch.cat([val_data_to_eval.pos_edge_label, val_data_to_eval.neg_edge_label], dim=0)
        
        scores = self.model.decoder(z, val_edges).sigmoid()

        y_true = val_labels.cpu()
        y_pred = scores.cpu()
        
        pos_scores = y_pred[y_true == 1]
        neg_scores = y_pred[y_true == 0]

        if pos_scores.numel() == 0 or neg_scores.numel() == 0:
            return 0.0

        ranks = (pos_scores.view(-1, 1) <= neg_scores.view(1, -1)).float().sum(dim=1) + 1
        mrr = (1.0 / ranks).mean().item()
        
        return mrr

    def predict_edges(self, graph_data: Data, edges_to_predict: torch.Tensor) -> torch.Tensor:
        print("Generating predictions with trained GraphSAGE...")
        self.model.eval()
        with torch.no_grad():
            graph_data = graph_data.to(device)
            edges_to_predict = edges_to_predict.to(device)
            z = self.model.encoder(graph_data.x, graph_data.edge_index)
            scores = self.model.decoder(z, edges_to_predict)
            return scores.sigmoid().cpu()