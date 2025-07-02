import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from torch_geometric.utils import k_hop_subgraph, to_networkx
from torch_geometric.nn import GCNConv, global_mean_pool
import networkx as nx
import copy
from ..LinkPredModel import LinkPredictionModel

if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

class SEAL_GNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, emb_dim, dropout):
        super().__init__()
        self.dropout = dropout
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.lin = torch.nn.Linear(hidden_channels, 1)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        x = self.conv3(x, edge_index)
        x = global_mean_pool(x, batch)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin(x)
        return x

class SEALModel(LinkPredictionModel):
    def __init__(self, in_channels, hidden_channels, emb_dim, num_hops,
                 epochs, lr, dropout, patience, use_feature=True):
        print("Initialized Self-Contained SEAL Link Prediction Model.")
        self.num_hops = num_hops
        self.use_feature = use_feature
        self.epochs = epochs
        self.patience = patience

        self.model = SEAL_GNN(in_channels, hidden_channels, emb_dim, dropout).to(device)
        self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=lr)
        self.criterion = torch.nn.BCEWithLogitsLoss()

    def _extract_subgraph(self, graph_data, edge):
        u, v = edge[0].item(), edge[1].item()
        
        subset, sub_edge_index, mapping, edge_mask = k_hop_subgraph(
            torch.tensor([u, v]), self.num_hops, graph_data.edge_index, relabel_nodes=True)
        
        if subset.numel() == 0:
            return None
            
        sub_x = graph_data.x[subset]
        
        u_new, v_new = mapping[0].item(), mapping[1].item()
        
        g = to_networkx(Data(edge_index=sub_edge_index, num_nodes=subset.size(0)), to_undirected=True)
        
        drnl_label = torch.zeros(subset.size(0), 2, dtype=torch.long)
        try:
            path_u = nx.shortest_path_length(g, source=u_new)
            path_v = nx.shortest_path_length(g, source=v_new)
            for i in range(subset.size(0)):
                drnl_label[i, 0] = path_u.get(i, 0)
                drnl_label[i, 1] = path_v.get(i, 0)
        except nx.NodeNotFound:
            return None

        drnl_label = drnl_label.float() / (self.num_hops + 1)
        
        if self.use_feature and sub_x is not None:
            subgraph_x = torch.cat([sub_x, drnl_label.to(sub_x.device)], dim=-1)
        else:
            subgraph_x = drnl_label
            
        return Data(x=subgraph_x, edge_index=sub_edge_index)

    @torch.no_grad()
    def _calculate_mrr_for_validation(self, val_data):
        self.model.eval()
        pos_edges = val_data.pos_edge_label_index.t()
        neg_edges = val_data.neg_edge_label_index.t()

        pos_subgraphs = [self._extract_subgraph(val_data, edge) for edge in pos_edges]
        neg_subgraphs = [self._extract_subgraph(val_data, edge) for edge in neg_edges]

        pos_subgraphs = [g for g in pos_subgraphs if g is not None]
        neg_subgraphs = [g for g in neg_subgraphs if g is not None]

        if not pos_subgraphs or not neg_subgraphs:
            return 0.0

        pos_loader = DataLoader(pos_subgraphs, batch_size=128)
        neg_loader = DataLoader(neg_subgraphs, batch_size=128)

        pos_scores = []
        for batch in pos_loader:
            batch = batch.to(device)
            pos_scores.append(self.model(batch.x, batch.edge_index, batch.batch).sigmoid())
        pos_scores = torch.cat(pos_scores, dim=0).cpu()

        neg_scores = []
        for batch in neg_loader:
            batch = batch.to(device)
            neg_scores.append(self.model(batch.x, batch.edge_index, batch.batch).sigmoid())
        neg_scores = torch.cat(neg_scores, dim=0).cpu()
        
        ranks = (pos_scores.view(-1, 1) <= neg_scores.view(1, -1)).float().sum(dim=1) + 1
        mrr = (1.0 / ranks).mean().item()
        return mrr

    def train(self, train_data, val_data=None):
        print("Extracting subgraphs for SEAL training...")
        pos_edges = train_data.pos_edge_label_index.t()
        neg_edges = train_data.neg_edge_label_index.t()

        all_edges = torch.cat([pos_edges, neg_edges], dim=0)
        train_subgraphs = [self._extract_subgraph(train_data, edge) for edge in all_edges]
        
        train_labels_list = []
        filtered_subgraphs = []
        for i, g in enumerate(train_subgraphs):
            if g is not None:
                filtered_subgraphs.append(g)
                train_labels_list.append(1.0 if i < pos_edges.size(0) else 0.0)

        train_labels = torch.tensor(train_labels_list, dtype=torch.float)

        if not filtered_subgraphs:
             print("Warning: No valid subgraphs could be extracted for training.")
             return 0.0

        train_loader = DataLoader(
            list(zip(filtered_subgraphs, train_labels)), batch_size=64, shuffle=True)

        best_val_mrr = 0
        patience_counter = 0
        best_model_state = None
        
        print("Starting SEAL model training...")
        for epoch in range(1, self.epochs + 1):
            self.model.train()
            total_loss = 0
            for subgraphs, labels in train_loader:
                subgraphs = subgraphs.to(device)
                labels = labels.to(device)
                
                self.optimizer.zero_grad()
                out = self.model(subgraphs.x, subgraphs.edge_index, subgraphs.batch).squeeze()
                loss = self.criterion(out, labels)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item() * subgraphs.num_graphs

            if epoch % 5 == 0 and val_data is not None:
                val_mrr = self._calculate_mrr_for_validation(val_data)
                print(f"Epoch: {epoch:03d}, Loss: {total_loss / len(train_loader.dataset):.4f}, Val MRR: {val_mrr:.4f}")

                if val_mrr > best_val_mrr:
                    best_val_mrr = val_mrr
                    best_model_state = copy.deepcopy(self.model.state_dict())
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                if patience_counter >= self.patience:
                    print(f"Early stopping at epoch {epoch}.")
                    break
        
        if best_model_state:
            print(f"Training finished. Loading best model with Val MRR: {best_val_mrr:.4f}")
            self.model.load_state_dict(best_model_state)
            return best_val_mrr 
        return 0.0

    def predict_edges(self, graph_data, edges_to_predict):
        self.model.eval()
        edges_list = edges_to_predict.t()
        
        pred_subgraphs = [self._extract_subgraph(graph_data, edge) for edge in edges_list]
        pred_subgraphs = [g for g in pred_subgraphs if g is not None]
        
        pred_loader = DataLoader(pred_subgraphs, batch_size=128)
        
        scores = []
        with torch.no_grad():
            for batch in pred_loader:
                batch = batch.to(device)
                score = self.model(batch.x, batch.edge_index, batch.batch).sigmoid()
                scores.append(score)
        
        if not scores:
            return torch.zeros(edges_to_predict.size(1))
            
        return torch.cat(scores, dim=0).squeeze().cpu()