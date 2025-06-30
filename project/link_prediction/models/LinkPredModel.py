from abc import ABC, abstractmethod
from torch_geometric.data import Data
import torch
from typing import Tuple

class LinkPredictionModel(ABC):

    @abstractmethod
    def train(self, train_data: Data = None, val_data: Data = None):
        pass

    @abstractmethod
    def predict_edges(self, graph_data: Data, edges_to_predict: torch.Tensor) -> torch.Tensor:
        pass
    
    def recommend_connections(self, graph_data: Data, node: int, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
        num_nodes = graph_data.num_nodes
        assert 0 <= node < num_nodes, f"Node index {node} is out of bounds for graph with {num_nodes} nodes."
        assert k > 0, "The number of recommendations (k) must be greater than 0."

        # 1. Identify all nodes currently connected to the target `node`.
        edge_index = graph_data.edge_index
        neighbors_row = edge_index[1][edge_index[0] == node]
        neighbors_col = edge_index[0][edge_index[1] == node]
        existing_neighbors = torch.unique(torch.cat([neighbors_row, neighbors_col]))

        # 2. Determine candidate nodes for new connections.
        all_nodes = torch.arange(num_nodes, device=edge_index.device)
        is_candidate = torch.ones(num_nodes, dtype=torch.bool, device=edge_index.device)
        is_candidate[node] = False
        if existing_neighbors.numel() > 0:
            is_candidate[existing_neighbors] = False
        
        candidate_nodes = all_nodes[is_candidate]

        if candidate_nodes.numel() == 0:
            print(f"Warning: Node {node} has no potential new connections.")
            return torch.tensor([], dtype=torch.long), torch.tensor([], dtype=torch.float)

        # 3. Formulate all potential new edges to be scored.
        source_nodes = torch.full_like(candidate_nodes, fill_value=node)
        edges_to_score = torch.stack([source_nodes, candidate_nodes], dim=0)

        # 4. Use the model's `predict_edges` method to get scores.
        scores = self.predict_edges(graph_data=graph_data, edges_to_predict=edges_to_score)

        # 5. Identify the top k recommendations from the scored candidates.
        k = min(k, candidate_nodes.numel())
        top_k_scores, top_k_indices = torch.topk(scores, k=k)
        top_k_nodes = candidate_nodes[top_k_indices]

        return top_k_nodes, top_k_scores