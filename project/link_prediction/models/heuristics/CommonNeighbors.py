import torch
import networkx as nx
from torch_geometric.utils import to_networkx
from torch_geometric.data import Data
from ..LinkPredModel import LinkPredictionModel

class CommonNeighborsModel(LinkPredictionModel):
    def __init__(self):
        print("Initialized Common Neighbors Model.")

    def train(self, train_data: Data = None, val_data: Data = None):
        print("Common Neighbors is a heuristic model and does not require training.")
        pass

    def predict(self, graph_data: Data, edges_to_predict: torch.Tensor) -> torch.Tensor:
        if edges_to_predict.size(1) == 0:
            return torch.empty(0, dtype=torch.float)

        try:
            nx_graph = to_networkx(graph_data, to_undirected=True)
            if len(nx_graph.nodes()) == 0:
                return torch.zeros(edges_to_predict.size(1), dtype=torch.float)

            edge_list = edges_to_predict.t().tolist()
            scores = [len(list(nx.common_neighbors(nx_graph, u, v))) for u, v in edge_list]
            
            return torch.tensor(scores, dtype=torch.float)

        except Exception as e:
            print(f"Error in Common Neighbors prediction: {e}")
            return torch.zeros(edges_to_predict.size(1), dtype=torch.float)