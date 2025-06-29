import torch
import networkx as nx
from torch_geometric.utils import to_networkx
from torch_geometric.data import Data
from ..LinkPredModel import LinkPredictionModel

class PreferentialAttachmentModel(LinkPredictionModel):
    def __init__(self):
        print("Initialized Preferential Attachment Model.")

    def train(self, train_data: Data = None, val_data: Data = None):
        print("Preferential Attachment is a heuristic model and does not require training.")
        pass

    def predict(self, graph_data: Data, edges_to_predict: torch.Tensor) -> torch.Tensor:
        if edges_to_predict.size(1) == 0:
            return torch.empty(0, dtype=torch.float)

        try:
            nx_graph = to_networkx(graph_data, to_undirected=True)
            if len(nx_graph.nodes()) == 0:
                return torch.zeros(edges_to_predict.size(1), dtype=torch.float)

            edge_list = edges_to_predict.t().tolist()

            predictions = nx.preferential_attachment(nx_graph, edge_list)
            scores = [p for _, _, p in predictions]

            return torch.tensor(scores, dtype=torch.float)

        except Exception as e:
            print(f"Error in Preferential Attachment prediction: {e}")
            return torch.zeros(edges_to_predict.size(1), dtype=torch.float)