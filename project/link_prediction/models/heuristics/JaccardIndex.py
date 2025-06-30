import torch
import networkx as nx
from torch_geometric.utils import to_networkx
from torch_geometric.data import Data
from ..LinkPredModel import LinkPredictionModel

class JaccardIndexModel(LinkPredictionModel):
    def __init__(self):
        print("Initialized Jaccard Index Model.")

    def train(self, train_data: Data = None, val_data: Data = None):
        print("Jaccard Index is a heuristic model and does not require training.")
        pass

    def predict(self, graph_data: Data, edges_to_predict: torch.Tensor) -> torch.Tensor:
        nx_graph = to_networkx(graph_data, to_undirected=True)
        edge_list = edges_to_predict.t().tolist()
        predictions = nx.jaccard_coefficient(nx_graph, edge_list)
        scores = [p for _, _, p in predictions]
        return torch.tensor(scores, dtype=torch.float)