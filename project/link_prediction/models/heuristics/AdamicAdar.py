import torch
import networkx as nx
from math import log
from torch_geometric.utils import to_networkx
from torch_geometric.data import Data
from ..LinkPredModel import LinkPredictionModel


def robust_adamic_adar_index(G, ebunch=None):
    if ebunch is None:
        ebunch = nx.non_edges(G)
    def predict(u, v):
        try:
            return sum(1 / log(G.degree(w)) for w in nx.common_neighbors(G, u, v) if G.degree(w) > 1)
        except (ValueError, ZeroDivisionError):
            return 0.0
    return ((u, v, predict(u, v)) for u, v in ebunch)


class AdamicAdarModel(LinkPredictionModel):
    def __init__(self):
        print("Initialized Adamic-Adar Model.")

    def train(self, train_data: Data = None, val_data: Data = None):
        print("Adamic-Adar is a heuristic model and does not require training.")
        pass

    def predict(self, graph_data: Data, edges_to_predict: torch.Tensor) -> torch.Tensor:
        nx_graph = to_networkx(graph_data, to_undirected=True)
        edge_list = edges_to_predict.t().tolist()
        predictions = robust_adamic_adar_index(nx_graph, edge_list)
        scores = [p for _, _, p in predictions]
        return torch.tensor(scores, dtype=torch.float)