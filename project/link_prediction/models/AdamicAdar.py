import torch
import networkx as nx
from torch_geometric.utils import to_networkx
from torch_geometric.data import Data
from .LinkPredictionModel import LinkPredictionModel

class AdamicAdarModel(LinkPredictionModel):

    def __init__(self):
        print("Initialized Adamic-Adar Model.")

    def train(self, train_data: Data, val_data: Data = None):
        print("Adamic-Adar is a heuristic model and does not require training.")
        pass

    def predict(self, graph_data: Data, edges_to_predict: torch.Tensor) -> torch.Tensor:
        # 1. Convert the graph data to a NetworkX graph for the calculation
        nx_graph = to_networkx(graph_data, to_undirected=True)
        
        # 2. Get the list of edges to score
        edge_list = edges_to_predict.t().tolist()
        
        # 3. Use the NetworkX implementation of Adamic-Adar
        predictions = nx.adamic_adar_index(nx_graph, edge_list)

        # 4. Extract just the scores in the correct order
        scores = [p for _, _, p in predictions]
        
        return torch.tensor(scores, dtype=torch.float)