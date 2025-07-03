import torch
import networkx as nx
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx
from ..LinkPredModel import LinkPredictionModel
from .FeatureExtractor import HeuristicFeatureExtractor

class DecisionTreeModel(LinkPredictionModel):
    def __init__(self, **kwargs):
        print("Initialized Decision Tree Link Prediction Model.")
        if 'random_state' not in kwargs:
            kwargs['random_state'] = 42
        
        self.model = DecisionTreeClassifier(**kwargs)
        self.feature_extractor = HeuristicFeatureExtractor()
        self._is_trained = False

    def train(self, train_data: Data, val_data: Data = None):

        print("Training Decision Tree model...")
        print("Building graph for feature extraction using training edges only...")
        graph_for_features = Data(edge_index=train_data.edge_index, num_nodes=train_data.num_nodes)
        nx_graph = to_networkx(graph_for_features, to_undirected=True)
        pos_train_edges = train_data.pos_edge_label_index.t().cpu().numpy().tolist()
        neg_train_edges = train_data.neg_edge_label_index.t().cpu().numpy().tolist()

        print(f"Generating features for {len(pos_train_edges)} positive and {len(neg_train_edges)} negative training samples...")
        
        X_pos = self.feature_extractor.calculate_features(nx_graph, pos_train_edges)
        X_neg = self.feature_extractor.calculate_features(nx_graph, neg_train_edges)

        X_train = np.vstack([X_pos, X_neg])
        y_train = np.hstack([np.ones(X_pos.shape[0]), np.zeros(X_neg.shape[0])])

        print("Fitting the DecisionTreeClassifier...")
        self.model.fit(X_train, y_train)
        self._is_trained = True
        print("Training complete.")

    def predict_edges(self, graph_data: Data, edges_to_predict: torch.Tensor) -> torch.Tensor:
        if not self._is_trained:
            raise RuntimeError("Model has not been trained yet. Call train() first.")
        
        print(f"Predicting on {edges_to_predict.size(1)} edges using Decision Tree...")
        print("Re-building graph for prediction using training edges only...")
        graph_for_features = Data(edge_index=graph_data.edge_index, num_nodes=graph_data.num_nodes)
        nx_graph = to_networkx(graph_for_features, to_undirected=True)
        
        edge_list = edges_to_predict.t().cpu().numpy().tolist()
        X_test = self.feature_extractor.calculate_features(nx_graph, edge_list)
        probs = self.model.predict_proba(X_test)[:, 1]
        return torch.tensor(probs, dtype=torch.float)