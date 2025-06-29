import torch
import networkx as nx
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx
from ..LinkPredModel import LinkPredictionModel
from .FeatureExtractor import HeuristicFeatureExtractor

class LogisticRegressionModel(LinkPredictionModel):
    def __init__(self, **kwargs):
        print("Initialized Logistic Regression Link Prediction Model.")
        if 'random_state' not in kwargs:
            kwargs['random_state'] = 42
        if 'solver' not in kwargs:
            kwargs['solver'] = 'liblinear'

        self.model = LogisticRegression(**kwargs)
        self.feature_extractor = HeuristicFeatureExtractor()
        self.scaler = StandardScaler()
        self.nx_graph = None
        self._is_trained = False

    def train(self, train_data: Data, val_data: Data = None):
        print("Training Logistic Regression model...")
        self.nx_graph = to_networkx(train_data, to_undirected=True, node_attrs=None)

        pos_train_edges = train_data.pos_edge_label_index.t().cpu().numpy().tolist()
        neg_train_edges = train_data.neg_edge_label_index.t().cpu().numpy().tolist()

        print(f"Generating features for {len(pos_train_edges)} positive and {len(neg_train_edges)} negative training samples...")
        
        X_pos = self.feature_extractor.calculate_features(self.nx_graph, pos_train_edges)
        X_neg = self.feature_extractor.calculate_features(self.nx_graph, neg_train_edges)

        X_train = np.vstack([X_pos, X_neg])
        y_train = np.hstack([np.ones(X_pos.shape[0]), np.zeros(X_neg.shape[0])])

        print("Scaling features and fitting the LogisticRegression model...")
        # Fit the scaler on the training data and transform it
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Train the model on the scaled data
        self.model.fit(X_train_scaled, y_train)
        self._is_trained = True
        print("Training complete.")

    def predict(self, graph_data: Data, edges_to_predict: torch.Tensor) -> torch.Tensor:
        if not self._is_trained:
            raise RuntimeError("Model has not been trained yet. Call train() first.")

        print(f"Predicting on {edges_to_predict.size(1)} edges using Logistic Regression...")
        if self.nx_graph is None:
            self.nx_graph = to_networkx(graph_data, to_undirected=True)
        
        edge_list = edges_to_predict.t().cpu().numpy().tolist()
        
        X_test = self.feature_extractor.calculate_features(self.nx_graph, edge_list)
        X_test_scaled = self.scaler.transform(X_test)
        probs = self.model.predict_proba(X_test_scaled)[:, 1]

        return torch.tensor(probs, dtype=torch.float)