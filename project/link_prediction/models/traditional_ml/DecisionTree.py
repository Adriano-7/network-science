import torch
import networkx as nx
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx
from math import log
from ..LinkPredModel import LinkPredictionModel

class DecisionTreeModel(LinkPredictionModel):
    def __init__(self, **kwargs):
        print("Initialized Decision Tree Link Prediction Model.")
        if 'random_state' not in kwargs:
            kwargs['random_state'] = 42
            
        self.model = DecisionTreeClassifier(**kwargs)
        self.nx_graph = None
        self._is_trained = False

    def _calculate_features(self, G: nx.Graph, edges: list) -> np.ndarray:
        all_features = []
        degrees = dict(G.degree())

        for u, v in edges:
            if u not in G or v not in G:
                all_features.append(np.zeros(4)) 
                continue
            
            common_neighbors = list(nx.common_neighbors(G, u, v))
            cn_score = len(common_neighbors)

            try:
                jaccard_preds = nx.jaccard_coefficient(G, [(u, v)])
                _, _, jaccard_score = next(jaccard_preds)
            except StopIteration:
                 jaccard_score = 0.0

            aa_score = sum(1 / log(degrees[w]) for w in common_neighbors if degrees[w] > 1)

            pa_score = degrees.get(u, 0) * degrees.get(v, 0)
            all_features.append([cn_score, jaccard_score, aa_score, pa_score])
        
        return np.array(all_features)

    def train(self, train_data: Data, val_data: Data = None):
        print("Training Decision Tree model...")

        self.nx_graph = to_networkx(train_data, to_undirected=True, node_attrs=None)

        pos_train_edges = train_data.pos_edge_label_index.t().cpu().numpy().tolist()
        neg_train_edges = train_data.neg_edge_label_index.t().cpu().numpy().tolist()

        print(f"Generating features for {len(pos_train_edges)} positive and {len(neg_train_edges)} negative training samples...")
        
        X_pos = self._calculate_features(self.nx_graph, pos_train_edges)
        X_neg = self._calculate_features(self.nx_graph, neg_train_edges)

        X_train = np.vstack([X_pos, X_neg])
        y_train = np.hstack([np.ones(X_pos.shape[0]), np.zeros(X_neg.shape[0])])

        print("Fitting the DecisionTreeClassifier...")
        self.model.fit(X_train, y_train)
        self._is_trained = True
        print("Training complete.")

    def predict(self, graph_data: Data, edges_to_predict: torch.Tensor) -> torch.Tensor:
        if not self._is_trained:
            raise RuntimeError("Model has not been trained yet. Call train() first.")
        
        print(f"Predicting on {edges_to_predict.size(1)} edges using Decision Tree...")
        if self.nx_graph is None:
             self.nx_graph = to_networkx(graph_data, to_undirected=True)
        
        edge_list = edges_to_predict.t().cpu().numpy().tolist()
        X_test = self._calculate_features(self.nx_graph, edge_list)
        probs = self.model.predict_proba(X_test)[:, 1]
        return torch.tensor(probs, dtype=torch.float)