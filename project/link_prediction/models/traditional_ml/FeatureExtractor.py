import networkx as nx
import numpy as np
from math import log

class HeuristicFeatureExtractor:
    """
    Calculates heuristic-based features for link prediction.
    This class can be shared across multiple traditional ML models.
    """
    def calculate_features(self, G: nx.Graph, edges: list) -> np.ndarray:
        all_features = []
        
        # Pre-calculate degrees for efficiency
        degrees = dict(G.degree())

        for u, v in edges:
            # For nodes not in the graph, features are zero.
            if u not in G or v not in G:
                all_features.append(np.zeros(4)) 
                continue
            
            common_neighbors = list(nx.common_neighbors(G, u, v))

            # Feature 1: Common Neighbors Score
            cn_score = len(common_neighbors)

            # Feature 2: Jaccard Coefficient Score
            try:
                jaccard_preds = nx.jaccard_coefficient(G, [(u, v)])
                _, _, jaccard_score = next(jaccard_preds)
            except StopIteration:
                 jaccard_score = 0.0

            # Feature 3: Adamic-Adar Index (Robust Implementation)
            aa_score = sum(1 / log(degrees[w]) for w in common_neighbors if degrees.get(w, 1) > 1)

            # Feature 4: Preferential Attachment Score
            pa_score = degrees.get(u, 0) * degrees.get(v, 0)

            all_features.append([cn_score, jaccard_score, aa_score, pa_score])
        
        return np.array(all_features)