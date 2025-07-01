import networkx as nx
import numpy as np
from math import log

class HeuristicFeatureExtractor:
    def calculate_features(self, G: nx.Graph, edges: list) -> np.ndarray:
        """
        Calculates a set of heuristic features for a given list of edges.
        """
        if not edges:
            return np.empty((0, 6))

        all_features = []
        
        print("Pre-computing graph features...")
        try:
            communities = list(nx.community.louvain_communities(G))
            node_to_community = {node: i for i, comm in enumerate(communities) for node in comm}
            print(f"Found {len(communities)} communities.")
        except ImportError:
            node_to_community = {}
            print("Warning: python-louvain is not installed. The 'Same Community' feature will be 0.")

        jaccard_preds = { (u,v): p for u, v, p in nx.jaccard_coefficient(G, edges) }
        ra_preds = { (u,v): p for u, v, p in nx.resource_allocation_index(G, edges) }
        pa_preds = { (u,v): p for u, v, p in nx.preferential_attachment(G, edges) }
        
        print("Extracting features for each edge...")
        for u, v in edges:
            if u not in G or v not in G:
                all_features.append(np.zeros(6)) 
                continue

            common_neighbors = list(nx.common_neighbors(G, u, v))
            
            # Feature 1: Common Neighbors Score
            cn_score = len(common_neighbors)

            # Feature 2: Jaccard Coefficient Score (from pre-computed values)
            jaccard_score = jaccard_preds.get((u, v), 0.0)

            # Feature 3: Adamic-Adar Index (requires degrees of common neighbors)
            aa_score = sum(1 / log(G.degree(w)) for w in common_neighbors if G.degree(w) > 1)

            # Feature 4: Preferential Attachment Score (from pre-computed values)
            pa_score = pa_preds.get((u, v), 0.0)

            # Feature 5: Resource Allocation Index (from pre-computed values)
            ra_score = ra_preds.get((u, v), 0.0)

            # Feature 6: Same Community
            community_u = node_to_community.get(u, -1)
            community_v = node_to_community.get(v, -2)
            same_community = 1.0 if (community_u != -1 and community_u == community_v) else 0.0

            all_features.append([cn_score, jaccard_score, aa_score, pa_score, ra_score, same_community])
        
        return np.array(all_features)