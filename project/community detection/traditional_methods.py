import networkx as nx
import community as co 

def run_louvain(G):
    """Runs the Louvain community detection algorithm."""
    partition = co.best_partition(G)
    return partition

def run_girvan_newman(G):
    """
    Runs the Girvan-Newman community detection algorithm.
    WARNING: Very slow for large graphs. Only suitable for small graphs like KarateClub.
    """
    comp = nx.community.girvan_newman(G)

    best_partition = None
    max_modularity = -1.0
    
    iteration_limit = 2 * G.number_of_nodes() 
    if G.number_of_nodes() < 50: 
        iteration_limit = 5 * G.number_of_nodes()
    
    current_iteration = 0
    
    try:
        for communities_tuple in comp:
            current_iteration += 1
            if current_iteration > iteration_limit:
                break

            current_partition = {}
            for comm_id, community_set in enumerate(communities_tuple):
                for node in community_set:
                    current_partition[node] = comm_id
            
            if not current_partition:
                continue

            current_modularity = co.modularity(current_partition, G)

            if current_modularity > max_modularity:
                max_modularity = current_modularity
                best_partition = current_partition
                
    except Exception as e:
        return {} 

    if best_partition is None:
        if G.number_of_nodes() > 0:
             partition = {node: 0 for node in G.nodes()} 
             return partition
        return {} 
    return best_partition


def run_label_propagation(G):
    """Runs the Label Propagation community detection algorithm."""
    communities = nx.community.label_propagation_communities(G)
    
    partition = {}
    for comm_id, community_set in enumerate(communities):
        for node in community_set:
            partition[node] = comm_id
    return partition