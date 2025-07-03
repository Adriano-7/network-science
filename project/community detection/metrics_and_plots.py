# metrics_and_plotting.py

import networkx as nx
import community as co 
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, fowlkes_mallows_score
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import os

def calculate_internal_metrics(G, partition):
    """
    Calculates Modularity and average Conductance.
    Conductance is manually calculated as: cut_edges / (internal_edges + cut_edges)
    """
    modularity = co.modularity(partition, G)

    conductances = []
    communities_sets = {}
    for node, comm_id in partition.items():
        if comm_id not in communities_sets:
            communities_sets[comm_id] = set()
        communities_sets[comm_id].add(node)

    for comm_id, nodes_in_comm in communities_sets.items():
        if len(nodes_in_comm) > 0: 
            internal_edges_count = G.subgraph(nodes_in_comm).number_of_edges()
            cut_edges_count = len(list(nx.edge_boundary(G, nodes_in_comm)))

            denominator = internal_edges_count + cut_edges_count
            
            if denominator == 0:
                cond = 1.0 
            else:
                cond = cut_edges_count / denominator
            
            conductances.append(cond)
        else:
            conductances.append(np.nan) 

    avg_conductance = np.mean(conductances) if conductances else np.nan

    return modularity, avg_conductance

def calculate_external_metrics(ground_truth, predicted_partition):
    """Calculates NMI, ARI, and Fowlkes-Mallows Score."""
    predicted_labels = []
    for i in range(len(ground_truth)):
        predicted_labels.append(predicted_partition.get(i, -1))

    nmi = normalized_mutual_info_score(ground_truth, predicted_labels)
    ari = adjusted_rand_score(ground_truth, predicted_labels)
    fms = fowlkes_mallows_score(ground_truth, predicted_labels)
    return nmi, ari, fms

def plot_graph_communities(G, partition, title, figsize=(8, 8)):
    plt.figure(figsize=figsize)
    
    node_ids = list(G.nodes())
    valid_partition_nodes = [node for node in node_ids if node in partition]
    
    if not valid_partition_nodes:
        plt.close()
        return

    unique_communities = sorted(list(set(partition[node] for node in valid_partition_nodes)))
    colors = plt.cm.get_cmap('viridis', max(len(unique_communities), 1)) 
    
    node_colors = []
    for node_id in node_ids:
        if node_id in partition:
            try:
                color_idx = unique_communities.index(partition[node_id])
                node_colors.append(colors(color_idx))
            except ValueError:
                node_colors.append('lightgray') 
        else:
            node_colors.append('lightgray') 

    pos = nx.spring_layout(G, seed=42) 
    
    with_labels = G.number_of_nodes() <= 50 

    nx.draw_networkx(G, 
                     pos=pos, 
                     with_labels=with_labels, 
                     node_color=node_colors, 
                     node_size=300 if with_labels else 100, 
                     font_size=8, 
                     font_color='black',
                     edge_color='gray',
                     width=0.5)
    
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    os.makedirs('plots', exist_ok=True)
    plt.savefig(f"plots/{title.replace(' ', '_').replace(':', '').replace('/', '_')}_graph.png", dpi=300)
    plt.close()

def plot_embeddings_2d(embeddings, labels, title, figsize=(8, 8)):
    plt.figure(figsize=figsize)
    
    if len(embeddings) <= 1:
        plt.close()
        return

    perplexity_val = min(30, len(embeddings) - 1)
    if perplexity_val < 1: 
        plt.close()
        return

    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity_val, max_iter=1000) 
    embeddings_2d = tsne.fit_transform(embeddings)

    labels_int = labels.astype(int) if isinstance(labels, np.ndarray) else np.array(labels).astype(int)
    unique_labels = sorted(list(np.unique(labels_int)))
    colors = plt.cm.get_cmap('viridis', max(len(unique_labels), 1)) 
    
    legend_handles = []
    for i, label in enumerate(unique_labels):
        indices = labels_int == label
        if np.any(indices):
            plt.scatter(embeddings_2d[indices, 0], embeddings_2d[indices, 1], 
                        color=colors(i), label=f'Community {label}', alpha=0.7, s=50) 
            legend_handles.append(plt.Line2D([0], [0], marker='o', color='w', 
                                             markerfacecolor=colors(i), markersize=10, label=f'Community {label}'))
    
    plt.title(title)
    if len(legend_handles) > 0:
        plt.legend(handles=legend_handles, loc='best', fontsize='small', bbox_to_anchor=(1.05, 1), borderaxespad=0.)
    plt.grid(True)
    plt.tight_layout(rect=[0, 0, 0.85, 1]) 
    os.makedirs('plots', exist_ok=True)
    plt.savefig(f"plots/{title.replace(' ', '_').replace(':', '').replace('/', '_')}_embeddings.png", dpi=300)
    plt.close()