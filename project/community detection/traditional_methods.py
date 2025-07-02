import torch
import networkx as nx
import community as co 
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, fowlkes_mallows_score
from torch_geometric.datasets import Planetoid, KarateClub 
from torch_geometric.utils import to_networkx

from torch_geometric.data import Data 
import numpy as np
import pandas as pd
import warnings
import random
import os 
import requests 

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from deep_learning_methods import run_gnn, GCN, GraphSage 

warnings.filterwarnings("ignore")

def get_cora_subgraph_pyg(full_data_pyg, num_nodes_subgraph=150):
    """
    Extracts a connected subgraph of a specified number of nodes from the full Cora graph.
    Returns a PyTorch Geometric Data object with relabeled nodes (0-indexed),
    node features, ground truth, and randomly generated train/val/test masks.
    """
    G_full = to_networkx(full_data_pyg, to_undirected=True)
    
    nodes_in_full_graph = list(G_full.nodes())
    if len(nodes_in_full_graph) < num_nodes_subgraph:
        print(f"Warning: Full graph (Cora) has fewer nodes ({len(nodes_in_full_graph)}) than requested subgraph size ({num_nodes_subgraph}). Returning full graph data.")
        return full_data_pyg

    subgraph_nodes = set()
    
    attempts = 0
    max_attempts = 10 
    while len(subgraph_nodes) < num_nodes_subgraph and attempts < max_attempts:
        attempts += 1
        start_node = random.choice(nodes_in_full_graph)
        current_subgraph_nodes = set()
        queue = [start_node]
        visited = {start_node}

        while queue and len(current_subgraph_nodes) < num_nodes_subgraph:
            current_node = queue.pop(0)
            current_subgraph_nodes.add(current_node)
            for neighbor in G_full.neighbors(current_node):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        
        if len(current_subgraph_nodes) >= num_nodes_subgraph * 0.8: 
            subgraph_nodes = current_subgraph_nodes
            break
        else: 
            subgraph_nodes = set()
            
    if len(subgraph_nodes) < num_nodes_subgraph:
        print(f"Warning: BFS method struggled to find a connected subgraph of {num_nodes_subgraph} nodes after {max_attempts} attempts. Taking a random sample of {num_nodes_subgraph} nodes instead (might be disconnected).")
        subgraph_nodes = set(random.sample(nodes_in_full_graph, num_nodes_subgraph))


    subgraph = G_full.subgraph(subgraph_nodes).copy() 
    
    old_to_new_node_map = {old_id: new_id for new_id, old_id in enumerate(sorted(subgraph_nodes))}
    
    subgraph_edges = []
    for u, v in G_full.edges():
        if u in subgraph_nodes and v in subgraph_nodes:
            subgraph_edges.append([old_to_new_node_map[u], old_to_new_node_map[v]])
    if not subgraph_edges:
        edge_index_subgraph = torch.empty((2, 0), dtype=torch.long)
    else:
        edge_index_subgraph = torch.tensor(subgraph_edges, dtype=torch.long).t().contiguous()

    subgraph_x = torch.stack([full_data_pyg.x[old_node_id] for old_node_id in sorted(subgraph_nodes)])
    subgraph_y = torch.stack([full_data_pyg.y[old_node_id] for old_node_id in sorted(subgraph_nodes)])

    num_nodes_actual = len(subgraph_nodes)
    idx = np.arange(num_nodes_actual)
    np.random.shuffle(idx)

    train_size = int(0.6 * num_nodes_actual)
    val_size = int(0.2 * num_nodes_actual)
    
    train_mask = torch.zeros(num_nodes_actual, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes_actual, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes_actual, dtype=torch.bool)

    train_mask[idx[:train_size]] = True
    val_mask[idx[train_size : train_size + val_size]] = True
    test_mask[idx[train_size + val_size :]] = True

    subgraph_data = Data(x=subgraph_x, edge_index=edge_index_subgraph, y=subgraph_y,
                         train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)
    
    print(f"Created PyG Cora subgraph with {subgraph_data.num_nodes} nodes and {subgraph_data.num_edges} edges.")
    return subgraph_data


# Dataset Loading and Preparation 

def load_and_prepare_dataset(dataset_name):
    """Loads a PyTorch Geometric dataset and prepares it."""
    print(f"\n--- Loading {dataset_name} Dataset ---")
    G, ground_truth = None, None
    data_pyg = None 

    if dataset_name == 'Cora':
        dataset = Planetoid(root='/tmp/Cora', name='Cora')
        data_pyg = dataset[0]
        G = to_networkx(data_pyg, to_undirected=True)
        ground_truth = data_pyg.y.cpu().numpy()
    elif dataset_name == 'CiteSeer':
        dataset = Planetoid(root='/tmp/CiteSeer', name='CiteSeer')
        data_pyg = dataset[0]
        G = to_networkx(data_pyg, to_undirected=True)
        ground_truth = data_pyg.y.cpu().numpy()
    elif dataset_name == 'PubMed':
        dataset = Planetoid(root='/tmp/PubMed', name='PubMed')
        data_pyg = dataset[0]
        G = to_networkx(data_pyg, to_undirected=True)
        ground_truth = data_pyg.y.cpu().numpy()
    elif dataset_name == 'KarateClub':
        dataset = KarateClub()
        data_pyg = dataset[0]
        G = to_networkx(data_pyg, to_undirected=True)
        ground_truth = data_pyg.y.cpu().numpy()
    elif dataset_name == 'Cora_Subset':
        full_dataset = Planetoid(root='/tmp/Cora', name='Cora')
        full_data_pyg = full_dataset[0]
        data_pyg = get_cora_subgraph_pyg(full_data_pyg, num_nodes_subgraph=150)
        G = to_networkx(data_pyg, to_undirected=True) 
        ground_truth = data_pyg.y.cpu().numpy() 
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}. Please choose from: KarateClub, Cora_Subset, Cora, CiteSeer, PubMed.")

    if ground_truth is not None and len(ground_truth) != G.number_of_nodes():
         print(f"Warning: Ground truth size {len(ground_truth)} ({len(np.unique(ground_truth))} communities) does not match graph nodes {G.number_of_nodes()}. Skipping external metrics.")
         ground_truth = None
         
    print(f"Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")
    if ground_truth is not None:
        print(f"Ground truth communities: {len(np.unique(ground_truth))}")
    return G, ground_truth, data_pyg 


# Metric Calculation Functions

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

# Community Detection Algorithms - Traditional Methods

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
        print(f"    Error during Girvan-Newman iteration: {e}")
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


# Plotting Functions for Visualization 

def plot_graph_communities(G, partition, title, ground_truth_labels=None, figsize=(8, 8)):
    plt.figure(figsize=figsize)
    
    node_ids = list(G.nodes())
    valid_partition_nodes = [node for node in node_ids if node in partition]
    
    if not valid_partition_nodes:
        print(f"Warning: No valid nodes in partition for plotting {title}. Skipping graph plot.")
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
        print(f"Warning: Not enough embeddings for t-SNE in {title}. Skipping embedding plot.")
        plt.close()
        return

    perplexity_val = min(30, len(embeddings) - 1)
    if perplexity_val < 1: 
        print(f"Warning: Perplexity value ({perplexity_val}) is too low for t-SNE in {title}. Skipping embedding plot.")
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


# Main Execution Loo

datasets_to_test = [
    'KarateClub',    # Small (34 nodes) - has GT, good for graph plot & embeddings
    'Cora_Subset',   # Small-Medium (approx. 150 nodes) - has GT, G-N feasible, good for embeddings
    'Cora',          # Medium (2708 nodes) - has GT, good for embeddings (but computationally heavier)
    'PubMed'         # Large (19717 nodes) - has GT, good for embeddings (but computationally heavier)
]

GRAPH_PLOT_DATASETS = ['KarateClub', 'Cora_Subset', 'Cora', 'PubMed'] 

EMBEDDING_PLOT_DATASETS = ['KarateClub', 'Cora_Subset', 'Cora', 'PubMed'] 

results = []

for dataset_name in datasets_to_test:
    G, ground_truth, data_pyg = load_and_prepare_dataset(dataset_name)

    partitions_for_plotting = {}
    embeddings_for_plotting_data = {} 

    traditional_algorithms = {
        "Louvain": run_louvain,
        "Label Propagation": run_label_propagation,
    }

    if G.number_of_nodes() <= 185: 
        traditional_algorithms["Girvan-Newman"] = run_girvan_newman
    else:
        print(f"Skipping Girvan-Newman for {dataset_name} due to graph size (>185 nodes).")

    for algo_name, algo_func in traditional_algorithms.items():
        print(f"  - Running {algo_name}...")
        try:
            partition = algo_func(G)
            
            if not partition and G.number_of_nodes() > 0:
                print(f"    {algo_name} returned an empty partition for a non-empty graph. Metrics will be NaN.")
                num_communities = 0
            elif not partition and G.number_of_nodes() == 0:
                num_communities = 0
            else:
                num_communities = len(set(partition.values()))
            
            modularity, avg_conductance = np.nan, np.nan
            if partition and G.number_of_nodes() > 0:
                modularity, avg_conductance = calculate_internal_metrics(G, partition)

            nmi, ari, fms = np.nan, np.nan, np.nan
            if ground_truth is not None and partition and G.number_of_nodes() > 0:
                try:
                    nmi, ari, fms = calculate_external_metrics(ground_truth, partition)
                except ValueError as e:
                    print(f"    Error calculating external metrics for {algo_name} on {dataset_name}: {e}")

            row = {
                'Dataset': dataset_name,
                'Algorithm': algo_name,
                'Num_Communities': num_communities,
                'Modularity': modularity,
                'Avg_Conductance': avg_conductance,
                'NMI': nmi,
                'ARI': ari,
                'Fowlkes_Mallows': fms
            }
            results.append(row)
            print(f"    Completed {algo_name}.")

            if dataset_name in GRAPH_PLOT_DATASETS:
                partitions_for_plotting[algo_name] = partition

        except Exception as e:
            print(f"    Error running {algo_name} on {dataset_name}: {e}")
            row = {
                'Dataset': dataset_name,
                'Algorithm': algo_name,
                'Num_Communities': np.nan,
                'Modularity': np.nan,
                'Avg_Conductance': np.nan,
                'NMI': np.nan,
                'ARI': np.nan,
                'Fowlkes_Mallows': np.nan
            }
            results.append(row)
    
    dl_algorithms = {
        "GNN (GCN)": GCN,
        "GNN (GraphSage)": GraphSage, 
    }

    for dl_algo_name, dl_model_class in dl_algorithms.items():
        print(f"  - Running {dl_algo_name}...")
        
        if (data_pyg is not None and hasattr(data_pyg, 'x') and hasattr(data_pyg, 'edge_index') and 
            hasattr(data_pyg, 'y') and hasattr(data_pyg, 'train_mask') and data_pyg.train_mask.any()): 
            try:
                num_classes_for_gnn = -1 

                if dataset_name == 'Cora_Subset':
                    temp_full_cora_dataset = Planetoid(root='/tmp/Cora', name='Cora')
                    num_classes_for_gnn = temp_full_cora_dataset[0].y.max().item() + 1 
                else: 
                    num_classes_for_gnn = data_pyg.y.unique().size(0)
                
                if not isinstance(num_classes_for_gnn, int) or num_classes_for_gnn <= 0:
                    raise ValueError(f"Failed to determine valid num_classes ({num_classes_for_gnn}) for GNN on {dataset_name}.")
                
                gnn_partition, gnn_embeddings = run_gnn(dl_model_class, data_pyg, num_classes=num_classes_for_gnn) 
                
                num_communities = len(set(gnn_partition.values()))
                modularity, avg_conductance = calculate_internal_metrics(G, gnn_partition) 

                nmi, ari, fms = np.nan, np.nan, np.nan
                if ground_truth is not None:
                    try:
                        nmi, ari, fms = calculate_external_metrics(ground_truth, gnn_partition)
                    except ValueError as e:
                        print(f"    Error calculating external metrics for {dl_algo_name} on {dataset_name}: {e}")

                row = {
                    'Dataset': dataset_name,
                    'Algorithm': dl_algo_name,
                    'Num_Communities': num_communities,
                    'Modularity': modularity,
                    'Avg_Conductance': avg_conductance,
                    'NMI': nmi,
                    'ARI': ari,
                    'Fowlkes_Mallows': fms
                }
                results.append(row)
                print(f"    Completed {dl_algo_name}.")

                if dataset_name in GRAPH_PLOT_DATASETS:
                    partitions_for_plotting[dl_algo_name] = gnn_partition
                if dataset_name in EMBEDDING_PLOT_DATASETS:
                    predicted_labels_array_for_plot = np.array([gnn_partition[i] for i in range(G.number_of_nodes())])
                    embeddings_for_plotting_data[dl_algo_name] = (gnn_embeddings, predicted_labels_array_for_plot)


            except Exception as e:
                print(f"    Error running {dl_algo_name} on {dataset_name}: {e}")
                row = {
                    'Dataset': dataset_name,
                    'Algorithm': dl_algo_name,
                    'Num_Communities': np.nan,
                    'Modularity': np.nan,
                    'Avg_Conductance': np.nan,
                    'NMI': np.nan,
                    'ARI': np.nan,
                    'Fowlkes_Mallows': np.nan
                }
                results.append(row)
        else:
            print(f"    Skipping {dl_algo_name} for {dataset_name}: Missing required PyG Data attributes (x, edge_index, y, train_mask).")
            row = {
                    'Dataset': dataset_name,
                    'Algorithm': dl_algo_name,
                    'Num_Communities': np.nan,
                    'Modularity': np.nan,
                    'Avg_Conductance': np.nan,
                    'NMI': np.nan,
                    'ARI': np.nan,
                    'Fowlkes_Mallows': np.nan
                }
            results.append(row)
    
    if dataset_name in GRAPH_PLOT_DATASETS:
        if ground_truth is not None:
            gt_partition = {i: ground_truth[i] for i in range(len(ground_truth))}
            plot_graph_communities(G, gt_partition, f"{dataset_name} - Ground Truth Communities")

        for algo_name, partition in partitions_for_plotting.items():
            plot_graph_communities(G, partition, f"{dataset_name} - {algo_name} Communities")
    
    if dataset_name in EMBEDDING_PLOT_DATASETS:
        if ground_truth is not None and embeddings_for_plotting_data: 
            sample_embeddings_tuple = None
            if "GNN (GCN)" in embeddings_for_plotting_data:
                sample_embeddings_tuple = embeddings_for_plotting_data["GNN (GCN)"]
            elif "GNN (GraphSage)" in embeddings_for_plotting_data: 
                sample_embeddings_tuple = embeddings_for_plotting_data["GNN (GraphSage)"]
            
            if sample_embeddings_tuple is not None:
                sample_embeddings = sample_embeddings_tuple[0] 
                plot_embeddings_2d(sample_embeddings, ground_truth, f"{dataset_name} - GNN Embeddings (Ground Truth)")

        for algo_name, (embeddings, predicted_labels_array) in embeddings_for_plotting_data.items():
            plot_embeddings_2d(embeddings, predicted_labels_array, f"{dataset_name} - {algo_name} (Predicted Communities)")

# Display results in a DataFrame
results_df = pd.DataFrame(results)
print("\n--- Summary of Community Detection Results ---")
print(results_df.round(4).to_markdown(index=False))

# Save results as a markdown table
markdown_table = results_df.round(4).to_markdown(index=False)
with open("community_detection_results.md", "w", encoding="utf-8") as f:
    f.write("# Summary of Community Detection Results\n\n")
    f.write(markdown_table)
    f.write("\n")