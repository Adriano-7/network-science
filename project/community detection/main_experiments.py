import torch
import networkx as nx
from torch_geometric.datasets import Planetoid, KarateClub 
from torch_geometric.utils import to_networkx
from torch_geometric.data import Data 

import numpy as np
import pandas as pd
import warnings
import random
import os 
import requests 

from traditional_methods import run_louvain, run_girvan_newman, run_label_propagation
from deep_learning_methods import run_gnn, GCN, GraphSage, calculate_graphlet_features
from metrics_and_plots import calculate_internal_metrics, calculate_external_metrics, plot_graph_communities, plot_embeddings_2d

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

def get_dataset_stats(G, data_pyg=None):
    stats = {
        'Nodes': G.number_of_nodes(),
        'Edges': G.number_of_edges(),
        'Avg. Node Degree': f"{np.mean([d for n, d in G.degree()]):.2f}",
        'Density': f"{nx.density(G):.5f}",
        'Is Connected': nx.is_connected(G),
        '# Components': nx.number_connected_components(G),
    }
    if data_pyg is not None:
        if hasattr(data_pyg, 'x') and data_pyg.x is not None:
            stats['Node Feature Dim.'] = data_pyg.num_node_features
        if hasattr(data_pyg, 'y') and data_pyg.y is not None:
            stats['# Classes / GT Communities'] = data_pyg.y.unique().size(0)
    return stats


# Main Execution Loop

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

    print("\n  --- Running GNNs with Graphlet-Enhanced Features ---")

    try:
        graphlet_features_tensor = calculate_graphlet_features(G).cpu()

        for dl_algo_name, dl_model_class in dl_algorithms.items():
            print(f"  - Running {dl_algo_name} + Graphlets...")

            if (data_pyg is not None and hasattr(data_pyg, 'x') and hasattr(data_pyg, 'edge_index') and
                hasattr(data_pyg, 'y') and hasattr(data_pyg, 'train_mask') and data_pyg.train_mask.any()):
                
                data_with_graphlets = data_pyg.clone()

                if dataset_name == 'KarateClub':
                    print("    Strategy: Replacing original features with graphlet features.")
                    data_with_graphlets.x = graphlet_features_tensor
                else:
                    print("    Strategy: Augmenting original features with graphlet features.")
                    original_x_float = data_pyg.x.to(torch.float)
                    data_with_graphlets.x = torch.cat([original_x_float, graphlet_features_tensor], dim=1)
                
                try:
                    if dataset_name == 'Cora_Subset':
                        temp_full_cora_dataset = Planetoid(root='/tmp/Cora', name='Cora')
                        num_classes_for_gnn = temp_full_cora_dataset[0].y.max().item() + 1
                    else:
                        num_classes_for_gnn = data_pyg.y.unique().size(0)
                    
                    if not isinstance(num_classes_for_gnn, int) or num_classes_for_gnn <= 0:
                        raise ValueError(f"Failed to determine valid num_classes ({num_classes_for_gnn}) for GNN on {dataset_name}.")
                    
                    gnn_partition, gnn_embeddings = run_gnn(dl_model_class, data_with_graphlets, num_classes=num_classes_for_gnn)
                    
                    num_communities = len(set(gnn_partition.values()))
                    modularity, avg_conductance = calculate_internal_metrics(G, gnn_partition)

                    nmi, ari, fms = np.nan, np.nan, np.nan
                    if ground_truth is not None:
                        nmi, ari, fms = calculate_external_metrics(ground_truth, gnn_partition)

                    row = {
                        'Dataset': dataset_name,
                        'Algorithm': f"{dl_algo_name} + Graphlets", 
                        'Num_Communities': num_communities,
                        'Modularity': modularity,
                        'Avg_Conductance': avg_conductance,
                        'NMI': nmi,
                        'ARI': ari,
                        'Fowlkes_Mallows': fms
                    }
                    results.append(row)
                    print(f"    Completed {dl_algo_name} + Graphlets.")

                except Exception as e:
                    print(f"    Error running {dl_algo_name} + Graphlets on {dataset_name}: {e}")
            else:
                print(f"    Skipping {dl_algo_name} + Graphlets for {dataset_name}: Missing required PyG Data attributes.")
                
    except Exception as e:
        print(f"  Failed to run GNNs with graphlet features for {dataset_name}: {e}")    
    
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