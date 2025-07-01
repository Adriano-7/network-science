import torch
from torch_geometric.datasets import Planetoid, GNNBenchmarkDataset, Actor
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from models.RoleDiscoveryModel import RoleDiscoveryModel
from models.FeatureBasedRoles import FeatureBasedRoles
from models.GNNEmbedder import GNNEmbedder
from models.DGIEmbedder import DGIEmbedder
from utils.visualization import visualize_roles_tsne

def get_dataset(name: str, root: str = '/tmp/'):
    path = os.path.join(root, name)
    if name.lower() == 'cora':
        return Planetoid(root=path, name='Cora')
    elif name.lower() == 'cluster':
        return GNNBenchmarkDataset(root=path, name='CLUSTER')
    elif name.lower() == 'actor':
        return Actor(root=path)
    else:
        raise ValueError(f"Unknown dataset: {name}")

def run_role_discovery_experiment(dataset_name: str):

    print("\n" + "#"*60)
    print(f"RUNNING ROLE DISCOVERY EXPERIMENT ON: {dataset_name.upper()}")
    print("#"*60)

    output_dir = f"results/role_discovery/{dataset_name}"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Results will be saved to: {output_dir}")

    dataset = get_dataset(dataset_name)
    data = dataset[0]
    print(f"Successfully loaded '{dataset_name}' dataset.")
    print(f"Graph properties: {data.num_nodes} nodes, {data.num_edges} edges, {data.num_features} features.")

    models_to_test = {
        "Feature-Based_Roles": FeatureBasedRoles(),
        "GNN_Embedder_GAE": GNNEmbedder(hidden_channels=128, emb_dim=32),
        "GNN_Embedder_DGI": DGIEmbedder(in_channels=data.num_features, hidden_channels=128)
    }

    k_values = [3, 4, 5, 6, 7] 
    for model_name, model in models_to_test.items():
        print("\n" + "="*50)
        print(f"Testing Model: {model_name}")
        print("="*50)

        if hasattr(model, 'train'):
            model.train(data)

        results = []
        best_k = -1
        best_score = -2  
        best_result = None

        for k in k_values:
            embeddings, role_labels = model.predict(data, k)
            
            try:
                silhouette = silhouette_score(embeddings, role_labels)
                davies_bouldin = davies_bouldin_score(embeddings, role_labels)
                calinski_harabasz = calinski_harabasz_score(embeddings, role_labels)
                
                results.append({
                    "k": k,
                    "Silhouette Score": silhouette,
                    "Davies-Bouldin Index": davies_bouldin,
                    "Calinski-Harabasz Index": calinski_harabasz
                })

                if silhouette > best_score:
                    best_score = silhouette
                    best_k = k
                    best_result = (embeddings, role_labels)
            except ValueError as e:
                print(f"Could not compute metrics for k={k}. Error: {e}. Skipping.")
                continue

        results_df = pd.DataFrame(results).set_index('k')
        print("\n### Clustering Evaluation Summary")
        print(results_df.to_string(float_format="%.4f"))
        print(f"\nBest result found for k={best_k} with a Silhouette Score of {best_score:.4f}")

        csv_path = os.path.join(output_dir, f"{model_name}_clustering_metrics.csv")
        results_df.to_csv(csv_path)
        print(f"Evaluation metrics saved to {csv_path}")

        if best_result:
            embeddings, role_labels = best_result
            viz_title = f"t-SNE of Roles from {model_name} on {dataset_name} (k={best_k})"
            plot_path = os.path.join(output_dir, f"{model_name}_k{best_k}_tsne.png")
            visualize_roles_tsne(embeddings, role_labels, viz_title, best_score, save_path=plot_path)

def generate_comparison_summary(datasets_to_run: list):
    print("\n" + "#"*60)
    print("GENERATING COMPARISON SUMMARY TABLES AND PLOTS")
    print("#"*60)

    all_results = []
    for dataset_name in datasets_to_run:
        output_dir = f"results/role_discovery/{dataset_name}"
        model_names = ["Feature-Based_Roles", "GNN_Embedder_GAE", "GNN_Embedder_DGI"]
        
        for model_name in model_names:
            csv_path = os.path.join(output_dir, f"{model_name}_clustering_metrics.csv")
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path, index_col='k')
                best_row = df.loc[df['Silhouette Score'].idxmax()]
                all_results.append({
                    "Dataset": dataset_name,
                    "Model": model_name,
                    "Best k": int(best_row.name),
                    "Silhouette Score": best_row["Silhouette Score"],
                    "Davies-Bouldin Index": best_row["Davies-Bouldin Index"],
                    "Calinski-Harabasz Index": best_row["Calinski-Harabasz Index"]
                })
            else:
                print(f"Warning: {csv_path} not found. Skipping.")

    summary_df = pd.DataFrame(all_results)
    summary_df_path = "results/role_discovery/comparison_summary.csv"
    summary_df.to_csv(summary_df_path, index=False)
    print(f"Comparison summary table saved to {summary_df_path}")
    print("\n### Overall Comparison Summary")
    print(summary_df.to_string(float_format="%.4f"))

    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(20, 7), sharey=False)
    metrics = ["Silhouette Score", "Davies-Bouldin Index", "Calinski-Harabasz Index"]
    titles = ["Best Silhouette Score", "Best Davies-Bouldin Index (Lower is Better)", "Best Calinski-Harabasz Index (Higher is Better)"]
    
    palette = sns.color_palette("rocket", n_colors=len(summary_df['Model'].unique()))
    color_map = {model: palette[i] for i, model in enumerate(summary_df['Model'].unique())}

    for i, metric in enumerate(metrics):
        ax = axes[i]
        plot_df = summary_df.pivot(index='Dataset', columns='Model', values=metric)
        plot_df.plot(kind='bar', ax=ax, rot=0, color=[color_map[col] for col in plot_df.columns])
        ax.set_title(titles[i])
        ax.set_ylabel(metric)
        ax.tick_params(axis='x', rotation=45)
        ax.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')
        
        for container in ax.containers:
            ax.bar_label(container, fmt='%.2f', label_type='edge')

    plt.tight_layout()
    plot_save_path = "results/role_discovery/comparison_metrics_plot.png"
    plt.savefig(plot_save_path)
    print(f"Comparison metrics plot saved to {plot_save_path}")
    plt.close()


def main():
    datasets_to_run = ['Cora', 'CLUSTER', 'Actor']
    for dataset in datasets_to_run:
        run_role_discovery_experiment(dataset)
    print("\nAll role discovery experiments finished successfully.")
    generate_comparison_summary(datasets_to_run)

if __name__ == '__main__':
    main()