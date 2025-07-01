import torch
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path
from torch_geometric.utils import to_networkx

def analyze_role_characteristics(data: torch.Tensor, role_labels: torch.Tensor, model_name: str, dataset_name: str, k: int, output_dir: Path):
    print(f"\nAnalyzing characteristics for {k} roles discovered by {model_name} on {dataset_name}")

    G = to_networkx(data, to_undirected=True)
    if not G.nodes:
        print("Warning: Graph has no nodes. Skipping role analysis.")
        return

    print("Calculating centrality metrics...")
    k_betweenness = min(100, data.num_nodes - 1) if data.num_nodes > 100 else None
    
    centrality_measures = {
        'degree': nx.degree_centrality(G),
        'betweenness': nx.betweenness_centrality(G, k=k_betweenness, seed=42),
        'closeness': nx.closeness_centrality(G),
        'eigenvector': nx.eigenvector_centrality_numpy(G),
        'pagerank': nx.pagerank(G, alpha=0.85),
        'clustering_coeff': nx.clustering(G)
    }

    node_ids = sorted(list(G.nodes()))
    analysis_data = {
        'node_id': node_ids,
        'role_id': role_labels.cpu().numpy()
    }

    for metric_name, values_dict in centrality_measures.items():
        analysis_data[metric_name] = [values_dict.get(node, 0) for node in node_ids]

    analysis_df = pd.DataFrame(analysis_data)
    role_summary_df = analysis_df.groupby('role_id').agg(
        num_nodes=('node_id', 'count'),
        avg_degree=('degree', 'mean'),
        avg_betweenness=('betweenness', 'mean'),
        avg_closeness=('closeness', 'mean'),
        avg_eigenvector=('eigenvector', 'mean'),
        avg_pagerank=('pagerank', 'mean'),
        avg_clustering_coeff=('clustering_coeff', 'mean')
    ).reset_index()

    print("\nAverage Structural Properties per Role:")
    print(role_summary_df.to_string(float_format="%.4f"))

    analysis_save_path = output_dir / f"{model_name}_k{k}_role_analysis.csv"
    role_summary_df.to_csv(analysis_save_path, index=False)
    print(f"\nRole analysis summary saved to: {analysis_save_path}")


def generate_comparison_summary(datasets_to_run: list):
    """
    Reads all individual experiment results, creates a summary CSV, and generates comparison plots.
    """
    print("\n" + "#"*60)
    print("GENERATING COMPARISON SUMMARY AND PLOTS")
    print("#"*60)

    all_results = []
    root_results_dir = Path("results/role_discovery/")
    for dataset_name in datasets_to_run:
        output_dir = root_results_dir / dataset_name
        model_names = ["Feature-Based_Roles", "GNN_Embedder_GAE", "GNN_Embedder_DGI"]
        
        for model_name in model_names:
            csv_path = output_dir / f"{model_name}_clustering_metrics.csv"
            if csv_path.exists():
                try:
                    df = pd.read_csv(csv_path).set_index('k')
                    if not df.empty and 'Silhouette Score' in df.columns and df['Silhouette Score'].notna().any():
                        best_row = df.loc[df['Silhouette Score'].idxmax()]
                        all_results.append({
                            "Dataset": dataset_name, "Model": model_name, "Best k": int(best_row.name),
                            "Silhouette Score": best_row["Silhouette Score"],
                            "Davies-Bouldin Index": best_row["Davies-Bouldin Index"],
                            "Calinski-Harabasz Index": best_row["Calinski-Harabasz Index"]
                        })
                except (ValueError, KeyError, pd.errors.EmptyDataError) as e:
                    print(f"Warning: Could not process file {csv_path}. Error: {e}")
            else:
                print(f"Warning: {csv_path} not found. Skipping for {dataset_name}.")

    if not all_results:
        print("No results found to generate a comparison summary.")
        return

    summary_df = pd.DataFrame(all_results)
    summary_df_path = root_results_dir / "comparison_summary.csv"
    summary_df.to_csv(summary_df_path, index=False)
    print(f"Comparison summary table saved to {summary_df_path}")
    print("\n### Overall Comparison Summary ###")
    print(summary_df.to_string(float_format="%.4f"))
    
    for dataset_name in datasets_to_run:
        dataset_df = summary_df[summary_df['Dataset'] == dataset_name]
        if dataset_df.empty: continue
        
        fig, axes = plt.subplots(1, 3, figsize=(20, 7))
        fig.suptitle(f'Model Comparison on {dataset_name}', fontsize=20)
        metrics = ["Silhouette Score", "Davies-Bouldin Index", "Calinski-Harabasz Index"]
        titles = ["Silhouette (Higher is Better)", "Davies-Bouldin (Lower is Better)", "Calinski-Harabasz (Higher is Better)"]
        palette = sns.color_palette("magma", n_colors=len(dataset_df['Model'].unique()))
        
        for i, metric in enumerate(metrics):
            sns.barplot(data=dataset_df, x='Model', y=metric, ax=axes[i], hue='Model', palette=palette, legend=False)
            axes[i].set_title(titles[i], fontsize=14)
            axes[i].set_ylabel(metric)
            axes[i].set_xlabel("Model")
            axes[i].tick_params(axis='x', rotation=45, labelsize=10)
            for container in axes[i].containers:
                axes[i].bar_label(container, fmt='%.2f', label_type='edge')
                
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(root_results_dir / dataset_name / "comparison_metrics_plot.png")
        plt.close(fig)

    if len(datasets_to_run) > 1:
        fig, axes = plt.subplots(1, 3, figsize=(20, 7), sharey=False)
        metrics = ["Silhouette Score", "Davies-Bouldin Index", "Calinski-Harabasz Index"]
        titles = ["Aggregate Silhouette", "Aggregate Davies-Bouldin", "Aggregate Calinski-Harabasz"]
        palette = sns.color_palette("magma", n_colors=len(summary_df['Model'].unique()))
        color_map = {model: palette[i] for i, model in enumerate(sorted(summary_df['Model'].unique()))}
        
        for i, metric in enumerate(metrics):
            plot_df = summary_df.pivot(index='Dataset', columns='Model', values=metric)
            plot_df[sorted(plot_df.columns)].plot(kind='bar', ax=axes[i], rot=0, color=[color_map[col] for col in sorted(plot_df.columns)])
            axes[i].set_title(titles[i])
            axes[i].legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')
            for container in axes[i].containers:
                axes[i].bar_label(container, fmt='%.2f', label_type='edge', fontsize=8)

        plt.tight_layout()
        plt.savefig(root_results_dir / "aggregate_comparison_metrics_plot.png")
        plt.close(fig)