import torch
from torch_geometric.datasets import Planetoid, GNNBenchmarkDataset, Actor
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import argparse
import itertools
from pathlib import Path

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

def clean_params(params: dict) -> dict:
  cleaned = {}
  for k, v in params.items():
    if pd.isna(v):
      continue
    native_v = v.item() if isinstance(v, np.generic) else v
    if k in ['hidden_channels', 'emb_dim']:
      cleaned[k] = int(native_v)
    else:
      cleaned[k] = native_v
  return cleaned

def run_role_discovery_experiment(dataset_name: str, use_tuned_models: bool = False):
  print("\n" + "#"*60)
  print(f"RUNNING ROLE DISCOVERY EXPERIMENT ON: {dataset_name.upper()}")
  print("#"*60)

  output_dir = Path(f"results/role_discovery/{dataset_name}")
  output_dir.mkdir(parents=True, exist_ok=True)
  print(f"Results will be saved to: {output_dir}")

  dataset = get_dataset(dataset_name)
  data = dataset[0]
  print(f"Successfully loaded '{dataset_name}' dataset.")
  print(f"Graph properties: {data.num_nodes} nodes, {data.num_edges} edges, {data.num_features} features.")

  if use_tuned_models:
    print("\n### Using BEST pre-tuned models for experiment ###")
    tuning_results_path = output_dir / "hyperparameter_tuning_results.csv"
    if not tuning_results_path.exists():
      raise FileNotFoundError(f"Tuning results not found at {tuning_results_path}. Please run with --mode tune first.")
   
    tuning_df = pd.read_csv(tuning_results_path)
    models_to_test = {"Feature-Based_Roles": FeatureBasedRoles()}

    gae_results = tuning_df[tuning_df['model_name'] == 'GNN_Embedder_GAE']
    if not gae_results.empty:
      best_gae_params = clean_params(gae_results.iloc[0].drop(['model_name', 'best_silhouette']).to_dict())
      gae_model_path = output_dir / "best_GNN_Embedder_GAE_model.pt"
      models_to_test["GNN_Embedder_GAE"] = GNNEmbedder(**best_gae_params, model_path=str(gae_model_path), force_retrain=False)
    else:
      print("Warning: No tuned parameters found for GNN_Embedder_GAE. Skipping in inference mode.")

    dgi_results = tuning_df[tuning_df['model_name'] == 'GNN_Embedder_DGI']
    if not dgi_results.empty:
      best_dgi_params = clean_params(dgi_results.iloc[0].drop(['model_name', 'best_silhouette']).to_dict())
      dgi_model_path = output_dir / "best_GNN_Embedder_DGI_model.pt"
      models_to_test["GNN_Embedder_DGI"] = DGIEmbedder(in_channels=data.num_features, **best_dgi_params, model_path=str(dgi_model_path), force_retrain=False)
    else:
      print("Warning: No tuned parameters found for GNN_Embedder_DGI. Skipping in inference mode.")

  else:
    print("\n### Using DEFAULT models for experiment ###")
    models_to_test = {
      "Feature-Based_Roles": FeatureBasedRoles(),
      "GNN_Embedder_GAE": GNNEmbedder(hidden_channels=128, emb_dim=32, force_retrain=True),
      "GNN_Embedder_DGI": DGIEmbedder(in_channels=data.num_features, hidden_channels=128, force_retrain=True)
    }

  k_values = [3, 4, 5, 6, 7]
  for model_name, model in models_to_test.items():
    print("\n" + "="*50)
    print(f"Testing Model: {model_name}")
    print("="*50)
   
    if not model: continue

    results, best_k, best_score, best_result = [], -1, -2, None

    for k in k_values:
      embeddings, role_labels = model.predict(data, k)
      if len(np.unique(role_labels.numpy())) > 1:
        silhouette = silhouette_score(embeddings.cpu().numpy(), role_labels.cpu().numpy())
        davies_bouldin = davies_bouldin_score(embeddings.cpu().numpy(), role_labels.cpu().numpy())
        calinski_harabasz = calinski_harabasz_score(embeddings.cpu().numpy(), role_labels.cpu().numpy())
        results.append({"k": k, "Silhouette Score": silhouette, "Davies-Bouldin Index": davies_bouldin, "Calinski-Harabasz Index": calinski_harabasz})
        if silhouette > best_score:
          best_score, best_k, best_result = silhouette, k, (embeddings, role_labels)
   
    results_df = pd.DataFrame(results).set_index('k')
    print("\n### Clustering Evaluation Summary")
    print(results_df.to_string(float_format="%.4f"))
    print(f"\nBest result found for k={best_k} with a Silhouette Score of {best_score:.4f}")

    csv_path = output_dir / f"{model_name}_clustering_metrics.csv"
    results_df.to_csv(csv_path)
    print(f"Evaluation metrics saved to {csv_path}")

    if best_result:
      embeddings, role_labels = best_result
      viz_title, plot_path = f"t-SNE of Roles from {model_name} on {dataset_name} (k={best_k})", output_dir / f"{model_name}_k{best_k}_tsne.png"
      visualize_roles_tsne(embeddings, role_labels, viz_title, best_score, save_path=str(plot_path))

def run_hyperparameter_tuning(dataset_name: str):
  print("\n" + "#"*60)
  print(f"RUNNING HYPERPARAMETER TUNING ON: {dataset_name.upper()}")
  print("#"*60)

  dataset = get_dataset(dataset_name)
  data = dataset[0]
 
  output_dir = Path(f"results/role_discovery/{dataset_name}")
  output_dir.mkdir(parents=True, exist_ok=True)
  tuning_results_path = output_dir / "hyperparameter_tuning_results.csv"

  param_grid = {
    'GNN_Embedder_GAE': {'lr': [0.01, 0.005], 'hidden_channels': [128, 256], 'emb_dim': [32, 64]},
    'GNN_Embedder_DGI': {'lr': [0.001, 0.0005], 'hidden_channels': [128, 256]}
  }
 
  all_tuning_results, k_values_for_eval = [], [3, 4, 5, 6, 7]

  for model_name, grid in param_grid.items():
    keys, values = zip(*grid.items())
    print(f"\n### Tuning {model_name} ###")

    for param_combination in itertools.product(*values):
      params = dict(zip(keys, param_combination))
      print(f"\nTesting params: {params}")
     
      model = GNNEmbedder(**params, force_retrain=True) if model_name == 'GNN_Embedder_GAE' else DGIEmbedder(in_channels=data.num_features, **params, force_retrain=True)
     
      best_silhouette = -2
      embeddings, _ = model.predict(data, k=k_values_for_eval[0])
      for k in k_values_for_eval:
        _, role_labels = model.predict(data, k)
        if len(np.unique(role_labels.numpy())) > 1:
          silhouette = silhouette_score(embeddings.cpu().numpy(), role_labels.cpu().numpy())
          best_silhouette = max(best_silhouette, silhouette)
     
      if best_silhouette > -2:
        print(f"Best Silhouette Score for this param set: {best_silhouette:.4f}")
        all_tuning_results.append({'model_name': model_name, 'best_silhouette': best_silhouette, **params})
 
  if not all_tuning_results:
    print("Tuning finished, but no results were recorded.")
    return

  tuning_df = pd.DataFrame(all_tuning_results).sort_values('best_silhouette', ascending=False)
  tuning_df.to_csv(tuning_results_path, index=False)
  print("\n### Hyperparameter Tuning Summary ###")
  print(tuning_df.to_string())

  for model_name in param_grid.keys():
    model_results = tuning_df[tuning_df['model_name'] == model_name]
    if not model_results.empty:
      best_params_row = model_results.iloc[0]
      best_params = clean_params(best_params_row.drop(['model_name', 'best_silhouette']).to_dict())

      print(f"\nBest params for {model_name}: {best_params}")
      print(f"Retraining best {model_name} and saving for inference...")
     
      model_save_path = output_dir / f"best_{model_name}_model.pt"
      best_model = GNNEmbedder(**best_params, model_path=str(model_save_path), force_retrain=True) if model_name == 'GNN_Embedder_GAE' else DGIEmbedder(in_channels=data.num_features, **best_params, model_path=str(model_save_path), force_retrain=True)
      best_model.train(data)
      print(f"Best model for {model_name} saved successfully.")


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
                try:
                    df = pd.read_csv(csv_path, index_col='k')
                    if not df.empty and 'Silhouette Score' in df.columns and df['Silhouette Score'].notna().any():
                        best_row = df.loc[df['Silhouette Score'].idxmax()]
                        all_results.append({
                            "Dataset": dataset_name,
                            "Model": model_name,
                            "Best k": int(best_row.name),
                            "Silhouette Score": best_row["Silhouette Score"],
                            "Davies-Bouldin Index": best_row["Davies-Bouldin Index"],
                            "Calinski-Harabasz Index": best_row["Calinski-Harabasz Index"]
                        })
                except (ValueError, KeyError) as e:
                    print(f"Warning: Could not process file {csv_path}. Error: {e}")
            else:
                print(f"Warning: {csv_path} not found. Skipping for {dataset_name}.")

    if not all_results:
        print("No results found to generate a comparison summary. Make sure you have run inference first.")
        return

    summary_df = pd.DataFrame(all_results)
    summary_df_path = "results/role_discovery/comparison_summary.csv"
    summary_df.to_csv(summary_df_path, index=False)
    print(f"Comparison summary table saved to {summary_df_path}")
    print("\n### Overall Comparison Summary")
    print(summary_df.to_string(float_format="%.4f"))

    # --- Generate Individual Plots for each dataset ---
    print("\n### Generating Individual Dataset Plots ###")
    for dataset_name in datasets_to_run:
        dataset_df = summary_df[summary_df['Dataset'] == dataset_name]
        if dataset_df.empty:
            continue
        
        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(20, 7))
        fig.suptitle(f'Model Comparison on {dataset_name}', fontsize=20)
        metrics = ["Silhouette Score", "Davies-Bouldin Index", "Calinski-Harabasz Index"]
        titles = ["Silhouette Score (Higher is Better)", "Davies-Bouldin Index (Lower is Better)", "Calinski-Harabasz Index (Higher is Better)"]
        
        palette = sns.color_palette("magma", n_colors=len(dataset_df['Model'].unique()))

        for i, metric in enumerate(metrics):
            ax = axes[i]
            sns.barplot(data=dataset_df, x='Model', y=metric, ax=ax, hue='Model', palette=palette, legend=False)
            ax.set_title(titles[i], fontsize=14)
            ax.set_ylabel(metric)
            ax.set_xlabel("Model")
            ax.tick_params(axis='x', rotation=45, labelsize=10)

            for container in ax.containers:
                ax.bar_label(container, fmt='%.2f', label_type='edge')
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plot_save_path = f"results/role_discovery/{dataset_name}/comparison_metrics_plot.png"
        plt.savefig(plot_save_path)
        print(f"Individual comparison plot saved to {plot_save_path}")
        plt.close(fig)

    # --- Generate Aggregate Plot if more than one dataset is processed ---
    if len(datasets_to_run) > 1:
        print("\n### Generating Aggregate Comparison Plot ###")
        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(20, 7), sharey=False)
        metrics = ["Silhouette Score", "Davies-Bouldin Index", "Calinski-Harabasz Index"]
        titles = ["Aggregate Silhouette Score (Higher is Better)", "Aggregate Davies-Bouldin Index (Lower is Better)", "Aggregate Calinski-Harabasz Index (Higher is Better)"]
        
        palette = sns.color_palette("viridis", n_colors=len(summary_df['Model'].unique()))
        color_map = {model: palette[i] for i, model in enumerate(sorted(summary_df['Model'].unique()))}

        for i, metric in enumerate(metrics):
            ax = axes[i]
            plot_df = summary_df.pivot(index='Dataset', columns='Model', values=metric)
            plot_df = plot_df[sorted(plot_df.columns)]
            plot_df.plot(kind='bar', ax=ax, rot=0, color=[color_map[col] for col in plot_df.columns])
            ax.set_title(titles[i])
            ax.set_ylabel(metric)
            ax.tick_params(axis='x', rotation=0)
            ax.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')
            
            for container in ax.containers:
                ax.bar_label(container, fmt='%.2f', label_type='edge', fontsize=8)

        plt.tight_layout()
        plot_save_path = "results/role_discovery/aggregate_comparison_metrics_plot.png"
        plt.savefig(plot_save_path)
        print(f"Aggregate comparison metrics plot saved to {plot_save_path}")
        plt.close(fig)
    else:
        print("\nSkipping aggregate plot as only one dataset was processed.")

def main():
    parser = argparse.ArgumentParser(description="Run Role Discovery experiments.")
    # --- ADDED 'summarize' to choices ---
    parser.add_argument('--mode', type=str, default='run', choices=['run', 'tune', 'inference', 'summarize'],
                        help="Execution mode: 'run', 'tune', 'inference', or 'summarize' (only generate plots from existing results).")
    parser.add_argument('--dataset', type=str, nargs='+', default=['Cora'],
                        help="A list of datasets to run on or summarize, e.g., --dataset Cora Twitch-DE.")
    
    args = parser.parse_args()

    if args.mode == 'tune':
        for dataset in args.dataset:
            run_hyperparameter_tuning(dataset)
    elif args.mode == 'inference':
        print("Running in INFERENCE mode using best pre-tuned models.")
        for dataset in args.dataset:
            run_role_discovery_experiment(dataset, use_tuned_models=True)
        print("\nAll role discovery experiments finished successfully.")
        generate_comparison_summary(args.dataset)
    # --- ADDED LOGIC FOR 'summarize' MODE ---
    elif args.mode == 'summarize':
        print(f"Running in SUMMARIZE mode. Will generate plots for: {', '.join(args.dataset)}")
        generate_comparison_summary(args.dataset)
    else: # mode == 'run'
        print("Running in standard RUN mode with default hyperparameters.")
        for dataset in args.dataset:
            run_role_discovery_experiment(dataset, use_tuned_models=False)
        print("\nAll role discovery experiments finished successfully.")
        generate_comparison_summary(args.dataset)


if __name__ == '__main__':
    main()