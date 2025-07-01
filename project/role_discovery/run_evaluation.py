import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

# Project-specific imports
from .models.FeatureBasedRoles import FeatureBasedRoles
from .models.FeatureBasedRolesGraphlets import FeatureBasedRolesGraphlets
from .models.GNNEmbedder import GNNEmbedder
from .models.DGIEmbedder import DGIEmbedder
from .utils.experiment_utils import get_dataset, clean_params
from .utils.visualization import visualize_roles_tsne
from .utils.analysis import analyze_role_characteristics, create_and_visualize_role_adjacency

def run_role_discovery_experiment(dataset_name: str, use_tuned_models: bool = False):
    """
    Runs the core role discovery experiment for a given dataset.
    """
    print("\n" + "#"*60)
    print(f"RUNNING EVALUATION ON: {dataset_name.upper()}")
    print(f"Using tuned models: {use_tuned_models}")
    print("#"*60)

    output_dir = Path(f"results/role_discovery/{dataset_name}")
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Results will be saved to: {output_dir}")

    dataset = get_dataset(dataset_name)
    data = dataset[0]
    print(f"Successfully loaded '{dataset_name}': {data.num_nodes} nodes, {data.num_edges} edges.")

    models_to_test = {}
    if use_tuned_models:
        print("\n### Loading BEST pre-tuned models ###")
        tuning_results_path = output_dir / "hyperparameter_tuning_results.csv"
        if not tuning_results_path.exists():
            raise FileNotFoundError(f"Tuning results not found at {tuning_results_path}. Run tuning first.")
        
        tuning_df = pd.read_csv(tuning_results_path)
        
        gae_results = tuning_df[tuning_df['model_name'] == 'GNN_Embedder_GAE']
        if not gae_results.empty:
            raw_gae_params = gae_results.iloc[0].to_dict()
            raw_gae_params.pop('model_name', None)
            raw_gae_params.pop('best_silhouette', None)
            gae_params = clean_params(raw_gae_params)
            models_to_test["GNN_Embedder_GAE"] = GNNEmbedder(**gae_params, model_path=str(output_dir / "best_GNN_Embedder_GAE_model.pt"), force_retrain=False)

        dgi_results = tuning_df[tuning_df['model_name'] == 'GNN_Embedder_DGI']
        if not dgi_results.empty:
            raw_dgi_params = dgi_results.iloc[0].to_dict()
            raw_dgi_params.pop('model_name', None)
            raw_dgi_params.pop('best_silhouette', None)
            dgi_params = clean_params(raw_dgi_params)
            models_to_test["GNN_Embedder_DGI"] = DGIEmbedder(in_channels=data.num_features, **dgi_params, model_path=str(output_dir / "best_GNN_Embedder_DGI_model.pt"), force_retrain=False)

    else:
        print("\n### Using DEFAULT models ###")
        models_to_test = {
            "GNN_Embedder_GAE": GNNEmbedder(hidden_channels=128, emb_dim=32, force_retrain=True),
            "GNN_Embedder_DGI": DGIEmbedder(in_channels=data.num_features, hidden_channels=128, force_retrain=True)
        }
    
    models_to_test["Feature-Based_Roles"] = FeatureBasedRoles() 
    models_to_test["Feature-Based_Roles_Graphlets"] = FeatureBasedRolesGraphlets()

    k_values = [3, 4, 5, 6, 7]
    for model_name, model in models_to_test.items():
        print("\n" + "="*50)
        print(f"Testing Model: {model_name}")
        print("="*50)
        
        results, best_k, best_score, best_result = [], -1, -2, None

        for k in k_values:
            print(f"  - Evaluating for k={k}...")
            embeddings, role_labels = model.predict(data, k)
            if len(np.unique(role_labels.numpy())) > 1:
                silhouette = silhouette_score(embeddings.cpu().numpy(), role_labels.cpu().numpy())
                davies_bouldin = davies_bouldin_score(embeddings.cpu().numpy(), role_labels.cpu().numpy())
                calinski_harabasz = calinski_harabasz_score(embeddings.cpu().numpy(), role_labels.cpu().numpy())
                results.append({"k": k, "Silhouette Score": silhouette, "Davies-Bouldin Index": davies_bouldin, "Calinski-Harabasz Index": calinski_harabasz})
                if silhouette > best_score:
                    best_score, best_k, best_result = silhouette, k, (embeddings, role_labels)
        
        if not results:
            print(f"Could not generate any valid clustering results for {model_name}.")
            continue

        results_df = pd.DataFrame(results).set_index('k')
        print("\n### Clustering Evaluation Summary ###")
        print(results_df.to_string(float_format="%.4f"))
        print(f"\nBest result found for k={best_k} with Silhouette Score: {best_score:.4f}")

        csv_path = output_dir / f"{model_name}_clustering_metrics.csv"
        results_df.to_csv(csv_path)
        print(f"Evaluation metrics saved to {csv_path}")

        if best_result:
            embeddings, role_labels = best_result
            visualize_roles_tsne(
                embeddings, role_labels,
                title=f"Roles from {model_name} on {dataset_name} (k={best_k})",
                score=best_score,
                save_path=str(output_dir / f"{model_name}_k{best_k}_tsne.png")
            )
            analyze_role_characteristics(data, role_labels, model_name, dataset_name, best_k, output_dir)
            create_and_visualize_role_adjacency(data, role_labels, best_k, model_name, dataset_name, output_dir)

def main():
    parser = argparse.ArgumentParser(description="Run Role Discovery evaluation experiments.")
    parser.add_argument('--dataset', type=str, nargs='+', default=['Cora'], help="List of datasets to run on.")
    parser.add_argument('--use_tuned', action='store_true', help="Use best pre-tuned models instead of defaults.")
    args = parser.parse_args()

    for dataset in args.dataset:
        run_role_discovery_experiment(dataset, use_tuned_models=args.use_tuned)
    
    print("\nAll evaluation runs finished successfully.")

if __name__ == '__main__':
    main()