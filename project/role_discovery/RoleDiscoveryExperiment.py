import torch
from torch_geometric.datasets import Planetoid
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import os
import pandas as pd

from models.RoleDiscoveryModel import RoleDiscoveryModel
from models.FeatureBasedRoles import FeatureBasedRoles
from models.GNNEmbedder import GNNEmbedder
from utils.visualization import visualize_roles_tsne

def run_role_discovery_experiment(dataset_name: str):

    print("\n" + "#"*60)
    print(f"RUNNING ROLE DISCOVERY EXPERIMENT ON: {dataset_name.upper()}")
    print("#"*60)

    output_dir = f"results/role_discovery/{dataset_name}"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Results will be saved to: {output_dir}")

    dataset = Planetoid(root='/tmp/' + dataset_name, name=dataset_name)
    data = dataset[0]
    print(f"Successfully loaded '{dataset_name}' dataset.")
    print(f"Graph properties: {data.num_nodes} nodes, {data.num_edges} edges, {data.num_features} features.")

    models_to_test = {
        "Feature-Based_Roles": FeatureBasedRoles(),
        "GNN_Embedder_GAE": GNNEmbedder(hidden_channels=128, emb_dim=32)
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

def main():
    datasets_to_run = ['Cora'] 
    
    for dataset in datasets_to_run:
        run_role_discovery_experiment(dataset)
        
    print("\nAll role discovery experiments finished successfully.")

if __name__ == '__main__':
    main()