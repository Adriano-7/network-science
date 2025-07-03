import argparse
import itertools
import pandas as pd
from pathlib import Path
import numpy as np
from sklearn.metrics import silhouette_score

from .models.GNNEmbedder import GNNEmbedder
from .models.DGIEmbedder import DGIEmbedder
from .models.GNNEmbedderGraphlets import GNNEmbedderGraphlets
from .models.DGIEmbedderGraphlets import DGIEmbedderGraphlets
from .utils.experiment_utils import get_dataset, clean_params

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
        'GNN_Embedder_DGI': {'lr': [0.001, 0.0005], 'hidden_channels': [128, 256]},
        'GNN_Embedder_GAE_Graphlets': {'lr': [0.01, 0.005], 'hidden_channels': [128, 256], 'emb_dim': [32, 64]},
        'GNN_Embedder_DGI_Graphlets': {'lr': [0.001, 0.0005], 'hidden_channels': [128, 256]},
    }
    
    all_tuning_results = []
    k_values_for_eval = [3, 4, 5, 6, 7] 

    for model_name, grid in param_grid.items():
        keys, values = zip(*grid.items())
        print(f"\n### Tuning {model_name} ###")

        for param_combination in itertools.product(*values):
            params = dict(zip(keys, param_combination))
            print(f"\nTesting params: {params}")
            
            if model_name == 'GNN_Embedder_GAE':
                model = GNNEmbedder(**params, force_retrain=True)
            elif model_name == 'GNN_Embedder_DGI':
                model = DGIEmbedder(in_channels=data.num_features, **params, force_retrain=True)
            elif model_name == 'GNN_Embedder_GAE_Graphlets':
                model = GNNEmbedderGraphlets(**params, force_retrain=True)
            elif model_name == 'GNN_Embedder_DGI_Graphlets':
                model = DGIEmbedderGraphlets(**params, force_retrain=True)
            
            best_silhouette_for_params = -2
            
            for k in k_values_for_eval:
                embeddings, role_labels = model.predict(data, k)
                if len(np.unique(role_labels.numpy())) > 1:
                    silhouette = silhouette_score(embeddings.cpu().numpy(), role_labels.cpu().numpy())
                    if silhouette > best_silhouette_for_params:
                        best_silhouette_for_params = silhouette
            
            if best_silhouette_for_params > -2:
                print(f"Best Silhouette Score for this param set: {best_silhouette_for_params:.4f}")
                all_tuning_results.append({'model_name': model_name, 'best_silhouette': best_silhouette_for_params, **params})
    
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
            best_params = clean_params(model_results.iloc[0].drop(['model_name', 'best_silhouette']).to_dict())
            print(f"\nRetraining best {model_name} with params: {best_params}")
            
            model_save_path = str(output_dir / f"best_{model_name}_model.pt")
            if model_name == 'GNN_Embedder_GAE':
                best_model = GNNEmbedder(**best_params, model_path=model_save_path, force_retrain=True)
            elif model_name == 'GNN_Embedder_DGI':
                best_model = DGIEmbedder(in_channels=data.num_features, **best_params, model_path=model_save_path, force_retrain=True)
            elif model_name == 'GNN_Embedder_GAE_Graphlets':
                best_model = GNNEmbedderGraphlets(**best_params, model_path=model_save_path, force_retrain=True)
            elif model_name == 'GNN_Embedder_DGI_Graphlets':
                best_model = DGIEmbedderGraphlets(**best_params, model_path=model_save_path, force_retrain=True)
            
            best_model.train(data) 
            print(f"Best model for {model_name} saved successfully.")

def main():
    parser = argparse.ArgumentParser(description="Run Role Discovery hyperparameter tuning.")
    parser.add_argument('--dataset', type=str, nargs='+', default=['Cora'], help="List of datasets to tune on.")
    args = parser.parse_args()

    for dataset in args.dataset:
        run_hyperparameter_tuning(dataset)
    
    print("\nAll tuning runs finished successfully.")

if __name__ == '__main__':
    main()