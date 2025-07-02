import os
import csv
import torch
import optuna
from datetime import datetime
from DatasetManager import DatasetManager
from models.deep_learning.GCN1 import GCNModel1
from models.deep_learning.GraphSAGE1 import GraphSAGEModel1
from models.deep_learning.GCN2 import GCNModel2
from models.deep_learning.GraphSAGE2 import GraphSAGEModel2
from models.deep_learning.SEAL import SEALModel

if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

def create_objective(model_class, dataset_manager, num_node_features):
    train_data = dataset_manager.train_data
    val_data = dataset_manager.val_data

    def objective(trial: optuna.trial.Trial) -> float:
        params = {
            'lr': trial.suggest_float('lr', 1e-4, 5e-3, log=True),
            'hidden_channels': trial.suggest_categorical('hidden_channels', [32, 64]),
            'emb_dim': trial.suggest_categorical('emb_dim', [16, 32]),
            'dropout': trial.suggest_float('dropout', 0.2, 0.6),
        }

        model_name = model_class.__name__
        if 'SEAL' in model_name:
            params['num_hops'] = trial.suggest_categorical('num_hops', [1, 2])
            params['use_feature'] = trial.suggest_categorical('use_feature', [True, False])
            
            drnl_feature_dim = 2
            if params['use_feature']:
                in_channels = num_node_features + drnl_feature_dim
            else:
                in_channels = drnl_feature_dim

        else:
            in_channels = num_node_features

        try:
            model = model_class(
                in_channels=in_channels,
                **params,
                epochs=100,
                patience=10
            )
            final_score = model.train(train_data, val_data)
            
            print(f"Trial {trial.number} finished. Final Score: {final_score:.4f} with params: {params}")
            return final_score

        except Exception as e:
            print(f"Trial {trial.number} failed with error: {e}")
            import traceback
            traceback.print_exc()
            raise optuna.exceptions.TrialPruned()

    return objective


def save_tuning_results(dataset_name, model_name, best_trial, output_file):
    results_dir = os.path.dirname(output_file)
    os.makedirs(results_dir, exist_ok=True)

    results = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'dataset': dataset_name,
        'model': model_name,
        'best_mrr_val': f"{best_trial.value:.4f}",
        **best_trial.params
    }

    header = list(results.keys())
    write_header = not os.path.exists(output_file)
    with open(output_file, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=header)
        if write_header:
            writer.writeheader()
        writer.writerow(results)
    
    print(f"Best results for {model_name} on {dataset_name} saved to {output_file}")


def main(dataset_name):
    DATASET_TO_TUNE = dataset_name
    NUM_TRIALS = 25

    print(f"Starting hyperparameter tuning on the '{DATASET_TO_TUNE}' dataset.")
    print(f"Device: {device}")
    
    dataset_manager = DatasetManager(dataset_name=DATASET_TO_TUNE, seed=42)
    num_node_features = dataset_manager.train_data.num_features
    output_file = "results/tune/gnns.csv"

    models_to_tune = {
        # 'GCNModel1': GCNModel1,
        # 'GraphSAGEModel1': GraphSAGEModel1,
        # 'GCNModel2': GCNModel2,
        # 'GraphSAGEModel2': GraphSAGEModel2,
        'SEALModel': SEALModel,
    }

    for model_name, model_class in models_to_tune.items():
        print("\n" + "="*60)
        print(f"TUNING MODEL: {model_name}")
        print("="*60)

        objective_fn = create_objective(model_class, dataset_manager, num_node_features)
        study = optuna.create_study(
            direction='maximize',
            pruner=optuna.pruners.MedianPruner(n_warmup_steps=5)
        )
        study.optimize(objective_fn, n_trials=NUM_TRIALS, timeout=7200)

        best_trial = study.best_trial
        print("\n" + "-"*30)
        print("Tuning Complete!")
        print(f"Best Trial for {model_name}:")
        print(f"  Validation MRR: {best_trial.value:.4f}")
        print("  Best Hyperparameters:")
        for key, value in best_trial.params.items():
            print(f"    {key}: {value}")
        print("-" * 30 + "\n")

        save_tuning_results(DATASET_TO_TUNE, model_name, best_trial, output_file)

    print("All tuning processes finished successfully.")


if __name__ == '__main__':
    datasets = [
        # 'Cora', 
        # 'Citeseer', 
        # 'Twitch-EN', 
        # 'Twitch-DE'
    ]
    for dataset in datasets:
        main(dataset)