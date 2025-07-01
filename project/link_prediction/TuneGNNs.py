import os
import csv
import torch
import optuna
from datetime import datetime
from DatasetManager import DatasetManager
from models.deep_learning.GCN import GCNModel
from models.deep_learning.GraphSAGE import GraphSAGEModel

if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")


def calculate_mrr(model, train_data, val_data):
    model.model.eval()
    with torch.no_grad():
        train_data = train_data.to(device)
        val_data = val_data.to(device)
        z = model.model.encoder(train_data.x, train_data.edge_index)
        
        val_edges = torch.cat([val_data.pos_edge_label_index, val_data.neg_edge_label_index], dim=-1)
        val_labels = torch.cat([val_data.pos_edge_label, val_data.neg_edge_label], dim=0)
        
        scores = model.model.decoder(z, val_edges).sigmoid()

        y_true = val_labels.cpu()
        y_pred = scores.cpu()
        pos_scores = y_pred[y_true == 1]
        neg_scores = y_pred[y_true == 0]

        if len(pos_scores) == 0 or len(neg_scores) == 0:
            return 0.0

        ranks = (pos_scores.view(-1, 1) <= neg_scores.view(1, -1)).float().sum(dim=1) + 1
        mrr = (1.0 / ranks).mean().item()
        
        return mrr

def create_objective(model_class, dataset_manager, num_node_features):
    train_data = dataset_manager.train_data
    val_data = dataset_manager.val_data

    def objective(trial: optuna.trial.Trial) -> float:
        params = {
            'lr': trial.suggest_float('lr', 1e-4, 1e-2, log=True),
            'hidden_channels': trial.suggest_categorical('hidden_channels', [64, 128, 256]),
            'emb_dim': trial.suggest_categorical('emb_dim', [32, 64, 128]),
            'dropout': trial.suggest_float('dropout', 0.2, 0.6),
        }

        try:
            model = model_class(
                in_channels=num_node_features,
                hidden_channels=params['hidden_channels'],
                emb_dim=params['emb_dim'],
                dropout=params['dropout'],
                lr=params['lr'],
                epochs=200,
                patience=20
            )
            model.train(train_data, val_data)
            mrr_score = calculate_mrr(model, train_data, val_data)

            print(f"Trial {trial.number} finished. MRR: {mrr_score:.4f} with params: {params}")
            return mrr_score

        except Exception as e:
            print(f"Trial {trial.number} failed with error: {e}")
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


def main():
    DATASET_TO_TUNE = 'Citeseer'  # Options: 'Cora', 'Twitch-DE', 'Twitch-EN'
    NUM_TRIALS = 50

    print(f"Starting hyperparameter tuning on the '{DATASET_TO_TUNE}' dataset.")
    print(f"Device: {device}")
    
    dataset_manager = DatasetManager(dataset_name=DATASET_TO_TUNE, seed=42)
    num_node_features = dataset_manager.train_data.num_features
    output_file = "results/tune/gnns.csv"

    models_to_tune = {
        'GCNModel': GCNModel,
        'GraphSAGEModel': GraphSAGEModel
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
        study.optimize(objective_fn, n_trials=NUM_TRIALS, timeout=3600)

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
    main()