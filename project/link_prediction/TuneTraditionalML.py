import os
import json
import torch
import optuna
from datetime import datetime
from DatasetManager import DatasetManager
from models.traditional_ml.DecisionTree import DecisionTreeModel
from models.traditional_ml.LogisticRegression import LogisticRegressionModel
from models.traditional_ml.RandomForest import RandomForestModel
from models.traditional_ml.KNN import KNNModel

def calculate_mrr(predictions: torch.Tensor, ground_truth: torch.Tensor) -> float:
    """Calculates the Mean Reciprocal Rank (MRR) for a set of predictions."""
    y_true = ground_truth.cpu()
    y_pred = predictions.cpu()
    
    pos_scores = y_pred[y_true == 1]
    neg_scores = y_pred[y_true == 0]

    if len(pos_scores) == 0 or len(neg_scores) == 0:
        return 0.0

    ranks = (pos_scores.view(-1, 1) <= neg_scores.view(1, -1)).float().sum(dim=1) + 1
    mrr = (1.0 / ranks).mean().item()
    return mrr

def get_model_and_params(model_name: str, trial: optuna.trial.Trial):
    """Suggests hyperparameters and instantiates a model for a given trial."""
    if model_name == 'DecisionTree':
        params = {
            'criterion': trial.suggest_categorical('criterion', ['gini', 'entropy']),
            'max_depth': trial.suggest_int('max_depth', 3, 25),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 30),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 30),
            'random_state': 42
        }
        model = DecisionTreeModel(**params)
    elif model_name == 'LogisticRegression':
        params = {
            'C': trial.suggest_float('C', 1e-4, 1e2, log=True),
            'penalty': 'l2',
            'solver': trial.suggest_categorical('solver', ['lbfgs', 'liblinear', 'newton-cg', 'sag']),
            'random_state': 42
        }
        model = LogisticRegressionModel(**params)
    elif model_name == 'RandomForest':
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 400, step=25),
            'max_depth': trial.suggest_int('max_depth', 5, 50),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20),
            'criterion': trial.suggest_categorical('criterion', ['gini', 'entropy']),
            'random_state': 42,
            'n_jobs': -1
        }
        model = RandomForestModel(**params)
    elif model_name == 'KNN':
        params = {
            'n_neighbors': trial.suggest_int('n_neighbors', 3, 50),
            'weights': trial.suggest_categorical('weights', ['uniform', 'distance']),
            'metric': trial.suggest_categorical('metric', ['euclidean', 'manhattan', 'minkowski']),
            'n_jobs': -1
        }
        model = KNNModel(**params)
    else:
        raise ValueError(f"Unknown model name for tuning: {model_name}")

    return model, params

def create_objective(model_name, dataset_manager):
    """Creates an Optuna objective function for a given model."""
    train_data = dataset_manager.train_data
    val_edges = dataset_manager.all_val_edges
    val_labels = dataset_manager.all_val_labels

    def objective(trial: optuna.trial.Trial) -> float:
        try:
            model, params = get_model_and_params(model_name, trial)
            print(f"\n[Trial {trial.number}] Testing {model_name} with params: {params}")
            
            # Train model on the full training set
            model.train(train_data)

            # Predict on the validation set
            val_scores = model.predict_edges(train_data, val_edges)
            
            # Calculate MRR score for evaluation
            mrr_score = calculate_mrr(val_scores, val_labels)

            print(f"  > Trial {trial.number} for {model_name} finished. Validation MRR: {mrr_score:.4f}")
            return mrr_score
        except Exception as e:
            print(f"  > Trial {trial.number} for {model_name} FAILED with error: {e}")
            raise optuna.exceptions.TrialPruned()
            
    return objective

def save_tuning_results(dataset_name, model_name, best_trial, output_file):
    """Saves the best trial's results to a JSON file."""
    results_dir = os.path.dirname(output_file)
    os.makedirs(results_dir, exist_ok=True)

    new_result = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'dataset': dataset_name,
        'model': model_name,
        'best_mrr_val': f"{best_trial.value:.4f}",
        'params': best_trial.params 
    }

    all_results = []
    if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
        with open(output_file, 'r') as f:
            try:
                all_results = json.load(f)
                if not isinstance(all_results, list):
                    all_results = []
            except json.JSONDecodeError:
                all_results = []
    
    all_results.append(new_result)
    
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=4)
    
    print(f"Best results for {model_name} on {dataset_name} saved to {output_file}")


def main(dataset_name: str):
    DATASET_TO_TUNE = dataset_name
    NUM_TRIALS = 100
    OUTPUT_FILE = "results/tune/traditional_ml.json"

    print(f"Starting hyperparameter tuning on the '{DATASET_TO_TUNE}' dataset for traditional ML models.")
    
    dataset_manager = DatasetManager(dataset_name=DATASET_TO_TUNE, seed=42)

    models_to_tune = [
        # 'DecisionTree', 
        # 'LogisticRegression', 
        # 'RandomForest',
        'KNN'
    ]

    for model_name in models_to_tune:
        print("\n" + "="*60)
        print(f"TUNING MODEL: {model_name}")
        print("="*60)

        objective_fn = create_objective(model_name, dataset_manager)
        study = optuna.create_study(direction='maximize')
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

        save_tuning_results(DATASET_TO_TUNE, model_name, best_trial, OUTPUT_FILE)

    print("All traditional ML tuning processes finished successfully.")

if __name__ == '__main__':
    datasets = ['Cora', 'Citeseer', 'Twitch-EN', 'Twitch-DE']
    for dataset in datasets:
        main(dataset)