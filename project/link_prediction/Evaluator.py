import os
import torch
import csv
from datetime import datetime
from sklearn.metrics import roc_auc_score

class Evaluator:
    """
    Handles the evaluation of link prediction models and saves results to a CSV file.
    """
    def __init__(self, dataset_name: str):
        """
        Args:
            dataset_name (str): The name of the dataset, used for saving results.
        """
        self.dataset_name = dataset_name
        # The user's provided code used 'results/', so we will adhere to that.
        self.output_dir = f"results/{self.dataset_name}"
        os.makedirs(self.output_dir, exist_ok=True)
        # Change the output file to a .csv
        self.results_path = os.path.join(self.output_dir, "evaluation_report.csv")

    def _calculate_ranking_metrics(self, y_true, y_pred, K_values=None):
        """
        Calculates Hits@K and Mean Reciprocal Rank (MRR).
        """
        if K_values is None:
            K_values = [10, 50, 100]

        pos_scores = y_pred[y_true == 1]
        neg_scores = y_pred[y_true == 0]

        hits_at_k = {}
        for k in K_values:
            # In case there are fewer negative samples than K
            eff_k = min(k, len(neg_scores))
            num_hits = (pos_scores.view(-1, 1) > neg_scores.view(1, -1)).float().sum(dim=1)
            hits_at_k[f'Hits@{k}'] = (num_hits >= (len(neg_scores) - eff_k)).float().mean().item()

        ranks = (pos_scores.view(-1, 1) <= neg_scores.view(1, -1)).float().sum(dim=1) + 1
        mrr = (1.0 / ranks).mean().item()

        return hits_at_k, mrr

    def evaluate(self, model_name: str, predictions: torch.Tensor, ground_truth: torch.Tensor):
        """
        Runs all evaluations, prints the results, and saves them to a CSV file.

        Args:
            model_name (str): The name of the model being evaluated.
            predictions (torch.Tensor): The prediction scores from the model.
            ground_truth (torch.Tensor): The ground truth labels (0 or 1).
        """
        y_true = ground_truth.cpu()
        y_pred = predictions.cpu()

        # --- Calculate all metrics ---
        auc = roc_auc_score(y_true, y_pred)
        hits, mrr = self._calculate_ranking_metrics(y_true, y_pred)

        # --- Print a formatted report to the console ---
        print("\n--- Evaluation Report ---")
        print(f"Model: {model_name}")
        print(f"AUC Score: {auc:.4f}")
        print(f"Mean Reciprocal Rank (MRR): {mrr:.4f}")
        for k, v in hits.items():
            print(f"{k}: {v:.4f}")
        print("-" * 25)

        # --- Save results to CSV ---
        results_dict = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'model_name': model_name,
            'AUC': f"{auc:.4f}",
            'MRR': f"{mrr:.4f}",
            **{k: f"{v:.4f}" for k, v in hits.items()}
        }
        
        header = results_dict.keys()
        write_header = not os.path.exists(self.results_path)
        with open(self.results_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=header)
            if write_header:
                writer.writeheader()
            writer.writerow(results_dict)
        print(f"Evaluation report appended to {self.results_path}")