import os
import torch
import csv
from datetime import datetime
from sklearn.metrics import roc_auc_score

class Evaluator:
    def __init__(self, dataset_name: str):
        self.dataset_name = dataset_name
        self.output_dir = f"results/datasets/"
        os.makedirs(self.output_dir, exist_ok=True)
        self.results_path = os.path.join(self.output_dir, f"{dataset_name}_metrics.csv")

    def _calculate_ranking_metrics(self, y_true, y_pred, K_values=None):
        if K_values is None:
            K_values = [10, 50, 100]

        pos_scores = y_pred[y_true == 1]
        neg_scores = y_pred[y_true == 0]

        if len(pos_scores) == 0 or len(neg_scores) == 0:
            return {f'Hits@{k}': 0.0 for k in K_values}, 0.0

        hits_at_k = {}
        for k in K_values:
            k = min(k, len(neg_scores))
            num_better = (pos_scores.view(-1, 1) > neg_scores.view(1, -1)).float().sum(dim=1)
            hits_at_k[f'Hits@{k}'] = (num_better >= (len(neg_scores) - k + 1)).float().mean().item()

        ranks = (pos_scores.view(-1, 1) <= neg_scores.view(1, -1)).float().sum(dim=1) + 1
        mrr = (1.0 / ranks).mean().item()

        return hits_at_k, mrr

    def evaluate(self, model_name: str, predictions: torch.Tensor, ground_truth: torch.Tensor):
        try:
            y_true = ground_truth.cpu()
            y_pred = predictions.cpu()

            auc = roc_auc_score(y_true, y_pred)
            hits, mrr = self._calculate_ranking_metrics(y_true, y_pred)

            print("\n--- Evaluation Report ---")
            print(f"Model: {model_name}")
            print(f"AUC Score: {auc:.4f}")
            print(f"Mean Reciprocal Rank (MRR): {mrr:.4f}")
            for k, v in hits.items():
                print(f"{k}: {v:.4f}")
            print("-" * 25)

            results_dict = {
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'model_name': model_name,
                'AUC': f"{auc:.4f}",
                'MRR': f"{mrr:.4f}",
                **{k: f"{v:.4f}" for k, v in hits.items()}
            }
            
            header = list(results_dict.keys())
            write_header = not os.path.exists(self.results_path)
            
            with open(self.results_path, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=header)
                if write_header:
                    writer.writeheader()
                writer.writerow(results_dict)
                
            print(f"Evaluation report appended to {self.results_path}")
            
        except Exception as e:
            print(f"Error during evaluation: {e}")
            raise