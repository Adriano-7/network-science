from DatasetManager import DatasetManager
from models.Evaluator import Evaluator
from models.LinkPredModel import LinkPredictionModel
from models.heuristics.AdamicAdar import AdamicAdarModel
from models.heuristics.CommonNeighbors import CommonNeighborsModel
from models.heuristics.JaccardIndex import JaccardIndexModel
from models.heuristics.PreferentialAttachment import PreferentialAttachmentModel

def run_experiment(model: LinkPredictionModel, dataset_manager: DatasetManager, evaluator: Evaluator):

    print("\n" + "="*50)
    print(f"Starting experiment for model: {model.__class__.__name__}")
    print("="*50)

    model.train(dataset_manager.train_data, dataset_manager.val_data)
    
    print(f"\nPredicting scores for test edges using {model.__class__.__name__}...")
    test_scores = model.predict(
        graph_data=dataset_manager.train_data,
        edges_to_predict=dataset_manager.all_test_edges
    )
    
    if test_scores.size(0) != dataset_manager.all_test_labels.size(0):
        print(f"Warning: Mismatch in number of predictions ({test_scores.size(0)}) and ground truth labels ({dataset_manager.all_test_labels.size(0)}).")
        return

    print(f"Generated {len(test_scores)} scores for the test edges.")

    evaluator.evaluate(
        model_name=model.__class__.__name__,
        predictions=test_scores,
        ground_truth=dataset_manager.all_test_labels
    )
    print(f"Finished experiment for model: {model.__class__.__name__}")


def main():
    print("Initializing experiment suite...")
    dataset_manager = DatasetManager(dataset_name='Twitch', seed=42)
    evaluator = Evaluator(dataset_name=dataset_manager.dataset_name)
    
    print(f"Dataset loaded: {dataset_manager.dataset_name}")
    print(f"Total nodes: {dataset_manager.train_data.num_nodes}")
    print(f"Train edges (for message passing): {dataset_manager.train_data.edge_index.size(1)}")
    print(f"Validation edges (pos+neg): {dataset_manager.all_val_edges.size(1)}")
    print(f"Test edges (pos+neg): {dataset_manager.all_test_edges.size(1)}")
    
    models_to_run = [
        AdamicAdarModel(),
        CommonNeighborsModel(),
        JaccardIndexModel(),
        PreferentialAttachmentModel(),
    ]

    for model in models_to_run:
        run_experiment(model, dataset_manager, evaluator)
    
    print("\nAll experiments finished successfully.")

if __name__ == '__main__':
    main()