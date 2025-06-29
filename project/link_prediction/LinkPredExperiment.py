from DatasetManager import DatasetManager
from models.heuristics.AdamicAdar import AdamicAdarModel
from models.Evaluator import Evaluator

def main():
    print("Initializing experiment...")
    dataset_manager = DatasetManager(dataset_name='Twitch')
    
    print(f"Dataset loaded: {dataset_manager.dataset_name}")
    print(f"Train edges: {dataset_manager.all_train_edges.size(1)}")
    print(f"Val edges: {dataset_manager.all_val_edges.size(1)}")
    print(f"Test edges: {dataset_manager.all_test_edges.size(1)}")
    
    model = AdamicAdarModel()
    model.train(dataset_manager.train_data, dataset_manager.val_data)
    
    print("\nPredicting scores for test edges...")
    test_scores = model.predict(
        graph_data=dataset_manager.train_data,
        edges_to_predict=dataset_manager.all_test_edges
    )
    
    print(f"Generated {len(test_scores)} scores for the test edges using {model.__class__.__name__}.")

    evaluator = Evaluator(dataset_name=dataset_manager.dataset_name)
    evaluator.evaluate(
        model_name=model.__class__.__name__,
        predictions=test_scores,
        ground_truth=dataset_manager.all_test_labels
    )
    
    print("\nExperiment finished successfully.")

if __name__ == '__main__':
    main()