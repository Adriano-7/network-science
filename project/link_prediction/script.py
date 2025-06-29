from DatasetManager import DatasetManager
from models.AdamicAdar import AdamicAdarModel
from Evaluator import Evaluator

def main():
    
    print("Initializing experiment...")
    dataset_manager = DatasetManager(dataset_name='Twitch')
    model = AdamicAdarModel()
    model.train(dataset_manager.train_data)
    print("\nPredicting scores for test edges...")
    test_scores = model.predict(
        graph_data=dataset_manager.train_data,
        edges_to_predict=dataset_manager.all_test_edges
    )
    print("\n--- Model Predictions ---")
    print(f"Generated {len(test_scores)} scores for the test edges using {model.__class__.__name__}.")

    # --- Evaluating Model Performance ---
    print("\n--- Evaluating Model Performance ---")
    
    evaluator = Evaluator(dataset_name=dataset_manager.dataset_name)
    evaluator.evaluate(
        model_name=model.__class__.__name__,
        predictions=test_scores,
        ground_truth=dataset_manager.all_test_labels
    )
    
    print("\nExperiment finished.")


if __name__ == '__main__':
    main()