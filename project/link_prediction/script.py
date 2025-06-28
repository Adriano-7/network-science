from DatasetManager import DatasetManager
from models.AdamicAdar import AdamicAdarModel

def main():
    print("Initializing experiment...")
    dataset_manager = DatasetManager(dataset_name='Twitch')
    model = AdamicAdarModel()
    print("\nPredicting scores for test edges...")
    test_scores = model.predict(
        graph_data=dataset_manager.train_data,
        edges_to_predict=dataset_manager.test_data.edge_label_index
    )
    print("\n--- Model Predictions ---")
    print(f"Generated {len(test_scores)} scores for the test edges using {model.__class__.__name__}.")
    print("Example scores:", test_scores[:10])
    # evaluate the model
    print("\nExperiment finished.")


if __name__ == '__main__':
    main()