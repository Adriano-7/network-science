from DatasetManager import DatasetManager
from models.Evaluator import Evaluator
from models.LinkPredModel import LinkPredictionModel
from models.heuristics.AdamicAdar import AdamicAdarModel
from models.heuristics.CommonNeighbors import CommonNeighborsModel
from models.heuristics.JaccardIndex import JaccardIndexModel
from models.heuristics.PreferentialAttachment import PreferentialAttachmentModel
from models.traditional_ml.DecisionTree import DecisionTreeModel
from models.traditional_ml.LogisticRegression import LogisticRegressionModel
from models.traditional_ml.RandomForest import RandomForestModel
from models.traditional_ml.KNN import KNNModel
from models.deep_learning.GCN1 import GCNModel1
from models.deep_learning.GraphSAGE1 import GraphSAGEModel1
from models.deep_learning.GCN2 import GCNModel2
from models.deep_learning.GraphSAGE2 import GraphSAGEModel2


def run_experiment(model: LinkPredictionModel, dataset_manager: DatasetManager, evaluator: Evaluator):
    print("\n" + "="*50)
    print(f"Starting experiment for model: {model.__class__.__name__}")
    print("="*50)

    model.train(dataset_manager.train_data, dataset_manager.val_data)
    
    print(f"\nPredicting scores for test edges using {model.__class__.__name__}...")
    test_scores = model.predict_edges(
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


def run_experiments_on_dataset(dataset_name: str, seed: int = 42):
    print("\n" + "#"*60)
    print(f"LOADING DATASET: {dataset_name}")
    print("#"*60)
    dataset_manager = DatasetManager(dataset_name=dataset_name, seed=seed)
    evaluator = Evaluator(dataset_name=dataset_manager.dataset_name)
    
    num_node_features = dataset_manager.train_data.num_features

    print("\n" + "="*50)
    print(f"Dataset loaded: {dataset_manager.dataset_name}")
    print(f"Total nodes: {dataset_manager.train_data.num_nodes}")
    print(f"Node features: {num_node_features}")
    print(f"Train edges (for message passing): {dataset_manager.train_data.edge_index.size(1)}")
    print(f"Test edges (pos+neg): {dataset_manager.all_test_edges.size(1)}")
    print("="*50)

    dec_tree = DecisionTreeModel(
        max_depth=7,
        min_samples_split=30,
        min_samples_leaf=27,
        criterion='gini',
        random_state=seed
    )
    log_reg = LogisticRegressionModel(
        C=0.0001,
        penalty='l2',
        solver='newton-cg',
        random_state=seed
    )
    random_forest = RandomForestModel(
        n_estimators=375,
        max_depth=16,
        min_samples_split=9,
        min_samples_leaf=12,
        criterion='entropy',
        random_state=seed,
        n_jobs=-1
    )
    knn = KNNModel(
        n_neighbors=49,
        weights='uniform',
        metric='euclidean',
    )


    gcn = GCNModel1(
        in_channels=num_node_features,
        epochs=500,
        patience=50,
        dropout=0.50,
        hidden_channels=128,
        emb_dim=32,
        lr=0.005,
    )
    graphsage = GraphSAGEModel1(
        in_channels=num_node_features,
        epochs=500,
        patience=50,
        dropout=0.31,
        hidden_channels=128,
        emb_dim=128,
        lr=0.001,
    )

    models_to_run = [
        # CommonNeighborsModel(),
        # JaccardIndexModel(),
        # AdamicAdarModel(),
        # PreferentialAttachmentModel(),
        # dec_tree,
        # log_reg,
        # random_forest,
        # knn,
        gcn,
        graphsage,
    ]

    for model in models_to_run:
        run_experiment(model, dataset_manager, evaluator)

def main():
    print("Initializing experiment suite")

    datasets_to_test = [
        # 'Cora',
        # 'Citeseer',
        # 'Twitch-EN',
        'Twitch-DE',
    ]
    
    for dataset in datasets_to_test:
        run_experiments_on_dataset(dataset, seed=42)

    print("\nAll experiments finished successfully.")

if __name__ == '__main__':
    main()