import torch
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid, Twitch

class DatasetManager:
    """
    Manages loading and splitting a graph dataset for link prediction.
    """
    def __init__(self, dataset_name: str, path: str = '/tmp/', seed: int = 42):
        self.dataset_name = dataset_name
        self.path = path
        self.seed = seed
        self._load_and_split()

        self.all_train_edges = torch.cat(
            [self.train_data.pos_edge_index, self.train_data.neg_edge_index], dim=-1
        )
        self.all_train_labels = torch.cat(
            [self.train_data.pos_edge_label, self.train_data.neg_edge_label], dim=0
        )
        self.all_val_edges = torch.cat(
            [self.val_data.pos_edge_index, self.val_data.neg_edge_index], dim=-1
        )
        self.all_val_labels = torch.cat(
            [self.val_data.pos_edge_label, self.val_data.neg_edge_label], dim=0
        )
        self.all_test_edges = torch.cat(
            [self.test_data.pos_edge_label_index, self.test_data.neg_edge_label_index], dim=-1
        )
        self.all_test_labels = torch.cat(
            [self.test_data.pos_edge_label, self.test_data.neg_edge_label], dim=0
        )

    def _load_and_split(self):
        print(f"Loading {self.dataset_name} dataset...")

        if self.dataset_name.lower() == 'twitch':
            dataset = Twitch(root=self.path + 'Twitch', name='EN')
        elif self.dataset_name.lower() in ['cora', 'citeseer', 'pubmed']:
            dataset = Planetoid(root=self.path + self.dataset_name, name=self.dataset_name)
        else:
            raise ValueError(f"Dataset '{self.dataset_name}' not supported by this manager.")

        torch.manual_seed(self.seed)
        data = dataset[0]
        transform = T.RandomLinkSplit(
            num_val=0.1,
            num_test=0.15,
            is_undirected=True,
            add_negative_train_samples=True,
            split_labels=True,
        )

        print(f"Applying link prediction split with random seed {self.seed}...")
        self.train_data, self.val_data, self.test_data = transform(data)
        print("Dataset ready. Splits are now fixed and reproducible.")