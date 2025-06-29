import torch
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid, Twitch

class DatasetManager:
    def __init__(self, dataset_name: str, path: str = '/tmp/', seed: int = 42, 
                 val_ratio: float = 0.1, test_ratio: float = 0.15):
        self.dataset_name = dataset_name
        self.path = path
        self.seed = seed
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self._load_and_split()
        self._prepare_edge_sets()

    def _load_and_split(self):
        print(f"Loading {self.dataset_name} dataset...")

        if self.dataset_name.lower() == 'twitch':
            dataset = Twitch(root=self.path + 'Twitch', name='EN')
        elif self.dataset_name.lower() in ['cora', 'citeseer', 'pubmed']:
            dataset = Planetoid(root=self.path + self.dataset_name, name=self.dataset_name)
        else:
            raise ValueError(f"Dataset '{self.dataset_name}' not supported by this manager.")

        if len(dataset) == 0:
            raise ValueError(f"Dataset {self.dataset_name} is empty")

        torch.manual_seed(self.seed)
        data = dataset[0]
        
        if data.edge_index.size(1) == 0:
            raise ValueError("Dataset contains no edges")
            
        transform = T.RandomLinkSplit(
            num_val=self.val_ratio,
            num_test=self.test_ratio,
            is_undirected=True,
            add_negative_train_samples=True,
            split_labels=True,
        )

        print(f"Applying link prediction split with random seed {self.seed}...")
        self.train_data, self.val_data, self.test_data = transform(data)
        print("Dataset ready. Splits are now fixed and reproducible.")

    def _prepare_edge_sets(self):
        """Prepare combined edge sets for training, validation and testing."""
        def safe_concat(pos_edges, neg_edges, pos_labels, neg_labels):
            if pos_edges.size(1) == 0 and neg_edges.size(1) == 0:
                return torch.empty((2, 0), dtype=torch.long), torch.empty(0, dtype=torch.float)
            elif pos_edges.size(1) == 0:
                return neg_edges, neg_labels
            elif neg_edges.size(1) == 0:
                return pos_edges, pos_labels
            else:
                edges = torch.cat([pos_edges, neg_edges], dim=-1)
                labels = torch.cat([pos_labels, neg_labels], dim=0)
                return edges, labels

        # Training edges (only has edge_index, not separate pos/neg)
        if hasattr(self.train_data, 'pos_edge_label_index'):
            self.all_train_edges, self.all_train_labels = safe_concat(
                self.train_data.pos_edge_label_index, self.train_data.neg_edge_label_index,
                self.train_data.pos_edge_label, self.train_data.neg_edge_label
            )
        else:
            # For training data, we only have the main edge_index
            self.all_train_edges = self.train_data.edge_index
            self.all_train_labels = torch.ones(self.train_data.edge_index.size(1))
        
        # Validation edges
        self.all_val_edges, self.all_val_labels = safe_concat(
            self.val_data.pos_edge_label_index, self.val_data.neg_edge_label_index,
            self.val_data.pos_edge_label, self.val_data.neg_edge_label
        )
        
        # Test edges
        self.all_test_edges, self.all_test_labels = safe_concat(
            self.test_data.pos_edge_label_index, self.test_data.neg_edge_label_index,
            self.test_data.pos_edge_label, self.test_data.neg_edge_label
        )