from abc import ABC, abstractmethod
from torch_geometric.data import Data
import torch

class LinkPredictionModel(ABC):
    """
    Abstract base class for all link prediction models.

    This class defines the common interface that all models (Heuristic,
    Traditional ML, and GNN-based) must implement. This allows for them
    to be used interchangeably in the evaluation pipeline.
    """

    @abstractmethod
    def train(self, train_data: Data, val_data: Data = None):
        """
        Trains the link prediction model.

        For heuristic models, this method might be empty. For GNNs, this
        is where the training loop will be implemented.

        Args:
            train_data (Data): The training graph data.
            val_data (Data): The validation data, used for monitoring model performance.
        """
        pass

    @abstractmethod
    def predict(self, data: Data) -> torch.Tensor:
        """
        Makes predictions for the links specified in the data object.

        Args:
            data (Data): A data object containing the graph structure and
                         the `edge_label_index` of edges to be scored.

        Returns:
            torch.Tensor: A tensor of scores for each edge in `data.edge_label_index`.
                          Higher scores indicate a higher probability of a link.
        """
        pass