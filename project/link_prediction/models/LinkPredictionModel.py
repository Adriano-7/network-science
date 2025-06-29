from abc import ABC, abstractmethod
from torch_geometric.data import Data
import torch

class LinkPredictionModel(ABC):
    """
    Abstract base class for all link prediction models.
    """

    @abstractmethod
    def train(self, train_data: Data = None, val_data: Data = None):
        pass

    @abstractmethod
    def predict(self, graph_data: Data, edges_to_predict: torch.Tensor) -> torch.Tensor:
        pass