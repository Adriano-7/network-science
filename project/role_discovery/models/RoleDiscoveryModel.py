from abc import ABC, abstractmethod
import torch
from torch_geometric.data import Data

class RoleDiscoveryModel(ABC):
    """Abstract base class for all role discovery models."""

    def train(self, graph_data: Data):
        """
        Models that do not require training can just pass.
        """
        pass

    @abstractmethod
    def predict(self, graph_data: Data, k: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Assigns each node in the graph to one of k roles.
        """
        pass