from abc import ABC, abstractmethod
from typing import Tuple
import torch
from torch_geometric.data import Data

class RoleDiscoveryModel(ABC):
    def train(self, graph_data: Data):
        """
        Optional training step for models that require it.
        """
        pass

    @abstractmethod
    def predict(self, graph_data: Data, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Assigns each node in the graph to one of k roles.
        """
        pass