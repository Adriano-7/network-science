import os
import pandas as pd
import numpy as np
import torch
from torch_geometric.datasets import Planetoid, GNNBenchmarkDataset, Actor

def get_dataset(name: str, root: str = '/tmp/'):
    path = os.path.join(root, name)
    if name.lower() == 'cora':
        return Planetoid(root=path, name='Cora')
    elif name.lower() == 'cluster':
        return GNNBenchmarkDataset(root=path, name='CLUSTER')
    elif name.lower() == 'actor':
        return Actor(root=path)
    else:
        raise ValueError(f"Unknown dataset: {name}")

def clean_params(params: dict) -> dict:
    cleaned = {}
    for k, v in params.items():
        if pd.isna(v):
            continue
        native_v = v.item() if isinstance(v, np.generic) else v
        if k in ['hidden_channels', 'emb_dim']:
            cleaned[k] = int(native_v)
        else:
            cleaned[k] = native_v
    return cleaned