import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def visualize_roles_tsne(embeddings: torch.Tensor, role_labels: torch.Tensor, title: str):
    """
    Visualizes node embeddings in 2D using t-SNE, colored by their assigned role.

    Args:
        embeddings (torch.Tensor): The node embeddings (N x F).
        role_labels (torch.Tensor): The role ID for each node (N).
        title (str): The title for the plot.
    """
    print(f"Generating t-SNE visualization for '{title}'...")
    
    embeddings_np = embeddings.cpu().detach().numpy()
    labels_np = role_labels.cpu().numpy()
    
    tsne = TSNE(n_components=2, perplexity=30, learning_rate='auto', 
                max_iter=1000, random_state=42, init='pca')
    embeddings_2d = tsne.fit_transform(embeddings_np)
    
    plt.figure(figsize=(12, 10))
    unique_labels = np.unique(labels_np)
    
    colors = plt.get_cmap('rocket', len(unique_labels))
    
    for i, label in enumerate(unique_labels):
        idx = labels_np == label
        plt.scatter(embeddings_2d[idx, 0], embeddings_2d[idx, 1], 
                    color=colors(i), label=f'Role {label}', alpha=0.8)
            
    plt.title(title, fontsize=18, pad=20)
    plt.xlabel("t-SNE Dimension 1", fontsize=14)
    plt.ylabel("t-SNE Dimension 2", fontsize=14)
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.show()