import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def visualize_roles_tsne(embeddings: torch.Tensor, role_labels: torch.Tensor, title: str,
                         score: float, save_path: str = None):
    """
    Visualizes node embeddings in 2D using t-SNE, colored by their assigned role.
    """
    print(f"Generating t-SNE visualization for: {title}")
    
    embeddings_np = embeddings.cpu().detach().numpy()
    labels_np = role_labels.cpu().numpy()
    
    tsne = TSNE(n_components=2, perplexity=30, learning_rate='auto', 
                max_iter=1000, random_state=42, init='pca')
    embeddings_2d = tsne.fit_transform(embeddings_np)
    
    plt.figure(figsize=(12, 10))
    unique_labels = np.unique(labels_np)
    
    colors = plt.get_cmap('rocket', len(unique_labels))
    
    for i, label in enumerate(unique_labels):
        idx = (labels_np == label)
        plt.scatter(embeddings_2d[idx, 0], embeddings_2d[idx, 1], 
                    color=colors(i), label=f'Role {label}', alpha=0.8, s=50)
            
    plt.title(f"{title}\nSilhouette Score: {score:.4f}", fontsize=18, pad=20)
    plt.xlabel("t-SNE Dimension 1", fontsize=14)
    plt.ylabel("t-SNE Dimension 2", fontsize=14)
    plt.legend(title='Discovered Roles', fontsize=12, title_fontsize=14)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Plot saved to {save_path}")
        plt.close()  
    else:
        plt.show()