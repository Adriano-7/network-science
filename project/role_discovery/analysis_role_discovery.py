#!/usr/bin/env python
# coding: utf-8

# # Role Discovery: Result Analysis
# 
# The primary objective is to compare the efficacy of two distinct approaches:
# 
# 1.  **Traditional Method (`Feature-Based_Roles`)**: This method involves extracting a vector of structural features (e.g., centrality metrics, clustering coefficient) for each node and then applying K-Means clustering.
# 2.  **GNN-based Methods (`GNN_Embedder_GAE`, `GNN_Embedder_DGI`)**: These methods use an unsupervised Graph Neural Network (a Graph Autoencoder or Deep Graph Infomax model) to learn low-dimensional node embeddings. These embeddings, which capture the structural context of nodes, are then clustered using K-Means.
# 
# The analysis is performed on three datasets: **Cora**, **Actor**, and **CLUSTER**. We will evaluate the models based on internal clustering metrics and qualitative analysis, aiming to answer the following key questions:
# 
#   - Which method produces better-separated roles (clusters)?
#   - How do we determine the optimal number of roles, $k$?
#   - Can we assign meaningful, interpretable labels (e.g., "hub," "bridge," "periphery") to the discovered roles?
# 

# In[2]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import warnings

warnings.filterwarnings('ignore', category=FutureWarning)
sns.set_theme(style="whitegrid", palette="magma")
plt.rcParams['figure.figsize'] = (12, 7)
plt.rcParams['font.size'] = 12

RESULTS_DIR = Path("../results/role_discovery/")
DATASETS = ['Cora', 'Actor', 'CLUSTER']
MODELS = ["Feature-Based_Roles", "GNN_Embedder_GAE", "GNN_Embedder_DGI"]

print(f"Analysis Notebook Setup Complete.")
print(f"Results Directory: {RESULTS_DIR.resolve()}")
print(f"Datasets to Analyze: {DATASETS}")


# ## 1\. High-Level Comparison Across All Datasets
# 

# In[3]:


summary_path = RESULTS_DIR / "comparison_summary.csv"
if not summary_path.exists():
    print(f"Error: The summary file was not found at {summary_path}")
    print("Please run `python -m project.role_discovery.generate_report` first.")
    summary_df = pd.DataFrame()
else:
    summary_df = pd.read_csv(summary_path)

print("--- Overall Model Performance (based on best Silhouette Score) ---")
display(summary_df.style.format({
    'Silhouette Score': '{:.4f}',
    'Davies-Bouldin Index': '{:.4f}',
    'Calinski-Harabasz Index': '{:,.2f}'
}))


# ### 1.1. Visualizing Overall Performance
# 
# We will create a separate plot for each key metric.
#   - **Silhouette Score**: Higher is better. Measures how similar a node is to its own role compared to other roles.
#   - **Davies-Bouldin Index**: Lower is better. Measures the average similarity ratio of each role with its most similar one.
#   - **Calinski-Harabasz Index**: Higher is better. The ratio of between-cluster dispersion to within-cluster dispersion.
# 

# In[4]:


if not summary_df.empty:
    fig, axes = plt.subplots(1, 3, figsize=(24, 7))
    fig.suptitle('Overall Model Performance Comparison', fontsize=20, y=1.03)

    metrics = [("Silhouette Score", "Higher is Better"), 
               ("Davies-Bouldin Index", "Lower is Better"), 
               ("Calinski-Harabasz Index", "Higher is Better")]

    for i, (metric, interpretation) in enumerate(metrics):
        sns.barplot(data=summary_df, x='Dataset', y=metric, hue='Model', ax=axes[i])
        axes[i].set_title(f'{metric}\n({interpretation})', fontsize=14)
        axes[i].set_xlabel("Dataset", fontsize=12)
        axes[i].set_ylabel("Score", fontsize=12)
        axes[i].tick_params(axis='x', rotation=15)
        if i == 0:
            axes[i].legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')
        else:
            axes[i].get_legend().remove()

    plt.tight_layout()
    plt.show()


# **Interpretation:**
# The visualizations clearly show that the GNN-based methods (GAE and DGI) consistently outperform the traditional `Feature-Based_Roles` approach across all datasets and nearly all metrics. Between the two GNN methods, there isn't a single clear winner; their performance is competitive, with GAE slightly ahead expecially for the Actor and cluster datasets.

# ## 2\. Deep Dive: Analysis of the "Actor" Dataset
# 
# Now, we perform a analysis on a single dataset to understand the results more deeply. We choose the **Actor** dataset as it represents a real-world network where roles are likely to be complex and interesting.
# 
# 
# ### 2.1. Finding the Optimal Number of Roles ($k$)
# 
# To select the best number of roles, $k$, for each model, we plot the Silhouette Score against different values of $k$ that were tested. A peak or an "elbow" in the plot suggests an optimal value for $k$.
# 

# In[5]:


fig, ax = plt.subplots(figsize=(14, 7))

for model_name in MODELS:
    metrics_path = RESULTS_DIR / f"Actor/{model_name}_clustering_metrics.csv"
    if metrics_path.exists():
        df = pd.read_csv(metrics_path)
        ax.plot(df['k'], df['Silhouette Score'], marker='o', linestyle='-', label=model_name)

ax.set_title('Silhouette Score vs. Number of Roles (k) on Actor Dataset', fontsize=16)
ax.set_xlabel('Number of Roles (k)', fontsize=12)
ax.set_ylabel('Silhouette Score (Higher is better)', fontsize=12)
ax.set_xticks(df['k'].unique())
ax.grid(True, which='both', linestyle='--', linewidth=0.5)
ax.legend(title="Model")
plt.show()


# **Interpretation:**
# 
# For the "Actor" dataset, the GNN models (`GAE` and `DGI`) show their best performance at $k=3$, after which the score begins to decline. This indicates that three is likely the most natural number of distinct structural roles in this graph. The `Feature-Based_Roles` method peaks later at $k=5$, its much lower score implies these groupings are less coherent.

# ### 2.2. Visualizing the Discovered Roles with t-SNE
# t-SNE is a dimensionality reduction technique that allows us to visualize the high-dimensional node embeddings (or feature vectors) in 2D. In a good model, the nodes belonging to the same role should form distinct, well-separated visual clusters.
# 
# We will load and display the pre-generated t-SNE plots for the best $k$ value of each model.
# 

# In[ ]:


import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pathlib import Path
import pandas as pd

fig, axes = plt.subplots(1, 3, figsize=(24, 8))
fig.suptitle("Comparison of t-SNE Role Visualizations on the 'Actor' Dataset", fontsize=22)

best_k_series = summary_df[summary_df['Dataset'] == 'Actor'].set_index('Model')['Best k']

for ax, model_name in zip(axes.flatten(), MODELS):
    if model_name in best_k_series:
        k = best_k_series[model_name]
        image_path = RESULTS_DIR / f"Actor/{model_name}_k{k}_tsne.png"

        ax.set_title(f"{model_name}\n(Best k={k})", fontsize=16, pad=10)

        if image_path.exists():
            img = mpimg.imread(image_path)
            ax.imshow(img)
        else:
            ax.text(0.5, 0.5, f'Image not found', ha='center', va='center', fontsize=12)

    ax.axis('off')

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()


# 
# **Interpretation:** 
# The t-SNE visualizations provide strongly reinforcing the quantitative metrics. The `Feature-Based_Roles` model results in a chaotic plot with heavily overlapping and poorly defined clusters, visually explaining its low Silhouette Score. In stark contrast, both the `GNN_Embedder_GAE` and `GNN_Embedder_DGI` models generate remarkably clean plots with distinct, well-separated groups, confirming that their learned embeddings are far more effective for discovering meaningful structural roles.
# 
# Comparing the top two methods, the GAE model produces exceptionally sharp, linear structures, while the DGI model forms more globular clusters. Both are excellent representations, but the GAE’s visual separation is slightly crisper, aligning with its higher score. Ultimately, these visualizations provide powerful qualitative proof that the GNNs successfully learned low-dimensional embeddings that effectively capture and separate the distinct structural roles within the Actor network.

# ### 2.3. Interpreting Role Characteristics
# Moving beyond scores and visualizations to understand *what these roles represent*. We load the analysis files, which contain the average structural properties (degree, betweenness, etc.) for nodes within each role. By examining these properties, we can assign intuitive labels.

# In[9]:


def plot_role_profiles(df, dataset_name, model_name, k, ax):
    metrics_to_plot = ['avg_degree', 'avg_betweenness', 'avg_closeness', 'avg_eigenvector', 'avg_clustering_coeff']
    profiles = df[metrics_to_plot]
    profiles_normalized = (profiles - profiles.min()) / (profiles.max() - profiles.min())
    profiles_normalized = profiles_normalized.fillna(0) 

    labels = profiles_normalized.columns
    num_vars = len(labels)

    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1] 

    cmap = plt.get_cmap('plasma')
    colors = cmap(np.linspace(0, 1, len(df)))

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    ax.set_rlabel_position(0)
    ax.set_xticks(angles[:-1], labels, size=10)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8"], color="grey", size=8)

    for i, row in profiles_normalized.iterrows():
        values = row.values.flatten().tolist()
        values += values[:1] 
        ax.plot(angles, values, color=colors[i], linewidth=2, linestyle='solid', label=f"Role {df.at[i, 'role_id']}")
        ax.fill(angles, values, color=colors[i], alpha=0.25)

    ax.set_title(f"{model_name} (k={k})", pad=20, fontsize=14)
    ax.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))

fig, axes = plt.subplots(1, 3, figsize=(22, 7), subplot_kw=dict(polar=True))
fig.suptitle('Structural Role Profiles on Actor Dataset', fontsize=20, y=1.1)

for i, model_name in enumerate(MODELS):
    if model_name in best_k_series:
        k = best_k_series[model_name]
        analysis_path = RESULTS_DIR / f"Actor/{model_name}_k{k}_role_analysis.csv"
        if analysis_path.exists():
            role_df = pd.read_csv(analysis_path)
            print(f"\n--- Analysis for {model_name} (k={k}) ---")
            display(role_df)
            plot_role_profiles(role_df, "Actor", model_name, k, axes[i])
        else:
            print(f"Analysis file not found for {model_name}")
            fig.delaxes(axes[i]) 

plt.tight_layout()
plt.show()


# **Interpretation of GNN\_Embedder\_GAE (k=3) on Actor:**
# 
# Based on the table and the radar chart for the `GNN_Embedder_GAE` model:
# 
#   - **Role 0 (Periphery Nodes, \~6737 nodes)**: This is the largest group. It is characterized by extremely low values across all centrality metrics (`avg_degree`, `avg_betweenness`, `avg_closeness`, `avg_eigenvector`). These are the vast majority of actors who are not well-connected and exist on the fringes of the network.
#   - **Role 1 (Connectors/Brokers, \~862 nodes)**: This group has moderately high values for `avg_degree` and `avg_betweenness` relative to the periphery. The high betweenness suggests these actors play a crucial role in connecting different parts of the network, acting as bridges or brokers between different communities (e.g., genres or film series).
#   - **Role 2 (Super-Hub, 1 node)**: This role contains a single node with exceptionally high values for every single centrality metric. Its `avg_betweenness` and `avg_eigenvector` scores dominate all others. This is the ultimate "hub" of the network, a superstar actor who is not only highly connected but is also connected to other important actors, making them central to the entire graph structure.
# 

# ## 3\. Conclusions
# 
# The analysis yields several key conclusions:
# 
# 1.  **GNNs are Superior for Role Discovery**: Both GNN-based methods, GAE and DGI, significantly outperformed the traditional feature-based approach. The learned embeddings capture complex structural patterns that are missed by a hand-picked set of metrics, resulting in more cohesive and well-separated roles.
# 
# 2.  **Role Interpretability is High**: By analyzing the structural characteristics of the nodes within each GNN-discovered role, we were able to assign meaningful labels like "hub," "broker," and "periphery." This confirms that the unsupervised learning process is identifying functionally relevant node groups.
