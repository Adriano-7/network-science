#!/usr/bin/env python
# coding: utf-8

# # Role Discovery: Result Analysis
# 
# The primary objective is to compare the efficacy of two distinct approaches:
# 
# 1.  **Traditional Method (`Feature-Based_Roles`)**: This method involves extracting a vector of structural features (e.g., centrality metrics, clustering coefficient) for each node and then applying K-Means clustering.
# 2.  **GNNs with Simple Features**: Unsupervised GNNs (`GAE`, `DGI`) learn node embeddings from the graph structure, using only node degrees as initial features.
# 3.  **GNNs with Graphlet Features**: The same GNNs are provided with pre-computed graphlet features as input, combining engineered features with learned representations.
# 
# The analysis is performed on three datasets: **Cora**, **Actor**, and **CLUSTER**. We will evaluate the models based on internal clustering metrics and qualitative analysis, aiming to answer the following key questions:
# 
#   - Which method produces better-separated roles (clusters)?
#   - How do we determine the optimal number of roles, $k$?
#   - Can we assign meaningful, interpretable labels (e.g., "hub," "bridge," "periphery") to the discovered roles?
# 
#   - How crucial are input features for graph neural networks in the context of role discovery?

# In[1]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import warnings
import sys
from IPython.display import display

warnings.filterwarnings('ignore', category=FutureWarning)
sns.set_theme(style="whitegrid", palette="magma")
plt.rcParams['figure.figsize'] = (12, 7)
plt.rcParams['font.size'] = 12

RESULTS_DIR = Path("results/role_discovery/")
DATASETS = ['Cora', 'Actor', 'CLUSTER']
MODELS = [
    "Feature-Based_Roles", "Feature-Based_Roles_Graphlets",
    "GNN_Embedder_GAE", "GNN_Embedder_GAE_Graphlets",
    "GNN_Embedder_DGI", "GNN_Embedder_DGI_Graphlets"
]

print(f"Analysis Notebook Setup Complete.")
print(f"Results Directory: {RESULTS_DIR.resolve()}")
print(f"Datasets to Analyze: {DATASETS}")


# ## 1\. High-Level Comparison Across All Datasets
# 

# In[2]:


summary_path = RESULTS_DIR / "comparison_summary.csv"
if not summary_path.exists():
    print(f"Error: The summary file was not found at {summary_path}")
    print("Please run `python -m project.role_discovery.generate_report` first.")
    summary_df = pd.DataFrame()
else:
    summary_df = pd.read_csv(summary_path)

print("--- Overall Model Performance (based on best Silhouette Score) ---")
if not summary_df.empty:
    display(summary_df.sort_values('Silhouette Score', ascending=False).style.format({
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

# In[ ]:


if not summary_df.empty:
    fig, axes = plt.subplots(1, 3, figsize=(24, 8))
    fig.suptitle('Overall Model Performance Comparison Across Datasets', fontsize=20, y=1.03)

    metrics = [("Silhouette Score", "Higher is Better"),
               ("Davies-Bouldin Index", "Lower is Better"),
               ("Calinski-Harabasz Index", "Higher is Better")]

    for i, (metric, interpretation) in enumerate(metrics):
        sorted_df = summary_df.sort_values(by=['Dataset', 'Silhouette Score'], ascending=[True, False])
        sns.barplot(data=sorted_df, x='Dataset', y=metric, hue='Model', ax=axes[i],
                    hue_order=MODELS) 
        axes[i].set_title(f'{metric}\n({interpretation})', fontsize=14)
        axes[i].set_xlabel("Dataset", fontsize=12)
        axes[i].set_ylabel("Score", fontsize=12)
        axes[i].tick_params(axis='x', rotation=0)
        if i == 0:
            axes[i].legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')
        else:
            axes[i].get_legend().remove()

    plt.tight_layout(rect=[0, 0, 0.9, 1])
    plt.show()


# **Interpretation:**
# 
# The charts clearly demonstrate that **graphlet-based models** (e.g., *Feature-Based\_Roles\_Graphlets* and *GNN\_Embedder\_DGI\_Graphlets*) consistently outperform others across all three metrics—**Silhouette Score** (↑), **Davies-Bouldin Index** (↓), and **Calinski-Harabasz Index** (↑). This indicates they discover roles that are both cohesive and well-separated.
# 
# Notably, models that **combine graphlet features with GNNs** show the strongest results, confirming the effectiveness of **hybrid approaches**.
# 
# Performance varies by dataset. On *CLUSTER*, models perform similarly and score lower overall.
# 
# Lastly, the baseline *Feature-Based\_Roles* method performs the worst, showing that **simple centrality features are insufficient** for capturing complex role patterns.
# 

# ## 2\. Analysis of the "Actor" Dataset
# 
# Now, we perform a analysis on a single dataset to understand the results more deeply. We choose the **Actor** dataset.
# 
# 
# ### 2.1. Finding the Optimal Number of Roles ($k$)
# 
# To select the best number of roles, $k$, for each model, we plot the Silhouette Score against different values of $k$ that were tested. A peak or an "elbow" in the plot suggests an optimal value for $k$.
# 

# In[4]:


best_k_series = summary_df.set_index(['Dataset', 'Model'])['Best k']
actor_best_k = best_k_series.loc['Actor']

fig, ax = plt.subplots(figsize=(14, 8))

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
ax.legend(title="Model", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()


# **Interpretation:**
# 
# This plot provides a clear answer for the optimal number of roles for the Actor dataset.
# 
# For the top-performing models—`Feature-Based_Roles_Graphlets` and `GNN_Embedder_DGI_Graphlets`, the Silhouette Score peaks at **k=3** and then consistently declines. This signal indicates that forcing the data into more than three roles leads to less dense and less meaningful clusters.
# 
# While some of the other models show slightly different behavior, they all achieve significantly lower scores overall. The evidence overwhelmingly suggests that **3 is the optimal number of structural roles** for this network.

# ### 2.2. Visualizing the Discovered Roles with t-SNE
# t-SNE is a dimensionality reduction technique that allows us to visualize the high-dimensional node embeddings (or feature vectors) in 2D. In a good model, the nodes belonging to the same role should form distinct, well-separated visual clusters.
# 
# We will load and display the pre-generated t-SNE plots for the best $k$ value of each model.
# 

# In[5]:


import matplotlib.image as mpimg

fig, axes = plt.subplots(2, 3, figsize=(24, 16))
axes = axes.flatten()
fig.suptitle("Comparison of t-SNE Role Visualizations on the 'Actor' Dataset", fontsize=22)

for i, model_name in enumerate(MODELS):
    ax = axes[i]
    if model_name in actor_best_k:
        k = actor_best_k[model_name]
        image_path = RESULTS_DIR / f"Actor/{model_name}_k{k}_tsne.png"
        ax.set_title(f"{model_name}\n(Best k={k})", fontsize=16, pad=10)

        if image_path.exists():
            img = mpimg.imread(image_path)
            ax.imshow(img)
        else:
            ax.text(0.5, 0.5, 'Image not found', ha='center', va='center', fontsize=12)
    else:
        ax.set_visible(False)
    ax.axis('off')

for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()


# 
# **Interpretation:** 
# 
# These t-SNE plots provide a  visual confirmation of the quantitative results from the previous step. The goal is to see tight, well-separated clusters, which indicates a high-quality role assignment.
# 
# * **Excellent Separation (Graphlet-based models):** The two top-performing models, `Feature-Based_Roles_Graphlets` and `GNN_Embedder_DGI_Graphlets`, show remarkably clean visualizations. They both identify one massive, dense primary role (in blue) and a few very small, perfectly isolated satellite roles. This visual clarity directly corresponds to their near-perfect Silhouette scores and suggests they have uncovered a strong core-periphery structure in the network.
# a
# * **Moderate Separation (Other GNNs):** The `GNN_Embedder_GAE` and `GNN_Embedder_GAE_Graphlets` models lso manage to separate out the smaller roles, but their main cluster appears more diffuse and less tightly packed. The `GNN_Embedder_DGI` model without graphlets struggles more, with its roles appearing as less cohesive, string like structures.
# 
# * **Poor Separation (Baseline Model):** The `Feature-Based_Roles` model, which had the lowest scores, shows exactly what we would expect: heavily intermingled and poorly defined clusters. It is visually difficult to distinguish between the different roles, confirming that the standard features were insufficient for this task.

# ### 2.3. Interpreting Role Characteristics
# Moving beyond scores and visualizations to understand *what these roles represent*. We load the analysis files, which contain the average structural properties (degree, betweenness, etc.) for nodes within each role. By examining these properties, we can assign intuitive labels.

# In[6]:


def plot_role_profiles(df, dataset_name, model_name, k, ax):
    metrics_to_plot = ['avg_degree', 'avg_betweenness', 'avg_closeness', 'avg_eigenvector', 'avg_clustering_coeff']
    profiles = df[metrics_to_plot]

    min_vals = profiles.min()
    range_vals = profiles.max() - min_vals
    range_vals[range_vals == 0] = 1.0
    profiles_normalized = (profiles - min_vals) / range_vals
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
    ax.set_xticks(angles[:-1], labels, size=11)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8"], color="grey", size=9)

    for i, row in profiles_normalized.iterrows():
        values = row.values.flatten().tolist()
        values += values[:1]
        ax.plot(angles, values, color=colors[i], linewidth=2.5, linestyle='solid', label=f"Role {df.at[i, 'role_id']}")
        ax.fill(angles, values, color=colors[i], alpha=0.3)

    ax.legend(loc='upper right', bbox_to_anchor=(0.15, 0.15))

best_model_name = summary_df.loc[summary_df[summary_df['Dataset'] == 'Actor']['Silhouette Score'].idxmax()]['Model']
k = actor_best_k[best_model_name]
analysis_path = RESULTS_DIR / f"Actor/{best_model_name}_k{k}_role_analysis.csv"

fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
fig.suptitle(f'Structural Role Profiles on Actor Dataset\n(Best Model: {best_model_name})', fontsize=18)

if analysis_path.exists():
    role_df = pd.read_csv(analysis_path)
    print(f"\n--- Analysis for {best_model_name} (k={k}) ---")
    display(role_df)
    plot_role_profiles(role_df, "Actor", best_model_name, k, ax)
else:
    ax.set_visible(False)
    print(f"Analysis file not found for {best_model_name}")

plt.tight_layout(pad=3.0)
plt.show()


# 
# ### **Interpretation of Role Characteristics**
# 
# The radar chart give us a structural "fingerprint" for each of the three roles discovered by the `GNN_Embedder_DGI_Graphlets` model. By analyzing these fingerprints, we can assign them meaningful labels:
# 
# * **Role 0 (Dark Blue) - "Periphery":** This role contains the vast majority of nodes in the network (**7,592**). Its structural profile is the inverse of a hub: it has the lowest average centrality scores across the board but the highest average clustering coefficient. This describes typical members of local communities—they are part of tightly knit local structures but have little to no influence on the global network structure.
# 
# * **Role 1 (Pink) - "Connectors":** This is a role containing just **2 nodes**. These nodes have moderately high eigenvector and closeness centrality, suggesting they are well connected to other influential nodes. However, they are not primary hubs, as their degree and betweenness are lower than Role 2. They likely serve as important secondary connectors or links between specific clusters and the main network core.
# 
# * **Role 2 (Yellow) - "Global Hubs":** This is the most distinct and influential role. Despite comprising only **6 nodes**, it dominates every centrality metric: it has the highest average degree, betweenness, closeness, and eigenvector centrality. Its extremely low clustering coefficient signifies that these nodes connect to many other nodes that are not themselves connected. This is the classic signature of a **hub** or **broker** that bridges disparate parts of the network.

# ### 2.4. Analyzing Role Interactions with Adjacency Matrices
# 
# While role profiles tell us about node properties *within* a role, they don't describe how roles connect *to each other*. To analyze this, we compute a **role-to-role adjacency matrix**.
# 
# - **Raw Counts Matrix**: The entry `(i, j)` shows the absolute number of edges connecting nodes from role `i` to nodes in role `j`.
# - **Normalized Connectivity Matrix**: This matrix is row-normalized by the total degree sum of each role. The entry `(i, j)` represents the proportion of role `i`'s total connections (stubs) that go to role `j`. This reveals preferences:
#   - A high diagonal value (`~1.0`) indicates an insular role (a community).
#   - High off-diagonal values for a row `i` show it's a "bridge" or "connector" role.
# 
# 

# In[7]:


import matplotlib.pyplot as plt
import matplotlib.image as mpimg

try:
    best_model_name
except NameError:
    print("Warning: 'best_model_name' not defined. Calculating it now.")
    best_model_name = summary_df.loc[summary_df[summary_df['Dataset'] == 'Actor']['Silhouette Score'].idxmax()]['Model']

fig, ax = plt.subplots(1, 1, figsize=(10, 8))

model_to_plot = best_model_name
dataset_name = 'Actor'

if model_to_plot in actor_best_k:
    k = actor_best_k[model_to_plot]

    norm_path = RESULTS_DIR / f"{dataset_name}/{model_to_plot}_k{k}_role_adj_heatmap_normalized.png"

    ax.set_title(f"Normalized Role Connectivity for {model_to_plot} (k={k})", fontsize=16, pad=15)

    if norm_path.exists():
        img_norm = mpimg.imread(norm_path)
        ax.imshow(img_norm)
    else:
        ax.text(0.5, 0.5, f'Image not found at:\n{norm_path}', ha='center', va='center', wrap=True)

    ax.axis('off')
else:
    print(f"Could not find best k for '{model_to_plot}' on {dataset_name}")
    ax.axis('off')

plt.show()


# This heatmap of normalized role connectivity perfectly complements the previous analysis, moving from what the roles *are* to what they *do*. It reveals a clear **Core-Satellite structure** within the Actor network.
# 
# * **Role 0 acts as the central "Core":** This role is highly insular, with the heatmap showing that **99%** of connections originating from Role 0 nodes connect to other nodes *within* Role 0 (the bright cell at `[0, 0]`). This indicates that it represents a massive, densely interconnected component—likely the mainstream actors who frequently collaborate.
# 
# * **Roles 1 and 2 are distinct "Satellite" groups:** Their connectivity is not diverse; instead, it's highly specialized:
#     * **Role 1** directs **100%** of its connections exclusively to the "Core" (Role 0). It has zero connections to other nodes in Role 1 or to any nodes in the "Hub" role (Role 2).
#     * **Role 2** (the "Global Hubs") behaves almost identically, directing **99%** of its connections to the "Core" (Role 0).
# 
# This heatmap shows that our best-performing model did not find three separate communities. Instead, it identified a network architecture defined by function: a large, central core of actors, and two different types of specialized peripheral actors whose primary function is to connect to that core. 
# 

# ## 3. Bridging Structure and Semantics: Case Study on Cora
# 
# So far, our analysis has been purely structural. The `GNN_Embedder_DGI_Graphlets` model, which we identified as the top performer on real-world datasets, was trained using only the graph's structure (via graphlet features) and had no access to the actual content or subject area of the papers in the Cora dataset.
# 
# This allows us to ask a question: **Do the purely structural roles discovered by our best model correspond to distinct semantic fields?** In other words, do papers in "Neural Networks" have a different structural signature in the citation graph than papers on "Genetic Algorithms"?
# 
# Answering this will reveal the relationship between a paper's structural role (e.g., foundational paper, survey paper, niche paper) and its academic subject.
# 

# In[8]:


from torch_geometric.datasets import Planetoid
import torch
from pathlib import Path
from IPython.display import display
import pandas as pd
import matplotlib.pyplot as plt
import sys
import matplotlib.ticker as mticker

project_root = str(Path().resolve().parent)
if project_root not in sys.path:
    print(f"Adding project root to path: {project_root}")
    sys.path.append(project_root)

from role_discovery.models.GNNEmbedder import GNNEmbedder
from role_discovery.models.DGIEmbedder import DGIEmbedder
from role_discovery.models.FeatureBasedRoles import FeatureBasedRoles
from role_discovery.models.FeatureBasedRolesGraphlets import FeatureBasedRolesGraphlets
from role_discovery.models.GNNEmbedderGraphlets import GNNEmbedderGraphlets
from role_discovery.models.DGIEmbedderGraphlets import DGIEmbedderGraphlets
from role_discovery.utils.experiment_utils import clean_params

cora_subject_names = {
    0: 'Theory', 1: 'Reinforcement Learning', 2: 'Genetic Algorithms',
    3: 'Neural Networks', 4: 'Probabilistic Methods', 5: 'Case-Based',
    6: 'Rule-Learning'
}

dataset = Planetoid(root='/tmp/Cora', name='Cora')
data = dataset[0]

print("\nCell executed successfully: All modules imported.")


# In[9]:


dataset_name = "Cora"
cora_summary = summary_df[summary_df['Dataset'] == dataset_name]
best_model_name = cora_summary.loc[cora_summary['Silhouette Score'].idxmax()]['Model']
print(f"The best performing model for {dataset_name} is: '{best_model_name}'")

cora_best_k_series = best_k_series.loc['Cora']
best_k = cora_best_k_series[best_model_name]

model_class_map = {
    'GNN_Embedder_GAE': GNNEmbedder,
    'GNN_Embedder_DGI': DGIEmbedder,
    'Feature-Based_Roles': FeatureBasedRoles,
    'Feature-Based_Roles_Graphlets': FeatureBasedRolesGraphlets,
    'GNN_Embedder_GAE_Graphlets': GNNEmbedderGraphlets,
    'GNN_Embedder_DGI_Graphlets': DGIEmbedderGraphlets
}

ModelClass = model_class_map.get(best_model_name)
model = None

if 'GNN' in best_model_name:
    print("Best model is a GNN. Loading hyperparameters...")
    tuning_results_path = RESULTS_DIR / dataset_name / "hyperparameter_tuning_results.csv"
    tuning_df = pd.read_csv(tuning_results_path)
    model_tuning_results = tuning_df[tuning_df['model_name'] == best_model_name]

    if not model_tuning_results.empty:
        best_model_params_raw = model_tuning_results.iloc[0].to_dict()
        params = clean_params({k: v for k, v in best_model_params_raw.items() if k not in ['model_name', 'best_silhouette', 'emb_dim', 'in_channels']})
        print(f"Loading model with k={best_k} and params: {params}")

        init_params = {
            **params,
            "model_path": str(RESULTS_DIR / dataset_name / f"best_{best_model_name}_model.pt"),
            "force_retrain": False
        }
        if best_model_name == 'GNN_Embedder_DGI':
            init_params['in_channels'] = data.num_features

        model = ModelClass(**init_params)
    else:
        print(f"ERROR: GNN model '{best_model_name}' not found in tuning results file.")

else: 
    print(f"Best model is feature-based. Instantiating '{best_model_name}' directly.")
    model = ModelClass()


# In[10]:


if model:
    embeddings, role_labels = model.predict(data, k=best_k)
    print(f"\nSuccessfully assigned {data.num_nodes} nodes to {best_k} roles using {best_model_name}.")

    semantic_df = pd.DataFrame({
        'role_id': role_labels.numpy(),
        'subject_id': data.y.numpy()
    })
    semantic_df['subject_name'] = semantic_df['subject_id'].map(cora_subject_names)

    crosstab = pd.crosstab(semantic_df['role_id'], semantic_df['subject_name'])
    crosstab_norm = crosstab.div(crosstab.sum(axis=1), axis=0)

    print("\n--- Subject Distribution within Each Structural Role ---")
    display(crosstab_norm.style.format("{:.2%}"))


# Although the GNN model was only given the network structure, the roles it discovered correspond to distinct subject distributions.
# 
# * **Role 1:** This was a remarkable discovery, consisting **100%** of papers on "Genetic Algorithms." This indicates that the citation patterns within this field are so unique that they create a distinct structural footprint that the model could perfectly isolate.
# 
# * **Role 2:** This role captured a "meta-role" of closely related AI topics, primarily grouping "Neural Networks" (38%), "Reinforcement Learning" (31%), and other related papers. The model correctly identified that these fields share a common structure in the citation network.
# 
# * **Role 0:** This larger, more diverse role represents the interdisciplinary core of the network, containing a mix of all subjects.

# In[11]:


if model:
    fig, ax = plt.subplots(figsize=(14, 7))
    crosstab_norm.plot(
        kind='bar',
        stacked=True,
        ax=ax,
        cmap='rocket',
        width=0.8
    )

    ax.set_title(f'Semantic Analysis: Paper Subject Distribution per Structural Role on Cora\n(Model: {best_model_name})', fontsize=18, pad=15)
    ax.set_xlabel('Discovered Structural Role ID', fontsize=14)
    ax.set_ylabel('Proportion of Papers', fontsize=14)
    ax.tick_params(axis='x', rotation=0)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0)) 
    plt.legend(title='Paper Subject', bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.show()

