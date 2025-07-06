# Network Science Tasks: Classical vs Modern Approaches

> **Project for Network Science 2024/2025**
> <br />
> Master's in Artificial Intelligence (MIA)
> <br />
> Faculty of Engineering, University of Porto
> <br />
> [Course Link](https://www.dcc.fc.up.pt/~pribeiro/aulas/ns2425/)
> <br />
---

## Project Goals

We compared classical network algorithms against modern Graph Neural Networks across three fundamental network analysis tasks. Using seven diverse datasets including citation networks, social networks, and synthetic benchmarks, we evaluated the performance trade-offs between topology-based and feature-based approaches.

* **Community Detection** - Finding groups of nodes that hang out together
* **Link Prediction** - Guessing which connections are missing or might form next  
* **Role Discovery** - Figuring out which nodes do similar jobs in the network

### Community Detection

We compared traditional topology-based algorithms with feature-aware GNNs for node partitioning.

**Methods tested:**
- **Traditional**: Louvain, Girvan-Newman, Label Propagation
- **GNNs**: Graph Convolutional Network (GCN), GraphSAGE


<p align="center">
  <img src="project/docs/Ground_Truth_embeddings.png" width="45%" alt="t-SNE colored by ground-truth communities">
  <img src="project/docs/Predicted_Communities_embeddings.png" width="45%" alt="t-SNE colored by GCN-predicted communities">
</p>

**Figure 1.** t‑SNE visualization of GCN embeddings on Cora. Left shows ground‑truth communities, right shows what the GCN predicted. Pretty close match!

### Link Prediction

Here we looked at two things: how well models can tell if a link exists (AUC) and how good they are at ranking potential links (MRR, Hits@K). 

**What we tested:**
- **Simple heuristics**: Adamic-Adar, Common Neighbors, Jaccard Index, Preferential Attachment
- **Traditional ML**: Random Forest, Logistic Regression, Decision Tree, KNN (all trained on features from the heuristics)
- **GNNs**: GCN and GraphSAGE with different decoders

**What we found:**
- GNNs got the best classification scores, GCN with a Dot Product decoder hit 0.900 AUC on average.
- But for ranking tasks, **Random Forest** was actually the most practical choice. Better ranking performance than GNNs and way faster to train.
- Simple Dot Product decoders consistently beat fancy MLP decoders for GNNs. Turns out good embeddings matter more than complex decoding.

<p align="center">
  <img src="project/docs/link_pred_best_models.png" width="80%" alt="Comparison of the best models from each category for link prediction.">
</p>

**Figure 2.** Best model from each category. GNNs win at classification, Random Forest dominates ranking.

### Role Discovery  

This task focuses on identifying structurally equivalent nodes regardless of community membership.

**What we tested:**
- **Feature-based clustering**: K-Means on centrality features or graphlet degree vectors
- **GNN-based clustering**: K-Means on embeddings from Graph Auto-Encoder (GAE) and Deep Graph Infomax (DGI)

**What we found:**
- Engineered graphlet features consistently produced the highest quality roles across all datasets. The traditional Feature-Based_Roles_Graphlets approach often matched or exceeded GNN performance.

<p align="center">
  <img src="project/docs/silhouette_score.png" width="80%" alt="Bar chart comparing the Silhouette Score for all role discovery models across three datasets.">
</p>

**Figure 3.** Silhouette scores across Actor, CLUSTER, and Cora datasets. Graphlet features consistently produce the most cohesive roles.

<p align="center">
  <img src="project/docs/t-sne_feature_based_graphlet.png" width="60%" alt="t-SNE visualization of the three roles discovered by the Feature-Based Graphlets model on Cora.">
</p>

**Figure 4.** t‑SNE viz of the three roles found by graphlet features on Cora. Clean separation = high Silhouette Score (0.9372).

<p align="center">
  <img src="project/docs/structural_role_analysis.png" width="90%" alt="Structural 'fingerprints' and property distributions of the three roles discovered on Cora.">
</p>

**Figure 5.** Structural "fingerprints" of the three roles on Cora. The radar plot shows each role's centrality signature, violin plots show the distributions.

## How to run it

Each task has its own folder. From the project root:

### Community Detection

```bash
# Run experiments
python project/community_detection/main_experiments.py

# Check results
jupyter notebook project/community_detection/analysis.ipynb
```

### Link Prediction

```bash
# Optional: tune hyperparameters
python project/link_prediction/TuneTraditionalML.py
python project/link_prediction/TuneGNNs.py

# Run main experiments
python project/link_prediction/LinkPredExperiment.py

# Analyze results
jupyter notebook project/link_prediction/ResultAnalysis.ipynb
```

### Role Discovery

```bash
# Optional: tune GNN hyperparameters
python -m project.role_discovery.tune_hyperparams --dataset Cora Actor CLUSTER

# Run evaluation (add --use_tuned for best hyperparameters)
python -m project.role_discovery.run_evaluation --dataset Cora Actor CLUSTER --use_tuned

# Generate reports
python -m project.role_discovery.generate_report --dataset Cora Actor CLUSTER
```

## Dependencies

- Python
- PyTorch
- PyTorch Geometric  
- NetworkX
- Scikit-learn
- Pandas
- Matplotlib
- Seaborn

## Team

- **Adriano Machado** (up202105352)
- **Francisco da Ana** (up202108762)  
- **João Lima** (up202108891)