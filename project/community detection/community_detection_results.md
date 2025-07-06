# Summary of Community Detection Results

| Dataset     | Algorithm                   |   Num_Communities |   Modularity |   Avg_Conductance |    NMI |     ARI |   Fowlkes_Mallows |
|:------------|:----------------------------|------------------:|-------------:|------------------:|-------:|--------:|------------------:|
| KarateClub  | Louvain                     |                 4 |       0.4188 |            0.4421 | 0.8932 |  0.8653 |            0.9029 |
| KarateClub  | Label Propagation           |                 3 |       0.3251 |            0.4315 | 0.551  |  0.5115 |            0.6859 |
| KarateClub  | Girvan-Newman               |                 5 |       0.4013 |            0.5761 | 0.8301 |  0.8077 |            0.8609 |
| KarateClub  | GNN (GCN)                   |                 4 |       0.3476 |            0.4869 | 0.6361 |  0.4745 |            0.6282 |
| KarateClub  | GNN (GraphSage)             |                 4 |       0.2772 |            0.6514 | 0.5058 |  0.4474 |            0.6219 |
| KarateClub  | GNN (GCN) + Graphlets       |                 4 |      -0.0639 |            0.8139 | 0.2762 |  0.1135 |            0.3648 |
| KarateClub  | GNN (GraphSage) + Graphlets |                 4 |      -0.0865 |            0.8526 | 0.3879 |  0.205  |            0.4444 |
| Cora_Subset | Louvain                     |                 9 |       0.6566 |            0.3579 | 0.1211 |  0.0064 |            0.3032 |
| Cora_Subset | Label Propagation           |                17 |       0.5821 |            0.4919 | 0.1476 | -0.0039 |            0.3013 |
| Cora_Subset | Girvan-Newman               |                11 |       0.6391 |            0.4008 | 0.1358 |  0.0038 |            0.2754 |
| Cora_Subset | GNN (GCN)                   |                 6 |       0.0285 |            0.7376 | 0.579  |  0.6728 |            0.9242 |
| Cora_Subset | GNN (GraphSage)             |                 6 |       0.0052 |            0.8024 | 0.6092 |  0.6606 |            0.9256 |
| Cora_Subset | GNN (GCN) + Graphlets       |                 5 |       0.0103 |            0.7349 | 0.4627 |  0.5543 |            0.9044 |
| Cora_Subset | GNN (GraphSage) + Graphlets |                 6 |       0.0053 |            0.7879 | 0.5673 |  0.6201 |            0.9189 |
| Cora        | Louvain                     |               101 |       0.8137 |            0.047  | 0.4498 |  0.226  |            0.3447 |
| Cora        | Label Propagation           |               454 |       0.6722 |            0.46   | 0.4143 |  0.0761 |            0.2014 |
| Cora        | GNN (GCN)                   |                 7 |       0.6967 |            0.2619 | 0.6036 |  0.612  |            0.6791 |
| Cora        | GNN (GraphSage)             |                 7 |       0.6861 |            0.2821 | 0.5889 |  0.5892 |            0.6596 |
| Cora        | GNN (GCN) + Graphlets       |                 7 |       0.6998 |            0.2593 | 0.6052 |  0.6044 |            0.6726 |
| Cora        | GNN (GraphSage) + Graphlets |                 7 |       0.6634 |            0.3204 | 0.5711 |  0.5607 |            0.6357 |
| PubMed      | Louvain                     |                43 |       0.7675 |            0.2304 | 0.1982 |  0.1094 |            0.2869 |
| PubMed      | Label Propagation           |              1740 |       0.6185 |            0.5768 | 0.1805 |  0.0389 |            0.1683 |
| PubMed      | GNN (GCN)                   |                 3 |       0.5001 |            0.2636 | 0.3799 |  0.4301 |            0.6322 |
| PubMed      | GNN (GraphSage)             |                 3 |       0.4938 |            0.2575 | 0.3493 |  0.3966 |            0.6093 |
| PubMed      | GNN (GCN) + Graphlets       |                 3 |       0.5003 |            0.2753 | 0.372  |  0.4244 |            0.6294 |
| PubMed      | GNN (GraphSage) + Graphlets |                 3 |       0.4563 |            0.2737 | 0.3335 |  0.3742 |            0.5978 |
