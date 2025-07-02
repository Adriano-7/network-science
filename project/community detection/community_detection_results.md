# Summary of Community Detection Results

| Dataset     | Algorithm         |   Num_Communities |   Modularity |   Avg_Conductance |    NMI |    ARI |   Fowlkes_Mallows |
|:------------|:------------------|------------------:|-------------:|------------------:|-------:|-------:|------------------:|
| KarateClub  | Louvain           |                 4 |       0.4198 |            0.4417 | 0.815  | 0.7665 |            0.8309 |
| KarateClub  | Label Propagation |                 3 |       0.3251 |            0.4315 | 0.551  | 0.5115 |            0.6859 |
| KarateClub  | Girvan-Newman     |                 5 |       0.4013 |            0.5761 | 0.8301 | 0.8077 |            0.8609 |
| KarateClub  | GNN (GCN)         |                 4 |       0.3476 |            0.4869 | 0.6361 | 0.4745 |            0.6282 |
| KarateClub  | GNN (GraphSage)   |                 4 |      -0.1294 |            0.906  | 0.2884 | 0.1192 |            0.3798 |
| Cora_Subset | Louvain           |                 9 |       0.6978 |            0.3107 | 0.3097 | 0.1307 |            0.2781 |
| Cora_Subset | Label Propagation |                29 |       0.5906 |            0.5245 | 0.4037 | 0.0778 |            0.1974 |
| Cora_Subset | Girvan-Newman     |                11 |       0.6925 |            0.3318 | 0.3142 | 0.1228 |            0.2673 |
| Cora_Subset | GNN (GCN)         |                 6 |       0.3311 |            0.5429 | 0.5937 | 0.5862 |            0.6865 |
| Cora_Subset | GNN (GraphSage)   |                 6 |       0.2147 |            0.6341 | 0.6332 | 0.6095 |            0.711  |
| Cora        | Louvain           |               105 |       0.8159 |            0.0571 | 0.4542 | 0.2317 |            0.3527 |
| Cora        | Label Propagation |               454 |       0.6722 |            0.46   | 0.4143 | 0.0761 |            0.2014 |
| Cora        | GNN (GCN)         |                 7 |       0.7002 |            0.257  | 0.5989 | 0.6091 |            0.6766 |
| Cora        | GNN (GraphSage)   |                 7 |       0.6687 |            0.3036 | 0.5895 | 0.5935 |            0.6635 |
| PubMed      | Louvain           |                40 |       0.7679 |            0.2403 | 0.2074 | 0.1146 |            0.2937 |
| PubMed      | Label Propagation |              1740 |       0.6185 |            0.5768 | 0.1805 | 0.0389 |            0.1683 |
| PubMed      | GNN (GCN)         |                 3 |       0.4986 |            0.2682 | 0.3634 | 0.4103 |            0.6179 |
| PubMed      | GNN (GraphSage)   |                 3 |       0.5096 |            0.2382 | 0.3603 | 0.4121 |            0.6196 |
