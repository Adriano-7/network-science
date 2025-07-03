# Summary of Community Detection Results

| Dataset     | Algorithm         |   Num_Communities |   Modularity |   Avg_Conductance |    NMI |    ARI |   Fowlkes_Mallows |
|:------------|:------------------|------------------:|-------------:|------------------:|-------:|-------:|------------------:|
| KarateClub  | Louvain           |                 4 |       0.4188 |            0.4421 | 0.8932 | 0.8653 |            0.9029 |
| KarateClub  | Label Propagation |                 3 |       0.3251 |            0.4315 | 0.551  | 0.5115 |            0.6859 |
| KarateClub  | Girvan-Newman     |                 5 |       0.4013 |            0.5761 | 0.8301 | 0.8077 |            0.8609 |
| KarateClub  | GNN (GCN)         |                 4 |       0.3476 |            0.4869 | 0.6361 | 0.4745 |            0.6282 |
| KarateClub  | GNN (GraphSage)   |                 4 |      -0.152  |            0.8671 | 0.355  | 0.1736 |            0.4417 |
| Cora_Subset | Louvain           |                 8 |       0.6785 |            0.3135 | 0.2195 | 0.0887 |            0.3388 |
| Cora_Subset | Label Propagation |                16 |       0.6313 |            0.4767 | 0.264  | 0.0764 |            0.306  |
| Cora_Subset | Girvan-Newman     |                 8 |       0.679  |            0.2801 | 0.2693 | 0.1107 |            0.3667 |
| Cora_Subset | GNN (GCN)         |                 4 |       0.287  |            0.619  | 0.6249 | 0.7319 |            0.8575 |
| Cora_Subset | GNN (GraphSage)   |                 4 |       0.1231 |            0.7138 | 0.643  | 0.7565 |            0.8742 |
| Cora        | Louvain           |               102 |       0.8156 |            0.0526 | 0.4508 | 0.2191 |            0.3402 |
| Cora        | Label Propagation |               454 |       0.6722 |            0.46   | 0.4143 | 0.0761 |            0.2014 |
| Cora        | GNN (GCN)         |                 7 |       0.7033 |            0.2519 | 0.6043 | 0.6109 |            0.6781 |
| Cora        | GNN (GraphSage)   |                 7 |       0.6812 |            0.2834 | 0.583  | 0.5863 |            0.6576 |
| PubMed      | Louvain           |                44 |       0.765  |            0.2494 | 0.2003 | 0.1077 |            0.285  |
| PubMed      | Label Propagation |              1740 |       0.6185 |            0.5768 | 0.1805 | 0.0389 |            0.1683 |
| PubMed      | GNN (GCN)         |                 3 |       0.4703 |            0.287  | 0.3749 | 0.4265 |            0.6318 |
| PubMed      | GNN (GraphSage)   |                 3 |       0.4839 |            0.2523 | 0.3483 | 0.3978 |            0.6122 |
