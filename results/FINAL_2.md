# FINAL 2 (seed_sweep_2026-10-03_11-50-07)

1 seed(s), 6 training combination(s). Each model set is tested on the dataset left out of training. Accuracies are the average over the ensembles of a seed unless stated otherwise.

## Summary over seeds

| Training | Test | Seeds | Mean | Std | Min | Max |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: |
| ALL_UM | AGFD | 1 | 87.93% | – | 87.93% | 87.93% |
| ALL_UA | MCC5 | 1 | 70.00% | – | 70.00% | 70.00% |
| ALL_MA | UNSW | 1 | 95.83% | – | 95.83% | 95.83% |
| ALL_U | AGFD | 1 | 86.44% | – | 86.44% | 86.44% |
| ALL_M | AGFD | 1 | 86.96% | – | 86.96% | 86.96% |
| ALL_A | UNSW | 1 | 90.83% | – | 90.83% | 90.83% |

## Mean accuracy per seed

| Seed | ALL_UM → AGFD | ALL_UA → MCC5 | ALL_MA → UNSW | ALL_U → AGFD | ALL_U → MCC5 | ALL_M → AGFD | ALL_M → UNSW | ALL_A → UNSW | ALL_A → MCC5 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1249 | 87.93% | 70.00% | 95.83% | 86.44% | 67.78 | 86.96% | 74.2 | 90.83% | 45.6 |

## Individual ensembles

### ALL_UM → AGFD

| Seed | Weights | Ensemble | Accuracy |
| :--- | :--- | ---: | ---: |
| 1249 | `UM_1` | 1 | 87.41% |
| 1249 | `UM_1` | 2 | 87.04% |
| 1249 | `UM_1` | 3 | 87.78% |
| 1249 | `UM_1` | 4 | 88.89% |
| 1249 | `UM_1` | 5 | 88.52% |

### ALL_UA → MCC5

| Seed | Weights | Ensemble | Accuracy |
| :--- | :--- | ---: | ---: |
| 1249 | `UA_1` | 1 | 75.00% |
| 1249 | `UA_1` | 2 | 69.44% |
| 1249 | `UA_1` | 3 | 75.00% |
| 1249 | `UA_1` | 4 | 66.67% |
| 1249 | `UA_1` | 5 | 63.89% |

### ALL_MA → UNSW

| Seed | Weights | Ensemble | Accuracy |
| :--- | :--- | ---: | ---: |
| 1249 | `MA_1` | 1 | 95.83% |
| 1249 | `MA_1` | 2 | 95.83% |
| 1249 | `MA_1` | 3 | 95.83% |
| 1249 | `MA_1` | 4 | 95.83% |
| 1249 | `MA_1` | 5 | 95.83% |

### ALL_U → AGFD

| Seed | Weights | Ensemble | Accuracy |
| :--- | :--- | ---: | ---: |
| 1249 | `U_1` | 1 | 84.81% |
| 1249 | `U_1` | 2 | 86.30% |
| 1249 | `U_1` | 3 | 90.74% |
| 1249 | `U_1` | 4 | 82.59% |
| 1249 | `U_1` | 5 | 87.78% |

### ALL_M → AGFD

| Seed | Weights | Ensemble | Accuracy |
| :--- | :--- | ---: | ---: |
| 1249 | `M_1` | 1 | 85.56% |
| 1249 | `M_1` | 2 | 86.30% |
| 1249 | `M_1` | 3 | 87.78% |
| 1249 | `M_1` | 4 | 87.41% |
| 1249 | `M_1` | 5 | 87.78% |

### ALL_A → UNSW

| Seed | Weights | Ensemble | Accuracy |
| :--- | :--- | ---: | ---: |
| 1249 | `A_1` | 1 | 93.75% |
| 1249 | `A_1` | 2 | 89.58% |
| 1249 | `A_1` | 3 | 91.67% |
| 1249 | `A_1` | 4 | 85.42% |
| 1249 | `A_1` | 5 | 93.75% |

