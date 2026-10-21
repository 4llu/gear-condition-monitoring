# FINAL 1 (seed_sweep_2026-10-02_16-30-24)

1 seed(s), 6 training combination(s). Each model set is tested on the dataset left out of training. Accuracies are the average over the ensembles of a seed unless stated otherwise.

## Summary over seeds

| Training | Test | Seeds | Mean | Std | Min | Max |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: |
| ALL_UM | AGFD | 1 | 89.85% | – | 89.85% | 89.85% |
| ALL_UA | MCC5 | 1 | 67.78% | – | 67.78% | 67.78% |
| ALL_MA | UNSW | 1 | 95.83% | – | 95.83% | 95.83% |
| ALL_U | AGFD | 1 | 86.67% | – | 86.67% | 86.67% |
| ALL_M | AGFD | 1 | 87.56% | – | 87.56% | 87.56% |
| ALL_A | UNSW | 1 | 89.17% | – | 89.17% | 89.17% |

## Mean accuracy per seed

| Seed | ALL_UM → AGFD | ALL_UA → MCC5 | ALL_MA → UNSW | ALL_U → AGFD | ALL_U → MCC5 | ALL_M → AGFD | ALL_M → UNSW | ALL_A → UNSW | ALL_A → MCC5 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1234 | 89.85% | 67.78% | 95.83% | 86.67% | 65.0 | 87.56% | 80.8 | 89.17% | 45.6 |

## Individual ensembles

### ALL_UM → AGFD

| Seed | Weights | Ensemble | Accuracy |
| :--- | :--- | ---: | ---: |
| 1234 | `UM` | 1 | 91.85% |
| 1234 | `UM` | 2 | 90.37% |
| 1234 | `UM` | 3 | 88.89% |
| 1234 | `UM` | 4 | 90.37% |
| 1234 | `UM` | 5 | 87.78% |

### ALL_UA → MCC5

| Seed | Weights | Ensemble | Accuracy |
| :--- | :--- | ---: | ---: |
| 1234 | `UA` | 1 | 75.00% |
| 1234 | `UA` | 2 | 63.89% |
| 1234 | `UA` | 3 | 58.33% |
| 1234 | `UA` | 4 | 69.44% |
| 1234 | `UA` | 5 | 72.22% |

### ALL_MA → UNSW

| Seed | Weights | Ensemble | Accuracy |
| :--- | :--- | ---: | ---: |
| 1234 | `MA` | 1 | 95.83% |
| 1234 | `MA` | 2 | 95.83% |
| 1234 | `MA` | 3 | 95.83% |
| 1234 | `MA` | 4 | 95.83% |
| 1234 | `MA` | 5 | 95.83% |

### ALL_U → AGFD

| Seed | Weights | Ensemble | Accuracy |
| :--- | :--- | ---: | ---: |
| 1234 | `U` | 1 | 88.52% |
| 1234 | `U` | 2 | 86.30% |
| 1234 | `U` | 3 | 85.19% |
| 1234 | `U` | 4 | 84.81% |
| 1234 | `U` | 5 | 88.52% |

### ALL_M → AGFD

| Seed | Weights | Ensemble | Accuracy |
| :--- | :--- | ---: | ---: |
| 1234 | `M` | 1 | 88.89% |
| 1234 | `M` | 2 | 86.30% |
| 1234 | `M` | 3 | 85.93% |
| 1234 | `M` | 4 | 89.63% |
| 1234 | `M` | 5 | 87.04% |

### ALL_A → UNSW

| Seed | Weights | Ensemble | Accuracy |
| :--- | :--- | ---: | ---: |
| 1234 | `A` | 1 | 89.58% |
| 1234 | `A` | 2 | 89.58% |
| 1234 | `A` | 3 | 89.58% |
| 1234 | `A` | 4 | 89.58% |
| 1234 | `A` | 5 | 87.50% |

