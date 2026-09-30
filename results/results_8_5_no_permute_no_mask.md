# results_2026-09-30_01-22-26

3 seed(s), 3 training combination(s). Each model set is tested on the dataset left out of training. Accuracies are the average over the ensembles of a seed unless stated otherwise.

## Summary over seeds

| Training | Test | Seeds | Mean | Std | Min | Max |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: |
| ALL_UM | AGFD | 3 | 88.27% | 0.43% | 88.02% | 88.77% |
| ALL_UA | MCC5 | 3 | 70.37% | 0.93% | 69.44% | 71.30% |
| ALL_MA | UNSW | 3 | 92.13% | 4.01% | 87.50% | 94.44% |

## Mean accuracy per seed

| Seed | ALL_UM → AGFD | ALL_UA → MCC5 | ALL_MA → UNSW |
| :--- | ---: | ---: | ---: |
| 600 | 88.02% | 71.30% | 87.50% |
| 700 | 88.77% | 69.44% | 94.44% |
| 800 | 88.02% | 70.37% | 94.44% |

## Individual ensembles

### ALL_UM → AGFD

| Seed | Weights | Ensemble | Accuracy |
| :--- | :--- | ---: | ---: |
| 600 | `UM` | 1 | 90.37% |
| 600 | `UM` | 2 | 86.30% |
| 600 | `UM` | 3 | 87.41% |
| 700 | `UM_1` | 1 | 89.63% |
| 700 | `UM_1` | 2 | 87.78% |
| 700 | `UM_1` | 3 | 88.89% |
| 800 | `UM_2` | 1 | 87.41% |
| 800 | `UM_2` | 2 | 88.89% |
| 800 | `UM_2` | 3 | 87.78% |

### ALL_UA → MCC5

| Seed | Weights | Ensemble | Accuracy |
| :--- | :--- | ---: | ---: |
| 600 | `UA` | 1 | 63.89% |
| 600 | `UA` | 2 | 72.22% |
| 600 | `UA` | 3 | 77.78% |
| 700 | `UA_1` | 1 | 61.11% |
| 700 | `UA_1` | 2 | 75.00% |
| 700 | `UA_1` | 3 | 72.22% |
| 800 | `UA_2` | 1 | 75.00% |
| 800 | `UA_2` | 2 | 69.44% |
| 800 | `UA_2` | 3 | 66.67% |

### ALL_MA → UNSW

| Seed | Weights | Ensemble | Accuracy |
| :--- | :--- | ---: | ---: |
| 600 | `MA` | 1 | 91.67% |
| 600 | `MA` | 2 | 79.17% |
| 600 | `MA` | 3 | 91.67% |
| 700 | `MA_1` | 1 | 91.67% |
| 700 | `MA_1` | 2 | 95.83% |
| 700 | `MA_1` | 3 | 95.83% |
| 800 | `MA_2` | 1 | 95.83% |
| 800 | `MA_2` | 2 | 95.83% |
| 800 | `MA_2` | 3 | 91.67% |

