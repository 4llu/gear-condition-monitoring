# results_2026-09-30_10-42-08

3 seed(s), 3 training combination(s). Each model set is tested on the dataset left out of training. Accuracies are the average over the ensembles of a seed unless stated otherwise.

## Summary over seeds

| Training | Test | Seeds | Mean | Std | Min | Max |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: |
| ALL_UM | AGFD | 3 | 87.90% | 0.99% | 86.91% | 88.89% |
| ALL_UA | MCC5 | 3 | 71.60% | 3.51% | 67.59% | 74.07% |
| ALL_MA | UNSW | 3 | 90.74% | 2.23% | 88.19% | 92.36% |

## Mean accuracy per seed

| Seed | ALL_UM → AGFD | ALL_UA → MCC5 | ALL_MA → UNSW |
| :--- | ---: | ---: | ---: |
| 600 | 86.91% | 74.07% | 88.19% |
| 700 | 88.89% | 67.59% | 92.36% |
| 800 | 87.90% | 73.15% | 91.67% |

## Individual ensembles

### ALL_UM → AGFD

| Seed | Weights | Ensemble | Accuracy |
| :--- | :--- | ---: | ---: |
| 600 | `UM` | 1 | 88.52% |
| 600 | `UM` | 2 | 85.19% |
| 600 | `UM` | 3 | 87.04% |
| 700 | `UM_1` | 1 | 88.89% |
| 700 | `UM_1` | 2 | 87.41% |
| 700 | `UM_1` | 3 | 90.37% |
| 800 | `UM_2` | 1 | 88.89% |
| 800 | `UM_2` | 2 | 87.78% |
| 800 | `UM_2` | 3 | 87.04% |

### ALL_UA → MCC5

| Seed | Weights | Ensemble | Accuracy |
| :--- | :--- | ---: | ---: |
| 600 | `UA` | 1 | 72.22% |
| 600 | `UA` | 2 | 75.00% |
| 600 | `UA` | 3 | 75.00% |
| 700 | `UA_1` | 1 | 63.89% |
| 700 | `UA_1` | 2 | 63.89% |
| 700 | `UA_1` | 3 | 75.00% |
| 800 | `UA_2` | 1 | 72.22% |
| 800 | `UA_2` | 2 | 72.22% |
| 800 | `UA_2` | 3 | 75.00% |

### ALL_MA → UNSW

| Seed | Weights | Ensemble | Accuracy |
| :--- | :--- | ---: | ---: |
| 600 | `MA` | 1 | 91.67% |
| 600 | `MA` | 2 | 81.25% |
| 600 | `MA` | 3 | 91.67% |
| 700 | `MA_1` | 1 | 87.50% |
| 700 | `MA_1` | 2 | 95.83% |
| 700 | `MA_1` | 3 | 93.75% |
| 800 | `MA_2` | 1 | 91.67% |
| 800 | `MA_2` | 2 | 95.83% |
| 800 | `MA_2` | 3 | 87.50% |

