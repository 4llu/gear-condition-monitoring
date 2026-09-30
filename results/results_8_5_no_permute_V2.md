# seed_sweep_2026-10-01_01-36-01

3 seed(s), 3 training combination(s). Each model set is tested on the dataset left out of training. Accuracies are the average over the ensembles of a seed unless stated otherwise.

## Summary over seeds

| Training | Test | Seeds | Mean | Std | Min | Max |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: |
| ALL_UM | AGFD | 3 | 89.05% | 0.62% | 88.40% | 89.63% |
| ALL_UA | MCC5 | 3 | 73.15% | 1.85% | 71.30% | 75.00% |
| ALL_MA | UNSW | 3 | 92.82% | 2.23% | 90.28% | 94.44% |

## Mean accuracy per seed

| Seed | ALL_UM → AGFD | ALL_UA → MCC5 | ALL_MA → UNSW |
| :--- | ---: | ---: | ---: |
| 100 | 89.14% | 75.00% | 90.28% |
| 200 | 88.40% | 73.15% | 94.44% |
| 500 | 89.63% | 71.30% | 93.75% |

## Individual ensembles

### ALL_UM → AGFD

| Seed | Weights | Ensemble | Accuracy |
| :--- | :--- | ---: | ---: |
| 100 | `UM` | 1 | 86.30% |
| 100 | `UM` | 2 | 92.22% |
| 100 | `UM` | 3 | 88.89% |
| 200 | `UM_1` | 1 | 88.15% |
| 200 | `UM_1` | 2 | 89.63% |
| 200 | `UM_1` | 3 | 87.41% |
| 500 | `UM_2` | 1 | 89.26% |
| 500 | `UM_2` | 2 | 88.52% |
| 500 | `UM_2` | 3 | 91.11% |

### ALL_UA → MCC5

| Seed | Weights | Ensemble | Accuracy |
| :--- | :--- | ---: | ---: |
| 100 | `UA` | 1 | 72.22% |
| 100 | `UA` | 2 | 75.00% |
| 100 | `UA` | 3 | 77.78% |
| 200 | `UA_1` | 1 | 72.22% |
| 200 | `UA_1` | 2 | 75.00% |
| 200 | `UA_1` | 3 | 72.22% |
| 500 | `UA_2` | 1 | 69.44% |
| 500 | `UA_2` | 2 | 69.44% |
| 500 | `UA_2` | 3 | 75.00% |

### ALL_MA → UNSW

| Seed | Weights | Ensemble | Accuracy |
| :--- | :--- | ---: | ---: |
| 100 | `MA` | 1 | 87.50% |
| 100 | `MA` | 2 | 91.67% |
| 100 | `MA` | 3 | 91.67% |
| 200 | `MA_1` | 1 | 95.83% |
| 200 | `MA_1` | 2 | 95.83% |
| 200 | `MA_1` | 3 | 91.67% |
| 500 | `MA_2` | 1 | 91.67% |
| 500 | `MA_2` | 2 | 93.75% |
| 500 | `MA_2` | 3 | 95.83% |

