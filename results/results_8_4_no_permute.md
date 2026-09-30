# results_2026-09-29_23-23-26

3 seed(s), 3 training combination(s). Each model set is tested on the dataset left out of training. Accuracies are the average over the ensembles of a seed unless stated otherwise.

## Summary over seeds

| Training | Test | Seeds | Mean | Std | Min | Max |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: |
| ALL_UM | AGFD | 3 | 87.49% | 0.88% | 86.54% | 88.27% |
| ALL_UA | MCC5 | 3 | 47.53% | 1.41% | 46.30% | 49.07% |
| ALL_MA | UNSW | 3 | 75.46% | 2.12% | 73.61% | 77.78% |

## Mean accuracy per seed

| Seed | ALL_UM → AGFD | ALL_UA → MCC5 | ALL_MA → UNSW |
| :--- | ---: | ---: | ---: |
| 300 | 87.65% | 49.07% | 75.00% |
| 400 | 86.54% | 46.30% | 73.61% |
| 500 | 88.27% | 47.22% | 77.78% |

## Individual ensembles

### ALL_UM → AGFD

| Seed | Weights | Ensemble | Accuracy |
| :--- | :--- | ---: | ---: |
| 300 | `UM_3` | 1 | 85.93% |
| 300 | `UM_3` | 2 | 88.89% |
| 300 | `UM_3` | 3 | 88.15% |
| 400 | `UM_4` | 1 | 85.19% |
| 400 | `UM_4` | 2 | 87.41% |
| 400 | `UM_4` | 3 | 87.04% |
| 500 | `UM_5` | 1 | 88.89% |
| 500 | `UM_5` | 2 | 88.15% |
| 500 | `UM_5` | 3 | 87.78% |

### ALL_UA → MCC5

| Seed | Weights | Ensemble | Accuracy |
| :--- | :--- | ---: | ---: |
| 300 | `UA_3` | 1 | 47.22% |
| 300 | `UA_3` | 2 | 52.78% |
| 300 | `UA_3` | 3 | 47.22% |
| 400 | `UA_4` | 1 | 47.22% |
| 400 | `UA_4` | 2 | 41.67% |
| 400 | `UA_4` | 3 | 50.00% |
| 500 | `UA_5` | 1 | 50.00% |
| 500 | `UA_5` | 2 | 47.22% |
| 500 | `UA_5` | 3 | 44.44% |

### ALL_MA → UNSW

| Seed | Weights | Ensemble | Accuracy |
| :--- | :--- | ---: | ---: |
| 300 | `MA_3` | 1 | 70.83% |
| 300 | `MA_3` | 2 | 72.92% |
| 300 | `MA_3` | 3 | 81.25% |
| 400 | `MA_4` | 1 | 75.00% |
| 400 | `MA_4` | 2 | 70.83% |
| 400 | `MA_4` | 3 | 75.00% |
| 500 | `MA_5` | 1 | 70.83% |
| 500 | `MA_5` | 2 | 89.58% |
| 500 | `MA_5` | 3 | 72.92% |

