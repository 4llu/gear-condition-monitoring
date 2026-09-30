# results_2026-09-30_00-12-40

3 seed(s), 3 training combination(s). Each model set is tested on the dataset left out of training. Accuracies are the average over the ensembles of a seed unless stated otherwise.

## Summary over seeds

| Training | Test | Seeds | Mean | Std | Min | Max |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: |
| ALL_UM | AGFD | 3 | 85.84% | 0.56% | 85.31% | 86.42% |
| ALL_UA | MCC5 | 3 | 68.83% | 1.07% | 67.59% | 69.44% |
| ALL_MA | UNSW | 3 | 90.28% | 1.39% | 88.89% | 91.67% |

## Mean accuracy per seed

| Seed | ALL_UM → AGFD | ALL_UA → MCC5 | ALL_MA → UNSW |
| :--- | ---: | ---: | ---: |
| 600 | 85.80% | 69.44% | 88.89% |
| 700 | 85.31% | 69.44% | 91.67% |
| 800 | 86.42% | 67.59% | 90.28% |

## Individual ensembles

### ALL_UM → AGFD

| Seed | Weights | Ensemble | Accuracy |
| :--- | :--- | ---: | ---: |
| 600 | `UM` | 1 | 86.30% |
| 600 | `UM` | 2 | 86.67% |
| 600 | `UM` | 3 | 84.44% |
| 700 | `UM_1` | 1 | 84.81% |
| 700 | `UM_1` | 2 | 86.67% |
| 700 | `UM_1` | 3 | 84.44% |
| 800 | `UM_2` | 1 | 87.04% |
| 800 | `UM_2` | 2 | 85.19% |
| 800 | `UM_2` | 3 | 87.04% |

### ALL_UA → MCC5

| Seed | Weights | Ensemble | Accuracy |
| :--- | :--- | ---: | ---: |
| 600 | `UA` | 1 | 69.44% |
| 600 | `UA` | 2 | 69.44% |
| 600 | `UA` | 3 | 69.44% |
| 700 | `UA_1` | 1 | 72.22% |
| 700 | `UA_1` | 2 | 72.22% |
| 700 | `UA_1` | 3 | 63.89% |
| 800 | `UA_2` | 1 | 72.22% |
| 800 | `UA_2` | 2 | 63.89% |
| 800 | `UA_2` | 3 | 66.67% |

### ALL_MA → UNSW

| Seed | Weights | Ensemble | Accuracy |
| :--- | :--- | ---: | ---: |
| 600 | `MA` | 1 | 89.58% |
| 600 | `MA` | 2 | 89.58% |
| 600 | `MA` | 3 | 87.50% |
| 700 | `MA_1` | 1 | 93.75% |
| 700 | `MA_1` | 2 | 91.67% |
| 700 | `MA_1` | 3 | 89.58% |
| 800 | `MA_2` | 1 | 93.75% |
| 800 | `MA_2` | 2 | 89.58% |
| 800 | `MA_2` | 3 | 87.50% |

