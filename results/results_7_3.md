# results_2026-09-29_19-43-26

3 seed(s), 3 training combination(s). Each model set is tested on the dataset left out of training. Accuracies are the average over the ensembles of a seed unless stated otherwise.

## Summary over seeds

| Training | Test | Seeds | Mean | Std | Min | Max |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: |
| ALL_UM | AGFD | 3 | 83.99% | 1.04% | 83.33% | 85.19% |
| ALL_UA | MCC5 | 3 | 39.51% | 1.07% | 38.89% | 40.74% |
| ALL_MA | UNSW | 3 | 62.50% | 3.18% | 59.03% | 65.28% |

## Mean accuracy per seed

| Seed | ALL_UM → AGFD | ALL_UA → MCC5 | ALL_MA → UNSW |
| :--- | ---: | ---: | ---: |
| 0 | 83.46% | 38.89% | 63.19% |
| 100 | 83.33% | 38.89% | 59.03% |
| 200 | 85.19% | 40.74% | 65.28% |

## Individual ensembles

### ALL_UM → AGFD

| Seed | Weights | Ensemble | Accuracy |
| :--- | :--- | ---: | ---: |
| 0 | `UM` | 1 | 83.33% |
| 0 | `UM` | 2 | 84.81% |
| 0 | `UM` | 3 | 82.22% |
| 100 | `UM_1` | 1 | 82.59% |
| 100 | `UM_1` | 2 | 82.96% |
| 100 | `UM_1` | 3 | 84.44% |
| 200 | `UM_2` | 1 | 85.19% |
| 200 | `UM_2` | 2 | 86.67% |
| 200 | `UM_2` | 3 | 83.70% |

### ALL_UA → MCC5

| Seed | Weights | Ensemble | Accuracy |
| :--- | :--- | ---: | ---: |
| 0 | `UA` | 1 | 41.67% |
| 0 | `UA` | 2 | 30.56% |
| 0 | `UA` | 3 | 44.44% |
| 100 | `UA_1` | 1 | 38.89% |
| 100 | `UA_1` | 2 | 44.44% |
| 100 | `UA_1` | 3 | 33.33% |
| 200 | `UA_2` | 1 | 41.67% |
| 200 | `UA_2` | 2 | 41.67% |
| 200 | `UA_2` | 3 | 38.89% |

### ALL_MA → UNSW

| Seed | Weights | Ensemble | Accuracy |
| :--- | :--- | ---: | ---: |
| 0 | `MA` | 1 | 68.75% |
| 0 | `MA` | 2 | 54.17% |
| 0 | `MA` | 3 | 66.67% |
| 100 | `MA_1` | 1 | 60.42% |
| 100 | `MA_1` | 2 | 58.33% |
| 100 | `MA_1` | 3 | 58.33% |
| 200 | `MA_2` | 1 | 66.67% |
| 200 | `MA_2` | 2 | 66.67% |
| 200 | `MA_2` | 3 | 62.50% |

