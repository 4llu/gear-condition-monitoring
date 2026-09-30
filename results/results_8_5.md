# results_2026-09-29_21-01-28

3 seed(s), 3 training combination(s). Each model set is tested on the dataset left out of training. Accuracies are the average over the ensembles of a seed unless stated otherwise.

## Summary over seeds

| Training | Test | Seeds | Mean | Std | Min | Max |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: |
| ALL_UM | AGFD | 3 | 86.21% | 0.58% | 85.56% | 86.67% |
| ALL_UA | MCC5 | 3 | 59.26% | 1.60% | 58.33% | 61.11% |
| ALL_MA | UNSW | 3 | 88.43% | 2.12% | 86.11% | 90.28% |

## Mean accuracy per seed

| Seed | ALL_UM → AGFD | ALL_UA → MCC5 | ALL_MA → UNSW |
| :--- | ---: | ---: | ---: |
| 0 | 86.67% | 58.33% | 86.11% |
| 100 | 86.42% | 61.11% | 90.28% |
| 200 | 85.56% | 58.33% | 88.89% |

## Individual ensembles

### ALL_UM → AGFD

| Seed | Weights | Ensemble | Accuracy |
| :--- | :--- | ---: | ---: |
| 0 | `UM` | 1 | 88.15% |
| 0 | `UM` | 2 | 86.30% |
| 0 | `UM` | 3 | 85.56% |
| 100 | `UM_1` | 1 | 84.81% |
| 100 | `UM_1` | 2 | 86.30% |
| 100 | `UM_1` | 3 | 88.15% |
| 200 | `UM_2` | 1 | 85.56% |
| 200 | `UM_2` | 2 | 86.67% |
| 200 | `UM_2` | 3 | 84.44% |

### ALL_UA → MCC5

| Seed | Weights | Ensemble | Accuracy |
| :--- | :--- | ---: | ---: |
| 0 | `UA` | 1 | 58.33% |
| 0 | `UA` | 2 | 55.56% |
| 0 | `UA` | 3 | 61.11% |
| 100 | `UA_1` | 1 | 61.11% |
| 100 | `UA_1` | 2 | 61.11% |
| 100 | `UA_1` | 3 | 61.11% |
| 200 | `UA_2` | 1 | 61.11% |
| 200 | `UA_2` | 2 | 55.56% |
| 200 | `UA_2` | 3 | 58.33% |

### ALL_MA → UNSW

| Seed | Weights | Ensemble | Accuracy |
| :--- | :--- | ---: | ---: |
| 0 | `MA` | 1 | 87.50% |
| 0 | `MA` | 2 | 81.25% |
| 0 | `MA` | 3 | 89.58% |
| 100 | `MA_1` | 1 | 89.58% |
| 100 | `MA_1` | 2 | 91.67% |
| 100 | `MA_1` | 3 | 89.58% |
| 200 | `MA_2` | 1 | 89.58% |
| 200 | `MA_2` | 2 | 85.42% |
| 200 | `MA_2` | 3 | 91.67% |

