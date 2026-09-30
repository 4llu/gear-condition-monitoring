# results_2026-09-29_22-36-13

3 seed(s), 3 training combination(s). Each model set is tested on the dataset left out of training. Accuracies are the average over the ensembles of a seed unless stated otherwise.

## Summary over seeds

| Training | Test | Seeds | Mean | Std | Min | Max |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: |
| ALL_UM | AGFD | 3 | 87.65% | 1.46% | 86.42% | 89.26% |
| ALL_UA | MCC5 | 3 | 72.22% | 1.85% | 70.37% | 74.07% |
| ALL_MA | UNSW | 3 | 93.52% | 0.40% | 93.06% | 93.75% |

## Mean accuracy per seed

| Seed | ALL_UM → AGFD | ALL_UA → MCC5 | ALL_MA → UNSW |
| :--- | ---: | ---: | ---: |
| 300 | 86.42% | 70.37% | 93.75% |
| 400 | 87.28% | 74.07% | 93.06% |
| 500 | 89.26% | 72.22% | 93.75% |

## Individual ensembles

### ALL_UM → AGFD

| Seed | Weights | Ensemble | Accuracy |
| :--- | :--- | ---: | ---: |
| 300 | `UM` | 1 | 84.81% |
| 300 | `UM` | 2 | 86.30% |
| 300 | `UM` | 3 | 88.15% |
| 400 | `UM_1` | 1 | 85.93% |
| 400 | `UM_1` | 2 | 86.67% |
| 400 | `UM_1` | 3 | 89.26% |
| 500 | `UM_2` | 1 | 87.78% |
| 500 | `UM_2` | 2 | 89.26% |
| 500 | `UM_2` | 3 | 90.74% |

### ALL_UA → MCC5

| Seed | Weights | Ensemble | Accuracy |
| :--- | :--- | ---: | ---: |
| 300 | `UA` | 1 | 72.22% |
| 300 | `UA` | 2 | 77.78% |
| 300 | `UA` | 3 | 61.11% |
| 400 | `UA_1` | 1 | 69.44% |
| 400 | `UA_1` | 2 | 75.00% |
| 400 | `UA_1` | 3 | 77.78% |
| 500 | `UA_2` | 1 | 72.22% |
| 500 | `UA_2` | 2 | 75.00% |
| 500 | `UA_2` | 3 | 69.44% |

### ALL_MA → UNSW

| Seed | Weights | Ensemble | Accuracy |
| :--- | :--- | ---: | ---: |
| 300 | `MA` | 1 | 93.75% |
| 300 | `MA` | 2 | 91.67% |
| 300 | `MA` | 3 | 95.83% |
| 400 | `MA_1` | 1 | 91.67% |
| 400 | `MA_1` | 2 | 91.67% |
| 400 | `MA_1` | 3 | 95.83% |
| 500 | `MA_2` | 1 | 91.67% |
| 500 | `MA_2` | 2 | 93.75% |
| 500 | `MA_2` | 3 | 95.83% |

