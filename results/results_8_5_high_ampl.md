# results_2026-09-29_21-50-03

3 seed(s), 3 training combination(s). Each model set is tested on the dataset left out of training. Accuracies are the average over the ensembles of a seed unless stated otherwise.

## Summary over seeds

| Training | Test | Seeds | Mean | Std | Min | Max |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: |
| ALL_UM | AGFD | 3 | 84.73% | 1.01% | 83.58% | 85.43% |
| ALL_UA | MCC5 | 3 | 52.47% | 1.41% | 50.93% | 53.70% |
| ALL_MA | UNSW | 3 | 87.27% | 1.75% | 85.42% | 88.89% |

## Mean accuracy per seed

| Seed | ALL_UM → AGFD | ALL_UA → MCC5 | ALL_MA → UNSW |
| :--- | ---: | ---: | ---: |
| 0 | 85.19% | 52.78% | 85.42% |
| 100 | 85.43% | 53.70% | 88.89% |
| 200 | 83.58% | 50.93% | 87.50% |

## Individual ensembles

### ALL_UM → AGFD

| Seed | Weights | Ensemble | Accuracy |
| :--- | :--- | ---: | ---: |
| 0 | `UM` | 1 | 86.30% |
| 0 | `UM` | 2 | 84.07% |
| 0 | `UM` | 3 | 85.19% |
| 100 | `UM_1` | 1 | 82.96% |
| 100 | `UM_1` | 2 | 85.93% |
| 100 | `UM_1` | 3 | 87.41% |
| 200 | `UM_2` | 1 | 83.70% |
| 200 | `UM_2` | 2 | 85.56% |
| 200 | `UM_2` | 3 | 81.48% |

### ALL_UA → MCC5

| Seed | Weights | Ensemble | Accuracy |
| :--- | :--- | ---: | ---: |
| 0 | `UA` | 1 | 61.11% |
| 0 | `UA` | 2 | 47.22% |
| 0 | `UA` | 3 | 50.00% |
| 100 | `UA_1` | 1 | 58.33% |
| 100 | `UA_1` | 2 | 52.78% |
| 100 | `UA_1` | 3 | 50.00% |
| 200 | `UA_2` | 1 | 55.56% |
| 200 | `UA_2` | 2 | 52.78% |
| 200 | `UA_2` | 3 | 44.44% |

### ALL_MA → UNSW

| Seed | Weights | Ensemble | Accuracy |
| :--- | :--- | ---: | ---: |
| 0 | `MA` | 1 | 87.50% |
| 0 | `MA` | 2 | 77.08% |
| 0 | `MA` | 3 | 91.67% |
| 100 | `MA_1` | 1 | 87.50% |
| 100 | `MA_1` | 2 | 91.67% |
| 100 | `MA_1` | 3 | 87.50% |
| 200 | `MA_2` | 1 | 91.67% |
| 200 | `MA_2` | 2 | 83.33% |
| 200 | `MA_2` | 3 | 87.50% |

