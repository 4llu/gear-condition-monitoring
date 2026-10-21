# Gear-CM

By: *Aleksanteri Hämäläinen* (aleksanteri.hamalainen@aalto.fi)

Few-shot detection of localized gear tooth faults that generalizes across datasets: an
embedding model is trained on two gear datasets and tested on the third with
prototype classification. Classes are healthy (no localized tooth fault), crack and
missing tooth.

Findings, design decisions and every experiment behind the current configuration:
[results/local_fault/HANDOFF.md](results/local_fault/HANDOFF.md) (summary) and
[results/local_fault/LOG.md](results/local_fault/LOG.md) (full log).

## Overview

| Part | Where |
| :--- | :--- |
| Configuration | `configs/FINAL.yaml` |
| Data loading and episodes | `src/data/sampler_ALL.py` (datasets in `sampler_UNSW.py`, `sampler_MCC5.py`, `sampler_AGFD.py`) |
| Preprocessing and augmentation | `src/data/preprocessing_individual.py`, `src/data/preprocessing_batch.py` |
| Model | `src/models/embedding.py` (triplet loss), backbones in `src/models/backbones/` (`MlpMixer`, `MLP`) |
| Training loop | `src/training/training.py` |
| Testing | `test.py` |
| All scenarios, train + test | `results.py` |
| Final results and oracle | `results/local_fault/final/`, oracle code `results/local_fault/scripts/oracle.py` |

### Data

`data/` holds one feather file per dataset with order-tracked, time-synchronously
averaged spectra: `UNSW_V6.feather`, `MCC5_V6.feather`, `AGFD_V6.feather`. Training reads
the files named in the config's `data_files`; `test.py` always tests on the V6 files.

Inputs are log10 spectra at 8 gear mesh harmonics with the 5 nearest and 5 furthest
sidebands on each side, minus a healthy anchor of the same operating condition.

- **UNSW:** healthy (H and S, the mildest crack, used as a second healthy gear) and
  crack (M, L). The four are separate gears.
- **MCC5:** healthy (plus mild wear as an extra healthy unit), crack (M, L), missing
  tooth.
- **AGFD:** healthy (8 gears, plus severe wear as an extra healthy unit), crack (mild),
  missing tooth (the "severe crack", relabelled when loading).

### Configuration (`configs/FINAL.yaml`)

- `train_datasets`: `[ALL_<letters>]`, one letter per dataset (U: UNSW, M: MCC5,
  A: AGFD), e.g. `[ALL_MA]`.
- `validation_datasets`: an `ALL_<letters>` dataset that is only monitored during
  training (loss and retrieval metrics, no checkpoint selection), default `[ALL_U]`.
  It draws from its own random generator, so it doesn't change the training samples.
- `<dataset>_train_*` / `<dataset>_validation_*`: speeds, loads, sensors, classes and
  severities of each dataset.
- `backbone`: `MlpMixer` (default) or `MLP` (`model_args: {hidden_sizes: [...]}`, input
  size defaults to the selected bins), for backbone comparisons.

## How to run everything

Run every script from the repository root (`data/` and `model_weights/` are resolved
relative to it).

### Training and testing every scenario (`results.py`)

`results.py` trains every two-dataset combination and tests it on the dataset left out
(`ALL_UM` -> AGFD, `ALL_UA` -> MCC5, `ALL_MA` -> UNSW), writing one CSV row per
ensemble. This is how the final results are produced.

```
python results.py <seed> [<seed> ...] [--models N] [--scenarios ALL_UM ...] [--override JSON] [--all-sensors]
```

- `--models`: models per training combination (default: 9, three ensembles of three).
- `--override`: config entries to change for every training, stored in each run's
  `run_info.yaml`.
- `--all-sensors`: test with all sensors voting (see below).

Final results, both seeds (the exact commands that wrote `results/local_fault/final/`,
with output paths, the oracle and the weight directories, are in
[HANDOFF section 4](results/local_fault/HANDOFF.md#reproduce)):

```bash
python results.py 1000 2000
```

The unit-variation augmentation (only meant for training without AGFD's healthy gears):

```bash
python results.py 1000 2000 --scenarios ALL_UM --override '{"preprocessing_difference_batch": ["mask_blocks_non_episodic", "augment_amplitude", "add_unit_variation"]}'
```

Parallel `results.py` calls are safe: each training claims its own
`model_weights/<code>_<n>` directory.

### Training an ensemble (`train_repeated.py`)

`train_repeated.py` trains a model several times with the same config, so the runs can
be tested together as an ensemble.

```
python train_repeated.py <config> [--runs N] [--seed SEED]
```

- `config`: config name in `configs/`, without `.yaml` (e.g. `FINAL`).
- `--runs`: number of training runs (default: 9, three ensembles of `ENSEMBLE_SIZE` = 3
  in `test.py`).
- `--seed`: base seed. Run *i* uses `SEED + i`, so the ensemble members differ but the
  whole ensemble is reproducible. Overrides the config's `seed`; if neither is set, a
  random base seed is picked and logged.

All runs are saved to `model_weights/<datasets>`, named after the config's
`train_datasets` with one letter per dataset in the order U (UNSW), M (MCC5), A (AGFD).
For example, `[ALL_MA]` gives `MA` and `[ALL_UM]` gives `UM`. If that directory already
exists, the first free one of `<datasets>_1`, `<datasets>_2`, ... is used instead, so
every invocation gets its own directory. Each run gets its own subdirectory named after
its start time and seed. If the config has `save: false`, it is overridden to `true`.

Train an ensemble with `FINAL` (trains on `ALL_MA`, so the weights go to
`model_weights/MA`) and test it on the dataset left out:

```bash
python train_repeated.py FINAL --seed 1000
python test.py MA UNSW
```

### Reproducing a run

Every run folder contains a `run_info.yaml` next to its checkpoints, with the run's
`seed`, the git commit (and whether the working tree had uncommitted changes), and the
full config used. To reproduce a run, check out that commit, make sure the config
matches the saved one, and train with the same seed:

```bash
python main.py --config FINAL --seed 1000
```

Single runs with `main.py` take the seed the same way: `--seed` first, then `seed` in
the config, otherwise a random seed that is still saved in `run_info.yaml`. Runs are
only bit-for-bit reproducible on the CPU with the same number of threads; MPS and CUDA
don't guarantee it.

### Testing a trained ensemble (`test.py`)

`test.py` evaluates a trained ensemble on one dataset with few-shot prototype
classification.

```
python test.py <weight_dir> <dataset> [--all-sensors]
```

- `weight_dir`: the ensemble weight directory. Relative names are resolved against
  `model_weights/`.
- `dataset`: one of `UNSW`, `MCC5` or `AGFD` (case-insensitive).
- `--all-sensors`: let all sensors of the dataset vote instead of only the training
  sensor.

```
model_weights/
  <weight_dir>/
    <member_1>/5.pth
    <member_2>/5.pth
    ...
```

Each subdirectory of `<weight_dir>` is one ensemble member. The subdirectories are
sorted and split into groups of `ENSEMBLE_SIZE`, and `ENSEMBLE_CHECKPOINT_NUM` selects
which checkpoint (`<n>.pth`) to load (5, saved at batch 500). Both are set at the top of
`test.py`. Leftover members that don't fill a complete ensemble are ignored. The input
and model settings are read from the members' `run_info.yaml`.

The script prints the accuracy for each operating condition, the average for each
ensemble, and the overall average across all ensembles.

#### What is tested

The ensemble members vote by averaging their distances to the class prototypes. Window
embeddings are L2-normalized, averaged per recording, and normalized again. By default
each dataset is tested with its training sensor (AGFD acc4, MCC5 gearbox x); with
`--all-sensors`, every (model, sensor) pair votes over all sensors of the tested dataset
(AGFD acc3 + acc4, MCC5 gearbox x + y + z).

- **AGFD:** healthy vs missing tooth. Support and query come from two different healthy
  gears out of `AGFD_CLEAN_HEALTHY_GPS` (gears that never had a tested fault seeded in
  them), the anchor is averaged over the remaining ones, and all ordered gear pairs are
  used. Fault prototype and queries come from different installations. Healthy vs mild
  crack is reported separately as the detection limit, and healthy recall is printed
  per gear.
- **MCC5:** healthy / crack / missing tooth, prototypes from the torque run, queries from
  the speed run, crack also across severities (M <-> L).
- **UNSW:** healthy / crack, the first half of H1 is the anchor and the rest the healthy
  prototype, H2 (= S) the healthy queries, crack M <-> L.
