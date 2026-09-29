# Gear-CM

By: *Aleksanteri Hämäläinen* (aleksanteri.hamalainen@aalto.fi)

## How to run everything

### Training an ensemble (`train_repeated.py`)

`train_repeated.py` trains a model several times with the same config, so the runs can be tested together as an ensemble.

```
python train_repeated.py <config> [--runs N] [--seed SEED]
```

- `config`: config name in `configs/`, without `.yaml` (e.g. `FINAL`).
- `--runs`: number of training runs (default: 5, matching `ENSEMBLE_SIZE` in `test.py`).
- `--seed`: base seed. Run *i* uses `SEED + i`, so the ensemble members differ but the whole ensemble is reproducible. Overrides the config's `seed`; if neither is set, a random base seed is picked and logged.

All runs are saved to `model_weights/<datasets>`, named after the config's `train_datasets` with one letter per dataset in the order U (UNSW), M (MCC5), A (AGFD), S (ASD). For example, `[ALL_A]` gives `A`, `[ALL_UM]` gives `UM` and `[ALL_AM]` gives `MA`. If that directory already exists, the first free one of `<datasets>_1`, `<datasets>_2`, ... is used instead (e.g. `A_1`), so every invocation gets its own directory. Each run gets its own timestamped subdirectory inside it. If the config has `save: false`, it is overridden to `true`.

#### Examples

Train 5 models with the `FINAL` config:

```bash
python train_repeated.py FINAL
```

Train 10 models with the `FINAL` config:

```bash
python train_repeated.py FINAL --runs 10
```

Train a reproducible ensemble of 5 models with seeds 42–46:

```bash
python train_repeated.py FINAL --seed 42
```

Test the resulting ensemble (here `FINAL` trains on `ALL_A`, so the weights are in `model_weights/A`):

```bash
python test.py A UNSW
```

### Reproducing a run

Every run folder contains a `run_info.yaml` next to its checkpoints, with the run's `seed`, the git commit (and whether the working tree had uncommitted changes), and the full config used. To reproduce a run, check out that commit, make sure the config matches the saved one, and train with the same seed:

```bash
python main.py --config FINAL --seed 42
```

Single runs with `main.py` take the seed the same way: `--seed` first, then `seed` in the config, otherwise a random seed that is still saved in `run_info.yaml`. Runs are only bit-for-bit reproducible on the CPU; MPS and CUDA don't guarantee it.

### Testing a trained ensemble (`test.py`)

`test.py` evaluates a trained embedding-model ensemble on one dataset using few-shot prototype classification.

```
python test.py <weight_dir> <dataset>
```

- `weight_dir`: the ensemble weight directory. Relative names are resolved against `model_weights/`.
- `dataset`: one of `UNSW`, `MCC5` or `AGFD` (case-insensitive).

Run it from the repository root, because both `data/` and `model_weights/` are resolved relative to the current directory.

#### Expected layout

```
data/
  UNSW_gear_crack_OT_V2.feather
  MCC5-THU_OT_V3.feather
  AGFD_OT_V2.feather
model_weights/
  <weight_dir>/
    <member_1>/5.pth
    <member_2>/5.pth
    ...
```

Each subdirectory of `<weight_dir>` is one ensemble member. The subdirectories are sorted and split into groups of `ENSEMBLE_SIZE`, and `ENSEMBLE_CHECKPOINT_NUM` selects which checkpoint (`<n>.pth`) to load. Both are set at the top of `test.py`. Leftover members that don't fill a complete ensemble are ignored.

#### Examples

Test the `AGFD_1` weights on UNSW:

```bash
python test.py AGFD_1 UNSW
```

Test the `AGFD_1` weights on MCC5:

```bash
python test.py AGFD_1 MCC5
```

Test the `AGFD_1` weights on AGFD:

```bash
python test.py AGFD_1 AGFD
```

Show the help text:

```bash
python test.py --help
```

The script prints the accuracy for each operating condition, the average for each ensemble, and the overall average across all ensembles.

