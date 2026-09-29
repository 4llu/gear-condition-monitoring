# Gear-CM

By: *Aleksanteri Hämäläinen* (aleksanteri.hamalainen@aalto.fi)

## How to run everything

### Training an ensemble (`train_repeated.py`)

`train_repeated.py` trains a model several times with the same config, so the runs can be tested together as an ensemble.

```
python train_repeated.py <config> [--runs N]
```

- `config`: config name in `configs/`, without `.yaml` (e.g. `FINAL`).
- `--runs`: number of training runs (default: 5, matching `ENSEMBLE_SIZE` in `test.py`).

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

Test the resulting ensemble (here `FINAL` trains on `ALL_A`, so the weights are in `model_weights/A`):

```bash
python test.py A UNSW
```

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

