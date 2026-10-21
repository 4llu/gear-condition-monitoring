# Final path and cleanup plan

State as of 2026-10-22. Evidence for every decision is in [LOG.md](LOG.md); the short
version is in [HANDOFF.md](HANDOFF.md).

## 1. Final path and design decisions

### Data

- V6 files: `AGFD_V6`, `MCC5_V6`, `UNSW_V6` (regenerated with ~80 windows per
  recording), set by `data_files` in `configs/FINAL.yaml`.
- Input: log10 order spectra, minus a healthy anchor; selected bins = 8 mesh harmonics
  with the 5 nearest and 5 furthest sidebands per side (checked against the full
  spectrum: no crack information is lost, the selection works as denoising).

### Classes

- healthy (= no localized tooth fault) / crack / missing tooth.
- AGFD severe "crack" (GP5) is relabelled missing tooth; AGFD mild crack (GP9) is in
  the crack class.
- MCC5: crack M and L, missing tooth (single severity).
- UNSW: S (mild crack) is used as the second healthy gear (H2); H, S, M, L are four
  separate, separately installed gears.
- Wear units whose signature doesn't align with a local fault are extra healthy units:
  AGFD severe wear (GP2), MCC5 mild wear.
- Dropped: wear as a class, pitting, broken tooth.

### Training

- MLP-Mixer, 2-D embedding, triplet loss (margin 1.4, all-triplet miner) on
  L2-normalized embeddings, 501 batches, checkpoint 5 (bit-identical to 1001 batches).
- Training on two datasets, testing on the third: UM -> AGFD, UA -> MCC5, MA -> UNSW.
- 10 anchors per episode; augmentations: block masking + amplitude scaling.
- AGFD: all 8 healthy gears, sensor acc4. MCC5: gearbox x. Batches unbalanced (wear adds
  to healthy).
- 9 models per combination = 3 ensembles of 3; seeds 1000 and 2000.
- Optional, UM only: white-noise unit-variation augmentation (RMS 0.25, sized from
  MCC5), only justified for training without healthy-gear variation.

### Testing (`test.py`)

- AGFD: healthy vs missing tooth, healthy support/query/anchor rotated over gears
  1, 2, 3, 4, 6, 7; faults across installations; mild crack reported separately as the
  detection limit; per-gear healthy recall; GP1-4 subset as a secondary result.
- MCC5: healthy / crack / missing tooth, torque-run prototypes, speed-run queries,
  crack across severities (report per direction, and same severity as a variant).
- UNSW: healthy / crack, first half of H1 as anchor, rest as prototype, H2 (= S)
  queries, crack M <-> L.
- Default test sensor = training sensor (AGFD acc4, MCC5 gearbox x); `--all-sensors`
  lets all sensors vote (AGFD acc3 + acc4, MCC5 gearbox x + y + z). Window embeddings
  L2-normalized, averaged per unit, normalized; soft voting over members (and sensors);
  no outlier dropping.
- Reported with every result: per-class recall, oracle ceiling (`scripts/oracle.py`),
  both seeds.

### Still to do for the final numbers

- ~~Rerun the official results on the ~80-window UNSW file~~ done: `final/`, commands
  and table in HANDOFF section 4.
- Add the MCC5 per-direction breakdown (and same-severity variant) to `test.py`.
- Add the code that creates the V6 files to the repo (`src/data/data_processing_*_simple.py`
  are untracked so far).

## 2. Cleanup: what the final results don't need

Anything that changes how random numbers are drawn during training (e.g. removing the
validation/test loaders) changes the trained models, so clean up first and rerun the
official results once afterwards.

### Remove from the code

| Where | What | Note |
| :--- | :--- | :--- |
| repo root | `test_secondary.py` (~~`SCENARIO_*.ipynb`~~ deleted in the working tree) | old evaluation |
| ~~`src/data/sampler_ASD.py` + ASD branches~~ | ASD dataset | done (c02c63c) |
| ~~`src/data/sampler_AGFD.py`, `sampler_MCC5.py`, `sampler_UNSW.py` standalone samplers~~ | standalone loaders, episodic batch samplers, `get_*_data` | done: only the `*_Dataset` classes remain |
| ~~`src/data/data.py`~~ | non-`ALL_` dataset names | done |
| ~~`src/data/preprocessing_batch.py`~~ | unused collate and preprocessing functions | done: `collate_all`, `amplitude_log10`, `mask_blocks_non_episodic`, `augment_amplitude`, `add_unit_variation` remain |
| `src/data/sampler_ALL.py` | `balance_healthy_like` / `num_query_groups` | tested, not adopted; off by default |
| ~~`src/models/prototypical.py`, `classical.py`~~ | other models | done; the MLP backbone stays for the backbone comparison |
| ~~`src/models/embedding.py`~~ | NCA, contrastive, circle, SupCon, proto_loss branches; `intermediate_L2` in MlpMixer | done: triplet only |
| `src/training/training.py` | ~~test loop, Optuna hooks~~ (done; validation kept on `validation_datasets: [ALL_U]` with its own random generator), TensorBoard, commented code | |
| `test.py` | `remove_anchor_outliers`, `remove_batch_outliers`, outlier/averaging switches in `embedding_specs`, hard voting (~~commented-out config block~~ done) | final test uses fixed settings |
| `configs/FINAL.yaml` | ASD, validation and commented-out test blocks, SupCon block, `episodic_training`, `k_shot` / `n_query` if unused after the above | keep only what the final run reads |
| `src/data/data_processing_*.ipynb`, `src/data/signal_analysis.py`, `signal_processing/order_tracking.py`, `src/utils/signals.py` | create the old data files | replace with the V6 preparation code, or keep only what it uses |
| tracked junk | `.DS_Store` files, `__pycache__/*.pyc` | `.gitignore` has `.DS_STore` (wrong case); fix and untrack |

### Keep as notes and logs (not code paths)

- `results/local_fault/HANDOFF.md`, `LOG.md`, `RTF_PLAN.md`, this file, the final CSVs.
- `results/local_fault/scripts/oracle.py`; the other analysis scripts were deleted
  (they are in git history before 01f301c; LOG.md names them).

### Outside git (disk only)

- `model_weights/`: the final runs (HANDOFF section 4: `UM_2`-`UM_5`, `UA_2`, `UA_3`,
  `MA_2`, `MA_3`). The archives, ablations and all older runs were deleted.
- `data/`: the old files (`AGFD_OT_V2`, `MCC5-THU_OT_V3`, `UNSW_gear_crack_OT_V2`) are
  only needed to reproduce old-data comparisons in the log.
- `results/`: old result pages (`results_*.md`, `FINAL_1.md`, `FINAL_2.md`,
  `results_FINAL_FULL.md`, `csv_to_md.py`) belong to the earlier class setup.
- `logs/`: emptied (TensorBoard writes there when enabled); the untracked run logs in `results/local_fault/` were deleted, the final runs keep theirs in `final/`.
