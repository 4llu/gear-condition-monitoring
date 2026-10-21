# Handoff: local-fault findings for the paper

State as of 2026-10-22, branch `local-only`. Every experiment, with its reason and
result, is in [LOG.md](LOG.md); the oracle is [scripts/oracle.py](scripts/oracle.py) and
the final outputs are in [final/](final/) (rerun commands in section 4). This note is the short version
to write from.

## 1. Final configuration (what to describe in the methods)

| Choice | Setting | Reason |
| :--- | :--- | :--- |
| Task | train on two datasets, test on the third (UM -> AGFD, UA -> MCC5, MA -> UNSW) | cross-dataset generalisation |
| Classes | healthy (= no localized tooth fault) / crack / missing tooth | see section 2 |
| Data | V6 files (`data_files` in FINAL.yaml) | cleaner, less overlapping windows |
| Model | MLP-Mixer, 2-D embedding, triplet loss (margin 1.4, all-triplet miner), 501 batches, checkpoint 5 | unchanged; SupCon and 8-D tested, not better; 501 batches give a bit-identical checkpoint 5 to 1001 |
| Input | log10 order-spectrum bins around 8 mesh harmonics, minus a healthy anchor | unchanged |
| Training anchors | 10 per episode (was 20) | with 20, anchors from the old ~20-window UNSW file were ~the recording mean; kept at 10 on the regenerated ~80-window file |
| Extra healthy units | wear units whose signature doesn't align with a local fault (AGFD severe wear, MCC5 mild wear) | MCC5 has one healthy unit, UNSW two healthy-labelled gears |
| AGFD training | all 8 healthy gears; mild crack in the crack class | max. healthy variation; correct labels |
| Training sensors | AGFD acc4, MCC5 gearbox x | all-sensor training traded ~6 UNSW points for ~4 MCC5 |
| Test sensors | training sensor by default (AGFD acc4, MCC5 gearbox x); `--all-sensors` lets all sensors vote (AGFD acc3+acc4, MCC5 gearbox x+y+z) | the results table below uses the default (training sensor) |
| Inference | L2-normalize window embeddings, average per unit, normalize; distances averaged over 3 members and all sensors | training geometry is normalized |
| Optional (UM only) | white-noise unit-variation augmentation, RMS 0.25 (sized from MCC5) | only justified without AGFD's healthy gears |

Reproduce: see [section 4](#4-final-results-resultspy-3-ensembles-of-3-per-seed).

## 2. Classes and dataset facts

- **AGFD "severe crack" (Severe_TFF_GP5) is an entirely missing tooth**; AGFD's only real
  crack is mild (GP9). AGFD faults were seeded into healthy gears: mild pitting GP1,
  severe pitting GP6, mild crack GP9, missing tooth GP5, mild wear GP7, severe wear GP2.
- **Dropped:** wear (no signature that transfers between gears; AGFD wear separability
  is gear identity), pitting (AGFD and MCC5 signatures don't align, cos -0.18 to 0.29),
  broken tooth (MCC5 only, tiny deviation).
- **UNSW S (mild crack) is the second healthy recording (H2)**, as in the original
  pipeline. State it: S is 0.32-0.35 log10 RMS from H, M 0.38-0.40.
- **UNSW H, S, M and L are four separate gears, each installed separately** (per the
  author). So every UNSW query comes from a different gear and installation than its
  prototype, for both classes: UNSW is the only dataset where the test is symmetric
  across gears for all classes (AGFD: one gear per fault; MCC5: one healthy unit and a
  fingerprint per unit). Remaining limits: two gears per class, one rig, and possible
  slow drift if the gears were recorded in severity order.
- **Missing tooth is consistent across datasets** (AGFD vs MCC5 cos 0.80); MCC5 crack M
  and L point in different directions (cos 0.20).
- **MCC5:** each (fault, severity) is its own unit with a fingerprint, so binary
  healthy-vs-fault is trivially separable; crack doesn't transfer between severities
  even in-domain (oracle crack 42-50%).
- **AGFD remounting changes a gear's spectrum as much as swapping gears** (0.29-0.40 vs
  0.37; within a recording 0.12), so own-gear baselines can't be evaluated on AGFD.
- **AGFD GP6/GP7** healthy spectra point along the AGFD fault direction; kept and
  reported per gear (no evidence of a measurement error).
- **AGFD missing tooth installation 1 = installation 2** at 500 rpm / 6 Nm and 1000 rpm /
  11 Nm (old and V6 files); excluding those pairs changes the result by 0.1 points.

## 3. Test protocols (`test.py`)

- **AGFD:** healthy support and query from two different gears out of 1, 2, 3, 4, 6, 7
  (never made faulty for a tested class), anchor = mean of the other four, all ordered
  pairs; fault prototype and queries from different installations (same physical gear,
  state it). Mild crack reported separately (detection limit). Healthy recall per gear.
  GP1-4 subset (secondary, not in the final table): set `AGFD_CLEAN_HEALTHY_GPS = [1, 2,
  3, 4]` in `test.py` and rerun `test.py` on the saved weights.
- **MCC5:** prototypes from the torque run, queries from the speed run; crack M <-> L.
  Anchor = first 7 torque-run healthy windows.
- **UNSW:** anchor = first half of H1, H1 rest = healthy prototype, H2 (= S) healthy
  queries, crack M <-> L.
- **Oracle:** logistic regression on the same inputs, trained on the test dataset's own
  labels under the same hold-outs.

## 4. Final results (`results.py`, 3 ensembles of 3 per seed)

Commit 01f301c, V6 data (UNSW regenerated with ~80 windows per recording), test sensor
= training sensor. Outputs in [final/](final/) (`final_s1000.csv`, `final_s2000.csv`,
`final_UM_augmentation.csv`, each with its full `.log`; `oracle.txt`). The runs'
`run_info.yaml` say `dirty: true` only because of uncommitted deletions of old
notebooks and result pages and a `.gitignore` fix; no code differed from 01f301c.

| Training -> test | Seed 1000 | Seed 2000 | Ensembles (s1000 / s2000) | Oracle |
| :--- | ---: | ---: | :--- | ---: |
| MA -> UNSW, H / crack | **97.9%** | **98.6%** | 95.8, 100, 97.9 / 100, 97.9, 97.9 | 72.9% (in-domain oracle calls S crack) |
| UA -> MCC5, H / crack / MT | **64.8%** | **71.3%** | 66.7, 66.7, 61.1 / 75.0, 69.4, 69.4 | 80.6% (x), 83.3% (z); crack 42-50 |
| UM -> AGFD, H / MT | **78.4%** | **77.0%** | 77.4, 76.9, 81.1 / 79.0, 77.5, 74.3 | 98.5% |
| UM + augmentation -> AGFD, H / MT | 80.2% | 79.6% | 80.6, 80.1, 79.8 / 80.8, 79.3, 78.6 | |
| UM -> AGFD mild crack (detection limit) | 59.1% | 56.8% | | 84.6% |
| UM + augmentation -> AGFD mild crack | 59.9% | 59.0% | | |

AGFD recall (H / MT): base 65.8 / 91.1 (s1000), 64.3 / 89.6 (s2000); augmented
71.3 / 89.1, 71.8 / 87.4. Healthy recall per query gear, base: GP1 81-84, GP2 86-88,
GP3 80-81, GP4 86-87, GP6 30-32, GP7 22-23; augmented: GP1-4 87-94, GP6 38-41,
GP7 24-25. MCC5 and UNSW per-class recall aren't printed by `test.py`; earlier
diagnostics (LOG.md) had MCC5 H ~69, crack ~40, MT 100, i.e. crack is the limit, as in
the oracle. The previous table (~20-window UNSW file, all-sensor voting) is in LOG.md.

Oracle (`final/oracle.txt`, logistic regression on the same inputs and hold-outs): MCC5
x/y/z 80.6/80.6/83.3%; AGFD acc4 MT 98.5% (GP1-4: 99.4%), mild crack 84.6% (86.4%);
acc3 MT 99.4% (100%), mild crack 72.0% (77.8%); UNSW 72.9% (H 50, crack 96).

### Reproduce

Run from the repo root with `~/micromamba/envs/torch3/bin/python`; each `results.py`
call takes ~1 h on the CPU (9 models x 3 scenarios), the oracle a minute. The calls can
run in parallel.

```bash
python results.py 1000 --output results/local_fault/final/final_s1000.csv > results/local_fault/final/final_s1000.log 2>&1
python results.py 2000 --output results/local_fault/final/final_s2000.csv > results/local_fault/final/final_s2000.log 2>&1
python results.py 1000 2000 --scenarios ALL_UM --output results/local_fault/final/final_UM_augmentation.csv --override '{"preprocessing_difference_batch": ["mask_blocks_non_episodic", "augment_amplitude", "add_unit_variation"]}' > results/local_fault/final/final_UM_augmentation.log 2>&1
python results/local_fault/scripts/oracle.py > results/local_fault/final/oracle.txt
```

Retest saved weights without training: `python test.py <weight_dir> AGFD|MCC5|UNSW`
(add `--all-sensors` for all-sensor voting). Weights of these runs (on disk only):

| Run | s1000 | s2000 |
| :--- | :--- | :--- |
| UM -> AGFD | `model_weights/UM_2` | `UM_3` |
| UM + augmentation | `UM_4` | `UM_5` |
| UA -> MCC5 | `UA_3` | `UA_2` |
| MA -> UNSW | `MA_3` | `MA_2` |

New `results.py` runs claim the next free `<code>_<n>` directory, so they never overwrite
these.

## 5. Story the results support

1. **Missing tooth transfers** across datasets and gear sets (AGFD 87-91%, MCC5 100%
   recall).
2. **Crack transfers where it exceeds healthy-unit variation** (UNSW M/L, 98-99%); MCC5 crack
   doesn't transfer between severities even in-domain.
3. **Mild faults mark a detection limit** set by healthy-gear and remount variation: AGFD
   mild crack deviates from its own healthy gear about as much as two healthy gears
   differ, and remounted gears without a local fault look like mild crack.
4. **The main limitation is missing healthy-unit variation in training:**
   - UNSW 74% -> 97-99% when wear units without a local-fault signature are added as
     healthy units (balancing batches showed it's diversity, not class weighting);
   - AGFD oracle (sees the healthy gears) ~99% vs 77-78% for the model trained without
     them;
   - per-gear AGFD healthy recall 80-88% on GP1-4, 22-32% on GP6/GP7;
   - unit-variation augmentation +2 to +3 points (healthy recall +6-7) (supporting only: white noise is a
     generic regularizer, the structured variant hurt).
5. **Fewer training windows reduce transfer:** V6 UNSW (~20 windows per recording)
   lowered MCC5 by ~11 points; subsampling old UNSW to 20 windows reproduces it, and
   10 training anchors recover most of it. The final results use the regenerated
   ~80-window file.

## 6. Tried and not adopted (one line each, details in LOG.md)

SupCon (UNSW -8, AGFD +2); 8-D embedding (MCC5 +5, nothing on top of 10 anchors);
balanced healthy/wear batches (MCC5 lower); all-sensor training; single best test
sensor (selection on test); own-gear anchors for AGFD faults (~70%); structured
augmentation; old UNSW file (MCC5 +, AGFD -7); removing amplitude augmentation; old six
AGFD gears; checkpoint 12.

## 7. Open items

- Why AGFD mild crack helps UNSW but costs MCC5; why AGFD severe wear as healthy lowers
  MCC5 healthy recall.
- The cos < 0.5 wear rule was written after seeing that AGFD mild wear hurts.
- Why fewer windows per recording hurt transfer (mechanism behind the anchor fix).
- Which block-level information the structured augmentation destroyed.
- Not updated to the new classes: `test_secondary.py` (FINAL doesn't use it).
- Run-to-failure dataset (test-only, no remount) is the natural next validation; plan in
  [RTF_PLAN.md](RTF_PLAN.md).
