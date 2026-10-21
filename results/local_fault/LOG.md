# Local-fault retraining log

Classes: healthy / crack / missing tooth. Training on two datasets, testing on the
third. 9 models per training combination (3 ensembles of 3), seed 1000, checkpoint 5.
AGFD training uses all eight healthy gears. Test protocols as in `test.py`:

- AGFD: healthy vs missing tooth, healthy roles rotated over gears 1, 2, 3, 4, 6, 7
  (anchor averaged over the four gears not used as support or query), faults across
  installations. Mild crack reported separately (detection limit).
- MCC5: healthy / crack / missing tooth, prototypes from the torque run, queries from
  the speed run, crack also across severities (M <-> L).
- UNSW: healthy / crack, H1 prototype, H2 (= S) queries, crack M <-> L.

Every variant below lists its reason, so the final choice does not rest on test scores
alone. "Unjustified improvements" are changes that improved results without a clear
reason yet; they are noted for later investigation and not adopted.

## v0 baseline (commit 3c6b976)

| Training -> test | Mean | Ensembles |
| :--- | ---: | :--- |
| UM -> AGFD (healthy / missing tooth) | 74.1% | 74.4, 72.4, 75.4 |
| UA -> MCC5 | 78.7% | 80.6, 80.6, 75.0 |
| MA -> UNSW | 74.3% | 72.9, 72.9, 77.1 |

AGFD: healthy recall 68%, missing tooth 80%; healthy per query gear GP1 80, GP2 86,
GP3 85, GP4 86, GP6 36, GP7 32. Mild crack (detection limit): 61.4%.

For comparison, the old class setup (healthy / crack / wear, AGFD missing tooth labelled
crack) gave MA -> UNSW 95.8% (results/FINAL_2.md).

### v0 diagnostics (inference variants on the same weights)

Training uses the triplet loss on L2-normalized embeddings
(`lpnorm_embeddings_training: true`), but `test.py` compares raw, batch-averaged
embeddings with Euclidean distance (`lpnorm_embeddings_inference: false`). Training only
constrains the direction of an embedding, so its length is unconstrained noise at test
time. `norm_mean` normalizes each window embedding, averages, and normalizes again,
which is the geometry training optimizes. `*_win` classifies every window separately
(secondary metric).

| Test | raw_mean (test.py) | norm_mean | raw_win | norm_win |
| :--- | ---: | ---: | ---: | ---: |
| AGFD healthy / missing tooth | 74.1% (H 68, MT 80) | 76.3% (H 72, MT 80) | 72.7% | 74.5% |
| AGFD healthy / mild crack | 61.4% | 60.2% | 60.5% | 59.7% |
| MCC5 | 78.7% (H 83, C 53, MT 100) | 86.1% (H 92, C 67, MT 100) | 76.5% | 80.3% |
| UNSW | 74.3% (H 65, C 83) | 72.2% (H 69, C 75) | 73.1% | 71.6% |

Old MA weights (old classes) on the identical UNSW protocol: 95.4% (H 91, C 100), so
the UNSW drop comes from training, not from the test.

## Round 1: MA ablations (test UNSW), seed 1000, FINAL + one override each

| Variant | Reason | raw_mean | norm_mean |
| :--- | :--- | ---: | ---: |
| v0 baseline | | 74.3% (H 65, C 83) | 72.2% (H 69, C 75) |
| no amplitude augmentation | augmentation scales the dB difference by 0.8-1.2, may blur severity | 72.9% (H 71, C 75) | 68.1% |
| AGFD healthy gears 2,3,4,5,6,9 (old six) | GP1/GP7 flagged "not good" in old config, GP6/GP7 look fault-like | 73.6% (H 72, C 75) | 75.7% |
| AGFD mild crack added to crack class | correctly labelled crack data that was excluded; AGFD crack direction aligns with UNSW crack (cos 0.52) | **79.9%** (H 68, C 92) | **84.0%** (H 89, C 79) |
| missing tooth labelled crack (old semantics, diagnostic only) | locate the regression | 63.2% (H 46, C 81) | 75.0% |

Neither the augmentation nor the healthy gear choice explains the drop. Adding AGFD
mild crack back helps; relabelling missing tooth as crack does not recover the old
95%.

## Round 2: locating the UNSW regression (MA, test UNSW)

| Variant | raw_mean | norm_mean |
| :--- | ---: | ---: |
| old configuration on new code (healthy/crack/wear, missing tooth labelled crack, six AGFD gears) | 95.8% (H 92, C 100) | 93.1% |
| new classes + wear as training-only class, AGFD mild crack in crack | **91.7%** (H 83, C 100) | **92.4%** (H 85, C 100) |
| v0 baseline, seed 2000 | 66.7% (H 60, C 74) | 68.8% |
| AGFD mild crack added, seed 2000 | 77.8% (H 67, C 89) | 79.2% |

- The code reproduces the old 95.8%, so the regression comes from the class setup.
- Seed noise for one 9-model training is about 4-8 points (74.3 vs 66.7 for the
  baseline). The AGFD mild crack gain (+10) holds for both seeds.
- Wear as an extra training class restores UNSW to ~92%. The mechanism is not yet
  clear, so it is tested further in round 3.

## Round 3: why does wear help? (MA, test UNSW, AGFD mild crack in crack class)

| Variant | raw_mean | norm_mean |
| :--- | ---: | ---: |
| wear (separate class) from MCC5 only | 85.4% (H 76, C 94) | 88.9% |
| wear (separate class) from AGFD only | 89.6% (H 79, C 100) | 95.1% |
| MCC5 broken tooth as extra class instead of wear | 82.6% (H 68, C 97) | 86.1% |
| **wear labelled healthy** | **96.5%** (H 93, C 100) | 95.1% (H 93, C 97) |

Interpretation: wear does not help as a fault class; it helps as additional units
without a local tooth fault. UNSW and MCC5 have one healthy unit each, so the model
otherwise never sees unit-to-unit variation and calls it crack. Earlier measurements
support treating these units as "no local fault": AGFD severe wear sits next to its
own healthy gear (GP2), and MCC5 mild/medium wear was not recognisable on an unseen
severity. The healthy class therefore means "no localized tooth fault".

Exception: MCC5 severe wear points in the same direction as missing tooth (cos 0.86),
so it is not labelled healthy (would contradict the missing tooth class). Wear used as
healthy: AGFD S and L, MCC5 S and M. Needs per-class MCC5 severities
(`MCC5_<split>_class_severities`).

## v1: AGFD mild crack in crack + wear as healthy (AGFD S/L, MCC5 S/M), all combinations

| Training -> test | raw_mean | norm_mean | raw_win | norm_win |
| :--- | ---: | ---: | ---: | ---: |
| UM -> AGFD healthy / missing tooth | 71.9% (H 65, MT 78) | 72.7% | 70.2% | 70.5% |
| UM -> AGFD healthy / mild crack | 62.2% | 62.4% | 60.8% | 61.0% |
| UA -> MCC5 | 75.9% (H 78, C 50, MT 100) | 79.6% | 72.2% | 71.4% |
| MA -> UNSW | **68.8%** (H 51, C 86) | 72.9% | 64.6% | 69.6% |

v1 is worse than round 3's wear-as-healthy variant (96.5%). The two differ only in
which wear units are labelled healthy: round 3 used MCC5 wear M and L plus AGFD wear L;
v1 used MCC5 wear S and M plus AGFD wear S and L. Round 4 tests which units matter.

Note on the UNSW test: crack recall is high in almost every variant (75-100%), and
almost all variation is in healthy recall. UNSW healthy queries are S (mild crack,
labelled healthy by design; 0.32 log10 RMS from H vs 0.36 for M). So the UNSW score
mostly measures whether the model ignores a mild crack.

## Round 4: which wear units help as "healthy"? (MA, test UNSW, AGFD mild crack in crack)

| Wear units labelled healthy | raw_mean | norm_mean |
| :--- | ---: | ---: |
| MCC5 M, L + AGFD L, seed 2000 (round 3 variant replicated) | 89.6% (H 79, C 100) | 93.8% |
| MCC5 L only | 89.6% (H 85, C 94) | 90.3% |
| MCC5 S, M + AGFD L | **91.7%** (H 83, C 100) | **94.4%** |
| MCC5 S, M, L + AGFD S, L | 72.9% (H 62, C 83) | 78.5% |
| (v1) MCC5 S, M + AGFD S, L | 68.8% (H 51, C 86) | 72.9% |

The two failing variants are the ones that label AGFD mild wear (S) healthy.

Input-space signatures (mean log-spectrum difference to healthy, cosine):

| Wear unit | Strongest alignment with a local-fault signature | Healthy-like? |
| :--- | :--- | :--- |
| AGFD wear S (gear GP7) | AGFD crack 0.67, AGFD missing tooth 0.62; own gear GP7 0.85 | no |
| AGFD wear L (gear GP2) | negative with all faults (-0.29 to -0.63) | yes |
| MCC5 wear S | max 0.45 (MCC5 crack M) | yes |
| MCC5 wear M | MCC5 missing tooth 0.69 | no |
| MCC5 wear L | MCC5 missing tooth 0.86 | no |

Rule: a wear unit is labelled healthy only if its signature does not align with a
local-fault signature (cos < 0.5). Caveat: this rule was written down after seeing that
AGFD wear S hurts; the same reasoning had been applied to MCC5 wear L beforehand.
v2 uses AGFD wear L + MCC5 wear S.

## v2: AGFD mild crack in crack + wear as healthy (AGFD L, MCC5 S)

| Training -> test | raw_mean | norm_mean | raw_win | norm_win |
| :--- | ---: | ---: | ---: | ---: |
| MA -> UNSW | **98.6%** (H 97, C 100) | 98.6% | 93.1% | 93.1% |
| MA -> UNSW, seed 2000 | 97.2% (H 94, C 100) | 97.2% | 92.1% | 92.0% |
| UM -> AGFD healthy / missing tooth | 74.1% (H 72, MT 76) | 74.7% (H 73, MT 76) | 71.7% | 72.1% |
| UM -> AGFD healthy / mild crack | 62.1% | 60.7% | 59.9% | 59.7% |
| UA -> MCC5 | 73.1% (H 72, C 53, MT 94) | 75.9% (H 81, C 53, MT 94) | 69.4% | 69.9% |

AGFD healthy recall per query gear (norm_mean): GP1 83, GP2 88, GP3 87, GP4 87, GP6 49,
GP7 45.

UNSW is solved and stable over seeds. AGFD is unchanged from v0. MCC5 is lower than v0
(78.7 / 86.1); round 5 separates seed noise from the two v2 components.

## Round 5: is the MCC5 drop real? (UA, test MCC5)

| Variant | raw_mean | norm_mean |
| :--- | ---: | ---: |
| v0, seed 1000 | 78.7% (H 83, C 53, MT 100) | 86.1% (H 92, C 67) |
| v0, seed 2000 | 78.7% (H 86, C 50, MT 100) | 79.6% (H 86, C 53) |
| v2, seed 1000 | 73.1% (H 72, C 53, MT 94) | 75.9% (H 81, C 53) |
| v2, seed 2000 | 70.4% (H 67, C 44, MT 100) | 72.2% (H 67, C 50) |
| v0 + AGFD wear L as healthy only | 71.3% | 74.1% |
| v0 + AGFD mild crack as crack only | 72.2% | 75.0% |

The MCC5 cost of v2 is real (about 5-8 points) and comes from both components, almost
entirely through healthy recall.

MCC5 oracle under the same protocol (logistic regression on the model inputs, crack
severity s1 + torque run for training, other severity + speed run for testing):
**77.8%** (healthy 100, crack **33**, missing tooth 100). Crack does not transfer
between MCC5 severities even in-domain (MCC5 crack M vs L signatures: cos 0.20), so
MCC5 crack recall around 50% is a data limit, not a model failure.

## Decision

Adopted v2 + normalized inference (commit 0c7f8bd):

- AGFD mild crack in the crack class (correctly labelled crack data).
- Wear units labelled healthy where their signature does not align with a local fault
  (AGFD L, MCC5 S): extra units without a local fault, which UNSW and MCC5 lack.
- L2-normalize window embeddings before averaging at test time (training geometry).
- Same pipeline for all three combinations. Cost: MCC5 -5 to -8 points vs v0, gain:
  UNSW +23 points.

Rejected: removing amplitude augmentation (no gain), restricting AGFD healthy gears to
the old six (no gain).

## Improvements without a clear justification (to investigate later)

1. **AGFD mild crack in the crack class raises UNSW by ~10 points** (both seeds) but
   lowers MCC5 by ~5. The label is correct, but why it transfers to UNSW and costs MCC5
   is not understood; AGFD mild crack is at the detection limit within AGFD.
2. **Wear-as-healthy selection rule.** The cos < 0.5 rule was written down after seeing
   that AGFD mild wear (S) hurts (UNSW 69-73% with it, 92-99% without). The threshold
   and the mechanism (why a fault-aligned "healthy" unit damages training so much) need
   a proper study.
3. **Any extra class helped UNSW** (MCC5 broken tooth +8, wear as a separate class
   +11-15). Wear-as-healthy explains most of it, but more classes may also help the
   triplet loss through more negatives; untested.
4. **AGFD severe wear as healthy lowers MCC5 healthy recall** (round 5) for unknown
   reasons.
5. **Normalized inference** helped MCC5 by +7 points for one seed and +1 for the
   other, and changed UNSW by -2 to +6 depending on the variant. Justified by the
   training geometry, but the size of its effect is seed-dependent.

## Final results (commit 0c7f8bd + a9c04cc, two seeds, 3 ensembles of 3 each)

Official runs with `results.py` (MA, UA) and, after a directory collision between the
two parallel `results.py` calls (fixed in a9c04cc, see below), separately retrained UM.

| Training -> test | Seed 1000 | Seed 2000 | Per-class recall (seed 1000 / 2000) | Per-window | Oracle |
| :--- | ---: | ---: | :--- | ---: | ---: |
| MA -> UNSW healthy / crack | **99.3%** | **96.5%** | H 99 / 93, crack 100 / 100 | 93.5 / 91.8% | - |
| UA -> MCC5 healthy / crack / missing tooth | **77.8%** | **75.9%** | H 78 / 78, crack 58 / 50, MT 97 / 100 | 71.1 / 70.6% | 77.8% (crack 33) |
| UM -> AGFD healthy / missing tooth | **73.4%** | **74.1%** | H 72 / 72, MT 75 / 77 | - | 97.2% |
| UM -> AGFD healthy / mild crack (detection limit) | 60.3% | 62.3% | H 62 / 63, crack 59 / 62 | - | 83.9% |

Weights: `model_weights/MA_3`, `MA_4`, `UA_3`, `UA_4`, `ablations/final_UM_s1000`,
`ablations/final_UM_s2000`. `model_weights/UM_3` is corrupted (two seeds wrote into the
same run directories) and must not be used.

AGFD healthy recall per query gear (healthy / missing tooth): GP1 83/84, GP2 88/87,
GP3 86/84, GP4 87/86, GP6 45/54, GP7 42/36.

## What limits the results

- **UNSW:** essentially solved. Its healthy queries are mild crack (S) by design, so
  the score also reflects that convention.
- **MCC5:** at the in-domain ceiling. Crack does not transfer between MCC5 severities
  even for a classifier trained on MCC5 itself (crack recall 33%); missing tooth and
  healthy are near perfect.
- **AGFD:** 23 points below the oracle. Missing tooth transfers (75-77%), but
  UNSW + MCC5 training contains almost no variation between healthy units, so unseen
  healthy gears are often called faulty. Four of six test gears reach 83-88% healthy
  recall; GP6 and GP7 (whose healthy spectra point along the AGFD fault direction,
  cos 0.6-0.9) reach 36-54%. More healthy units in training (or the run-to-failure
  test) is the way forward, not further tuning.
- **Seed noise:** one 9-model training varies by about 2-8 points between seeds;
  report both seeds.
- Training on CPU with different thread counts is not bit-reproducible (official
  MA_3 99.3% vs the same configuration in the ablation runner 98.6%).

## Augmentation for UNSW + MCC5 training (UM -> AGFD only)

UNSW and MCC5 have one healthy unit each, so a model trained on them never sees how
healthy units differ. `add_unit_variation` (preprocessing_batch.py, off by default)
adds a random deviation to every training sample after the anchor difference, labels
unchanged. Only used for UM (training without AGFD); MA and UA are unchanged, so their
results above stand. Sizes were taken from MCC5 before testing (run-to-run variation
0.11-0.30, near-healthy units ~0.23-0.30 log10 RMS); per-sample RMS is drawn uniformly
from [0, 2 x rms].

- structured: random offset per harmonic block + smooth tilt over blocks (sensor path
  gain differences)
- white: independent noise per bin (control)

AGFD, normalized inference. "All gears" rotates healthy roles over 1, 2, 3, 4, 6, 7;
"GP1-4" over 1-4 only (GP5 is the missing tooth gear and can't be a healthy test gear).

| UM training | All gears, healthy / missing tooth | GP1-4, healthy / missing tooth | All gears, mild crack | GP1-4, mild crack |
| :--- | ---: | ---: | ---: | ---: |
| no augmentation, seed 1000 | 73.4% (H 72, MT 75) | 87.4% (H 87, MT 88) | 60.3% | 81.2% |
| no augmentation, seed 2000 | 74.1% (H 72, MT 76) | 88.3% (H 87, MT 89) | 62.3% | 84.0% |
| structured 0.15, seed 1000 | 66.6% | 73.8% | 64.1% | 77.5% |
| structured 0.25, seed 1000 | 69.8% | 75.1% | 62.2% | 72.6% |
| white 0.15, seed 1000 | 74.9% | 88.7% | 60.8% | 83.5% |
| **white 0.25, seed 1000** | **77.6%** (H 75, MT 80) | **90.1%** (H 88, MT 92) | 60.1% | 82.8% |
| **white 0.25, seed 2000** | **79.5%** (H 77, MT 82) | **90.2%** (H 88, MT 92) | 61.4% | 84.3% |
| oracle (AGFD labels) | 97.2% | 98.3% | 83.9% | 85.3% |

Healthy recall per query gear, all gears, white 0.25 (seed 1000 / 2000): GP1 89/91,
GP2 90/91, GP3 90/90, GP4 91/91, GP6 55/62, GP7 38/37 (no augmentation: GP6 45/54,
GP7 42/36).

Interpretation:
- White 0.25 improves healthy / missing tooth by +4 to +5 points on all gears and +2
  to +3 on GP1-4, consistently over seeds; mild crack is unchanged.
- The physically motivated structured deviation hurts: the level of whole harmonic
  blocks evidently carries fault information.
- GP1-4 reaches 90% against a 98% oracle; mild crack on GP1-4 (82-84%) is at its
  oracle (85%). GP7 does not improve.
- Caveat for the paper: white noise is a generic regularizer, so this result alone
  shows that invariance to small deviations helps, not specifically that healthy-gear
  variation is missing. Stronger evidence for "missing healthy units": UNSW rose from
  74% to 97-99% when wear units without a local-fault signature were added as extra
  healthy units (rounds 3-4), and the AGFD oracle, which sees eight healthy gears,
  reaches 97%.

Not adopted / to investigate: whether larger white noise helps further (0.25 > 0.15,
but going beyond the MCC5-based size would be tuning on the AGFD test).

## Own-gear anchors for AGFD faults (UM weights, seed 1000)

Question: does anchoring each fault on the healthy recording of its own gear (as in
deployment, where a gear is measured healthy after installation) work better?

Remount effect first (mean log10 RMS distance between mean spectra):

| Faulty unit | vs own healthy gear | vs other healthy gears | between its own remounts |
| :--- | ---: | ---: | ---: |
| missing tooth (GP5) | 0.53 | 0.48 | 0.29 |
| mild crack (GP9) | 0.34 | 0.31 | 0.34 |
| mild pitting (GP1) | 0.43 | 0.34 | 0.40 |
| severe pitting (GP6) | 0.34 | 0.37 | 0.34 |
| mild wear (GP7) | 0.31 | 0.37 | 0.31 |
| severe wear (GP2) | 0.29 | 0.32 | 0.30 |

Healthy, same recording (first vs last third): 0.12. Healthy, gear to gear: 0.37.
Remounting the same gear changes its spectrum as much as swapping gears, so the "gear
fingerprint" is mostly an installation fingerprint, and a faulty gear is no closer to
its own healthy recording than to other gears.

Evaluation (`scripts/agfd_own_anchor.py`): faults anchored on their own healthy gear,
prototype and queries from different installations; healthy anchor, support and query
are three different gears from 1, 2, 3, 4, 6, 7 (all ordered triples), so the healthy
side isn't trivial. Wear units (no local fault, also own-gear anchored) are queried as
extra healthy queries.

| UM training | Task | Accuracy | Healthy | Fault | Wear S called healthy | Wear L called healthy |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: |
| no augmentation | healthy / missing tooth | 69.9% | 69% | 71% | 69% | 71% |
| no augmentation | healthy / mild crack | 61.7% | 60% | 64% | 47% | 44% |
| white 0.25 | healthy / missing tooth | 69.5% | 68% | 72% | 70% | 74% |
| white 0.25 | healthy / mild crack | 62.3% | 56% | 68% | 44% | 42% |

Healthy recall per query gear (missing tooth task, no augmentation): GP1 74, GP2 76,
GP3 75, GP4 75, GP6 59, GP7 52.

- Own-gear anchors don't help: healthy / missing tooth drops from 73-78% (anchor =
  mean of other gears) to ~70%. A single-gear anchor carries its own installation
  fingerprint, and the own-gear anchor can't cancel the remount change.
- Against missing tooth, wear units are called healthy about as often as healthy gears
  (69-74% vs 68-69%), consistent with wear having no local fault.
- Against mild crack, more than half of the wear units are called crack: mild crack
  can't be told apart from a remounted gear without a local fault.
- AGFD can't evaluate baseline-referenced monitoring fairly because every fault was
  seeded by remounting. A run-to-failure dataset (no remount) can.

## Class balance with wear-as-healthy (one ensemble, seeds 1000-1002)

In v2 the sampler draws wear as its own query group, so label 0 (healthy) gets twice
the samples of each fault label in AGFD and MCC5 batches (100 vs 50; UNSW batches are
balanced). The v2 gain could therefore come from unit diversity or from healthy's
doubled weight in the loss. `balance_healthy_like: true` (sampler_ALL.py) makes
healthy-like classes take one of healthy's query groups instead: 50 per label, label 0
= 25 healthy + 25 wear.

Single-ensemble comparison (same seeds as the first ensemble of the v2 runs; normalized
inference):

| Training -> test | v0 (no wear) | v2 (wear, healthy doubled) | balanced (wear, healthy not doubled) |
| :--- | ---: | ---: | ---: |
| MA -> UNSW | 72.9% | 100.0% | **95.8%** (H 96, C 96) |
| UA -> MCC5 | 86.1% | 80.6% | 72.2% (H 67, C 50, MT 100) |
| UM -> AGFD healthy / missing tooth | 75.6% | 75.4% | 75.2% (H 72, MT 79) |
| UM -> AGFD mild crack | 60.3% | 59.9% | 59.3% |

One ensemble has 48 (UNSW) or 36 (MCC5) queries, so one query = 2-3 points; these are
indicative only.

- UNSW keeps most of the gain without the doubled healthy weight (95.8% vs 72.9%), so
  unit diversity, not class weighting, explains most of the v2 improvement.
- MCC5 is lower again (72.2% vs 80.6%), within single-ensemble noise but in the same
  direction as the earlier wear-as-healthy cost on MCC5.
- AGFD is unchanged.

## SupCon vs triplet (balanced batches)

### Implementation check

- `losses.SupConLoss` (pytorch-metric-learning 2.8.1) with `CosineSimilarity`:
  similarity / temperature, denominator over all other samples (positives and
  negatives, self excluded), mean over anchors with positives. No miner (correct).
- `intermediate_L2: true` L2-normalizes the pooled features before the final
  projection; cosine similarity makes the output length irrelevant.
- Inference reads `similarity: cosine` from run_info.yaml; `prototype_distances` uses
  1 - cosine on normalized embeddings; validation `predict` uses dot products of
  normalized embeddings.
- The training loop divides the (already mean-reduced) loss by the batch size. With
  AdamW this mostly cancels, but UNSW batches (100 samples) weigh ~1.5x AGFD/MCC5 batches
  (150). Same for both losses, left unchanged.
- Settings: temperature 0.1, cosine, `intermediate_L2: true` (the canonical SupCon
  settings in the config history), everything else as v2 + `balance_healthy_like`.

### How many batches? (pilot, one model per combination, 2501 batches)

Checkpoint chosen from training loss only: first checkpoint where the 100-batch moving
average of the training loss has completed 95% of its total decrease.

| Pilot | 90% of decrease | 95% of decrease | Fraction at checkpoint 5 / 10 |
| :--- | ---: | ---: | :--- |
| SupCon MA | 4 | 7 | 0.93 / 0.99 |
| SupCon UA | 7 | 10 | 0.87 / 0.96 |
| SupCon UM | 5 | 8 | 0.91 / 0.98 |
| triplet MA | 11 | 12 | 0.40 / 0.85 |

SupCon: 1001 batches, checkpoint 10 for all combinations (2501 not needed). Triplet
converges later: its canonical checkpoint 5 is at ~40% of the loss decrease, so triplet
is evaluated at both checkpoint 5 (canonical) and 10 (latest with 1001 batches).

### Results (two ensembles of three, seeds 1000-1005, balanced batches, normalized inference)

| Test | triplet ckpt 5 | triplet ckpt 10 | SupCon ckpt 10 |
| :--- | ---: | ---: | ---: |
| MA -> UNSW | **96.9%** (H 98, C 96; ens 95.8, 97.9) | 96.9% (H 94, C 100) | 88.5% (H 83, C 94; ens 87.5, 89.6) |
| UA -> MCC5 | 68.1% (H 62, C 46, MT 96; ens 72.2, 63.9) | **73.6%** (H 67, C 58, MT 96) | 72.2% (H 83, C 33, MT 100; ens 72.2, 72.2) |
| UM -> AGFD all gears, H / MT | 75.1% (ens 75.2, 75.0) | 73.1% | **76.6%** (H 74, MT 79; ens 77.7, 75.4) |
| UM -> AGFD GP1-4, H / MT | 88.2% | 88.1% | **90.5%** (H 90, MT 91) |
| UM -> AGFD all gears, mild crack | 60.7% | 60.2% | 61.4% |
| UM -> AGFD GP1-4, mild crack | 81.1% | **81.5%** | 80.3% |

Per-window (norm_win) follows the same pattern, 2-7 points lower.

- SupCon is not better overall: UNSW -8 points, MCC5 about equal (healthy recall up,
  crack recall down to 33%, which is the MCC5 in-domain oracle's crack recall), AGFD
  healthy / missing tooth +1.5 to +2.4 points.
- SupCon's two ensembles agree more closely (MCC5 72.2 / 72.2 vs triplet 72.2 / 63.9 at
  checkpoint 5), consistent with lower seed variance, but two ensembles are too few to
  claim it.
- Triplet checkpoint 5 vs 10: no consistent difference (MCC5 +5.5 at 10, AGFD -2 at
  10). Kept at the canonical 5; the convergence criterion would pick 12 for triplet,
  which is an open question.
- Balanced vs unbalanced (v2) triplet: UNSW similar, AGFD similar, MCC5 lower with
  balanced batches (68.1% vs 75.9-77.8%), in line with the single-ensemble test.

Decision: keep triplet. SupCon's AGFD gain is small and its UNSW loss large.

## V6 datasets (commit 48ba615): v2 baseline retrained, two seeds, 3 ensembles each

V6 has fewer, less overlapping windows (AGFD 65 per recording at every speed, MCC5
9-28, UNSW ~20); spectra agree with the old files (median correlation 0.96-0.98).
UNSW test anchor = first half of the H1 windows (was a fixed 16). The AGFD missing
tooth installation 1 = installation 2 duplication (500 rpm / 6 Nm, 1000 rpm / 11 Nm)
exists in both old and V6 files.

| Test (normalized inference) | V6 seed 1000 / 2000 | old data seed 1000 / 2000 |
| :--- | ---: | ---: |
| MA -> UNSW | 99.3 / 97.9% (H 96-100, C 99-100) | 99.3 / 96.5% |
| UA -> MCC5 | 63.9 / 67.6% (H 64, C 31-42, MT 97) | 77.8 / 75.9% |
| UM -> AGFD all gears, H / MT | 78.6 / 77.2% (H 67, MT 88-90) | 73.4 / 74.1% |
| UM -> AGFD GP1-4, H / MT | 88.3 / 88.0% (H 81, MT 95-96) | 87.4 / 88.3% |
| UM -> AGFD mild crack, all / GP1-4 | 59.3 / 81.4%, 58.9 / 80.8% | 60.3 / 81.2%, 62.3 / 84.0% |

AGFD healthy per query gear (all gears, seed 1000): GP1 85, GP2 89, GP3 86, GP4 87,
GP6 32, GP7 24. Per-window accuracy is 1-3 points below the averaged numbers.

- AGFD gains through missing tooth recall (75-77 -> 88-90%); GP6/GP7 healthy recall
  drops.
- MCC5 drops ~11 points. Old UA weights score 77.8% on the V6 MCC5 test, so the cause
  is training on V6 UNSW + AGFD, not the V6 test data. Open.

## Sensors (V6, normalized inference)

### Test sensor only (V6 baseline weights, 3 ensembles; seed 1000 / 2000)

| Test | acc4 / x (baseline) | acc3 / y | z | all sensors (each with its own anchor, distances averaged) |
| :--- | ---: | ---: | ---: | ---: |
| UM -> AGFD all gears, H / MT | 78.6 / 77.2% | acc3 83.5 / 83.1% | - | 82.4 / 82.1% |
| UM -> AGFD GP1-4, H / MT | 88.3 / 88.0% | acc3 85.2 / 85.2% | - | 87.6 / 87.6% |
| UM -> AGFD GP1-4, mild crack | 81.4 / 80.8% | acc3 67.7 / 68.1% | - | 78.4 / 78.2% |
| UA -> MCC5 | 63.9 / 67.6% | y 65.7 / 67.6% | 74.1 / 72.2% | 68.5 / 68.5% |

acc3 moves healthy recall between gears (GP6 32 -> 60-64%, GP1 85 -> 52%; GP1 only 30%
on GP1-4), so sensor-specific effects are partly gear-specific. Picking the best test
sensor would be selection on the test set; averaging all sensors needs no selection.

### Training sensors (seed 1000, 2 ensembles; baseline trains on acc4 / x)

| Training sensors | MA -> UNSW | UA -> MCC5, test x / all | UM -> AGFD all gears, test acc4 / both | UM -> AGFD GP1-4, test acc4 / both |
| :--- | ---: | ---: | ---: | ---: |
| baseline | 100.0% | 63.9 / 68.5% (3 ens) | 78.5 / 81.7% | 88.2 / 87.6% (3 ens) |
| AGFD acc3 | 92.7% | 72.2 / 70.8% | - | - |
| MCC5 y | 83.3% | - | 75.8 / 81.5% | 88.3 / 87.7% |
| MCC5 z | 97.9% | - | 78.3 / 82.8% | 88.0 / 87.4% |
| all (AGFD acc3+acc4, MCC5 x+y+z) | 96.9% | 69.4 / 73.6% | 75.8 / 81.1% | 86.9 / 86.4% |

- Training sensor matters for MCC5 (AGFD acc3 or all sensors: +5 to +9 points).
- Test-time sensor averaging helps AGFD (+3-4) regardless of training sensors.
- "All sensors in training and testing" is close to the best per dataset with one rule
  and no test-set selection. Seed 2000 replicate below.

## Round C (V6, multi-sensor test, seed 1000, 2 ensembles)

Baseline (same seeds, multi-sensor test): UNSW 100.0%, MCC5 68.1%, AGFD all gears
81.7%, GP1-4 87.5%.

| Variant | Result |
| :--- | :--- |
| AGFD duplicated installation pairs excluded (3 ens) | 82.3% vs 82.4%: negligible |
| embedding_len 8 | UNSW 100.0%, MCC5 72.2%, AGFD 80.9% / 86.9% (within noise) |
| unit-variation augmentation (UM, white 0.25) | AGFD 84.3% / 88.2% (+2.6 / +0.7) |
| UA trained on old UNSW (full) + V6 AGFD | MCC5 79.2% |
| UA trained on old UNSW subsampled to ~20 windows per recording + V6 AGFD | MCC5 70.8% |
| UA trained on V6 UNSW + old AGFD | MCC5 70.8% |

The MCC5 drop on V6 comes from the number of UNSW training windows: subsampling old
UNSW to V6's ~20 windows per recording reproduces it. Possible mechanism: training
anchors average 20 windows drawn from ~20, so every anchor is almost the recording mean
and the model sees no anchor-to-anchor variation. Round D tests 10 anchors.

## Round D (V6, multi-sensor test, 2 ensembles; seed 1000 / 2000)

Two-seed baseline: UNSW 100.0 / 99.0%, MCC5 68.1 / 65.3%, AGFD all gears 81.7 / 81.3%,
GP1-4 87.5 / 87.5%.

| Variant | UNSW | MCC5 | AGFD all gears | AGFD GP1-4 |
| :--- | ---: | ---: | ---: | ---: |
| embedding_len 8 | 100.0 / 97.9% | **72.2 / 70.8%** | 80.9 / 81.9% | 86.9 / - |
| ALL_train_num_anchors 10 (seed 1000) | 97.9% | 72.2% | 82.0% | - |
| unit-variation augmentation (UM) | - | - | **84.3 / 84.9%** | 88.2 / 88.3% |
| UM trained on old UNSW | - | - | 74.4% | - |

- 8-D embedding: MCC5 +4 to +5 for both seeds, nothing lost elsewhere. A priori reason:
  with a 2-D L2-normalized embedding all variation (including unit-to-unit variation)
  must fit into one angle.
- Old UNSW helps MCC5 (79.2%) but costs AGFD 7 points, so it is no fix; V6 stays.
- Augmentation replicates on V6 (+2.6 / +3.7 all gears, +0.7 / +0.8 GP1-4).

## Round E and decision (V6, multi-sensor test, 2 ensembles)

Two-seed means (seed 1000 / 2000 where given):

| Variant | UNSW | MCC5 | AGFD all gears | AGFD GP1-4 |
| :--- | ---: | ---: | ---: | ---: |
| baseline | 99.5% | 66.7% | 81.5% | 87.5% |
| 10 training anchors | 97.9% (97.9 / 97.9) | **74.3%** (72.2 / 76.4) | 82.2% (82.0 / 82.4) | 86.2% (s1000) |
| embedding_len 8 | 99.0% | 71.5% | 81.4% | 86.9% (s1000) |
| 8-D + 10 anchors (s1000) | 99.0% | 73.6% | 81.4% | 87.0% |
| 8-D + augmentation, UM (s1000) | - | - | 84.3% | 87.6% |

Decision (commit 81afead): `ALL_train_num_anchors: 10`. Reason independent of the scores:
V6 has ~20 windows per UNSW recording, so averaging 20 random anchors made every
training anchor almost the recording mean; 10 also matches the test anchor sizes better
(UNSW 10, MCC5 7). The 8-D embedding is not adopted: smaller MCC5 gain, nothing added on
top of 10 anchors, and it gives up the 2-D visualisation.

Test-time sensors (commit d1d942a): all sensors vote (AGFD acc3 + acc4, MCC5 gearbox
x + y + z). Training sensors stay single (AGFD acc4, MCC5 x): training on all sensors
traded ~6 UNSW points for ~4 MCC5 and ~1 AGFD.

## Oracle ceilings on V6 (`scripts/oracle_v6.py`)

| Test | Sensor | Oracle |
| :--- | :--- | ---: |
| MCC5 healthy / crack / missing tooth | x / y / z | 80.6 / 80.6 / 83.3% (H 100, crack 42-50, MT 100) |
| AGFD all gears, H / MT | acc4 / acc3 | 98.5 / 99.4% |
| AGFD GP1-4, H / MT | acc4 / acc3 | 99.4 / 100.0% |
| AGFD all gears, mild crack | acc4 / acc3 | 84.6 / 72.0% |
| AGFD GP1-4, mild crack | acc4 / acc3 | 86.4 / 77.8% |

## Final official results (commit 81afead, `results.py`, 3 ensembles of 3 per seed)

| Training -> test | Seed 1000 | Seed 2000 | Oracle (V6) |
| :--- | ---: | ---: | ---: |
| MA -> UNSW | 97.9% | 96.5% | - |
| UA -> MCC5 | 70.4% | 69.4% | 80.6-83.3% |
| UM -> AGFD all gears, H / MT | 82.9% (H 72, MT 94) | 84.0% (H 72, MT 96) | 98.5-99.4% |
| UM + unit-variation augmentation -> AGFD, H / MT | 83.9% (H 74, MT 94) | 85.5% (H 75, MT 96) | |
| UM (+aug) -> AGFD all gears, mild crack | 58.9-60.0% | | 72.0-84.6% |

Weights: `model_weights/UM_4`, `UM_5` (UM), `UM_6`, `UM_7` (UM + augmentation),
`UA_5`, `UA_6`, `MA_5`, `MA_6`. CSVs and logs: `results/local_fault/final_v6_*`.

AGFD healthy recall per gear (baseline, both seeds): GP1 74, GP2 94-95, GP3 92-94,
GP4 93-96, GP6 47-49, GP7 27-29. With 10 anchors the augmentation's gain is smaller
(+1.0 / +1.6) than with 20 anchors.

### GP1-4 and per-window on the official weights

| UM training | GP1-4 H / MT (seed 1000 / 2000) | GP1-4 mild crack | per-window H / MT (all gears not run) |
| :--- | ---: | ---: | :--- |
| baseline (UM_4, UM_5) | 87.3 / 88.1% (H 77-78, MT 97-98) | 77.4 / 78.1% | GP1-4: 85.8 / 86.8% |
| + augmentation (UM_6, UM_7) | 88.5 / 88.4% (H 79-80, MT 97-98) | 78.1 / 78.0% | GP1-4: 86.9 / 87.3% |

GP1-4 healthy recall per gear: GP1 38-46%, GP2 99-100%, GP3 80-82%, GP4 90-92%.
Per-window: UNSW 95.8 / 94.1%, MCC5 69.4 / 69.9%.

## Checkpoint sensitivity (seed 1000, 2 ensembles, max_batches 1301, same weights)

| Test | checkpoint 5 (used) | checkpoint 12 (training-loss rule) |
| :--- | ---: | ---: |
| MA -> UNSW | 99.0% | 99.0% |
| UA -> MCC5 | 72.2% | 76.4% |
| UM -> AGFD all gears, H / MT | 82.1% | 79.9% |

The two checkpoints move MCC5 and AGFD in opposite directions by 2-4 points and are
equal overall, so checkpoint 5 (fixed in advance) stays; report this as robustness.

## Regenerated UNSW_V6 with ~80 windows per recording (2026-10-22)

Data check: 7,865 rows, signal OT only, 62-92 windows per recording (balanced over
severities), 270 orders (selected bins need up to 229), values finite and positive,
no duplicate spectra, `freqs` = 0..269. Agreement with the old UNSW file: median
correlation of mean log spectra 0.964 (worst only at speed 2, unused), mesh-frequency
level unchanged. Consecutive-window / random-pair distance 0.50 (old 0.44, 20-window V6
0.67); within-recording spread 0.130 (old 0.152); distance to H: S 0.35, M 0.40, L 0.42.

Retrained, multi-sensor test, 2 ensembles (seed 1000 / 2000):

| Training -> test | UNSW ~80 windows, 10 anchors | UNSW ~80 windows, 20 anchors | UNSW ~20 windows, 10 anchors (official, first 2 ens) | UNSW ~20 windows, 20 anchors |
| :--- | ---: | ---: | ---: | ---: |
| UA -> MCC5 | **79.2 / 70.8%** (mean 75.0) | 70.8 / 75.0% (72.9) | 70.8 / 66.7% (68.8) | 68.1 / 65.3% (66.7) |
| MA -> UNSW | 97.9 / 94.8% | - | 99.0 / 96.9% | - |
| UM -> AGFD all gears, H / MT | 81.7 / 83.7% | - | 82.0 / 83.1% | - |
| UM -> AGFD GP1-4, H / MT | 86.9 / 87.5% | - | 87.3 / 88.1% (3 ens) | - |

- The ~80-window UNSW file recovers most of the MCC5 drop (+6 points over the
  20-window file, both anchor settings); the old file with all windows gave 79.2%.
- Unlike the old UNSW file (AGFD -7), it doesn't hurt AGFD. UNSW is within noise
  (-1 to -2 points, one query = 2.1 points).
- With ~80 windows, 10 vs 20 training anchors makes no clear difference (75.0 vs
  72.9%). 10 stays (it also matches the test anchor sizes better).

## Spread-scaled prototype distances (test-time, no retraining)

`norm_scaled` (scripts/diag_eval.py): as norm_mean, but each query-prototype distance is
divided by the mean distance of that prototype's normalized support windows to it
(relative distance). Same rule for all datasets, only support data used.

| Weights -> test | norm_mean (test.py) | norm_scaled |
| :--- | ---: | ---: |
| UA_5 -> MCC5 | 70.4% (H 69, C 42, MT 100) | 63.9% (H 83, C 25, MT 83) |
| UA_6 -> MCC5 | 69.4% (H 69, C 39, MT 100) | 63.9% (H 89, C 25, MT 78) |
| UA ~80-window UNSW, s1000 -> MCC5 | 79.2% (H 79, C 58, MT 100) | 62.5% (H 83, C 21, MT 83) |
| UA ~80-window UNSW, s2000 -> MCC5 | 70.8% (H 71, C 42, MT 100) | 66.7% (H 92, C 25, MT 83) |
| MA_5 / MA_6 -> UNSW (now the ~80-window file) | 97.2 / 95.8% | 92.4 / 88.9% |
| UM_4 / UM_5 -> AGFD all gears, H / MT | 82.9 / 84.0% | 77.2 / 79.2% |
| UM_4 / UM_5 -> AGFD GP1-4, H / MT | 87.3 / 88.1% | 82.8 / 83.9% |

Healthy recall rises on MCC5 (+10 to +21) and partly on AGFD, but crack and missing
tooth recall fall more (MCC5 crack 21-25%, missing tooth 78-83%), so accuracy drops on
every test by 4-17 points. The healthy support windows spread more than the fault
windows, so scaling pulls every query toward healthy. Not adopted.

## Unit-variation augmentation for UA (~80-window UNSW, 10 anchors, 2 ensembles)

| UA training | Seed 1000 | Seed 2000 | Mean healthy / crack recall |
| :--- | ---: | ---: | :--- |
| no augmentation | 79.2% (H 79, C 58, MT 100) | 70.8% (H 71, C 42, MT 100) | 75 / 50% |
| white noise 0.25 | 72.2% (H 79, C 38, MT 100) | 75.0% (H 92, C 33, MT 100) | 85.5 / 35.5% |

Confusion with augmentation: healthy queries called crack 21% / 8%; crack queries
called healthy 54% / 58%. Crack recall stays lowest at 2000 rpm (12%) and for severe ->
medium (25%).

The augmentation moves the healthy/crack boundary toward crack: healthy recall +10 on
average (one seed +21, the other +0), crack -15, accuracy about equal (73.6 vs 75.0%).
Same trade-off as the spread-scaled distances, milder. MCC5 crack and run-to-run
healthy variation are the same size, so improving healthy recall costs crack recall.
Not adopted for UA (it is also only justified for training without healthy-unit
variation).

## Selected bins vs full spectrum: is crack information dropped? (`scripts/order_information.py`)

The selection keeps 8 harmonics with the 5 nearest and 5 furthest sidebands per side.
Dropped: low shaft orders (MCC5 1-18, UNSW 1-13, AGFD 1-7), middle sidebands between
harmonics (MCC5 offsets 6-12, 119 orders; UNSW offsets 6-8, 48 orders; none in AGFD,
where 15 teeth make the blocks overlap: 168 bins = 120 unique orders), and everything
above harmonic 8 (MCC5 54, UNSW 40, AGFD 22 orders).

Per-order AUC (healthy vs crack), direction fixed on the training side of the protocol,
measured on the held-out side (MCC5: M <-> L, torque -> speed run; UNSW: M <-> L vs H
halves; AGFD: mild crack, installations 1-2 / gears 1-3 -> installation 3 / gears 4, 6,
7):

| Data | Median AUC selected / dropped | Orders with AUC >= 0.75, selected / dropped | Best dropped orders |
| :--- | :--- | :--- | :--- |
| MCC5 x | 0.53 / 0.49 | 9 of 168 / 3 of 191 | 322 (h9-2, 0.82), 344 (h10-16, 0.78), 96 (h3-12, 0.78) |
| MCC5 z | 0.54 / 0.50 | 13 / 5 | 96 (h3-12, 0.80), 240 (h7-12), 60 (h2-12) |
| UNSW | 0.66 / 0.65 | 53 of 168 / 24 of 101 | 255, 258 (h9+12, h10-12, 0.89), 247, 264, 33 (h1+6) |
| AGFD acc4 / acc3, mild crack | 0.55 / 0.48; 0.56 / 0.55 | 11 / 1; 13 / 2 | 144 (h10-6, 0.78-0.80), 129 (h9-6) |

The most informative orders are almost all selected (UNSW and AGFD: all of the top 8-10;
MCC5: 9 of the top 12). Dropped orders with information sit mostly above harmonic 8
(harmonics 9-10) and, in MCC5, at offsets of 12 orders (24, 48, 60, 96, 240, ...),
which repeats every 12 orders and may belong to another component of the gearbox rather
than the cracked gear.

Oracle with different feature sets (logistic regression, same hold-outs):

| Feature set | MCC5 x | MCC5 z | UNSW | AGFD acc4 mild crack | AGFD acc3 mild crack |
| :--- | ---: | ---: | ---: | ---: | ---: |
| selected (current) | **80.6%** (C 42) | **83.3%** (C 50) | 97.9% | **84.8%** | 71.7% |
| full spectrum | 75.0% (C 25) | 75.0% (C 25) | 95.8% | 81.3% | 74.4% |
| dropped orders only | 72.2% (C 17) | 69.4% (C 8) | 91.7% | 75.9% | 75.4% |
| selected + low orders | 80.6% | 83.3% | 100.0% | 84.6% | 71.5% |
| selected + orders above harmonic 8 | 77.8% | 80.6% | 97.9% | 81.9% | 74.6% |

- No crack information that the selection loses: adding the dropped orders lowers the
  oracle on MCC5 (-5 to -8), UNSW (-2) and AGFD acc4 (-3). The selection works as
  denoising. Only AGFD acc3 gains from the full spectrum (+3) and UNSW from the low
  orders (+2, one query).
- The dropped orders still carry some information on their own (UNSW 92%, AGFD 76%):
  crack information is spread over many orders, as expected from a localized fault.
- The earlier old-data benchmark found the full spectrum slightly better for MCC5
  under a different protocol (fault-type classification on a held-out severity); on V6
  with the test protocol, the selection is better.

## UA trained on AGFD acc3 + acc4 (~80-window UNSW, 10 anchors, 2 ensembles)

| UA training | Seed 1000 | Seed 2000 | Mean |
| :--- | ---: | ---: | ---: |
| AGFD acc4 (current) | 79.2% (H 79, C 58, MT 100) | 70.8% (H 71, C 42, MT 100) | 75.0% |
| AGFD acc3 + acc4 | 69.4% (H 71, C 38, MT 100) | 70.8% (H 71, C 42, MT 100) | 70.1% |
| oracle (MCC5 labels, x / z) | | | 80.6 / 83.3% (H 100, C 42-50) |

No gain: healthy recall stays at 71% (both seeds), crack 38-42%. The earlier +4 with
both AGFD sensors (20-window UNSW, 20 anchors) doesn't hold in the current setup.
Healthy queries called crack 29%; crack weakest at 2000 rpm (12%).

## Training length and test sensor default (2026-10-22)

- `max_batches: 501`: checkpoint 5 is saved at batch 500 and is bit-identical to the
  1001-batch runs (same seed and threads: max weight difference 0.0), so all results
  above still hold.
- Test sensors: default is now the training sensor (AGFD acc4, MCC5 gearbox x);
  `test.py --all-sensors` / `results.py --all-sensors` restores all-sensor voting.
  UA_5 -> MCC5: 74.1% with gearbox x only, 70.4% with all sensors (the official result).

## Code cleanup: standalone samplers and training-time test loop (2026-10-22)

- `sampler_AGFD.py`, `sampler_MCC5.py`, `sampler_UNSW.py` now only hold the `*_Dataset`
  classes used by `sampler_ALL.py`; the standalone/episodic samplers, `get_*_data` and
  the episodic collate functions are gone. `data.py` only accepts `ALL_<letters>`.
- The test loop during training is gone. Validation stays (monitoring only): it uses an
  `ALL_` dataset from `validation_datasets` (default `ALL_U`, `<dataset>_validation_*`
  entries in the config for all three datasets) and logs loss, precision at 1,
  R-precision and AMI within each batch.
- Validation draws from its own random generator (`[seed, 1]`), so the validation setup
  no longer changes the training samples of a seed. This, and removing the test loader,
  changes the training random stream compared with all runs above: models trained from
  now on differ from earlier ones with the same seed, so the official results have to be
  rerun.

## Code cleanup: models and losses (2026-10-22)

Removed the prototypical and classical models, every loss except triplet (NCA,
contrastive, circle, SupCon, proto_loss), the episodic `Embedding.predict`, the SupCon-only
`intermediate_L2` option of the MLP-Mixer, and `src/training/utils.py`. The MLP backbone
stays for the backbone comparison; its `input_size` now defaults to the number of
selected bins (168), e.g. `backbone: MLP`, `model_args: {hidden_sizes: [64, 32]}` (12,962
parameters vs 14,632 for the MLP-Mixer). A model trained before and after the cleanup with
the same seed is bit-identical (same parameter names, max weight difference 0.0), so
existing triplet weights still load.
