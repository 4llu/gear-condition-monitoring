# Plan: run-to-failure experiment (train on UMA, test on run-to-failure)

## Why this experiment

It answers the two questions the seeded datasets can't:

1. **Deployment scenario.** A gear is measured healthy after installation and then
   monitored without remounting. On AGFD every fault involved a remount, which changes
   the spectrum as much as swapping gears, so own-gear baselines couldn't be evaluated
   fairly. In a run-to-failure run the baseline and the fault share one installation.
2. **Natural vs seeded faults.** Does a model trained on seeded local faults recognise a
   naturally developing failure, and how early?

It also tests the main claim of the thesis directly: training with variation between
healthy units (now available, since AGFD's eight healthy gears are in UMA) is what
limits generalisation.

## 0. Data checks before anything else

- Failure mode of each run at teardown (crack, tooth breakage/missing tooth, pitting or
  spalling, combination). Decides which training classes make sense (section 2).
- Which gear failed, its tooth count, the shaft it sits on, and whether a tachometer or
  encoder signal exists for order tracking on that shaft.
- Operating conditions over each run (constant or varying speed/load).
- Number of runs, length, timestamps of the ultrasound-verified healthy measurements and
  of the RMS alarm / rig stop.
- Run-in period at the start (spectra usually change during the first hours).

## 1. Preprocessing

Same pipeline as the seeded datasets: order tracking on the failed gear's shaft, TSA
spectrum over the same number of revolutions (30-40), selected bins (8 harmonics,
block size 5) for the failed gear's tooth count, log10. One "unit" = one timestamp
(recording); all windows of a timestamp are embedded and batch-averaged as in test.py,
with per-window results as a secondary metric.

## 2. Training (UMA, test-only on run-to-failure)

Use the current pipeline (classes healthy / crack / missing tooth, healthy = no
localized tooth fault):

- healthy: all healthy units, all 8 AGFD gears, plus the wear units without a
  local-fault signature (AGFD severe wear, MCC5 mild wear)
- crack: UNSW M/L, MCC5 M/L, AGFD mild crack
- missing tooth: AGFD GP5, MCC5 missing tooth
- no unit-variation augmentation: it is only justified for training without
  healthy-unit variation, and UMA has AGFD's eight healthy gears
- normalized inference, 3 ensembles of 3, two seeds, checkpoint 5

Class choice depends on the teardown result:

- **Tooth breakage / crack (most likely for "broken"):** the core setup above fits.
  Missing tooth is the most consistent signature across datasets (cos 0.80), crack has
  UNSW as its consistent source.
- **Pitting / spalling:** add pitting as a class only as a secondary variant (MCC5 S/M/L,
  AGFD S/L). Pitting signatures disagree between AGFD and MCC5, so expect it to act as a
  generic "fault" direction rather than a reliable pitting class.
- Broken tooth (MCC5) stays out: its signature is tiny and doesn't resemble missing tooth.

Comparison models (same test, to support the thesis):

| Model | Purpose |
| :--- | :--- |
| UMA, core setup | main result |
| UMA without wear-as-healthy | effect of extra healthy units |
| UMA with AGFD restricted to 2 healthy gears | effect of healthy-gear variation directly |
| UM, with and without unit-variation augmentation | training without healthy-gear variation |
| UMA + pitting class | only if teardown shows pitting/spalling |

## 3. Test protocol (no information from the future of the tested run)

Anchor and healthy prototype come from the run's own early, ultrasound-verified healthy
timestamps (after run-in), never from the same timestamp:

- anchor: earliest verified-healthy timestamp after run-in
- healthy prototype: a later verified-healthy timestamp (the gap imitates slow change)
- remaining verified-healthy timestamps: healthy queries (false-alarm check)

Two complementary evaluations:

**A. Health indicator (needs no fault data from the target):** the distance of each
timestamp's embedding to the healthy prototype, over the whole run.

- Alarm threshold fixed in advance from the verified-healthy timestamps only
  (e.g. mean + 3 std of their distances, or the maximum), the factor chosen before
  looking at the rest of the run.
- Metrics: false alarms over the verified-healthy period, first persistent alarm time,
  lead time against the RMS alarm / rig stop, monotonicity (Spearman with time).

**B. Few-shot diagnosis with leave-one-run-out:** fault prototypes from the final
(verified broken) timestamps of the *other* runs, healthy prototype from the tested
run's own early data. Classify every timestamp of the tested run.

- Metrics: recall of the end state, false alarms in the healthy period, time at which
  the run is first classified as faulty, and which fault prototype the run moves toward.
- Optional: source-domain class prototypes (crack, missing tooth from UMA) to see
  whether the late-stage data points toward the expected fault class.

## 4. Baselines

- RMS of the raw signal (what stopped the rig) and its alarm time
- Classical TSA-residual indicators (FM4/kurtosis, NA4)
- Distance in the input space (log-spectrum difference to the anchor, no model):
  shows what the learned embedding adds

## 5. What would support the thesis

- Earlier, stable detection than RMS with few false alarms in the verified-healthy period.
- UMA (with healthy-gear variation) clearly better than UM and than UMA restricted to
  two healthy gears, especially in false alarms. That's direct evidence that healthy-unit
  variation in training is what limits generalisation.
- If the end state is tooth breakage, late-stage timestamps approaching the missing
  tooth prototype would link the seeded and natural fault results.

## 6. Risks

- Few runs: report every run separately, no pooled tuning.
- Run-in, temperature drift and changing operating conditions can look like faults:
  check the healthy period's trend, and use only matching conditions if they vary.
- Unknown fault onset between the last verified-healthy time and the stop: only the end
  state has a fault label, so detection time is judged against the RMS alarm, not a true
  onset.
- If the failed gear's shaft has no speed reference, order tracking must be estimated
  from the signal itself; check its quality before trusting spectra.
