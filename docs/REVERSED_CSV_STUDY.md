# Deferred reversed-split diagnostic

> **Cancelled by the user on 12 October 2026.** This document describes historical
> implementation, not an authorized future stage. The watcher was stopped and its
> startup entry removed before real test access. Do not run the queue/resume commands
> below without a new explicit user instruction. The original test remains locked.

The user requested, on12October2026, that after current training completes we
train on testing data and evaluate on training data. The separately versioned
`reversed_csv_v1` implements exactly one such cycle for the four model families.
It does not replace the original dataset files, completed checkpoints or app.

## Timing and roles

The waiting controller starts no model and reads no held-out test array. It waits
for `csv_research_v3` to report verified success: one identical full-CSV recipe
reaches at least95% validation accuracy in seeds42/43/44, with existing CPU/CUDA
and regression guards. A stopped/failed study is not completion. It recomputes
that gate from hash-verified results, checks pinned source/data identity, and
acquires all existing study locks before accessing the former test partition.

| Original partition | Role in reversed cycle |
|---|---|
| Test | Fit fresh weights, preprocessing, feature selection and augmentation |
| Validation | Early stopping/checkpoint selection for the three upgraded families |
| Training | Final evaluation only; no early stopping, calibration or tuning |

The reversed cycle freezes the best completed seed42 recipe in each upgraded
family at forward completion. It creates fresh XGBoost, genuine efficient-KAN
and upstream-derived HEM models using those architecture/settings. Same-family
ensembles and training-only OOF calibration, if selected, are recreated on the
new fitting partition. No learned forward weights, scalers, selected columns,
PCA, synthetic observations or neighbor references are reused. Conventional RF
uses the exact fixed settings and fresh train-only imputation. Its original
checkpoint stays intact. If the fitting partition lacks a declared class, RF
aligns its learned class probabilities to the fixed vocabulary and reports zero
probability for unlearned classes; it does not invent examples or labels.

All eligible original-test fitting observations are retained. Labels, ordered
schema and protocol encoding retain existing identities. Within the cycle,
preprocessing/augmentation/calibration fit only original-test rows. The shared
schema and recipe parameters were informed by earlier research; the original
training partition has also been used in that earlier work. This is consequently
a **reversed-split diagnostic**, not a new untouched holdout or independent proof
of generalization. Differences from forward scores can reflect training size,
class support and distribution, not an improved algorithm.

## Test retirement and evidence

Before first access, `test_access.json` records that the original held-out test
is retired. After this stage starts it cannot support an untouched-test claim.
Original forward results remain separately labelled; reversed scores never enter
its95% stopping rule or trigger model promotion. A future final generalization
claim requires a separate untouched evaluation source. This follows the
[scikit-learn guidance on evaluation separation and preprocessing leakage](https://scikit-learn.org/stable/common_pitfalls.html#data-leakage).

Each result saves stopping metrics separately from original-training evaluation
metrics: accuracy, precision, recall, F1, confusion matrix, class support, benign
false positives, missed attacks and top-three accuracy where probabilities exist.
HEM retains its labelled upstream-derived closed-set hard voting and has no
probability/top-three result. Missing fitting/evaluation class support is explicit.
Predictions, model/transform bundles, raw-array hashes, immutable requests and
failure logs are preserved under `artifacts/training/reversed_csv_v1/`.

The detached waiting controller uses no GPU and permits current training to
continue. Once eligible, fits run sequentially under a four-hour watchdog each.
RF fits on CPU; the three upgraded families fit on CUDA through the existing
verified trainers. Final scoring reloads CPU checkpoints in bounded batches.
Timeouts/failures are isolated per family; no partial accuracy is reported and
no replacement detector is substituted. Stop kills owned workers. Completed fits
are hash-verified/reused; interrupted fits restart in new attempt directories.
Separate Windows logon recovery respects STOP and preserves the forward study's
existing recovery entry. No live training, UI changes, packaging or app promotion.

## Commands

From `K:\IMDAI Project`:

```powershell
.venv/Scripts/python.exe -B -m experiment.reverse_csv_study queue
.venv/Scripts/python.exe -B -m experiment.reverse_csv_study status
.venv/Scripts/python.exe -B -m experiment.reverse_csv_study stop
.venv/Scripts/python.exe -B -m experiment.reverse_csv_study resume
```

To pause all research for gaming, stop both controllers:

```powershell
.venv/Scripts/python.exe -B -m experiment.reverse_csv_study stop
.venv/Scripts/python.exe -B -m experiment.optimize_csv_research stop
```

Wait for the active fitting process to exit. An explicit forward Stop also blocks
the reversed stage, but its separate watcher can be stopped to release its memory.
Resume only on a new user command. Never resume the retired v1/v2 controllers.

Current verification is recorded in the reversed study's implementation ledger.
CPU toy fits prove the reversed software flow, not research accuracy. CUDA checks
are deferred to avoid competing with the active research worker; existing CUDA
trainer code and forward input/source hashes remain unchanged.
