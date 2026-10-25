# Dataset-specific TabM study

Status: implementation and execution requested on 16 October 2026. Results are
measured artifacts, never implied by this specification.

Models: `tabm_ne` (flat 84-class numerical-embedding model), `tabm_hier`
(shared backbone, binary and 83-type heads), and `tabm_hier_ne` (both changes).
Use the official Apache-2.0 `tabm==0.0.3` and MIT
`rtdl-num-embeddings==0.0.12` packages; our hierarchy and data handling are
adaptations. Sources: https://github.com/yandex-research/tabm and
https://github.com/yandex-research/rtdl-num-embeddings.

## Dataset compatibility

Only verified cleaned DataSense v2 train/validation files are permitted.
The complete ordered 655-feature schema and 84 labels are preserved. Fit all
preprocessing on training alone. Unknown indicators remain distinguishable from
absence through explicit missing masks. Numerical values retain sign; compare
signed-log robust scaling against train-fitted quantile-normal scaling.
Numerical embeddings use train-fitted quantile boundaries, version B, no
activation. Binary/presence/quality features bypass embeddings. Constants and
all-missing columns bypass embeddings but retain their input and mask positions.
For numerical columns with a majority of zeros, reserve zero and fit remaining
edges to nonzero training values; this prevents rare-event bins from collapsing.
Check incoming feature names/order, dimensions, finite output and label range.
No raw datasets are overwritten, no labels invented and no test arrays read.

## Training and selection

The initial matched controls use k=16, three blocks, width256, dropout0.1,
AdamW lr0.002 and weight decay0.0003; embedding dimension12 and16 bins.
A frozen search varies scaling, capacity, learning rate, regularization,
embedding resolution, class weights and full/strict training arms in named
recipes. Selected recipes repeat with seeds43/44. Every real fit uses all rows
of its declared arm; small smoke fixtures are explicitly software tests.
Early stopping and recipe/seed choice use only frozen tuning-role metrics:
accuracy primary, macro F1 tie-break; binary regressions are reported.
Search never chooses against calibration or assessment performance.

Per-member losses are averaged independently. Hierarchical loss is BCE plus
true-attack-masked subtype CE, averaged over all rows/members. Starting weight1
is the joint hierarchical negative log-likelihood. Combine each member's
attack and subtype probabilities before ensemble averaging.

## Calibration and assessment

Reuse the existing purged temporal tuning/calibration/assessment roles.
After recipe selection, fit scalar temperature on calibration rows to the
composed84-class probabilities, preserving argmax. Fit a separate binary
Platt calibrator for attack probability. Select the binary alert threshold on
tuning rows under a declared false-alarm cap; subtype uncertainty never changes
an attack alert into benign. Assess calibrators only on the final assessment
role and report improvements and regressions. Prior validation reuse makes
these exploratory results, not an independent final benchmark.

Report full-vocabulary accuracy, recall/F1, per-class support, false alarms,
misses, probability NLL/Brier/ECE, threshold decisions and subtype accuracy on
true attacks. Keep uncalibrated and calibrated scores separate. A target is not
a guarantee and calibration does not increase unchanged argmax accuracy.

## Execution and evidence

Default execution is CPU/offline/credential-free; CUDA is an explicit study
option. Share the existing GPU controller locks and never stop other studies.
Use bounded fits, persistent per-epoch telemetry, distinct attempts, explicit
failures, stop/resume, data/source hashes, CPU checkpoint replay and calibration
reload checks. No model promotion, live capture or installed-app changes.
No third-party service is needed; Plugin Management recommends existing local
capabilities for this task.

Run the default demo, all four sample options and the entire evaluation suite
after implementation. Record actual execution separately from this design.

The first search has eight named recipes per model (24 fits), followed by
seeds43/44 for each model's selected recipe (six fits), totaling30 planned fits.
Per-fit training is bounded to3600 seconds plus preprocessing, CPU replay and
reporting; the worker watchdog is7200 seconds. Failures remain recorded and
never count as successful fits. There is no promise that the search is globally
optimal. Selected models and seed repeats are calibrated automatically.

```powershell
.venv/Scripts/python.exe -B -m experiment.train_tabm start --device cuda
.venv/Scripts/python.exe -B -m experiment.train_tabm status
.venv/Scripts/python.exe -B -m experiment.train_tabm stop
.venv/Scripts/python.exe -B -m experiment.train_tabm resume
```

The controller survives chat closure; after reboot, run resume explicitly.
State, source snapshots, preprocessing, epoch telemetry, checkpoints and
calibration evidence are in
`artifacts/training/tabm_verified_v2_20261016_v1/`. Calibration uses6,963 rows,
assessment7,225 and tuning14,545;1,267 crossing/duplicate rows are purged from
role-based comparisons while remaining in the separately labelled full report.
