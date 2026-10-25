# Three neural/tree hybrids

Authorized 17 October 2026 after the architecture discussion:

1. Flat TabM with numerical embeddings + XGBoost.
2. The same TabM + XGBoost + genuine efficient-KAN.
3. An upstream TabPack experiment + the same XGBoost.

All branches predict the identical 84 ordered DataSense classes from the verified
655-feature v2 schema. This is soft probability fusion. There is no binary hard
gate. KAN can receive zero global weight; adding it does not guarantee a gain.
The default application and its deployed models are outside this study.

## Data and fitting

Use `dataset/experiments/datasense_verified_clean_20261014_v2` exclusively.
Freeze two class-conditional chronological training folds. Fold 0 uses the first
50% of distinct training start times for fitting, the next 10% for early stopping,
and 60–80% for out-of-fold predictions. Fold 1 uses 70%, 10%, and the last 20%.
Purge windows touching/crossing boundaries and duplicate feature vectors across
roles. These percentages refer to timestamps, not row counts. Preserve row IDs.
KAN's fitting rows also intersect its selected strict-training arm. Every fitting
role must retain all 84 labels; report unsupported early-stopping/OOF classes.
OOF labels never control the branch checkpoint, preprocessing, or member selection.

Reuse the existing selected seed-42 recipes and final checkpoints for TabM,
XGBoost and KAN. Their original trials remain unchanged. Fit new fold checkpoints
for OOF predictions; pretrained full-training checkpoints cannot generate OOF.
Fit TabPack's folds and a full-training checkpoint separately. The full TabPack
run may use only the original frozen validation tuning role for early stopping.

Learn nonnegative global blend weights by OOF log loss. Also compare a narrowly
scoped class-specific blend with weights shrunk toward the global weights
(fixed penalties 0.01 and 0.1). Select the combiner using only frozen validation
tuning accuracy, then macro F1, then log loss. Fit temperature and binary
calibration on the frozen calibration role; assess the later assessment role.
Report individual branches, equal weights, learned weights, false alarms, attack
recall, macro F1 and reliability. Prior validation/recipe reuse makes comparisons
exploratory. The held-out test is never opened, scored, or used for selection.

## TabPack compatibility experiment

Official source: https://github.com/yandex-research/tabpack
Pinned commit: `05a89e21b955f12de84889d662e15ca534019aaa`.
Keep the unmodified checkout in `external/tabpack`. The upstream project requires
Python >=3.12 and documents Linux GPU/macOS support. This environment is Windows
Python 3.11. A local adapter may load the actual upstream ModelPack,
MLPBackbonePack and AdamWPack with explicit type-annotation compatibility edits
in memory. It must verify pinned source hashes, preserve numerical code, and
pass CPU/CUDA forward/backward and checkpoint tests before research fitting.

Use eight fixed heterogeneous MLP/AdamW configurations, training-only quantile
preprocessing with explicit missing masks, and greedy member selection during
training on the early-stopping role. Save the selected pack and member weights.
This bounded upstream-based TabPack adaptation does not reproduce the paper's
full dynamic resampling/search protocol. No top-level upstream license file was
found at the pinned revision; retain attribution and do not redistribute its code
as project-owned code. This checkout is for the requested local experiment.

## Execution and evidence

Active additive study: `artifacts/training/hybrids_verified_v2_20261018_v2`.
Record source, package, data, checkpoint and fold hashes. GPU training is an
explicit CLI option; inference defaults to CPU. Serialize GPU workers with the
existing study locks. Use hidden detached workers, per-fit watchdogs, resumable
completed jobs, explicit failures and no automatic promotion. Separate numerical
CPU/GPU comparison from same-device checkpoint replay; record label changes and
margins. Do not rewrite earlier failed runs or relax their pinned checks.

Run meaningful leakage, probability alignment, combiner and real upstream tests,
then all evaluation tests and all five demo commands. A smoke run is software
verification, not a research score. Training status must distinguish built,
running, completed and failed.

## Commands and execution state

The implementation was completed and the frozen study initialized on 18 October
2026. Nine fits share the OOF
predictions needed by the three requested combinations. Fold 0 has 238,144 fitting
rows (190,442 for strict KAN), 33,469 early-stopping rows, and 71,776 OOF rows.
Fold 1 has 335,119 fitting rows (267,144 strict KAN), 34,435 early-stopping rows,
and 75,150 OOF rows. The combiner uses 146,926 training OOF observations.

```powershell
.venv/Scripts/python.exe -B -m experiment.train_hybrids start --device cuda
.venv/Scripts/python.exe -B -m experiment.train_hybrids status
.venv/Scripts/python.exe -B -m experiment.train_hybrids stop
.venv/Scripts/python.exe -B -m experiment.train_hybrids resume
```

Resume preserves completed fits and restarts interrupted fits in new attempt
directories. Failed fits remain skipped for diagnosis. Source/input changes
require a separately versioned study; do not edit the sealed protocol to make
a changed implementation look like the original run.

Each hybrid publishes its `hybrids/<name>/result.json` only after CPU bundle
replay. Normal bundle loading requires that completion record and verifies the
exported hashes and exact ordered schemas. The candidate bundle stays pending
until replay passes. Check `state.json` and actual workers for progress; built
and smoke-tested models do not constitute completed research training.

The initial `hybrids_verified_v2_20261017_v1` protocol remains preserved as an
initialized snapshot. It launched a controller but no fitting job because free
RAM was below 4 GiB. The v2 revision removes PyTorch from controller imports
(measured controller import RSS about 0.14 GiB instead of 0.62 GiB) and streams
neural fitting rows/columns instead of copying full folds. Neural fits require
2.5 GiB physical and 6 GiB commit headroom; XGBoost/KAN retain the 4 GiB physical
requirement. The scheduler can run an eligible neural job while a heavier job
waits. All observations, fold identities, recipes and nine planned fits remain
the same; no earlier fitting result was changed.

## Recorded optimization continuation and completion alert

After six successful fits, the three-model classwise combiner exhausted its
150-iteration L-BFGS-B budget. The completed TabM/XGBoost hybrid remained valid.
`experiment/train_hybrids_optimizer_compat.py` records a versioned controller
adapter in `controller_adapters/class_blend_iterations_v1.json` and extends only
that iteration limit to 500. Convergence tolerances and input/recipe pins stay
the same. Its own source digest and snapshot are recorded; new state and result
records identify the adapter. Resume this continuation with:

```powershell
.venv/Scripts/python.exe -B -m experiment.train_hybrids_optimizer_compat resume
```

The user's completion-alert request is served by the hidden local watcher at
`completion_monitor/monitor.py`. Automatic chat scheduling was unavailable.
Every minute it checks progress without routine notifications. A Windows
notification-area alert reports verified completion or a new actionable failure.
Success requires all nine requested fits, all three calibrated/replayed hybrid
bundles, and verified saved artifact hashes. It saves a final `REPORT.md` beside
its status and notification records, then exits. The PC and watcher must remain
running; no email or external messaging service is used.
