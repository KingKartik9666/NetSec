# Full-CSV research training

## Current study: csv_research_v3 (11 October 2026)

The accepted training-only study supersedes the bounded v2 controller described
below. XGBoost, genuine efficient-KAN and the upstream-derived closed-set HEM
adaptation use the unchanged full-CSV training/validation partitions. Conventional
RF is fixed. The held-out test, live versions, UI and packaging are deferred.
Success requires one identical recipe to achieve at least 95% validation accuracy
in seeds 42, 43 and 44, verified CUDA execution, and CPU checkpoint reload. A macro
recall decline greater than one percentage point or benign false-positive increase
greater than half a percentage point blocks accepting a winner against its control.
This is a validation stopping criterion, not independent test or industrial accuracy.

There is no overall deadline. Fits have 20-minute, one-hour or four-hour watchdogs;
CUDA fitting is sequential. Explicit Stop preserves completed evidence; interrupted
fits restart from the beginning in a new attempt directory. Requests, trial results,
logs, recursive bundle hashes and Optuna SQLite are preserved. The controller checks
source/input hashes before recovery. It holds locks across the old/new studies,
inhibits sleep only while active, and uses a Windows job object for descendant cleanup.

The study verifies completed v1/v2 checkpoint hashes, raw measurement/label hashes,
manifest and split identities before adoption. Its recipes cover independent MI
selection, safe count fractions/coefficient-of-variation features, constrained
resampling, real conditional tabular CTGAN, class weighting, supported losses and
neural scheduling, followed by compatible pairs and evidence-supported combinations.
Inactive knobs and identical fits are deduplicated. Optuna TPE tunes active model and
recipe parameters; median pruning is reported to the iterative trainers. Capacity
probes can interleave after six stagnant completed fits instead of waiting for every
pair. Their requests record the sampled training/validation gap and diagnosis.

Every final candidate keeps all 480,405 original training observations. MI estimation
uses a deterministic stratified sample capped at 30,000, without dropping those
observations from subsequent model fitting. Feature transforms, neural preprocessing,
augmentation and calibration fit on training only. Validation remains 30,000 real
rows; it supports 83 of 84 target classes. Macro metrics include the complete label
vocabulary and explicitly retain unsupported classes. Original input order remains
170 columns; each saved research transform reconstructs the selected/engineered
model input on CPU. No guessed per-second rates are derived from unverified interval
units. The combined user-provided CSV's publisher identity has not been independently
established by a publisher checksum; source results must retain that provenance limit.

SMOTENC, Borderline-1 SMOTE and ADASYN here are explicitly our constrained adaptations,
not unchanged imbalanced-learn estimators. Their neighbors retain categorical protocol
values, missingness and window contexts; generated counts are audited and exact real
donor IDs are saved. CTGAN uses the actual `ctgan.CTGAN` implementation on CUDA, trained
with joint label/context conditioning. Condition mismatches, invalid or duplicate rows
and collapsed output are rejected. Synthetic labels inherit donor categories; schema
validity does not independently establish real attack semantics. HEM's nearest-neighbor
references contain only original observations, never synthetic additions.

Same-family ensembles average three seed models uniformly (HEM uses its explicit hard
votes). There is no learned validation-weighted stack. Scalar temperature calibration
for XGBoost/KAN uses three training-only outer folds; each has a separate inner stopping
split, keeping prediction-fold rows out of checkpoint selection. Outer validation is
never used to fit calibration. HEM has no probability API, so its top-three metric and
probability calibration remain unavailable. A positive scalar temperature preserves
argmax accuracy; calibration is evaluated for probability quality rather than promoted
as an accuracy enhancement. Model enlargement remains a hypothesis until validation
supports it; no recipe promises 95%.

Commands from `K:\IMDAI Project` after implementation verification:

```powershell
.venv/Scripts/python.exe -B -m experiment.optimize_csv_research start
.venv/Scripts/python.exe -B -m experiment.optimize_csv_research status
.venv/Scripts/python.exe -B -m experiment.optimize_csv_research stop
.venv/Scripts/python.exe -B -m experiment.optimize_csv_research resume
```

Do not resume the retired controllers. Launch status and exact next work belong in
`docs/RESUME_CONTEXT.md`. New research dependencies are isolated in
`requirements-research.txt`; the app's default imports remain CPU/offline/credential-free.
Verification evidence lives in `artifacts/training/csv_research_v3/`: CUDA toy checks
prove execution and reload, not dataset accuracy; the full suite and all five demo
logs verify integration. Original models, data and evidence are preserved. No automatic
promotion or installed-app replacement occurs.

Method references: [Optuna sampling and pruning](https://optuna.readthedocs.io/en/v4.5.0/tutorial/10_key_features/003_efficient_optimization_algorithms.html),
[training-only resampling](https://imbalanced-learn.org/stable/common_pitfalls.html),
and [upstream CTGAN](https://github.com/sdv-dev/CTGAN).

## Historical bounded v2 study (retired)

The versioned `csv_specialization_v2` study trains XGBoost, efficient-KAN and the GitHub-derived closed-set HEM adaptation on complete CIC IIoT2025/DataSense training observations. The conventional Random Forest baseline remains fixed. This replaces the mixed full/live-feature training objective; separate live models remain blocked pending CSV completion and validation.

The three optimized families use the existing 480,405 training rows, 170 ordered features and 84 real category labels. Train/validation row identities are preserved; the held-out test is not loaded. RF retains its previously documented numeric-input compatibility configuration. It is our conventional implementation of the paper's model family, not an exact reproduction of unpublished author settings.

Accuracy, macro precision, recall, F1, benign false-positive rate and attacks missed as benign are persisted in `progress.json` each XGBoost round or neural epoch. Final metrics include class support and confusion matrices. Intermediate scores are validation observations, not independent test results or live-PC accuracy.

After independent ablations, supported combinations and local search, a gain below 0.1 percentage point over six completed trials triggers a broader one-factor proposal. Supported changes include weighting, regularization, learning rate, tree depth/binning, KAN capacity/grid size, HEM PCA/capacity, scheduling and training duration. Changes occur between fits. This is deterministic heuristic search, not Bayesian optimization or evidence that a proposed change works.

Each proposal records its reason and measured predecessor metrics. A recall decline exceeding one percentage point or a benign false-positive increase exceeding half a percentage point is flagged. Promotion requires improvement against controls and matched seed-43/44 repeats; promotion is explicit, never an automatic application replacement. Accuracy selection uses macro F1 as its tie-breaker. The 96% target is an aspiration, not a promised result.

The original wall-clock deadline remains 12 October 2026, 13:54:23 IST. Fits have a 20-minute outer budget and a shorter internal training budget. CUDA fits run sequentially; CPU checkpoint reload is checked. Immutable requests, hashes, independent process control, recovery and failure isolation preserve experiment evidence. No deadline extension is implicit.

Commands from the project root:

```powershell
.venv/Scripts/python.exe -B -m experiment.optimize_csv_models status
.venv/Scripts/python.exe -B -m experiment.optimize_csv_models stop
.venv/Scripts/python.exe -B -m experiment.optimize_csv_models resume
```

The prior full-CSV XGBoost control is retained: validation accuracy 79.50%, macro recall 69.39%, macro F1 67.33%, benign false-positive rate 1.07%. Its full 30,000-row CPU reload was verified. These results do not meet the target.

Verification after the adaptive changes: 365 evaluation tests passed; one existing joblib physical-core discovery warning. Two independent review findings (unstable resumed combinations and incomplete repeats marked completed) were fixed with regression tests. Current source and input hashes are pinned when the versioned study starts. Default application execution remains CPU-only, offline and credential-free.
