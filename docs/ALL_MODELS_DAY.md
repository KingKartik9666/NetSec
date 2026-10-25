# All-model 24-hour optimization

Authorized on 20 October 2026. This is research execution on verified DataSense
v2: 655 features, 84 original classes, 480,405 full-training and 30,000 source
validation observations. Source-validation comparisons remain exploratory.
The held-out test is never loaded. No app/model promotion or live capture occurs.

## Models and method

The ten reported models are fixed RF, XGBoost, efficient-KAN, HEM-derived,
TabM-NE, TabM-Hier, TabM-Hier-NE, and the three existing hybrids. TabPack is an
additional shared branch of TabPack + XGBoost, not a new eleventh report model.

Seven branches receive persistent Optuna multivariate/group TPE proposals in
SQLite. The first measured configuration uses the prior selected recipe. Every
family receives three trial slots before improvement-per-second allocation with
an exploration bonus. Subsequent proposals search numerical regularization and
learning rates logarithmically, conditional capacities categorically, train-only
scaling, full/strict training arms and supported class weighting. RF stays fixed.

The three TabM variants retain separate hypotheses. Hierarchical subtype losses
use the true attack mask, with benign/attack and subtype probabilities composed
by the existing implementation. HEM remains a closed-set upstream-derived
adaptation with original neural/vote code, tunable capacity/PCA/neighbors and
preserved training-only neighbor references, not original-paper reproduction.

Early stopping remains in every backend. Optuna asynchronous successive halving
compares tuning accuracy within each family, after at least six epochs or 60
boosting rounds. The first three trials are never pruned. Pruned attempts are
retained and counted separately from failures. See the official
[Optuna pruning API](https://optuna.readthedocs.io/en/v4.5.0/reference/generated/optuna.pruners.SuccessiveHalvingPruner.html).

The initial 24-hour budget reserves elapsed hours 0-14 for recovery/search,
14-17 for seed-43/44 repeats, 17-21 for two forward OOF fits per hybrid branch,
and 21-24 for assessment, calibration and CPU bundles. Completed earlier phases
may advance without waiting. Recovery of the old interrupted hybrid study is
capped at two hours within this budget. Normal fits have a 35-minute watchdog
and most searched configurations a 25-minute optimization limit. Actual capacity
and remaining time determine the completed trial count; no accuracy is promised.

Checkpoint selection uses frozen tuning accuracy, macro-F1 tie-break and guards
against increases greater than 0.5 percentage points in false alarms or attack
misses, or a macro-recall drop greater than one point, relative to the historic
control. These guards use tuning only. Prior checkpoints remain selected when no
eligible measured gain exists. Seeds are evaluated for stability, not selected
after assessment. Calibration and assessment never feed the next model proposal.

OOF rows follow the existing purged forward folds. Selected recipes are refit
with their exact training arm. Global nonnegative OOF log-loss weights, equal
averaging and regularized classwise weights are compared; classwise shrinkage
gets twelve TPE slots per hybrid. Temperature scaling and binary Platt scaling
use the frozen calibration role. HEM correctness calibration does not fabricate
ensemble multiclass probabilities. CPU replay gates precede bundle completion.

## Detached loop and recovery

The supervisor is an ordinary hidden Python process, independent of Codex token
or context limits. It owns a controller, which runs one ML worker at a time.
Durable requests, Optuna trials, unique attempt directories, source/data pins,
process identities, bounded restarts and atomic JSON replacement retries permit
recovery. Historical sources and successful attempts are not overwritten.

The absolute start/deadline is written once in study.json and never reset by
resume. Three controller restarts and two worker attempts bound recovery.
Permanent numerical failures are recorded rather than retried endlessly. A STOP
marker stops new work and terminates owned active processes. Windows is requested
to remain awake only while the supervisor runs; no persistent power settings are
changed. Power loss, shutdown, manual sleep or killing the supervisor stops
execution; use resume after restarting Windows, within the original deadline.

```powershell
.venv/Scripts/python.exe -B -m experiment.optimize_all_models_day start
.venv/Scripts/python.exe -B -m experiment.optimize_all_models_day status
.venv/Scripts/python.exe -B -m experiment.optimize_all_models_day stop
.venv/Scripts/python.exe -B -m experiment.optimize_all_models_day resume
```

Evidence: artifacts/training/all_models_day_20261020_v1/study.json,
supervisor.json, state.json, runs/, selected.json, oof_registry/, assessment/,
hybrids/, report.json and REPORT.md. Missing stages remain pending/failed in the
report; a completed loop does not imply ten successful models.

Required software checks: the complete evaluation suite, default demo and all
four sample modes. Genuine-row reduced CPU smoke fits are explicitly separate
from full research results. No external repository is modified.

## Explicit pause and continuation — 21 October

The user paused all work after approximately fourteen active hours, then
explicitly requested resuming everything after the original wall-clock deadline.
The additive `experiment.resume_all_models_day` entry point resumes the frozen
selection and remaining stages under
`artifacts/training/all_models_resume_20261021_v1`. It restores only the unused
active-time budget (about ten hours), with a new recorded deadline. The original
protocol/deadline, search trials, checkpoints and STOP marker remain preserved.
Historical/cancelled studies are not restarted by this continuation.

```powershell
.venv/Scripts/python.exe -B -m experiment.resume_all_models_day resume
.venv/Scripts/python.exe -B -m experiment.resume_all_models_day status
.venv/Scripts/python.exe -B -m experiment.resume_all_models_day stop
```
