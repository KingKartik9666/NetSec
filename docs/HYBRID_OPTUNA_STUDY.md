# Optuna optimization of the three completed hybrids

Authorized 18 October 2026: retrain with different parameters using Bayesian
optimization. Use the installed Optuna 4.5.0 TPE sampler. Keep all previous
models, study protocols and research results intact.

The new study is `artifacts/training/hybrids_optuna_20261018_v1`.
Optimize the same flat TabM with numerical embeddings, XGBoost, genuine
efficient-KAN and bounded upstream TabPack branches used by the three hybrids.
There are 12 trials per family, including a measured adopted baseline, followed
by selected-recipe fold/full-data retraining. Continuous learning-rate and
regularization ranges and categorical capacity/preprocessing ranges are frozen
in source before execution. TPE uses four startup trials, deterministic seeds,
SQLite ask/tell storage, unique recipes and preserved failed/interrupted attempts.
Existing per-fit early stopping and runtime bounds remain active. There is no
standalone-score median pruning: the search objective is hybrid compatibility.

## Training-only objective

Reuse the exact two purged chronological training folds of the completed hybrid
study, including strict KAN fitting IDs. Within fold 0's OOF observations, create
an additional chronological, window/duplicate-purged split: earlier rows fit
global nonnegative blend weights, later rows score candidate hybrids. No OOF
target selects its branch checkpoint; the preceding early-stopping role does.
OOF classes with short time coverage are scoring-only and are reported.

Optimize coordinates in order: TabM, XGBoost, KAN, TabPack. Each candidate is
scored with the currently frozen partner predictions for its affected hybrids.
TabM affects the pair and trio; XGBoost affects all three; KAN affects only the
trio; TabPack affects its pair. After each family, freeze its selected recipe
before optimizing the next. This is one bounded coordinate search, not an
exhaustive joint grid. Primary objective is mean affected-hybrid accuracy with
an explicitly tiny macro-F1 tie-break; individual branch scores remain diagnostic.
All model-search labels are from the original training partition.

For each selected branch, retain/reuse its measured fold-0 checkpoint and fit
fold 1 and the full original training arm. Full-data early stopping uses only
the original validation tuning role. A selected baseline may reuse its verified
equivalent prior checkpoint; adoption records bind config, arm, row IDs and bytes.
Final models keep 655 ordered inputs, all 84 labels and explicit missing handling.

## Combiner optimization and final comparison

Use all 146,926 training OOF predictions for final combiner fitting. For each
hybrid retain a global-weight control and run 12 TPE trials over log-scaled
class-specific shrinkage in [0.0001,1]. Select combiners on frozen validation
tuning accuracy/macro F1 only. Classwise L-BFGS-B has a frozen 1000-iteration
budget; failed candidates remain explicit failures, not fabricated results.
Only converged candidates can be selected. Preserve the global control if it
scores higher. Calibrate after selection on validation role1 and assess role2.
Record full-validation and assessment metrics plus the previous hybrid controls.
Repeated within-capture validation remains exploratory; the test stays locked.

## Execution

Keep historical training sources unchanged. Reuse their model implementations,
GPU locks, checkpoint loaders, row streaming and bundle publication checks through
an explicitly documented new controller. Pin all relevant source/package/input
and partner prediction hashes. Completed and interrupted trials survive resume;
failed trials remain reported in Optuna. Serialize GPU fitting and respect RAM
headroom. Default application execution stays offline, CPU and credential-free.
Before launch run focused tests, authentic-schema CPU/CUDA smoke verification,
all evaluation tests and all five demos, plus one independent final review.

Run from the project environment:

```powershell
.venv/Scripts/python.exe -B -m experiment.optimize_hybrids start --device cuda
.venv/Scripts/python.exe -B -m experiment.optimize_hybrids status
.venv/Scripts/python.exe -B -m experiment.optimize_hybrids stop
.venv/Scripts/python.exe -B -m experiment.optimize_hybrids resume
```

The initial campaign was launched after 584 evaluation tests and all five demos
passed. Four sampled recipes also passed short authentic-schema CUDA training
and CPU checkpoint reload checks; these smoke checks are not research scores.
All eight historical fold-1/full adoption paths passed a pinned-artifact preflight.
The inner role counts are 32,664 blend-fitting rows, 34,675 scoring rows and
4,437 purged rows. Original validation roles remain 14,545 tuning, 6,963
calibration, 7,225 assessment and 1,267 purged rows.

Trial observations, requests and failures are durable in `optuna.sqlite3` and
`runs/`. The controller serializes GPU training; a selected baseline may be
adopted without another identical fit. The completion watcher stays quiet while
work continues and sends a local Windows notification on verified completion
or controller failure. Completion messages report any failed search trials.
It does not schedule a message in the chat. Verification logs are in the study's
`verification/` directory; final comparison tables will be written by the watcher
after calibrated bundles pass their reload checks.

Sources:
- https://optuna.readthedocs.io/en/v4.5.0/reference/samplers/generated/optuna.samplers.TPESampler.html
- https://optuna.readthedocs.io/en/v4.5.0/reference/generated/optuna.study.Study.html
