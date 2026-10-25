# Hybrid Optuna implementation plan

> **For agentic workers:** Use superpowers:executing-plans for inline implementation and one final independent review.

**Goal:** Run Bayesian parameter optimization and selected-recipe retraining for
the three completed neural/tree hybrids.
**Architecture:** Frozen training-only coordinate TPE, persistent trial requests,
shared selected branches, then OOF combiner TPE and post-selection calibration.
**Tech stack:** Existing Optuna 4.5.0, SQLite, PyTorch, XGBoost, NumPy/SciPy.
**Spec:** `docs/HYBRID_OPTUNA_STUDY.md`.

## Global constraints

- Verified v2 dataset, exact 655 inputs and 84 classes; held-out test locked.
- No edits to historical pinned sources or old study results.
- Model search and blend fitting use original training labels only.
- Calibration/assessment never select models or combiners.
- Explicit local CUDA experiments; default application remains offline CPU.

## Review focus

- Candidate checkpoint selection must precede, and exclude, OOF scoring rows.
- Restarts must recover unfinished Optuna requests without zombie duplicates.
- Adopted baseline models must match exact recipes, fitting rows and predictions.
- Refits and inference must consume the selected effective parameters.
- Failed optional candidates must not invalidate successful controls or publish
  an unverified completion.

### Task 1: Search policy and objective

Create `framework/training/hybrid_hpo.py` and focused tests in
`evaluation/test_hybrid_hpo.py`.
- [x] Freeze parameter spaces and seeded persistent TPE lifecycle.
- [x] Test recipe effects, restart identity, hybrid score and no assessment use.

### Task 2: Protocol and executable workers

Create `framework/training/hybrid_hpo_data.py` and `experiment/optimize_hybrids.py`.
- [x] Freeze source/data/partner hashes and purged meta roles.
- [x] Implement baseline adoption, serial workers, selected refits and resumable
  controller/status/stop/start. Preserve locks, memory policy and CPU reload.
- [x] Test role purging, adoption rejection and worker request identity.

### Task 3: Bayesian combiner selection and export

Create `framework/training/hybrid_hpo_finalize.py`.
- [x] Implement persistent combiner TPE, selection by frozen tuning only and retain
  converged global/classwise controls.
- [x] Reuse calibrated bundle publication/replay checks and report prior controls.
- [x] Verify selected parameter identity and calibrator reload behavior in regression checks.

### Task 4: Verification and launch

- [x] Run all required checks and one final independent review; fix findings.
- [x] Start the authorized CUDA search and verify an actual retraining trial.
- [x] Preserve completion notification intent and record execution evidence.

## Ledger

User explicitly authorized optimization/retraining of the existing hybrids.
Implementation proceeds directly. No Git repository exists; use additive source
files and snapshots. The completed hybrid study has nine successful fits and
three verified calibrated bundles; none needs to be overwritten.

Final review found and resolved baseline full-checkpoint path normalization,
failure-record recovery during combiner ask/tell, and controller identity checks
in the completion watcher. Each has a regression test. Verification: 584 passed,
11 dependency warnings; all five demos passed; all four branch CUDA recipe
smokes and all eight historical adoption checks passed. The study is now launched;
selected-model research scores and final calibration remain pending execution.
Live launch verification observed baseline trial 0 scored and recorded, trial 1
training on all 238,144 fold-0 fitting rows with 33,469 preceding early-stopping
rows, and two completed CUDA epochs. Persistent trial requests rejected a
modified configuration. The controller and quiet local Windows watcher were alive.
